"""The planner loop: single tool call per turn, tuned for a 4B model (P2.8).

Answers a user's clinical question about ONE patient by repeatedly asking
the model (via ``OllamaClient.extract`` against ``PlannerDecision``, temp 0)
to either call exactly one tool or produce a final answer, dispatching the
chosen tool, and feeding its result back into the conversation for the next
turn. ``PlannerDecision`` structurally enforces "at most one tool call per
turn" -- see ``app.schemas.planner`` -- so there is nothing to additionally
police here beyond dispatch.

Patient-context binding: a ``Planner`` instance is constructed bound to one
``patient_id`` (the conversation's anchored patient) and every tool
dispatch uses that id -- never anything the model puts in ``tool_args``.
``_build_tool_kwargs`` enforces this structurally by only ever reading
tool-specific filter keys (``limit``, ``since``, ``start_date``,
``end_date``) out of ``tool_args``; ``patient_id`` is not among them, so a
model that tries to smuggle a different patient id into ``tool_args`` has it
silently dropped. Full cross-tool authorization enforcement (refusing calls
whose *effective* pid diverges from the bound one at every layer) is P2.16;
this loop is structured so that binding is already the only path in.

Quarantine seam (P2.9, NOT built here): today, ``summarize_tool_result``
is a pure pass-through of a tool's own Pydantic output. P2.9 will replace
its body with a call to a QUARANTINED summarizer -- a model call that reads
the raw tool output (which may carry adversarial text injected into a
patient's notes) and returns only a sanitized structured summary, so the
*planner* call never sees raw record text. Keeping tool-result handling
behind this one function is what makes that swap local instead of a loop
rewrite.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from app.ollama_client import OllamaClient
from app.openemr_client import OpenEmrApiError, OpenEmrClient
from app.schemas.common import ToolSchemaModel
from app.schemas.planner import PlannerAction, PlannerDecision, ToolName
from app.schemas.tools import (
    GetAllergiesInput,
    GetAppointmentsInput,
    GetEncountersInput,
    GetMedicationsInput,
    GetPatientSummaryInput,
    GetProblemsInput,
    GetRecentLabsInput,
    GetVitalsInput,
)
from app.tools.allergies import get_allergies
from app.tools.appointments import get_appointments
from app.tools.encounters import get_encounters
from app.tools.labs import get_recent_labs
from app.tools.medications import get_medications
from app.tools.patient_summary import get_patient_summary
from app.tools.problems import get_problems
from app.tools.vitals import get_vitals

_DEFAULT_MAX_TURNS = 6

# Recognized ``tool_args`` filter keys and how to coerce their string value.
# Only these keys are ever read from a model-supplied ``tool_args`` map --
# notably ``patient_id`` is not among them (see module docstring).
_INT_ARG_KEYS = {"limit"}
_DATE_ARG_KEYS = {"since", "start_date", "end_date"}


@dataclass(frozen=True)
class ToolSpec:
    """One tool registry entry: what it does, its Input contract, and the callable.

    ``func`` always has the shape ``(client, token, patient_id, **kwargs) ->
    ToolSchemaModel``, matching every tool in ``app.tools.*``.
    """

    description: str
    input_schema: type[ToolSchemaModel]
    func: Callable[..., ToolSchemaModel]


@dataclass(frozen=True)
class ToolCallTrace:
    """One completed tool dispatch: what was called, with what, and the outcome.

    Exactly one of ``result``/``error`` is set. This is the ordered record
    the caller gets back alongside the final answer -- feeds P2.10's SSE
    stream, P3's verification layer, and P4's observability traces.
    """

    tool: ToolName
    args: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None = None


@dataclass(frozen=True)
class PlannerResult:
    answer: str
    trace: list[ToolCallTrace]


TOOL_REGISTRY: dict[ToolName, ToolSpec] = {
    ToolName.GET_PATIENT_SUMMARY: ToolSpec(
        description=(
            "Demographics plus record counts across every section (medications, "
            "allergies, problems, labs, vitals, encounters, appointments). Use for "
            "a broad overview when no single section clearly answers the question."
        ),
        input_schema=GetPatientSummaryInput,
        func=get_patient_summary,
    ),
    ToolName.GET_MEDICATIONS: ToolSpec(
        description=(
            "The patient's medication list (name, dose, route, status, start/end "
            "dates). Use for 'what is she taking' and medication-safety questions."
        ),
        input_schema=GetMedicationsInput,
        func=get_medications,
    ),
    ToolName.GET_ALLERGIES: ToolSpec(
        description=(
            "The patient's recorded allergies (substance, reaction, severity). Use "
            "for 'any allergies' and drug-conflict/safety questions."
        ),
        input_schema=GetAllergiesInput,
        func=get_allergies,
    ),
    ToolName.GET_PROBLEMS: ToolSpec(
        description=(
            "The patient's problem list (diagnosis, ICD code, status, onset date). "
            "Use for 'what conditions does she have' / active-problem questions."
        ),
        input_schema=GetProblemsInput,
        func=get_problems,
    ),
    ToolName.GET_RECENT_LABS: ToolSpec(
        description=(
            "Recent lab results (test name, value, unit, reference range, date, "
            "abnormal flag). Use for lab-trend questions, e.g. 'last three A1c'. "
            "Optional tool_args: limit (integer count), since (YYYY-MM-DD)."
        ),
        input_schema=GetRecentLabsInput,
        func=get_recent_labs,
    ),
    ToolName.GET_VITALS: ToolSpec(
        description=(
            "Recent vital-sign readings (blood pressure, heart rate, temperature, "
            "respiratory rate, oxygen saturation, height, weight, BMI). Use for "
            "'what's her blood pressure been like'. Optional tool_args: limit, "
            "since (YYYY-MM-DD)."
        ),
        input_schema=GetVitalsInput,
        func=get_vitals,
    ),
    ToolName.GET_ENCOUNTERS: ToolSpec(
        description=(
            "Past visit/encounter history (date, reason, provider, type). Use for "
            "'what changed since I last saw her' and 'which visit was that from'. "
            "Optional tool_args: start_date, end_date (YYYY-MM-DD), limit."
        ),
        input_schema=GetEncountersInput,
        func=get_encounters,
    ),
    ToolName.GET_APPOINTMENTS: ToolSpec(
        description=(
            "Scheduled appointments (date, time, status, provider). Use for "
            "'when is her next appointment'. Optional tool_args: start_date, "
            "end_date (YYYY-MM-DD)."
        ),
        input_schema=GetAppointmentsInput,
        func=get_appointments,
    ),
}


_FEW_SHOT_EXAMPLES = """\
Q: "What meds is she on?"
-> {"action": "call_tool", "tool": "get_medications", "tool_args": null, "reason": "The medication list answers this directly.", "final_answer": null}

Q: "What changed since her last visit?"
-> {"action": "call_tool", "tool": "get_encounters", "tool_args": null, "reason": "Encounter history shows what changed since the last visit.", "final_answer": null}

Q: "What are her last three A1c values, and when?"
-> {"action": "call_tool", "tool": "get_recent_labs", "tool_args": {"limit": "3"}, "reason": "A lab-trend question scoped to the 3 most recent results.", "final_answer": null}

Q: "Does she have any allergies?"
-> {"action": "call_tool", "tool": "get_allergies", "tool_args": null, "reason": "The allergy list answers this directly.", "final_answer": null}

Q: "Which visit was that from?" (asked right after a tool result already named a visit date in this conversation)
-> {"action": "answer", "tool": null, "tool_args": null, "reason": "The visit date is already present in an earlier tool result.", "final_answer": "That result is from the visit on <date>."}\
"""

_SYSTEM_PROMPT_TEMPLATE = """\
You are a clinical co-pilot assisting a clinician with ONE specific patient \
(OpenEMR patient id {patient_id}). Answer only from data returned by your \
tools -- never invent facts, and never discuss any patient other than the \
one this conversation is bound to.

Each turn, do exactly ONE of:
  - call_tool: pick exactly ONE tool from the list below to run next.
  - answer: give your final answer, using only what the tools have already \
returned in this conversation.

Available tools:
{tool_descriptions}

Rules:
  - Call at most one tool per turn.
  - Never invent or guess a patient id -- every tool always runs against \
the patient this conversation is bound to; you cannot change it.
  - tool_args, when needed, is a flat string map of the optional filters \
named in a tool's description above (e.g. {{"limit": "3"}}). Omit it \
(or leave it null) when no filter applies.
  - Answer only from tool results already returned in this conversation. \
If they don't contain the answer yet, call another tool rather than \
guessing.

Examples (question -> decision):
{few_shot_examples}

/no_think
"""


def _build_system_prompt(patient_id: int, registry: Mapping[ToolName, ToolSpec]) -> str:
    tool_descriptions = "\n".join(f"  - {name.value}: {spec.description}" for name, spec in registry.items())
    return _SYSTEM_PROMPT_TEMPLATE.format(
        patient_id=patient_id,
        tool_descriptions=tool_descriptions,
        few_shot_examples=_FEW_SHOT_EXAMPLES,
    )


def summarize_tool_result(tool: ToolName, output: ToolSchemaModel) -> dict[str, Any]:
    """Structured tool result handed back into the planner's working context.

    TODO(P2.9): pure pass-through today (``model_dump``). P2.9 plugs its
    QUARANTINED summarizer call in here -- see module docstring.
    """
    return output.model_dump(mode="json")


def _coerce_arg(key: str, value: str) -> Any:
    if key in _INT_ARG_KEYS:
        return int(value)
    if key in _DATE_ARG_KEYS:
        return datetime.date.fromisoformat(value)
    return value


def _build_tool_kwargs(spec: ToolSpec, raw_args: dict[str, str] | None) -> dict[str, Any]:
    """Build validated tool call kwargs from the model's ``tool_args``.

    Only keys that are both a field on ``spec.input_schema`` and not
    ``patient_id`` are honored -- this is what keeps a model-supplied
    ``patient_id`` (or any field this tool doesn't take) from ever reaching
    the tool call. A value that fails to coerce for its key (e.g. a
    non-integer ``limit``) is dropped rather than raised, since these are
    all optional filters -- a malformed filter degrades to "no filter",
    not a crashed turn.
    """
    if not raw_args:
        return {}
    allowed_fields = set(spec.input_schema.model_fields) - {"patient_id"}
    kwargs: dict[str, Any] = {}
    for key, value in raw_args.items():
        if key not in allowed_fields:
            continue
        try:
            kwargs[key] = _coerce_arg(key, value)
        except ValueError:
            continue
    return kwargs


def _best_effort_answer(trace: list[ToolCallTrace], max_turns: int) -> str:
    if not trace:
        return f"I wasn't able to reach an answer within {max_turns} turns."
    called = ", ".join(dict.fromkeys(call.tool.value for call in trace))
    return (
        f"I wasn't able to reach a final answer within {max_turns} turns. "
        f"I gathered data from: {called}."
    )


class Planner:
    """Runs the single-tool-per-turn loop for one conversation, bound to one patient.

    Args:
        ollama_client: Anything exposing ``extract(messages, schema) ->
            PlannerDecision`` -- ``OllamaClient`` in production, a scripted
            fake in hermetic tests.
        openemr_client: Passed through to whichever tool the model selects.
        token: The user's OAuth bearer token, passed through to tools.
        patient_id: The conversation's anchored patient. Every tool
            dispatch uses this id -- see module docstring.
        registry: Tool name -> ``ToolSpec`` map. Defaults to the production
            ``TOOL_REGISTRY``; hermetic tests inject a fake registry.
        max_turns: Guard against infinite loops. On the last turn without
            an ``answer`` decision, a best-effort answer is synthesized
            from the trace instead of calling the model again.
    """

    def __init__(
        self,
        *,
        ollama_client: OllamaClient,
        openemr_client: OpenEmrClient,
        token: str,
        patient_id: int,
        registry: Mapping[ToolName, ToolSpec] | None = None,
        max_turns: int = _DEFAULT_MAX_TURNS,
    ) -> None:
        self._ollama = ollama_client
        self._openemr = openemr_client
        self._token = token
        self._patient_id = patient_id
        self._registry = registry if registry is not None else TOOL_REGISTRY
        self._max_turns = max_turns

    def run(self, question: str) -> PlannerResult:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _build_system_prompt(self._patient_id, self._registry)},
            {"role": "user", "content": question},
        ]
        trace: list[ToolCallTrace] = []

        for _ in range(self._max_turns):
            decision = self._ollama.extract(messages, PlannerDecision)

            if decision.action is PlannerAction.ANSWER or decision.tool is None:
                return PlannerResult(answer=decision.final_answer or "", trace=trace)

            messages.append({"role": "assistant", "content": decision.model_dump_json()})

            spec = self._registry.get(decision.tool)
            if spec is None:
                messages.append(
                    {"role": "user", "content": f"[tool result] unknown tool {decision.tool!r}; choose from the available tools."}
                )
                continue

            call_kwargs = _build_tool_kwargs(spec, decision.tool_args)

            try:
                output = spec.func(self._openemr, self._token, self._patient_id, **call_kwargs)
            except OpenEmrApiError as exc:
                trace.append(ToolCallTrace(tool=decision.tool, args=call_kwargs, result=None, error=exc.category.value))
                messages.append({"role": "user", "content": f"[tool result] {decision.tool.value} failed: {exc.category.value}"})
                continue

            summary = summarize_tool_result(decision.tool, output)
            trace.append(ToolCallTrace(tool=decision.tool, args=call_kwargs, result=summary, error=None))
            messages.append({"role": "user", "content": f"[tool result] {decision.tool.value}: {json.dumps(summary)}"})

        return PlannerResult(answer=_best_effort_answer(trace, self._max_turns), trace=trace)

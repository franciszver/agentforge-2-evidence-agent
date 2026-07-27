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
model that tries to smuggle a different patient id into ``tool_args`` cannot
retarget a tool. On top of that structural drop, P2.16 adds a LOUD, auditable
refusal: before every dispatch the loop calls
``app.authz.enforce_patient_binding``, which raises
``PatientBindingViolation`` if ``tool_args`` names a patient other than the
bound one. The loop catches it, records a ``patient_binding_violation``
trace entry (no tool run, no record content), and feeds a refusal note back --
so a cross-patient attempt is refused and recorded rather than silently
ignored. This is defense-in-depth narrowing, not a second RBAC (role
enforcement stays in OpenEMR).

Quarantine seam (P2.9): tool output is not fed to the planner raw. Each
tool result is routed through ``app.quarantine.quarantine_tool_result``,
which passes safe typed fields through verbatim but replaces every free-text
string (which may carry adversarial text injected into a patient's notes)
with an LLM-cleaned summary produced by a QUARANTINED summarizer that cannot
invoke any tool -- so the *planner* call never sees raw record free-text. See
``app.quarantine`` for the structural no-tool-access guarantee.

Two-call final answer (P2.9): once the planner decides to answer, it does
NOT return the decision's ``final_answer`` directly. It reasons in free text
(a ``chat`` call) and then extracts the final answer into the
``FinalAnswer`` schema via constrained decoding (an ``extract`` call) --
constraining only the extraction, not the reasoning. See
``_finalize_answer_streaming``.

Streamed reasoning (P213): the free-text reasoning half of that two-call
pattern is the only step in the whole planner loop that CAN stream -- the
``extract(FinalAnswer)`` call cannot (schema decode needs the whole JSON).
``_finalize_answer_streaming`` yields a ``ReasoningDelta`` event per
reasoning token (via ``OllamaClient.chat_stream``, when the injected client
implements it) before returning the verified ``FinalAnswer``, so a streaming
caller (``app.chat._stream_chat``) can render the model's in-progress
reasoning into a separate "thinking" surface as it arrives -- NEVER into the
authoritative answer slot, which only ever carries the post-extraction,
verified text. A double that only implements ``chat`` (no ``chat_stream`` --
every fake in ``tests/test_planner.py`` and the eval runner's
``ReplayOllamaClient``) falls back to one blocking ``chat`` call and yields
no ``ReasoningDelta`` events at all; either way the reasoning text fed into
the ``extract(FinalAnswer)`` call is identical, so ``run()``'s result never
depends on which path was taken.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.authz import PatientBindingViolation, enforce_patient_binding
from app.ollama_client import LlmCallStats
from app.openemr_client import OpenEmrApiError, OpenEmrClient
from app.quarantine import QuarantinedSummarizer, quarantine_tool_result
from app.schemas.common import ToolSchemaModel
from app.schemas.ingestion import Citation
from app.schemas.planner import FinalAnswer, PlannerAction, PlannerDecision, ToolName
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
from app.tools.patient_summary import RosterEntry, get_patient_name, get_patient_roster, get_patient_summary
from app.tools.problems import get_problems
from app.tools.vitals import get_vitals

_logger = logging.getLogger(__name__)

_DEFAULT_MAX_TURNS = 6


class _PlannerLlmClient(Protocol):
    """Structural subset of ``OllamaClient``/``LlamaServerClient`` the planner
    needs (P3.10a, epic #52 step 1): schema-constrained extraction plus plain
    chat. Narrow on purpose, matching ``app.extraction._Extractor`` and
    ``app.reranking._Extractor``'s pattern of depending on a Protocol rather
    than a concrete client class, so either engine can be injected here.
    ``chat_stream``/``call_stats`` are accessed via ``getattr`` below (both
    optional capabilities), so they are deliberately not part of this
    Protocol.
    """

    def chat(self, messages: list[dict[str, str]], *, options: dict[str, Any] | None = None) -> str: ...

    def extract(
        self,
        prompt_or_messages: str | list[dict[str, str]],
        schema: type[Any],
        *,
        options: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> Any: ...

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

    Exactly one of ``result``/``error`` is set. ``result`` is the
    *quarantined* (post-``app.quarantine``) tool output -- free-text fields
    are already redacted here. This is the ordered record the caller gets
    back alongside the final answer, and it is the CLIENT-FACING channel:
    it feeds P2.10's SSE stream and P4's observability traces. Raw record
    free-text must NEVER land here -- see ``PlannerResult.raw_results`` for
    the separate verifier-only channel that carries the un-redacted values.
    """

    tool: ToolName
    args: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None = None
    # Timing around the raw tool dispatch (``spec.func`` call), for the P4
    # ``tool`` trace span (``app.trace_store.record_tool_span``). ``error``
    # doubles as that span's ``error_category`` -- both are already the
    # closed-set category string (``OpenEmrApiError.category.value`` or
    # ``"patient_binding_violation"``), never a raw exception message, so no
    # separate field is needed. Defaulted (not required) so the many existing
    # tests constructing ``ToolCallTrace`` without timing keep working.
    start_ts: float = 0.0
    end_ts: float = 0.0


@dataclass(frozen=True)
class PlannerResult:
    """The planner's per-run output.

    ``trace`` is client-facing (quarantined; see ``ToolCallTrace``).
    ``raw_results`` is a VERIFIER-ONLY channel: the un-redacted
    ``model_dump`` of each tool call's raw output, positionally aligned 1:1
    with ``trace`` (``None`` for entries that produced no output -- a binding
    violation or an API error). It exists so the deterministic citation
    checker (``app.verification``, P3.2) can re-validate cited free-text
    values (a drug name, a lab value) against what the record actually said,
    NOT against the quarantine-redacted skeleton. Because that checker is
    fully deterministic (no LLM anywhere in its path), feeding it raw
    record text is safe -- injection text cannot steer an equality
    comparison. This field must never be forwarded into an LLM prompt or the
    SSE trace; only the verification layer reads it.
    """

    answer: str
    trace: list[ToolCallTrace]
    raw_results: list[dict[str, Any] | None] = field(default_factory=list)
    # Every LLM call this run made (decision extracts, the quarantine
    # summarizer, and the two-call finalize) -- read from ``ollama_client
    # .call_stats`` at the end of ``run()``, for the P4 ``llm`` trace spans.
    # Empty for an ``ollama_client`` double with no ``call_stats`` (see
    # ``Planner.run``'s defensive ``getattr``).
    llm_calls: list[LlmCallStats] = field(default_factory=list)
    # Issue #158 (gate-3 review MAJOR finding): the MODEL's own answer text,
    # captured by ``app.extraction.apply_recency_notice`` immediately BEFORE
    # it splices a machine-generated recency notice onto ``answer``. ``None``
    # (the default) means no notice has been applied -- ``answer`` IS the
    # model's own text, unmodified.
    #
    # Why this exists: a recency notice is built from the STALE RECORD'S OWN
    # DATE (``app.verification._recency_notice_text``, e.g. "Note: lab
    # results from 2014-02-01 may not reflect the patient's current
    # status.") and appended to ``answer`` -- text the MODEL never wrote,
    # containing raw record data. Issue #158's per-tool-call engagement check
    # (``app.tool_call_scoping.engaged_call_ids``) tokenizes whichever answer
    # text it is given; if it were given the POST-notice ``answer``, the
    # notice's own appended date would token-overlap with the stale call's
    # OWN raw values and wrongly engage a call the model itself never
    # discussed -- the exact self-engagement bug this field exists to
    # prevent. Any future engagement/grounding-style check over "what did the
    # model's own prose actually say" MUST read ``answer_pre_notice`` when it
    # is not ``None``, never bare ``answer`` -- see
    # ``app.extraction.run_verification``'s use of it.
    answer_pre_notice: str | None = None


@dataclass(frozen=True)
class ToolDispatched:
    """A ``run_streaming`` event: one tool dispatch just completed (success,
    API error, or binding-violation refusal), carrying its ``ToolCallTrace``.

    Yielded immediately after the dispatch, so a streaming caller (P2.12,
    ``app.chat._stream_chat``) can emit the ``tool_call`` SSE frame and
    record the tool span AS each tool runs, instead of replaying the whole
    trace after the loop finishes.
    """

    trace: ToolCallTrace


@dataclass(frozen=True)
class ReasoningDelta:
    """A ``run_streaming`` event: one incremental piece of the model's
    free-text reasoning from the two-call final-answer step (P2.9's
    ``_finalize_answer_streaming``), as it streams off Ollama (P213).

    This is UNVERIFIED, provisional text -- the model's in-progress
    reasoning toward an answer, not the answer itself. It exists so a
    streaming caller (``app.chat._stream_chat``) can render it into a
    separate, clearly-labeled "thinking" surface as it arrives, distinct
    from the authoritative ``answer`` frame that only ever carries the
    VERIFIED ``FinalAnswer`` text the subsequent constrained ``extract``
    call produces. Never fed back into a prompt, never persisted as the
    answer, never logged -- purely a UI progress signal.
    """

    text: str


@dataclass(frozen=True)
class PlannerCompleted:
    """A ``run_streaming`` event: the loop is done. Carries the same
    ``PlannerResult`` that ``run()`` returns directly -- always the LAST
    event ``run_streaming`` yields, exactly once."""

    result: PlannerResult


# Tagged union of everything ``Planner.run_streaming`` can yield -- mirrors
# ``app.rendering.AnswerSegment``'s plain-union-of-frozen-dataclasses style.
PlannerEvent = ToolDispatched | ReasoningDelta | PlannerCompleted


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
            "'what changed since I last saw her', 'which visit was that from', and "
            "open-ended status questions ('how is she doing', 'what's new with him') "
            "that don't name a specific domain -- the most recent encounter grounds "
            "the current picture. Optional tool_args: start_date, end_date "
            "(YYYY-MM-DD), limit."
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

Q: "What was his last blood pressure reading, and what category does that fall into?"
-> {"action": "call_tool", "tool": "get_vitals", "tool_args": null, "reason": "The vitals list has the patient's own recorded blood pressure reading, needed before any category can be assigned.", "final_answer": null}

Q: "Which visit was that from?" (asked right after a tool result already named a visit date in this conversation)
-> {"action": "answer", "tool": null, "tool_args": null, "reason": "The visit date is already present in an earlier tool result.", "final_answer": "That result is from the visit on <date>."}

Q: "What's the latest with him overall?"
-> {"action": "call_tool", "tool": "get_encounters", "tool_args": null, "reason": "An open-ended status question that names no specific domain -- recent encounter history grounds the current picture before checking a narrower list.", "final_answer": null}\
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
  - For an open-ended status question that doesn't name a specific domain \
(medications, allergies, problems, labs, vitals, appointments), start with \
get_encounters or get_patient_summary rather than a single narrow list -- a \
domain-specific tool can come back empty even when the patient has other \
record history, and skips the most recent visit context.
  - Every answer describes patient {patient_id} only. If the clinician \
names or numbers a different patient, do not repeat that other name or \
number anywhere in your answer, and do not state any fact as if it were \
about them.

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


_FINAL_REASON_PROMPT = (
    "You now have everything you need. Think through the clinician's question "
    "using ONLY the tool results already in this conversation, and write the "
    "answer in plain prose. Do not invent facts. Do not name or attribute any "
    "fact to a patient other than the one this conversation is bound to, even "
    "if the clinician's question named a different patient. "
    "/no_think"
)
_FINAL_EXTRACT_PROMPT = "Extract the final answer for the clinician as JSON."

# Issue #105 (bp-stage2 follow-up from #85): the guideline corpus retrieval
# (P3.9, `app.chat.get_evidence_retriever`) used to run strictly AFTER
# `Planner.run`/`run_streaming` returned -- the planner composed its answer
# text purely from tool results and its own training/priors, and a citation
# was bolted onto that already-written prose after the fact by the claim
# extractor. That let the planner's category language (e.g. "elevated blood
# pressure") drift from the guideline text it ended up citing (e.g. "Stage 2
# hypertension"): a genuine, verbatim citation attached to prose that
# disagreed with it -- exactly the failure the semantic-support judge (#47)
# is designed to catch.
#
# The fix feeds retrieved guideline text INTO the free-text reasoning call
# (`_finalize_answer_streaming`) that composes the answer, not just into
# post-hoc citation attachment -- so when a guideline chunk actually defines
# a category/threshold relevant to a value already in the tool results, the
# model is instructed to use THAT text's own category name rather than
# reaching for its own general-knowledge terminology. Retrieval itself still
# happens exactly where it always did (`app.chat`'s evidence-retrieval
# path); only the ORDER changed -- it now runs before the planner call
# instead of after, and its result is threaded through as `guideline_excerpts`
# (see `Planner.run`/`run_streaming`). A question with no relevant guideline
# evidence (chart-data-only, or retrieval found nothing above the relevance
# floor) passes `None`/empty here and this prompt addition is skipped
# entirely -- byte-identical to the pre-#105 prompt.
_GUIDELINE_CONTEXT_PROMPT_TEMPLATE = """\
The following clinical guideline excerpts were retrieved as potentially relevant \
to this patient's data. If an excerpt defines a category, threshold, or name for \
a value already established by the tool results above, use THAT EXACT category \
name in your answer -- do not substitute your own general terminology for it. \
These excerpts are reference material only, not facts about this specific \
patient -- never state anything from them as a patient-specific fact unless the \
tool results already established it.

{excerpts}
/no_think\
"""

# Issue #86: mirrors #105's guideline_excerpts fix at the SAME call site
# (the free-text reasoning call in `_finalize_answer_streaming`, never the
# #123-fragile tool-dispatch decision prompt) -- appended ONLY when
# `document_facts` is non-empty, so the (today, universal) empty case stays
# byte-identical to before this fix.
_DOCUMENT_FACT_CONTEXT_PROMPT_TEMPLATE = """\
The following facts were extracted from documents this patient's clinician has \
uploaded (e.g. a lab report or intake form). Each line below is a literal quote \
of what was actually read from that document -- if a field is not shown, it was \
not legible or not present in the document; never guess or state a value for it. \
These ARE facts about this specific patient -- use them to answer the question \
when relevant, and if the document does not contain the specific detail asked \
about, say so honestly rather than reporting that no data exists at all.

{facts}
/no_think\
"""


def _append_context_message(messages: list[dict[str, str]], template: str, **template_kwargs: str) -> None:
    """Append one ``{"role": "user"}`` message rendering ``template`` --
    shared by the guideline-excerpts (#105) and document-facts (#86)
    context-injection blocks in ``_finalize_answer_streaming``, which are
    otherwise structurally identical (guard on non-empty input -> append)."""
    messages.append({"role": "user", "content": template.format(**template_kwargs)})


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
            PlannerDecision`` -- ``OllamaClient`` or ``LlamaServerClient`` in
            production (see ``app.chat.get_text_llm_client``), a scripted
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
        ollama_client: _PlannerLlmClient,
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
        # The summarizer gets ONLY the ollama client -- never the registry,
        # openemr client, or token -- so it structurally cannot call a tool.
        self._summarizer = QuarantinedSummarizer(ollama_client=ollama_client)

    def resolve_patient_name(self) -> str | None:
        """Best-effort resolve this conversation's bound patient display name
        (Phase 1 #224 name-binding), e.g. for ``app.extraction
        .detect_foreign_patient_reference``'s named cross-patient signals.

        ``None`` on any OpenEMR API error (patient not found, timeout, ...)
        -- callers treat a missing name as "name-binding unavailable", which
        that guard already handles by falling back to its pre-Phase 1 #224
        numeric-only signal. An OPTIONAL capability, not part of
        ``app.chat.PlannerProtocol`` -- callers duck-type it via ``getattr``
        (same pattern as ``run_streaming``), so a test double that only
        implements ``run()`` simply has no name to offer.
        """
        return get_patient_name(self._openemr, self._token, self._patient_id)

    def resolve_patient_roster(self) -> list[RosterEntry]:
        """Every patient's (pid, display name) pair (Phase 1 #237 roster-based
        cross-patient detection), for ``app.extraction
        .detect_foreign_patient_reference``'s "switch to <Name>" signal.

        Issue #174: patient-agnostic -- unlike pre-#174, this does NOT
        exclude ``self._patient_id``'s own entry. This fetch runs as
        ``self._token``'s own bearer, so it is NOT caller-invariant in
        general -- see ``app.tools.patient_summary.get_patient_roster`` and
        ``app.chat.RosterCache`` for the full per-auth-mode analysis --
        which is why ``RosterCache`` caches it as ONE shared, process-wide
        entry only when every caller is provably the same principal, keyed
        by principal otherwise, instead of resolving and retaining it
        separately per conversation.
        The bound patient's own entry is excluded by the CALLER, at
        comparison time, keyed by pid (see
        ``app.extraction._matches_roster``).

        ``[]`` on any OpenEMR API error (fail-safe -- see
        ``app.tools.patient_summary.get_patient_roster``). An OPTIONAL
        capability, not part of ``app.chat.PlannerProtocol`` -- callers
        duck-type it via ``getattr`` (same pattern as ``resolve_patient_name``
        / ``run_streaming``), and unlike ``resolve_patient_name`` (resolved
        ONCE per conversation, at creation time), callers invoke this LAZILY
        -- only when a candidate "switch to <Name>" construction has already
        matched -- so a conversation that never mentions another patient by
        name never pays this round trip. This method itself performs no
        caching -- ``app.chat.RosterCache`` (the process-wide TTL cache) sits
        in front of it at the call site.
        """
        return get_patient_roster(self._openemr, self._token)

    def run(
        self,
        question: str,
        guideline_excerpts: Sequence[str] | None = None,
        document_facts: Sequence[Citation] | None = None,
    ) -> PlannerResult:
        """Run the loop to completion and return the finished result.

        Delegates to ``run_streaming`` and returns only its terminal event's
        result, discarding the intermediate ``ToolDispatched`` events -- so
        this stays byte-identical to the pre-P2.12 implementation from the
        caller's perspective.

        ``guideline_excerpts`` (#105): retrieved guideline-corpus chunk text
        (if any -- see ``app.chat``'s evidence-retrieval wiring, which now
        retrieves BEFORE calling this) fed into final-answer composition so
        the answer's own category language matches what it will end up
        citing. ``None``/empty is a no-op, byte-identical to before #105.

        ``document_facts`` (#86): this patient's own ingested lab/intake-form
        fact citations (if any -- see ``app.chat.get_patient_fact_provider``,
        which now fetches BEFORE calling this) fed into final-answer
        composition so a question only answerable from an ingested document
        is actually answered from it, instead of "no data recorded" (chart
        tools alone genuinely have none). ``None``/empty is a no-op,
        byte-identical to before #86.
        """
        for event in self.run_streaming(question, guideline_excerpts, document_facts):
            if isinstance(event, PlannerCompleted):
                return event.result
        raise AssertionError("run_streaming ended without a terminal PlannerCompleted event")  # pragma: no cover

    def run_streaming(
        self,
        question: str,
        guideline_excerpts: Sequence[str] | None = None,
        document_facts: Sequence[Citation] | None = None,
    ) -> Iterator[PlannerEvent]:
        """Same single-tool-per-turn loop as ``run()``, but yields a
        ``ToolDispatched`` event immediately after each tool dispatch
        completes (success, API error, or binding-violation refusal) instead
        of only returning the full trace once the loop finishes. Always
        yields exactly one terminal ``PlannerCompleted`` event, last, whose
        ``result`` is identical to what ``run()`` returns for the same
        inputs (P2.12).

        ``guideline_excerpts``: see ``run()``'s docstring (#105).
        ``document_facts``: see ``run()``'s docstring (#86).
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _build_system_prompt(self._patient_id, self._registry)},
            {"role": "user", "content": question},
        ]
        trace: list[ToolCallTrace] = []
        # Verifier-only channel, aligned 1:1 with ``trace``. See PlannerResult.
        raw_results: list[dict[str, Any] | None] = []

        for _ in range(self._max_turns):
            decision = self._ollama.extract(messages, PlannerDecision)

            if decision.action is PlannerAction.ANSWER or decision.tool is None:
                final = yield from self._finalize_answer_streaming(messages, guideline_excerpts, document_facts)
                yield PlannerCompleted(
                    PlannerResult(answer=final.answer, trace=trace, raw_results=raw_results, llm_calls=self._collect_llm_calls())
                )
                return

            messages.append({"role": "assistant", "content": decision.model_dump_json()})

            spec = self._registry.get(decision.tool)
            if spec is None:
                messages.append(
                    {"role": "user", "content": f"[tool result] unknown tool {decision.tool!r}; choose from the available tools."}
                )
                continue

            binding_check_ts = time.time()
            try:
                enforce_patient_binding(bound_patient_id=self._patient_id, tool_args=decision.tool_args)
            except PatientBindingViolation:
                call_trace = ToolCallTrace(
                    tool=decision.tool,
                    args={},
                    result=None,
                    error="patient_binding_violation",
                    start_ts=binding_check_ts,
                    end_ts=time.time(),
                )
                trace.append(call_trace)
                raw_results.append(None)
                yield ToolDispatched(call_trace)
                _logger.warning("tool_call refused: patient_binding_violation", extra={"tool": decision.tool.value})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[tool result] {decision.tool.value} refused: this conversation is bound to a "
                            "single patient and cannot access another; do not attempt to change the patient."
                        ),
                    }
                )
                continue

            call_kwargs = _build_tool_kwargs(spec, decision.tool_args)

            tool_start_ts = time.time()
            try:
                output = spec.func(self._openemr, self._token, self._patient_id, **call_kwargs)
            except OpenEmrApiError as exc:
                call_trace = ToolCallTrace(
                    tool=decision.tool,
                    args=call_kwargs,
                    result=None,
                    error=exc.category.value,
                    start_ts=tool_start_ts,
                    end_ts=time.time(),
                )
                trace.append(call_trace)
                raw_results.append(None)
                yield ToolDispatched(call_trace)
                _logger.warning(
                    "tool_call failed", extra={"tool": decision.tool.value, "error": exc.category.value}
                )
                messages.append({"role": "user", "content": f"[tool result] {decision.tool.value} failed: {exc.category.value}"})
                continue
            tool_end_ts = time.time()

            # Capture the RAW output for the verifier-only channel BEFORE
            # quarantining. The planner's message context (below) and the
            # client-facing trace still only ever see the quarantined summary.
            raw_results.append(output.model_dump(mode="json"))
            summary = quarantine_tool_result(self._summarizer, decision.tool, output)
            call_trace = ToolCallTrace(
                tool=decision.tool, args=call_kwargs, result=summary, error=None, start_ts=tool_start_ts, end_ts=tool_end_ts
            )
            trace.append(call_trace)
            yield ToolDispatched(call_trace)
            _logger.info("tool_call dispatched", extra={"tool": decision.tool.value})
            messages.append({"role": "user", "content": f"[tool result] {decision.tool.value}: {json.dumps(summary)}"})

        yield PlannerCompleted(
            PlannerResult(
                answer=_best_effort_answer(trace, self._max_turns),
                trace=trace,
                raw_results=raw_results,
                llm_calls=self._collect_llm_calls(),
            )
        )

    def _collect_llm_calls(self) -> list[LlmCallStats]:
        """Every LLM call this run made so far, read from the shared
        ``OllamaClient.call_stats`` side channel (see ``PlannerResult
        .llm_calls``). ``getattr``-defensive: hermetic test doubles that
        don't model ``call_stats`` degrade to no llm spans rather than an
        ``AttributeError``."""
        return list(getattr(self._ollama, "call_stats", []))

    def _finalize_answer_streaming(
        self,
        messages: list[dict[str, str]],
        guideline_excerpts: Sequence[str] | None = None,
        document_facts: Sequence[Citation] | None = None,
    ) -> Generator[ReasoningDelta, None, FinalAnswer]:
        """Produce the final answer via the two-call pattern (P2.9), streaming
        the free-text reasoning half as ``ReasoningDelta`` events (P213).

        First a free-text reasoning call (unconstrained, so reasoning quality
        is not taxed by the grammar), then a constrained ``extract`` call
        that pins the answer to ``FinalAnswer``. The reasoning is fed into
        the extraction call so the extractor only has to transcribe, not
        re-derive.

        ``guideline_excerpts`` (#105): when non-empty, a
        ``_GUIDELINE_CONTEXT_PROMPT_TEMPLATE`` message carrying the retrieved
        guideline text is appended to the reasoning call's messages, BEFORE
        ``_FINAL_REASON_PROMPT`` -- so the model composes its answer with
        that text already in view rather than reaching for its own
        general-knowledge category language and having a citation bolted on
        after the fact. ``None``/empty is a no-op: no message is appended,
        and the prompt is byte-identical to before #105.

        ``document_facts`` (#86): when non-empty, a
        ``_DOCUMENT_FACT_CONTEXT_PROMPT_TEMPLATE`` message carrying this
        patient's own ingested document facts (their literal citation quotes
        only) is appended AFTER the
        guideline-excerpts message (if any) and BEFORE
        ``_FINAL_REASON_PROMPT``, so the model can answer from an ingested
        document instead of reporting no data exists for it.
        ``None``/empty is a no-op: no message is appended, and the prompt is
        byte-identical to before #86.

        When ``self._ollama`` exposes ``chat_stream`` (the real
        ``OllamaClient``), the reasoning call streams: each delta is yielded
        as a ``ReasoningDelta`` event as it arrives, and the reasoning text
        fed to the extraction call is assembled from exactly those deltas --
        byte-identical to what a plain ``chat`` call would have returned
        (see ``OllamaClient.chat_stream``'s docstring). Falls back to one
        blocking ``chat`` call with no ``ReasoningDelta`` events for a
        double that only implements ``chat`` (every fake in
        ``tests/test_planner.py`` and the eval runner) -- this is what keeps
        those 18 direct-``run()`` tests and the eval replay suite green.
        """
        reason_messages = list(messages)
        if guideline_excerpts:
            _append_context_message(
                reason_messages,
                _GUIDELINE_CONTEXT_PROMPT_TEMPLATE,
                excerpts="\n\n".join(guideline_excerpts),
            )
        if document_facts:
            _append_context_message(
                reason_messages,
                _DOCUMENT_FACT_CONTEXT_PROMPT_TEMPLATE,
                facts="\n".join(f"- {fact.quote_or_value}" for fact in document_facts),
            )
        reason_messages.append({"role": "user", "content": _FINAL_REASON_PROMPT})
        chat_stream = getattr(self._ollama, "chat_stream", None)
        if chat_stream is not None:
            reasoning_parts: list[str] = []
            for delta in chat_stream(reason_messages):
                reasoning_parts.append(delta)
                yield ReasoningDelta(delta)
            reasoning = "".join(reasoning_parts)
        else:
            reasoning = self._ollama.chat(reason_messages)
        extract_messages = messages + [
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": _FINAL_EXTRACT_PROMPT},
        ]
        return self._ollama.extract(extract_messages, FinalAnswer)

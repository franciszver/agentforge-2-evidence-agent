"""Hermetic tests for the planner loop (P2.8).

Both the Ollama client and the tool registry are faked -- no HTTP, no
Ollama, no OpenEMR. ``_ScriptedOllamaClient`` returns a pre-scripted
sequence of ``PlannerDecision``s (one per turn), so each test asserts
exactly the loop behaviour it names rather than depending on real model
output. The offline eval suite (``evals/cases/tool_selection/`` + the P4.7
record/replay runner, ``evals/test_cases.py``) exercises the real qwen3:4b
model's tool-selection behavior via committed recordings.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from app.ollama_client import LlmCallStats
from app.openemr_client import ErrorCategory, OpenEmrApiError
from app.planner import _FEW_SHOT_EXAMPLES, _FINAL_REASON_PROMPT, Planner, ToolSpec
from app.quarantine import REDACTED_SENTINEL, QuarantineSummary
from app.schemas.ingestion import Citation
from app.schemas.planner import FinalAnswer, PlannerAction, PlannerDecision, ToolName
from app.schemas.tools import (
    GetMedicationsInput,
    GetRecentLabsInput,
    MedicationItem,
    MedicationsOutput,
    RecentLabsOutput,
)


class _ScriptedOllamaClient:
    """Fake ``OllamaClient`` dispatching ``extract`` by requested schema.

    ``PlannerDecision`` extractions return the next scripted decision;
    ``QuarantineSummary`` and ``FinalAnswer`` extractions return canned values
    (the ``FinalAnswer`` echoes the last scripted decision's ``final_answer``
    so the two-call finalize step reproduces the intended answer). ``chat``
    (the reasoning half of the two-call answer) returns a fixed string.
    ``calls`` records only the PlannerDecision extractions.
    """

    def __init__(self, decisions: list[PlannerDecision]) -> None:
        self._decisions = list(decisions)
        self.calls: list[list[dict[str, str]]] = []
        self.chat_calls: list[list[dict[str, str]]] = []
        self._last_final_answer = ""
        # Mirrors the real ``OllamaClient.call_stats`` side channel (#149) so
        # tests can assert ``PlannerResult.llm_calls`` is read from it.
        self.call_stats: list[LlmCallStats] = []

    def _record_call_stats(self) -> None:
        self.call_stats.append(
            LlmCallStats(model="qwen3:4b", start_ts=0.0, end_ts=0.1, ok=True, tokens_in=10, tokens_out=5)
        )

    def extract(self, messages: list[dict[str, str]], schema: type):
        self._record_call_stats()
        if schema is QuarantineSummary:
            return QuarantineSummary(summary="quarantined summary")
        if schema is FinalAnswer:
            return FinalAnswer(answer=self._last_final_answer)
        self.calls.append(messages)
        if not self._decisions:
            raise AssertionError("scripted decisions exhausted -- planner looped too many times")
        decision = self._decisions.pop(0)
        if decision.final_answer:
            self._last_final_answer = decision.final_answer
        return decision

    def chat(self, messages: list[dict[str, str]], *, options=None) -> str:
        self._record_call_stats()
        self.chat_calls.append(messages)
        return "reasoning"


class _AlwaysCallToolOllamaClient:
    """Fake ``OllamaClient``: always decides to call the same tool -- used to
    drive the max-turns guard without needing an unbounded script."""

    def __init__(self, tool: ToolName) -> None:
        self._tool = tool
        self.call_count = 0

    def extract(self, messages: list[dict[str, str]], schema: type) -> PlannerDecision:
        self.call_count += 1
        return PlannerDecision(action=PlannerAction.CALL_TOOL, tool=self._tool, reason="looping")

    def chat(self, messages: list[dict[str, str]], *, options=None) -> str:
        return "reasoning"


def _fake_medications_spec(fn: MagicMock) -> ToolSpec:
    return ToolSpec(description="fake medications tool", input_schema=GetMedicationsInput, func=fn)


def _fake_labs_spec(fn: MagicMock) -> ToolSpec:
    return ToolSpec(description="fake labs tool", input_schema=GetRecentLabsInput, func=fn)


BOUND_PATIENT_ID = 42


def _make_planner(ollama_client: Any, registry: dict[ToolName, ToolSpec], max_turns: int = 6) -> Planner:
    return Planner(
        ollama_client=ollama_client,
        openemr_client=object(),
        token="tok",
        patient_id=BOUND_PATIENT_ID,
        registry=registry,
        max_turns=max_turns,
    )


# --- single tool call per turn ----------------------------------------------


def test_single_tool_call_then_answer_returns_trace_of_one_call():
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[MedicationItem(name="Lisinopril", dose="", route="", status="active")]))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}

    decisions = [
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="the question asks about meds"),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="She is on Lisinopril.", reason="medication list answers the question"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    assert result.answer == "She is on Lisinopril."
    assert len(result.trace) == 1
    assert result.trace[0].tool == ToolName.GET_MEDICATIONS
    assert result.trace[0].error is None
    medications_fn.assert_called_once_with(planner._openemr, "tok", BOUND_PATIENT_ID)


# --- multi-turn --------------------------------------------------------------


def test_multi_turn_two_tool_calls_then_answer_returns_ordered_trace_of_two():
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[]))
    labs_fn = MagicMock(return_value=RecentLabsOutput(items=[]))
    registry = {
        ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn),
        ToolName.GET_RECENT_LABS: _fake_labs_spec(labs_fn),
    }

    decisions = [
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="check meds first"),
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_RECENT_LABS, reason="then check labs"),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="No active meds; no recent labs.", reason="both sections empty"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("Anything notable about meds or labs?")

    assert result.answer == "No active meds; no recent labs."
    assert [call.tool for call in result.trace] == [ToolName.GET_MEDICATIONS, ToolName.GET_RECENT_LABS]
    medications_fn.assert_called_once()
    labs_fn.assert_called_once()


# --- max-turns guard ----------------------------------------------------------


def test_max_turns_guard_stops_looping_and_returns_best_effort_answer():
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[]))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    ollama = _AlwaysCallToolOllamaClient(ToolName.GET_MEDICATIONS)
    planner = _make_planner(ollama, registry, max_turns=3)

    result = planner.run("What meds is she on?")

    assert ollama.call_count == 3
    assert len(result.trace) == 3
    assert result.answer != ""
    assert isinstance(result.answer, str)


# --- patient-context binding ---------------------------------------------------


def test_smuggled_divergent_patient_id_is_refused_loudly_not_silently_run():
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[]))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}

    decisions = [
        # The model tries to smuggle a different patient_id into tool_args.
        PlannerDecision(
            action=PlannerAction.CALL_TOOL,
            tool=ToolName.GET_MEDICATIONS,
            tool_args={"patient_id": "999999"},
            reason="attempting cross-patient access",
        ),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="done", reason="done"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    # The tool is NOT dispatched -- the binding violation is refused before
    # any patient data is fetched (loud + auditable, not a silent drop).
    medications_fn.assert_not_called()
    # The refusal is recorded in the trace as a typed, auditable category,
    # and carries NO record content (zero PHI on refusal).
    assert len(result.trace) == 1
    refusal = result.trace[0]
    assert refusal.tool == ToolName.GET_MEDICATIONS
    assert refusal.result is None
    assert refusal.error == "patient_binding_violation"
    assert refusal.args == {}
    # The loop continues rather than crashing.
    assert result.answer == "done"


def test_tool_args_are_filtered_to_the_tools_own_input_schema_fields():
    labs_fn = MagicMock(return_value=RecentLabsOutput(items=[]))
    registry = {ToolName.GET_RECENT_LABS: _fake_labs_spec(labs_fn)}

    decisions = [
        PlannerDecision(
            action=PlannerAction.CALL_TOOL,
            tool=ToolName.GET_RECENT_LABS,
            tool_args={"limit": "3", "bogus_field": "x"},
            reason="last three labs",
        ),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="done", reason="done"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    planner.run("What are her last three A1c values?")

    labs_fn.assert_called_once_with(planner._openemr, "tok", BOUND_PATIENT_ID, limit=3)


# --- tool error handling -------------------------------------------------------


def test_tool_error_is_surfaced_without_crashing_the_loop():
    medications_fn = MagicMock(side_effect=OpenEmrApiError(ErrorCategory.FORBIDDEN, "forbidden"))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}

    decisions = [
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="check meds"),
        PlannerDecision(
            action=PlannerAction.ANSWER,
            final_answer="I couldn't retrieve the medication list (access denied).",
            reason="tool call failed",
        ),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    assert result.answer == "I couldn't retrieve the medication list (access denied)."
    assert len(result.trace) == 1
    assert result.trace[0].tool == ToolName.GET_MEDICATIONS
    assert result.trace[0].result is None
    assert result.trace[0].error is not None
    assert "forbidden" in result.trace[0].error.lower()


# --- system prompt sanity -------------------------------------------------------


def test_system_prompt_sent_to_ollama_includes_no_think_and_every_registered_tool_name():
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[]))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    decisions = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="ok", reason="ok")]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    planner.run("What meds is she on?")

    first_call_messages = ollama.calls[0]
    system_message = next(m["content"] for m in first_call_messages if m["role"] == "system")
    assert "/no_think" in system_message
    assert ToolName.GET_MEDICATIONS.value in system_message
    assert str(BOUND_PATIENT_ID) in system_message


def test_few_shot_examples_include_a_vitals_domain_example():
    """Issue #93 (fix 4/4, mitigation): before this fix, every OTHER
    domain tool (medications, encounters, labs, allergies) had a dedicated
    few-shot example demonstrating when to call it, but vitals had none --
    a plausible contributor to the observed live nondeterminism where an
    identical vitals-needing question (e.g. the bp-stage2-question eval
    case's "What was his last blood pressure reading...") sometimes skipped
    ``get_vitals`` and sometimes did not. This does not PROVE the mechanism
    (that would require live GPU-level tracing -- see the PR body's honest
    accounting of what is confirmed vs. hypothesis); it pins the mitigation
    actually shipped: a concrete vitals example is present, in the same
    ``call_tool`` -> ``get_vitals`` shape as every other domain example."""
    assert '"tool": "get_vitals"' in _FEW_SHOT_EXAMPLES
    assert "blood pressure" in _FEW_SHOT_EXAMPLES.lower()


# --- verifier-only raw channel + safety boundary (P3.2 / #130) -----------------


def test_raw_results_carry_unredacted_values_while_trace_stays_quarantined():
    """The hard safety boundary: the RAW (un-redacted) tool output travels
    ONLY on ``PlannerResult.raw_results`` (the verifier-only channel the P3.2
    citation checker reads). The client-facing ``ToolCallTrace.result`` -- which
    feeds the SSE stream + observability -- still only ever sees the
    quarantined skeleton, so raw record free-text never leaks there."""
    medications_fn = MagicMock(
        return_value=MedicationsOutput(items=[MedicationItem(name="Lisinopril", dose="10mg", route="oral", status="active")])
    )
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    decisions = [
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="meds"),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="She is on Lisinopril.", reason="done"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    # raw_results is aligned 1:1 with the trace and carries the un-redacted name.
    assert len(result.raw_results) == len(result.trace) == 1
    assert result.raw_results[0]["items"][0]["name"] == "Lisinopril"
    # The client-facing trace has the name REDACTED, never the raw value.
    trace_result = result.trace[0].result
    assert trace_result["data"]["items"][0]["name"] == REDACTED_SENTINEL
    assert "Lisinopril" not in str(trace_result)


def test_raw_results_hold_none_for_a_refused_call_keeping_positional_alignment():
    """A binding-violation refusal produces a trace entry but no output; the
    raw channel carries ``None`` at the same position so call_N still lines up."""
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[]))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    decisions = [
        PlannerDecision(
            action=PlannerAction.CALL_TOOL,
            tool=ToolName.GET_MEDICATIONS,
            tool_args={"patient_id": "999999"},
            reason="cross-patient",
        ),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="done", reason="done"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    medications_fn.assert_not_called()
    assert len(result.raw_results) == len(result.trace) == 1
    assert result.raw_results[0] is None
    assert result.trace[0].error == "patient_binding_violation"


# --- span emission: tool timing + llm call stats (#149) ------------------------


def test_tool_call_trace_carries_start_and_end_timestamps_for_a_successful_dispatch():
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[]))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    decisions = [
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="meds"),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="done", reason="done"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    call = result.trace[0]
    assert call.start_ts > 0
    assert call.end_ts >= call.start_ts


def test_tool_call_trace_carries_timestamps_on_the_error_path_too():
    medications_fn = MagicMock(side_effect=OpenEmrApiError(ErrorCategory.FORBIDDEN, "forbidden"))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    decisions = [
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="meds"),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="done", reason="done"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    call = result.trace[0]
    assert call.start_ts > 0
    assert call.end_ts >= call.start_ts


def test_tool_call_trace_carries_timestamps_on_a_binding_violation_refusal():
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[]))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    decisions = [
        PlannerDecision(
            action=PlannerAction.CALL_TOOL,
            tool=ToolName.GET_MEDICATIONS,
            tool_args={"patient_id": "999999"},
            reason="cross-patient",
        ),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="done", reason="done"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    call = result.trace[0]
    assert call.start_ts > 0
    assert call.end_ts >= call.start_ts


def test_planner_result_collects_llm_call_stats_from_the_ollama_client():
    medications_fn = MagicMock(return_value=MedicationsOutput(items=[]))
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    decisions = [
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="meds"),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="done", reason="done"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry)

    result = planner.run("What meds is she on?")

    # Every extract()/chat() call the loop made (2 PlannerDecision extracts +
    # the two-call finalize: 1 chat + 1 extract) shows up as an llm call.
    assert result.llm_calls == ollama.call_stats
    assert len(result.llm_calls) == 4


def test_planner_result_llm_calls_defaults_to_empty_list_for_a_client_without_call_stats():
    # A minimal fake that has no ``call_stats`` attribute at all must not
    # crash the planner -- the field degrades to an empty list rather than
    # raising AttributeError.
    class _BareOllamaClient:
        def extract(self, messages, schema):
            if schema is QuarantineSummary:
                return QuarantineSummary(summary="s")
            if schema is FinalAnswer:
                return FinalAnswer(answer="done")
            return PlannerDecision(action=PlannerAction.ANSWER, final_answer="done", reason="done")

        def chat(self, messages, *, options=None) -> str:
            return "reasoning"

    registry: dict[ToolName, ToolSpec] = {}
    planner = _make_planner(_BareOllamaClient(), registry)

    result = planner.run("anything?")

    assert result.llm_calls == []


# --- resolve_patient_name (#224 name-binding) --------------------------------
#
# Best-effort resolution of the bound patient's own display name, for the
# #224 cross-patient guard signals (app.extraction
# .detect_foreign_patient_reference). A single demographics-only round trip
# (app.tools.patient_summary.get_patient_name) via the planner's own
# openemr_client/token/patient_id -- no new capability, just a getattr-duck-
# typed optional method (same pattern as run_streaming) that app.chat's
# conversation-creation wiring calls once per new conversation.


def test_resolve_patient_name_returns_first_and_last_name(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(
                200,
                json={
                    "validationErrors": [],
                    "internalErrors": [],
                    "data": [
                        {
                            "pid": BOUND_PATIENT_ID,
                            "fname": "Wanda",
                            "lname": "Moore",
                            "DOB": "1950-01-01",
                            "sex": "Female",
                            "uuid": "u1",
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected request: {request.url.path}")

    planner = Planner(
        ollama_client=object(),
        openemr_client=make_openemr_client(handler),
        token="tok",
        patient_id=BOUND_PATIENT_ID,
        registry={},
    )

    assert planner.resolve_patient_name() == "Wanda Moore"


def test_resolve_patient_name_returns_none_when_patient_not_found(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(200, json={"validationErrors": [], "internalErrors": [], "data": []})
        raise AssertionError(f"unexpected request: {request.url.path}")

    planner = Planner(
        ollama_client=object(),
        openemr_client=make_openemr_client(handler),
        token="tok",
        patient_id=BOUND_PATIENT_ID,
        registry={},
    )

    assert planner.resolve_patient_name() is None


# --- resolve_patient_roster (#237 roster-based cross-patient detection) -----
#
# Best-effort every-OTHER-patient's display name, for the roster-based
# "switch to <Name>" signal (app.extraction.detect_foreign_patient_reference)
# -- same getattr-duck-typed OPTIONAL-capability pattern as
# resolve_patient_name, called lazily (only when a candidate name
# construction actually matched) rather than at conversation-creation time --
# see app.chat's wiring comment for why.


def test_resolve_patient_roster_returns_every_other_patients_name(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(
                200,
                json={
                    "validationErrors": [],
                    "internalErrors": [],
                    "data": [
                        {
                            "pid": BOUND_PATIENT_ID,
                            "fname": "Wanda",
                            "lname": "Moore",
                            "DOB": "1950-01-01",
                            "sex": "Female",
                            "uuid": "u1",
                        },
                        {
                            "pid": BOUND_PATIENT_ID + 1,
                            "fname": "Bob",
                            "lname": "Smith",
                            "DOB": "1960-01-01",
                            "sex": "Male",
                            "uuid": "u2",
                        },
                    ],
                },
            )
        raise AssertionError(f"unexpected request: {request.url.path}")

    planner = Planner(
        ollama_client=object(),
        openemr_client=make_openemr_client(handler),
        token="tok",
        patient_id=BOUND_PATIENT_ID,
        registry={},
    )

    assert planner.resolve_patient_roster() == ["Bob Smith"]


def test_resolve_patient_roster_returns_empty_list_on_api_error(make_openemr_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/default/api/patient":
            return httpx.Response(403, json={"error": "insufficient_scope"})
        raise AssertionError(f"unexpected request: {request.url.path}")

    planner = Planner(
        ollama_client=object(),
        openemr_client=make_openemr_client(handler),
        token="tok",
        patient_id=BOUND_PATIENT_ID,
        registry={},
    )

    assert planner.resolve_patient_roster() == []


# --- issue #105: guideline-corpus retrieval feeds answer composition --------
#
# Root cause: `Planner.run`/`run_streaming` never saw retrieved guideline
# text at all before this fix -- retrieval only ran AFTER the planner had
# already composed its final answer (see `app.chat._stream_chat`'s pre-#105
# ordering), so the model reached for its own general-knowledge category
# language (e.g. "elevated blood pressure") instead of the guideline's own
# category name (e.g. "Stage 2 hypertension") -- a genuine, verbatim
# citation ended up attached to prose that used the wrong category name for
# what it cited. The fix threads retrieved guideline chunk TEXT into the
# free-text reasoning call that composes the answer (`_finalize_answer_
# streaming`), as an extra `guideline_excerpts` parameter on `run`/
# `run_streaming`.


def test_finalize_answer_includes_guideline_excerpts_in_the_reasoning_call():
    """When `guideline_excerpts` is passed to `run`, the free-text reasoning
    call (the "chat" half of the two-call finalize) must receive that text --
    so the model can use the guideline's own category language when
    composing the answer, instead of finding out about it only after the
    fact via a bolted-on citation."""
    decisions = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry={})

    guideline_text = (
        "Stage 2 hypertension: systolic 140 mmHg or higher OR diastolic 90 mmHg or higher."
    )
    planner.run("What category is 148/94 mmHg?", guideline_excerpts=[guideline_text])

    assert ollama.chat_calls, "expected the reasoning (chat) call to have run"
    reasoning_messages = ollama.chat_calls[-1]
    joined = " ".join(message["content"] for message in reasoning_messages)
    assert guideline_text in joined, (
        "the retrieved guideline excerpt must reach the reasoning call that "
        "composes the answer, not just post-hoc citation attachment"
    )


def test_finalize_answer_omits_guideline_context_block_when_no_excerpts_given():
    """No guideline_excerpts (the default, e.g. a chart-data-only question)
    must be a complete no-op -- the reasoning call's prompt stays exactly
    what it was before #105."""
    decisions = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry={})

    planner.run("What meds is she on?")

    reasoning_messages = ollama.chat_calls[-1]
    assert reasoning_messages[-1]["content"] == _FINAL_REASON_PROMPT


# --- issue #86: ingested document facts feed answer composition ------------
#
# Root cause (DEMO_SCRIPT.md beat 3): `app.chat.get_patient_fact_provider`
# reads `LocalIngestionStore.list_citations_for_patient` -- but only AFTER
# `Planner.run`/`run_streaming` already returned, feeding ONLY the post-hoc
# claim-extraction/verification step (P3.9a, issue #46). The planner itself
# never sees a patient's own ingested lab/intake-form facts while composing
# its answer, so a question only answerable from an ingested PDF gets
# answered as if the document doesn't exist (chart tools alone). This fix
# mirrors #105's `guideline_excerpts` mechanism exactly: retrieved document
# facts are threaded into the SAME free-text reasoning call
# (`_finalize_answer_streaming`) via a new, purely-additive `document_facts`
# parameter -- appended to the reasoning messages ONLY when non-empty, so a
# turn with nothing ingested (every citation_present eval case today) is
# byte-identical to before this fix.


_A1C_FACT = Citation(
    source_type="lab_pdf",
    source_id="deadbeef" * 4,
    page_or_section="page 1",
    field_or_chunk_id="Hemoglobin A1c#page1-row0",
    quote_or_value="Hemoglobin A1c: 5.4",
)

# Mirrors tests/test_ingestion.py's deliberately-unreadable-field fixture:
# Creatinine's value is legible but its collection_date is not, so
# `_quote_for_row` never mentions a date at all for this row.
_REDACTED_CREATININE_FACT = Citation(
    source_type="lab_pdf",
    source_id="cafebabe" * 4,
    page_or_section="page 2",
    field_or_chunk_id="Creatinine#page2-row7",
    quote_or_value="Creatinine: 0.9",
)


@pytest.mark.parametrize(
    ("fact", "question", "check_no_fabricated_date"),
    [
        pytest.param(_A1C_FACT, "What is her A1c?", False, id="includes_document_facts_in_the_reasoning_call"),
        pytest.param(
            _REDACTED_CREATININE_FACT,
            "What was the collection date for his creatinine result?",
            True,
            id="never_states_a_field_the_document_fact_quote_omits",
        ),
    ],
)
def test_finalize_answer_reasoning_call_reflects_document_facts(fact, question, check_no_fabricated_date):
    """When `document_facts` is passed to `run`, the free-text reasoning call
    must receive each fact's literal citation quote -- so the model can
    answer from the patient's own ingested document instead of reporting
    "no lab results recorded" when a relevant document actually exists.

    The no-fabrication contract (`app.ingestion._quote_for_row`) guarantees
    an unreadable field is simply ABSENT from a fact's citation quote (e.g.
    the Creatinine case's illegible collection date). Since the reasoning
    call only ever sees that literal quote text, verbatim, it structurally
    cannot pass a fabricated collection date to the model."""
    decisions = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]
    ollama = _ScriptedOllamaClient(decisions)
    planner = _make_planner(ollama, registry={})

    planner.run(question, document_facts=[fact])

    assert ollama.chat_calls, "expected the reasoning (chat) call to have run"
    reasoning_messages = ollama.chat_calls[-1]
    joined = " ".join(message["content"] for message in reasoning_messages)
    assert fact.quote_or_value in joined, (
        "the ingested document fact's literal quote must reach the reasoning "
        "call that composes the answer"
    )
    if check_no_fabricated_date:
        # No fabricated date has any way to appear: the quote fed to the
        # model names only the test and its value, never a date the source
        # document never actually gave up.
        assert "2026" not in joined and "collection_date" not in joined.lower().replace("_", "")


def test_finalize_answer_omits_document_fact_context_block_when_no_facts_given():
    """No document_facts (the default -- every turn today, since nothing in
    the citation_present eval suite has an ingested document) must be a
    complete no-op: the reasoning call's prompt is BYTE-IDENTICAL to a call
    with no document_facts argument at all -- the entire safety argument for
    this being a prompt-neutral change for every existing case."""
    decisions_a = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]
    decisions_b = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]
    ollama_a = _ScriptedOllamaClient(decisions_a)
    ollama_b = _ScriptedOllamaClient(decisions_b)
    planner_a = _make_planner(ollama_a, registry={})
    planner_b = _make_planner(ollama_b, registry={})

    planner_a.run("What meds is she on?")
    planner_b.run("What meds is she on?", document_facts=None)

    assert ollama_a.chat_calls[-1] == ollama_b.chat_calls[-1]
    assert ollama_b.chat_calls[-1][-1]["content"] == _FINAL_REASON_PROMPT

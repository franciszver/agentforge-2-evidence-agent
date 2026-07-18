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

from app.ollama_client import LlmCallStats
from app.openemr_client import ErrorCategory, OpenEmrApiError
from app.planner import Planner, ToolSpec
from app.quarantine import REDACTED_SENTINEL, QuarantineSummary
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

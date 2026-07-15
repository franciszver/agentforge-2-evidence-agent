"""Hermetic tests for the planner loop (P2.8).

Both the Ollama client and the tool registry are faked -- no HTTP, no
Ollama, no OpenEMR. ``_ScriptedOllamaClient`` returns a pre-scripted
sequence of ``PlannerDecision``s (one per turn), so each test asserts
exactly the loop behaviour it names rather than depending on real model
output. A separate ``@pytest.mark.integration`` eval (``evals/tool_selection/``)
exercises the real qwen3:4b model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.openemr_client import ErrorCategory, OpenEmrApiError
from app.planner import Planner, ToolSpec
from app.quarantine import QuarantineSummary
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

    def extract(self, messages: list[dict[str, str]], schema: type):
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

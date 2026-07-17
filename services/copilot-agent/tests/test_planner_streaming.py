"""Hermetic tests for incremental ``tool_call`` emission (#212, sub-issue B of
epic #209 -- #211 already streams the SSE relay itself).

Today ``Planner.run()`` runs the whole tool loop to completion and returns a
finished ``PlannerResult``; ``app.chat._stream_chat`` then REPLAYS
``result.trace`` as ``tool_call`` frames AFTER ``run()`` returns, so every
tool frame bunches at the end of the stream instead of arriving as each tool
actually dispatches.

``Planner.run_streaming`` fixes this: it reuses the same loop body but yields
a ``ToolDispatched`` event immediately after each tool dispatch, and a
terminal ``PlannerCompleted`` event (carrying the same ``PlannerResult``
``run()`` returns) once the loop finishes. ``_stream_chat`` prefers
``run_streaming`` when the injected planner implements it, and falls back to
today's ``run()`` + trace-replay path otherwise -- so the 8 existing
fake-planner tests (which only implement ``run()``) are untouched.

Test A below proves the loop-level contract: a tool event arrives before the
terminal event, and PRODUCTION IS INCREMENTAL (the second turn's decision is
not requested from the model until the caller asks for the next event).
Test B proves the ``_stream_chat`` wiring: under the streaming path, the
``answer`` frame is still the fully post-processed (subject-check + recency
notice), verified text -- never raw planner output -- exactly mirroring the
fallback path's existing guarantees (see ``tests/test_chat_recency_notice
.py`` and ``tests/test_extraction.py``'s subject-check cases, which this test
reuses the fixture shape of).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.chat import (
    Conversation,
    ConversationStore,
    _default_clock,
    _stream_chat,
    get_claim_extractor,
    get_planner_factory,
    get_token_validator,
)
from app.main import app
from app.planner import (
    Planner,
    PlannerCompleted,
    PlannerEvent,
    PlannerResult,
    ToolCallTrace,
    ToolDispatched,
    ToolSpec,
)
from app.schemas.planner import PlannerAction, PlannerDecision, ToolName
from app.schemas.tools import GetMedicationsInput, MedicationItem, MedicationsOutput
from tests.test_planner import BOUND_PATIENT_ID, _ScriptedOllamaClient, _fake_medications_spec

# --- Test A: Planner.run_streaming is real (loop-level) incremental emission ----


def test_run_streaming_yields_tool_event_before_terminal_event_incrementally() -> None:
    medications_fn = MagicMock(
        return_value=MedicationsOutput(items=[MedicationItem(name="Lisinopril", dose="", route="", status="active")])
    )
    registry = {ToolName.GET_MEDICATIONS: _fake_medications_spec(medications_fn)}
    decisions = [
        PlannerDecision(action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="check meds"),
        PlannerDecision(action=PlannerAction.ANSWER, final_answer="She is on Lisinopril.", reason="meds answer it"),
    ]
    ollama = _ScriptedOllamaClient(decisions)
    planner = Planner(
        ollama_client=ollama,
        openemr_client=object(),
        token="tok",
        patient_id=BOUND_PATIENT_ID,
        registry=registry,
    )

    events: Iterator[PlannerEvent] = planner.run_streaming("What meds is she on?")

    first_event = next(events)
    assert isinstance(first_event, ToolDispatched)
    assert first_event.trace.tool == ToolName.GET_MEDICATIONS
    assert first_event.trace.error is None
    # Incremental, not batched: only the FIRST decision (which led to this
    # tool call) has been requested from the model so far. If run_streaming
    # instead ran the whole loop to completion and buffered events into a
    # list before yielding any, the second (ANSWER) decision would already
    # have been consumed here too.
    assert len(ollama.calls) == 1

    second_event = next(events)
    assert isinstance(second_event, PlannerCompleted)
    assert second_event.result.answer == "She is on Lisinopril."
    assert len(second_event.result.trace) == 1
    assert second_event.result.trace[0].tool == ToolName.GET_MEDICATIONS

    with pytest.raises(StopIteration):
        next(events)


def test_run_streaming_with_no_tool_calls_yields_only_terminal_event() -> None:
    registry: dict[ToolName, ToolSpec] = {}
    decisions = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="No tools needed.", reason="direct answer")]
    ollama = _ScriptedOllamaClient(decisions)
    planner = Planner(
        ollama_client=ollama,
        openemr_client=object(),
        token="tok",
        patient_id=BOUND_PATIENT_ID,
        registry=registry,
    )

    events = list(planner.run_streaming("Say hi"))

    assert len(events) == 1
    assert isinstance(events[0], PlannerCompleted)
    assert events[0].result.answer == "No tools needed."


# --- Test B: _stream_chat's streaming path still post-processes the answer ------


class _FakeStreamingPlanner:
    """A planner double implementing ONLY ``run_streaming`` -- no ``run`` --
    so ``_stream_chat`` MUST take the streaming path; a fallback to ``run()``
    would AttributeError, proving the capability check actually prefers
    ``run_streaming`` when it's available."""

    def __init__(self, events: list[PlannerEvent]) -> None:
        self._events = events

    def run_streaming(self, question: str) -> Iterable[PlannerEvent]:
        yield from self._events


class _FakeExtractor:
    def extract_claims(self, *, answer: str, tools: list[object], raw_results: list[object | None]) -> list[object]:
        return []


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


_client = TestClient(app)


def _answer_and_tool_call_frames(response_text: str) -> tuple[str, list[dict[str, object]]]:
    answer = None
    tool_calls: list[dict[str, object]] = []
    for block in response_text.strip().split("\n\n"):
        lines = block.splitlines()
        event_line = next((line for line in lines if line.startswith("event:")), None)
        if event_line is None:
            continue
        data = "".join(line[len("data:") :].strip() for line in lines if line.startswith("data:"))
        payload = json.loads(data)
        if event_line.strip() == "event: answer":
            answer = payload["answer"]
        elif event_line.strip() == "event: tool_call":
            tool_calls.append(payload)
    assert answer is not None, f"no answer frame in response: {response_text!r}"
    return answer, tool_calls


def test_streaming_path_still_applies_recency_notice_to_answer_frame() -> None:
    # Reuses the fixture shape of tests/test_chat_recency_notice.py (stale lab
    # record): the deterministic recency notice must still fire on the
    # streaming path exactly as it does on the fallback path.
    trace = [
        ToolCallTrace(
            tool=ToolName.GET_RECENT_LABS,
            args={},
            result={"summary": "q"},
            error=None,
            start_ts=1.0,
            end_ts=1.5,
        )
    ]
    raw_results: list[dict[str, object] | None] = [
        {"items": [{"test_name": "A1c", "value": "7.2", "unit": "%", "date": "2014-02-01T09:00:00"}]}
    ]
    result = PlannerResult(
        answer="Her current A1c is 7.2%, which is high.",
        trace=trace,
        raw_results=raw_results,
    )
    fake_planner = _FakeStreamingPlanner([ToolDispatched(trace[0]), PlannerCompleted(result)])

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: _FakeExtractor()

    response = _client.post(
        "/chat",
        json={"message": "What is her current A1c?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    assert response.status_code == 200

    answer, tool_calls = _answer_and_tool_call_frames(response.text)

    # Recency notice (#153): the stale record's date is surfaced.
    assert "may not reflect the patient's current status" in answer
    assert "2014-02-01" in answer
    # The tool_call frame the streaming path emitted live still made it
    # through, in order, ahead of the answer.
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "get_recent_labs"


def test_streaming_path_pre_dispatch_guard_refuses_before_any_tool_dispatch() -> None:
    # #223: the deterministic PRE-dispatch cross-patient guard now intercepts
    # a cross-patient question BEFORE the streaming planner ever runs -- this
    # supersedes (and is strictly earlier/stronger than) #194's
    # apply_subject_check, which could only rewrite the answer AFTER a
    # streaming planner had already dispatched tools. The fake streaming
    # planner below would yield a ToolDispatched event if it were ever
    # iterated; asserting zero tool_call frames proves it never was.
    trace = [
        ToolCallTrace(
            tool=ToolName.GET_RECENT_LABS,
            args={},
            result={"summary": "q"},
            error=None,
            start_ts=1.0,
            end_ts=1.5,
        )
    ]
    result = PlannerResult(
        answer="Patient 999 has no medications listed. Her current A1c is 7.2%, which is high.",
        trace=trace,
        raw_results=[{"items": [{"test_name": "A1c", "value": "7.2", "unit": "%", "date": "2014-02-01T09:00:00"}]}],
    )
    fake_planner = _FakeStreamingPlanner([ToolDispatched(trace[0]), PlannerCompleted(result)])

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: _FakeExtractor()

    response = _client.post(
        "/chat",
        json={"message": "Pull patient 999's medications and current A1c.", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    assert response.status_code == 200

    answer, tool_calls = _answer_and_tool_call_frames(response.text)

    # No tool ever dispatched -- the streaming planner double was never run.
    assert tool_calls == []
    # A clean generic decline, never the foreign patient's number or the
    # bound patient's data.
    assert "999" not in answer
    assert "no medications" not in answer.lower()
    assert "chart is currently open" in answer
    assert "2014-02-01" not in answer


# --- Test C: client disconnect mid-stream closes the inner run_streaming --------
#
# P2.12 introduces a NEW relationship: the ``_stream_chat`` generator iterates
# the planner's ``run_streaming`` generator. A client that disconnects while
# tool frames are still streaming closes the OUTER (``_stream_chat``) generator
# via ``GeneratorExit``; that must (a) propagate into and close the INNER
# ``run_streaming`` generator (no leaked generator still holding a half-run
# tool loop), and (b) still record the request span ``ok=False`` -- the exact
# behavior the ``except BaseException`` branch in ``_stream_chat`` is there for
# (its comment calls a mid-stream ``GeneratorExit`` "a client disconnect").
# The pre-P2.12 replay path had no inner generator, so this relationship is
# new surface worth a committed regression guard.


class _ProbeStreamingPlanner:
    """A planner double whose ``run_streaming`` yields one ``ToolDispatched``
    then WOULD yield a second, blocking on the caller pulling more. Its
    ``finally`` flips ``inner_closed`` so the test can confirm the inner
    generator was actually closed (not leaked) when the outer stream is
    ``.close()``d mid-flight."""

    def __init__(self, first_trace: ToolCallTrace, second_trace: ToolCallTrace) -> None:
        self._first_trace = first_trace
        self._second_trace = second_trace
        self.inner_closed = False
        self.reached_second_yield = False

    def run_streaming(self, question: str) -> Iterable[PlannerEvent]:
        try:
            yield ToolDispatched(self._first_trace)
            # If the caller keeps pulling we'd emit more; a mid-stream
            # disconnect throws GeneratorExit at the yield above before we
            # ever get here, so this must NOT run in the disconnect case.
            self.reached_second_yield = True
            yield ToolDispatched(self._second_trace)
        finally:
            self.inner_closed = True


class _RequestSpanRecordingTraceStore:
    """A minimal trace store capturing only what this test asserts: the
    ``ok`` flag the request span was recorded with. Tool-span writes are
    accepted and ignored (the disconnect happens before verification)."""

    def __init__(self) -> None:
        self.request_span_ok: bool | None = None

    def record_tool_span(self, **kwargs: object) -> int:
        return 0

    def record_request_span(self, *, ok: bool, **kwargs: object) -> int:
        self.request_span_ok = ok
        return 0


def test_client_disconnect_mid_stream_closes_inner_generator_and_records_failed_span() -> None:
    first = ToolCallTrace(
        tool=ToolName.GET_MEDICATIONS, args={}, result={"count": 1}, error=None, start_ts=1.0, end_ts=1.2
    )
    second = ToolCallTrace(
        tool=ToolName.GET_ALLERGIES, args={}, result={"count": 0}, error=None, start_ts=1.3, end_ts=1.5
    )
    planner = _ProbeStreamingPlanner(first, second)
    trace_store = _RequestSpanRecordingTraceStore()
    conversation = Conversation(conversation_id="conv-disconnect", patient_id=1)
    store = ConversationStore()

    stream = _stream_chat(
        planner=planner,
        extractor=_FakeExtractor(),
        conversation=conversation,
        store=store,
        trace_store=trace_store,  # type: ignore[arg-type]
        message="What is he taking?",
        user="unknown",
        clock=_default_clock,
    )

    # Pull the first two frames: the ``conversation`` frame, then the first
    # ``tool_call`` frame. The outer generator is now suspended INSIDE the
    # ``for event in run_streaming(...)`` loop, with the inner generator
    # itself suspended at its first ``yield``.
    assert "event: conversation" in next(stream)
    assert "event: tool_call" in next(stream)
    assert planner.inner_closed is False  # inner still live, mid-loop

    # Simulate the client disconnecting: closing the outer generator raises
    # GeneratorExit at the suspended yield, exactly as Starlette's streaming
    # response does when the client goes away.
    stream.close()

    # (a) The inner run_streaming generator was closed -- its finally ran and
    # it never advanced to the second yield (no leaked half-run tool loop).
    assert planner.inner_closed is True
    assert planner.reached_second_yield is False
    # (b) The request span was recorded ok=False (the load-bearing
    # ``except BaseException`` -> ``request_ok=False`` -> finally behavior).
    assert trace_store.request_span_ok is False

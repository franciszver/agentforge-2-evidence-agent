"""Hermetic tests for streaming the planner's free-text reasoning (#213,
sub-issue C of epic #209 -- #211 streams the SSE relay, #212 streams
``tool_call`` frames incrementally).

Today ``Planner._finalize_answer`` reasons in one blocking ``chat()`` call,
then extracts the verified ``FinalAnswer`` via a constrained ``extract()``
call. ``.extract()`` cannot stream (schema decode needs the whole JSON), but
the free-text reasoning `chat()` call can -- so this streams THAT half:
``Planner.run_streaming`` now yields a ``ReasoningDelta`` event per reasoning
token, from a new ``_finalize_answer_streaming`` generator, BEFORE its
terminal ``PlannerCompleted`` event. ``app.chat._stream_chat`` forwards each
as a ``reasoning_delta`` SSE frame.

The owner's non-negotiable UX rule this enforces: the answer bubble must
ONLY EVER show the VERIFIED ``answer`` frame text -- reasoning deltas are
provisional, unverified model text that must render in a separate,
clearly-labeled zone, never the answer slot. Test C below is the direct
regression guard for that separation.

Test A: ``Planner.run_streaming`` emits ``ReasoningDelta`` events (via a
fake ``chat_stream``-bearing Ollama client) before the terminal event, and
falls back to zero such events (plain ``chat()``) for a double that only
implements ``chat`` -- keeping the 18 existing direct-``run()`` tests +
eval replay (none of which model ``chat_stream``) green.

Test B: the run()-equivalence guarantee -- the reasoning fed into the
``extract(FinalAnswer)`` call, and thus the final answer, is byte-identical
whether ``chat_stream`` is available or not.

Test C: ``_stream_chat`` forwards ``ReasoningDelta`` events as
``reasoning_delta`` SSE frames, in order, before the ``answer`` frame -- and
the ``answer`` frame never contains any reasoning-only text.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.chat import get_claim_extractor, get_planner_factory, get_token_validator
from app.main import app
from app.planner import Planner, PlannerCompleted, PlannerEvent, ReasoningDelta
from app.schemas.planner import FinalAnswer, PlannerAction, PlannerDecision
from tests.test_chat_endpoint import _iter_sse_events
from tests.test_planner import BOUND_PATIENT_ID, _ScriptedOllamaClient
from tests.test_planner_streaming import _FakeExtractor

# --- Test A: run_streaming emits ReasoningDelta events (and falls back cleanly) --


class _ChatStreamOllamaClient(_ScriptedOllamaClient):
    """Same scripted-decision behavior as ``_ScriptedOllamaClient``, plus a
    ``chat_stream`` that yields a fixed sequence of reasoning deltas --
    proving the REAL capability-check path (a ``chat_stream``-bearing
    client) emits ``ReasoningDelta`` events."""

    def __init__(self, decisions: list[PlannerDecision], reasoning_deltas: list[str]) -> None:
        super().__init__(decisions)
        self._reasoning_deltas = reasoning_deltas
        self.chat_stream_calls: list[list[dict[str, str]]] = []

    def chat_stream(self, messages: list[dict[str, str]], *, options=None) -> Iterator[str]:
        self._record_call_stats()
        self.chat_stream_calls.append(messages)
        yield from self._reasoning_deltas


def test_run_streaming_emits_reasoning_delta_events_before_the_terminal_event() -> None:
    decisions = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]
    ollama = _ChatStreamOllamaClient(decisions, reasoning_deltas=["Let me ", "check...", " done."])
    planner = Planner(ollama_client=ollama, openemr_client=object(), token="tok", patient_id=BOUND_PATIENT_ID, registry={})

    events: list[PlannerEvent] = list(planner.run_streaming("What meds is she on?"))

    reasoning_events = [e for e in events if isinstance(e, ReasoningDelta)]
    assert [e.text for e in reasoning_events] == ["Let me ", "check...", " done."]
    assert isinstance(events[-1], PlannerCompleted)
    assert events[-1].result.answer == "Answer."
    # Every ReasoningDelta arrives strictly before the terminal event.
    terminal_index = events.index(events[-1])
    assert all(events.index(e) < terminal_index for e in reasoning_events)
    assert len(ollama.chat_stream_calls) == 1


def test_run_streaming_falls_back_to_plain_chat_with_no_reasoning_delta_events() -> None:
    """A double that only implements ``chat`` (no ``chat_stream``) -- e.g.
    the 18 existing ``_ScriptedOllamaClient``-based tests and the eval
    runner's ``ReplayOllamaClient`` -- must produce zero ``ReasoningDelta``
    events, not error."""
    decisions = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]
    ollama = _ScriptedOllamaClient(decisions)
    planner = Planner(ollama_client=ollama, openemr_client=object(), token="tok", patient_id=BOUND_PATIENT_ID, registry={})

    events = list(planner.run_streaming("What meds is she on?"))

    assert not any(isinstance(e, ReasoningDelta) for e in events)
    assert isinstance(events[-1], PlannerCompleted)
    assert events[-1].result.answer == "Answer."


# --- Test B: run()-equivalence guarantee -------------------------------------


class _ExtractRecordingOllamaClient(_ScriptedOllamaClient):
    """Records every ``messages`` list sent to ``extract(FinalAnswer)`` so
    the run()-equivalence guarantee can be asserted directly (not just
    inferred from the final answer text)."""

    def __init__(self, decisions: list[PlannerDecision]) -> None:
        super().__init__(decisions)
        self.finalize_extract_messages: list[list[dict[str, str]]] = []

    def extract(self, messages: list[dict[str, str]], schema: type):
        if schema is FinalAnswer:
            self.finalize_extract_messages.append(messages)
        return super().extract(messages, schema)


class _ExtractRecordingOllamaClientWithChatStream(_ExtractRecordingOllamaClient):
    def __init__(self, decisions: list[PlannerDecision], reasoning_deltas: list[str]) -> None:
        super().__init__(decisions)
        self._reasoning_deltas = reasoning_deltas

    def chat_stream(self, messages: list[dict[str, str]], *, options=None) -> Iterator[str]:
        self._record_call_stats()
        yield from self._reasoning_deltas


def test_run_result_and_extract_messages_are_byte_identical_with_and_without_chat_stream() -> None:
    decisions_plain = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]
    decisions_streaming = [PlannerDecision(action=PlannerAction.ANSWER, final_answer="Answer.", reason="direct")]

    plain = _ExtractRecordingOllamaClient(decisions_plain)
    # Deltas join to "reasoning" -- the exact literal _ScriptedOllamaClient.chat()
    # (the fallback path) always returns, so the assembled reasoning text is
    # identical whichever path assembled it.
    streaming = _ExtractRecordingOllamaClientWithChatStream(decisions_streaming, reasoning_deltas=["rea", "soning"])

    planner_plain = Planner(ollama_client=plain, openemr_client=object(), token="tok", patient_id=BOUND_PATIENT_ID, registry={})
    planner_streaming = Planner(
        ollama_client=streaming, openemr_client=object(), token="tok", patient_id=BOUND_PATIENT_ID, registry={}
    )

    result_plain = planner_plain.run("What meds is she on?")
    result_streaming = planner_streaming.run("What meds is she on?")

    assert result_plain.answer == result_streaming.answer == "Answer."
    assert plain.finalize_extract_messages == streaming.finalize_extract_messages
    assert plain.finalize_extract_messages[0][-2] == {"role": "assistant", "content": "reasoning"}


# --- Test C: _stream_chat forwards reasoning_delta frames, answer stays verified-only --


class _FakeStreamingPlanner:
    def __init__(self, events: list[PlannerEvent]) -> None:
        self._events = events

    def run_streaming(self, question: str) -> Iterable[PlannerEvent]:
        yield from self._events


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


_client = TestClient(app)


def _sse_frames(response_text: str) -> list[tuple[str, dict[str, object]]]:
    # Reuse test_chat_endpoint's block parser; just JSON-decode each frame's
    # data payload here (it returns the raw data string).
    return [(name, json.loads(data)) for name, data in _iter_sse_events(response_text) if name]


def test_stream_chat_emits_reasoning_delta_frames_before_answer_frame_verified_only() -> None:
    from app.planner import PlannerResult

    result = PlannerResult(answer="She is on Lisinopril.", trace=[], raw_results=[])
    fake_planner = _FakeStreamingPlanner(
        [
            ReasoningDelta(text="Let me check "),
            ReasoningDelta(text="the medication list..."),
            PlannerCompleted(result),
        ]
    )

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: _FakeExtractor()

    response = _client.post(
        "/chat",
        json={"message": "What meds is she on?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    assert response.status_code == 200

    frames = _sse_frames(response.text)
    events_in_order = [name for name, _ in frames]

    reasoning_frames = [data for name, data in frames if name == "reasoning_delta"]
    answer_frames = [data for name, data in frames if name == "answer"]

    assert [f["text"] for f in reasoning_frames] == ["Let me check ", "the medication list..."]
    assert len(answer_frames) == 1
    assert answer_frames[0]["answer"] == "She is on Lisinopril."

    # Reasoning frames arrive strictly before the answer frame -- typewriter
    # first, verified pop-in second.
    assert events_in_order.index("reasoning_delta") < events_in_order.index("answer")

    # The hard safety invariant: no reasoning text ever leaks into the
    # answer frame's payload.
    full_answer_text = json.dumps(answer_frames[0])
    assert "Let me check" not in full_answer_text
    assert "the medication list" not in full_answer_text


def test_stream_chat_with_no_reasoning_delta_events_emits_no_reasoning_delta_frames() -> None:
    """A planner double that only implements ``run()`` (all pre-#212 fake
    planners) never emits reasoning_delta frames -- the fallback replay path
    has no such events to replay, exactly like it has no live tool_call
    frames either."""
    from app.planner import PlannerResult

    class _FakeRunOnlyPlanner:
        def run(self, question: str):
            return PlannerResult(answer="No tools needed.", trace=[], raw_results=[])

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: _FakeRunOnlyPlanner())
    app.dependency_overrides[get_claim_extractor] = lambda: _FakeExtractor()

    response = _client.post(
        "/chat",
        json={"message": "Say hi", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    assert response.status_code == 200

    frames = _sse_frames(response.text)
    assert not any(name == "reasoning_delta" for name, _ in frames)

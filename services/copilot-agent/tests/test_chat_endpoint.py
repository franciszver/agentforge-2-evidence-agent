"""Hermetic tests for the POST /chat SSE endpoint (P2.10).

Everything is faked: the token validator and the planner factory are both
injected via FastAPI dependency overrides, so no real OpenEMR or Ollama
service is ever contacted. See ``app/chat.py`` for the seams.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.chat import (
    ChatEvent,
    ConversationStore,
    PatientMismatchError,
    TokenValidationError,
    Turn,
    get_conversation_store,
    get_planner_factory,
    get_token_validator,
)
from app.main import app
from app.planner import PlannerResult, ToolCallTrace
from app.schemas.planner import ToolName


class FakePlanner:
    """Scripted planner double: records the question it was asked and
    returns a fixed trace + answer."""

    def __init__(self, trace: list[ToolCallTrace], answer: str) -> None:
        self._trace = trace
        self._answer = answer
        self.questions: list[str] = []

    def run(self, question: str) -> PlannerResult:
        self.questions.append(question)
        return PlannerResult(answer=self._answer, trace=self._trace)


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _iter_sse_events(text: str) -> list[tuple[str, str]]:
    """Parse ``event: X\\ndata: Y\\n\\n`` blocks into (event, data) pairs."""
    events: list[tuple[str, str]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        events.append((event_name, "\n".join(data_lines)))
    return events


def _conversation_id(response_text: str) -> str:
    """Pull the ``conversation_id`` value out of the ``conversation`` frame."""
    data = next(data for name, data in _iter_sse_events(response_text) if name == "conversation")
    return json.loads(data)["conversation_id"]


def _override_ok_validator() -> None:
    def _validator(token: str) -> None:
        return None

    app.dependency_overrides[get_token_validator] = lambda: _validator


def _override_planner_factory(fake_planner: FakePlanner) -> None:
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)


client = TestClient(app)


def test_stream_emits_tool_call_answer_done_frames_in_order():
    trace = [
        ToolCallTrace(tool=ToolName.GET_MEDICATIONS, args={}, result={"count": 2}, error=None),
    ]
    fake_planner = FakePlanner(trace=trace, answer="She is on lisinopril and metformin.")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "What meds is she on?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _iter_sse_events(response.text)
    event_names = [name for name, _ in events]

    assert "conversation" in event_names
    assert event_names.index("tool_call") < event_names.index("answer")
    assert event_names.index("answer") < event_names.index("done")

    tool_call_data = next(data for name, data in events if name == "tool_call")
    assert "get_medications" in tool_call_data

    answer_data = next(data for name, data in events if name == "answer")
    assert "lisinopril" in answer_data

    assert fake_planner.questions == ["What meds is she on?"]


def test_new_conversation_returns_a_fresh_conversation_id():
    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    conversation_id = _conversation_id(response.text)
    assert conversation_id  # non-empty conversation_id present in the frame


def test_resume_with_same_conversation_id_continues_history():
    fake_planner = FakePlanner(trace=[], answer="first answer")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    first = client.post(
        "/chat",
        json={"message": "first question", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    conversation_id = _conversation_id(first.text)

    second = client.post(
        "/chat",
        json={
            "message": "second question",
            "patient_id": 1,
            "conversation_id": conversation_id,
        },
        headers={"Authorization": "Bearer good-token"},
    )
    assert second.status_code == 200
    second_conversation_id = _conversation_id(second.text)
    assert second_conversation_id == conversation_id

    # The store now holds both turns for this conversation.
    conversation = store.get(conversation_id)
    assert conversation is not None
    assert len(conversation.history) == 2
    assert fake_planner.questions == ["first question", "second question"]


def test_resume_with_mismatched_patient_id_is_rejected():
    fake_planner = FakePlanner(trace=[], answer="first answer")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    first = client.post(
        "/chat",
        json={"message": "first question", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    conversation_id = _conversation_id(first.text)

    second = client.post(
        "/chat",
        json={
            "message": "second question",
            "patient_id": 2,
            "conversation_id": conversation_id,
        },
        headers={"Authorization": "Bearer good-token"},
    )

    assert second.status_code in (400, 409)


def test_missing_token_returns_401_and_never_invokes_planner():
    fake_planner = FakePlanner(trace=[], answer="should not be called")
    _override_planner_factory(fake_planner)
    # No validator override -- default stub still requires a header, but we
    # additionally force a rejecting validator to prove the 401 path.

    def _rejecting_validator(token: str) -> None:
        raise TokenValidationError("no token")

    app.dependency_overrides[get_token_validator] = lambda: _rejecting_validator

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
    )

    assert response.status_code == 401
    assert fake_planner.questions == []


def test_rejected_token_returns_401_and_never_invokes_planner():
    fake_planner = FakePlanner(trace=[], answer="should not be called")
    _override_planner_factory(fake_planner)

    def _rejecting_validator(token: str) -> None:
        raise TokenValidationError("bad token")

    app.dependency_overrides[get_token_validator] = lambda: _rejecting_validator

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer bad-token"},
    )

    assert response.status_code == 401
    assert fake_planner.questions == []


def _dev_bearer(username: str, sub: int, pid: int) -> str:
    """Build a DevAgentToken-shaped bearer (``base64url(payload).sig``).

    Mirrors ``DevAgentToken::mint`` on the PHP side closely enough for the
    agent's best-effort identity read: the payload segment carries the
    ``username``/``sub`` claims; the signature segment is opaque filler (the
    agent does not verify it -- that is the deferred introspection work).
    """
    payload = json.dumps(
        {"sub": sub, "username": username, "pid": pid, "typ": "copilot-dev"}
    ).encode()
    segment = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    return f"{segment}.signature-not-verified"


def test_per_turn_record_captures_user_patient_and_correlation_id():
    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    response = client.post(
        "/chat",
        json={"message": "hi", "patient_id": 7},
        headers={"Authorization": "Bearer " + _dev_bearer("dr.house", 42, 7)},
    )
    assert response.status_code == 200

    conversation_id = _conversation_id(response.text)
    conversation = store.get(conversation_id)
    assert conversation is not None
    assert len(conversation.history) == 1

    turn = conversation.history[0]
    assert isinstance(turn, Turn)
    assert turn.user == "dr.house"
    assert turn.patient_id == 7
    assert turn.correlation_id  # non-empty per-turn id
    assert turn.question == "hi"
    assert turn.answer == "ok"


def test_correlation_id_is_unique_per_turn():
    fake_planner = FakePlanner(trace=[], answer="a")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    headers = {"Authorization": "Bearer " + _dev_bearer("dr.house", 42, 3)}
    first = client.post(
        "/chat", json={"message": "one", "patient_id": 3}, headers=headers
    )
    conversation_id = _conversation_id(first.text)
    client.post(
        "/chat",
        json={"message": "two", "patient_id": 3, "conversation_id": conversation_id},
        headers=headers,
    )

    conversation = store.get(conversation_id)
    assert conversation is not None
    assert len(conversation.history) == 2
    correlation_ids = {turn.correlation_id for turn in conversation.history}
    assert len(correlation_ids) == 2, "each turn must get a distinct correlation id"


def test_unparseable_token_records_unknown_user():
    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    response = client.post(
        "/chat",
        json={"message": "hi", "patient_id": 5},
        headers={"Authorization": "Bearer not-a-real-dev-token"},
    )
    conversation_id = _conversation_id(response.text)
    conversation = store.get(conversation_id)
    assert conversation is not None
    assert conversation.history[0].user == "unknown"


def test_chat_event_enum_matches_frame_names():
    # Sanity: the frame names used above are exactly the ChatEvent values,
    # so the P2.14 UI has one source of truth for the SSE contract.
    assert ChatEvent.CONVERSATION.value == "conversation"
    assert ChatEvent.TOOL_CALL.value == "tool_call"
    assert ChatEvent.ANSWER.value == "answer"
    assert ChatEvent.VERIFICATION.value == "verification"
    assert ChatEvent.DONE.value == "done"


def test_stream_emits_pending_verification_frame_after_answer_before_done():
    # P3.8: every response carries a verification frame (verdict badge /
    # citation chips / warning banner contract). The answer->claims/meds
    # extraction pipeline is not built yet, so the live frame is the *pending*
    # payload (verdict null, no segments, no warnings); the frame's position
    # in the stream and its contract shape are pinned here.
    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    events = _iter_sse_events(response.text)
    event_names = [name for name, _ in events]

    assert "verification" in event_names
    assert event_names.index("answer") < event_names.index("verification")
    assert event_names.index("verification") < event_names.index("done")

    verification_data = json.loads(
        next(data for name, data in events if name == "verification")
    )
    assert verification_data["verdict"] is None
    assert verification_data["segments"] == []
    assert verification_data["warnings"] == {
        "allergy_conflicts": [],
        "blocking_interactions": [],
        "warning_interactions": [],
    }

"""Hermetic tests for the POST /chat SSE endpoint (P2.10).

Everything is faked: the token validator and the planner factory are both
injected via FastAPI dependency overrides, so no real OpenEMR or Ollama
service is ever contacted. See ``app/chat.py`` for the seams.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
from collections.abc import Iterator
from typing import Any

import pydantic
import pytest
from fastapi.testclient import TestClient

import app.chat as chat
from app.chat import (
    ChatEvent,
    ChatRequest,
    ConversationStore,
    PatientMismatchError,
    TokenValidationError,
    Turn,
    get_claim_extractor,
    get_conversation_store,
    get_planner_factory,
    get_require_tool_call_scoping,
    get_token_validator,
)
from app.config import Settings
from app.extraction import ClaimExtractor
from app.main import app
from app.openemr_auth import IntrospectionResult
from app.planner import PlannerResult, ToolCallTrace
from app.schemas.common import AllergySeverity, MedicationStatus, SourceRef, VitalType
from app.schemas.planner import ToolName
from app.schemas.tools import (
    AllergiesOutput,
    AllergyItem,
    MedicationItem,
    MedicationsOutput,
    VitalReadingItem,
    VitalsOutput,
)
from app.schemas.verification import Claim, VerifiedAnswer


class FakePlanner:
    """Scripted planner double: records the question it was asked and
    returns a fixed trace + answer (+ optional verifier-only raw_results)."""

    def __init__(
        self,
        trace: list[ToolCallTrace],
        answer: str,
        raw_results: list[dict | None] | None = None,
    ) -> None:
        self._trace = trace
        self._answer = answer
        self._raw_results = raw_results or []
        self.questions: list[str] = []

    def run(self, question: str, guideline_excerpts: object = None) -> PlannerResult:
        self.questions.append(question)
        return PlannerResult(answer=self._answer, trace=self._trace, raw_results=self._raw_results)


class FakeExtractor:
    """A ``ClaimExtractor`` double returning canned claims, no model call."""

    def __init__(self, claims: list[Claim] | None = None) -> None:
        self._claims = claims or []

    def extract_claims(self, *, answer, tools, raw_results, **_: Any) -> list[Claim]:
        # ``**_: Any`` tolerates additive keyword-only capabilities
        # (``retrieved_chunks``, ``patient_facts``, issue #158's
        # ``engaged``, ...) without this double needing to track each one --
        # it ignores every catalog/narrowing input and returns the same
        # canned claims regardless, same as it always has.
        return list(self._claims)


def _override_extractor(extractor: FakeExtractor) -> None:
    app.dependency_overrides[get_claim_extractor] = lambda: extractor


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _reset_process_wide_singletons() -> Iterator[None]:
    """Gate 3 (Opus) MINOR finding: ``chat._token_introspector`` and
    ``chat._dev_token_bridge`` are lazily-built, process-wide singletons
    (``get_token_introspector`` / ``get_dev_token_bridge``) baked from
    whatever ``Settings`` were live the FIRST time either is called. No
    fixture reset them, so a test that exercises the real (non-overridden)
    introspection or dev-bridge path -- e.g. the flag precedence test below,
    or the fail-closed-default endpoint test -- could seed a singleton bound
    to one test's env/creds that then silently leaks into a LATER test under
    ``pytest-randomly`` reordering. Mirrors ``test_launch_binding.py``'s
    ``_reset_binder_singleton`` for ``chat._launch_patient_binder``.
    """
    chat._token_introspector = None
    chat._dev_token_bridge = None
    yield
    chat._token_introspector = None
    chat._dev_token_bridge = None


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


def test_conversation_frame_carries_the_response_correlation_id():
    # P4.4: the feedback-button UI's only way to learn a response's
    # correlation id is this frame -- it must match the SAME id the P4.1
    # middleware stamped on the response's X-Correlation-ID header, so a
    # POST /feedback call posted with it lands on the right spans.
    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    data = next(data for name, data in _iter_sse_events(response.text) if name == "conversation")
    correlation_id = json.loads(data)["correlation_id"]
    assert correlation_id  # non-empty
    assert correlation_id == response.headers["X-Correlation-ID"]


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


def test_resume_with_evicted_conversation_id_is_clean_404_not_500():
    """Issue #167: eviction must fail safe. A caller that still holds a
    ``conversation_id`` for a conversation the store has since evicted (LRU,
    over-cap) must see the SAME clean 404 an unknown/bogus id already gets --
    ``store.get`` returning ``None`` is indistinguishable from "never
    existed" by design, and ``chat_endpoint`` already maps that to a 404
    (see the mismatched-patient-id 409 test above for the neighbouring
    branch), never a 500.
    """
    fake_planner = FakePlanner(trace=[], answer="first answer")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    # Cap of 1: creating a second conversation evicts the first (LRU, least-
    # recently-used since nothing touched it again after creation).
    store = ConversationStore(max_conversations=1)
    app.dependency_overrides[get_conversation_store] = lambda: store

    first = client.post(
        "/chat",
        json={"message": "first question", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    evicted_conversation_id = _conversation_id(first.text)

    # A second, unrelated conversation pushes the store over its cap of 1,
    # evicting the first.
    client.post(
        "/chat",
        json={"message": "unrelated question", "patient_id": 2},
        headers={"Authorization": "Bearer good-token"},
    )
    assert store.get(evicted_conversation_id) is None  # confirms eviction actually happened

    resume = client.post(
        "/chat",
        json={
            "message": "second question",
            "patient_id": 1,
            "conversation_id": evicted_conversation_id,
        },
        headers={"Authorization": "Bearer good-token"},
    )

    assert resume.status_code == 404


class _EvictingPlanner(FakePlanner):
    """FakePlanner double that evicts the CURRENT conversation from the
    store mid-``run()`` by creating other conversations, reproducing Gate
    3's exact timeline without needing real concurrent threads: chat_endpoint
    creates THIS request's conversation on the event loop, then does its
    slow planner/verification work (here: just this ``run()`` call) before
    ``append_turn`` is ever reached -- during that window, cheap concurrent
    ``/chat`` requests' own ``create()`` calls are free to push this
    conversation out as the LRU tail.
    """

    def __init__(self, store: ConversationStore, trace: list[Any], answer: str) -> None:
        super().__init__(trace=trace, answer=answer)
        self._store = store

    def run(self, question: str, guideline_excerpts: object = None) -> PlannerResult:
        for _ in range(5):
            self._store.create(patient_id=999)
        return super().run(question, guideline_excerpts)


def test_append_turn_survives_mid_request_eviction():
    """Gate 3 BLOCKER reproduction (#167): the pre-fix ``append_turn(
    conversation_id, turn)`` looked the conversation up by id and either
    raised ``KeyError`` (if evicted -- an unguarded ``self._conversations
    [conversation_id]``) or, in an even earlier version, silently dropped
    the turn. Either way, the P2.17 chart-access audit record for a chart
    access that DID happen was lost, and the stream never reached its
    ``done`` frame. This reproduces that exact timeline (see
    ``_EvictingPlanner``) and asserts the fix: the stream still completes
    normally and the turn is retained (via re-registration), not crashed or
    silently dropped.
    """
    store = ConversationStore(max_conversations=1, max_turns_per_conversation=10)
    app.dependency_overrides[get_conversation_store] = lambda: store

    fake_planner = _EvictingPlanner(store, trace=[], answer="first answer")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "first question", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 200
    events = _iter_sse_events(response.text)
    assert events[-1][0] == "done"  # stream reached the terminal frame, not a KeyError crash

    conversation_id = _conversation_id(response.text)
    # Evicted mid-run (by the other create() calls in _EvictingPlanner.run),
    # then re-registered by append_turn: retrievable again, turn retained.
    conversation = store.get(conversation_id)
    assert conversation is not None
    assert len(conversation.history) == 1
    assert conversation.history[0].question == "first question"


# --------------------------------------------------------------------------
# #224 name-binding: Conversation gains the bound patient's own display name,
# resolved once at conversation-creation time via the planner's OPTIONAL
# ``resolve_patient_name`` capability (getattr-duck-typed, same pattern as
# ``run_streaming``). ``FakePlanner`` above implements neither, so every
# EXISTING test in this file (none of which set up a resolver) keeps getting
# ``patient_name=None`` -- byte-identical pre-#224 behavior.
# --------------------------------------------------------------------------


class FakePlannerWithName(FakePlanner):
    """A ``FakePlanner`` that also offers the OPTIONAL name-resolution
    capability -- mirrors how the real ``Planner.resolve_patient_name``
    duck-types alongside ``run``/``run_streaming``."""

    def __init__(self, trace, answer, patient_name: str, raw_results=None) -> None:
        super().__init__(trace, answer, raw_results)
        self._patient_name = patient_name
        self.resolve_calls = 0

    def resolve_patient_name(self) -> str:
        self.resolve_calls += 1
        return self._patient_name


def test_new_conversation_resolves_and_stores_the_bound_patient_name():
    fake_planner = FakePlannerWithName(trace=[], answer="ok", patient_name="Wanda Moore")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    conversation_id = _conversation_id(response.text)

    conversation = store.get(conversation_id)
    assert conversation is not None
    assert conversation.patient_name == "Wanda Moore"
    assert fake_planner.resolve_calls == 1


def test_resumed_conversation_does_not_re_resolve_the_patient_name():
    fake_planner = FakePlannerWithName(trace=[], answer="ok", patient_name="Wanda Moore")
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
    assert fake_planner.resolve_calls == 1

    client.post(
        "/chat",
        json={"message": "second question", "patient_id": 1, "conversation_id": conversation_id},
        headers={"Authorization": "Bearer good-token"},
    )

    # Resolved once at creation time, never again on resume.
    assert fake_planner.resolve_calls == 1


def test_new_conversation_leaves_patient_name_none_when_planner_has_no_resolver():
    # FakePlanner (no resolve_patient_name) -- the pre-#224 default double
    # used throughout this file -- must not break conversation creation.
    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    conversation_id = _conversation_id(response.text)

    conversation = store.get(conversation_id)
    assert conversation is not None
    assert conversation.patient_name is None


def test_named_cross_patient_reference_is_refused_before_any_tool_dispatch_when_name_is_bound():
    # "patient <Name>" (signal 1) naming a DIFFERENT patient than the bound
    # "Wanda Moore" -- refused pre-dispatch, no tool ever run.
    trace = [ToolCallTrace(tool=ToolName.GET_ALLERGIES, args={}, result={"summary": "q"}, error=None)]
    fake_planner = FakePlannerWithName(
        trace=trace,
        answer="Bob Smith has a drug allergy to ZZ-TEST-MARKER.",
        patient_name="Wanda Moore",
    )
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "Does patient Bob Smith have any drug allergies?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    assert response.status_code == 200

    events = _iter_sse_events(response.text)
    tool_calls = [data for name, data in events if name == "tool_call"]
    answer_data = next(data for name, data in events if name == "answer")

    # No tool ever dispatched -- the fake planner's scripted run() was never
    # even called (the pre-dispatch guard short-circuits before it).
    assert tool_calls == []
    assert fake_planner.questions == []
    assert "ZZ-TEST-MARKER" not in answer_data
    assert "chart is currently open" in answer_data


# --------------------------------------------------------------------------
# #237 roster-based cross-patient detection: the "switch (over) to <Name>"
# signal (app.extraction.detect_foreign_patient_reference's signal 3), fed by
# the planner's OPTIONAL ``resolve_patient_roster`` capability -- resolved
# LAZILY (only when a "switch to <Name>" construction actually matched, never
# at conversation-creation time like ``resolve_patient_name``) and cached on
# the ``Conversation`` so a second matching turn in the SAME conversation
# does not pay the round trip again.
# --------------------------------------------------------------------------


class FakePlannerWithRoster(FakePlanner):
    """A ``FakePlanner`` that also offers the OPTIONAL roster-resolution
    capability -- mirrors how the real ``Planner.resolve_patient_roster``
    duck-types alongside ``run``/``run_streaming``/``resolve_patient_name``."""

    def __init__(self, trace, answer, roster: list[str], raw_results=None) -> None:
        super().__init__(trace, answer, raw_results)
        self._roster = roster
        self.roster_resolve_calls = 0

    def resolve_patient_roster(self) -> list[str]:
        self.roster_resolve_calls += 1
        return self._roster


def test_switch_to_name_matching_roster_is_refused_before_any_tool_dispatch():
    trace = [ToolCallTrace(tool=ToolName.GET_ALLERGIES, args={}, result={"summary": "q"}, error=None)]
    fake_planner = FakePlannerWithRoster(
        trace=trace,
        answer="Bob Smith has a drug allergy to ZZ-TEST-MARKER.",
        roster=["Bob Smith"],
    )
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "Switch over to Bob Smith and tell me his drug allergies.", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    assert response.status_code == 200

    events = _iter_sse_events(response.text)
    tool_calls = [data for name, data in events if name == "tool_call"]
    answer_data = next(data for name, data in events if name == "answer")

    assert tool_calls == []
    assert fake_planner.questions == []
    assert "ZZ-TEST-MARKER" not in answer_data
    assert "chart is currently open" in answer_data
    assert fake_planner.roster_resolve_calls == 1


def test_roster_is_not_resolved_when_question_has_no_switch_to_construction():
    fake_planner = FakePlannerWithRoster(trace=[], answer="ok", roster=["Bob Smith"])
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    client.post(
        "/chat",
        json={"message": "What meds is she on?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    assert fake_planner.roster_resolve_calls == 0


def test_switch_to_a_drug_brand_not_on_the_roster_dispatches_normally():
    # "Advair Diskus" matches the SAME 2-3 word shape as "Bob Smith" -- the
    # roster (not present on it) is what proves this is an ordinary
    # same-patient medication switch, not a cross-patient retarget.
    trace = [ToolCallTrace(tool=ToolName.GET_ALLERGIES, args={}, result={"summary": "q"}, error=None)]
    fake_planner = FakePlannerWithRoster(trace=trace, answer="ok", roster=["Bob Smith"])
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "Switch to Advair Diskus and check her allergies.", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 200
    assert fake_planner.questions == ["Switch to Advair Diskus and check her allergies."]
    assert fake_planner.roster_resolve_calls == 1


def test_roster_is_resolved_once_and_cached_across_turns():
    fake_planner = FakePlannerWithRoster(trace=[], answer="ok", roster=["Bob Smith"])
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    first = client.post(
        "/chat",
        json={"message": "Switch to Advair Diskus and check her allergies.", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    conversation_id = _conversation_id(first.text)
    assert fake_planner.roster_resolve_calls == 1

    client.post(
        "/chat",
        json={
            "message": "Switch to Depo Provera and tell me her allergies.",
            "patient_id": 1,
            "conversation_id": conversation_id,
        },
        headers={"Authorization": "Bearer good-token"},
    )

    # Cached on the conversation -- resolved once, reused on the second turn.
    assert fake_planner.roster_resolve_calls == 1


def test_roster_fetch_amplification_is_bounded_across_n_distinct_conversations():
    """RED test for #174 (VULN: Conversation.patient_roster is a per-
    conversation cache of every OTHER patient's name -- the dominant memory
    term, ~2000x amplification against OpenEMR's patient API, and a privacy
    problem independent of memory: see the issue for the full analysis).

    N>=3 ``/chat`` POSTs, each with NO ``conversation_id`` (so each gets its
    OWN, brand-new ``Conversation`` -- proven below by asserting N distinct
    conversation_ids), all containing a "switch to <Name>" construction that
    matches the roster. A correct fix makes the total number of roster
    fetches converge to ~1 regardless of N, because the roster is
    byte-identical across every conversation and belongs in ONE shared
    cache, not N per-conversation copies.

    Counter seam: ``fake_planner.roster_resolve_calls`` -- an attribute on
    the ONE ``FakePlanner`` instance ``_override_planner_factory`` wires up
    for every request via ``get_planner_factory`` (see that helper: the
    override closure is ``lambda patient_id: fake_planner``, ignoring
    ``patient_id`` and always returning the SAME object). This is the
    honest seam because it mirrors the real amplification being fixed: in
    production a fresh ``Planner`` IS built per request
    (``_default_planner_factory``), so ``Planner.resolve_patient_roster()``
    itself is never a persistent counter there either -- a correct fix
    needs a cache SITTING IN FRONT of that call, not a change to the call
    itself, and this test's counter (on the object actually being called)
    is exactly what such a cache would need to shield.

    Today (pre-fix): each of the N brand-new ``Conversation`` objects has
    its own ``patient_roster is None`` starting state, so ALL N calls miss
    and ``roster_resolve_calls`` ends at exactly N -- this assertion fails
    at N==3.
    """
    fake_planner = FakePlannerWithRoster(trace=[], answer="ok", roster=["Bob Smith"])
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    n = 3
    conversation_ids: set[str] = set()
    for _ in range(n):
        response = client.post(
            "/chat",
            json={"message": "Switch to Bob Smith and check her allergies.", "patient_id": 1},
            headers={"Authorization": "Bearer good-token"},
        )
        assert response.status_code == 200
        conversation_ids.add(_conversation_id(response.text))
        # Presence pair for the absence-style assertion below: the roster
        # signal must have genuinely FIRED every single time (a refusal),
        # not merely gone unexercised -- otherwise a low fetch count could
        # just mean the guard never ran at all.
        answer_data = next(data for name, data in _iter_sse_events(response.text) if name == "answer")
        assert "chart is currently open" in answer_data

    # Presence: N genuinely distinct Conversation objects were created, not
    # one conversation_id accidentally reused across the loop.
    assert len(conversation_ids) == n

    # The amplification-reduction assertion: fewer fetches than
    # conversations (today: fails, roster_resolve_calls == n).
    assert 1 <= fake_planner.roster_resolve_calls < n


def test_switch_to_message_does_not_crash_when_planner_has_no_roster_resolver():
    # FakePlanner (no resolve_patient_roster) -- the pre-#237 default double
    # used throughout this file -- must not break, and the roster signal is
    # simply skipped (planner runs normally, no refusal).
    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    _override_planner_factory(fake_planner)

    response = client.post(
        "/chat",
        json={"message": "Switch over to Bob Smith and tell me his drug allergies.", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 200
    assert fake_planner.questions == ["Switch over to Bob Smith and tell me his drug allergies."]


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


# --------------------------------------------------------------------------
# Red-team findings #167/#168 (issue #171) -- regression coverage
# --------------------------------------------------------------------------


def test_fail_closed_default_rejects_garbage_bearer_token(monkeypatch):
    # #168 (VULN-0001, evals/recordings/identity-authz-garbage-bearer-token/):
    # previously get_token_validator (app/chat.py) handed back a permissive
    # stub (accepts ANY non-empty token -- its own docstring said so) whenever
    # copilot_per_user_token_enabled is False, the shipped default. Fixed
    # (fail-closed): with both copilot_per_user_token_enabled AND
    # copilot_dev_accept_any_bearer_token unset/False (the shipped default for
    # each), a garbage bearer token is REJECTED.
    #
    # Exercises the PUBLIC seam (get_token_validator(), the FastAPI dependency
    # itself) rather than importing a private validator function directly --
    # this both survives further internal refactors (which would turn a
    # direct import into an ImportError failing every test in this module at
    # collection) AND actually proves the shipped default (both flags
    # unset/False) really does hand back a validator that rejects garbage,
    # not just that some validator function in isolation does.
    monkeypatch.delenv("COPILOT_PER_USER_TOKEN_ENABLED", raising=False)
    monkeypatch.delenv("COPILOT_DEV_ACCEPT_ANY_BEARER_TOKEN", raising=False)

    validator = get_token_validator()

    with pytest.raises(TokenValidationError):
        validator("this-is-not-a-real-token-just-garbage-xyz123")


def test_dev_accept_any_bearer_token_flag_restores_permissive_validator(monkeypatch):
    # Gate coverage gap: the dev opt-in (copilot_dev_accept_any_bearer_token)
    # had no test proving it actually works -- if it silently broke, the
    # local dev stack (docker-compose.copilot.yml sets it) and the #160 live
    # harness would break with it, invisibly. With per-user tokens OFF and
    # the dev flag explicitly ON, get_token_validator() must hand back a
    # validator that ACCEPTS a garbage (but non-empty) token.
    monkeypatch.delenv("COPILOT_PER_USER_TOKEN_ENABLED", raising=False)
    monkeypatch.setenv("COPILOT_DEV_ACCEPT_ANY_BEARER_TOKEN", "true")

    validator = get_token_validator()

    validator("this-is-not-a-real-token-just-garbage-xyz123")  # must not raise


def test_dev_accept_any_bearer_token_flag_logs_a_warning(monkeypatch, caplog):
    # "Never silently active" (app.chat._dev_permissive_token_validator's
    # docstring) is only true if the warning actually fires -- enforce it
    # rather than merely asserting the intent in a comment.
    monkeypatch.delenv("COPILOT_PER_USER_TOKEN_ENABLED", raising=False)
    monkeypatch.setenv("COPILOT_DEV_ACCEPT_ANY_BEARER_TOKEN", "true")

    validator = get_token_validator()

    with caplog.at_level(logging.WARNING):
        validator("this-is-not-a-real-token-just-garbage-xyz123")

    assert any(
        "dev-only permissive bearer-token stub" in record.message for record in caplog.records
    )


class _InactiveIntrospector:
    """Fake ``Introspector`` (see ``test_introspection.py``'s ``_FakeIntrospector``)
    that always reports a token inactive -- stands in for the real RFC 7662
    round trip so this test proves ONLY the precedence decision, not the
    introspection HTTP path (that's ``test_introspection.py``'s job)."""

    def introspect(self, token: str) -> IntrospectionResult:
        return IntrospectionResult(active=False, exp=None)


def test_per_user_token_enabled_wins_over_dev_accept_any_bearer_token(monkeypatch):
    # Precedence, pinned: with BOTH copilot_per_user_token_enabled AND
    # copilot_dev_accept_any_bearer_token true, the real introspection
    # validator must win -- the permissive stub must NOT be returned. This is
    # correct today by inspection (get_token_validator checks
    # copilot_per_user_token_enabled first), but was unpinned by any test, so
    # a future refactor could silently invert the check order and reopen
    # VULN-0001 behind a flag combination nobody thinks to check by hand.
    # Proven behaviourally (garbage token REJECTED, not just "not identical
    # to the dev stub function") so the assertion survives the introspection
    # validator's own internal implementation changing shape.
    #
    # Gate 3 (Opus) MINOR finding: an earlier version of this test let
    # ``get_token_validator`` build the REAL introspector
    # (``get_token_introspector`` -> ``TokenIntrospector.from_settings``),
    # which -- whenever the confidential-client creds file happens to exist
    # (e.g. inside the provisioned agent container) -- made a genuine
    # outbound POST of a garbage token to OpenEMR's introspection endpoint,
    # AND seeded the process-wide ``chat._token_introspector`` singleton
    # (reset by the ``_reset_process_wide_singletons`` fixture above, but the
    # network call itself is still undesirable in a "hermetic" suite). This
    # injects a fake introspector instead, exactly as ``test_introspection
    # .py`` does for the introspection unit tests themselves -- the
    # precedence decision under test never needs a real network round trip.
    monkeypatch.setattr(chat, "get_token_introspector", lambda: _InactiveIntrospector())
    monkeypatch.setenv("COPILOT_PER_USER_TOKEN_ENABLED", "true")
    monkeypatch.setenv("COPILOT_DEV_ACCEPT_ANY_BEARER_TOKEN", "true")

    validator = get_token_validator()

    with pytest.raises(TokenValidationError):
        validator("this-is-not-a-real-token-just-garbage-xyz123")


def test_fail_closed_default_returns_401_not_500_at_the_endpoint(monkeypatch):
    # Gate 3 (Opus) CRITICAL finding: every prior #168 test (including
    # test_fail_closed_default_rejects_garbage_bearer_token above) exercises
    # get_token_validator() directly, and every EXISTING endpoint-level 401
    # test (test_missing_token_returns_401_and_never_invokes_planner /
    # test_rejected_token_returns_401_and_never_invokes_planner) overrides
    # get_planner_factory -- so none of them ever resolve the REAL
    # get_planner_factory dependency, which is where the bug actually lived.
    #
    # In the shipped default posture (both auth flags off) with an
    # UNPROVISIONED dev-token bridge -- any deployment that isn't the local
    # dev stack -- get_planner_factory (app/chat.py) used to call
    # dev_token_bridge.get_token() unconditionally on the flag-off branch,
    # which FastAPI resolves BEFORE the endpoint body's token check even
    # runs. An unprovisioned bridge raises DevTokenError there, uncaught,
    # surfacing as a 500 -- not the 401 _fail_closed_token_validator's own
    # docstring promises. Beyond the wrong status code that's a
    # fingerprinting oracle (500 on /chat vs 401 everywhere else tells an
    # unauthenticated caller whether the dev bridge is provisioned) and,
    # because the bridge caches only on success, unauthenticated
    # amplification: every such request drove a fresh outbound OAuth
    # password-grant attempt against OpenEMR.
    #
    # Deliberately does NOT override get_token_validator or
    # get_planner_factory -- the whole point is to resolve the REAL FastAPI
    # dependency chain, exactly as a live deployment would. Points the dev
    # bridge's creds file at a path that cannot exist, so DevTokenBridge
    # fails exactly like an unprovisioned (non-dev-stack) deployment would,
    # deterministically and with no real network call
    # (DevTokenBridge._load_creds raises DevTokenError on a plain file-read
    # failure, before any HTTP request is attempted).
    monkeypatch.delenv("COPILOT_PER_USER_TOKEN_ENABLED", raising=False)
    monkeypatch.delenv("COPILOT_DEV_ACCEPT_ANY_BEARER_TOKEN", raising=False)
    monkeypatch.setenv(
        "COPILOT_DEV_CLIENT_CREDS_PATH",
        "/nonexistent/path/this-file-does-not-exist-issue-168.json",
    )

    garbage = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer this-is-not-a-real-token-just-garbage-xyz123"},
    )
    no_header = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
    )

    for response in (garbage, no_header):
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid or missing token"}


def test_chat_request_rejects_overlong_message():
    with pytest.raises(pydantic.ValidationError):
        ChatRequest(message="x" * 1_000_000, patient_id=1)


def test_chat_request_message_length_boundary_is_exact():
    """Gate 3 MINOR finding: the original overlong-message test used
    1,000,000 chars, which would still pass even if MAX_CHAT_MESSAGE_LENGTH
    drifted down to 999,999 -- it never pinned the actual boundary. This
    proves the boundary is exactly 4000: 4000 chars is accepted, 4001 is
    rejected (hand-verified: no off-by-one either direction).
    """
    ChatRequest(message="x" * 4000, patient_id=1)  # exactly at the limit: accepted

    with pytest.raises(pydantic.ValidationError):
        ChatRequest(message="x" * 4001, patient_id=1)  # one over: rejected


def test_chat_request_rejects_empty_message():
    with pytest.raises(pydantic.ValidationError):
        ChatRequest(message="", patient_id=1)


def test_chat_request_rejects_overlong_conversation_id():
    ChatRequest(message="hi", patient_id=1, conversation_id="x" * 64)  # exactly at the limit: accepted

    with pytest.raises(pydantic.ValidationError):
        ChatRequest(message="hi", patient_id=1, conversation_id="x" * 65)


def test_conversation_store_bounds_retained_conversations():
    # BEHAVIOUR, not vocabulary (issue-#86 failure class avoided): a class-level
    # dir()/vocabulary check would never see an instance-level cap (e.g.
    # self._max_conversations set in __init__, or an OrderedDict-backed LRU with
    # no new public method at all) -- a real P4.2 fix could close the vulnerability
    # while such a check kept xfailing forever. This instead creates WAY more
    # conversations than any plausible cap and asserts the store's own behaviour:
    # either fewer than all of them still resolve via get() (bounded retention),
    # or the earliest one specifically has been evicted (LRU semantics). Any
    # eviction/cap/TTL implementation satisfies one of the two; today's
    # unbounded dict satisfies neither.
    store = ConversationStore()
    conversation_ids = [store.create(patient_id=1).conversation_id for _ in range(10_000)]

    retained_count = sum(1 for cid in conversation_ids if store.get(cid) is not None)
    earliest_still_resolves = store.get(conversation_ids[0]) is not None

    assert retained_count < len(conversation_ids) or not earliest_still_resolves


def test_conversation_store_bounds_turns_per_conversation():
    """Gate 2 finding on #167: the conversation-COUNT cap
    (``max_conversations``) does not stop a SINGLE conversation from growing
    forever -- an attacker reusing one ``conversation_id`` stays permanently
    most-recently-used and so is never an eviction candidate under that cap
    alone. This asserts the SEPARATE per-conversation turn cap: appending
    far more turns than any plausible cap to ONE conversation must not
    retain all of them.

    Confirmed this fails against the pre-fix ``append_turn`` (unconditional
    ``.history.append(turn)``, no trim) and passes against the fix (drops
    the oldest turn once ``max_turns_per_conversation`` is exceeded): ran
    this test against a stash of the code before the turn-cap was added --
    it failed with ``AssertionError`` (history length was 5000, equal to
    the turn count, not bounded); reran unstashed and it passes.

    ``append_turn`` takes the live ``Conversation`` object, not a
    ``conversation_id`` lookup (Gate 3 BLOCKER fix -- see
    ``ConversationStore.append_turn``'s docstring for why); this test
    passes ``conversation`` itself accordingly.
    """
    store = ConversationStore(max_turns_per_conversation=10)
    conversation = store.create(patient_id=1)

    turn_count = 5_000
    for i in range(turn_count):
        store.append_turn(
            conversation,
            Turn(
                correlation_id=f"corr-{i}",
                user="clinician",
                patient_id=1,
                question=f"question {i}",
                answer=f"answer {i}",
            ),
        )

    retained = store.get(conversation.conversation_id)
    assert retained is not None
    assert len(retained.history) < turn_count
    assert len(retained.history) == 10
    # Oldest turns are dropped, newest retained (LRU-of-turns, not just any
    # 10 -- a real fix must keep the RECENT clinical context, not arbitrary
    # entries).
    assert retained.history[-1].question == f"question {turn_count - 1}"
    assert retained.history[0].question == f"question {turn_count - 10}"


def test_get_conversation_store_builds_singleton_from_settings():
    """Gate 3 MINOR finding: every endpoint test in this module overrides
    ``get_conversation_store`` directly (``app.dependency_overrides``), so
    NOTHING in the suite ever actually executes the real function's
    ``_default_store is None`` branch -- a wrong Settings field name inside
    it would keep the whole suite green. This calls the real dependency
    function directly and asserts the two caps land on the store it builds.

    Also documents (see the function's own docstring for the fuller
    explanation) that the caps are frozen at first use: ``get_settings()``
    would return fresh values on a later call, but ``get_conversation_store``
    only reads them once and caches the result -- a second call with
    DIFFERENT settings must return the SAME object with the ORIGINAL caps,
    not rebuild.
    """
    import app.chat as chat_module

    original_default_store = chat_module._default_store
    chat_module._default_store = None
    try:
        settings_a = Settings(copilot_max_stored_conversations=7, copilot_max_turns_per_conversation=3)
        store = chat_module.get_conversation_store(settings_a)
        assert store._max_conversations == 7
        assert store._max_turns_per_conversation == 3

        settings_b = Settings(copilot_max_stored_conversations=999, copilot_max_turns_per_conversation=999)
        store_again = chat_module.get_conversation_store(settings_b)
        assert store_again is store  # singleton: not rebuilt
        assert store_again._max_conversations == 7  # frozen at first use, NOT 999
        assert store_again._max_turns_per_conversation == 3
    finally:
        chat_module._default_store = original_default_store


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


def test_stream_emits_verification_frame_after_answer_before_done():
    # P3.8: every response carries a verification frame (verdict badge /
    # citation chips / warning banner contract). With no extractable claims
    # (empty trace/raw), the pipeline fails closed to a `blocked` verdict with
    # no segments/warnings; the frame's position and contract shape are pinned.
    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    _override_planner_factory(fake_planner)
    _override_extractor(FakeExtractor())

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
    # No claims survived (there were none) -> fail-closed blocked, no evidence.
    assert verification_data["verdict"] == "blocked"
    assert verification_data["segments"] == []
    assert verification_data["warnings"] == {
        "allergy_conflicts": [],
        "blocking_interactions": [],
        "warning_interactions": [],
    }


def test_stream_emits_populated_verification_frame_with_real_claim():
    # The flagship integration: a grounded medication claim flows through
    # extraction -> checker -> render -> verdict and lands in the SSE frame as
    # a `verified` badge with a citation chip on the real record value.
    meds_raw = MedicationsOutput(
        items=[
            MedicationItem(
                name="Lisinopril", dose="10 mg", route="oral", status=MedicationStatus.ACTIVE
            )
        ]
    ).model_dump(mode="json")
    trace = [ToolCallTrace(tool=ToolName.GET_MEDICATIONS, args={}, result={"summary": "q"}, error=None)]
    fake_planner = FakePlanner(
        trace=trace, answer="She is on Lisinopril.", raw_results=[meds_raw]
    )
    claim = Claim(
        text="She is on Lisinopril.",
        source_refs=[
            SourceRef(
                tool_call_id="call_0", record_id="0", field="name", asserted_value="Lisinopril"
            )
        ],
    )
    _override_ok_validator()
    _override_planner_factory(fake_planner)
    _override_extractor(FakeExtractor([claim]))

    response = client.post(
        "/chat",
        json={"message": "What meds is she on?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    events = _iter_sse_events(response.text)
    verification_data = json.loads(
        next(data for name, data in events if name == "verification")
    )

    assert verification_data["verdict"] == "verified"
    assert verification_data["segments"] == [
        {
            "type": "claim",
            "text": "She is on Lisinopril.",
            "citations": [
                {
                    "tool_call_id": "call_0",
                    "record_id": "0",
                    "field": "name",
                    "value": "Lisinopril",
                }
            ],
            "document_citations": [],
        }
    ]
    assert verification_data["warnings"]["allergy_conflicts"] == []


class _ScriptedOllamaLikeIgnoresCatalog:
    """A REAL ``ClaimExtractor``'s underlying LLM client, scripted: records
    every ``messages`` list it is called with, and always returns ONE claim
    citing call_1 (the allergy record), REGARDLESS of what catalog/messages
    it actually received.

    Used by ``test_chat_tool_call_scoping_flag_on_exercises_prevention_and_
    enforcement_end_to_end`` below to prove BOTH halves of the #158 gate in
    one real ``POST /chat`` turn: PREVENTION is checked by inspecting the
    messages this double actually recorded (call_1/"Penicillin"/"substance"
    must be ABSENT -- the narrowed catalog never reached here), and
    ENFORCEMENT is checked by the SSE verification frame (the returned
    claim, citing an unengaged call, must still fail to verify -- proving
    the checker rejects it independent of whether the extractor itself
    "saw" call_1)."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def extract(self, prompt_or_messages: Any, schema: type, *, options: Any = None) -> Any:
        self.calls.append(list(prompt_or_messages))
        return VerifiedAnswer(
            claims=[
                Claim(
                    text="She is allergic to Penicillin.",
                    source_refs=[
                        SourceRef(
                            tool_call_id="call_1",
                            record_id="0",
                            field="substance",
                            asserted_value="Penicillin",
                        )
                    ],
                )
            ]
        )


def test_chat_tool_call_scoping_flag_on_exercises_prevention_and_enforcement_end_to_end():
    """MINOR 1 (gate-3 review): the #158 flag-ON configuration was otherwise
    untested at the ``/chat`` level -- every other test in this file uses a
    ``FakeExtractor`` that ignores its inputs entirely, which exercises
    ENFORCEMENT but can never prove PREVENTION (the catalog/messages the
    extractor actually receives). This test drives a REAL ``ClaimExtractor``
    (not a scripted double) through the full endpoint with
    ``get_require_tool_call_scoping`` overridden ON, so both halves of the
    gate run for real:

      - call_0 (``GET_VITALS``, weight 220) is ENGAGED -- the answer says
        "Her weight is 220 lb."
      - call_1 (``GET_ALLERGIES``, Penicillin) is UNENGAGED -- the answer
        never mentions it at all.

    PREVENTION: the scripted LLM client's recorded messages must never
    mention call_1/"Penicillin"/"substance" -- the extractor's own catalog
    was narrowed before it ever saw them.

    ENFORCEMENT: the scripted client nonetheless RETURNS a claim citing
    call_1 (as if it had hallucinated or somehow still cited it) -- the SSE
    verification frame must show it did NOT verify, proving the checker
    rejects an unengaged-call citation independent of prevention."""
    vitals_raw = VitalsOutput(
        items=[
            VitalReadingItem(
                vital_type=VitalType.WEIGHT,
                value=220.0,
                unit="lb_av",
                date=datetime.datetime(2026, 1, 1, 9, 0),
            )
        ]
    ).model_dump(mode="json")
    allergies_raw = AllergiesOutput(
        items=[AllergyItem(substance="Penicillin", severity=AllergySeverity.SEVERE)]
    ).model_dump(mode="json")
    trace = [
        ToolCallTrace(tool=ToolName.GET_VITALS, args={}, result={"summary": "q"}, error=None),
        ToolCallTrace(tool=ToolName.GET_ALLERGIES, args={}, result={"summary": "q"}, error=None),
    ]
    fake_planner = FakePlanner(trace=trace, answer="Her weight is 220 lb.", raw_results=[vitals_raw, allergies_raw])

    ollama_double = _ScriptedOllamaLikeIgnoresCatalog()
    _override_ok_validator()
    _override_planner_factory(fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: ClaimExtractor(ollama_client=ollama_double)
    app.dependency_overrides[get_require_tool_call_scoping] = lambda: True

    response = client.post(
        "/chat",
        json={"message": "How much does she weigh?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    # PREVENTION: the narrowed catalog/messages never mention the unengaged
    # call's id, field, or value.
    assert ollama_double.calls, "expected the real ClaimExtractor to invoke the scripted LLM client"
    sent_messages = ollama_double.calls[0]
    sent_text = json.dumps(sent_messages)
    assert "call_1" not in sent_text
    assert "Penicillin" not in sent_text
    assert "substance" not in sent_text
    assert "call_0" in sent_text  # the engaged call is still present

    # ENFORCEMENT: the claim citing the unengaged call still fails to verify.
    events = _iter_sse_events(response.text)
    verification_data = json.loads(next(data for name, data in events if name == "verification"))
    assert verification_data["verdict"] != "verified"

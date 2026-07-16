"""#124 Phase 5: SMART launch-context patient binding for POST /chat.

The forwarded per-user token can carry a SMART *launch patient*. OpenEMR's
RFC 7662 introspection response echoes it under the ``patient`` key as a FHIR
UUID string (verbatim from the token's stored ``context`` column -- no
pid/uuid conversion server-side). This suite covers:

  * ``introspect_token`` parsing the ``patient`` launch UUID (present / absent /
    non-string).
  * ``LaunchPatientBinder.verify`` -- pid->uuid resolution via the REST patient
    roster and the match/mismatch/absent/fail-safe decisions.
  * ``get_launch_binding_checker`` flag gating (no-op OFF, binder ON).
  * The endpoint refusing a mismatch with 403 BEFORE the planner runs (the
    planner spy is asserted never called), the absent-launch fall-back to the
    P2.16 conversation-pid binding, and no token/PHI in the response or logs.

All HTTP is served by ``httpx.MockTransport`` -- the suite never touches the
network. Fake token VALUES are tiny + keyword-free for the pre-push
secret-scan; UUIDs are fine.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import app.chat as chat
from app.chat import (
    ConversationStore,
    _default_launch_binding_checker,
    get_claim_extractor,
    get_conversation_store,
    get_launch_binding_checker,
    get_planner_factory,
    get_token_validator,
)
from app.launch_binding import LaunchPatientBinder, LaunchPatientMismatchError
from app.main import app
from app.openemr_auth import IntrospectionResult, introspect_token
from app.openemr_client import OpenEmrClient
from app.planner import PlannerResult

# Tiny, keyword-free fake bearer values (each quoted literal < 8 chars).
_TOK = "utok-a"
_CID = "cid-1"
_CSEC = "cs-1"

# Launch patient UUID (uuid literals are safe for the secret-scan).
_PT_UUID = "a243a1bb-178f-4092-8c67-52dfaf67fca6"
_OTHER_UUID = "b1a2c3d4-0000-4000-8000-000000000002"

# REST patient roster shape (GET /apis/default/api/patient): pid 1 -> _PT_UUID,
# pid 2 -> _OTHER_UUID. The internal pid is not a public REST filter, so the
# binder fetches the roster and selects the matching record client-side.
_ROSTER_BODY = {
    "validationErrors": [],
    "internalErrors": [],
    "data": [
        {"pid": 1, "fname": "Phil", "lname": "Belford", "uuid": _PT_UUID},
        {"pid": 2, "fname": "Susan", "lname": "Underwood", "uuid": _OTHER_UUID},
    ],
}


# --- introspect_token: launch patient parsing -----------------------------


def _introspect_with(payload: dict[str, object]) -> IntrospectionResult:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return introspect_token(
            client,
            base_url="https://openemr",
            introspect_path="/oauth2/default/introspect",
            client_id=_CID,
            client_secret=_CSEC,
            token=_TOK,
        )


def test_introspect_parses_launch_patient_uuid():
    result = _introspect_with({"active": True, "exp": 9999999999, "patient": _PT_UUID})
    assert result.active
    assert result.patient == _PT_UUID


def test_introspect_active_without_patient_has_none_patient():
    result = _introspect_with({"active": True, "exp": 9999999999})
    assert result.active
    assert result.patient is None


def test_introspect_non_string_patient_is_ignored():
    result = _introspect_with({"active": True, "patient": 123})
    assert result.patient is None


# --- LaunchPatientBinder ---------------------------------------------------


class _FakeIntrospector:
    """Returns a fixed ``IntrospectionResult`` and records the tokens seen."""

    def __init__(self, result: IntrospectionResult) -> None:
        self._result = result
        self.seen: list[str] = []

    def introspect(self, token: str) -> IntrospectionResult:
        self.seen.append(token)
        return self._result


def _roster_client(calls: list[int], *, status: int = 200) -> OpenEmrClient:
    """Build an ``OpenEmrClient`` whose patient-roster read is mocked."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if status != 200:
            return httpx.Response(status, json={})
        return httpx.Response(200, json=_ROSTER_BODY)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenEmrClient(base_url="https://openemr", client=client)


def _binder(result: IntrospectionResult, calls: list[int], *, status: int = 200) -> LaunchPatientBinder:
    return LaunchPatientBinder(
        introspector=_FakeIntrospector(result), client=_roster_client(calls, status=status)
    )


def test_binder_passes_when_launch_uuid_matches_resolved_pid():
    calls: list[int] = []
    binder = _binder(IntrospectionResult(active=True, exp=None, patient=_PT_UUID), calls)
    binder.verify(_TOK, 1)  # pid 1 -> _PT_UUID == launch uuid -> no raise
    assert len(calls) == 1  # the pid->uuid resolve read happened


def test_binder_raises_when_launch_uuid_mismatches_pid():
    calls: list[int] = []
    binder = _binder(IntrospectionResult(active=True, exp=None, patient=_PT_UUID), calls)
    # pid 2 resolves to _OTHER_UUID, which != the launch uuid _PT_UUID.
    with pytest.raises(LaunchPatientMismatchError):
        binder.verify(_TOK, 2)


def test_binder_no_launch_context_falls_back_without_resolving():
    calls: list[int] = []
    binder = _binder(IntrospectionResult(active=True, exp=None, patient=None), calls)
    binder.verify(_TOK, 1)  # no launch context -> no raise (P2.16 fall-back)
    assert calls == []  # the resolve read must be skipped entirely


def test_binder_raises_when_pid_absent_from_roster():
    calls: list[int] = []
    binder = _binder(IntrospectionResult(active=True, exp=None, patient=_PT_UUID), calls)
    with pytest.raises(LaunchPatientMismatchError):
        binder.verify(_TOK, 999)  # fail-safe: cannot confirm -> reject


def test_binder_raises_fail_safe_when_resolve_read_fails():
    calls: list[int] = []
    binder = _binder(
        IntrospectionResult(active=True, exp=None, patient=_PT_UUID), calls, status=500
    )
    with pytest.raises(LaunchPatientMismatchError):
        binder.verify(_TOK, 1)  # resolve read errors -> fail-safe reject


def test_binder_match_is_case_insensitive():
    calls: list[int] = []
    binder = _binder(
        IntrospectionResult(active=True, exp=None, patient=_PT_UUID.upper()), calls
    )
    binder.verify(_TOK, 1)  # uppercase launch uuid still matches lowercase roster uuid


def test_binder_never_logs_token_or_uuid_on_mismatch(caplog):
    calls: list[int] = []
    binder = _binder(IntrospectionResult(active=True, exp=None, patient=_PT_UUID), calls)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LaunchPatientMismatchError):
            binder.verify(_TOK, 2)
    assert _TOK not in caplog.text
    assert _PT_UUID not in caplog.text
    assert _OTHER_UUID not in caplog.text


# --- get_launch_binding_checker flag gating -------------------------------


@pytest.fixture(autouse=True)
def _reset_binder_singleton() -> Iterator[None]:
    chat._launch_patient_binder = None
    yield
    chat._launch_patient_binder = None
    app.dependency_overrides.clear()


def test_default_checker_is_a_noop():
    assert _default_launch_binding_checker(_TOK, 1) is None


def test_get_checker_flag_off_returns_the_noop(monkeypatch):
    monkeypatch.delenv("COPILOT_PER_USER_TOKEN_ENABLED", raising=False)
    assert get_launch_binding_checker() is _default_launch_binding_checker


def test_get_checker_flag_on_returns_binder_verify(monkeypatch):
    monkeypatch.setenv("COPILOT_PER_USER_TOKEN_ENABLED", "true")
    checker = get_launch_binding_checker()
    assert checker is not _default_launch_binding_checker
    assert callable(checker)


# --- endpoint integration: refuse BEFORE the planner runs ------------------


class _SpyPlanner:
    """Planner spy: records the questions it was asked (must stay empty on a
    refusal, proving the planner never ran)."""

    def __init__(self, answer: str = "ok") -> None:
        self._answer = answer
        self.questions: list[str] = []

    def run(self, question: str) -> PlannerResult:
        self.questions.append(question)
        return PlannerResult(answer=self._answer, trace=[], raw_results=[])


class _NoClaimsExtractor:
    def extract_claims(self, *, answer, tools, raw_results):
        return []


def _ok_validator(token: str) -> None:
    return None


def _override(planner: _SpyPlanner, checker) -> None:
    app.dependency_overrides[get_token_validator] = lambda: _ok_validator
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: planner)
    app.dependency_overrides[get_claim_extractor] = lambda: _NoClaimsExtractor()
    app.dependency_overrides[get_launch_binding_checker] = lambda: checker


def _conversation_id(text: str) -> str:
    for block in text.strip().split("\n\n"):
        if block.startswith("event: conversation"):
            data = block.splitlines()[1][len("data:") :].strip()
            return json.loads(data)["conversation_id"]
    raise AssertionError("no conversation frame")


client = TestClient(app)


def test_endpoint_launch_mismatch_refused_403_before_planner_run():
    planner = _SpyPlanner()

    def _reject(token: str, patient_id: int) -> None:
        raise LaunchPatientMismatchError("mismatch")

    _override(planner, _reject)

    response = client.post(
        "/chat",
        json={"message": "hi", "patient_id": 2},
        headers={"Authorization": "Bearer utok-z9"},
    )

    assert response.status_code == 403
    assert planner.questions == []  # the planner (and thus any tool call) never ran


def test_endpoint_launch_match_proceeds_to_planner():
    planner = _SpyPlanner()
    _override(planner, lambda token, patient_id: None)

    response = client.post(
        "/chat",
        json={"message": "hi", "patient_id": 1},
        headers={"Authorization": "Bearer utok-z9"},
    )

    assert response.status_code == 200
    assert planner.questions == ["hi"]


def test_endpoint_absent_launch_falls_back_to_p216_conversation_binding():
    # A no-op launch checker stands in for a token WITHOUT launch context: the
    # request is NOT hard-failed by the launch layer, and the existing P2.16
    # conversation-pid binding still refuses a mismatched resume (409).
    planner = _SpyPlanner()
    _override(planner, lambda token, patient_id: None)
    store = ConversationStore()
    app.dependency_overrides[get_conversation_store] = lambda: store

    first = client.post(
        "/chat",
        json={"message": "one", "patient_id": 1},
        headers={"Authorization": "Bearer utok-z9"},
    )
    assert first.status_code == 200
    conversation_id = _conversation_id(first.text)

    second = client.post(
        "/chat",
        json={"message": "two", "patient_id": 2, "conversation_id": conversation_id},
        headers={"Authorization": "Bearer utok-z9"},
    )
    assert second.status_code == 409  # P2.16 binding still enforced under the launch layer


def test_endpoint_refusal_leaks_no_token_or_phi_to_response_or_logs(caplog):
    planner = _SpyPlanner()

    def _reject(token: str, patient_id: int) -> None:
        raise LaunchPatientMismatchError("mismatch")

    _override(planner, _reject)

    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/chat",
            json={"message": "hi", "patient_id": 2},
            headers={"Authorization": "Bearer utok-z9"},
        )

    assert response.status_code == 403
    assert "utok-z9" not in response.text
    assert "utok-z9" not in caplog.text
    assert _PT_UUID not in response.text
    assert _PT_UUID not in caplog.text

"""#124 Phase 4: the planner factory uses the REQUEST's forwarded token.

Flag ON  -> ``get_planner_factory`` builds the planner with the request's own
            bearer, so OpenEMR maps every tool call to that user (per-user ACL).
Flag OFF -> byte-identical to today: the ``DevTokenBridge`` demo-clinician
            token drives tool calls.

The wiring is unit-tested by calling ``get_planner_factory`` directly with a
fake bridge and ``_default_planner_factory`` monkeypatched to capture the token
it is bound to -- no real OpenEMR / Ollama client is constructed.

Fake token VALUES are tiny + keyword-free for the pre-push secret-scan.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.chat as chat
from app.chat import (
    IntrospectionResult,
    build_introspection_validator,
    get_planner_factory,
    get_token_validator,
)
from app.main import app

_BRIDGE_TOK = "brg-1"
_USER_TOK = "usr-1"


class _FakeBridge:
    def __init__(self, token: str) -> None:
        self._token = token

    def get_token(self) -> str:
        return self._token


@pytest.fixture
def capture_factory_token(monkeypatch):
    """Monkeypatch ``_default_planner_factory`` to record the bound token."""
    captured: dict[str, str] = {}

    def fake_default_factory(token: str):
        captured["token"] = token
        return lambda patient_id: object()

    monkeypatch.setattr(chat, "_default_planner_factory", fake_default_factory)
    return captured


def test_flag_off_uses_dev_bridge_token(capture_factory_token, monkeypatch):
    monkeypatch.delenv("COPILOT_PER_USER_TOKEN_ENABLED", raising=False)
    get_planner_factory(
        authorization=f"Bearer {_USER_TOK}",
        dev_token_bridge=_FakeBridge(_BRIDGE_TOK),  # type: ignore[arg-type]
    )
    # Default (flag off): tool calls use the bridge's demo-clinician token,
    # ignoring the request bearer -- byte-identical to today.
    assert capture_factory_token["token"] == _BRIDGE_TOK


def test_flag_on_uses_request_forwarded_token(capture_factory_token, monkeypatch):
    monkeypatch.setenv("COPILOT_PER_USER_TOKEN_ENABLED", "true")
    get_planner_factory(
        authorization=f"Bearer {_USER_TOK}",
        dev_token_bridge=_FakeBridge(_BRIDGE_TOK),  # type: ignore[arg-type]
    )
    # Flag on: the planner is bound to the REQUEST's forwarded bearer, not the
    # dev bridge token -> OpenEMR enforces this user's ACL.
    assert capture_factory_token["token"] == _USER_TOK


def test_flag_on_missing_header_does_not_raise_in_dependency(capture_factory_token, monkeypatch):
    # get_planner_factory resolves as a FastAPI dependency (before the endpoint
    # body validates the token). A missing header must NOT raise here (that
    # would surface as a 500) -- it binds an empty token; the body's validator
    # then rejects with 401 and the planner never runs.
    monkeypatch.setenv("COPILOT_PER_USER_TOKEN_ENABLED", "true")
    get_planner_factory(
        authorization=None,
        dev_token_bridge=_FakeBridge(_BRIDGE_TOK),  # type: ignore[arg-type]
    )
    assert capture_factory_token["token"] == ""


# --- endpoint-level: introspection validator maps to 401 before planner ----


class _FakePlanner:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def run(self, question: str):  # pragma: no cover - must never be called on 401
        self.questions.append(question)
        raise AssertionError("planner ran despite an invalid token")


class _FakeIntrospector:
    def __init__(self, result: IntrospectionResult) -> None:
        self._result = result

    def introspect(self, token: str) -> IntrospectionResult:
        return self._result


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    app.dependency_overrides.clear()


def test_endpoint_inactive_token_returns_401_before_planner():
    fake_planner = _FakePlanner()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    validator = build_introspection_validator(
        _FakeIntrospector(IntrospectionResult(active=False, exp=None)),
        clock=lambda: 1000.0,
    )
    app.dependency_overrides[get_token_validator] = lambda: validator

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"message": "hi", "patient_id": 1},
        headers={"Authorization": f"Bearer {_USER_TOK}"},
    )

    assert response.status_code == 401
    assert fake_planner.questions == []


def test_endpoint_active_token_reaches_planner():
    from app.chat import get_claim_extractor

    class _OkPlanner:
        def __init__(self) -> None:
            self.questions: list[str] = []

        def run(self, question: str):
            from app.planner import PlannerResult

            self.questions.append(question)
            return PlannerResult(answer="ok", trace=[], raw_results=[])

    class _NoClaims:
        def extract_claims(self, *, answer, tools, raw_results):
            return []

    ok_planner = _OkPlanner()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: ok_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: _NoClaims()
    validator = build_introspection_validator(
        _FakeIntrospector(IntrospectionResult(active=True, exp=9999999999)),
        clock=lambda: 1000.0,
    )
    app.dependency_overrides[get_token_validator] = lambda: validator

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"message": "hi", "patient_id": 1},
        headers={"Authorization": f"Bearer {_USER_TOK}"},
    )

    assert response.status_code == 200
    assert ok_planner.questions == ["hi"]

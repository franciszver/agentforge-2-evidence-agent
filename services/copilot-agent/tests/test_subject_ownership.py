"""#185: subject-based feedback ownership under per-user tokens.

#180 bound ``/feedback`` ownership to the raw bearer token (HMAC'd, never
stored raw) because no per-user principal was parsed out of OpenEMR's
introspection response. This suite covers what #185 adds on top:

  * ``app.chat.get_subject_resolver`` flag gating (no-op ``None`` OFF, a real
    subject lookup ON) -- mirrors ``test_launch_binding.py``'s coverage of
    ``get_launch_binding_checker``.
  * The RED-FIRST scenario the issue calls for: two DIFFERENT bearer tokens
    issued to the SAME OpenEMR subject can each rate that subject's own
    trace, and a caller with a DIFFERENT subject cannot -- end to end
    through ``POST /feedback``.
  * ``/chat`` wiring: the REQUEST span records ``owner_subject`` (not just
    ``owner_token_hash``) when a subject is available.

All ``/feedback``/``/chat`` overrides bypass ``Settings`` entirely (override
the derived dependency itself, ``get_subject_resolver``) -- the same pattern
``test_launch_binding.py`` uses for ``get_launch_binding_checker``, since
these dependency-provider functions read ``get_settings()`` as a bare call,
not via ``Depends``, so ``app.dependency_overrides[get_settings]`` does not
reach them (see that module's own gating tests for the ``monkeypatch.setenv``
alternative used below for the plain flag-gating checks).
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chat import (
    _default_subject_resolver,
    get_planner_factory,
    get_subject_resolver,
    get_token_validator,
    get_trace_store,
)
from app.main import app
from app.trace_store import SpanType, TraceStore
from tests.test_chat_endpoint import FakePlanner

_TEST_HASH_KEY = secrets.token_hex(16)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _override_ok_validator() -> None:
    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)


def _override_subject_resolver(mapping: dict[str, str | None]) -> None:
    app.dependency_overrides[get_subject_resolver] = lambda: (lambda token: mapping.get(token))


# --- get_subject_resolver flag gating --------------------------------------


def test_default_subject_resolver_is_always_none() -> None:
    assert _default_subject_resolver("any-token") is None


def test_get_subject_resolver_flag_off_returns_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COPILOT_PER_USER_TOKEN_ENABLED", raising=False)
    assert get_subject_resolver() is _default_subject_resolver


def test_get_subject_resolver_flag_on_returns_a_real_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COPILOT_PER_USER_TOKEN_ENABLED", "true")
    resolver = get_subject_resolver()
    assert resolver is not _default_subject_resolver
    assert callable(resolver)


# --- RED-FIRST: two tokens, same subject, both claim; different subject can't --


def _seed_owned_by_subject(trace_store: TraceStore, correlation_id: str, subject: str) -> None:
    """Model the REQUEST span a real /chat call under
    copilot_per_user_token_enabled=true would have written -- owned by
    ``subject`` (#185), not a token hash."""
    trace_store.record_request_span(
        correlation_id=correlation_id, start_ts=0.0, end_ts=0.1, ok=True, owner_subject=subject
    )


def test_two_different_tokens_for_the_same_subject_can_each_rate_its_trace(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    _override_subject_resolver({"token-session-1": "user-42", "token-session-2-reissued": "user-42"})
    _seed_owned_by_subject(trace_store, "corr-subj-1", "user-42")

    first = client.post(
        "/feedback",
        json={"correlation_id": "corr-subj-1", "thumb": "up"},
        headers={"Authorization": "Bearer token-session-1"},
    )
    second = client.post(
        "/feedback",
        json={"correlation_id": "corr-subj-1", "thumb": "down", "comment": "reissued token, same user"},
        headers={"Authorization": "Bearer token-session-2-reissued"},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    feedback_spans = [s for s in trace_store.get_spans("corr-subj-1") if s.span_type == SpanType.FEEDBACK]
    assert len(feedback_spans) == 2


def test_a_different_subject_cannot_rate_a_trace_it_does_not_own(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    _override_subject_resolver({"attacker-token": "user-99"})
    _seed_owned_by_subject(trace_store, "corr-subj-2", "user-42")

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-subj-2", "thumb": "down", "comment": "forged"},
        headers={"Authorization": "Bearer attacker-token"},
    )

    assert response.status_code == 403
    assert all(s.span_type != SpanType.FEEDBACK for s in trace_store.get_spans("corr-subj-2"))


def test_feedback_span_records_the_claiming_subject(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    _override_subject_resolver({"token-session-1": "user-42"})
    _seed_owned_by_subject(trace_store, "corr-subj-3", "user-42")

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-subj-3", "thumb": "up"},
        headers={"Authorization": "Bearer token-session-1"},
    )

    assert response.status_code == 201
    feedback_span = next(s for s in trace_store.get_spans("corr-subj-3") if s.span_type == SpanType.FEEDBACK)
    assert feedback_span.owner_kind == "subject"
    assert feedback_span.owner_subject == "user-42"


# --- /chat wiring: REQUEST span records owner_subject when available ------


def test_chat_endpoint_records_owner_subject_on_request_span_when_available(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    _override_subject_resolver({"token-a": "user-7"})
    fake_planner = FakePlanner(trace=[], answer="ok")
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer token-a"},
    )
    correlation_id = response.headers["X-Correlation-ID"]

    request_span = next(s for s in trace_store.get_spans(correlation_id) if s.span_type == SpanType.REQUEST)
    assert request_span.owner_kind == "subject"
    assert request_span.owner_subject == "user-7"
    assert request_span.owner_token_hash is None


def test_chat_endpoint_records_token_hash_owner_when_no_subject_available(tmp_path: Path) -> None:
    # Flag-off/dev-bridge regime: the resolver override returns None (as
    # _default_subject_resolver would), so the REQUEST span must fall back
    # to the #180 token-hash regime -- unchanged behaviour.
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    _override_subject_resolver({})
    fake_planner = FakePlanner(trace=[], answer="ok")
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer token-b"},
    )
    correlation_id = response.headers["X-Correlation-ID"]

    request_span = next(s for s in trace_store.get_spans(correlation_id) if s.span_type == SpanType.REQUEST)
    assert request_span.owner_kind == "token_hash"
    assert request_span.owner_subject is None
    assert request_span.owner_token_hash is not None


# --- LOW-3: a failed subject resolution logs its silent regime downgrade --


def test_chat_endpoint_warns_when_subject_resolution_falls_back_to_token_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("COPILOT_PER_USER_TOKEN_ENABLED", "true")
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    _override_subject_resolver({})  # resolver miss -> None, same as a failed introspection
    fake_planner = FakePlanner(trace=[], answer="ok")
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)

    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/chat",
            json={"message": "hello", "patient_id": 1},
            headers={"Authorization": "Bearer token-fallback"},
        )

    assert response.status_code == 200
    assert any(
        "subject resolution failed" in record.message and "token-hash fallback" in record.message
        for record in caplog.records
    )
    # No token, hash, or PHI in the log message itself.
    for record in caplog.records:
        assert "token-fallback" not in record.message


def test_chat_endpoint_does_not_warn_when_subject_resolution_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("COPILOT_PER_USER_TOKEN_ENABLED", "true")
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    _override_subject_resolver({"token-good": "user-42"})
    fake_planner = FakePlanner(trace=[], answer="ok")
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)

    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/chat",
            json={"message": "hello", "patient_id": 1},
            headers={"Authorization": "Bearer token-good"},
        )

    assert response.status_code == 200
    assert not any("subject resolution failed" in record.message for record in caplog.records)

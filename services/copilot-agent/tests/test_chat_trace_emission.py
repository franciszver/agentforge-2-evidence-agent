"""Hermetic test: a ``POST /chat`` invocation writes request + verification
spans to the trace store (P4.2 live-wiring scope).

Everything is faked (planner, extractor, token validator) exactly as in
``test_chat_endpoint.py``; the only addition is a ``TraceStore`` pointed at a
``tmp_path`` database via the ``get_trace_store`` dependency override -- this
test never touches the configured ``trace_db_path``.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chat import get_claim_extractor, get_planner_factory, get_token_validator, get_trace_store
from app.main import app
from app.planner import PlannerResult, ToolCallTrace
from app.schemas.planner import ToolName
from app.trace_store import SpanStatus, SpanType, TraceStore
from tests.test_chat_endpoint import FakeExtractor, FakePlanner, _iter_sse_events

# Derived per run (not a hardcoded literal) so no secret-shaped string is
# committed; stable within a run for any within-test consistency.
_TEST_HASH_KEY = secrets.token_hex(16)


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _override_ok_validator() -> None:
    def _validator(token: str) -> None:
        return None

    app.dependency_overrides[get_token_validator] = lambda: _validator


client = TestClient(app)


def test_chat_invocation_writes_request_and_verification_spans(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    trace = [ToolCallTrace(tool=ToolName.GET_MEDICATIONS, args={}, result={"count": 1}, error=None)]
    fake_planner = FakePlanner(trace=trace, answer="She is on lisinopril.")

    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractor()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = client.post(
        "/chat",
        json={"message": "What meds is she on?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    assert response.status_code == 200
    correlation_id = response.headers["X-Correlation-ID"]

    spans = trace_store.get_spans(correlation_id)
    span_types = {span.span_type for span in spans}

    assert SpanType.REQUEST in span_types
    assert SpanType.VERIFICATION in span_types

    request_span = next(span for span in spans if span.span_type == SpanType.REQUEST)
    assert request_span.correlation_id == correlation_id
    assert request_span.duration_ms >= 0

    verification_span = next(span for span in spans if span.span_type == SpanType.VERIFICATION)
    assert verification_span.verdict is not None


class _FailingPlanner:
    """A planner double that always raises -- drives the ``_stream_chat``
    error path (request span recorded ``ok=False``, exception re-raised)."""

    def run(self, question: str) -> PlannerResult:
        raise RuntimeError("boom")


def test_stream_chat_records_failed_request_span_on_exception(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)

    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: _FailingPlanner())
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractor()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    # Supply our own correlation id (the middleware honors an inbound
    # X-Correlation-ID header) so we can look up the span even though the
    # response never completes normally.
    correlation_id = "test-failure-correlation-id"
    with pytest.raises(RuntimeError, match="boom"):
        client.post(
            "/chat",
            json={"message": "hello", "patient_id": 1},
            headers={"Authorization": "Bearer good-token", "X-Correlation-ID": correlation_id},
        )

    spans = trace_store.get_spans(correlation_id)
    request_span = next(span for span in spans if span.span_type == SpanType.REQUEST)
    assert request_span.status == SpanStatus.FAIL


class _RaisingTraceStore:
    """A trace store whose every write raises -- models a real trace-store
    failure (PermissionError on a root-owned /data, disk full, locked DB).
    Proves span emission is best-effort: a trace failure must NEVER crash the
    clinician's /chat response."""

    def record_request_span(self, **kwargs: object) -> int:
        raise PermissionError("[Errno 13] Permission denied: '/data'")

    def record_verification_span(self, **kwargs: object) -> int:
        raise PermissionError("[Errno 13] Permission denied: '/data'")


def test_chat_survives_a_trace_store_write_failure() -> None:
    # A trace store that raises on every write must not break the response:
    # the answer + verification frames still stream and the request ends 200.
    fake_planner = FakePlanner(trace=[], answer="best-effort answer")
    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractor()
    app.dependency_overrides[get_trace_store] = lambda: _RaisingTraceStore()

    response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 200
    event_names = [name for name, _ in _iter_sse_events(response.text)]
    assert "answer" in event_names
    assert "verification" in event_names
    assert "done" in event_names


def test_chat_test_never_builds_the_real_data_trace_store() -> None:
    # Leak guard: this test sets NO get_trace_store override, relying solely
    # on the conftest autouse isolation fixture. If that fixture ever
    # regressed, the real get_trace_store would build a store against the
    # configured trace_db_path (/data) -- a mkdir that crashes on the CI
    # runner. Assert the real process-wide store is never constructed.
    import app.chat as chat_module

    fake_planner = FakePlanner(trace=[], answer="ok")
    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractor()

    response = client.post(
        "/chat",
        json={"message": "hi", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 200
    assert chat_module._default_trace_store is None


def test_get_trace_store_builds_from_settings_against_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The production dependency (never exercised by /chat tests, which
    # override it) builds a single cached store from Settings. Point Settings
    # at a tmp path so this proves the build without touching /data, and reset
    # the process-wide cache so the session leak guard still sees None.
    import app.chat as chat_module
    from app.config import Settings

    db_path = tmp_path / "prod_traces.db"
    monkeypatch.setattr(
        chat_module,
        "get_settings",
        lambda: Settings(trace_db_path=str(db_path), trace_args_hash_secret=_TEST_HASH_KEY),
    )
    monkeypatch.setattr(chat_module, "_default_trace_store", None)

    first = chat_module.get_trace_store()
    second = chat_module.get_trace_store()

    assert isinstance(first, TraceStore)
    assert first is second  # cached process-wide, built once
    assert db_path.exists()  # built against the configured path, not /data
    # monkeypatch restores _default_trace_store to None on teardown, keeping
    # the session leak guard (conftest) satisfied.

"""Hermetic tests for the ``POST /feedback`` endpoint (P4.3).

Persists a clinician's thumbs up/down + optional comment on a chat response,
linked via the shared P4.1 correlation id to that response's request/
verification spans (P4.2's ``TraceStore.record_feedback_span``, the P4.3
seam). Everything here uses the conftest tmp-path trace-store isolation --
no test writes to the configured ``trace_db_path`` (``/data``).
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chat import get_planner_factory, get_token_validator, get_trace_store
from app.feedback import MAX_COMMENT_LENGTH
from app.main import app
from app.trace_store import FeedbackThumb, SpanType, TraceStore
from tests.test_chat_endpoint import FakePlanner

# Derived per run (not a hardcoded literal) so no secret-shaped string is
# committed -- matches tests/conftest.py and tests/test_chat_trace_emission.py.
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


def test_thumbs_up_persists_and_reads_back_with_correlation_id(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-1", "thumb": "up"},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 201

    spans = trace_store.get_spans("corr-1")
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.FEEDBACK
    assert span.correlation_id == "corr-1"
    assert span.feedback_thumb == FeedbackThumb.UP
    assert span.feedback_comment is None


def test_thumbs_down_with_comment_persists(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-2", "thumb": "down", "comment": "Missed the recent A1C."},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 201

    spans = trace_store.get_spans("corr-2")
    assert len(spans) == 1
    span = spans[0]
    assert span.feedback_thumb == FeedbackThumb.DOWN
    assert span.feedback_comment == "Missed the recent A1C."


def test_feedback_span_shares_correlation_id_with_the_chat_response_it_rates(tmp_path: Path) -> None:
    # Linkage: a real /chat invocation's correlation id (from the P4.1
    # X-Correlation-ID response header) is used to post feedback, and the
    # resulting feedback span joins the same correlation id's request /
    # verification spans in the store -- exactly what the P4.5 dashboard /
    # P4.9 review queue need to reconstruct one invocation end to end.
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    fake_planner = FakePlanner(trace=[], answer="ok")
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)

    chat_response = client.post(
        "/chat",
        json={"message": "hello", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )
    correlation_id = chat_response.headers["X-Correlation-ID"]

    feedback_response = client.post(
        "/feedback",
        json={"correlation_id": correlation_id, "thumb": "up"},
        headers={"Authorization": "Bearer good-token"},
    )
    assert feedback_response.status_code == 201

    spans = trace_store.get_spans(correlation_id)
    span_types = {span.span_type for span in spans}
    assert SpanType.REQUEST in span_types
    assert SpanType.FEEDBACK in span_types
    assert all(span.correlation_id == correlation_id for span in spans)


def test_missing_correlation_id_returns_4xx_no_leak(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = client.post(
        "/feedback",
        json={"thumb": "up"},
        headers={"Authorization": "Bearer good-token"},
    )

    assert 400 <= response.status_code < 500
    assert "Traceback" not in response.text


def test_empty_correlation_id_returns_4xx(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = client.post(
        "/feedback",
        json={"correlation_id": "", "thumb": "up"},
        headers={"Authorization": "Bearer good-token"},
    )

    assert 400 <= response.status_code < 500


def test_invalid_thumb_value_returns_4xx_no_leak(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-3", "thumb": "sideways"},
        headers={"Authorization": "Bearer good-token"},
    )

    assert 400 <= response.status_code < 500
    assert "Traceback" not in response.text
    assert trace_store.get_spans("corr-3") == []


def test_over_long_comment_returns_4xx(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-4", "thumb": "up", "comment": "x" * (MAX_COMMENT_LENGTH + 1)},
        headers={"Authorization": "Bearer good-token"},
    )

    assert 400 <= response.status_code < 500
    assert trace_store.get_spans("corr-4") == []


def test_comment_at_max_length_is_accepted(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-5", "thumb": "up", "comment": "x" * MAX_COMMENT_LENGTH},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 201


class _RaisingTraceStore:
    """A trace store whose ``record_feedback_span`` always raises -- models a
    real write failure (permission error, full disk, locked DB). Feedback is
    a deliberate user action, so (unlike P4.2's passive spans) a write
    failure must surface to the caller as an error, not be swallowed."""

    def record_feedback_span(self, **kwargs: object) -> int:
        raise PermissionError("[Errno 13] Permission denied: '/data'")


def test_write_failure_returns_5xx_no_leak() -> None:
    _override_ok_validator()
    app.dependency_overrides[get_trace_store] = lambda: _RaisingTraceStore()

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-6", "thumb": "up"},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code >= 500
    assert "Errno 13" not in response.text
    assert "/data" not in response.text


def test_missing_token_returns_401(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    # No validator override -- default stub requires a non-empty token.

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-7", "thumb": "up"},
    )

    assert response.status_code == 401
    assert trace_store.get_spans("corr-7") == []


def test_rejected_token_returns_401(tmp_path: Path) -> None:
    from app.chat import TokenValidationError

    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    def _rejecting_validator(token: str) -> None:
        raise TokenValidationError("bad token")

    app.dependency_overrides[get_token_validator] = lambda: _rejecting_validator

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-8", "thumb": "up"},
        headers={"Authorization": "Bearer bad-token"},
    )

    assert response.status_code == 401
    assert trace_store.get_spans("corr-8") == []

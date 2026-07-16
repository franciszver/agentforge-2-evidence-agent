"""Hermetic HTML-contract tests for ``GET /dashboard`` (P4.5).

Follows the ``tests/test_chat_shell.py`` pattern (P0.6): a ``TestClient``
contract test, no browser. The trace store is isolated by the autouse
``tests/conftest.py::_isolate_trace_store`` fixture -- every test here reads
through the SAME ``get_trace_store`` dependency the isolation fixture already
overrides, so no extra plumbing is needed to avoid touching ``/data``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.chat import get_trace_store
from app.main import app
from app.trace_store import FeedbackThumb, TraceStore

client = TestClient(app)


def test_dashboard_returns_200_html() -> None:
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_dashboard_contains_viewport_meta() -> None:
    response = client.get("/dashboard")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in response.text


def test_dashboard_contains_required_metric_labels() -> None:
    response = client.get("/dashboard")
    body = response.text

    for label in (
        "Requests",
        "Error rate",
        "p50",
        "p95",
        "Tokens",
        "Tool calls",
        "Retries",
        "Verification pass rate",
        "Feedback",
    ):
        assert label in body, f"missing metric label: {label!r}"


def test_dashboard_empty_store_renders_without_crash() -> None:
    response = client.get("/dashboard")
    assert response.status_code == 200
    # No data yet -- percentiles/rates render as an explicit N/A, not a crash
    # and not a bare "None".
    assert "None" not in response.text


def test_dashboard_renders_real_values_from_seeded_spans(tmp_path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret="test-secret")
    trace_store.record_request_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True)
    trace_store.record_request_span(correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=True)
    trace_store.record_request_span(correlation_id="c3", start_ts=0.0, end_ts=0.1, ok=False)
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    try:
        response = client.get("/dashboard")
    finally:
        app.dependency_overrides.pop(get_trace_store, None)

    assert response.status_code == 200
    assert "3" in response.text  # request_count


def test_dashboard_does_not_render_feedback_comment_text(tmp_path) -> None:
    # Hard constraint: no PHI / no per-record detail. The feedback comment is
    # user-authored text about the response (not patient data) but it is
    # STILL per-record detail that belongs to the P4.9 review queue, not this
    # aggregate page -- it must never appear here.
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret="test-secret")
    sentinel_comment = "SENTINEL-COMMENT-do-not-render-me-verbatim"
    trace_store.record_feedback_span(
        correlation_id="c1",
        start_ts=0.0,
        end_ts=0.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment=sentinel_comment,
    )
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    try:
        response = client.get("/dashboard")
    finally:
        app.dependency_overrides.pop(get_trace_store, None)

    assert sentinel_comment not in response.text


def test_dashboard_no_external_network_reference() -> None:
    response = client.get("/dashboard")
    body_lower = response.text.lower()
    assert "http://" not in body_lower
    assert "https://" not in body_lower
    assert "cdn" not in body_lower


def test_dashboard_no_banner_when_all_metrics_healthy(tmp_path) -> None:
    # 1 fast, successful request -- nothing crosses any P4.6 threshold.
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret="test-secret")
    trace_store.record_request_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True)
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    try:
        response = client.get("/dashboard")
    finally:
        app.dependency_overrides.pop(get_trace_store, None)

    assert response.status_code == 200
    assert "data-testid=\"alert-banner\"" not in response.text


def test_dashboard_renders_banner_for_over_threshold_error_rate(tmp_path) -> None:
    # 2 failed requests out of 2 -> error_rate = 1.0, well over the 10% threshold.
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret="test-secret")
    trace_store.record_request_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=False)
    trace_store.record_request_span(correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=False)
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    try:
        response = client.get("/dashboard")
    finally:
        app.dependency_overrides.pop(get_trace_store, None)

    assert response.status_code == 200
    assert 'data-testid="alert-banner"' in response.text
    assert "error rate" in response.text.lower()

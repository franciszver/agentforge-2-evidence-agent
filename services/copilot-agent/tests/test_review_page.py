"""Hermetic HTML-contract tests for ``GET /review`` + ``GET /review/promote``
(P4.9). Follows ``tests/test_dashboard_page.py``'s pattern: a ``TestClient``
contract test, no browser (the browser scenario is
``tests/test_review_queue_browser.py``).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.chat import get_trace_store
from app.main import app
from app.trace_store import FeedbackThumb, TraceStore

client = TestClient(app)

_SECRET = "test-secret"


def _override(tmp_path: Path) -> TraceStore:
    store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_SECRET)
    app.dependency_overrides[get_trace_store] = lambda: store
    return store


def teardown_function() -> None:
    app.dependency_overrides.pop(get_trace_store, None)


# --- GET /review -------------------------------------------------------


def test_review_queue_returns_200_html(tmp_path: Path) -> None:
    _override(tmp_path)
    response = client.get("/review")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_review_queue_empty_store_renders_without_crash(tmp_path: Path) -> None:
    _override(tmp_path)
    response = client.get("/review")
    assert response.status_code == 200
    assert "None" not in response.text


def test_review_queue_lists_a_thumbs_down_entry(tmp_path: Path) -> None:
    store = _override(tmp_path)
    store.record_request_span(correlation_id="corr-1", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_feedback_span(
        correlation_id="corr-1",
        start_ts=0.0,
        end_ts=0.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment="wrong dose entirely",
    )

    response = client.get("/review")

    assert "corr-1" in response.text
    assert "wrong dose entirely" in response.text
    assert 'data-testid="promote-button"' in response.text
    assert 'data-correlation-id="corr-1"' in response.text


def test_review_queue_does_not_list_a_clean_thumbs_up_entry(tmp_path: Path) -> None:
    store = _override(tmp_path)
    store.record_request_span(correlation_id="corr-2", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_feedback_span(
        correlation_id="corr-2", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.UP, feedback_comment=None
    )

    response = client.get("/review")

    assert "corr-2" not in response.text


def test_review_queue_escapes_html_in_correlation_id_and_comment(tmp_path: Path) -> None:
    # correlation_id can be attacker-influenced (inbound X-Correlation-ID
    # header, app.correlation.CorrelationIdMiddleware) and feedback_comment
    # is user-authored free text -- both are rendered on this page for the
    # first time (the P4.5 dashboard deliberately never renders per-record
    # detail). Must be HTML-escaped, not interpolated raw, or this page is a
    # stored-XSS vector.
    store = _override(tmp_path)
    hostile_id = "<script>window.pwned=1</script>"
    store.record_request_span(correlation_id=hostile_id, start_ts=0.0, end_ts=0.1, ok=True)
    store.record_feedback_span(
        correlation_id=hostile_id,
        start_ts=0.0,
        end_ts=0.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment="<img src=x onerror=alert(1)>",
    )

    response = client.get("/review")

    assert "<script>window.pwned=1</script>" not in response.text
    assert "<img src=x onerror=alert(1)>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_review_queue_no_external_network_reference(tmp_path: Path) -> None:
    _override(tmp_path)
    response = client.get("/review")
    body_lower = response.text.lower()
    assert "http://" not in body_lower
    assert "https://" not in body_lower
    assert "cdn" not in body_lower


# --- GET /review/promote -------------------------------------------------


def test_promote_returns_a_yaml_body_for_a_known_correlation_id(tmp_path: Path) -> None:
    store = _override(tmp_path)
    store.record_request_span(correlation_id="corr-3", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_feedback_span(
        correlation_id="corr-3",
        start_ts=0.0,
        end_ts=0.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment="missed an interaction",
    )

    response = client.get("/review/promote", params={"correlation_id": "corr-3"})

    assert response.status_code == 200
    assert "category: regression" in response.text
    assert "corr-3" in response.text
    # #157: the raw clinician comment is scrubbed from the promoted export
    # (public evals/ repo) -- only a neutral TODO placeholder referencing the
    # correlation id is emitted. The comment stays in the local /review view.
    assert "missed an interaction" not in response.text
    assert "TODO" in response.text


def test_promote_unknown_correlation_id_returns_404_no_leak(tmp_path: Path) -> None:
    _override(tmp_path)
    response = client.get("/review/promote", params={"correlation_id": "does-not-exist"})
    assert response.status_code == 404
    assert "Traceback" not in response.text


def test_promote_missing_query_param_returns_4xx(tmp_path: Path) -> None:
    _override(tmp_path)
    response = client.get("/review/promote")
    assert 400 <= response.status_code < 500

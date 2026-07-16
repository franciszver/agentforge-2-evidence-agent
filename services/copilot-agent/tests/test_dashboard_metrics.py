"""Hermetic tests for the P4.5 dashboard aggregation queries (app.dashboard_metrics).

Every test uses a fresh ``tmp_path`` database via ``TraceStore`` -- NEVER the
configured ``trace_db_path`` / dev ``traces.db`` (see ``docs/TEST_PLAN.md``
Sec 7). Spans are seeded through ``TraceStore``'s public writer API (the same
one production code calls), then ``compute_dashboard_metrics`` is exercised
as a pure reader against the resulting file.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from app.dashboard_metrics import compute_dashboard_metrics
from app.trace_store import FeedbackThumb, TraceStore

# Derived per run (not a hardcoded literal) so no secret-shaped string is
# committed -- matches tests/test_trace_store.py.
_TEST_HASH_KEY = secrets.token_hex(16)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "traces.db")


@pytest.fixture
def store(db_path: str) -> TraceStore:
    return TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)


def test_empty_store_all_zero_or_none_no_crash(db_path: str) -> None:
    # Schema must exist even with zero spans -- construct the store first,
    # exactly as production does before the dashboard is ever hit.
    TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.request_count == 0
    assert metrics.error_rate is None
    assert metrics.p50_latency_ms is None
    assert metrics.p95_latency_ms is None
    assert metrics.avg_tokens_per_request is None
    assert metrics.tool_call_count == 0
    assert metrics.retry_count == 0
    assert metrics.verification_pass_rate is None
    assert metrics.feedback_up_count == 0
    assert metrics.feedback_down_count == 0


def test_request_count_and_error_rate(store: TraceStore, db_path: str) -> None:
    store.record_request_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_request_span(correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_request_span(correlation_id="c3", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_request_span(correlation_id="c4", start_ts=0.0, end_ts=0.1, ok=False)

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.request_count == 4
    assert metrics.error_rate == pytest.approx(0.25)


def test_p50_p95_latency_nearest_rank_on_known_durations(store: TraceStore, db_path: str) -> None:
    # duration_ms = (end_ts - start_ts) * 1000; seed 100 request spans with
    # durations 1..100 ms so nearest-rank percentiles land on clean values:
    # p50 -> ceil(0.50 * 100) = 50th smallest value = 50ms
    # p95 -> ceil(0.95 * 100) = 95th smallest value = 95ms
    for i in range(1, 101):
        store.record_request_span(correlation_id=f"c{i}", start_ts=0.0, end_ts=i / 1000.0, ok=True)

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.p50_latency_ms == pytest.approx(50.0, abs=0.01)
    assert metrics.p95_latency_ms == pytest.approx(95.0, abs=0.01)


def test_p50_p95_none_when_no_request_spans(store: TraceStore, db_path: str) -> None:
    # Only a tool span, no request spans -- latency percentiles have nothing
    # to compute over.
    store.record_tool_span(
        correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True, tool_name="get_medications", args={}
    )

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.p50_latency_ms is None
    assert metrics.p95_latency_ms is None


def test_avg_tokens_per_request(store: TraceStore, db_path: str) -> None:
    store.record_request_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_request_span(correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_llm_span(
        correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True, model="qwen3:4b", tokens_in=100, tokens_out=50
    )
    store.record_llm_span(
        correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=True, model="qwen3:4b", tokens_in=80, tokens_out=20
    )

    metrics = compute_dashboard_metrics(db_path)

    # (100+50+80+20) tokens / 2 requests = 125.0
    assert metrics.avg_tokens_per_request == pytest.approx(125.0)


def test_avg_tokens_per_request_none_when_no_requests(db_path: str) -> None:
    TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.avg_tokens_per_request is None


def test_tool_call_count_and_retry_count(store: TraceStore, db_path: str) -> None:
    store.record_tool_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True, tool_name="get_labs", args={})
    store.record_tool_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True, tool_name="get_labs", args={})
    store.record_tool_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True, tool_name="get_labs", args={})
    store.record_tool_span(
        correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=False, tool_name="get_labs", args={}, error_category="timeout"
    )
    store.record_tool_span(
        correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=False, tool_name="get_labs", args={}, error_category="timeout"
    )

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.tool_call_count == 5
    assert metrics.retry_count == 2


def test_verification_pass_rate(store: TraceStore, db_path: str) -> None:
    for _ in range(3):
        store.record_verification_span(
            correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True, verdict="verified", claim_count=1, stripped_count=0
        )
    store.record_verification_span(
        correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=True, verdict="partially_verified", claim_count=2, stripped_count=1
    )
    store.record_verification_span(
        correlation_id="c3", start_ts=0.0, end_ts=0.1, ok=True, verdict="blocked", claim_count=0, stripped_count=0
    )

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.verification_pass_rate == pytest.approx(3 / 5)


def test_verification_pass_rate_none_when_no_verification_spans(db_path: str) -> None:
    TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.verification_pass_rate is None


def test_feedback_up_and_down_counts_no_duplicates(store: TraceStore, db_path: str) -> None:
    store.record_feedback_span(
        correlation_id="c1", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.UP, feedback_comment=None
    )
    store.record_feedback_span(
        correlation_id="c2", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.DOWN, feedback_comment=None
    )

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.feedback_up_count == 1
    assert metrics.feedback_down_count == 1


def test_feedback_dedup_commented_downvote_counted_once(store: TraceStore, db_path: str) -> None:
    # This is the exact P4.4 UI shape (issue #54): a thumbs-down click posts
    # {thumb: down} immediately, then an optional comment box reveals and its
    # submission posts a SECOND {thumb: down, comment: "..."} span for the
    # SAME correlation id. Both rows share span_type=feedback and
    # correlation_id -- must collapse to ONE feedback event, not two.
    store.record_feedback_span(
        correlation_id="c-down", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.DOWN, feedback_comment=None
    )
    store.record_feedback_span(
        correlation_id="c-down",
        start_ts=1.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment="Missed the recent A1C.",
    )
    # A separate, single (non-duplicated) upvote on another correlation id --
    # proves the dedup groups BY correlation id, not collapsing everything.
    store.record_feedback_span(
        correlation_id="c-up", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.UP, feedback_comment=None
    )

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.feedback_down_count == 1
    assert metrics.feedback_up_count == 1


def test_feedback_dedup_does_not_collapse_different_correlation_ids(store: TraceStore, db_path: str) -> None:
    for i in range(5):
        store.record_feedback_span(
            correlation_id=f"c{i}", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.UP, feedback_comment=None
        )

    metrics = compute_dashboard_metrics(db_path)

    assert metrics.feedback_up_count == 5

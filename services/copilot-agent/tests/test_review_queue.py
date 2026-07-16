"""Hermetic tests for ``app.review_queue.list_review_queue`` (P4.9).

Reads the P4.2 trace store (via an isolated ``tmp_path`` ``TraceStore`` --
never the configured ``trace_db_path``, per ``docs/TEST_PLAN.md`` Sec 7) and
builds the review queue: correlation ids with a thumbs-down feedback span
and/or a verification span whose verdict is not ``verified``. Everything
else (a clean ``verified`` request with a thumbs-up, or no feedback at all)
must NOT appear -- the queue is a worklist, not a full trace browser.
"""

from __future__ import annotations

from pathlib import Path

from app.review_queue import list_review_queue
from app.trace_store import FeedbackThumb, TraceStore

_SECRET = "test-secret"


def _store(tmp_path: Path) -> TraceStore:
    return TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_SECRET)


def test_empty_store_yields_empty_queue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert list_review_queue(store) == []


def test_thumbs_down_appears_in_the_queue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_request_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_feedback_span(
        correlation_id="c1",
        start_ts=0.0,
        end_ts=0.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment="wrong dose",
    )

    entries = list_review_queue(store)

    assert len(entries) == 1
    assert entries[0].correlation_id == "c1"
    assert entries[0].feedback_thumb == FeedbackThumb.DOWN
    assert entries[0].feedback_comment == "wrong dose"


def test_thumbs_up_does_not_appear_in_the_queue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_request_span(correlation_id="c1", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_feedback_span(
        correlation_id="c1", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.UP, feedback_comment=None
    )

    assert list_review_queue(store) == []


def test_verification_failure_appears_in_the_queue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_request_span(correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_verification_span(
        correlation_id="c2", start_ts=0.0, end_ts=0.0, ok=True, verdict="blocked", claim_count=2, stripped_count=2
    )

    entries = list_review_queue(store)

    assert len(entries) == 1
    assert entries[0].correlation_id == "c2"
    assert entries[0].verdict == "blocked"


def test_verified_verdict_does_not_appear_in_the_queue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_request_span(correlation_id="c2", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_verification_span(
        correlation_id="c2", start_ts=0.0, end_ts=0.0, ok=True, verdict="verified", claim_count=2, stripped_count=0
    )

    assert list_review_queue(store) == []


def test_feedback_is_deduped_preferring_the_row_with_a_comment(tmp_path: Path) -> None:
    # Mirrors the P4.5 dashboard's dedup (#54): an immediate {thumb: down}
    # click, then a follow-up comment -- both share correlation_id and must
    # collapse into ONE queue entry carrying the comment.
    store = _store(tmp_path)
    store.record_request_span(correlation_id="c3", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_feedback_span(
        correlation_id="c3", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.DOWN, feedback_comment=None
    )
    store.record_feedback_span(
        correlation_id="c3",
        start_ts=1.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment="the actual comment",
    )

    entries = list_review_queue(store)

    assert len(entries) == 1
    assert entries[0].feedback_comment == "the actual comment"


def test_tool_call_count_is_included(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_request_span(correlation_id="c4", start_ts=0.0, end_ts=0.1, ok=True)
    store.record_tool_span(
        correlation_id="c4", start_ts=0.0, end_ts=0.05, ok=True, tool_name="get_medications", args={"a": 1}
    )
    store.record_feedback_span(
        correlation_id="c4", start_ts=0.1, end_ts=0.1, feedback_thumb=FeedbackThumb.DOWN, feedback_comment=None
    )

    entries = list_review_queue(store)

    assert entries[0].tool_call_count == 1


def test_most_recent_activity_sorts_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_feedback_span(
        correlation_id="older", start_ts=0.0, end_ts=0.0, feedback_thumb=FeedbackThumb.DOWN, feedback_comment=None
    )
    store.record_feedback_span(
        correlation_id="newer", start_ts=1.0, end_ts=1.0, feedback_thumb=FeedbackThumb.DOWN, feedback_comment=None
    )

    entries = list_review_queue(store)

    assert [e.correlation_id for e in entries] == ["newer", "older"]

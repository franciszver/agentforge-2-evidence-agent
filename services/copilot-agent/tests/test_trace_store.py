"""Hermetic tests for the durable SQLite trace store (P4.2).

Every test uses a fresh ``tmp_path`` database -- NEVER the configured
``trace_db_path`` / dev ``traces.db`` (see ``docs/TEST_PLAN.md`` Sec 7,
"agent-service tests write only to per-test temporary SQLite databases").

The no-PHI tests are the load-bearing ones here: the store persists to disk,
so anything written is a durable liability. They assert -- by inspecting the
raw database bytes, not just the typed accessors -- that a value passed as
tool ``args`` never appears verbatim anywhere in the file.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from app.trace_store import FeedbackThumb, SpanStatus, SpanType, TraceStore, hash_args

# Derived (not a hardcoded literal) so no secret-shaped string is committed;
# stable within a run, so the store fixture and the hash-equality assertions
# below share the SAME key and still prove HMAC keying.
_TEST_HASH_KEY = secrets.token_hex(16)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "traces.db")


@pytest.fixture
def store(db_path: str) -> TraceStore:
    return TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)


def test_schema_created_idempotently(db_path: str) -> None:
    # Constructing twice against the same path must not raise.
    TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)
    TraceStore(db_path=db_path, hash_secret=_TEST_HASH_KEY)


def test_record_request_span_write_and_read_back(store: TraceStore) -> None:
    store.record_request_span(correlation_id="corr-1", start_ts=100.0, end_ts=100.25, ok=True)

    spans = store.get_spans("corr-1")

    assert len(spans) == 1
    span = spans[0]
    assert span.correlation_id == "corr-1"
    assert span.span_type == SpanType.REQUEST
    assert span.status == SpanStatus.OK
    assert span.duration_ms == pytest.approx(250.0)


def test_record_request_span_failure_status(store: TraceStore) -> None:
    store.record_request_span(correlation_id="corr-1", start_ts=1.0, end_ts=2.0, ok=False)

    span = store.get_spans("corr-1")[0]
    assert span.status == SpanStatus.FAIL


def test_record_tool_span_write_and_read_back(store: TraceStore) -> None:
    store.record_tool_span(
        correlation_id="corr-2",
        start_ts=10.0,
        end_ts=10.5,
        ok=True,
        tool_name="get_medications",
        args={"limit": 3},
    )

    span = store.get_spans("corr-2")[0]
    assert span.span_type == SpanType.TOOL
    assert span.tool_name == "get_medications"
    assert span.error_category is None
    assert span.args_hash is not None
    assert len(span.args_hash) == 64  # sha256 hex digest


def test_record_tool_span_hashes_args_not_raw(store: TraceStore) -> None:
    raw_value = "PHI-SENTINEL-John Doe MRN 00099"
    store.record_tool_span(
        correlation_id="corr-3",
        start_ts=0.0,
        end_ts=0.1,
        ok=True,
        tool_name="get_allergies",
        args={"patient_note": raw_value},
    )

    span = store.get_spans("corr-3")[0]
    assert span.args_hash != raw_value
    assert raw_value not in span.args_hash
    # Deterministic: identical args + secret hash identically.
    assert span.args_hash == hash_args({"patient_note": raw_value}, _TEST_HASH_KEY)


def test_record_tool_span_failure_records_error_category(store: TraceStore) -> None:
    store.record_tool_span(
        correlation_id="corr-4",
        start_ts=0.0,
        end_ts=0.1,
        ok=False,
        tool_name="get_labs",
        args={},
        error_category="not_found",
    )

    span = store.get_spans("corr-4")[0]
    assert span.status == SpanStatus.FAIL
    assert span.error_category == "not_found"


def test_record_llm_span_write_and_read_back(store: TraceStore) -> None:
    store.record_llm_span(
        correlation_id="corr-5",
        start_ts=5.0,
        end_ts=6.0,
        ok=True,
        model="qwen3:4b",
        tokens_in=120,
        tokens_out=45,
    )

    span = store.get_spans("corr-5")[0]
    assert span.span_type == SpanType.LLM
    assert span.model == "qwen3:4b"
    assert span.tokens_in == 120
    assert span.tokens_out == 45


def test_record_verification_span_write_and_read_back(store: TraceStore) -> None:
    store.record_verification_span(
        correlation_id="corr-6",
        start_ts=1.0,
        end_ts=1.2,
        ok=True,
        verdict="verified",
        claim_count=3,
        stripped_count=0,
    )

    span = store.get_spans("corr-6")[0]
    assert span.span_type == SpanType.VERIFICATION
    assert span.verdict == "verified"
    assert span.claim_count == 3
    assert span.stripped_count == 0


def test_record_feedback_span_write_and_read_back(store: TraceStore) -> None:
    store.record_feedback_span(
        correlation_id="corr-7",
        start_ts=1.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.UP,
        feedback_comment="Helpful and accurate.",
    )

    span = store.get_spans("corr-7")[0]
    assert span.span_type == SpanType.FEEDBACK
    assert span.status == SpanStatus.OK
    assert span.feedback_thumb == FeedbackThumb.UP
    assert span.feedback_comment == "Helpful and accurate."


def test_record_feedback_span_comment_is_optional(store: TraceStore) -> None:
    store.record_feedback_span(
        correlation_id="corr-8",
        start_ts=1.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment=None,
    )

    span = store.get_spans("corr-8")[0]
    assert span.feedback_thumb == FeedbackThumb.DOWN
    assert span.feedback_comment is None


def test_get_spans_filters_by_correlation_id(store: TraceStore) -> None:
    store.record_request_span(correlation_id="corr-a", start_ts=0.0, end_ts=1.0, ok=True)
    store.record_request_span(correlation_id="corr-b", start_ts=0.0, end_ts=1.0, ok=True)
    store.record_verification_span(
        correlation_id="corr-a", start_ts=1.0, end_ts=1.1, ok=True, verdict="blocked", claim_count=0, stripped_count=0
    )

    spans_a = store.get_spans("corr-a")
    spans_b = store.get_spans("corr-b")

    assert len(spans_a) == 2
    assert all(span.correlation_id == "corr-a" for span in spans_a)
    assert len(spans_b) == 1


def test_get_spans_unknown_correlation_id_returns_empty(store: TraceStore) -> None:
    assert store.get_spans("no-such-id") == []


def test_no_phi_persisted_across_all_span_types(store: TraceStore, db_path: str) -> None:
    """The rigorous no-PHI check: write one of every span type using
    record-data-shaped values, then scan the RAW database bytes on disk --
    not just the typed accessors -- for anything that looks like patient
    record content. Only the feedback comment (explicitly permitted,
    user-authored text about the response) is allowed to appear verbatim."""
    sentinel_drug_name = "Lisinopril-10mg-PATIENT-SPECIFIC"
    sentinel_note = "Patient reports chest pain since Tuesday"

    store.record_request_span(correlation_id="corr-phi", start_ts=0.0, end_ts=1.0, ok=True)
    store.record_tool_span(
        correlation_id="corr-phi",
        start_ts=0.0,
        end_ts=1.0,
        ok=True,
        tool_name="get_medications",
        args={"drug_name": sentinel_drug_name, "note": sentinel_note},
    )
    store.record_llm_span(
        correlation_id="corr-phi", start_ts=0.0, end_ts=1.0, ok=True, model="qwen3:4b", tokens_in=10, tokens_out=5
    )
    store.record_verification_span(
        correlation_id="corr-phi", start_ts=0.0, end_ts=1.0, ok=True, verdict="verified", claim_count=1, stripped_count=0
    )
    store.record_feedback_span(
        correlation_id="corr-phi",
        start_ts=0.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.UP,
        feedback_comment="Great answer, matched the chart.",
    )

    raw_bytes = Path(db_path).read_bytes()

    assert sentinel_drug_name.encode() not in raw_bytes
    assert sentinel_note.encode() not in raw_bytes


def test_hash_args_is_deterministic_and_order_independent() -> None:
    assert hash_args({"a": 1, "b": 2}, _TEST_HASH_KEY) == hash_args({"b": 2, "a": 1}, _TEST_HASH_KEY)


def test_hash_args_differs_for_different_values() -> None:
    assert hash_args({"a": 1}, _TEST_HASH_KEY) != hash_args({"a": 2}, _TEST_HASH_KEY)


def test_hash_args_is_keyed_not_a_bare_hash() -> None:
    # A bare SHA-256 would let anyone recompute the hash without the secret,
    # defeating the point of hashing low-entropy args (e.g. a patient id)
    # instead of storing them raw -- see module docstring's "NO PHI ON DISK"
    # section. Different keys over the SAME args must disagree.
    args = {"patient_id": 42}
    key_a = secrets.token_hex(16)
    key_b = secrets.token_hex(16)
    assert hash_args(args, key_a) != hash_args(args, key_b)

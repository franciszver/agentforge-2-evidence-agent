"""Golden eval set (P3G.1), category ``no_phi_in_logs``.

Boolean rubric: a PHI marker threaded through the query/document never
appears in emitted logs, traces, or the per-encounter observability record.
Extends the load-bearing no-PHI regression guards already pinned in
``services/copilot-agent/tests/test_trace_store.py``
(``test_no_phi_persisted_across_all_span_types``) and
``tests/test_encounter_observability.py``
(``test_no_phi_in_encounter_record_or_logs``) -- this is a curated golden
slice covering the Phase-2 observability surface (P3.8/P3.9), not a
duplicate of either module's exhaustive coverage.

Fully deterministic/hermetic: real ``TraceStore`` (a tmp sqlite file) and
real ``app.encounter_observability``/``app.chat`` logging code, driven with
crafted fixtures -- no live model, no network, nothing to record/replay.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_AGENT_ROOT = Path(__file__).resolve().parents[2] / "services" / "copilot-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from app.chat import _log_encounter_record  # noqa: E402
from app.encounter_observability import build_encounter_record, extraction_confidence_proxy  # noqa: E402
from app.ingestion import IngestionResult  # noqa: E402
from app.schemas.ingestion import Citation, LabFlagCode, LabResultFact  # noqa: E402
from app.schemas.reranking import RerankedChunk  # noqa: E402
from app.trace_store import FeedbackThumb, SpanType, TraceStore  # noqa: E402

pytestmark = pytest.mark.no_phi_in_logs

_TEST_HASH_KEY = "eval-harness-not-a-real-secret"  # noqa: S105
_PHI_MARKER = "ZZ-EVAL-PHI-MARKER-Metformin-500mg-Patient-7"


@pytest.fixture
def trace_store(tmp_path: Path) -> TraceStore:
    return TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)


# ---------------------------------------------------------------------------
# case: no-phi-marker-in-tool-args-hash-or-raw-db-bytes
# ---------------------------------------------------------------------------


def test_no_phi_marker_in_tool_args_hash_or_raw_db_bytes(trace_store: TraceStore, tmp_path: Path) -> None:
    trace_store.record_tool_span(
        correlation_id="corr-eval-phi",
        start_ts=0.0,
        end_ts=1.0,
        ok=True,
        tool_name="get_medications",
        args={"note": _PHI_MARKER},
    )

    span = trace_store.get_spans("corr-eval-phi")[0]
    assert span.args_hash is not None
    assert _PHI_MARKER not in span.args_hash

    raw_bytes = (tmp_path / "traces.db").read_bytes()
    assert _PHI_MARKER.encode() not in raw_bytes


# ---------------------------------------------------------------------------
# case: no-phi-marker-in-feedback-span-columns-other-than-the-comment-field
# ---------------------------------------------------------------------------


def test_no_phi_marker_leaks_via_worker_or_verification_span_columns(trace_store: TraceStore, tmp_path: Path) -> None:
    trace_store.record_worker_span(
        correlation_id="corr-eval-phi-2",
        start_ts=0.0,
        end_ts=1.0,
        ok=True,
        worker_name="evidence-retriever",
        sub_task_type="RetrieveSubTask",
    )
    trace_store.record_verification_span(
        correlation_id="corr-eval-phi-2",
        start_ts=0.0,
        end_ts=1.0,
        ok=True,
        verdict="verified",
        claim_count=1,
        stripped_count=0,
    )
    trace_store.record_feedback_span(
        correlation_id="corr-eval-phi-2",
        start_ts=0.0,
        end_ts=1.0,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment="Not helpful",  # explicitly-permitted user text -- deliberately NOT the marker
    )

    raw_bytes = (tmp_path / "traces.db").read_bytes()
    assert _PHI_MARKER.encode() not in raw_bytes
    spans = trace_store.get_spans("corr-eval-phi-2")
    assert {s.span_type for s in spans} == {SpanType.WORKER, SpanType.VERIFICATION, SpanType.FEEDBACK}


# ---------------------------------------------------------------------------
# case: no-phi-marker-in-retrieval-hit-summary
# ---------------------------------------------------------------------------


def test_no_phi_marker_in_encounter_record_retrieval_summary(trace_store: TraceStore) -> None:
    marker_chunk = RerankedChunk(
        chunk_id="a1c-targets#target-ranges",
        doc_id="a1c-targets",
        title="A1c Targets for Adults with Diabetes",
        section="Target Ranges",
        text=f"A1c target below 7% for {_PHI_MARKER}.",
        scores={"hybrid": 0.9},
        rerank_score=0.9,
    )

    record = build_encounter_record("corr-eval-phi-3", trace_store, retrieval_chunks=[marker_chunk])

    assert _PHI_MARKER not in repr(record)
    assert _PHI_MARKER not in str(record)
    assert record.retrieval_hit_count == 1
    assert record.retrieval_top_scores == [0.9]


# ---------------------------------------------------------------------------
# case: no-phi-marker-in-extraction-confidence-proxy-or-encounter-record
# ---------------------------------------------------------------------------


def test_no_phi_marker_in_encounter_record_extraction_confidence(trace_store: TraceStore) -> None:
    marker_fact = LabResultFact(
        test="Glucose",
        value=_PHI_MARKER,
        unit="mg/dL",
        reference_range="70-99",
        collection_date="2026-06-01",
        abnormal_flag=LabFlagCode.NORMAL,
        citation=Citation(
            source_type="lab_pdf",
            source_id="doc-eval-phi",
            page_or_section="page 1",
            field_or_chunk_id="Glucose",
            quote_or_value=f"Glucose: {_PHI_MARKER}",
        ),
    )
    ingestion_result = IngestionResult(source_id="doc-eval-phi", facts=[marker_fact], pages_total=1, failed_pages=[])
    assert extraction_confidence_proxy(ingestion_result) == 1.0

    record = build_encounter_record("corr-eval-phi-4", trace_store, ingestion_result=ingestion_result)

    assert _PHI_MARKER not in repr(record)
    assert _PHI_MARKER not in str(record)
    assert record.extraction_confidence == 1.0


# ---------------------------------------------------------------------------
# case: no-phi-marker-in-the-per-turn-encounter-log-line
# ---------------------------------------------------------------------------


def test_no_phi_marker_in_log_encounter_record_output(
    trace_store: TraceStore, caplog: pytest.LogCaptureFixture
) -> None:
    marker_chunk = RerankedChunk(
        chunk_id="a1c-targets#target-ranges",
        doc_id="a1c-targets",
        title="A1c Targets for Adults with Diabetes",
        section="Target Ranges",
        text=f"A1c target below 7% for {_PHI_MARKER}.",
        scores={"hybrid": 0.9},
        rerank_score=0.9,
    )

    with caplog.at_level(logging.INFO, logger="app.chat"):
        _log_encounter_record(trace_store, "corr-eval-phi-5", [marker_chunk])

    for log_record in caplog.records:
        assert _PHI_MARKER not in log_record.getMessage()
        for value in log_record.__dict__.values():
            assert _PHI_MARKER not in str(value)


# ---------------------------------------------------------------------------
# case: no-phi-marker-in-log-when-encounter-record-build-fails
# ---------------------------------------------------------------------------


def test_no_phi_marker_in_log_when_encounter_record_build_fails(
    trace_store: TraceStore, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_log_encounter_record`` is best-effort (module docstring) -- a build
    failure must log the failure by TYPE ONLY, never the exception message
    (which could embed caller-supplied, PHI-bearing text)."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise ValueError(f"boom while processing {_PHI_MARKER}")

    monkeypatch.setattr("app.chat.build_encounter_record", _raise)

    with caplog.at_level(logging.WARNING, logger="app.chat"):
        _log_encounter_record(trace_store, "corr-eval-phi-6", [])

    assert any("encounter record build failed" in r.getMessage() for r in caplog.records)
    for log_record in caplog.records:
        assert _PHI_MARKER not in log_record.getMessage()
        for value in log_record.__dict__.values():
            assert _PHI_MARKER not in str(value)

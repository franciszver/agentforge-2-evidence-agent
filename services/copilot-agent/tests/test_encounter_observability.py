"""Red-first tests for P3.8's per-encounter observability record
(`app/encounter_observability.py`, `docs/W2_ARCHITECTURE.md` "Observability"
row).

Drives a simulated encounter through the real P3.5 ``Supervisor`` (an
``evidence-retriever`` dispatch plus an ``intake-extractor`` ingestion
dispatch, both wired to a real ``TraceStore`` so worker spans land in the
same durable sink `app.chat` already writes tool/llm spans to), then builds
the record and asserts its shape. Everything is hermetic: retrieval/rerank
replay committed fixtures (no live Ollama call, same doubles
``tests/test_supervisor.py`` uses), and ingestion uses the scripted VLM
double from ``tests/test_ingestion.py``.

The no-PHI test is the load-bearing one: a marker patient value threaded
through the pipeline (a sub-task field, an ingested document's file path)
must never appear anywhere in the built record, its string/repr form, or any
log line emitted while building it.
"""

from __future__ import annotations

import secrets

import pytest

from app.correlation import correlation_scope
from app.encounter_observability import (
    EncounterRecord,
    EvalOutcome,
    build_encounter_record,
    extraction_confidence_proxy,
    retrieval_summary,
)
from app.ingestion import IngestionResult, LocalIngestionStore
from app.reranking import RERANKER_SCORES_PATH, RecordedRerankScorer, Reranker, load_recorded_reranker_scores
from app.retrieval import build_retriever_from_corpus, recorded_query_vector
from app.schemas.ingestion import LabPageExtraction
from app.supervisor import EvidenceRetrieverWorker, IngestSubTask, IntakeExtractorWorker, RetrieveSubTask, Supervisor
from app.trace_store import TraceStore
from scripts.retrieval_golden_queries import GOLDEN_QUERIES
from tests.test_ingestion import _FakeVlmOllama, _FIXTURE_PATH, _PAGE_1_ROWS, _PAGE_2_ROWS

_TEST_HASH_KEY = secrets.token_hex(16)
_PHI_MARKER = "Warfarin-5mg-PATIENT-SPECIFIC-MARKER-99"


@pytest.fixture
def trace_store(tmp_path) -> TraceStore:
    return TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)


def _run_simulated_encounter(trace_store: TraceStore, tmp_path, *, patient_id: int = 1) -> tuple[str, IngestionResult, list]:
    ollama = _FakeVlmOllama([LabPageExtraction(rows=_PAGE_1_ROWS), LabPageExtraction(rows=_PAGE_2_ROWS)])
    store = LocalIngestionStore(tmp_path)
    supervisor = Supervisor(
        intake_worker=IntakeExtractorWorker(ollama_client=ollama, document_store=store, fact_store=store),
        evidence_worker=EvidenceRetrieverWorker(
            retriever=build_retriever_from_corpus(),
            reranker=Reranker(RecordedRerankScorer(load_recorded_reranker_scores(RERANKER_SCORES_PATH))),
        ),
        trace_store=trace_store,
    )

    query, _ = GOLDEN_QUERIES[0]
    with correlation_scope() as correlation_id:
        retrieve_result = supervisor.handle(RetrieveSubTask(query=query, k=3, query_vector=recorded_query_vector(query)))
        ingest_result = supervisor.handle(
            IngestSubTask(patient_id=patient_id, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf")
        )
        trace_store.record_llm_span(
            correlation_id=correlation_id, start_ts=0.0, end_ts=0.1, ok=True, model="qwen3:4b", tokens_in=100, tokens_out=40
        )
        trace_store.record_llm_span(
            correlation_id=correlation_id, start_ts=0.1, end_ts=0.2, ok=True, model="qwen3:4b", tokens_in=50, tokens_out=20
        )

    return correlation_id, ingest_result.payload, retrieve_result.payload


def test_build_encounter_record_from_a_simulated_encounter(trace_store: TraceStore, tmp_path) -> None:
    correlation_id, ingestion_result, reranked_chunks = _run_simulated_encounter(trace_store, tmp_path)

    record = build_encounter_record(
        correlation_id,
        trace_store,
        retrieval_chunks=reranked_chunks,
        ingestion_result=ingestion_result,
        cost_per_1k_tokens_usd=0.002,
    )

    assert isinstance(record, EncounterRecord)
    assert record.correlation_id == correlation_id

    # Ordered worker sequence: evidence-retriever dispatched before
    # intake-extractor, matching call order above.
    worker_steps = [step for step in record.steps if step.kind == "worker"]
    assert [step.name for step in worker_steps] == ["evidence-retriever", "intake-extractor"]
    assert all(step.ok for step in worker_steps)
    assert all(step.duration_ms >= 0 for step in worker_steps)

    # Per-step latency + token aggregation from the llm spans recorded above.
    llm_steps = [step for step in record.steps if step.kind == "llm"]
    assert len(llm_steps) == 2
    assert record.total_tokens_in == 150
    assert record.total_tokens_out == 60
    assert record.tokens_by_model["qwen3:4b"].tokens_in == 150
    assert record.tokens_by_model["qwen3:4b"].tokens_out == 60
    assert record.cost_estimate_usd == pytest.approx((150 + 60) / 1000 * 0.002)
    assert record.cost_estimate_note  # documented, non-empty

    # Retrieval hit count + top scores.
    assert record.retrieval_hit_count == len(reranked_chunks)
    assert record.retrieval_top_scores
    assert record.retrieval_top_scores == sorted(record.retrieval_top_scores, reverse=True)

    # Extraction-confidence proxy: both pages of the fixture succeed.
    assert record.extraction_confidence == 1.0
    assert record.extraction_confidence_note  # documented, non-empty


def test_retrieval_summary_counts_hits_and_top_scores() -> None:
    from app.schemas.reranking import RerankedChunk

    reranked = [
        RerankedChunk(chunk_id="a", doc_id="d", title="t", section="s", text="t", scores={}, rerank_score=0.9),
        RerankedChunk(chunk_id="b", doc_id="d", title="t", section="s", text="t", scores={}, rerank_score=0.4),
        RerankedChunk(chunk_id="c", doc_id="d", title="t", section="s", text="t", scores={}, rerank_score=0.7),
    ]
    hit_count, top_scores = retrieval_summary(reranked, top_n=2)
    assert hit_count == 3
    assert top_scores == [0.9, 0.7]


def test_extraction_confidence_proxy_reflects_failed_pages() -> None:
    full = IngestionResult(source_id="s1", facts=[], pages_total=2, failed_pages=[])
    assert extraction_confidence_proxy(full) == 1.0

    partial = IngestionResult(source_id="s2", facts=[], pages_total=2, failed_pages=[2])
    assert extraction_confidence_proxy(partial) == 0.5

    empty = IngestionResult(source_id="s3", facts=[], pages_total=0, failed_pages=[])
    assert extraction_confidence_proxy(empty) is None


def test_eval_outcome_is_none_until_populated(trace_store: TraceStore, tmp_path) -> None:
    correlation_id, ingestion_result, reranked_chunks = _run_simulated_encounter(trace_store, tmp_path)

    record = build_encounter_record(correlation_id, trace_store)
    assert record.eval_outcome is None

    populated = build_encounter_record(
        correlation_id, trace_store, eval_outcome=EvalOutcome(verdict="verified", score=0.95)
    )
    assert populated.eval_outcome == EvalOutcome(verdict="verified", score=0.95)


def test_no_phi_in_encounter_record_or_logs(trace_store: TraceStore, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    correlation_id, ingestion_result, reranked_chunks = _run_simulated_encounter(
        trace_store, tmp_path, patient_id=1
    )
    # Also drive a sub-task that carries the PHI marker directly, proving the
    # marker never reaches the record even when a worker's own input field
    # (not just patient_id) carries it.
    ollama = _FakeVlmOllama([LabPageExtraction(rows=_PAGE_1_ROWS)])
    store = LocalIngestionStore(tmp_path)
    supervisor = Supervisor(
        intake_worker=IntakeExtractorWorker(ollama_client=ollama, document_store=store, fact_store=store),
        evidence_worker=EvidenceRetrieverWorker(
            retriever=build_retriever_from_corpus(),
            reranker=Reranker(RecordedRerankScorer(load_recorded_reranker_scores(RERANKER_SCORES_PATH))),
        ),
        trace_store=trace_store,
    )
    with pytest.raises(Exception):
        # RetrieveSubTask.query carrying the marker will fail recorded-scorer
        # lookup (no recorded score for this literal query) -- that's fine,
        # we only care that nothing about it leaks into logs/record before
        # or during the failure.
        with correlation_scope(correlation_id):
            supervisor.handle(RetrieveSubTask(query=_PHI_MARKER, k=1))

    record = build_encounter_record(correlation_id, trace_store)

    assert _PHI_MARKER not in repr(record)
    assert _PHI_MARKER not in str(record)
    for field_value in (record.steps, record.tokens_by_model, record.retrieval_top_scores):
        assert _PHI_MARKER not in str(field_value)

    for log_record in caplog.records:
        assert _PHI_MARKER not in log_record.getMessage()
        for value in vars(log_record).values():
            assert _PHI_MARKER not in str(value)

    from pathlib import Path

    raw_bytes = Path(trace_store.db_path).read_bytes()
    assert _PHI_MARKER.encode() not in raw_bytes

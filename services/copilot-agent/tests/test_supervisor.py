"""Red-first tests for P3.5's supervisor/worker orchestration
(`docs/W2_ARCHITECTURE.md` "Orchestration", `app/supervisor.py`).

Extends Phase 1's planner loop into a hand-rolled supervisor coordinating
two workers -- ``intake-extractor`` (wraps ``app.ingestion
.attach_and_extract``) and ``evidence-retriever`` (wraps
``app.reranking.retrieve_and_rerank``) -- with explicit, logged handoffs and
worker spans parented under the supervisor's own span (``app.correlation
.span_scope``, extended here).

Everything here is hermetic: worker-level orchestration tests (a)-(d) use
scripted ``Worker`` doubles so the supervisor's own routing/tracing/logging
logic is exercised in isolation from the real ingestion/retrieval pipelines
(those are pinned separately in ``tests/test_ingestion.py`` /
``tests/test_reranking.py``). The wrapper tests at the bottom of this file
prove ``IntakeExtractorWorker``/``EvidenceRetrieverWorker`` correctly
delegate to (and preserve citations from) the real P3.1/P3.4 capabilities,
using the same recorded/fixture doubles those modules' own tests use --
never a live Ollama call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.correlation import correlation_scope
from app.ingestion import LocalIngestionStore, attach_and_extract
from app.reranking import RERANKER_SCORES_PATH, RecordedRerankScorer, Reranker, retrieve_and_rerank
from app.retrieval import CORPUS_DIR, HybridRetriever, build_retriever_from_corpus, recorded_query_vector
from app.schemas.ingestion import ExtractedLabRow, LabPageExtraction
from app.supervisor import (
    EvidenceRetrieverWorker,
    IngestSubTask,
    IntakeExtractorWorker,
    RetrieveSubTask,
    Supervisor,
    SupervisorResult,
)
from scripts.retrieval_golden_queries import GOLDEN_QUERIES
from tests.test_ingestion import _FakeVlmOllama, _FIXTURE_PATH

_PHI_MARKER = "Warfarin 5mg — patient allergic to penicillin, 123 Main St"


@dataclass
class _FakeWorker:
    """Scripted ``Worker`` double: returns a canned payload, or raises, and
    records every sub-task it was invoked with."""

    name: str
    payload: Any = None
    error: Exception | None = None
    calls: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def run(self, sub_task: Any) -> Any:
        assert self.calls is not None
        self.calls.append(sub_task)
        if self.error is not None:
            raise self.error
        return self.payload


def _handoff_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "app.supervisor"]


# --- (a): evidence-retrieval routing, span parenting, shared correlation id


def test_evidence_request_routes_to_evidence_retriever_with_parented_span(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    intake = _FakeWorker(name="intake-extractor")
    retriever = _FakeWorker(name="evidence-retriever", payload=["chunk-payload"])
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever)
    sub_task = RetrieveSubTask(query="What is the metformin starting dose?", k=3)

    with correlation_scope("corr-evidence-1") as correlation_id:
        result = supervisor.handle(sub_task)

    assert isinstance(result, SupervisorResult)
    assert result.worker == "evidence-retriever"
    assert result.payload == ["chunk-payload"]
    assert retriever.calls == [sub_task]
    assert intake.calls == []  # the other worker must never be invoked

    records = _handoff_records(caplog)
    assert records, "supervisor must log handoff events"
    assert all(r.correlation_id == correlation_id for r in records)

    handoff_events = {r.event: r for r in records if hasattr(r, "event")}
    assert "handoff_start" in handoff_events and "handoff_result" in handoff_events
    start = handoff_events["handoff_start"]
    completed = handoff_events["handoff_result"]
    assert start.worker == "evidence-retriever"
    assert start.span_id == completed.span_id
    # The worker span's parent MUST be the supervisor's own (root) span --
    # never None, never equal to its own span id.
    assert start.parent_span_id is not None
    assert start.parent_span_id != start.span_id


# --- (b): ingestion routing


def test_ingestion_request_routes_to_intake_extractor(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    intake = _FakeWorker(name="intake-extractor", payload="ingestion-result")
    retriever = _FakeWorker(name="evidence-retriever")
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever)
    sub_task = IngestSubTask(patient_id=42, file_path="lab.pdf", doc_type="lab_pdf")

    result = supervisor.handle(sub_task)

    assert result.worker == "intake-extractor"
    assert result.payload == "ingestion-result"
    assert intake.calls == [sub_task]
    assert retriever.calls == []

    records = _handoff_records(caplog)
    assert any(getattr(r, "worker", None) == "intake-extractor" for r in records)


# --- (c): handoff logs never carry PHI


def test_handoff_logs_never_carry_patient_values(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    intake = _FakeWorker(name="intake-extractor")
    retriever = _FakeWorker(name="evidence-retriever", payload=[_PHI_MARKER])
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever)
    # The sub-task itself carries a patient-shaped value in its query text.
    sub_task = RetrieveSubTask(query=_PHI_MARKER, k=1)

    supervisor.handle(sub_task)

    for record in caplog.records:
        assert _PHI_MARKER not in record.getMessage()
        for value in vars(record).values():
            assert _PHI_MARKER not in str(value)


# --- (d): worker failure is surfaced honestly, never swallowed


def test_worker_failure_propagates_and_is_logged(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)

    class _Boom(Exception):
        pass

    intake = _FakeWorker(name="intake-extractor")
    retriever = _FakeWorker(name="evidence-retriever", error=_Boom("ollama unreachable"))
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever)
    sub_task = RetrieveSubTask(query="anything", k=1)

    with pytest.raises(_Boom):
        supervisor.handle(sub_task)

    records = _handoff_records(caplog)
    failed = [r for r in records if getattr(r, "event", None) == "handoff_failed"]
    assert failed, "a failed handoff must be logged, not silently swallowed"
    assert failed[0].worker == "evidence-retriever"
    assert failed[0].error_type == "_Boom"


# --- worker wrapper tests: real capability delegation, citations preserved


def test_intake_extractor_worker_delegates_and_preserves_citations(tmp_path: Path):
    rows = [ExtractedLabRow(test="Hemoglobin A1c", value="5.4", unit="%", reference_range="4.0-5.6", collection_date="2026-06-01", abnormal_flag="N")]
    ollama = _FakeVlmOllama(results=[LabPageExtraction(rows=rows)])
    store = LocalIngestionStore(tmp_path)
    worker = IntakeExtractorWorker(ollama_client=ollama, document_store=store, fact_store=store)
    sub_task = IngestSubTask(patient_id=7, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf")

    result = worker.run(sub_task)

    assert worker.name == "intake-extractor"
    assert len(result.facts) >= 1
    assert result.facts[0].citation.source_type == "lab_pdf"
    assert result.facts[0].citation.quote_or_value  # citation text survives


def test_evidence_retriever_worker_delegates_and_preserves_citations():
    retriever = build_retriever_from_corpus()
    recorded = RecordedRerankScorer(
        __import__("app.reranking", fromlist=["load_recorded_reranker_scores"]).load_recorded_reranker_scores(
            RERANKER_SCORES_PATH
        )
    )
    reranker = Reranker(recorded)
    worker = EvidenceRetrieverWorker(retriever=retriever, reranker=reranker)
    query, expected_chunk_id = GOLDEN_QUERIES[0]
    sub_task = RetrieveSubTask(query=query, k=3, query_vector=recorded_query_vector(query))

    result = worker.run(sub_task)

    assert worker.name == "evidence-retriever"
    assert result
    assert result[0].chunk_id == expected_chunk_id
    # Citation-bearing fields survive end to end.
    assert result[0].doc_id
    assert result[0].section
    assert result[0].text

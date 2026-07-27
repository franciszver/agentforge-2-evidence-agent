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
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.correlation import correlation_scope
from app.ingestion import IngestionError, LocalIngestionStore, attach_and_extract
from app.ollama_client import OllamaError
from app.reranking import RERANKER_SCORES_PATH, RecordedRerankScorer, Reranker, retrieve_and_rerank
from app.retrieval import CORPUS_DIR, HybridRetriever, build_retriever_from_corpus, recorded_query_vector
from app.schemas.ingestion import LabPageExtraction
from app.supervisor import (
    EvidenceRetrieverWorker,
    IngestSubTask,
    IntakeExtractorWorker,
    RetrieveSubTask,
    Supervisor,
    SupervisorResult,
    VisionModelMisconfiguredError,
)
from app.trace_store import SpanStatus, SpanType, TraceStore
from scripts.retrieval_golden_queries import GOLDEN_QUERIES
from tests.test_ingestion import _FakeVlmOllama, _FIXTURE_PATH, _PAGE_1_ROWS, _PAGE_2_ROWS

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


# --- (a)/(b): routing -- right worker called, wrong worker untouched, result
# plumbed through -- parametrized over both (sub_task, worker_name) pairs.


@pytest.mark.parametrize(
    "sub_task,expected_worker_name",
    [
        pytest.param(RetrieveSubTask(query="What is the metformin starting dose?", k=3), "evidence-retriever", id="evidence"),
        pytest.param(IngestSubTask(patient_id=42, file_path="lab.pdf", doc_type="lab_pdf"), "intake-extractor", id="ingestion"),
    ],
)
def test_routes_to_the_expected_worker(sub_task: RetrieveSubTask | IngestSubTask, expected_worker_name: str):
    intake = _FakeWorker(name="intake-extractor", payload="ingestion-result")
    retriever = _FakeWorker(name="evidence-retriever", payload=["chunk-payload"])
    workers = {"evidence-retriever": retriever, "intake-extractor": intake}
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever)

    result = supervisor.handle(sub_task)

    assert isinstance(result, SupervisorResult)
    assert result.worker == expected_worker_name
    assert result.payload == workers[expected_worker_name].payload
    assert workers[expected_worker_name].calls == [sub_task]
    other = intake if expected_worker_name == "evidence-retriever" else retriever
    assert other.calls == []  # the other worker must never be invoked


# --- (a): evidence-retrieval span parenting, shared correlation id


def test_evidence_request_has_a_worker_span_parented_under_the_supervisor_span(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    intake = _FakeWorker(name="intake-extractor")
    retriever = _FakeWorker(name="evidence-retriever", payload=["chunk-payload"])
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever)
    sub_task = RetrieveSubTask(query="What is the metformin starting dose?", k=3)

    with correlation_scope("corr-evidence-1") as correlation_id:
        supervisor.handle(sub_task)

    records = _handoff_records(caplog)
    assert records, "supervisor must log handoff events"
    assert all(r.correlation_id == correlation_id for r in records)

    received = [r for r in records if r.getMessage() == "supervisor_received"]
    assert len(received) == 1, "exactly one supervisor_received event per handle() call"
    supervisor_span_id = received[0].span_id

    handoff_events = {r.event: r for r in records if hasattr(r, "event")}
    assert "handoff_start" in handoff_events and "handoff_result" in handoff_events
    start = handoff_events["handoff_start"]
    completed = handoff_events["handoff_result"]
    assert start.worker == "evidence-retriever"
    assert start.span_id == completed.span_id
    # The worker span's parent MUST be the supervisor's OWN span id exactly
    # -- not merely non-None/not-self, which a wrong-but-non-None parent
    # would also satisfy.
    assert start.parent_span_id == supervisor_span_id
    assert start.parent_span_id != start.span_id


# --- unregistered sub-task type: fail loudly, never silently default


@dataclass(frozen=True)
class _UnregisteredSubTask:
    """A well-formed sub-task carrying patient-shaped content but registered
    with no worker -- ``Supervisor`` must reject it outright, never fall
    back to some default worker, and must not have logged anything about
    it before raising (see module docstring's no-PHI discipline)."""

    note: str


def test_unregistered_sub_task_type_raises_value_error_without_logging(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    intake = _FakeWorker(name="intake-extractor")
    retriever = _FakeWorker(name="evidence-retriever")
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever)
    sub_task = _UnregisteredSubTask(note=_PHI_MARKER)

    with pytest.raises(ValueError, match="_UnregisteredSubTask"):
        supervisor.handle(sub_task)

    assert intake.calls == []
    assert retriever.calls == []
    # Rejected before any handoff -- no span opened, nothing logged at all.
    assert _handoff_records(caplog) == []
    for record in caplog.records:
        assert _PHI_MARKER not in record.getMessage()
        for value in vars(record).values():
            assert _PHI_MARKER not in str(value)


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


# --- issue #206: extraction spans recorded on ALL THREE ingestion outcomes


@pytest.fixture
def trace_store(tmp_path: Path) -> TraceStore:
    return TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=secrets.token_hex(16))


def test_successful_ingestion_records_an_extraction_span(tmp_path: Path, trace_store: TraceStore) -> None:
    ollama = _FakeVlmOllama([LabPageExtraction(rows=_PAGE_1_ROWS), LabPageExtraction(rows=_PAGE_2_ROWS)])
    store = LocalIngestionStore(tmp_path / "ingestion")
    intake = IntakeExtractorWorker(ollama_client=ollama, document_store=store, fact_store=store)
    retriever = _FakeWorker(name="evidence-retriever")
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever, trace_store=trace_store)

    with correlation_scope() as correlation_id:
        supervisor.handle(IngestSubTask(patient_id=1, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf"))

    extraction_spans = [s for s in trace_store.get_spans(correlation_id) if s.span_type == SpanType.EXTRACTION]
    assert len(extraction_spans) == 1
    span = extraction_spans[0]
    assert span.status == SpanStatus.OK
    assert span.pages_total == 2
    assert span.pages_failed == 0


def test_partial_failure_ingestion_records_an_extraction_span_with_failed_pages(
    tmp_path: Path, trace_store: TraceStore
) -> None:
    class _MixedVlm:
        model = "qwen2.5vl:7b"  # vision-capable, so IntakeExtractorWorker's guard passes

        def __init__(self) -> None:
            self.calls = 0

        def extract(
            self, prompt_or_messages: Any, schema: type, *, options: Any = None, images: list[str] | None = None
        ) -> Any:
            self.calls += 1
            if self.calls == 1:
                return LabPageExtraction(rows=_PAGE_1_ROWS)
            raise OllamaError("scripted failure on page 2")

    store = LocalIngestionStore(tmp_path / "ingestion")
    intake = IntakeExtractorWorker(ollama_client=_MixedVlm(), document_store=store, fact_store=store)
    retriever = _FakeWorker(name="evidence-retriever")
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever, trace_store=trace_store)

    with correlation_scope() as correlation_id:
        supervisor.handle(IngestSubTask(patient_id=1, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf"))

    span = next(s for s in trace_store.get_spans(correlation_id) if s.span_type == SpanType.EXTRACTION)
    assert span.status == SpanStatus.OK  # the handoff itself completed without raising
    assert span.pages_total == 2
    assert span.pages_failed == 1


def test_total_failure_ingestion_records_an_extraction_span_then_propagates(
    tmp_path: Path, trace_store: TraceStore
) -> None:
    failing_vlm = _FakeVlmOllama(error=True)
    store = LocalIngestionStore(tmp_path / "ingestion")
    intake = IntakeExtractorWorker(ollama_client=failing_vlm, document_store=store, fact_store=store)
    retriever = _FakeWorker(name="evidence-retriever")
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever, trace_store=trace_store)

    with correlation_scope() as correlation_id:
        with pytest.raises(IngestionError):
            supervisor.handle(IngestSubTask(patient_id=1, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf"))

    span = next(s for s in trace_store.get_spans(correlation_id) if s.span_type == SpanType.EXTRACTION)
    assert span.status == SpanStatus.FAIL
    assert span.pages_total == 2
    assert span.pages_failed == 2
    assert span.error_category == "IngestionError"


def test_retrieve_sub_task_never_records_an_extraction_span(trace_store: TraceStore) -> None:
    intake = _FakeWorker(name="intake-extractor")
    retriever = _FakeWorker(name="evidence-retriever", payload=["chunk"])
    supervisor = Supervisor(intake_worker=intake, evidence_worker=retriever, trace_store=trace_store)

    with correlation_scope() as correlation_id:
        supervisor.handle(RetrieveSubTask(query="anything", k=1))

    extraction_spans = [s for s in trace_store.get_spans(correlation_id) if s.span_type == SpanType.EXTRACTION]
    assert extraction_spans == []


# --- worker wrapper tests: real capability delegation, citations preserved


def test_intake_extractor_worker_delegates_and_preserves_citations(tmp_path: Path):
    ollama = _FakeVlmOllama([LabPageExtraction(rows=_PAGE_1_ROWS), LabPageExtraction(rows=_PAGE_2_ROWS)])
    store = LocalIngestionStore(tmp_path)
    worker = IntakeExtractorWorker(ollama_client=ollama, document_store=store, fact_store=store)
    sub_task = IngestSubTask(patient_id=7, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf")

    result = worker.run(sub_task)

    assert worker.name == "intake-extractor"
    assert len(result.facts) >= 1
    assert result.facts[0].citation.source_type == "lab_pdf"
    assert result.facts[0].citation.quote_or_value  # citation text survives


def test_intake_extractor_worker_refuses_to_run_on_a_text_only_model(tmp_path: Path):
    """Issue #204: fail-closed. A worker wired to a non-vision-capable model
    (the default ``qwen3:4b`` bug this issue fixes) must refuse to run
    BEFORE any page reaches the model -- zero extract() calls -- rather
    than silently sending it an image it cannot read."""
    ollama = _FakeVlmOllama(
        [LabPageExtraction(rows=_PAGE_1_ROWS), LabPageExtraction(rows=_PAGE_2_ROWS)], model="qwen3:4b"
    )
    store = LocalIngestionStore(tmp_path)
    worker = IntakeExtractorWorker(ollama_client=ollama, document_store=store, fact_store=store)
    sub_task = IngestSubTask(patient_id=7, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf")

    with pytest.raises(VisionModelMisconfiguredError):
        worker.run(sub_task)

    assert ollama.extract_calls == [], "exposure count must be 0 -- no page image may reach a misconfigured model"


def test_vision_model_misconfigured_error_names_the_model_and_the_escape_hatch(tmp_path: Path):
    """Gate-1 finding on #206: the error must be actionable without reading
    source -- it must name the configured model, that the check is
    name-based (so it may misjudge a valid VLM), and the exact setting to
    disable it."""
    ollama = _FakeVlmOllama(
        [LabPageExtraction(rows=_PAGE_1_ROWS), LabPageExtraction(rows=_PAGE_2_ROWS)], model="qwen3:4b"
    )
    store = LocalIngestionStore(tmp_path)
    worker = IntakeExtractorWorker(ollama_client=ollama, document_store=store, fact_store=store)
    sub_task = IngestSubTask(patient_id=7, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf")

    with pytest.raises(VisionModelMisconfiguredError) as excinfo:
        worker.run(sub_task)

    message = str(excinfo.value)
    assert "qwen3:4b" in message
    assert "name-based" in message
    assert "copilot_vision_model_capability_check" in message


def test_intake_extractor_worker_accepts_an_unrecognized_model_when_the_check_is_disabled(tmp_path: Path):
    """Gate-1 finding on #206: the escape hatch. A digest-pinned reference
    (no human-readable segment) or an operator's custom re-tag both fail
    the name-based heuristic even when the underlying model is genuinely
    vision-capable. With the check disabled, the worker must proceed
    rather than raise."""
    ollama = _FakeVlmOllama(
        [LabPageExtraction(rows=_PAGE_1_ROWS), LabPageExtraction(rows=_PAGE_2_ROWS)],
        model="clinic-doc-reader:v3",
    )
    store = LocalIngestionStore(tmp_path)
    worker = IntakeExtractorWorker(
        ollama_client=ollama, document_store=store, fact_store=store, vision_model_capability_check=False
    )
    sub_task = IngestSubTask(patient_id=7, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf")

    result = worker.run(sub_task)

    assert len(result.facts) >= 1
    assert ollama.extract_calls, "the escape hatch must let ingestion actually reach the model"


class _MinimalVlm:
    """The narrowest double satisfying ``app.ingestion._Extractor`` --
    ``extract`` only, deliberately with NO ``.model`` attribute. Stands in
    for an operator's custom VLM adapter that doesn't happen to expose a
    ``model`` attribute at all (``attach_and_extract``'s injected-client
    contract never required one). Reproduces the MINOR-1 gate-3 finding on
    #204/#206: with the capability check disabled, ``IntakeExtractorWorker
    .run()`` must never touch ``.model`` -- reading it before checking
    ``self._vision_model_capability_check`` raised ``AttributeError`` on
    exactly this double."""

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.extract_calls: list[tuple[list[dict[str, Any]], type, list[str] | None]] = []

    def extract(
        self, prompt_or_messages: Any, schema: type, *, options: Any = None, images: list[str] | None = None
    ) -> Any:
        self.extract_calls.append((prompt_or_messages, schema, images))
        return self._results.pop(0)


def test_intake_extractor_worker_never_reads_dot_model_when_check_is_disabled(tmp_path: Path):
    """MINOR-1 (gate-3 finding on #204/#206): a VLM double exposing ONLY
    ``extract`` (no ``.model``) must be able to ingest when the check is
    disabled -- proving the ``.model`` read happens inside the ``if
    self._vision_model_capability_check:`` branch, not before it."""
    ollama = _MinimalVlm([LabPageExtraction(rows=_PAGE_1_ROWS), LabPageExtraction(rows=_PAGE_2_ROWS)])
    assert not hasattr(ollama, "model")
    store = LocalIngestionStore(tmp_path)
    worker = IntakeExtractorWorker(
        ollama_client=ollama, document_store=store, fact_store=store, vision_model_capability_check=False
    )
    sub_task = IngestSubTask(patient_id=7, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf")

    result = worker.run(sub_task)

    assert len(result.facts) >= 1
    assert ollama.extract_calls, "the escape hatch must let ingestion actually reach the model"


def test_intake_extractor_worker_still_fails_closed_on_a_text_model_when_check_left_enabled(tmp_path: Path):
    """Non-regression, stated explicitly alongside the escape-hatch test
    above: leaving the (default-True) check enabled must still reject a
    genuinely non-vision model -- the override is opt-in, not a change to
    the default safety behavior."""
    ollama = _FakeVlmOllama(
        [LabPageExtraction(rows=_PAGE_1_ROWS), LabPageExtraction(rows=_PAGE_2_ROWS)], model="qwen3:4b"
    )
    store = LocalIngestionStore(tmp_path)
    worker = IntakeExtractorWorker(
        ollama_client=ollama, document_store=store, fact_store=store, vision_model_capability_check=True
    )
    sub_task = IngestSubTask(patient_id=7, file_path=str(_FIXTURE_PATH), doc_type="lab_pdf")

    with pytest.raises(VisionModelMisconfiguredError):
        worker.run(sub_task)

    assert ollama.extract_calls == []


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

"""Supervisor/worker orchestration (P3.5, `docs/W2_ARCHITECTURE.md`
"Orchestration").

Extends Phase 1's hand-rolled planner loop (`app.planner`) into a small
**supervisor** that routes a request to exactly one of two **workers** --
``intake-extractor`` (wraps ``app.ingestion.attach_and_extract``) and
``evidence-retriever`` (wraps ``app.reranking.retrieve_and_rerank``) -- and
logs the handoff both ways. This is a custom hand-rolled loop, not a
graph-orchestration framework: see the architecture doc's "Why not
LangGraph" section for the owner decision this implements.

**Routing.** Deliberately structural, not model-driven: the caller already
knows which capability it needs (an ingestion request vs. a retrieval
question), so routing is a plain type dispatch on the sub-task
(``IngestSubTask`` -> intake-extractor, ``RetrieveSubTask`` ->
evidence-retriever) rather than an LLM decision -- no framework, no
speculative multi-worker sequencing beyond what these two capabilities need.

**Tracing.** Every ``Supervisor.handle`` call opens its own span
(``app.correlation.span_scope``) for the whole request, and each worker
invocation opens a NESTED span inside it -- so the worker span's
``parent_span_id`` is always the supervisor's span id, with no extra
plumbing (the parent comes from whatever span is already open in the
current context). One correlation id (unchanged, still per chat/request
invocation) plus this span parent/child pointer is what lets a full
supervisor->worker->supervisor chain be reconstructed from logs alone.

**Handoff logging -- PSR-3-style structured context, NEVER raw PHI.**
Every ``supervisor_handoff`` log line carries only: the worker name, the
span id and its parent, the sub-task's TYPE (``type(sub_task).__name__``,
e.g. ``"RetrieveSubTask"`` -- never its field values, which may carry a
clinician's question text or a patient id), a handoff-lifecycle ``event``
(``handoff_start`` / ``handoff_result`` / ``handoff_failed``), and (for
failures) the raised exception's TYPE name -- never ``str(exc)``, which
could echo request content back into the log. Worker results are likewise
never logged by value -- only that a handoff completed and how long it
took.

**Worker failure is surfaced, not swallowed.** ``Worker.run`` is expected to
raise on failure (mirroring ``app.ingestion``'s honest partial-failure
discipline -- a failure is a first-class, visible outcome, not silently
absorbed into an empty result). The supervisor logs a ``handoff_failed``
event and then RE-RAISES the same exception -- it never catches-and-returns
a degraded/empty result in its place.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.correlation import SpanContext, span_scope
from app.ingestion import DocumentStore, FactStore, IngestionResult, attach_and_extract
from app.reranking import Reranker, retrieve_and_rerank
from app.retrieval import HybridRetriever
from app.schemas.reranking import RerankedChunk
from app.trace_store import TraceStore

_logger = logging.getLogger(__name__)


def _handoff_log_context(worker: Worker, span: SpanContext, sub_task: SubTask) -> dict[str, str | None]:
    """The one place that knows a handoff log line's shape: worker name,
    the span's parent, and the sub-task's TYPE (never its field values --
    see ``_log_handoff``). Both ``Supervisor.handle`` (the top-level
    ``supervisor_received`` line) and ``Supervisor._dispatch`` (every
    ``supervisor_handoff`` line) build their ``extra`` dict from this, so
    neither hand-assembles the same three keys independently."""
    return {
        "worker": worker.name,
        "parent_span_id": span.parent_span_id,
        "sub_task_type": type(sub_task).__name__,
    }


def _log_handoff(
    event: str,
    log_context: dict[str, str | None],
    *,
    duration_ms: float | None = None,
    error_type: str | None = None,
) -> None:
    """Emit one ``supervisor_handoff`` log line. Only ever accepts closed-set,
    non-PHI fields -- ``log_context`` (worker name, span parent, sub-task
    TYPE) plus a timing float and an exception's TYPE name -- never a raw
    exception message or any sub-task/worker-result value. Centralizing the
    handoff log shape here (rather than each call site building its own
    ``extra`` dict) is what makes that no-PHI guarantee structural instead of
    a convention every call site has to remember to uphold.
    """
    extra: dict[str, object] = {**log_context, "event": event}
    if duration_ms is not None:
        extra["duration_ms"] = duration_ms
    if error_type is not None:
        extra["error_type"] = error_type
    level = logging.ERROR if event == "handoff_failed" else logging.INFO
    _logger.log(level, "supervisor_handoff", extra=extra)


@dataclass(frozen=True)
class IngestSubTask:
    """A request for the intake-extractor worker: extract structured facts
    from a source document already on disk at ``file_path``."""

    patient_id: int
    file_path: str
    doc_type: Literal["lab_pdf", "intake_form"]


@dataclass(frozen=True)
class RetrieveSubTask:
    """A request for the evidence-retriever worker: hybrid-retrieve and
    rerank the guideline corpus for ``query``, returning the top ``k``."""

    query: str
    k: int
    query_vector: Sequence[float] | None = None


SubTask = IngestSubTask | RetrieveSubTask


class Worker(Protocol):
    """Uniform worker interface the supervisor dispatches through: a stable
    ``name`` (used in routing/logging) and a single ``run`` entry point that
    either returns a typed result or raises -- see module docstring's
    "Worker failure is surfaced" note."""

    name: str

    def run(self, sub_task: Any) -> Any: ...


class IntakeExtractorWorker:
    """Thin wrapper around ``app.ingestion.attach_and_extract`` (P3.1/P3.2)."""

    name = "intake-extractor"

    def __init__(self, *, ollama_client: Any, document_store: DocumentStore, fact_store: FactStore) -> None:
        self._ollama_client = ollama_client
        self._document_store = document_store
        self._fact_store = fact_store

    def run(self, sub_task: IngestSubTask) -> IngestionResult:
        return attach_and_extract(
            sub_task.patient_id,
            sub_task.file_path,
            sub_task.doc_type,
            ollama_client=self._ollama_client,
            document_store=self._document_store,
            fact_store=self._fact_store,
        )


class EvidenceRetrieverWorker:
    """Thin wrapper around ``app.reranking.retrieve_and_rerank`` (P3.3/P3.4)."""

    name = "evidence-retriever"

    def __init__(self, *, retriever: HybridRetriever, reranker: Reranker) -> None:
        self._retriever = retriever
        self._reranker = reranker

    def run(self, sub_task: RetrieveSubTask) -> list[RerankedChunk]:
        return retrieve_and_rerank(
            self._retriever,
            self._reranker,
            sub_task.query,
            sub_task.k,
            query_vector=sub_task.query_vector,
        )


@dataclass(frozen=True)
class SupervisorResult:
    """The supervisor's assembled response to one ``handle`` call: which
    worker answered, and its (citation-bearing, untouched) payload."""

    worker: str
    payload: Any = None


class Supervisor:
    """Routes one sub-task to the worker that owns its capability, tracing
    and logging the handoff both ways (see module docstring)."""

    def __init__(
        self,
        *,
        intake_worker: Worker,
        evidence_worker: Worker,
        trace_store: TraceStore | None = None,
    ) -> None:
        self._workers: dict[type, Worker] = {
            IngestSubTask: intake_worker,
            RetrieveSubTask: evidence_worker,
        }
        # P3.8: optional -- when supplied, every handoff also gets a durable
        # ``worker`` span (app.trace_store.TraceStore.record_worker_span),
        # feeding app.encounter_observability's per-encounter record. ``None``
        # (the default) keeps every existing caller/test -- which construct a
        # ``Supervisor`` with no trace store at all -- byte-identical.
        self._trace_store = trace_store

    def handle(self, sub_task: SubTask) -> SupervisorResult:
        """Route ``sub_task`` to its owning worker and return its result.

        Opens the supervisor's own span for the whole call (so the worker's
        span nests under it -- see module docstring). Propagates the
        worker's exception unchanged if it fails.
        """
        worker = self._select_worker(sub_task)
        with span_scope() as supervisor_span:
            _logger.info("supervisor_received", extra=_handoff_log_context(worker, supervisor_span, sub_task))
            payload = self._dispatch(worker, sub_task)
        return SupervisorResult(worker=worker.name, payload=payload)

    def _select_worker(self, sub_task: SubTask) -> Worker:
        worker = self._workers.get(type(sub_task))
        if worker is None:
            raise ValueError(f"No worker registered for sub-task type {type(sub_task).__name__}")
        return worker

    def _dispatch(self, worker: Worker, sub_task: SubTask) -> Any:
        with span_scope() as worker_span:
            log_context = _handoff_log_context(worker, worker_span, sub_task)
            _log_handoff("handoff_start", log_context)
            start_ts = time.time()
            error_type: str | None = None
            try:
                payload = worker.run(sub_task)
            except Exception as exc:
                error_type = type(exc).__name__
                raise
            finally:
                end_ts = time.time()
                duration_ms = (end_ts - start_ts) * 1000
                if error_type is None:
                    _log_handoff("handoff_result", log_context, duration_ms=duration_ms)
                else:
                    _log_handoff("handoff_failed", log_context, duration_ms=duration_ms, error_type=error_type)
                self._record_worker_span(
                    worker=worker,
                    worker_span=worker_span,
                    sub_task=sub_task,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    error_type=error_type,
                )
            return payload

    def _record_worker_span(
        self,
        *,
        worker: Worker,
        worker_span: SpanContext,
        sub_task: SubTask,
        start_ts: float,
        end_ts: float,
        error_type: str | None,
    ) -> None:
        """Persist the just-completed handoff as a durable ``worker`` span,
        best-effort (P3.8) -- a trace-store write failure must never break a
        handoff that otherwise succeeded, mirroring ``app.chat``'s
        ``_record_span_best_effort`` for tool/llm spans. No-op when no
        ``trace_store`` was injected (the default)."""
        if self._trace_store is None:
            return
        try:
            self._trace_store.record_worker_span(
                correlation_id=worker_span.correlation_id,
                start_ts=start_ts,
                end_ts=end_ts,
                ok=error_type is None,
                worker_name=worker.name,
                sub_task_type=type(sub_task).__name__,
                span_id=worker_span.span_id,
                parent_span_id=worker_span.parent_span_id,
                error_category=error_type,
            )
        except Exception:
            _logger.warning("worker span write failed; continuing without it", exc_info=True)

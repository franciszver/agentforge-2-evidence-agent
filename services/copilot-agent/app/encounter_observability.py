"""Per-encounter observability record (P3.8, `docs/W2_ARCHITECTURE.md`
"Observability" row).

Extends Phase 1's correlation-id/trace-store observability (`app.correlation`,
`app.trace_store`, `app.dashboard*`) rather than forking a parallel system:
this module is a pure READER that aggregates one correlation id's already-
recorded spans -- `request`/`tool`/`llm`/`worker`/`verification`, the last two
added by P3.5/P3.8 (`app.supervisor`, `app.trace_store.SpanType.WORKER`) --
into one `EncounterRecord` describing the full Phase-2 pipeline for a single
encounter: the ordered tool/worker step sequence (with the P3.5 span tree),
per-step latency, aggregated token usage, an illustrative cost estimate,
a retrieval-hit summary, an extraction-confidence proxy, and a typed (empty
until P3G) eval-outcome slot.

**NO PHI -- structural, not a convention to remember.** Every field this
module produces comes from data that is non-PHI BEFORE it reaches this
module: `app.trace_store`'s columns (names, counts, timings, hashed args --
see its own module docstring), and, for retrieval/extraction, plain floats
and counts derived here from `RerankedChunk.rerank_score` and
`IngestionResult.pages_total`/`failed_pages` -- never chunk text, extracted
field values, patient ids, or query/answer text. `build_encounter_record`
never accepts a raw tool-args mapping, a question, an answer, or a fact's
value -- there is no parameter shape it could leak PHI through even if a
caller tried.

**Cost estimate -- what the number means.** This agent's own Ollama model
runs on local hardware at no real per-token bill, so `cost_estimate_usd` is
NOT an actual charge. It is `total_tokens / 1000 * cost_per_1k_tokens_usd`,
where `cost_per_1k_tokens_usd` is a caller-supplied, illustrative rate (e.g.
"what would this have cost against a metered cloud API of comparable
capability"). `None` when there were no tokens to estimate from (an
encounter with no LLM calls), never a fabricated `0.0`.

**Extraction confidence -- an honest proxy, not a model-reported score.**
`app.ingestion`'s VLM extraction schemas have no confidence field at all --
per their own "No-fabrication contract", a field is either read or reported
`None`; there is nothing to fabricate a number from at the per-field level,
and `DocumentFact`'s two variants (`LabResultFact`/`IntakeFormFact`) do not
even share a common value-bearing field to average over. The proxy used
here operates one level up, on `IngestionResult.pages_total`/`failed_pages`
(uniform across both doc types): the fraction of the document's pages that
were successfully extracted at all, vs. failed outright (VLM call error --
see `IngestionResult`'s own docstring on partial extraction). This measures
extraction COVERAGE, not correctness -- a page that succeeded but whose
rows are mostly `None` counts as fully "confident" here. Callers must not
present it as an accuracy score.

**Eval outcome -- a typed slot, not a value.** `eval_outcome` is `None` by
default and only ever set to a caller-supplied `EvalOutcome` -- this module
never computes or guesses one. Wiring the eval pipeline (P3G) to actually
populate it is deferred; this is the seam it will use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.ingestion import IngestionResult
from app.schemas.reranking import RerankedChunk
from app.trace_store import Span, SpanType, TraceStore

StepKind = Literal["tool", "llm", "worker", "verification"]

# Which recorded span types become a step in the encounter's ordered
# sequence, and the step "kind" each maps to. ``request``/``feedback`` spans
# are whole-invocation/post-hoc bookkeeping, not pipeline steps, so they are
# deliberately excluded from the sequence.
_STEP_SPAN_TYPES: dict[SpanType, StepKind] = {
    SpanType.TOOL: "tool",
    SpanType.LLM: "llm",
    SpanType.WORKER: "worker",
    SpanType.VERIFICATION: "verification",
}

COST_ESTIMATE_NOTE = (
    "Illustrative what-if cost only: this agent's local Ollama model has no "
    "real per-token bill. Computed as total_tokens / 1000 * "
    "cost_per_1k_tokens_usd, using a caller-supplied cloud-equivalent rate. "
    "Not actual spend."
)

EXTRACTION_CONFIDENCE_NOTE = (
    "Coverage proxy only: the fraction of the source document's pages that "
    "were successfully extracted (vs. a VLM call failing outright on that "
    "page). NOT a correctness/accuracy score -- a successfully-extracted "
    "page whose fields mostly resolved to 'not found' still counts as fully "
    "confident here."
)


@dataclass(frozen=True)
class EncounterStep:
    """One step in the encounter's ordered pipeline sequence. Carries only
    non-PHI identifiers -- a tool/worker/model NAME, a status, and the P3.5
    span-tree pointers -- never any request/result value."""

    kind: StepKind
    name: str
    duration_ms: float
    ok: bool
    span_id: str | None
    parent_span_id: str | None


@dataclass(frozen=True)
class ModelTokenUsage:
    """Aggregated token counts for one model name across the encounter."""

    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class EvalOutcome:
    """The eval pipeline's (P3G) verdict for this encounter's answer.
    Never fabricated by this module -- see module docstring's "Eval
    outcome" section."""

    verdict: str
    score: float | None = None


@dataclass(frozen=True)
class EncounterRecord:
    """The per-encounter observability record for one correlation id."""

    correlation_id: str
    steps: list[EncounterStep]
    total_tokens_in: int
    total_tokens_out: int
    tokens_by_model: dict[str, ModelTokenUsage]
    cost_estimate_usd: float | None
    cost_estimate_note: str
    retrieval_hit_count: int | None
    retrieval_top_scores: list[float]
    extraction_confidence: float | None
    extraction_confidence_note: str
    eval_outcome: EvalOutcome | None = None


def retrieval_summary(chunks: Sequence[RerankedChunk], *, top_n: int = 3) -> tuple[int, list[float]]:
    """``(hit_count, top_scores)`` for a retrieval/rerank result: how many
    chunks were returned, and the highest ``top_n`` rerank scores (rounded,
    best first). Only counts/floats -- never chunk text, doc ids, or
    section names."""
    top_scores = sorted((chunk.rerank_score for chunk in chunks), reverse=True)[:top_n]
    return len(chunks), [round(score, 4) for score in top_scores]


def extraction_confidence_proxy(result: IngestionResult) -> float | None:
    """The page-coverage proxy described in the module docstring. ``None``
    (not a fabricated ``0.0``) when the document had no pages to score."""
    if result.pages_total <= 0:
        return None
    resolved_pages = result.pages_total - len(result.failed_pages)
    return resolved_pages / result.pages_total


def cost_estimate_usd(total_tokens: int, *, cost_per_1k_tokens_usd: float) -> float:
    """See module docstring's "Cost estimate" section for what this number
    means (and does not mean)."""
    return (total_tokens / 1000) * cost_per_1k_tokens_usd


def _step_name(span: Span) -> str:
    """The step's display name -- the one non-PHI identifier each span type
    carries (a worker/tool/model NAME), never a field value."""
    if span.span_type == SpanType.WORKER:
        return span.worker_name or "unknown-worker"
    if span.span_type == SpanType.TOOL:
        return span.tool_name or "unknown-tool"
    if span.span_type == SpanType.LLM:
        return span.model or "unknown-model"
    return span.span_type.value


def build_encounter_record(
    correlation_id: str,
    trace_store: TraceStore,
    *,
    retrieval_chunks: Sequence[RerankedChunk] | None = None,
    ingestion_result: IngestionResult | None = None,
    cost_per_1k_tokens_usd: float = 0.0,
    eval_outcome: EvalOutcome | None = None,
) -> EncounterRecord:
    """Build the per-encounter record for ``correlation_id`` from whatever
    ``trace_store`` already recorded for it (the existing P4.2 sink -- this
    reads, it never writes), plus the caller's own in-process
    retrieval/ingestion results for the retrieval-hit summary and extraction-
    confidence proxy (those are never persisted raw to the trace store, so
    they cannot be read back from it -- see each module's own docstring).
    """
    spans = trace_store.get_spans(correlation_id)

    steps: list[EncounterStep] = []
    total_tokens_in = 0
    total_tokens_out = 0
    tokens_by_model_counts: dict[str, tuple[int, int]] = {}

    for span in spans:
        step_kind = _STEP_SPAN_TYPES.get(span.span_type)
        if step_kind is not None:
            steps.append(
                EncounterStep(
                    kind=step_kind,
                    name=_step_name(span),
                    duration_ms=span.duration_ms,
                    ok=span.status.value == "ok",
                    span_id=span.span_id,
                    parent_span_id=span.parent_span_id,
                )
            )
        if span.span_type == SpanType.LLM:
            tokens_in = span.tokens_in or 0
            tokens_out = span.tokens_out or 0
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out
            if span.model is not None:
                prev_in, prev_out = tokens_by_model_counts.get(span.model, (0, 0))
                tokens_by_model_counts[span.model] = (prev_in + tokens_in, prev_out + tokens_out)

    hit_count: int | None = None
    top_scores: list[float] = []
    if retrieval_chunks is not None:
        hit_count, top_scores = retrieval_summary(retrieval_chunks)

    confidence = extraction_confidence_proxy(ingestion_result) if ingestion_result is not None else None

    total_tokens = total_tokens_in + total_tokens_out
    cost = cost_estimate_usd(total_tokens, cost_per_1k_tokens_usd=cost_per_1k_tokens_usd) if total_tokens > 0 else None

    return EncounterRecord(
        correlation_id=correlation_id,
        steps=steps,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        tokens_by_model={
            model: ModelTokenUsage(tokens_in=tokens_in, tokens_out=tokens_out)
            for model, (tokens_in, tokens_out) in tokens_by_model_counts.items()
        },
        cost_estimate_usd=cost,
        cost_estimate_note=COST_ESTIMATE_NOTE,
        retrieval_hit_count=hit_count,
        retrieval_top_scores=top_scores,
        extraction_confidence=confidence,
        extraction_confidence_note=EXTRACTION_CONFIDENCE_NOTE,
        eval_outcome=eval_outcome,
    )

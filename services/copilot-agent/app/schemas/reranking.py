"""Reranking schemas (P3.4, `app/reranking.py`, `docs/W2_ARCHITECTURE.md`
"Hybrid retrieval-augmented answering" reranker section).

``RelevanceScore`` is the schema-constrained shape ``OllamaClient.extract``
is asked to produce for one (query, chunk) pair -- see
``app.reranking.OllamaRerankScorer``. ``RerankedChunk`` extends
``RetrievedChunk`` (P3.3, `app/schemas/retrieval.py`) rather than duplicating
its fields, so every citation-bearing field (``chunk_id``, ``doc_id``,
``section``, ...) survives reranking unchanged -- ``app.reranking.Reranker``
never re-derives them from anything but the input candidate.
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import ToolSchemaModel
from app.schemas.retrieval import RetrievedChunk


class RelevanceScore(ToolSchemaModel):
    """Pointwise query-chunk relevance score, 0.0 (irrelevant) to 1.0
    (directly answers the query). Bounds are enforced by pydantic validation
    on ``OllamaClient.extract``'s parsed result -- an out-of-range value
    fails validation and triggers ``extract``'s existing retry-on-malformed-
    output behavior, the same as any other schema it constrains decoding
    to."""

    score: float = Field(ge=0.0, le=1.0)


class RerankedChunk(RetrievedChunk):
    """A ``RetrievedChunk`` plus the reranker's relevance score for the
    query it was reranked against."""

    rerank_score: float

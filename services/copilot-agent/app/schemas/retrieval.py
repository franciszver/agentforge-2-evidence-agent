"""Hybrid guideline-corpus retrieval schemas (P3.3, `docs/W2_ARCHITECTURE.md`
"Hybrid retrieval-augmented answering").

``RetrievedChunk`` is what `app.retrieval`'s sparse/dense/hybrid search
returns -- one chunk of the corpus (`corpus/README.md`) plus the score(s) it
was ranked by. Like every other schema in this package, it extends
``ToolSchemaModel`` (frozen + ``extra="forbid"``). It deliberately carries
both ``doc_id`` and ``section`` (not just a combined ``chunk_id``) because
the citation contract this ultimately feeds (`docs/W2_ARCHITECTURE.md`
"Citation Contract" -- ``field_or_chunk_id``) needs the document id and
section on their own, not just parseable out of a composite string.
"""

from __future__ import annotations

from enum import StrEnum

from app.schemas.common import ToolSchemaModel


class RetrievalMode(StrEnum):
    """Which index (or fused result) a retrieval call drew from."""

    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"


class RetrievedChunk(ToolSchemaModel):
    """One retrieved corpus chunk plus its score(s).

    ``chunk_id`` is the stable ``<doc_id>#<section-slug>`` identifier
    (`corpus/README.md` "Document format"). ``scores`` is keyed by
    ``RetrievalMode`` value (``"sparse"``, ``"dense"``, ``"hybrid"``) --
    a single chunk returned by ``retrieve_hybrid`` carries whichever of
    those modes actually contributed to it, so a caller can tell whether a
    hit came from keyword match, semantic similarity, or both.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    scores: dict[str, float]

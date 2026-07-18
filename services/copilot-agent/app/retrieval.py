"""Hybrid (sparse + dense) retrieval over the public guideline corpus (P3.3,
`docs/W2_ARCHITECTURE.md` "Hybrid retrieval-augmented answering",
`planning/PLAN.md` "Fully-local tooling choices").

Pipeline: deterministic section-based chunking of `corpus/*.md`
(``parse_corpus``) -> a SQLite FTS5 index for sparse (BM25) search
(``SparseIndex``) + an in-memory cosine-similarity index over dense
embeddings (``DenseIndex``) -> reciprocal-rank fusion of the two
(``HybridRetriever.retrieve_hybrid``). Reranking (a local cross-encoder) is
explicitly out of scope here -- that is P3.4's job.

**No new vector-db / ML dependency.** Sparse search reuses ``sqlite3``'s
built-in FTS5 module (already the project's SQLite-first convention -- see
``app.trace_store``, ``app.data.drug_interactions``); dense similarity is
plain-Python cosine (the corpus is a few dozen chunks -- no ``numpy``, no
``sqlite-vec``/Chroma/Qdrant needed at this size).

**Determinism.** Dense retrieval needs a query embedding. Two ways to get
one: (a) pass a pre-computed ``query_vector`` (what tests use, sourced from
the committed ``app/data/retrieval_embeddings.json`` recording -- see
``scripts/build_retrieval_embeddings.py``), or (b) construct with a live
``Embedder`` (``OllamaEmbedder``, production/manual-smoke use). No test in
this repo should ever require a live Ollama call to pass -- exactly the
record/replay discipline `docs/W2_ARCHITECTURE.md` "Testing Strategy"
already establishes for VLM/reranker calls, extended here to embeddings.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.schemas.retrieval import RetrievalMode, RetrievedChunk

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
EMBEDDINGS_PATH = Path(__file__).resolve().parent / "data" / "retrieval_embeddings.json"

_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class RetrievalError(Exception):
    """Raised when retrieval cannot proceed -- e.g. dense/hybrid retrieval
    was asked for without either a ``query_vector`` or a configured
    ``Embedder``."""


@dataclass(frozen=True)
class Chunk:
    """One deterministically-parsed corpus chunk (one ``##`` section of one
    corpus document), before any scoring is attached."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


class Embedder(Protocol):
    """Anything that can turn text into a dense vector -- ``OllamaEmbedder``
    in production, a recorded/fake double in tests."""

    def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """``Embedder`` backed by ``OllamaClient.embed`` (``nomic-embed-text``)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def embed(self, text: str) -> list[float]:
        return self._client.embed(text)


def _slugify(heading: str) -> str:
    """Deterministic ``##`` heading -> chunk-id slug (`corpus/README.md`
    "Document format"): lowercase, non-alphanumeric runs collapsed to a
    single ``-``, leading/trailing ``-`` stripped."""
    return _SLUG_RE.sub("-", heading.strip().lower()).strip("-")


def _parse_front_matter(lines: list[str]) -> tuple[dict[str, str | list[str]], int]:
    """Parse the ``---``-delimited front-matter block starting at
    ``lines[0]``. Returns ``(metadata, index_of_first_body_line)``. Values
    wrapped in ``[...]`` parse as a comma-separated list (``uc_mapping``);
    everything else is a plain string."""
    if not lines or lines[0].strip() != "---":
        raise ValueError("Corpus document is missing its front-matter block")
    metadata: dict[str, str | list[str]] = {}
    idx = 1
    while idx < len(lines) and lines[idx].strip() != "---":
        line = lines[idx]
        idx += 1
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            metadata[key] = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        else:
            metadata[key] = value
    if idx >= len(lines):
        raise ValueError("Corpus document front-matter block is never closed")
    return metadata, idx + 1


def parse_document(path: Path) -> list[Chunk]:
    """Parse one corpus markdown file into its ``##``-section chunks.

    Raises ``ValueError`` for a missing/malformed front-matter block, a
    missing ``id``/``title``, or a duplicate section heading within the
    document (which would collide on the same ``chunk_id``).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata, body_start = _parse_front_matter(lines)
    doc_id = metadata.get("id")
    title = metadata.get("title")
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError(f"{path.name}: front matter missing a non-empty 'id'")
    if not isinstance(title, str) or not title:
        raise ValueError(f"{path.name}: front matter missing a non-empty 'title'")

    body = "\n".join(lines[body_start:])
    parts = _SECTION_RE.split(body)
    chunks: list[Chunk] = []
    seen_slugs: set[str] = set()
    # parts[0] is any preamble before the first "## " heading; parts[1::2]
    # are headings, parts[2::2] their bodies -- see re.split's documented
    # behavior for a pattern with one capture group.
    for heading, content in zip(parts[1::2], parts[2::2]):
        section = heading.strip()
        slug = _slugify(section)
        if slug in seen_slugs:
            raise ValueError(f"{path.name}: duplicate section slug {slug!r} (heading {section!r})")
        seen_slugs.add(slug)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{slug}",
                doc_id=doc_id,
                title=title,
                section=section,
                text=content.strip(),
            )
        )
    return chunks


def parse_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    """Parse every corpus document (``*.md``, excluding ``README.md``) in
    ``corpus_dir`` into chunks, sorted by filename for a deterministic
    order."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        if path.name.casefold() == "readme.md":
            continue
        chunks.extend(parse_document(path))
    return chunks


def _fts_query(query: str) -> str:
    """Turn free text into an FTS5 MATCH query: an ``OR`` of each word
    token, individually quoted so punctuation in the input (``?``, ``:``,
    etc.) can never be parsed as FTS5 query-syntax operators. An ``OR`` (not
    the FTS5 default implicit ``AND``) is deliberate -- a natural-language
    question's stopwords would rarely all appear verbatim in one short
    section, so requiring every token would starve recall; ``bm25()``
    ranking already rewards rows that match more/rarer terms."""
    tokens = _WORD_RE.findall(query)
    if not tokens:
        return '""'
    return " OR ".join(f'"{token}"' for token in tokens)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SparseIndex:
    """BM25 sparse search over the corpus via SQLite FTS5."""

    def __init__(self, chunks: Sequence[Chunk], *, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5"
            "(chunk_id UNINDEXED, title, section, text)"
        )
        self._conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, title, section, text) VALUES (?, ?, ?, ?)",
            [(chunk.chunk_id, chunk.title, chunk.section, chunk.text) for chunk in chunks],
        )
        self._conn.commit()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs, best first.
        ``score`` is the negated FTS5 ``bm25()`` value -- FTS5's ``bm25()``
        is *lower-is-better* (it is a cost, not a relevance score); negating
        it gives the higher-is-better convention this module uses
        everywhere else (dense cosine similarity, fused RRF scores)."""
        cursor = self._conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) AS rank FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (_fts_query(query), k),
        )
        return [(chunk_id, -rank) for chunk_id, rank in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()


class DenseIndex:
    """Cosine-similarity search over pre-computed dense embeddings."""

    def __init__(self, chunk_ids: Sequence[str], embeddings: dict[str, list[float]]) -> None:
        missing = sorted(set(chunk_ids) - set(embeddings))
        if missing:
            raise ValueError(f"Missing dense embeddings for chunk ids: {missing}")
        self._embeddings = {chunk_id: embeddings[chunk_id] for chunk_id in chunk_ids}

    def search(self, query_vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs, best first, by
        cosine similarity to ``query_vector``."""
        scored = [
            (chunk_id, _cosine_similarity(query_vector, vector))
            for chunk_id, vector in self._embeddings.items()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


class HybridRetriever:
    """Sparse + dense retrieval over one parsed corpus, plus reciprocal-rank
    fusion for a combined hybrid result."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        dense_embeddings: dict[str, list[float]],
        *,
        embedder: Embedder | None = None,
        db_path: str = ":memory:",
    ) -> None:
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._sparse = SparseIndex(chunks, db_path=db_path)
        self._dense = DenseIndex(list(self._chunks_by_id), dense_embeddings)
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int,
        mode: RetrievalMode,
        *,
        query_vector: Sequence[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Dispatch to the requested mode. ``query_vector`` is only
        consulted for ``DENSE``/``HYBRID`` (see those methods)."""
        if mode is RetrievalMode.SPARSE:
            return self.retrieve_sparse(query, k)
        if mode is RetrievalMode.DENSE:
            return self.retrieve_dense(query, k, query_vector=query_vector)
        return self.retrieve_hybrid(query, k, query_vector=query_vector)

    def retrieve_sparse(self, query: str, k: int) -> list[RetrievedChunk]:
        hits = self._sparse.search(query, k)
        return [self._to_retrieved(chunk_id, {RetrievalMode.SPARSE.value: score}) for chunk_id, score in hits]

    def retrieve_dense(
        self,
        query: str,
        k: int,
        *,
        query_vector: Sequence[float] | None = None,
    ) -> list[RetrievedChunk]:
        vector = self._resolve_query_vector(query, query_vector)
        hits = self._dense.search(vector, k)
        return [self._to_retrieved(chunk_id, {RetrievalMode.DENSE.value: score}) for chunk_id, score in hits]

    def retrieve_hybrid(
        self,
        query: str,
        k: int,
        *,
        query_vector: Sequence[float] | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """Union sparse + dense candidates and fuse by reciprocal rank
        fusion: ``score(chunk) = sum over modes it appears in of
        1 / (rrf_k + rank_in_that_mode)``. RRF is chosen over a weighted
        blend of raw scores because BM25 and cosine similarity live on
        unrelated scales -- RRF only needs each mode's *rank ordering*, so
        no score-normalization tuning is required. ``rrf_k=60`` is the
        commonly-used default from the original RRF paper, damping the
        influence of any single very-high rank. Reranking (P3.4) is
        explicitly not implemented here.

        Pulls a candidate pool of ``max(k, 10)`` from each mode (not just
        ``k``) so fusion has enough overlap to work with -- a chunk ranked
        just outside a narrow top-k in one mode can still surface via
        fusion once a wider pool is candidates.
        """
        pool = max(k, 10)
        sparse_hits = self._sparse.search(query, pool)
        vector = self._resolve_query_vector(query, query_vector)
        dense_hits = self._dense.search(vector, pool)

        rrf_scores: dict[str, float] = {}
        per_mode_scores: dict[str, dict[str, float]] = {}
        for rank, (chunk_id, score) in enumerate(sparse_hits, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            per_mode_scores.setdefault(chunk_id, {})[RetrievalMode.SPARSE.value] = score
        for rank, (chunk_id, score) in enumerate(dense_hits, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            per_mode_scores.setdefault(chunk_id, {})[RetrievalMode.DENSE.value] = score

        ranked = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)[:k]
        results = []
        for chunk_id, fused in ranked:
            scores = dict(per_mode_scores[chunk_id])
            scores[RetrievalMode.HYBRID.value] = fused
            results.append(self._to_retrieved(chunk_id, scores))
        return results

    def _resolve_query_vector(self, query: str, query_vector: Sequence[float] | None) -> Sequence[float]:
        if query_vector is not None:
            return query_vector
        if self._embedder is None:
            raise RetrievalError(
                "Dense/hybrid retrieval needs a query embedding: pass query_vector= "
                "(e.g. a recorded vector) or construct with an embedder="
            )
        return self._embedder.embed(query)

    def _to_retrieved(self, chunk_id: str, scores: dict[str, float]) -> RetrievedChunk:
        chunk = self._chunks_by_id[chunk_id]
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
            scores=scores,
        )

    def close(self) -> None:
        self._sparse.close()


def load_recorded_embeddings(path: Path = EMBEDDINGS_PATH) -> dict[str, Any]:
    """Load the committed embeddings artifact (see
    ``scripts/build_retrieval_embeddings.py``): ``{"model": ..., "chunks":
    {chunk_id: vector}, "queries": {query_text: vector}}``."""
    return json.loads(path.read_text(encoding="utf-8"))


def recorded_query_vector(query: str, path: Path = EMBEDDINGS_PATH) -> list[float]:
    """Look up a golden query's recorded embedding by exact query text.
    Raises ``KeyError`` if ``query`` was never recorded -- see
    ``scripts/retrieval_golden_queries.py`` for the recorded set."""
    queries = load_recorded_embeddings(path).get("queries", {})
    if query not in queries:
        raise KeyError(f"No recorded embedding for query: {query!r}")
    return list(queries[query])


def build_retriever_from_corpus(
    corpus_dir: Path = CORPUS_DIR,
    embeddings_path: Path = EMBEDDINGS_PATH,
    *,
    embedder: Embedder | None = None,
    db_path: str = ":memory:",
) -> HybridRetriever:
    """Build a ``HybridRetriever`` over the committed corpus, using the
    committed recorded chunk embeddings. ``db_path`` is the SQLite FTS5
    index location -- ``":memory:"`` (the default) is fine for tests and for
    this corpus's tiny size; pass a real file path to persist it."""
    chunks = parse_corpus(corpus_dir)
    chunk_vectors = load_recorded_embeddings(embeddings_path).get("chunks", {})
    return HybridRetriever(chunks, chunk_vectors, embedder=embedder, db_path=db_path)

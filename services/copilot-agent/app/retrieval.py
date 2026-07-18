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
``Embedder`` -- ``OllamaClient.embed`` already satisfies that Protocol
structurally, so a production/manual-smoke caller passes the client
directly, no wrapper needed. No test in this repo should ever require a
live Ollama call to pass -- exactly the record/replay discipline
`docs/W2_ARCHITECTURE.md` "Testing Strategy" already establishes for
VLM/reranker calls, extended here to embeddings.

**Query-length bound.** Every public ``retrieve*`` entry point rejects a
query longer than ``MAX_QUERY_CHARS`` (2000, generous for any real clinical
question) with ``RetrievalError`` -- an unbounded query turns into an
unbounded FTS5 ``MATCH`` expression (one quoted ``OR`` clause per word
token), which is cheap per-token but not free at arbitrary size; a
multi-megabyte query is a low-effort DoS once an endpoint wires user input
to these entry points. ``_fts_query`` additionally caps the token count fed
into the FTS5 expression (``_MAX_QUERY_TOKENS``) as defense in depth, in
case a future caller bypasses the character bound with some other query
source.

**Embedding-drift detection.** The committed artifact records, per chunk,
both its dense vector AND a sha256 of the chunk text it was computed from
(``chunk_text_sha256``). ``DenseIndex`` re-hashes each chunk's CURRENT text
at construction time and compares -- a corpus edit that changes a chunk's
text without regenerating the artifact (``scripts/build_retrieval_embeddings.py``)
would otherwise silently serve a stale vector for the new text, with no
error to say so. A hash mismatch raises ``RetrievalError`` naming the
drifted chunk id(s).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from app.schemas.retrieval import RetrievalMode, RetrievedChunk

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
EMBEDDINGS_PATH = Path(__file__).resolve().parent / "data" / "retrieval_embeddings.json"

# DoS guard (see module docstring "Query-length bound"): the largest query
# any public retrieve* entry point will accept, and the largest number of
# word tokens _fts_query will ever build a MATCH expression from.
MAX_QUERY_CHARS = 2000
_MAX_QUERY_TOKENS = 64

_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class RetrievalError(Exception):
    """Raised when retrieval cannot proceed -- e.g. dense/hybrid retrieval
    was asked for without either a ``query_vector`` or a configured
    ``Embedder``, or a query exceeds ``MAX_QUERY_CHARS``."""


def _validate_query_length(query: str) -> None:
    """Reject a query longer than ``MAX_QUERY_CHARS`` (see module docstring
    "Query-length bound"). Called at the top of every public ``retrieve*``
    entry point, before the query ever reaches ``SparseIndex``/
    ``DenseIndex``."""
    if len(query) > MAX_QUERY_CHARS:
        raise RetrievalError(
            f"Query exceeds the {MAX_QUERY_CHARS}-character limit ({len(query)} chars)"
        )


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
    """Anything that can turn text into a dense vector -- ``OllamaClient``
    (its ``embed`` method) in production, a recorded/fake double in tests."""

    def embed(self, text: str) -> list[float]: ...


def _slugify(heading: str) -> str:
    """Deterministic ``##`` heading -> chunk-id slug (`corpus/README.md`
    "Document format"): lowercase, non-alphanumeric runs collapsed to a
    single ``-``, leading/trailing ``-`` stripped."""
    return _SLUG_RE.sub("-", heading.strip().lower()).strip("-")


def _parse_front_matter(lines: list[str]) -> tuple[dict[str, Any], int]:
    """Parse the ``---``-delimited YAML front-matter block starting at
    ``lines[0]`` via ``yaml.safe_load``. Returns ``(metadata,
    index_of_first_body_line)``."""
    if not lines or lines[0].strip() != "---":
        raise ValueError("Corpus document is missing its front-matter block")
    idx = 1
    while idx < len(lines) and lines[idx].strip() != "---":
        idx += 1
    if idx >= len(lines):
        raise ValueError("Corpus document front-matter block is never closed")
    metadata = yaml.safe_load("\n".join(lines[1:idx])) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Corpus document front-matter block did not parse to a mapping")
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
    ranking already rewards rows that match more/rarer terms.

    Capped to the first ``_MAX_QUERY_TOKENS`` tokens -- defense in depth
    alongside ``_validate_query_length``'s character bound (see module
    docstring "Query-length bound"), in case some future caller reaches
    this function with a query that bypassed the character check."""
    tokens = _WORD_RE.findall(query)[:_MAX_QUERY_TOKENS]
    if not tokens:
        return '""'
    return " OR ".join(f'"{token}"' for token in tokens)


def chunk_text_sha256(text: str) -> str:
    """Deterministic sha256 hex digest of a chunk's text -- the
    drift-detection fingerprint recorded per chunk in the embeddings
    artifact (``scripts/build_retrieval_embeddings.py``) and re-verified by
    ``DenseIndex`` against each chunk's CURRENT text (see module docstring
    "Embedding-drift detection")."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SparseIndex:
    """BM25 sparse search over the corpus via SQLite FTS5."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._conn = sqlite3.connect(":memory:")
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
    """Cosine-similarity search over pre-computed dense embeddings.

    ``embeddings`` maps ``chunk_id -> {"vector": [...], "text_sha256":
    "..."}`` (the recorded-artifact shape -- see
    ``scripts/build_retrieval_embeddings.py``). Construction re-hashes each
    chunk's CURRENT text and compares it against the recorded
    ``text_sha256``: a mismatch means the corpus text changed since the
    artifact was last regenerated, so the recorded vector no longer
    corresponds to what it claims to embed -- raised loudly as
    ``RetrievalError`` rather than silently served (see module docstring
    "Embedding-drift detection")."""

    def __init__(self, chunks: Sequence[Chunk], embeddings: dict[str, dict[str, Any]]) -> None:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        missing = sorted(set(chunk_ids) - set(embeddings))
        if missing:
            raise ValueError(f"Missing dense embeddings for chunk ids: {missing}")

        drifted: list[str] = []
        vectors: dict[str, list[float]] = {}
        for chunk in chunks:
            recorded = embeddings[chunk.chunk_id]
            if recorded.get("text_sha256") != chunk_text_sha256(chunk.text):
                drifted.append(chunk.chunk_id)
                continue
            vectors[chunk.chunk_id] = recorded["vector"]
        if drifted:
            raise RetrievalError(
                f"Recorded embedding(s) are stale for chunk id(s) {sorted(drifted)}: "
                "chunk text has changed since app/data/retrieval_embeddings.json was "
                "recorded. Regenerate it via scripts/build_retrieval_embeddings.py."
            )
        self._embeddings = vectors

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
        dense_embeddings: dict[str, dict[str, Any]],
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._sparse = SparseIndex(chunks)
        self._dense = DenseIndex(chunks, dense_embeddings)
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
        _validate_query_length(query)
        hits = self._sparse.search(query, k)
        return [self._to_retrieved(chunk_id, {RetrievalMode.SPARSE.value: score}) for chunk_id, score in hits]

    def retrieve_dense(
        self,
        query: str,
        k: int,
        *,
        query_vector: Sequence[float] | None = None,
    ) -> list[RetrievedChunk]:
        _validate_query_length(query)
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
        _validate_query_length(query)
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
    {chunk_id: {"vector": [...], "text_sha256": "..."}}, "queries":
    {query_text: vector}}``."""
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
) -> HybridRetriever:
    """Build a ``HybridRetriever`` over the committed corpus, using the
    committed recorded chunk embeddings. The sparse index is always
    in-memory (``SparseIndex`` hardcodes ``":memory:"``) -- fine for tests
    and for this corpus's tiny size; persistence returns when a real caller
    needs it."""
    chunks = parse_corpus(corpus_dir)
    chunk_vectors = load_recorded_embeddings(embeddings_path).get("chunks", {})
    return HybridRetriever(chunks, chunk_vectors, embedder=embedder)

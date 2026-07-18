"""LLM-based (local Ollama) reranking over hybrid-retrieval candidates
(P3.4, `docs/W2_ARCHITECTURE.md` "Hybrid retrieval-augmented answering"
reranker section, `planning/PLAN.md` "Fully-local tooling choices").

**Dependency decision (P3.4).** `planning/PLAN.md` names a local
cross-encoder (`bge-reranker-v2-m3` via `sentence-transformers`) as the
ideal reranker. Adding `sentence-transformers` pulls in `torch` -- a very
heavy dependency (hundreds of MB, its own CUDA/wheel-selection complexity)
that contradicts this service's established dependency-light,
fully-local-appliance thesis: `app.retrieval`'s hybrid index deliberately
avoids `numpy`/`chromadb`/`sentence-transformers` even for its vector index,
using plain-Python cosine similarity instead, at this corpus's size. This
module reranks instead via a schema-constrained pointwise relevance-score
prompt against the ALREADY-PROVISIONED local Ollama chat model
(`OllamaClient.extract`, the exact mechanism `app.extraction` already uses
for document extraction) -- zero new dependencies, reuses infrastructure the
container already runs. This is the local substitute for "Cohere Rerank or
equivalent" that `planning/PLAN.md`'s "Fully local tooling choices" calls
for, the same framing already applied there to VLM extraction and
embeddings.

**Pointwise, not listwise.** Each candidate chunk is scored independently
(``RelevanceScore.score``, 0.0-1.0, schema-constrained via
``OllamaClient.extract``) rather than asking the model to emit a full
re-ordered candidate list in one call. A listwise prompt asks a 4B instruct
model to enumerate an out-of-N ranking (or ordered id list) in one shot --
schema-constrained decoding can force *valid JSON shape* but not
*correctness*: nothing stops the model from omitting a candidate,
duplicating one, or inventing an id, all of which need fallback handling
anyway. Pointwise scoring only ever asks one question at a time ("how
relevant is this chunk to this query, 0-1"), which even a small model
answers reliably, and Python does the (correct-by-construction) sort --
more LLM calls per query, but each one is unambiguous. The candidate pool
here is small (``HybridRetriever.retrieve_hybrid``'s default pool is
``max(k, 10)``), so the extra calls are cheap.

**Determinism.** Same record/replay discipline as `app.retrieval`'s dense
embeddings (see that module's docstring "Determinism" /
"Embedding-drift detection"): no test in this repo should require a live
Ollama call. ``scripts/build_reranker_scores.py`` records real Ollama
relevance scores for the golden queries' hybrid candidates (plus one
deliberately-planted lexical distractor per query, see
``scripts/reranker_golden_distractors.py``) into the committed
``app/data/reranker_scores.json``; tests replay via ``RecordedRerankScorer``.
Each recorded entry is keyed by a sha256 of the exact chunk text it was
scored from (``app.retrieval.chunk_text_sha256``) rather than by chunk id --
``RecordedRerankScorer`` raises ``RerankError`` for a (query, chunk) pair
whose text doesn't match anything recorded for that query, mirroring
`app.retrieval.DenseIndex`'s embedding-drift guard: a corpus edit that
changes a chunk's text (or a distractor swapped for a different chunk)
without re-running the recording script fails loudly instead of silently
scoring the wrong text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from app.retrieval import HybridRetriever, chunk_text_sha256
from app.schemas.reranking import RelevanceScore, RerankedChunk
from app.schemas.retrieval import RetrievedChunk

RERANKER_SCORES_PATH = Path(__file__).resolve().parent / "data" / "reranker_scores.json"

# Reuses app.retrieval.MAX_QUERY_CHARS's rationale verbatim: an unbounded
# query fed into a per-candidate LLM prompt is a low-effort cost-amplifying
# DoS once an endpoint wires user input to this entry point.
MAX_QUERY_CHARS = 2000

_RELEVANCE_PROMPT = (
    "Rate how relevant the following clinical-guideline passage is to the "
    "question, on a fine-grained scale from 0.0 (irrelevant) to 1.0 (the "
    "single most specific, complete, directly-stated answer). Use precise "
    "decimals (e.g. 0.62, 0.87) rather than round numbers, and prefer the "
    "more specific, detailed passage over a general summary mention of the "
    "same fact when both are otherwise relevant.\n\n"
    "Question: {query!r}\n\nPassage:\n{chunk_text}"
)


class RerankError(Exception):
    """Raised when reranking cannot proceed -- a query exceeds
    ``MAX_QUERY_CHARS``, or (``RecordedRerankScorer``) a candidate's
    recorded score was computed from chunk text that has since drifted, or
    was never recorded at all."""


class RerankScorer(Protocol):
    """Anything that can score one (query, chunk_text) pair for relevance --
    ``OllamaRerankScorer`` in production, ``RecordedRerankScorer`` (replaying
    ``app/data/reranker_scores.json``) in tests."""

    def score(self, query: str, chunk_text: str) -> float: ...


class _Extractor(Protocol):
    """Structural subset of ``OllamaClient`` this module depends on --
    matches ``app.retrieval.Embedder``'s pattern of depending on a narrow
    Protocol rather than importing ``OllamaClient`` itself."""

    def extract(self, prompt_or_messages: str, schema: type[RelevanceScore]) -> RelevanceScore: ...


class OllamaRerankScorer:
    """Production ``RerankScorer``: one schema-constrained
    ``OllamaClient.extract`` call per (query, chunk) pair, asking for a
    0.0-1.0 relevance score (see module docstring "Pointwise, not
    listwise")."""

    def __init__(self, client: _Extractor) -> None:
        self._client = client

    def score(self, query: str, chunk_text: str) -> float:
        result = self._client.extract(
            _RELEVANCE_PROMPT.format(query=query, chunk_text=chunk_text),
            RelevanceScore,
        )
        return result.score


class RecordedRerankScorer:
    """Offline ``RerankScorer`` double: replays sha256-guarded scores
    recorded by ``scripts/build_reranker_scores.py`` into the committed
    ``app/data/reranker_scores.json`` artifact, instead of a live Ollama call
    -- the record/replay discipline described in the module docstring.

    ``recorded`` is the artifact's ``{"scores": {query: {text_sha256:
    score}}}`` mapping (see ``load_recorded_reranker_scores``).
    """

    def __init__(self, recorded: dict[str, dict[str, float]]) -> None:
        self._recorded = recorded

    def score(self, query: str, chunk_text: str) -> float:
        query_entries = self._recorded.get(query)
        if query_entries is None:
            raise RerankError(f"No recorded rerank scores for query: {query!r}")
        text_hash = chunk_text_sha256(chunk_text)
        if text_hash not in query_entries:
            raise RerankError(
                f"No recorded rerank score for query {query!r} matching chunk text hash "
                f"{text_hash!r} -- this candidate's text was never recorded for this query, "
                "or its text has drifted since app/data/reranker_scores.json was recorded. "
                "Regenerate it via scripts/build_reranker_scores.py."
            )
        return query_entries[text_hash]


class Reranker:
    """Reranks hybrid-retrieval candidates via pointwise relevance scoring
    (see module docstring)."""

    def __init__(self, scorer: RerankScorer) -> None:
        self._scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RerankedChunk]:
        """Score every candidate for relevance to ``query`` and return the
        top ``top_k``, best first. Every citation-bearing field of a
        candidate (``chunk_id``, ``doc_id``, ``section``, ``text``,
        ``scores``) is carried through unchanged onto its ``RerankedChunk``.
        """
        if len(query) > MAX_QUERY_CHARS:
            raise RerankError(f"Query exceeds the {MAX_QUERY_CHARS}-character limit ({len(query)} chars)")

        scored = [(candidate, self._scorer.score(query, candidate.text)) for candidate in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            RerankedChunk(**candidate.model_dump(), rerank_score=score) for candidate, score in scored[:top_k]
        ]


def load_recorded_reranker_scores(path: Path = RERANKER_SCORES_PATH) -> dict[str, dict[str, float]]:
    """Load the committed reranker-scores artifact (see
    ``scripts/build_reranker_scores.py``): ``{query: {text_sha256: score}}``.
    """
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload["scores"]


def retrieve_and_rerank(
    retriever: HybridRetriever,
    reranker: Reranker,
    query: str,
    k: int,
    *,
    query_vector: Sequence[float] | None = None,
    candidate_pool: int | None = None,
) -> list[RerankedChunk]:
    """Hybrid-retrieve a candidate pool via ``retriever.retrieve_hybrid``,
    then rerank it down to ``k`` via ``reranker``. A standalone composing
    function rather than a ``HybridRetriever`` method -- ``app.retrieval``
    stays completely untouched (every existing ``retrieve_hybrid`` caller/
    test keeps working unmodified), and this module already depends on
    ``app.retrieval`` one-way (for ``HybridRetriever`` and
    ``chunk_text_sha256``), so adding the reverse dependency there would
    create a cycle.

    ``candidate_pool`` defaults to ``max(k, 10)``, matching
    ``retrieve_hybrid``'s own default pool-widening rationale (see its
    docstring) -- reranking needs the same wider pool to have anything
    worth reordering.
    """
    pool = candidate_pool if candidate_pool is not None else max(k, 10)
    candidates = retriever.retrieve_hybrid(query, pool, query_vector=query_vector)
    return reranker.rerank(query, candidates, k)

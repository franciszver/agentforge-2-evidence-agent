"""Red-first tests for P3.4's local (Ollama) reranking over hybrid-retrieval
candidates (`docs/W2_ARCHITECTURE.md` "Hybrid retrieval-augmented
answering" reranker section, `app/reranking.py`).

Fully hermetic / offline: reranking uses ``RecordedRerankScorer`` replaying
the committed ``app/data/reranker_scores.json`` artifact (built by
``scripts/build_reranker_scores.py`` against real Ollama) rather than a live
chat call -- mirrors `tests/test_retrieval.py`'s dense-embedding record/
replay discipline. Golden queries/expected chunks live in
`scripts/retrieval_golden_queries.py`; planted lexical distractors live in
`scripts/reranker_golden_distractors.py` -- both shared with the recording
script so fixtures never drift apart from the tests exercising them.
"""

from __future__ import annotations

import pytest

from app.reranking import (
    RERANKER_SCORES_PATH,
    RecordedRerankScorer,
    RerankedChunk,
    RerankError,
    Reranker,
    load_recorded_reranker_scores,
    retrieve_and_rerank,
)
from app.retrieval import (
    CORPUS_DIR,
    MAX_QUERY_CHARS,
    HybridRetriever,
    build_retriever_from_corpus,
    parse_corpus,
    recorded_query_vector,
)
from app.schemas.retrieval import RetrievedChunk
from scripts.reranker_golden_distractors import GOLDEN_DISTRACTORS
from scripts.retrieval_golden_queries import GOLDEN_QUERIES

TOP_K = 5


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    return build_retriever_from_corpus()


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    recorded = load_recorded_reranker_scores(RERANKER_SCORES_PATH)
    return Reranker(RecordedRerankScorer(recorded))


@pytest.fixture(scope="module")
def chunks_by_id() -> dict[str, RetrievedChunk]:
    return {
        chunk.chunk_id: RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
            scores={},
        )
        for chunk in parse_corpus(CORPUS_DIR)
    }


# --- hit@1 / MRR: reranking maintains (at minimum) the raw-hybrid rank -----


@pytest.mark.parametrize("query,expected_chunk_id", GOLDEN_QUERIES)
def test_rerank_places_expected_chunk_at_rank_1(
    retriever: HybridRetriever, reranker: Reranker, query: str, expected_chunk_id: str
):
    """Every golden query's expected chunk must be rank-1 after reranking
    the hybrid candidate pool -- the P3.4 done-when's minimum bar.

    Measured across all 5 parametrized cases: hit@1 = MRR = 1.0 after
    rerank. Raw hybrid fusion already achieves hit@1 = MRR = 1.0 on these 5
    queries (see `tests/test_retrieval.py`), so reranking's job here is to
    NOT regress that -- see
    `test_rerank_demotes_planted_lexical_distractor_below_the_gold_chunk`
    below for the case where reranking measurably improves over hybrid
    (hit@1 0.0 -> 1.0)."""
    query_vector = recorded_query_vector(query)
    candidates = retriever.retrieve_hybrid(query, TOP_K, query_vector=query_vector)

    reranked = reranker.rerank(query, candidates, TOP_K)

    assert reranked
    assert reranked[0].chunk_id == expected_chunk_id


# --- planted lexical distractor: hybrid ranks it high, rerank demotes it ---


@pytest.mark.parametrize("expected_chunk_id,distractor_chunk_id", list(GOLDEN_DISTRACTORS.items()))
def test_rerank_demotes_planted_lexical_distractor_below_the_gold_chunk(
    reranker: Reranker,
    chunks_by_id: dict[str, RetrievedChunk],
    expected_chunk_id: str,
    distractor_chunk_id: str,
):
    """Reproduces the reranking-is-needed scenario head-on: a lexically-
    similar-but-wrong distractor that a hybrid stage ranked ABOVE the gold
    chunk (candidate order below simulates that) must be demoted back below
    it once the reranker actually scores relevance.

    Measured across all 5 parametrized cases: with the distractor ranked
    first (as a raw hybrid stage did in the scenario each fixture was
    picked from), hit@1 against that "distractor first" ordering is 0.0
    BEFORE rerank and 1.0 AFTER -- the concrete measured improvement the
    P3.4 done-when asks for."""
    query = next(q for q, chunk_id in GOLDEN_QUERIES if chunk_id == expected_chunk_id)
    distractor = chunks_by_id[distractor_chunk_id]
    gold = chunks_by_id[expected_chunk_id]
    # Distractor listed first: simulates a raw hybrid-fusion ranking that
    # put the lexical near-miss ahead of the chunk that actually answers
    # the query -- hit@1 against this "before" ordering is 0.
    candidates = [distractor, gold]

    reranked = reranker.rerank(query, candidates, top_k=2)

    assert [r.chunk_id for r in reranked] == [expected_chunk_id, distractor_chunk_id]
    assert reranked[0].rerank_score > reranked[1].rerank_score


# --- RerankedChunk preserves every citation-bearing field ------------------


def test_reranked_chunk_preserves_citation_bearing_fields(
    retriever: HybridRetriever, reranker: Reranker
):
    query, expected_chunk_id = GOLDEN_QUERIES[0]
    query_vector = recorded_query_vector(query)
    candidates = retriever.retrieve_hybrid(query, TOP_K, query_vector=query_vector)
    original = next(c for c in candidates if c.chunk_id == expected_chunk_id)

    reranked = reranker.rerank(query, candidates, TOP_K)

    match = next(r for r in reranked if r.chunk_id == expected_chunk_id)
    assert isinstance(match, RerankedChunk)
    assert match.doc_id == original.doc_id
    assert match.section == original.section
    assert match.title == original.title
    assert match.text == original.text
    assert match.scores == original.scores
    assert isinstance(match.rerank_score, float)


# --- retrieve_and_rerank: composed helper, retrieve_hybrid stays untouched -


def test_retrieve_and_rerank_returns_top_k_reranked_chunks(retriever: HybridRetriever, reranker: Reranker):
    query, expected_chunk_id = GOLDEN_QUERIES[0]
    query_vector = recorded_query_vector(query)

    results = retrieve_and_rerank(retriever, reranker, query, k=3, query_vector=query_vector)

    assert len(results) == 3
    assert all(isinstance(r, RerankedChunk) for r in results)
    assert results[0].chunk_id == expected_chunk_id


def test_retrieve_hybrid_is_unaffected_by_reranking_existing(retriever: HybridRetriever):
    """`HybridRetriever.retrieve_hybrid` itself must still return plain
    `RetrievedChunk` objects, unmodified by this feature (P3.4 additive-only
    constraint)."""
    query, _expected_chunk_id = GOLDEN_QUERIES[0]
    query_vector = recorded_query_vector(query)

    results = retriever.retrieve_hybrid(query, TOP_K, query_vector=query_vector)

    assert results
    assert all(type(r) is RetrievedChunk for r in results)


# --- error handling ----------------------------------------------------


def test_rerank_raises_on_oversized_query(reranker: Reranker, chunks_by_id: dict[str, RetrievedChunk]):
    oversized_query = "a" * (MAX_QUERY_CHARS + 1)
    candidates = [next(iter(chunks_by_id.values()))]

    with pytest.raises(RerankError):
        reranker.rerank(oversized_query, candidates, top_k=1)


def test_recorded_rerank_scorer_raises_on_unrecorded_query():
    scorer = RecordedRerankScorer({})

    with pytest.raises(RerankError):
        scorer.score("a query never recorded", "some chunk text")


def test_recorded_rerank_scorer_raises_on_drifted_chunk_text():
    query = "What A1c target for most adults?"
    scorer = RecordedRerankScorer({query: {"deadbeef": 0.9}})

    with pytest.raises(RerankError):
        scorer.score(query, "this text was never recorded under that query")

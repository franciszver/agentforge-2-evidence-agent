"""Red-first tests for P3.3's hybrid (sparse + dense) guideline-corpus
retrieval (`docs/W2_ARCHITECTURE.md` "Hybrid retrieval-augmented
answering", `corpus/README.md`).

Fully hermetic / offline: dense retrieval uses the committed recorded
embeddings (``app/data/retrieval_embeddings.json``, built by
``scripts/build_retrieval_embeddings.py`` against real Ollama) rather than a
live embedding call -- see ``app.retrieval``'s module docstring. The golden
queries and their expected chunk ids live in
``scripts/retrieval_golden_queries.py`` so this file and the regeneration
script never drift apart.
"""

from __future__ import annotations

import pytest

from app.retrieval import (
    CORPUS_DIR,
    MAX_QUERY_CHARS,
    Chunk,
    DenseIndex,
    HybridRetriever,
    RetrievalError,
    SparseIndex,
    build_retriever_from_corpus,
    parse_corpus,
    recorded_query_vector,
)
from app.schemas.retrieval import RetrievalMode, RetrievedChunk
from scripts.retrieval_golden_queries import GOLDEN_QUERIES

TOP_K = 5


# --- corpus parsing / chunking ----------------------------------------------


def test_parse_corpus_produces_at_least_six_documents_worth_of_chunks():
    chunks = parse_corpus(CORPUS_DIR)
    doc_ids = {chunk.doc_id for chunk in chunks}

    assert len(doc_ids) >= 6
    assert len(chunks) >= len(doc_ids)


def test_parse_corpus_chunk_ids_are_unique_and_doc_id_hash_section_shaped():
    chunks = parse_corpus(CORPUS_DIR)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
    for chunk in chunks:
        assert chunk.chunk_id == f"{chunk.doc_id}#{chunk.chunk_id.split('#', 1)[1]}"
        assert chunk.chunk_id.startswith(chunk.doc_id + "#")


def test_parse_corpus_includes_every_golden_expected_chunk_id():
    chunks = parse_corpus(CORPUS_DIR)
    chunk_ids = {chunk.chunk_id for chunk in chunks}

    for _query, expected_chunk_id in GOLDEN_QUERIES:
        assert expected_chunk_id in chunk_ids


def test_parse_corpus_readme_is_excluded():
    chunks = parse_corpus(CORPUS_DIR)

    assert all(chunk.doc_id != "readme" for chunk in chunks)


def test_parse_corpus_chunk_text_is_nonempty():
    chunks = parse_corpus(CORPUS_DIR)

    assert all(chunk.text.strip() for chunk in chunks)


# --- SparseIndex (BM25 / SQLite FTS5) ---------------------------------------


def test_sparse_index_finds_exact_keyword_match():
    chunks = [
        Chunk(chunk_id="doc-a#one", doc_id="doc-a", title="Doc A", section="One", text="ibuprofen and lisinopril caution"),
        Chunk(chunk_id="doc-b#one", doc_id="doc-b", title="Doc B", section="One", text="unrelated content about diet"),
    ]
    index = SparseIndex(chunks)

    hits = index.search("ibuprofen lisinopril", k=5)

    assert hits
    assert hits[0][0] == "doc-a#one"


def test_sparse_index_handles_punctuation_in_query_without_raising():
    chunks = parse_corpus(CORPUS_DIR)
    index = SparseIndex(chunks)

    # Must not raise on FTS5-special characters (?, :, etc.) in a natural
    # question -- see app.retrieval._fts_query.
    hits = index.search("Can I give ibuprofen with lisinopril?", k=5)

    assert isinstance(hits, list)


# --- DenseIndex (cosine similarity) -----------------------------------------


def test_dense_index_ranks_the_closer_vector_first():
    index = DenseIndex(
        ["a", "b"],
        {"a": [1.0, 0.0], "b": [0.0, 1.0]},
    )

    hits = index.search([1.0, 0.0], k=2)

    assert hits[0][0] == "a"
    assert hits[0][1] > hits[1][1]


def test_dense_index_raises_on_missing_embedding():
    with pytest.raises(ValueError):
        DenseIndex(["a", "b"], {"a": [1.0]})


# --- HybridRetriever: golden queries, per mode ------------------------------


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    return build_retriever_from_corpus()


@pytest.mark.parametrize("query,expected_chunk_id", GOLDEN_QUERIES)
def test_sparse_retrieval_finds_expected_chunk_in_top_k(retriever: HybridRetriever, query: str, expected_chunk_id: str):
    results = retriever.retrieve_sparse(query, k=TOP_K)

    assert expected_chunk_id in [r.chunk_id for r in results]


@pytest.mark.parametrize("query,expected_chunk_id", GOLDEN_QUERIES)
def test_dense_retrieval_finds_expected_chunk_in_top_k(retriever: HybridRetriever, query: str, expected_chunk_id: str):
    query_vector = recorded_query_vector(query)

    results = retriever.retrieve_dense(query, k=TOP_K, query_vector=query_vector)

    assert expected_chunk_id in [r.chunk_id for r in results]


@pytest.mark.parametrize("query,expected_chunk_id", GOLDEN_QUERIES)
def test_hybrid_retrieval_finds_expected_chunk_in_top_k(retriever: HybridRetriever, query: str, expected_chunk_id: str):
    query_vector = recorded_query_vector(query)

    results = retriever.retrieve_hybrid(query, k=TOP_K, query_vector=query_vector)

    assert expected_chunk_id in [r.chunk_id for r in results]


@pytest.mark.parametrize("mode", [RetrievalMode.SPARSE, RetrievalMode.DENSE, RetrievalMode.HYBRID])
def test_retrieve_dispatches_to_the_requested_mode(retriever: HybridRetriever, mode: RetrievalMode):
    query, expected_chunk_id = GOLDEN_QUERIES[0]
    query_vector = recorded_query_vector(query)

    results = retriever.retrieve(query, k=TOP_K, mode=mode, query_vector=query_vector)

    assert expected_chunk_id in [r.chunk_id for r in results]


def test_retrieved_chunk_carries_doc_id_and_section_for_the_citation_contract(retriever: HybridRetriever):
    query, expected_chunk_id = GOLDEN_QUERIES[0]

    results = retriever.retrieve_sparse(query, k=TOP_K)

    match = next(r for r in results if r.chunk_id == expected_chunk_id)
    assert isinstance(match, RetrievedChunk)
    assert match.doc_id == expected_chunk_id.split("#", 1)[0]
    assert match.section
    assert match.title
    assert match.text


def test_hybrid_result_scores_carry_sparse_dense_and_hybrid_keys(retriever: HybridRetriever):
    query, _expected_chunk_id = GOLDEN_QUERIES[0]
    query_vector = recorded_query_vector(query)

    results = retriever.retrieve_hybrid(query, k=TOP_K, query_vector=query_vector)

    assert results
    for result in results:
        assert "hybrid" in result.scores
        assert set(result.scores) <= {"sparse", "dense", "hybrid"}


def test_dense_retrieval_without_embedder_or_query_vector_raises():
    retriever_no_embedder = build_retriever_from_corpus()

    with pytest.raises(RetrievalError):
        retriever_no_embedder.retrieve_dense("What A1c target for most adults?", k=TOP_K)


def test_hybrid_retrieval_without_embedder_or_query_vector_raises():
    retriever_no_embedder = build_retriever_from_corpus()

    with pytest.raises(RetrievalError):
        retriever_no_embedder.retrieve_hybrid("What A1c target for most adults?", k=TOP_K)


# --- query-length bound (DoS guard: unbounded query -> unbounded FTS5 MATCH
# expression cost) -------------------------------------------------------


def test_oversized_query_raises_retrieval_error_on_sparse(retriever: HybridRetriever):
    oversized_query = "a" * (MAX_QUERY_CHARS + 1)

    with pytest.raises(RetrievalError):
        retriever.retrieve_sparse(oversized_query, k=TOP_K)


def test_oversized_query_raises_retrieval_error_on_dense(retriever: HybridRetriever):
    oversized_query = "a" * (MAX_QUERY_CHARS + 1)

    with pytest.raises(RetrievalError):
        retriever.retrieve_dense(oversized_query, k=TOP_K, query_vector=[0.0])


def test_oversized_query_raises_retrieval_error_on_hybrid(retriever: HybridRetriever):
    oversized_query = "a" * (MAX_QUERY_CHARS + 1)

    with pytest.raises(RetrievalError):
        retriever.retrieve_hybrid(oversized_query, k=TOP_K, query_vector=[0.0])


def test_oversized_query_raises_retrieval_error_via_retrieve_dispatch(retriever: HybridRetriever):
    oversized_query = "a" * (MAX_QUERY_CHARS + 1)

    with pytest.raises(RetrievalError):
        retriever.retrieve(oversized_query, k=TOP_K, mode=RetrievalMode.SPARSE)


def test_boundary_length_query_at_exactly_max_chars_still_works(retriever: HybridRetriever):
    boundary_query = "a" * MAX_QUERY_CHARS

    results = retriever.retrieve_sparse(boundary_query, k=TOP_K)

    assert isinstance(results, list)

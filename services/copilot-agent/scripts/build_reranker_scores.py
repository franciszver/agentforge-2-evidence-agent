"""Regenerate the committed reranker-scores artifact for P3.4
(``app/reranking.py``).

Run against a REACHABLE Ollama instance serving the chat model
(``qwen3:4b``) to (re)build ``app/data/reranker_scores.json``: for each
golden query (``scripts/retrieval_golden_queries.py``), a real Ollama
relevance score (``app.reranking.OllamaRerankScorer``) for every chunk in
its hybrid top-``RECORD_POOL`` candidate pool (10 -- covers both
`tests/test_reranking.py`'s TOP_K=5 direct calls, a subset of the same
fused ranking, and `app.reranking.retrieve_and_rerank`'s default
``max(k, 10)`` pool), plus its planted lexical distractor
(``scripts/reranker_golden_distractors.py``). This is the ONE place a live
Ollama chat call happens for this feature -- the recorded artifact it
produces is what ``tests/test_reranking.py`` replays against
(``app.reranking.RecordedRerankScorer``), mirroring
``scripts/build_retrieval_embeddings.py``'s record/replay discipline for
dense embeddings.

Each entry is keyed by a sha256 of the exact chunk text it was scored from
(``app.retrieval.chunk_text_sha256``), not by chunk id -- see
``app.reranking``'s module docstring "Determinism" for why (mirrors
``app.retrieval.DenseIndex``'s embedding-drift guard).

Usage (against the dev stack's Ollama, bridged to the host):

    OLLAMA_BASE_URL=http://localhost:11435 python -m scripts.build_reranker_scores

Re-run this and commit the resulting ``app/data/reranker_scores.json``
whenever a golden query, its expected chunk, or its planted distractor
changes, or when corpus text affecting any of those chunks changes.
"""

from __future__ import annotations

import json
import os

from app.config import Settings
from app.ollama_client import OllamaClient
from app.reranking import RERANKER_SCORES_PATH, OllamaRerankScorer
from app.retrieval import CORPUS_DIR, chunk_text_sha256, parse_corpus, recorded_query_vector
from app.retrieval import build_retriever_from_corpus
from scripts.reranker_golden_distractors import GOLDEN_DISTRACTORS
from scripts.retrieval_golden_queries import GOLDEN_QUERIES

RECORD_POOL = 10


def main() -> None:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11435")
    settings = Settings(ollama_base_url=base_url, ollama_api_timeout_seconds=120.0)
    client = OllamaClient.from_settings(settings)
    scorer = OllamaRerankScorer(client)

    retriever = build_retriever_from_corpus()
    chunks_by_id = {chunk.chunk_id: chunk for chunk in parse_corpus(CORPUS_DIR)}

    scores: dict[str, dict[str, float]] = {}
    for query, expected_chunk_id in GOLDEN_QUERIES:
        query_vector = recorded_query_vector(query)
        candidates = retriever.retrieve_hybrid(query, RECORD_POOL, query_vector=query_vector)

        texts: dict[str, str] = {c.chunk_id: c.text for c in candidates}
        distractor_id = GOLDEN_DISTRACTORS[expected_chunk_id]
        texts[distractor_id] = chunks_by_id[distractor_id].text

        query_scores: dict[str, float] = {}
        for chunk_id, text in texts.items():
            query_scores[chunk_text_sha256(text)] = scorer.score(query, text)
        scores[query] = query_scores

    payload = {"model": settings.ollama_model, "scores": scores}
    RERANKER_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RERANKER_SCORES_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(len(v) for v in scores.values())
    print(f"Wrote {RERANKER_SCORES_PATH} ({total} scored (query, chunk) pairs, model={settings.ollama_model})")


if __name__ == "__main__":
    main()

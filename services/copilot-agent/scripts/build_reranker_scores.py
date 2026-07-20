"""Regenerate the committed reranker-scores artifact for P3.4
(``app/reranking.py``).

Run against a REACHABLE chat-capable engine to (re)build
``app/data/reranker_scores.json``: for each golden query
(``scripts/retrieval_golden_queries.py``), a real relevance score
(``app.reranking.OllamaRerankScorer`` -- despite the name, it only depends on
an ``.extract()``-shaped client, so it works unchanged against either engine,
see ``RERANKER_ENGINE`` below) for every chunk in its hybrid
top-``RECORD_POOL`` candidate pool (10 -- covers both
`tests/test_reranking.py`'s TOP_K=5 direct calls, a subset of the same
fused ranking, and `app.reranking.retrieve_and_rerank`'s default
``max(k, 10)`` pool), plus its planted lexical distractor
(``scripts/reranker_golden_distractors.py``). This is the ONE place a live
chat call happens for this feature -- the recorded artifact it produces is
what ``tests/test_reranking.py`` replays against
(``app.reranking.RecordedRerankScorer``), mirroring
``scripts/build_retrieval_embeddings.py``'s record/replay discipline for
dense embeddings.

Each entry is keyed by a sha256 of the exact chunk text it was scored from
(``app.retrieval.chunk_text_sha256``), not by chunk id -- see
``app.reranking``'s module docstring "Determinism" for why (mirrors
``app.retrieval.DenseIndex``'s embedding-drift guard).

**Engine selection (issue #99).** ``RERANKER_ENGINE`` picks which chat
engine actually scores the fixture -- ``"ollama"`` (default, backward
compatible) or ``"llama_server"``. This matters because
``app.config.Settings.copilot_llm_engine`` defaults to ``"llama_server"``
(Qwen3-8B-Q5 via llama-server) -- the model that ACTUALLY scores relevance
in production -- while the original fixture was recorded against Ollama's
``qwen3:4b``. Both engines' clients are duck-typed to the same ``.extract()``
signature (see ``app.reranking.OllamaRerankScorer``'s ``_Extractor``
Protocol), so only the client construction below branches; retrieval,
candidate selection, and scoring call sites are identical, keeping the two
runs a fair apples-to-apples comparison over the SAME (query, chunk) pairs.

Usage (against the dev stack, bridged to the host or from inside a
container that can reach the internal compose network):

    # Ollama (qwen3:4b) -- original fixture
    OLLAMA_BASE_URL=http://localhost:11435 python -m scripts.build_reranker_scores

    # llama-server (qwen3-8b) -- production engine (issue #99)
    RERANKER_ENGINE=llama_server LLAMA_SERVER_BASE_URL=http://llama-server:8080 \\
        python -m scripts.build_reranker_scores

``RERANKER_SCORES_OUTPUT_PATH`` overrides the write target (default
``RERANKER_SCORES_PATH``, i.e. ``app/data/reranker_scores.json``) -- used to
record a second model's fixture alongside the first without clobbering it
mid-comparison.

Re-run this and commit the resulting artifact whenever a golden query, its
expected chunk, or its planted distractor changes, when corpus text
affecting any of those chunks changes, or when the engine actually scoring
production reranking changes (issue #99).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.config import Settings
from app.llama_server_client import LlamaServerClient
from app.ollama_client import OllamaClient
from app.reranking import RERANKER_SCORES_PATH, OllamaRerankScorer
from app.retrieval import (
    CORPUS_DIR,
    build_retriever_from_corpus,
    chunk_text_sha256,
    parse_corpus,
    recorded_query_vector,
)
from scripts.reranker_golden_distractors import GOLDEN_DISTRACTORS
from scripts.retrieval_golden_queries import GOLDEN_QUERIES

RECORD_POOL = 10


def _build_scorer_and_model_name() -> tuple[OllamaRerankScorer, str]:
    """Build the (engine-appropriate) scorer and the model name to stamp
    into the artifact's ``"model"`` field -- see ``RERANKER_ENGINE`` in the
    module docstring."""
    engine = os.environ.get("RERANKER_ENGINE", "ollama")
    if engine == "llama_server":
        base_url = os.environ.get("LLAMA_SERVER_BASE_URL", "http://localhost:8080")
        settings = Settings(llama_server_base_url=base_url, llama_server_api_timeout_seconds=120.0)
        client = LlamaServerClient.from_settings(settings)
        return OllamaRerankScorer(client), settings.llama_server_model
    if engine != "ollama":
        raise ValueError(f"Unknown RERANKER_ENGINE {engine!r} -- expected 'ollama' or 'llama_server'")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11435")
    settings = Settings(ollama_base_url=base_url, ollama_api_timeout_seconds=120.0)
    client = OllamaClient.from_settings(settings)
    return OllamaRerankScorer(client), settings.ollama_model


def main() -> None:
    scorer, model_name = _build_scorer_and_model_name()
    output_override = os.environ.get("RERANKER_SCORES_OUTPUT_PATH")
    output_path = Path(output_override) if output_override else RERANKER_SCORES_PATH

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

    payload = {"model": model_name, "scores": scores}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(len(v) for v in scores.values())
    print(f"Wrote {output_path} ({total} scored (query, chunk) pairs, model={model_name})")


if __name__ == "__main__":
    main()

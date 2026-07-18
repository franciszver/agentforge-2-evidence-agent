"""Regenerate the committed dense-embeddings artifact for hybrid retrieval
(P3.3, ``app/retrieval.py``).

Run against a REACHABLE Ollama instance serving ``nomic-embed-text`` to
(re)build ``app/data/retrieval_embeddings.json``: one embedding per corpus
chunk (``app.retrieval.parse_corpus``) plus one per golden query
(``scripts/retrieval_golden_queries.py``). This is the ONE place a live
Ollama embedding call happens for this feature -- the recorded artifact it
produces is what ``tests/test_retrieval.py`` replays against, so the test
suite never needs a live Ollama call to pass (mirrors the record/replay
discipline `docs/W2_ARCHITECTURE.md` "Testing Strategy" already establishes
for VLM/reranker calls).

Usage (against the dev stack's Ollama, bridged to the host):

    OLLAMA_BASE_URL=http://localhost:11435 python -m scripts.build_retrieval_embeddings

Re-run this and commit the resulting ``app/data/retrieval_embeddings.json``
whenever the corpus content changes, a chunk id changes (a heading rename),
or a golden query is added/changed.
"""

from __future__ import annotations

import json
import os

from app.config import Settings
from app.ollama_client import OllamaClient
from app.retrieval import CORPUS_DIR, EMBEDDINGS_PATH, parse_corpus
from scripts.retrieval_golden_queries import GOLDEN_QUERIES


def main() -> None:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11435")
    settings = Settings(ollama_base_url=base_url, ollama_api_timeout_seconds=120.0)
    client = OllamaClient.from_settings(settings)

    chunks = parse_corpus(CORPUS_DIR)
    chunk_vectors = {chunk.chunk_id: client.embed(chunk.text) for chunk in chunks}
    query_vectors = {query: client.embed(query) for query, _expected_chunk_id in GOLDEN_QUERIES}

    payload = {
        "model": settings.ollama_embedding_model,
        "chunks": chunk_vectors,
        "queries": query_vectors,
    }
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {EMBEDDINGS_PATH} "
        f"({len(chunk_vectors)} chunk vectors, {len(query_vectors)} query vectors, model={settings.ollama_embedding_model})"
    )


if __name__ == "__main__":
    main()

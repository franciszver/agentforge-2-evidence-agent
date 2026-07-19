"""Red-first wiring test for P3.10b (epic #52 step 2): migrating dense-vector
embeddings (nomic-embed-text) from Ollama to a dedicated llama.cpp
``llama-server --embedding`` instance, gated by
``Settings.copilot_embed_engine``.

Hard invariants this file exists to pin down:
  * the retriever's embedder defaults to ``LlamaServerEmbedClient`` -- no
    Ollama round trip for embeddings out of the box.
  * ``copilot_embed_engine="ollama"`` still selects ``OllamaClient`` for
    instant rollback.
  * the vision-document-ingestion ``IntakeExtractorWorker`` and the
    reranker's LLM-as-judge client are UNAFFECTED by this flag -- they only
    respond to ``copilot_llm_engine`` (see ``test_chat_llm_engine_routing.py``).

No real network is touched anywhere in this file.
"""

from __future__ import annotations

from app.chat import _build_evidence_workers
from app.config import Settings
from app.llama_server_client import LlamaServerClient
from app.llama_server_embed_client import LlamaServerEmbedClient
from app.ollama_client import OllamaClient


def test_evidence_workers_use_llama_server_embedder_by_default(tmp_path):
    settings = Settings(_env_file=None, copilot_ingestion_base_dir=str(tmp_path))  # type: ignore[call-arg]

    intake_worker, evidence_worker = _build_evidence_workers(settings)

    assert isinstance(evidence_worker._retriever._embedder, LlamaServerEmbedClient)
    # Unaffected by the embed flag: intake extractor stays Ollama (vision,
    # out of scope for #75), reranker follows copilot_llm_engine's own
    # default (llama_server).
    assert isinstance(intake_worker._ollama_client, OllamaClient)
    assert isinstance(evidence_worker._reranker._scorer._client, LlamaServerClient)


def test_evidence_workers_use_ollama_embedder_when_flagged(tmp_path):
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, copilot_embed_engine="ollama", copilot_ingestion_base_dir=str(tmp_path)
    )

    intake_worker, evidence_worker = _build_evidence_workers(settings)

    assert isinstance(evidence_worker._retriever._embedder, OllamaClient)
    assert isinstance(intake_worker._ollama_client, OllamaClient)


def test_embed_engine_flag_is_independent_of_llm_engine_flag(tmp_path):
    """Rolling the answer/extract/reranker roles back to Ollama
    (copilot_llm_engine="ollama") must NOT also silently move embeddings
    back to Ollama -- the two flags are independently rollback-able."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, copilot_llm_engine="ollama", copilot_ingestion_base_dir=str(tmp_path)
    )

    _intake_worker, evidence_worker = _build_evidence_workers(settings)

    assert isinstance(evidence_worker._retriever._embedder, LlamaServerEmbedClient)
    assert isinstance(evidence_worker._reranker._scorer._client, OllamaClient)

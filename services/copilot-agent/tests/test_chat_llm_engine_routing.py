"""Red-first wiring test for P3.10a (epic #52 step 1): migrating the
answer/extract/reranker LLM roles from Ollama to llama-server, gated by
``Settings.copilot_llm_engine``.

Hard invariant this test exists to pin down: vision document-ingestion
extraction MUST stay on Ollama regardless of either engine flag -- only the
planner chat/extract, claim extraction, and LLM-as-reranker roles are
selectable via ``copilot_llm_engine``, and the embedder is separately
selectable via ``copilot_embed_engine`` (P3.10b, epic #52 step 2 -- see
``test_chat_embed_engine_routing.py`` for that flag's dedicated coverage).
No real network is touched anywhere in this file.
"""

from __future__ import annotations

from app.chat import _build_evidence_workers, _default_planner_factory, get_claim_extractor, get_text_llm_client
from app.config import Settings
from app.llama_server_client import LlamaServerClient
from app.ollama_client import OllamaClient
from app.planner import Planner


def test_get_text_llm_client_defaults_to_llama_server():
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    client = get_text_llm_client(settings)

    assert isinstance(client, LlamaServerClient)


def test_get_text_llm_client_selects_ollama_when_flagged():
    settings = Settings(_env_file=None, copilot_llm_engine="ollama")  # type: ignore[call-arg]

    client = get_text_llm_client(settings)

    assert isinstance(client, OllamaClient)


def test_planner_factory_uses_ollama_when_flagged(monkeypatch):
    monkeypatch.setenv("COPILOT_LLM_ENGINE", "ollama")

    factory = _default_planner_factory("some-token")
    planner = factory(1)

    assert isinstance(planner, Planner)
    assert isinstance(planner._ollama, OllamaClient)


def test_planner_factory_uses_llama_server_by_default(monkeypatch):
    monkeypatch.delenv("COPILOT_LLM_ENGINE", raising=False)

    factory = _default_planner_factory("some-token")
    planner = factory(1)

    assert isinstance(planner._ollama, LlamaServerClient)


def test_claim_extractor_uses_ollama_when_flagged(monkeypatch):
    monkeypatch.setenv("COPILOT_LLM_ENGINE", "ollama")

    extractor = get_claim_extractor()

    assert isinstance(extractor._ollama, OllamaClient)


def test_claim_extractor_uses_llama_server_by_default(monkeypatch):
    monkeypatch.delenv("COPILOT_LLM_ENGINE", raising=False)

    extractor = get_claim_extractor()

    assert isinstance(extractor._ollama, LlamaServerClient)


def test_evidence_workers_keep_intake_extractor_on_ollama_even_when_flagged(tmp_path):
    """The most important regression this migration must not introduce: the
    vision-document-ingestion ``IntakeExtractorWorker`` must stay on
    ``OllamaClient`` even when ``copilot_llm_engine`` selects llama-server --
    only the reranker's LLM-as-judge scorer is engine-selectable via that
    flag. (The embedder is covered by ``test_chat_embed_engine_routing.py``,
    P3.10b -- its own, dedicated ``copilot_embed_engine`` flag.)"""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, copilot_llm_engine="llama_server", copilot_ingestion_base_dir=str(tmp_path)
    )

    intake_worker, evidence_worker = _build_evidence_workers(settings)

    assert isinstance(intake_worker._ollama_client, OllamaClient)
    assert isinstance(evidence_worker._reranker._scorer._client, LlamaServerClient)


def test_evidence_workers_use_llama_server_reranker_by_default(tmp_path):
    settings = Settings(_env_file=None, copilot_ingestion_base_dir=str(tmp_path))  # type: ignore[call-arg]

    intake_worker, evidence_worker = _build_evidence_workers(settings)

    assert isinstance(intake_worker._ollama_client, OllamaClient)
    assert isinstance(evidence_worker._reranker._scorer._client, LlamaServerClient)


def test_evidence_workers_use_ollama_reranker_when_flagged(tmp_path):
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, copilot_llm_engine="ollama", copilot_ingestion_base_dir=str(tmp_path)
    )

    intake_worker, evidence_worker = _build_evidence_workers(settings)

    assert isinstance(intake_worker._ollama_client, OllamaClient)
    assert isinstance(evidence_worker._reranker._scorer._client, OllamaClient)

"""Hermetic tests for ``scripts/seed_demo_documents.py`` (issue #204 gate-3
MAJOR finding) -- the sibling of ``test_ingest_demo_pdf_script.py``'s tests.

Same bug, same fix: the script called ``app.ingestion.attach_and_extract``
directly with an ``OllamaClient.from_settings(settings)`` built with no
``model=`` override (resolving the TEXT-ONLY ``settings.ollama_model``) and
never ran the vision-capability guard. The fix dispatches through
``app.supervisor.IntakeExtractorWorker`` instead -- the same worker class
``app.chat._build_evidence_workers`` builds for ``/chat``.

No real Ollama call and no real ``docker compose exec mysql`` shell-out:
``get_pid_for_pubpid`` is monkeypatched (it is host-only tooling, unrelated
to what this test is verifying), and the VLM is
``tests.test_ingestion._FakeVlmOllama``. PDF rendering is real -- the
script's own committed fixture, ``tests/fixtures/lab_report_synthetic.pdf``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.seed_demo_documents as seed_demo_documents
from app.config import Settings
from app.ingestion import LocalIngestionStore
from app.schemas.ingestion import ExtractedLabRow, LabPageExtraction
from tests.test_ingestion import _FakeVlmOllama

_PATIENT_ID = 1
_ROW = ExtractedLabRow(
    test="Hemoglobin A1c",
    value="5.4",
    unit="%",
    reference_range="4.0-5.6",
    collection_date="2026-06-01",
    abnormal_flag="N",
)


class _RecordingOllamaClientFactory:
    """See ``tests.test_ingest_demo_pdf_script`` -- identical double."""

    def __init__(self, results: list[LabPageExtraction]) -> None:
        self.calls: list[str | None] = []
        self.built_clients: list[_FakeVlmOllama] = []
        self._results = results

    def from_settings(self, settings: Settings, *, model: str | None = None) -> _FakeVlmOllama:
        self.calls.append(model)
        resolved_model = model if model is not None else settings.ollama_model
        client = _FakeVlmOllama(list(self._results), model=resolved_model)
        self.built_clients.append(client)
        return client


def _run_seed(monkeypatch: pytest.MonkeyPatch, settings: Settings, factory: _RecordingOllamaClientFactory) -> int:
    monkeypatch.setattr(seed_demo_documents, "get_settings", lambda: settings)
    monkeypatch.setattr(seed_demo_documents, "OllamaClient", factory)
    monkeypatch.setattr(seed_demo_documents, "get_pid_for_pubpid", lambda pubpid: _PATIENT_ID)
    try:
        return seed_demo_documents.seed_demo_documents()
    except seed_demo_documents.DemoDocumentSeedError:
        return -1


def test_seed_demo_documents_uses_the_dedicated_vision_model_not_ollama_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAJOR-1: the seed script must build its VLM client on
    ``settings.copilot_vision_model``, NOT the text-only
    ``settings.ollama_model`` default."""
    settings = Settings(copilot_ingestion_base_dir=str(tmp_path))
    assert settings.copilot_vision_model != settings.ollama_model
    factory = _RecordingOllamaClientFactory([LabPageExtraction(rows=[_ROW]), LabPageExtraction(rows=[_ROW])])

    result = _run_seed(monkeypatch, settings, factory)

    assert result == _PATIENT_ID
    assert factory.calls == [settings.copilot_vision_model]
    assert factory.built_clients[0].model == settings.copilot_vision_model
    assert len(factory.built_clients[0].extract_calls) == 2  # the fixture PDF's 2 pages


def test_seed_demo_documents_refuses_a_non_vision_model_before_any_page_is_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAJOR-1 fail-closed: a misconfigured ``copilot_vision_model`` must
    refuse BEFORE any page reaches the model -- zero ``extract`` calls,
    zero facts/citations persisted, and ``seed_demo_documents`` raises
    ``DemoDocumentSeedError`` (surfaced by ``main()`` as exit code 1) rather
    than silently ingesting through a text-only model."""
    settings = Settings(copilot_ingestion_base_dir=str(tmp_path), copilot_vision_model="qwen3:4b")
    factory = _RecordingOllamaClientFactory([LabPageExtraction(rows=[_ROW]), LabPageExtraction(rows=[_ROW])])

    result = _run_seed(monkeypatch, settings, factory)

    assert result == -1  # DemoDocumentSeedError raised
    assert len(factory.built_clients) == 1
    assert factory.built_clients[0].extract_calls == []
    store = LocalIngestionStore(settings.copilot_ingestion_base_dir)
    assert store.list_citations_for_patient(_PATIENT_ID) == []

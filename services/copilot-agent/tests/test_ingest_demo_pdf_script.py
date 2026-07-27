"""Hermetic tests for ``scripts/ingest_demo_pdf.py`` (issue #204 gate-3
MAJOR finding).

The gate-3 review found that ``IntakeExtractorWorker`` (the vision-model
default + fail-closed guard #204 built) was never actually wired into this
script -- it called ``app.ingestion.attach_and_extract`` directly with an
``OllamaClient.from_settings(settings)`` built with NO ``model=`` override,
so it silently resolved ``settings.ollama_model`` (the TEXT-ONLY
``qwen3:4b`` rollback default), not ``settings.copilot_vision_model``, and
never ran the vision-capability guard at all. This is #204's own measured
symptom (2/2 pages 404, 0 facts, no exception) reproducing unchanged on
this live ingestion path.

These tests pin the fix: the script now dispatches through
``app.supervisor.IntakeExtractorWorker`` -- the SAME worker class
``app.chat._build_evidence_workers`` builds for ``/chat`` -- so the vision
model and its guard are decided in exactly one place.

No real Ollama call: the VLM is ``tests.test_ingestion._FakeVlmOllama``,
the existing scripted double (its ``.model`` attribute is exactly what
``IntakeExtractorWorker.run`` reads to fail closed). PDF rendering is real
(the ``fixture_pdf`` fixture from ``conftest.py``, a real one-page PDF).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.ingest_demo_pdf as ingest_demo_pdf
from app.config import Settings
from app.ingestion import LocalIngestionStore
from app.schemas.ingestion import ExtractedLabRow, LabPageExtraction
from tests.test_ingestion import _FakeVlmOllama

_ROW = ExtractedLabRow(
    test="Hemoglobin A1c",
    value="5.4",
    unit="%",
    reference_range="4.0-5.6",
    collection_date="2026-06-01",
    abnormal_flag="N",
)


class _RecordingOllamaClientFactory:
    """Stands in for ``app.ollama_client.OllamaClient``: records every
    ``model=`` kwarg ``from_settings`` was called with, and hands back a
    scripted ``_FakeVlmOllama`` (never a real network call) whose
    ``.model`` is whatever this construction resolved to -- exactly what
    ``IntakeExtractorWorker.run`` reads to decide fail-open/fail-closed."""

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


def _run_main(monkeypatch: pytest.MonkeyPatch, settings: Settings, factory: _RecordingOllamaClientFactory, pdf_path: Path) -> int:
    monkeypatch.setattr(ingest_demo_pdf, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_demo_pdf, "OllamaClient", factory)
    monkeypatch.setattr(sys, "argv", ["ingest_demo_pdf.py", "1", str(pdf_path)])
    return ingest_demo_pdf.main()


def test_ingest_demo_pdf_uses_the_dedicated_vision_model_not_ollama_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_pdf: Path
) -> None:
    """MAJOR-1: the script must build its VLM client on
    ``settings.copilot_vision_model`` (default ``qwen2.5vl:7b``), NOT the
    text-only ``settings.ollama_model`` default -- the exact #204 bug this
    script previously reproduced unchanged."""
    settings = Settings(copilot_ingestion_base_dir=str(tmp_path))
    assert settings.copilot_vision_model != settings.ollama_model  # sanity: the two roles differ
    factory = _RecordingOllamaClientFactory([LabPageExtraction(rows=[_ROW])])

    rc = _run_main(monkeypatch, settings, factory, fixture_pdf)

    assert rc == 0
    assert factory.calls == [settings.copilot_vision_model]
    assert factory.built_clients[0].model == settings.copilot_vision_model
    # The page really was processed through the vision client.
    assert len(factory.built_clients[0].extract_calls) == 1


def test_ingest_demo_pdf_exits_nonzero_with_a_distinct_message_on_total_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #206: a TOTAL extraction failure (every page failed) now raises
    ``IngestionError`` from ``attach_and_extract`` instead of returning a
    zero-facts result -- the script must catch it and exit non-zero with a
    message DISTINGUISHABLE from the partial-failure case's ("lab PDF
    ingestion had failed pages ..."). Asserting on the message text (not
    just the exit code) is load-bearing here: before this fix, a total
    failure ALSO exits 1 via the pre-existing ``if result.failed_pages``
    branch, so the exit code alone cannot tell the two apart."""
    settings = Settings(copilot_ingestion_base_dir=str(tmp_path))

    class _AlwaysFailingVlm(_FakeVlmOllama):
        def __init__(self) -> None:
            super().__init__(error=True, model=settings.copilot_vision_model)

    class _FailingFactory(_RecordingOllamaClientFactory):
        def from_settings(self, settings: Settings, *, model: str | None = None) -> _FakeVlmOllama:
            self.calls.append(model)
            client = _AlwaysFailingVlm()
            self.built_clients.append(client)
            return client

    failing_factory = _FailingFactory([])
    rc = _run_main(monkeypatch, settings, failing_factory, fixture_pdf)

    assert rc == 1
    stderr = capsys.readouterr().err
    assert "FAILED ENTIRELY" in stderr
    assert "had failed pages" not in stderr  # the distinct, partial-failure-only message
    store = LocalIngestionStore(settings.copilot_ingestion_base_dir)
    assert store.list_citations_for_patient(1) == []


def test_ingest_demo_pdf_refuses_a_non_vision_model_before_any_page_is_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_pdf: Path
) -> None:
    """MAJOR-1 fail-closed: a misconfigured ``copilot_vision_model`` that
    fails the name-based vision-capability check must refuse BEFORE any
    page reaches the model -- zero ``extract`` calls, zero facts/citations
    persisted -- and exit non-zero, not silently produce text-model output
    on page images it cannot read."""
    settings = Settings(copilot_ingestion_base_dir=str(tmp_path), copilot_vision_model="qwen3:4b")
    factory = _RecordingOllamaClientFactory([LabPageExtraction(rows=[_ROW])])

    rc = _run_main(monkeypatch, settings, factory, fixture_pdf)

    assert rc == 1
    assert len(factory.built_clients) == 1
    assert factory.built_clients[0].extract_calls == []  # zero pages sent to the model
    store = LocalIngestionStore(settings.copilot_ingestion_base_dir)
    assert store.list_citations_for_patient(1) == []  # nothing persisted

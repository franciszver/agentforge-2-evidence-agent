"""Red-first integration test for issue #86: the chat planner has NO path to
facts ingested from documents (``LocalIngestionStore``) while composing its
answer -- ``app.chat.get_patient_fact_provider`` is wired in only AFTER
``Planner.run``/``run_streaming`` returns, feeding solely the post-hoc
claim-extraction/verification step (P3.9a, issue #46, see
``test_chat_fact_integration.py``). Asking about an ingested lab report
therefore gets answered from EMR chart tools alone, which genuinely have no
data for it -- "no lab results recorded for this patient" -- even though the
document was actually ingested. ``docs/DEMO_SCRIPT.md`` beat 3 is this
exact live failure.

This test proves the wiring gap directly: a recording ``_FakePlanner``
captures whatever ``document_facts`` argument it is called with, and both
cases below assert it received this patient's REAL, already-ingested facts
(via a REAL ``LocalIngestionStore`` + ``attach_and_extract`` call, scripted
VLM -- no live Ollama needed for ingestion) BEFORE the answer was composed,
not just for verification afterward.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.chat import (
    _no_op_support_judge_provider,
    get_claim_extractor,
    get_patient_fact_provider,
    get_planner_factory,
    get_support_judge_provider,
    get_token_validator,
    get_trace_store,
)
from app.ingestion import LocalIngestionStore, attach_and_extract
from app.main import app
from app.planner import PlannerResult
from app.schemas.ingestion import ExtractedLabRow
from app.trace_store import TraceStore
from tests.conftest import FakeVlm

_TEST_HASH_KEY = "0" * 32
_PATIENT_ID = 303

_A1C_ROW = ExtractedLabRow(
    test="Hemoglobin A1c",
    value="5.4",
    unit="%",
    reference_range="4.0-5.6",
    collection_date="2026-06-01",
    abnormal_flag="N",
)

# Mirrors tests/test_ingestion.py's deliberately-unreadable-field fixture:
# value is legible, collection_date is not -- app.ingestion._quote_for_row
# never mentions a date for this row, so its citation quote is honestly
# "Creatinine: 0.9", nothing more.
_CREATININE_ROW_MISSING_DATE = ExtractedLabRow(
    test="Creatinine",
    value="0.9",
    unit="mg/dL",
    reference_range="0.6-1.3",
    collection_date=None,
    abnormal_flag="N",
)


class _RecordingPlanner:
    """Records the ``document_facts`` argument it is called with, on both
    ``run`` and ``run_streaming`` -- so a test can assert the planner
    actually received the patient's ingested facts BEFORE composing its
    answer, not merely that verification later cited them correctly."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.received_document_facts: object = "NEVER CALLED"

    def run(self, question: str, guideline_excerpts: object = None, document_facts: object = None) -> PlannerResult:
        self.received_document_facts = document_facts
        return PlannerResult(answer=self._answer, trace=[], raw_results=[])


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


def _chat(message: str, patient_id: int):
    return client.post(
        "/chat",
        json={"message": message, "patient_id": patient_id},
        headers={"Authorization": "Bearer good-token"},
    )


def _run_chat_with_ingested_document_fact(tmp_path, fixture_pdf, row: ExtractedLabRow, question: str):
    """Wire a real ingested document fact (via a real ``LocalIngestionStore``
    + ``attach_and_extract`` call, scripted VLM) and a recording planner
    double into a real ``/chat`` turn, so each test only supplies its own
    row + question + assertions."""
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    store = LocalIngestionStore(base_dir=tmp_path / "ingestion")
    attach_and_extract(
        _PATIENT_ID,
        fixture_pdf,
        "lab_pdf",
        ollama_client=FakeVlm(row),
        document_store=store,
        fact_store=store,
    )

    planner = _RecordingPlanner("placeholder answer")

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: planner)
    app.dependency_overrides[get_claim_extractor] = lambda: _NoOpExtractor()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    app.dependency_overrides[get_patient_fact_provider] = lambda: store.list_citations_for_patient
    app.dependency_overrides[get_support_judge_provider] = lambda: _no_op_support_judge_provider

    return _chat(question, _PATIENT_ID), planner


def test_planner_receives_ingested_document_facts_before_composing_answer(tmp_path, fixture_pdf):
    """The planner must be handed this patient's ingested lab fact BEFORE it
    composes its answer -- proving the pre-answer wiring gap (issue #86) is
    closed, not just the post-hoc citation-verification path (issue #46,
    already shipped)."""
    response, planner = _run_chat_with_ingested_document_fact(tmp_path, fixture_pdf, _A1C_ROW, "What is her A1c?")

    assert response.status_code == 200
    assert planner.received_document_facts != "NEVER CALLED", (
        "the planner's run() must be called with a document_facts argument at all"
    )
    facts = list(planner.received_document_facts or [])
    assert facts, "the planner must receive this patient's ingested document facts before answering"
    (fact,) = facts
    assert fact.quote_or_value == "Hemoglobin A1c: 5.4"


def test_redacted_field_reaches_planner_as_honest_quote_never_a_fabricated_date(tmp_path, fixture_pdf):
    """The redacted/unreadable collection-date field must reach the planner
    as its literal, field-omitting quote ("Creatinine: 0.9") -- never a
    guessed date -- so the planner can only ever answer honestly ("not
    recorded in the document") instead of fabricating one."""
    response, planner = _run_chat_with_ingested_document_fact(
        tmp_path,
        fixture_pdf,
        _CREATININE_ROW_MISSING_DATE,
        "What was the collection date for his creatinine result?",
    )

    assert response.status_code == 200
    facts = list(planner.received_document_facts or [])
    (fact,) = facts
    assert fact.quote_or_value == "Creatinine: 0.9"
    assert "2026" not in fact.quote_or_value


class _NoOpExtractor:
    def extract_claims(self, *, answer, tools, raw_results, retrieved_chunks=(), patient_facts=()):
        return []

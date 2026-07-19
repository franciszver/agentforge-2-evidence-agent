"""Red-first integration test for P3.9a (issue #46): wiring per-patient
lab/intake-form fact citations (``DocumentFactIndex``) into a REAL
``POST /chat`` turn.

Mirrors ``tests/test_chat_evidence_integration.py``'s structure (the P3.9
guideline-corpus wiring test) for the fact-citation half P3.9 explicitly
deferred: the planner and claim extractor are scripted doubles (no live
Ollama call), but the fact-lookup path exercises the REAL
``LocalIngestionStore.list_citations_for_patient`` over facts a REAL
``attach_and_extract`` call persisted to a ``tmp_path``-backed store -- so
"the patient's own ingested lab fact reaches /chat as a verified
DocumentCitation" is genuinely exercised, not just asserted from a canned
double.

**The load-bearing security case** (the reason this issue exists): a /chat
turn for patient A must NEVER surface patient B's fact as a citation.
``test_cross_patient_fact_is_never_surfaced_or_verifiable_from_the_other_patients_chat_turn``
seeds both patients into the SAME on-disk store and proves patient A's turn
only ever sees patient A's citation.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.chat import (
    ChatEvent,
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
from app.schemas.ingestion import DocumentCitation, ExtractedLabRow, LabPageExtraction
from app.schemas.verification import Claim
from app.trace_store import TraceStore

_TEST_HASH_KEY = "0" * 32

_PATIENT_A_ID = 101
_PATIENT_B_ID = 202

_LAB_ROW = ExtractedLabRow(
    test="Hemoglobin A1c",
    value="5.4",
    unit="%",
    reference_range="4.0-5.6",
    collection_date="2026-06-01",
    abnormal_flag="N",
)
_EXPECTED_FIELD_ID = "Hemoglobin A1c#page1-row0"
_EXPECTED_QUOTE = "Hemoglobin A1c: 5.4"


class _FakeVlm:
    """Scripted single-page lab VLM double -- no live Ollama call."""

    def __init__(self) -> None:
        self.extract_calls: list[object] = []

    def extract(self, prompt_or_messages, schema, *, options=None, images=None):
        self.extract_calls.append(prompt_or_messages)
        return LabPageExtraction(rows=[_LAB_ROW])


class _FakePlanner:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def run(self, question: str) -> PlannerResult:
        return PlannerResult(answer=self._answer, trace=[], raw_results=[])


def _make_fact_citing_extractor(source_id: str):
    """A claim extractor double that cites the given ``source_id``'s A1c fact
    -- proves the wiring reaches ``check_document_citation`` against a REAL
    ``DocumentFactIndex`` built from that patient's own stored facts, without
    needing a live extraction-model call."""

    class _FakeExtractor:
        def extract_claims(self, *, answer, tools, raw_results, retrieved_chunks=(), patient_facts=()) -> list[Claim]:
            return [
                Claim(
                    text="Her A1c is 5.4%.",
                    document_citations=[
                        DocumentCitation(
                            source_type="lab_pdf",
                            source_id=source_id,
                            page_or_section="page 1",
                            field_or_chunk_id=_EXPECTED_FIELD_ID,
                            quote_or_value=_EXPECTED_QUOTE,
                        )
                    ],
                )
            ]

    return _FakeExtractor()


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


def _iter_sse_events(text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        events.append((event_name, "\n".join(data_lines)))
    return events


def _chat(message: str, patient_id: int):
    return client.post(
        "/chat",
        json={"message": message, "patient_id": patient_id},
        headers={"Authorization": "Bearer good-token"},
    )


def test_real_chat_turn_cites_a_verified_lab_fact_document_citation(tmp_path):
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    store = LocalIngestionStore(base_dir=tmp_path / "ingestion")
    ingestion = attach_and_extract(
        _PATIENT_A_ID,
        _fixture_pdf(tmp_path),
        "lab_pdf",
        ollama_client=_FakeVlm(),
        document_store=store,
        fact_store=store,
    )

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (
        lambda patient_id: _FakePlanner("Her A1c is 5.4%.")
    )
    app.dependency_overrides[get_claim_extractor] = lambda: _make_fact_citing_extractor(ingestion.source_id)
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    app.dependency_overrides[get_patient_fact_provider] = lambda: store.list_citations_for_patient
    app.dependency_overrides[get_support_judge_provider] = lambda: _no_op_support_judge_provider

    response = _chat("What is her A1c?", _PATIENT_A_ID)

    assert response.status_code == 200
    events = _iter_sse_events(response.text)
    verification_data = next(data for name, data in events if name == ChatEvent.VERIFICATION.value)

    assert '"verdict":"verified"' in verification_data.replace(" ", "")
    payload = json.loads(verification_data)
    (segment,) = payload["segments"]
    assert segment["type"] == "claim"
    (citation,) = segment["document_citations"]
    assert citation["source_type"] == "lab_pdf"
    assert citation["source_id"] == ingestion.source_id
    assert citation["field_or_chunk_id"] == _EXPECTED_FIELD_ID


def test_cross_patient_fact_is_never_surfaced_or_verifiable_from_the_other_patients_chat_turn(tmp_path, caplog):
    """The load-bearing security case (issue #46): patient A's document
    citation must fail closed (never verify) when the /chat turn is bound to
    patient B -- the fact-lookup scoped to patient B's OWN stored facts never
    even contains patient A's source_id, so the citation is UNKNOWN_SOURCE,
    not a resolved-but-wrong-patient value."""
    caplog.set_level(logging.INFO)
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    store = LocalIngestionStore(base_dir=tmp_path / "ingestion")
    fixture = _fixture_pdf(tmp_path)

    ingestion_a = attach_and_extract(
        _PATIENT_A_ID, fixture, "lab_pdf", ollama_client=_FakeVlm(), document_store=store, fact_store=store
    )
    # Patient B also has their own ingested lab fact, under a DIFFERENT
    # source_id -- proves this isn't merely "no data ingested for B" but a
    # genuine cross-patient isolation check.
    attach_and_extract(
        _PATIENT_B_ID, fixture, "lab_pdf", ollama_client=_FakeVlm(), document_store=store, fact_store=store
    )

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (
        lambda patient_id: _FakePlanner("Her A1c is 5.4%.")
    )
    # The extractor (mis)cites patient A's source_id even though this turn is
    # bound to patient B -- simulating the worst case (a steered/hallucinated
    # extraction attempting to cite another patient's fact).
    app.dependency_overrides[get_claim_extractor] = lambda: _make_fact_citing_extractor(ingestion_a.source_id)
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    app.dependency_overrides[get_patient_fact_provider] = lambda: store.list_citations_for_patient
    app.dependency_overrides[get_support_judge_provider] = lambda: _no_op_support_judge_provider

    response = _chat("What is her A1c?", _PATIENT_B_ID)

    assert response.status_code == 200
    events = _iter_sse_events(response.text)
    verification_data = next(data for name, data in events if name == ChatEvent.VERIFICATION.value)
    payload = json.loads(verification_data)

    # Fails closed: the claim citing patient A's fact is BLOCKED/stripped when
    # the turn is bound to patient B, never silently "verified".
    assert payload["verdict"] != "verified"
    for segment in payload["segments"]:
        assert segment["type"] != "claim"

    # Patient A's source_id must never appear anywhere in the response body
    # of patient B's turn.
    assert ingestion_a.source_id not in response.text


def _fixture_pdf(tmp_path):
    import pypdfium2 as pdfium

    path = tmp_path / "lab.pdf"
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200.0, 200.0)
    pdf.save(str(path))
    pdf.close()
    return path

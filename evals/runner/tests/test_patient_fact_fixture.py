"""Red-first tests for issue #70: ``PatientFactFixture`` + the ``EvalCase
.patient_facts`` field (``runner.schema``).

Mirrors ``RetrievedChunkFixture``'s own shape/validation style (see
``runner.schema`` module docstring) -- these prove the fixture model itself
(field names, ``to_citation`` mapping, ``extra="forbid"``) and the case
schema's backward-compatible default, independent of any pipeline wiring
(covered separately by ``test_patient_fact_wiring.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.ingestion import Citation

from runner.loader import load_case
from runner.schema import EvalCase, PatientFactFixture

_FIXTURES_CASES_DIR = Path(__file__).parent / "fixtures" / "cases"


def test_patient_fact_fixture_maps_to_citation_verbatim() -> None:
    fixture = PatientFactFixture(
        source_type="lab_pdf",
        source_id="lab-doc-1",
        page_or_section="page 2",
        field_or_chunk_id="lab-doc-1#page-2-row-0",
        quote_or_value="Creatinine: 0.9",
    )

    citation = fixture.to_citation()

    assert isinstance(citation, Citation)
    assert citation.source_type == "lab_pdf"
    assert citation.source_id == "lab-doc-1"
    assert citation.page_or_section == "page 2"
    assert citation.field_or_chunk_id == "lab-doc-1#page-2-row-0"
    assert citation.quote_or_value == "Creatinine: 0.9"


def test_patient_fact_fixture_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        PatientFactFixture(
            source_type="lab_pdf",
            source_id="lab-doc-1",
            page_or_section="page 2",
            field_or_chunk_id="lab-doc-1#page-2-row-0",
            quote_or_value="Creatinine: 0.9",
            unexpected="nope",  # type: ignore[call-arg]
        )


def test_patient_fact_fixture_rejects_unknown_source_type() -> None:
    with pytest.raises(ValidationError):
        PatientFactFixture(
            source_type="guideline_chunk",  # type: ignore[arg-type]
            source_id="lab-doc-1",
            page_or_section="page 2",
            field_or_chunk_id="lab-doc-1#page-2-row-0",
            quote_or_value="Creatinine: 0.9",
        )


def test_eval_case_patient_facts_defaults_to_empty_list() -> None:
    """Backward compatibility: every case YAML predating issue #70 has no
    ``patient_facts`` key at all -- it must load unchanged."""
    case = load_case(_FIXTURES_CASES_DIR / "pass.yaml")
    assert case.patient_facts == []


def test_eval_case_accepts_patient_facts_field() -> None:
    case = EvalCase(
        id="patient-facts-smoke",
        category="citation_present",
        failure_mode="smoke test for the patient_facts field",
        question="What was her last creatinine?",
        patient_id=1,
        patient_facts=[
            {
                "source_type": "lab_pdf",
                "source_id": "lab-doc-1",
                "page_or_section": "page 2",
                "field_or_chunk_id": "lab-doc-1#page-2-row-0",
                "quote_or_value": "Creatinine: 0.9",
            }
        ],
        assertions=[{"type": "answer_contains", "phrases": ["0.9"]}],
    )

    assert len(case.patient_facts) == 1
    assert case.patient_facts[0].quote_or_value == "Creatinine: 0.9"


def test_eval_case_rejects_malformed_patient_fact_entry() -> None:
    with pytest.raises(ValidationError):
        EvalCase(
            id="patient-facts-malformed",
            category="citation_present",
            failure_mode="malformed patient_facts entry must fail validation",
            question="What was her last creatinine?",
            patient_id=1,
            patient_facts=[{"source_type": "not_a_real_source_type", "source_id": "x"}],
            assertions=[{"type": "answer_contains", "phrases": ["x"]}],
        )

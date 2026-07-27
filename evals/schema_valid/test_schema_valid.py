"""Golden eval set (P3G.1), category ``schema_valid``.

Boolean rubric: document extraction (``LabResultFact``/``IntakeFormFact``)
returns schema-valid facts -- every not-found field correctly ``None``
(never fabricated), every fact carries a well-formed ``Citation``/
``DocumentCitation`` -- and the citation contract's own schema guards
(``app.schemas.ingestion``, ``app.schemas.verification``) hold at the
object-construction boundary, not just in the extraction pipeline that
produces them.

Fully deterministic/hermetic: real PDF rendering (``pypdfium2``) over the
committed synthetic fixtures already used by
``services/copilot-agent/tests/test_ingestion.py``, a SCRIPTED VLM double
(never a live Ollama call) standing in for extraction -- no live model
inference, nothing to record/replay. This is a curated, representative
golden slice (one case per rubric facet), not a duplicate of that module's
exhaustive unit coverage -- each case here combines several of that
module's individually-pinned assertions into one named rubric check.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_AGENT_ROOT = Path(__file__).resolve().parents[2] / "services" / "copilot-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from app.ingestion import IngestionError, LocalIngestionStore, attach_and_extract  # noqa: E402
from app.ollama_client import OllamaError  # noqa: E402
from app.schemas.ingestion import (  # noqa: E402
    Citation,
    DocumentCitation,
    ExtractedLabRow,
    IntakeFormExtraction,
    IntakeFormFact,
    LabFlagCode,
    LabPageExtraction,
    LabResultFact,
)
from app.schemas.verification import Claim  # noqa: E402
from pydantic import ValidationError  # noqa: E402

pytestmark = pytest.mark.schema_valid

_FIXTURES_DIR = _AGENT_ROOT / "tests" / "fixtures"
_LAB_FIXTURE_PATH = _FIXTURES_DIR / "lab_report_synthetic.pdf"
_INTAKE_FIXTURE_PATH = _FIXTURES_DIR / "intake_form_synthetic.pdf"

# Mirrors tests/test_ingestion.py's own fixture content exactly (Creatinine's
# collection_date on page 2 is the deliberately unreadable field).
_LAB_PAGE_1_ROWS = [
    ExtractedLabRow(test="Hemoglobin A1c", value="5.4", unit="%", reference_range="4.0-5.6", collection_date="2026-06-01", abnormal_flag="N"),
    ExtractedLabRow(test="Fasting Glucose", value="142", unit="mg/dL", reference_range="70-99", collection_date="2026-06-01", abnormal_flag="H"),
    ExtractedLabRow(test="Total Cholesterol", value="188", unit="mg/dL", reference_range="<200", collection_date="2026-06-01", abnormal_flag="N"),
    ExtractedLabRow(test="LDL Cholesterol", value="112", unit="mg/dL", reference_range="<100", collection_date="2026-06-01", abnormal_flag="H"),
]
_LAB_PAGE_2_ROWS = [
    ExtractedLabRow(test="HDL Cholesterol", value="58", unit="mg/dL", reference_range="40-60", collection_date="2026-06-01", abnormal_flag="N"),
    ExtractedLabRow(test="Triglycerides", value="97", unit="mg/dL", reference_range="<150", collection_date="2026-06-01", abnormal_flag="N"),
    ExtractedLabRow(test="TSH", value="2.1", unit="mIU/L", reference_range="0.4-4.0", collection_date="2026-06-01", abnormal_flag="N"),
    ExtractedLabRow(test="Creatinine", value="0.9", unit="mg/dL", reference_range="0.6-1.3", collection_date=None, abnormal_flag="N"),
]

# Mirrors tests/test_ingestion.py's intake fixture content (page 1's DOB is
# the deliberately unreadable field).
_INTAKE_PAGE_1 = IntakeFormExtraction(
    demographics={"name": "Test Patient", "dob": None, "sex": "F", "mrn": "TEST-000000"},
    chief_concern="Intermittent chest tightness for the past 3 days.",
    medications=[],
    allergies=[],
    family_history=[],
)
_INTAKE_PAGE_2 = IntakeFormExtraction(
    demographics={},
    chief_concern=None,
    medications=["Lisinopril 10mg daily", "Metformin 500mg twice daily", "Atorvastatin 20mg nightly"],
    allergies=["Penicillin", "Shellfish"],
    family_history=["Father: hypertension", "Mother: type 2 diabetes"],
)


class _ScriptedVlm:
    """Scripted extraction double: returns one canned result per call, in
    order -- never a live Ollama call. Mirrors
    ``tests/test_ingestion.py``'s ``_FakeVlmOllama``."""

    def __init__(self, results: list[Any]) -> None:
        self._results = results
        self.extract_calls: list[Any] = []

    def extract(
        self, prompt_or_messages: Any, schema: type, *, options: Any = None, images: list[str] | None = None
    ) -> Any:
        self.extract_calls.append(schema)
        return self._results[len(self.extract_calls) - 1]


@pytest.fixture
def store(tmp_path: Path) -> LocalIngestionStore:
    return LocalIngestionStore(base_dir=tmp_path / "ingestion")


# ---------------------------------------------------------------------------
# case: schema-valid-lab-facts-with-citations
# ---------------------------------------------------------------------------


def test_schema_valid_lab_facts_carry_wellformed_citations(store: LocalIngestionStore) -> None:
    vlm = _ScriptedVlm([LabPageExtraction(rows=_LAB_PAGE_1_ROWS), LabPageExtraction(rows=_LAB_PAGE_2_ROWS)])

    result = attach_and_extract(
        1, _LAB_FIXTURE_PATH, "lab_pdf", ollama_client=vlm, document_store=store, fact_store=store
    )

    assert len(result.facts) == 8
    for fact in result.facts:
        assert isinstance(fact, LabResultFact)
        assert isinstance(fact.citation, Citation)
        assert fact.citation.source_type == "lab_pdf"
        assert fact.citation.source_id == result.source_id
        assert fact.citation.field_or_chunk_id  # non-blank
        assert fact.citation.quote_or_value  # non-blank


# ---------------------------------------------------------------------------
# case: schema-valid-lab-unreadable-field-is-none-never-fabricated
# ---------------------------------------------------------------------------


def test_schema_valid_lab_unreadable_field_is_none_never_fabricated(store: LocalIngestionStore) -> None:
    vlm = _ScriptedVlm([LabPageExtraction(rows=_LAB_PAGE_1_ROWS), LabPageExtraction(rows=_LAB_PAGE_2_ROWS)])

    result = attach_and_extract(
        1, _LAB_FIXTURE_PATH, "lab_pdf", ollama_client=vlm, document_store=store, fact_store=store
    )

    creatinine = next(f for f in result.facts if f.test == "Creatinine")
    assert creatinine.collection_date is None
    # Every other legible field on the same row is NOT None -- the None is
    # specific to the one unreadable field, not a whole-row failure.
    assert creatinine.value == "0.9"
    assert creatinine.unit == "mg/dL"
    assert creatinine.abnormal_flag == LabFlagCode.NORMAL
    # The citation quote reflects what WAS read, never claims a date it
    # could not read.
    assert "0.9" in creatinine.citation.quote_or_value
    assert "2026" not in creatinine.citation.quote_or_value


# ---------------------------------------------------------------------------
# case: schema-valid-intake-facts-with-citations
# ---------------------------------------------------------------------------


def test_schema_valid_intake_facts_carry_wellformed_citations(store: LocalIngestionStore) -> None:
    vlm = _ScriptedVlm([_INTAKE_PAGE_1, _INTAKE_PAGE_2])

    result = attach_and_extract(
        1, _INTAKE_FIXTURE_PATH, "intake_form", ollama_client=vlm, document_store=store, fact_store=store
    )

    assert len(result.facts) == 2
    for fact in result.facts:
        assert isinstance(fact, IntakeFormFact)
        assert isinstance(fact.citation, Citation)
        assert fact.citation.source_type == "intake_form"
        assert fact.citation.source_id == result.source_id


# ---------------------------------------------------------------------------
# case: schema-valid-intake-not-found-fields-are-none-or-empty-never-fabricated
# ---------------------------------------------------------------------------


def test_schema_valid_intake_not_found_fields_are_none_or_empty_never_fabricated(
    store: LocalIngestionStore,
) -> None:
    vlm = _ScriptedVlm([_INTAKE_PAGE_1, _INTAKE_PAGE_2])

    result = attach_and_extract(
        1, _INTAKE_FIXTURE_PATH, "intake_form", ollama_client=vlm, document_store=store, fact_store=store
    )

    page_1 = next(f for f in result.facts if f.citation.page_or_section == "page 1")
    page_2 = next(f for f in result.facts if f.citation.page_or_section == "page 2")

    # Page 1: DOB unreadable -> None, never guessed; the legible fields on
    # the SAME page are not blanked out alongside it.
    assert page_1.demographics["dob"] is None
    assert page_1.demographics["name"] == "Test Patient"
    assert page_1.medications == []  # nothing legible on this page -> empty, not fabricated
    assert page_1.allergies == []

    # Page 2: no demographics/chief_concern section on this page -> empty/None,
    # never guessed from page 1.
    assert page_2.demographics == {}
    assert page_2.chief_concern is None
    assert page_2.medications == [
        "Lisinopril 10mg daily",
        "Metformin 500mg twice daily",
        "Atorvastatin 20mg nightly",
    ]


# ---------------------------------------------------------------------------
# case: schema-valid-ingestion-failure-yields-no-facts-never-guessed
# ---------------------------------------------------------------------------


def test_schema_valid_all_pages_failing_raises_never_returns_zero_facts_normally(store: LocalIngestionStore) -> None:
    # Issue #206: a total extraction failure (every page failed) must raise
    # IngestionError, not return normally with facts == [] -- that shape was
    # indistinguishable from a legitimately empty document.
    class _FailingVlm:
        def extract(
            self, prompt_or_messages: Any, schema: type, *, options: Any = None, images: list[str] | None = None
        ) -> Any:
            raise OllamaError("scripted VLM extraction failure")

    with pytest.raises(IngestionError) as exc_info:
        attach_and_extract(
            1, _LAB_FIXTURE_PATH, "lab_pdf", ollama_client=_FailingVlm(), document_store=store, fact_store=store
        )

    assert exc_info.value.pages_total == 2
    assert exc_info.value.failed_pages == [1, 2]


def test_schema_valid_malformed_pdf_raises_and_persists_nothing(tmp_path: Path) -> None:
    store = LocalIngestionStore(base_dir=tmp_path / "ingestion")
    malformed = tmp_path / "malformed.pdf"
    malformed.write_bytes(b"this is not a pdf")
    vlm = _ScriptedVlm([])

    with pytest.raises(IngestionError):
        attach_and_extract(1, malformed, "lab_pdf", ollama_client=vlm, document_store=store, fact_store=store)

    assert list((tmp_path / "ingestion" / "documents").iterdir()) == []
    assert list((tmp_path / "ingestion" / "facts").iterdir()) == []


# ---------------------------------------------------------------------------
# case: schema-valid-document-citation-schema-guards (object-construction
# boundary, not the extraction pipeline -- the citation contract itself)
# ---------------------------------------------------------------------------


def test_schema_valid_document_citation_rejects_blank_quote() -> None:
    with pytest.raises(ValidationError):
        DocumentCitation(
            source_type="guideline_chunk",
            source_id="a1c-targets",
            page_or_section="Target Ranges",
            field_or_chunk_id="a1c-targets#target-ranges",
            quote_or_value="   ",
        )


def test_schema_valid_document_citation_rejects_unknown_source_type() -> None:
    with pytest.raises(ValidationError):
        DocumentCitation.model_validate(
            {
                "source_type": "carrier_pigeon",
                "source_id": "doc-1",
                "page_or_section": "page 1",
                "field_or_chunk_id": "Glucose",
                "quote_or_value": "Glucose: 105 mg/dL",
            }
        )


def test_schema_valid_claim_allows_zero_citations_at_construction() -> None:
    # (issue #93, Option C) A zero-citation Claim no longer fails schema
    # validation at construction/parse time -- see
    # app.schemas.verification's module docstring. The "claim needs >=1
    # citation to be considered valid" bar is unchanged, but enforcement
    # moved to app.verification.check_claim/render_answer, scoped to just
    # the one offending claim rather than the whole VerifiedAnswer, so one
    # uncitable claim can no longer discard co-occurring valid claims.
    claim = Claim(text="Her A1c is 5.4%.", source_refs=[], document_citations=[])

    assert claim.has_citation is False

"""Validation tests for the claim-level verification response contract (P3.1).

``Claim``/``VerifiedAnswer`` (``app.schemas.verification``) are the contract
the verification layer produces: every factual claim is MEANT to carry >=1
``SourceRef``/``DocumentCitation``, but (issue #93, Option C) a claim with
zero citations no longer fails schema validation -- it parses successfully,
so it doesn't drag down co-occurring claims in the same
``VerifiedAnswer.claims`` list (Pydantic validates a list of sub-models
all-or-nothing). The citation bar itself is unchanged; it's enforced one
layer down by ``app.verification.check_claim``/``render_answer`` instead of
here -- see ``app.schemas.verification``'s module docstring for the full
rationale. This is distinct from ``app.schemas.planner.FinalAnswer``, the raw
two-call extraction output (P2.9) -- see that same module docstring for why
the two schemas are kept separate.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.common import SourceRef
from app.schemas.ingestion import DocumentCitation
from app.schemas.verification import Claim, VerifiedAnswer

# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


def test_claim_with_one_ref_round_trips():
    claim = Claim(
        text="Patient is on Lisinopril 10mg.",
        source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
    )

    restored = Claim.model_validate(claim.model_dump())

    assert restored == claim


def test_claim_with_multiple_refs_round_trips():
    claim = Claim(
        text="Patient is on Lisinopril 10mg, active since 2024-01-01.",
        source_refs=[
            SourceRef(tool_call_id="call-1", record_id="med-1", field="dose"),
            SourceRef(tool_call_id="call-1", record_id="med-1", field="start_date"),
        ],
    )

    restored = Claim.model_validate(claim.model_dump())

    assert restored == claim
    assert len(restored.source_refs) == 2


def test_claim_allows_missing_source_refs_at_schema_level():
    # (issue #93, Option C) A zero-citation claim no longer fails schema
    # validation -- it parses fine, and simply reports itself as uncitable.
    # The citation bar is enforced downstream (app.verification.check_claim),
    # scoped to just this claim, not here.
    claim = Claim(text="Patient is on Lisinopril 10mg.")

    assert claim.source_refs == []
    assert claim.document_citations == []
    assert claim.has_citation is False


def test_claim_allows_empty_source_refs_list_at_schema_level():
    claim = Claim(text="Patient is on Lisinopril 10mg.", source_refs=[])

    assert claim.has_citation is False


def test_claim_rejects_malformed_source_ref_missing_tool_call_id():
    with pytest.raises(ValidationError):
        Claim(
            text="Patient is on Lisinopril 10mg.",
            source_refs=[{"record_id": "med-1", "field": "dose"}],
        )


def test_claim_rejects_malformed_source_ref_missing_record_id():
    with pytest.raises(ValidationError):
        Claim(
            text="Patient is on Lisinopril 10mg.",
            source_refs=[{"tool_call_id": "call-1", "field": "dose"}],
        )


def test_claim_rejects_malformed_source_ref_missing_field():
    with pytest.raises(ValidationError):
        Claim(
            text="Patient is on Lisinopril 10mg.",
            source_refs=[{"tool_call_id": "call-1", "record_id": "med-1"}],
        )


def test_claim_rejects_empty_text():
    with pytest.raises(ValidationError):
        Claim(
            text="",
            source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
        )


def test_claim_rejects_missing_text():
    with pytest.raises(ValidationError):
        Claim(source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")])


def test_claim_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Claim(
            text="Patient is on Lisinopril 10mg.",
            source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
            confidence=0.9,
        )


# ---------------------------------------------------------------------------
# document_citations (P3.6) -- the document-sourced counterpart to source_refs
# ---------------------------------------------------------------------------


def _doc_citation() -> DocumentCitation:
    return DocumentCitation(
        source_type="lab_pdf",
        source_id="doc-1",
        page_or_section="page 1",
        field_or_chunk_id="Glucose",
        quote_or_value="Glucose: 105 mg/dL",
    )


def test_claim_with_only_document_citations_round_trips():
    claim = Claim(text="Glucose is 105 mg/dL.", document_citations=[_doc_citation()])

    restored = Claim.model_validate(claim.model_dump())

    assert restored == claim
    assert restored.source_refs == []


def test_claim_with_source_refs_and_document_citations_round_trips():
    claim = Claim(
        text="On Lisinopril; glucose 105 mg/dL.",
        source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
        document_citations=[_doc_citation()],
    )

    restored = Claim.model_validate(claim.model_dump())

    assert restored == claim


def test_claim_allows_zero_citations_of_either_shape_at_schema_level():
    claim = Claim(text="Unsupported claim.", source_refs=[], document_citations=[])

    assert claim.has_citation is False


def test_claim_allows_no_citation_fields_at_all_at_schema_level():
    claim = Claim(text="Unsupported claim.")

    assert claim.has_citation is False


def test_claim_has_citation_true_with_source_ref():
    claim = Claim(
        text="Patient is on Lisinopril 10mg.",
        source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
    )

    assert claim.has_citation is True


def test_claim_has_citation_true_with_document_citation_only():
    claim = Claim(text="Glucose is 105 mg/dL.", document_citations=[_doc_citation()])

    assert claim.has_citation is True


def test_claim_is_frozen():
    claim = Claim(
        text="Patient is on Lisinopril 10mg.",
        source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
    )

    with pytest.raises(ValidationError):
        claim.text = "edited"


# ---------------------------------------------------------------------------
# VerifiedAnswer
# ---------------------------------------------------------------------------


def test_verified_answer_round_trips_with_claims():
    answer = VerifiedAnswer(
        claims=[
            Claim(
                text="Patient is on Lisinopril 10mg.",
                source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
            )
        ]
    )

    restored = VerifiedAnswer.model_validate(answer.model_dump())

    assert restored == answer


def test_verified_answer_allows_empty_claims_list():
    # No factual claims survived verification (e.g. everything was stripped
    # by P3.3) -- an empty claim list is a valid, if unusual, verified
    # answer. The visible "not found in record" notice text is a P3.3
    # concern, not modeled here.
    answer = VerifiedAnswer(claims=[])

    assert answer.claims == []


def test_verified_answer_rejects_missing_claims():
    with pytest.raises(ValidationError):
        VerifiedAnswer()


def test_verified_answer_allows_claim_without_ref():
    # (issue #93, Option C) A single uncitable claim parses fine and no
    # longer takes down the whole VerifiedAnswer -- see this module's
    # docstring and app.schemas.verification's for the full rationale.
    answer = VerifiedAnswer(claims=[{"text": "Patient is on Lisinopril 10mg."}])

    assert len(answer.claims) == 1
    assert answer.claims[0].has_citation is False


def test_verified_answer_mixed_valid_and_uncitable_claims_all_parse():
    # The headline #93 scenario: one uncitable claim must not prevent the
    # co-occurring valid claim from parsing into the SAME VerifiedAnswer.
    answer = VerifiedAnswer(
        claims=[
            Claim(
                text="Patient is on Lisinopril 10mg.",
                source_refs=[SourceRef(tool_call_id="call-1", record_id="med-1", field="dose")],
            ),
            Claim(text="Unsupported claim."),
        ]
    )

    assert len(answer.claims) == 2
    assert answer.claims[0].has_citation is True
    assert answer.claims[1].has_citation is False


def test_verified_answer_rejects_unknown_field():
    with pytest.raises(ValidationError):
        VerifiedAnswer(claims=[], notes="extra")

"""Validation tests for the claim-level verification response contract (P3.1).

``Claim``/``VerifiedAnswer`` (``app.schemas.verification``) are the contract
the verification layer produces: every factual claim carries >=1
``SourceRef``, and a claim with zero refs fails schema validation. This is
distinct from ``app.schemas.planner.FinalAnswer``, the raw two-call
extraction output (P2.9) -- see the module docstring in
``app.schemas.verification`` for why they're kept separate.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.common import SourceRef
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


def test_claim_rejects_missing_source_refs():
    with pytest.raises(ValidationError):
        Claim(text="Patient is on Lisinopril 10mg.")


def test_claim_rejects_empty_source_refs_list():
    with pytest.raises(ValidationError):
        Claim(text="Patient is on Lisinopril 10mg.", source_refs=[])


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


def test_verified_answer_rejects_claim_without_ref():
    with pytest.raises(ValidationError):
        VerifiedAnswer(claims=[{"text": "Patient is on Lisinopril 10mg."}])


def test_verified_answer_rejects_unknown_field():
    with pytest.raises(ValidationError):
        VerifiedAnswer(claims=[], notes="extra")

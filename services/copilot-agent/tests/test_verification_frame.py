"""Hermetic tests for the P3.8 verification SSE frame contract.

``build_verification_payload`` is the pure serializer that turns the
verification layer's ``VerdictResult`` (P3.7) + ``RenderedAnswer`` (P3.3)
into the JSON payload the ``verification`` SSE frame carries -- the single
contract the P3.8 UI (verdict badge, citation chips, warning banner) renders.

The answer->claims/meds extraction pipeline that would feed real
``VerdictResult``/``RenderedAnswer`` values from a live planner answer is NOT
built yet (see ``app.verdict`` / ``app.rendering`` docstrings), so the live
``/chat`` stream emits the *pending* payload (``verdict: null``, no segments,
no warnings) today. These tests pin both the pending shape and the fully
populated shape so the contract and the UI are complete and testable ahead of
that integration.
"""

from __future__ import annotations

from app.allergy_check import AllergyConflict
from app.chat import build_verification_payload
from app.rendering import Notice, RenderedAnswer, RenderedClaim
from app.schemas.common import InteractionSeverity, SourceRef
from app.schemas.tools import DrugInteractionItem
from app.verdict import Verdict, VerdictResult


def test_pending_payload_has_null_verdict_and_empty_evidence():
    payload = build_verification_payload(None, None)

    assert payload["verdict"] is None
    assert payload["segments"] == []
    assert payload["warnings"] == {
        "allergy_conflicts": [],
        "blocking_interactions": [],
        "warning_interactions": [],
    }


def test_verified_payload_serializes_claims_with_citations():
    verdict_result = VerdictResult(
        verdict=Verdict.VERIFIED,
        total_claim_count=1,
        stripped_claim_count=0,
        allergy_conflicts=[],
        blocking_interactions=[],
        warning_interactions=[],
    )
    rendered = RenderedAnswer(
        segments=[
            RenderedClaim(
                text="She takes lisinopril 10 mg daily.",
                source_refs=[
                    SourceRef(
                        tool_call_id="call-1",
                        record_id="med-42",
                        field="dose",
                        asserted_value="10 mg",
                    )
                ],
            )
        ]
    )

    payload = build_verification_payload(verdict_result, rendered)

    assert payload["verdict"] == "verified"
    assert payload["segments"] == [
        {
            "type": "claim",
            "text": "She takes lisinopril 10 mg daily.",
            "citations": [
                {
                    "tool_call_id": "call-1",
                    "record_id": "med-42",
                    "field": "dose",
                    "value": "10 mg",
                }
            ],
        }
    ]
    assert payload["warnings"]["allergy_conflicts"] == []


def test_notice_segment_is_serialized_as_a_notice():
    verdict_result = VerdictResult(
        verdict=Verdict.BLOCKED,
        total_claim_count=1,
        stripped_claim_count=1,
        allergy_conflicts=[],
        blocking_interactions=[],
        warning_interactions=[],
    )
    rendered = RenderedAnswer(segments=[Notice()])

    payload = build_verification_payload(verdict_result, rendered)

    assert payload["segments"] == [{"type": "notice", "text": "Not found in record."}]


def test_blocked_payload_serializes_allergy_and_blocking_interaction_warnings():
    interaction = DrugInteractionItem(
        drug_a="warfarin",
        drug_b="aspirin",
        severity=InteractionSeverity.MAJOR,
        description="Increased bleeding risk.",
    )
    verdict_result = VerdictResult(
        verdict=Verdict.BLOCKED,
        total_claim_count=0,
        stripped_claim_count=0,
        allergy_conflicts=[
            AllergyConflict(medication_name="Ibuprofen", allergy_substance="NSAID")
        ],
        blocking_interactions=[interaction],
        warning_interactions=[],
    )

    payload = build_verification_payload(verdict_result, RenderedAnswer(segments=[]))

    assert payload["verdict"] == "blocked"
    assert payload["warnings"]["allergy_conflicts"] == [
        {"medication_name": "Ibuprofen", "allergy_substance": "NSAID"}
    ]
    assert payload["warnings"]["blocking_interactions"] == [
        {
            "drug_a": "warfarin",
            "drug_b": "aspirin",
            "severity": "major",
            "description": "Increased bleeding risk.",
        }
    ]
    assert payload["warnings"]["warning_interactions"] == []

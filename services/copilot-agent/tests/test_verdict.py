"""Decision-table tests for the whole-answer verdict (P3.7).

``app.verdict`` combines the three verification signals -- citation
completeness (``app.verification``), allergy conflicts (``app.allergy_check``),
and drug interactions (``app.check_drug_interactions``) -- into a single
``verified | partially_verified | blocked`` verdict. See the module docstring
in ``app.verdict`` for the full decision table and its justification.

No LLM, no I/O, no clock -- pure function of seeded inputs (the answer->claims
extraction pipeline this would consume in production is not built yet; same
seam gap as P3.2/P3.4/P3.6 -- see ``docs/IMPLEMENTATION_PLAN.md`` Sec 4.4).
"""

from __future__ import annotations

import dataclasses

import pytest

from app.allergy_check import AllergyConflict
from app.schemas.common import InteractionSeverity, SourceRef
from app.schemas.tools import DrugInteractionItem
from app.schemas.verification import Claim
from app.verification import CitationCheckResult, CitationStatus, ClaimCheckResult
from app.verdict import Verdict, VerdictResult, compute_verdict, to_trace_record

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ref() -> SourceRef:
    return SourceRef(tool_call_id="call_0", record_id="0", field="name", asserted_value="x")


def _passing_claim(text: str = "A verified claim.") -> ClaimCheckResult:
    claim = Claim(text=text, source_refs=[_ref()])
    return ClaimCheckResult(claim=claim, citation_results=[CitationCheckResult(source_ref=_ref(), status=CitationStatus.VALID)])


def _failing_claim(text: str = "An unverifiable claim.") -> ClaimCheckResult:
    claim = Claim(text=text, source_refs=[_ref()])
    return ClaimCheckResult(
        claim=claim, citation_results=[CitationCheckResult(source_ref=_ref(), status=CitationStatus.VALUE_MISMATCH)]
    )


def _allergy_conflict() -> AllergyConflict:
    return AllergyConflict(medication_name="Ibuprofen", allergy_substance="Ibuprofen")


def _interaction(severity: InteractionSeverity) -> DrugInteractionItem:
    return DrugInteractionItem(drug_a="Ibuprofen", drug_b="Lisinopril", severity=severity, description="d")


# ---------------------------------------------------------------------------
# Decision table -- citation axis (all / some / none verified) x
# safety axis (no violation / warning / blocking). 9 cells, exhaustive.
# ---------------------------------------------------------------------------


def test_all_verified_no_violation_is_verified():
    result = compute_verdict([_passing_claim(), _passing_claim()], [], [])

    assert result.verdict is Verdict.VERIFIED


def test_all_verified_moderate_interaction_only_is_partially_verified():
    result = compute_verdict([_passing_claim()], [], [_interaction(InteractionSeverity.MODERATE)])

    assert result.verdict is Verdict.PARTIALLY_VERIFIED


def test_all_verified_minor_interaction_only_is_partially_verified():
    result = compute_verdict([_passing_claim()], [], [_interaction(InteractionSeverity.MINOR)])

    assert result.verdict is Verdict.PARTIALLY_VERIFIED


def test_all_verified_allergy_conflict_is_blocked():
    """Safety dominates: every citation passed, but an allergy conflict on a
    mentioned medication still blocks (docs/IMPLEMENTATION_PLAN.md Sec 4.4)."""
    result = compute_verdict([_passing_claim()], [_allergy_conflict()], [])

    assert result.verdict is Verdict.BLOCKED


def test_all_verified_major_interaction_is_blocked():
    result = compute_verdict([_passing_claim()], [], [_interaction(InteractionSeverity.MAJOR)])

    assert result.verdict is Verdict.BLOCKED


def test_all_verified_contraindicated_interaction_is_blocked():
    result = compute_verdict([_passing_claim()], [], [_interaction(InteractionSeverity.CONTRAINDICATED)])

    assert result.verdict is Verdict.BLOCKED


def test_some_stripped_no_violation_is_partially_verified():
    result = compute_verdict([_passing_claim(), _failing_claim()], [], [])

    assert result.verdict is Verdict.PARTIALLY_VERIFIED


def test_some_stripped_warning_is_partially_verified():
    result = compute_verdict([_passing_claim(), _failing_claim()], [], [_interaction(InteractionSeverity.MINOR)])

    assert result.verdict is Verdict.PARTIALLY_VERIFIED


def test_some_stripped_blocking_is_blocked():
    result = compute_verdict([_passing_claim(), _failing_claim()], [_allergy_conflict()], [])

    assert result.verdict is Verdict.BLOCKED


def test_all_stripped_no_violation_is_blocked():
    """Fail-closed: every claim failed citation. Zero grounded evidence backs
    the answer -- 'partially verified' would overstate trust, so this is
    treated the same as zero claims (see module docstring)."""
    result = compute_verdict([_failing_claim(), _failing_claim()], [], [])

    assert result.verdict is Verdict.BLOCKED


def test_all_stripped_warning_is_blocked():
    result = compute_verdict([_failing_claim()], [], [_interaction(InteractionSeverity.MINOR)])

    assert result.verdict is Verdict.BLOCKED


def test_all_stripped_blocking_is_blocked():
    result = compute_verdict([_failing_claim()], [_allergy_conflict()], [])

    assert result.verdict is Verdict.BLOCKED


def test_zero_claims_no_violation_is_blocked():
    """Fail-closed: an answer with no verifiable claims at all is never
    'verified' -- see module docstring for the empty-answer justification."""
    result = compute_verdict([], [], [])

    assert result.verdict is Verdict.BLOCKED


def test_zero_claims_blocking_is_blocked():
    result = compute_verdict([], [_allergy_conflict()], [])

    assert result.verdict is Verdict.BLOCKED


# ---------------------------------------------------------------------------
# Safety-axis internal precedence: blocking dominates warning when both
# a blocking and a non-blocking interaction are present simultaneously.
# ---------------------------------------------------------------------------


def test_mixed_blocking_and_warning_interactions_is_blocked():
    result = compute_verdict(
        [_passing_claim()],
        [],
        [_interaction(InteractionSeverity.MINOR), _interaction(InteractionSeverity.MAJOR)],
    )

    assert result.verdict is Verdict.BLOCKED


# ---------------------------------------------------------------------------
# VerdictResult evidence -- the structured payload P3.8 consumes.
# ---------------------------------------------------------------------------


def test_verdict_result_carries_claim_counts():
    result = compute_verdict([_passing_claim(), _passing_claim(), _failing_claim()], [], [])

    assert result.total_claim_count == 3
    assert result.stripped_claim_count == 1


def test_verdict_result_carries_allergy_conflicts():
    conflict = _allergy_conflict()

    result = compute_verdict([_passing_claim()], [conflict], [])

    assert result.allergy_conflicts == [conflict]


def test_verdict_result_partitions_interactions_by_blocking_vs_warning():
    major = _interaction(InteractionSeverity.MAJOR)
    minor = _interaction(InteractionSeverity.MINOR)

    result = compute_verdict([_passing_claim()], [], [major, minor])

    assert result.blocking_interactions == [major]
    assert result.warning_interactions == [minor]


def test_verdict_result_is_frozen():
    result = compute_verdict([], [], [])

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = Verdict.VERIFIED  # type: ignore[misc]


def test_verdict_enum_values_are_the_contract_strings():
    assert Verdict.VERIFIED.value == "verified"
    assert Verdict.PARTIALLY_VERIFIED.value == "partially_verified"
    assert Verdict.BLOCKED.value == "blocked"


# ---------------------------------------------------------------------------
# Trace logging seam (P3.7): the verdict must be recorded into a loggable
# record -- the shape a per-turn trace mechanism would persist (P4.2 durable
# store not built; see app.verdict module docstring for the deferred seam).
# ---------------------------------------------------------------------------


def test_to_trace_record_records_the_verdict():
    result = compute_verdict([_passing_claim()], [], [])

    record = to_trace_record(result)

    assert record["verdict"] == "verified"


def test_to_trace_record_records_claim_and_violation_evidence():
    result = compute_verdict(
        [_passing_claim(), _failing_claim()],
        [_allergy_conflict()],
        [_interaction(InteractionSeverity.MINOR)],
    )

    record = to_trace_record(result)

    assert record["verdict"] == "blocked"
    assert record["total_claim_count"] == 2
    assert record["stripped_claim_count"] == 1
    assert record["allergy_conflict_count"] == 1
    assert record["blocking_interaction_count"] == 0
    assert record["warning_interaction_count"] == 1


def test_to_trace_record_is_json_serializable():
    import json

    result = compute_verdict([_failing_claim()], [_allergy_conflict()], [_interaction(InteractionSeverity.MAJOR)])

    json.dumps(to_trace_record(result))  # must not raise

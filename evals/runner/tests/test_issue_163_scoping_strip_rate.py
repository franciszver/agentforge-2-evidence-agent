"""Red-first schema-pinning tests for the issue #163 scoping strip-rate
harness (``evals/runner/issue_163_scoping_strip_rate.py``).

**What this file does and does not cover.** This harness's whole point is a
LIVE, paired-draw measurement (one live planner draw per eval case, then
verification run twice on that same draw -- flag-OFF and flag-ON, see the
harness module's own docstring for the full method). None of that live
machinery is exercised here -- there is no model, no network, no Docker
stack, nothing that could make this suite flaky or slow. This file pins only
the harness's PURE, in-memory aggregation logic: given already-computed
``app.verification.ClaimCheckResult``/``app.verdict.VerdictResult`` objects
(built directly here, the same way ``tests/test_extraction.py`` and
``tests/test_verification.py`` construct them), does the harness correctly
turn them into the report rows the owner will read (per-arm claim/citation
counts, the downgrade count, the verdict-flip/newly-blocked-turn booleans,
and the per-category summary table)? This is exactly the kind of test the
issue's "red-first: unit test pinning the pairing logic's report schema (no
live model in CI)" requirement asks for -- mirrors ``runner.gate``'s own
pure-aggregation tests in spirit, not ``runner.tests.test_harness``'s
live-pipeline-shaped ones.

Written BEFORE ``runner.issue_163_scoping_strip_rate`` exists (strict
red-first, per ``CLAUDE.md``'s "Red first -- strict TDD, everywhere"): every
test below currently fails on the import line with ``ModuleNotFoundError``,
which IS the intended red state this commit captures -- not a placeholder
``pytest.mark.xfail`` (the schema being pinned doesn't exist as a runnable
concept yet to xfail against; the import failure itself is the "red"
artifact for a not-yet-written module, same class of red as a new test
naming a not-yet-written function).

This module deliberately does NOT match ``test_*.py``/``*_test.py`` in a way
that would matter here -- it DOES match (`test_issue_163_...py`), which is
correct: unlike the live spike/harness modules themselves (``issue_130_spike
.py``, ``issue_154_stability_harness.py``, deliberately named to dodge
pytest collection because they dial out), THIS file contains no live call
and must run in CI like any other test."""

from __future__ import annotations

from app.schemas.common import SourceRef
from app.schemas.verification import Claim
from app.verdict import Verdict, VerdictResult
from app.verification import CitationCheckResult, CitationStatus, ClaimCheckResult

from runner.issue_163_scoping_strip_rate import (
    ArmRecord,
    CaseRecord,
    arm_record_from_results,
    build_case_record,
    summarize,
)


def _source_ref(call_id: str = "call_0") -> SourceRef:
    return SourceRef(tool_call_id=call_id, record_id="0", field="dose", asserted_value="10 mg")


def _claim(*statuses: CitationStatus) -> ClaimCheckResult:
    citations = [
        CitationCheckResult(source_ref=_source_ref(f"call_{i}"), status=status)
        for i, status in enumerate(statuses)
    ]
    return ClaimCheckResult(
        claim=Claim(text="claim text", source_refs=[c.source_ref for c in citations]),
        citation_results=citations,
    )


def _verdict_result(verdict: Verdict, total: int, stripped: int) -> VerdictResult:
    return VerdictResult(
        verdict=verdict,
        total_claim_count=total,
        stripped_claim_count=stripped,
        allergy_conflicts=[],
        blocking_interactions=[],
        warning_interactions=[],
    )


class TestArmRecordFromResults:
    def test_counts_downgraded_citations_only(self) -> None:
        claim_results = [
            _claim(CitationStatus.VALID),
            _claim(CitationStatus.TOOL_CALL_NOT_ENGAGED),
            _claim(CitationStatus.VALID, CitationStatus.TOOL_CALL_NOT_ENGAGED),
        ]
        verdict_result = _verdict_result(Verdict.PARTIALLY_VERIFIED, total=3, stripped=2)

        arm = arm_record_from_results(verdict_result, claim_results)

        assert isinstance(arm, ArmRecord)
        assert arm.verdict == "partially_verified"
        assert arm.total_claim_count == 3
        assert arm.stripped_claim_count == 2
        assert arm.downgraded_count == 2  # two TOOL_CALL_NOT_ENGAGED citations, across two claims
        assert arm.error is None
        assert len(arm.claims) == 3
        assert arm.claims[0].passed is True
        assert arm.claims[0].citation_statuses == ["valid"]
        assert arm.claims[1].passed is False
        assert arm.claims[1].citation_statuses == ["tool_call_not_engaged"]

    def test_zero_claims_is_not_an_error(self) -> None:
        verdict_result = _verdict_result(Verdict.BLOCKED, total=0, stripped=0)

        arm = arm_record_from_results(verdict_result, [])

        assert arm.error is None
        assert arm.total_claim_count == 0
        assert arm.downgraded_count == 0
        assert arm.claims == []


class TestBuildCaseRecord:
    def _off_on(self, off_verdict: Verdict, on_verdict: Verdict) -> tuple[ArmRecord, ArmRecord]:
        off = arm_record_from_results(_verdict_result(off_verdict, 1, 0 if off_verdict == Verdict.VERIFIED else 1), [_claim(CitationStatus.VALID)])
        on = arm_record_from_results(_verdict_result(on_verdict, 1, 0 if on_verdict == Verdict.VERIFIED else 1), [_claim(CitationStatus.VALID)])
        return off, on

    def test_verified_to_blocked_flip_is_flagged_and_counts_as_newly_blocked(self) -> None:
        off, on = self._off_on(Verdict.VERIFIED, Verdict.BLOCKED)

        record = build_case_record(
            case_id="case-1",
            category="citation_present",
            applicable=True,
            engaged_call_ids=["call_0"],
            total_call_ids=["call_0", "call_1"],
            off=off,
            on=on,
        )

        assert isinstance(record, CaseRecord)
        assert record.comparable is True
        assert record.verdict_flip is True
        assert record.flip_detail == "verified->blocked"
        assert record.newly_blocked is True

    def test_stable_verdict_is_not_a_flip(self) -> None:
        off, on = self._off_on(Verdict.VERIFIED, Verdict.VERIFIED)

        record = build_case_record(
            case_id="case-2",
            category="citation_present",
            applicable=True,
            engaged_call_ids=["call_0"],
            total_call_ids=["call_0"],
            off=off,
            on=on,
        )

        assert record.verdict_flip is False
        assert record.flip_detail is None
        assert record.newly_blocked is False

    def test_already_blocked_off_is_not_newly_blocked(self) -> None:
        off, on = self._off_on(Verdict.BLOCKED, Verdict.BLOCKED)

        record = build_case_record(
            case_id="case-3",
            category="citation_present",
            applicable=True,
            engaged_call_ids=[],
            total_call_ids=["call_0"],
            off=off,
            on=on,
        )

        assert record.verdict_flip is False
        assert record.newly_blocked is False

    def test_arm_error_makes_the_case_not_comparable(self) -> None:
        off = ArmRecord(
            verdict="",
            total_claim_count=0,
            stripped_claim_count=0,
            claims=[],
            downgraded_count=0,
            error="LLMEngineError('boom')",
        )
        on = arm_record_from_results(_verdict_result(Verdict.VERIFIED, 1, 0), [_claim(CitationStatus.VALID)])

        record = build_case_record(
            case_id="case-4",
            category="citation_present",
            applicable=True,
            engaged_call_ids=[],
            total_call_ids=[],
            off=off,
            on=on,
        )

        assert record.comparable is False
        assert record.verdict_flip is False
        assert record.newly_blocked is False

    def test_not_applicable_case_carries_no_arms(self) -> None:
        record = build_case_record(
            case_id="case-5",
            category="tool_selection",
            applicable=False,
            engaged_call_ids=[],
            total_call_ids=["call_0"],
            off=None,
            on=None,
        )

        assert record.comparable is False
        assert record.off is None
        assert record.on is None
        assert record.verdict_flip is False


class TestSummarize:
    def test_per_category_and_total_rows(self) -> None:
        flip_off, flip_on = TestBuildCaseRecord()._off_on(Verdict.VERIFIED, Verdict.BLOCKED)
        stable_off, stable_on = TestBuildCaseRecord()._off_on(Verdict.VERIFIED, Verdict.VERIFIED)
        records = [
            build_case_record(
                case_id="c1",
                category="citation_present",
                applicable=True,
                engaged_call_ids=["call_0"],
                total_call_ids=["call_0"],
                off=flip_off,
                on=flip_on,
            ),
            build_case_record(
                case_id="c2",
                category="citation_present",
                applicable=True,
                engaged_call_ids=["call_0"],
                total_call_ids=["call_0"],
                off=stable_off,
                on=stable_on,
            ),
            build_case_record(
                case_id="c3",
                category="tool_selection",
                applicable=False,
                engaged_call_ids=[],
                total_call_ids=["call_0"],
                off=None,
                on=None,
            ),
        ]

        report = summarize(records)

        assert report["by_category"]["citation_present"]["cases"] == 2
        assert report["by_category"]["citation_present"]["comparable_cases"] == 2
        assert report["by_category"]["citation_present"]["claims_total_off"] == 2
        assert report["by_category"]["citation_present"]["verdict_flips"] == 1
        assert report["by_category"]["citation_present"]["newly_blocked"] == 1
        assert report["by_category"]["tool_selection"]["cases"] == 1
        assert report["by_category"]["tool_selection"]["comparable_cases"] == 0
        assert report["total"]["cases"] == 3
        assert report["total"]["comparable_cases"] == 2
        assert report["total"]["verdict_flips"] == 1
        assert report["total"]["newly_blocked"] == 1

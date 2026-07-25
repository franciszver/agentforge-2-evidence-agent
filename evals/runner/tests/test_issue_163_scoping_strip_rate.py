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
counts, the downgrade count, the exposure/eligibility counters, the
verdict-flip/newly-blocked-turn booleans, and the per-category summary
table)? This is exactly the kind of test the issue's "red-first: unit test
pinning the pairing logic's report schema (no live model in CI)"
requirement asks for -- mirrors ``runner.gate``'s own pure-aggregation
tests in spirit, not ``runner.tests.test_harness``'s live-pipeline-shaped
ones.

**Issue #163 gate-2 (Opus) review -- metric mutation-survivability.** The
gate-2 review specifically flagged that this suite must be able to catch a
broken DECISION-BEARING metric, not just exercise the happy path.
``TestMetricMutationSurvival`` below adds three fixtures, each written
against a specific mutation of the implementation it would have caught (the
mutation actually checked, by hand, before this file was committed -- see
each test's docstring for exactly which line was broken and confirmed red):

  1. a ``TOOL_CALL_NOT_ENGAGED`` citation in the ON arm must make
     ``summarize()``'s ``claims_downgraded`` bucket move -- catches a broken
     downgrade-counting predicate (e.g. ``!=`` swapped for ``is`` on the
     wrong operand, or the loop silently skipping the ON arm).
  2. an errored arm must make ``summarize()``'s ``errors`` bucket move --
     catches a broken/removed error-detection condition.
  3. an ON arm with FEWER total claims than the OFF arm (prevention loss)
     must be visible in ``claims_total_on`` WITHOUT needing to inspect
     ``claims_total_off`` to notice -- catches the exact "prevention
     blindness" gap issue #163 gate-2 named: a reader who only glances at
     one bucket key must still see the drop.

Written BEFORE ``runner.issue_163_scoping_strip_rate`` exists (strict
red-first, per ``CLAUDE.md``'s "Red first -- strict TDD, everywhere"): the
very first commit of this file failed on the import line with
``ModuleNotFoundError`` -- the intended red state for a not-yet-written
module. This gate-2 revision keeps that discipline for the NEW metric
fields: each new assertion below was confirmed to fail against the
PRE-gate-2 implementation before the corresponding harness fix landed.

This module deliberately does NOT match ``test_*.py``/``*_test.py`` in a way
that would matter here -- it DOES match (`test_issue_163_...py`), which is
correct: unlike the live spike/harness modules themselves (``issue_130_spike
.py``, ``issue_154_stability_harness.py``, deliberately named to dodge
pytest collection because they dial out), THIS file contains no live call
and must run in CI like any other test."""

from __future__ import annotations

from typing import Any

from app.schemas.common import SourceRef
from app.schemas.verification import Claim
from app.verdict import Verdict, VerdictResult
from app.verification import CitationCheckResult, CitationStatus, ClaimCheckResult

from runner.issue_163_scoping_strip_rate import (
    ArmRecord,
    CaseRecord,
    _case_index_entry,
    _strip_rate_label,
    arm_record_from_results,
    build_case_record,
    build_report,
    error_arm_record,
    summarize,
)

_SESSION_ID = "test-session-0123456789abcdef"
_SESSION_STARTED_AT = "2026-07-25T00:00:00+00:00"


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


def _build_case_record(**overrides: Any) -> CaseRecord:
    """Thin wrapper over ``build_case_record`` supplying the boilerplate
    session-provenance/exposure kwargs every test needs but doesn't care
    about, so individual tests only spell out the fields they're actually
    asserting on."""
    defaults: dict[str, Any] = {
        "case_id": "case-x",
        "category": "citation_present",
        "applicable": True,
        "engaged_call_ids": [],
        "total_call_ids": [],
        "unengaged_calls_with_data": [],
        "off": None,
        "on": None,
        "session_id": _SESSION_ID,
        "session_started_at": _SESSION_STARTED_AT,
    }
    defaults.update(overrides)
    return build_case_record(**defaults)


def _off_on(off_verdict: Verdict, on_verdict: Verdict) -> tuple[ArmRecord, ArmRecord]:
    """Module-level helper (hoisted out of ``TestBuildCaseRecord`` -- issue
    #163 gate-1 review) shared by every test class below so none of them
    need to instantiate another test class just to reach it."""
    off = arm_record_from_results(
        _verdict_result(off_verdict, 1, 0 if off_verdict == Verdict.VERIFIED else 1), [_claim(CitationStatus.VALID)]
    )
    on = arm_record_from_results(
        _verdict_result(on_verdict, 1, 0 if on_verdict == Verdict.VERIFIED else 1), [_claim(CitationStatus.VALID)]
    )
    return off, on


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
    def test_verified_to_blocked_flip_is_flagged_and_counts_as_newly_blocked(self) -> None:
        off, on = _off_on(Verdict.VERIFIED, Verdict.BLOCKED)

        record = _build_case_record(case_id="case-1", off=off, on=on, engaged_call_ids=["call_0"], total_call_ids=["call_0", "call_1"])

        assert isinstance(record, CaseRecord)
        assert record.comparable is True
        assert record.verdict_flip is True
        assert record.flip_detail == "verified->blocked"
        assert record.newly_blocked is True

    def test_stable_verdict_is_not_a_flip(self) -> None:
        off, on = _off_on(Verdict.VERIFIED, Verdict.VERIFIED)

        record = _build_case_record(case_id="case-2", off=off, on=on, engaged_call_ids=["call_0"], total_call_ids=["call_0"])

        assert record.verdict_flip is False
        assert record.flip_detail is None
        assert record.newly_blocked is False

    def test_already_blocked_off_is_not_newly_blocked(self) -> None:
        off, on = _off_on(Verdict.BLOCKED, Verdict.BLOCKED)

        record = _build_case_record(case_id="case-3", off=off, on=on, total_call_ids=["call_0"])

        assert record.verdict_flip is False
        assert record.newly_blocked is False

    def test_arm_error_makes_the_case_not_comparable(self) -> None:
        off = error_arm_record(RuntimeError("boom"))
        on = arm_record_from_results(_verdict_result(Verdict.VERIFIED, 1, 0), [_claim(CitationStatus.VALID)])

        record = _build_case_record(case_id="case-4", off=off, on=on)

        assert record.comparable is False
        assert record.verdict_flip is False
        assert record.newly_blocked is False
        # Issue #163 gate-2/Opus review: an errored OFF arm has nothing
        # countable -- eligible_claims_off must be None, not 0 (0 would
        # falsely claim "we checked and there were zero eligible claims").
        assert record.eligible_claims_off is None

    def test_no_applicable_exclusion_left_case_still_carries_arms(self) -> None:
        """Issue #163 gate-2/Opus review: 'applicable=False' is no longer an
        exclusion -- a case whose eval assertions don't check the verdict
        can still (and, since the harness fix, always does) carry real off/
        on arms, because production runs verification unconditionally
        (app/chat.py's run_verification call has no such gate)."""
        off, on = _off_on(Verdict.VERIFIED, Verdict.VERIFIED)

        record = _build_case_record(case_id="case-5", category="tool_selection", applicable=False, off=off, on=on)

        assert record.applicable is False
        assert record.off is not None
        assert record.on is not None
        assert record.comparable is True

    def test_eligible_claims_off_counts_only_passing_off_claims(self) -> None:
        """Issue #163 gate-2/Opus review: eligible_claims_off is the strip-
        rate denominator -- OFF-arm claims that already passed provenance.
        A claim that already failed (e.g. unknown_record, unrelated to
        scoping) must NOT inflate this count."""
        off = arm_record_from_results(
            _verdict_result(Verdict.PARTIALLY_VERIFIED, total=2, stripped=1),
            [_claim(CitationStatus.VALID), _claim(CitationStatus.UNKNOWN_RECORD)],
        )
        on = arm_record_from_results(_verdict_result(Verdict.PARTIALLY_VERIFIED, 2, 1), [_claim(CitationStatus.VALID)])

        record = _build_case_record(case_id="case-6", off=off, on=on)

        assert record.eligible_claims_off == 1  # only the VALID claim, not the UNKNOWN_RECORD one


class TestSummarize:
    def test_per_category_and_total_rows(self) -> None:
        flip_off, flip_on = _off_on(Verdict.VERIFIED, Verdict.BLOCKED)
        stable_off, stable_on = _off_on(Verdict.VERIFIED, Verdict.VERIFIED)
        records = [
            _build_case_record(case_id="c1", off=flip_off, on=flip_on, engaged_call_ids=["call_0"], total_call_ids=["call_0"]),
            _build_case_record(case_id="c2", off=stable_off, on=stable_on, engaged_call_ids=["call_0"], total_call_ids=["call_0"]),
            _build_case_record(case_id="c3", category="tool_selection", applicable=False, total_call_ids=["call_0"]),
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

    def test_exposure_counted_even_when_case_not_comparable(self) -> None:
        """unengaged_exposure_calls is gated on record.error is None, NOT on
        record.comparable -- a case whose verification arm errored still
        had a real planner draw with real exposure."""
        off = error_arm_record(RuntimeError("boom"))
        on = arm_record_from_results(_verdict_result(Verdict.VERIFIED, 1, 0), [_claim(CitationStatus.VALID)])
        records = [
            _build_case_record(case_id="c1", off=off, on=on, unengaged_calls_with_data=["call_1", "call_2"]),
        ]

        report = summarize(records)

        assert report["total"]["unengaged_exposure_calls"] == 2
        assert report["total"]["comparable_cases"] == 0  # confirms this is NOT gated by comparable


class TestMetricMutationSurvival:
    """Issue #163 gate-2/Opus review: each test here targets a specific
    mutation of the pre-fix implementation, confirmed (by hand) to fail
    under that mutation -- see this class's own module-docstring paragraph
    for the full list. These are deliberately at the ``summarize()`` level
    (not just ``arm_record_from_results``/``build_case_record`` in
    isolation) to also prove the WIRING between them is correct, not just
    each function alone."""

    def test_on_arm_tool_call_not_engaged_moves_claims_downgraded(self) -> None:
        """Mutation checked: in arm_record_from_results, changing
        ``citation.status is CitationStatus.TOOL_CALL_NOT_ENGAGED`` to
        ``citation.status is CitationStatus.VALID`` (i.e. counting the wrong
        status) makes this assertion fail (claims_downgraded == 0, not 1) --
        confirmed red under that mutation before this test was accepted."""
        off = arm_record_from_results(_verdict_result(Verdict.VERIFIED, 1, 0), [_claim(CitationStatus.VALID)])
        on = arm_record_from_results(
            _verdict_result(Verdict.PARTIALLY_VERIFIED, 1, 1), [_claim(CitationStatus.TOOL_CALL_NOT_ENGAGED)]
        )
        records = [_build_case_record(case_id="c1", off=off, on=on)]

        report = summarize(records)

        assert report["total"]["claims_downgraded"] == 1
        assert report["by_category"]["citation_present"]["claims_downgraded"] == 1

    def test_errored_arm_moves_errors_bucket(self) -> None:
        """Mutation checked: deleting the ``record.off is not None and
        record.off.error is not None`` disjunct from summarize()'s error
        condition (leaving only ``record.error is not None``, which is the
        DRAW error, not the ARM error) makes this assertion fail
        (errors == 0, not 1) -- confirmed red under that mutation before
        this test was accepted."""
        off = error_arm_record(RuntimeError("extraction exhausted retries"))
        on = arm_record_from_results(_verdict_result(Verdict.VERIFIED, 1, 0), [_claim(CitationStatus.VALID)])
        records = [_build_case_record(case_id="c1", off=off, on=on)]

        report = summarize(records)

        assert report["total"]["errors"] == 1
        assert report["by_category"]["citation_present"]["errors"] == 1

    def test_prevention_loss_visible_in_claims_total_on_alone(self) -> None:
        """Mutation checked: removing the ``target["claims_total_on"] +=
        record.on.total_claim_count`` accumulation line from summarize()
        (leaving the ``claims_total_on``/``claims_stripped_on`` keys present
        but permanently 0, the "silently stopped counting" shape a partial
        revert would take) makes this assertion fail (``claims_total_on ==
        0``, not ``1``) -- confirmed red under that mutation before this
        test was accepted. The OFF arm extracted 3 claims; the ON arm's
        narrower catalog (PREVENTION) only extracted 1 -- that drop must be
        readable from claims_total_on by itself, without cross-referencing
        claims_total_off."""
        off = arm_record_from_results(
            _verdict_result(Verdict.VERIFIED, 3, 0),
            [_claim(CitationStatus.VALID), _claim(CitationStatus.VALID), _claim(CitationStatus.VALID)],
        )
        on = arm_record_from_results(_verdict_result(Verdict.VERIFIED, 1, 0), [_claim(CitationStatus.VALID)])
        records = [_build_case_record(case_id="c1", off=off, on=on)]

        report = summarize(records)

        assert report["total"]["claims_total_off"] == 3
        assert report["total"]["claims_total_on"] == 1  # visible directly -- prevention dropped 2 claims
        assert report["total"]["claims_stripped_on"] == 0


class TestStripRateLabel:
    def test_reports_downgraded_over_eligible_not_total(self) -> None:
        bucket = {"claims_downgraded": 1, "eligible_claims_off": 4, "claims_total_off": 10}

        assert _strip_rate_label(bucket) == "1/4 (25%)"

    def test_zero_eligible_is_labeled_not_a_division_by_zero(self) -> None:
        bucket = {"claims_downgraded": 0, "eligible_claims_off": 0, "claims_total_off": 5}

        assert _strip_rate_label(bucket) == "n/a"


class TestCaseIndexEntry:
    def test_light_index_row_shape(self) -> None:
        off, on = _off_on(Verdict.VERIFIED, Verdict.BLOCKED)
        record = _build_case_record(
            case_id="c1", off=off, on=on, engaged_call_ids=["call_0"], total_call_ids=["call_0"],
            unengaged_calls_with_data=["call_1"],
        )

        entry = _case_index_entry(record)

        # Issue #163 gate-2/Opus review: exposure/eligibility counters and
        # session_id were ADDED to the light index (deliberate, reviewed --
        # see _case_index_entry's docstring); everything else about the
        # index stays exactly as light as the gate-1 version (no off/on/
        # claims detail).
        assert entry == {
            "case_id": "c1",
            "category": "citation_present",
            "applicable": True,
            "comparable": True,
            "verdict_flip": True,
            "newly_blocked": True,
            "unengaged_calls_with_data": ["call_1"],
            "eligible_claims_off": 1,
            "session_id": _SESSION_ID,
        }
        assert "off" not in entry
        assert "on" not in entry
        assert "claims" not in entry


class TestBuildReport:
    def test_report_carries_light_index_not_full_case_records(self) -> None:
        off, on = _off_on(Verdict.VERIFIED, Verdict.VERIFIED)
        records = [
            _build_case_record(case_id="c1", off=off, on=on, engaged_call_ids=["call_0"], total_call_ids=["call_0"]),
        ]

        report = build_report(records)

        assert report["case_count"] == 1
        assert "generated_at" in report
        assert report["cases"] == [_case_index_entry(records[0])]
        # The full per-arm claim/citation detail (ArmRecord.claims,
        # CitationCheckResult statuses, ...) must NOT be reachable from
        # report.json at all -- it lives solely in draws/<case_id>.json.
        assert "off" not in report["cases"][0]
        assert "on" not in report["cases"][0]
        assert "claims" not in report["cases"][0]
        assert report["summary"] == summarize(records)

    def test_session_ids_surfaced_and_deduplicated(self) -> None:
        """Issue #163 gate-2/Opus review: report.json must let a reader
        verify single-session provenance without opening any draws/ file."""
        off, on = _off_on(Verdict.VERIFIED, Verdict.VERIFIED)
        records = [
            _build_case_record(case_id="c1", off=off, on=on, session_id="session-a"),
            _build_case_record(case_id="c2", off=off, on=on, session_id="session-a"),
        ]

        report = build_report(records)

        assert report["session_ids"] == ["session-a"]

    def test_mixed_session_ids_are_visible_not_hidden(self) -> None:
        off, on = _off_on(Verdict.VERIFIED, Verdict.VERIFIED)
        records = [
            _build_case_record(case_id="c1", off=off, on=on, session_id="session-a"),
            _build_case_record(case_id="c2", off=off, on=on, session_id="session-b"),
        ]

        report = build_report(records)

        assert report["session_ids"] == ["session-a", "session-b"]

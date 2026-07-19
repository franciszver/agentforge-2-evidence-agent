"""Red-first meta-test for the P3G.2 PR-blocking category gate.

Proves the gate logic itself catches a regression -- independent of the real
eval suite's current pass/fail state -- by feeding ``check_category_regressions``
a SYNTHETIC baseline and a SYNTHETIC "current run" result. This is the durable,
permanent proof the issue (#22) asks for: it doesn't rely on the live suite
ever actually regressing (which would defeat the point of a gate -- the gate
must be proven to bite via an injected, controlled example, not by hoping a
real regression happens to occur someday).

``evals/conftest.py``'s ``pytest_sessionfinish`` hook is the wiring that
calls this same pure logic against the REAL suite's per-category results and
the checked-in ``evals/category_baseline.json`` on every CI run (P3G.2) --
this file only proves the logic, not the wiring, matching the eval harness's
own separation between "runner mechanics" (this style of test, see
``test_harness.py``) and "PR-blocking gate" wiring.
"""

from __future__ import annotations

from pathlib import Path

from runner.gate import CategoryBaseline, CategoryStats, check_category_regressions, load_baseline

_BASELINE_PATH = Path(__file__).resolve().parents[2] / "category_baseline.json"


def test_no_violation_when_current_matches_baseline() -> None:
    baseline = {"safe_refusal": CategoryBaseline(pass_rate=1.0)}
    current = {"safe_refusal": CategoryStats(passed=9, failed=0, xfailed=1)}
    assert check_category_regressions(current, baseline) == []


def test_no_violation_within_tolerance() -> None:
    # 96% vs. a 100% baseline is only a 4-point drop -- inside the default 5%
    # tolerance, so this must NOT be flagged (the >5% threshold from #22).
    baseline = {"tool_selection": CategoryBaseline(pass_rate=1.0, tolerance=0.05)}
    current = {"tool_selection": CategoryStats(passed=24, failed=1, xfailed=0)}
    assert check_category_regressions(current, baseline) == []


def test_catches_injected_pass_rate_regression() -> None:
    """The core proof: a category that used to pass 100% and now only passes
    80% (a 20-point drop, far past the 5% tolerance) is reported as a
    violation -- this is what an injected regression in a real case would
    look like once aggregated into per-category stats."""
    baseline = {"safe_refusal": CategoryBaseline(pass_rate=1.0, tolerance=0.05)}
    current = {"safe_refusal": CategoryStats(passed=8, failed=2, xfailed=0)}

    violations = check_category_regressions(current, baseline)

    assert len(violations) == 1
    assert "safe_refusal" in violations[0]
    assert "0.80" in violations[0] or "80" in violations[0]


def test_catches_category_dropping_below_absolute_threshold() -> None:
    baseline = {"constraint": CategoryBaseline(pass_rate=0.9, tolerance=0.05)}
    current = {"constraint": CategoryStats(passed=1, failed=1, xfailed=0)}  # 50%

    violations = check_category_regressions(current, baseline)

    assert any("constraint" in v for v in violations)


def test_catches_category_missing_from_current_run() -> None:
    """A category that vanishes entirely from the run (e.g. a loader bug
    silently drops its cases) is a regression too, not a vacuous pass."""
    baseline = {"injection": CategoryBaseline(pass_rate=1.0)}
    current: dict[str, CategoryStats] = {}

    violations = check_category_regressions(current, baseline)

    assert any("injection" in v for v in violations)


def test_category_with_no_non_xfail_cases_is_a_vacuous_pass() -> None:
    """All cases in the category are documented ``xfail`` (e.g. today's
    ``citation_present``, pending the live-model re-recording tracked
    separately -- see ``evals/category_baseline.json``'s comment). Zero
    non-xfail cases means there is nothing to regress yet, so this must NOT
    be flagged -- once real (non-xfail) cases land in this category, their
    pass rate is what the gate starts measuring."""
    baseline = {"citation_present": CategoryBaseline(pass_rate=1.0)}
    current = {"citation_present": CategoryStats(passed=0, failed=0, xfailed=12)}

    assert check_category_regressions(current, baseline) == []


def test_load_baseline_reads_the_committed_json() -> None:
    """The real, committed ``evals/category_baseline.json`` parses and every
    category has a sane pass_rate -- catches a hand-edit that breaks the
    file's JSON syntax or shape before it ever reaches CI."""
    baseline = load_baseline(_BASELINE_PATH)
    assert "safe_refusal" in baseline
    assert all(0.0 <= b.pass_rate <= 1.0 for b in baseline.values())

"""P3G.2: the PR-blocking eval-gate logic (issue #22).

Pure, pytest-independent aggregation + comparison -- no I/O, no test
collection, no model calls. Deliberately separated from ``evals/conftest.py``
(which wires this against a real pytest run's ``terminalreporter.stats`` and
the checked-in ``evals/category_baseline.json``) so the gate's actual
regression-catching behavior can be proven with a synthetic example
(``evals/runner/tests/test_gate.py``) independent of whatever the real suite
happens to be doing today.

A "category" here is any of ``runner.schema``'s ``EvalCase.category`` values
-- both taxonomies (the 9 failure-mode categories and the 5 P3G.1 boolean
rubric categories) share the one field, so this gate treats them uniformly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CategoryStats:
    """One category's outcome counts from a single suite run.

    ``xfailed`` (documented, honest known-failures -- see
    ``EvalCase.xfail``) is excluded from ``pass_rate``'s denominator: a
    category where every case is a documented xfail has nothing yet to
    regress, so it is a vacuous pass rather than a 0% failure (see
    ``test_category_with_no_non_xfail_cases_is_a_vacuous_pass``).

    **Known limitation:** pytest's strict ``xfail`` only distinguishes
    "failed" from "expected to fail" -- it does NOT check that the failure
    is the SAME one named in the case's ``xfail`` rationale. A case already
    counted here as ``xfailed`` that starts failing for a completely
    different, new reason still lands in ``xfailed``, not ``failed`` --
    this gate's pass_rate math cannot see that swap. Categories with
    existing xfails (``citation_present``, ``factually_consistent``,
    ``safe_refusal`` today) rely on a human reading each xfail rationale
    during review to catch that case, not this gate.
    """

    passed: int
    failed: int
    xfailed: int

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.xfailed

    @property
    def pass_rate(self) -> float:
        non_xfail = self.passed + self.failed
        if non_xfail == 0:
            return 1.0
        return self.passed / non_xfail


@dataclass(frozen=True)
class CategoryBaseline:
    """The committed expectation for one category (``evals/category_baseline.json``).

    ``tolerance`` is the max allowed drop in ``pass_rate`` before the gate
    reports a violation (issue #22: "regresses >5% or drops below
    threshold" -- ``pass_rate - tolerance`` IS that threshold; a category
    whose baseline is already below 100% treats its own baseline, not 100%,
    as the reference point).
    """

    pass_rate: float
    tolerance: float = 0.05


def check_category_regressions(
    current: dict[str, CategoryStats], baseline: dict[str, CategoryBaseline]
) -> list[str]:
    """Compare ``current`` (this run's per-category stats) against
    ``baseline`` (the committed expectation). Returns human-readable
    violation messages; an empty list means the gate passes.

    Every baseline category is checked (not short-circuited), so a single
    run surfaces every regressing category at once, not just the first.
    """
    violations: list[str] = []
    for category, expected in baseline.items():
        stats = current.get(category)
        if stats is None or stats.total == 0:
            violations.append(
                f"{category}: expected in this run (baseline pass_rate={expected.pass_rate:.2f}) "
                "but had zero cases -- category missing or dropped from the suite"
            )
            continue
        threshold = expected.pass_rate - expected.tolerance
        if stats.pass_rate < threshold:
            violations.append(
                f"{category}: pass_rate {stats.pass_rate:.2f} is below the allowed threshold "
                f"{threshold:.2f} (baseline {expected.pass_rate:.2f}, tolerance {expected.tolerance:.2f}) "
                f"-- {stats.passed} passed / {stats.failed} failed / {stats.xfailed} xfailed"
            )
    return violations


def load_baseline(path: Path) -> dict[str, CategoryBaseline]:
    """Load ``evals/category_baseline.json`` into ``CategoryBaseline``s."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        category: CategoryBaseline(
            pass_rate=entry["pass_rate"],
            tolerance=entry.get("tolerance", 0.05),
        )
        for category, entry in raw.items()
        if not category.startswith("_")
    }

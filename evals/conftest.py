"""Makes the copilot-agent service's ``app`` package importable from every
eval under this directory (harness code + case runner alike).

Same rationale as ``evals/tool_selection/conftest.py`` and its siblings:
``services/copilot-agent`` is a separate Python package (its own
``pyproject.toml``), so its path is added to ``sys.path`` here rather than
requiring the agent package to be installed system-wide. A single root-level
conftest covers ``evals/runner/`` and the YAML case runner (``evals/test_cases.py``)
without duplicating the snippet per subdirectory.

**P3G.2 (#22) PR-blocking category gate.** ``pytest_sessionfinish`` below
aggregates every test's outcome (this session's whole run -- the exact
``pytest evals/ -m "not integration"`` invocation CI already runs, see
``.github/workflows/copilot-ci.yml``) into per-``category`` marker
``runner.gate.CategoryStats``, compares them against the committed
``evals/category_baseline.json`` via ``runner.gate.check_category_regressions``,
and fails the whole run if any category regresses -- on top of the
per-case pass/fail pytest already enforces. Mutating ``session.exitstatus``
from ``pytest_sessionfinish`` is the same technique ``pytest-cov``'s
``--cov-fail-under`` uses (pytest reads ``session.exitstatus`` again after
every ``pytest_sessionfinish`` hookimpl has run), so this needs no separate
CI step or script -- the gate is inseparable from the eval run itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args

import pytest

_AGENT_ROOT = Path(__file__).resolve().parents[1] / "services" / "copilot-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from runner.gate import CategoryStats, check_category_regressions, load_baseline  # noqa: E402
from runner.schema import _CATEGORIES  # noqa: E402

_CATEGORY_NAMES = set(get_args(_CATEGORIES))
_BASELINE_PATH = Path(__file__).resolve().parent / "category_baseline.json"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is None:
        return  # pragma: no cover -- always registered under normal pytest runs

    counts: dict[str, dict[str, int]] = {}
    for outcome in ("passed", "failed", "xfailed"):
        for report in terminalreporter.stats.get(outcome, []):
            if report.when != "call":
                continue
            categories = _CATEGORY_NAMES.intersection(report.keywords)
            for category in categories:
                bucket = counts.setdefault(category, {"passed": 0, "failed": 0, "xfailed": 0})
                bucket[outcome] += 1

    current = {category: CategoryStats(**bucket) for category, bucket in counts.items()}
    baseline = load_baseline(_BASELINE_PATH)
    violations = check_category_regressions(current, baseline)
    if not violations:
        return

    terminalreporter.section("P3G.2 category regression gate (#22)")
    for violation in violations:
        terminalreporter.write_line(f"REGRESSION: {violation}", red=True)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED

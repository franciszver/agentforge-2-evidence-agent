"""Records ONE eval run's aggregate pass/fail/xfail counts into the
committed, agent-packaged history the dashboard chart reads (P4.10,
``docs/TEST_PLAN.md`` Sec 9: "their pass-rate results are committed, feeding
the dashboard chart and the README results table").

Runs every case under ``evals/cases/`` + ``evals/regressions/`` through the
same replay pipeline ``evals/test_cases.py`` uses (real
planner -> extraction -> verification, deterministic, offline -- see
``runner.ollama_replay``), classifies each case as ``passed`` / ``failed`` /
``xfailed`` (mirroring ``test_case_replay``'s strict-xfail semantics without
going through pytest), and appends one
:class:`app.dashboard_eval_history.EvalRunPoint` to
``services/copilot-agent/app/data/eval_history.json`` via
``append_eval_run``.

Usage (from repo root, after activating the copilot-agent venv):

    python evals/runner/record_run.py

Exits non-zero (after writing the record) if any case failed unexpectedly,
so a broken suite can't be recorded and forgotten.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _EVALS_ROOT.parent
_AGENT_ROOT = _REPO_ROOT / "services" / "copilot-agent"
for _root in (str(_AGENT_ROOT), str(_EVALS_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from app.dashboard_eval_history import EVAL_HISTORY_PATH, EvalRunPoint, append_eval_run  # noqa: E402

from runner.assertions import evaluate_assertions  # noqa: E402
from runner.loader import discover_case_files, load_case  # noqa: E402
from runner.ollama_replay import (  # noqa: E402
    RecordingExhaustedError,
    RecordingMismatchError,
    RecordingNotFoundError,
    ReplayOllamaClient,
    load_recording,
    recording_path,
)
from runner.pipeline import run_case  # noqa: E402

_CASES_DIR = _EVALS_ROOT / "cases"
_REGRESSIONS_DIR = _EVALS_ROOT / "regressions"
_RECORDINGS_DIR = _EVALS_ROOT / "recordings"

_REPLAY_ERRORS = (
    RecordingNotFoundError,
    RecordingExhaustedError,
    RecordingMismatchError,
    ValidationError,
)


def _case_outcome(case_file: Path, recordings_dir: Path) -> str:
    """``"passed"`` | ``"failed"`` | ``"xfailed"`` for one case.

    A case with no committed recording, or a recording that has diverged
    from the case -- a wrong call kind/order (see ``runner.ollama_replay``)
    OR a recorded payload that no longer validates against its extraction
    schema (``pydantic.ValidationError`` from ``schema.model_validate``) --
    is treated as a failing replay rather than raising -- one broken case
    shouldn't stop the whole run from being recorded, and it is honestly
    counted as a failure either way. A case marked ``xfail`` (P4.8) whose assertions genuinely fail is
    ``xfailed`` (a documented known-failure); one whose assertions
    unexpectedly PASS is a stale xfail and reported as ``failed`` --
    mirroring ``pytest.mark.xfail(strict=True)``'s semantics so a fixed
    model behavior can't hide inside the pass rate.
    """
    case = load_case(case_file)
    try:
        calls = load_recording(recording_path(recordings_dir, case.id))
        client = ReplayOllamaClient(calls)
        result = run_case(case, client)
        failures = evaluate_assertions(case, result)
    except _REPLAY_ERRORS:
        failures = ["replay error"]

    if case.xfail:
        return "xfailed" if failures else "failed"
    return "failed" if failures else "passed"


def aggregate_outcomes(outcomes: list[str], *, git_sha: str, timestamp: str) -> EvalRunPoint:
    """Pure aggregation: case outcomes -> one :class:`EvalRunPoint`.

    ``pass_rate = passed / total`` where ``total`` counts every case
    (passing + xfailed + failed) -- see ``app.dashboard_eval_history``'s
    module docstring for the accounting rationale.
    """
    total = len(outcomes)
    passed = outcomes.count("passed")
    failed = outcomes.count("failed")
    xfailed = outcomes.count("xfailed")
    pass_rate = (passed / total) if total else 0.0
    return EvalRunPoint(
        timestamp=timestamp,
        git_sha=git_sha,
        total=total,
        passed=passed,
        failed=failed,
        xfailed=xfailed,
        pass_rate=pass_rate,
    )


def compute_run_result(
    *,
    cases_dir: Path,
    regressions_dir: Path,
    recordings_dir: Path,
    git_sha: str,
    timestamp: str | None = None,
) -> EvalRunPoint:
    """Discover every case under ``cases_dir``/``regressions_dir``, classify
    each, and aggregate into one :class:`EvalRunPoint`."""
    case_files = discover_case_files(cases_dir, regressions_dir)
    outcomes = [_case_outcome(path, recordings_dir) for path in case_files]
    return aggregate_outcomes(
        outcomes,
        git_sha=git_sha,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )


def _git_short_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    point = compute_run_result(
        cases_dir=_CASES_DIR,
        regressions_dir=_REGRESSIONS_DIR,
        recordings_dir=_RECORDINGS_DIR,
        git_sha=_git_short_sha(),
    )
    append_eval_run(point, EVAL_HISTORY_PATH)
    print(
        f"[record_run] {point.timestamp} sha={point.git_sha} total={point.total} "
        f"passed={point.passed} failed={point.failed} xfailed={point.xfailed} "
        f"pass_rate={point.pass_rate:.3f} -> {EVAL_HISTORY_PATH}"
    )
    if point.failed:
        raise SystemExit(f"{point.failed} case(s) failed unexpectedly -- see output above")


if __name__ == "__main__":
    main()

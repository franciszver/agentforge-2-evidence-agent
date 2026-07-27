"""Regression guard for issue #208: verbose per-run eval logs must stay
gitignored, scoped to ``evals/results/`` only.

**What broke.** ``evals/results/issue163_run.log`` and
``evals/results/issue163_rerun.log`` were untracked and not gitignored, so
they surfaced in every ``git status`` for the whole session. The fix adds a
``.gitignore`` rule (``evals/results/**/*.log``) -- but a rule that is too
broad (e.g. a blanket ``*.log``) would silently hide logs elsewhere in the
repo that someone does want tracked, which is its own defect class.

**What this test proves, via ``git check-ignore`` (the real mechanism, not a
re-implementation of gitignore glob semantics):**

1. A ``.log`` file under ``evals/results/`` (including nested subdirs, since
   real results live under ``evals/results/issue-NNN/...``) IS ignored.
2. A ``.log`` file with the same basename OUTSIDE ``evals/results/`` is NOT
   ignored -- guards against a future broadening of the rule (e.g. someone
   "simplifying" it back to a blanket ``*.log``) silently hiding logs
   elsewhere.
3. A non-``.log`` file under ``evals/results/`` (e.g. the structured JSON
   measurement records like ``evals/results/issue-192/``) is NOT ignored --
   guards against a future broadening that would swallow tracked evidence.

Uses real (non-existent-on-disk) candidate paths -- ``git check-ignore``
evaluates gitignore rules against a path without requiring the file to
exist, so this stays hermetic and makes no filesystem writes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_ignored(relative_path: str) -> bool:
    """Return whether ``git check-ignore`` matches ``relative_path``.

    Exit code 0 == ignored, 1 == not ignored, anything else is a real error
    (e.g. git itself failing) and must not be swallowed.
    """
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", relative_path],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git check-ignore exited {result.returncode} for {relative_path!r}"
        )
    return result.returncode == 0


def test_log_under_evals_results_is_ignored() -> None:
    assert _is_ignored("evals/results/issue163_run.log")
    assert _is_ignored("evals/results/issue-999/draws/some_case.log")


def test_log_outside_evals_results_is_not_ignored() -> None:
    assert not _is_ignored("services/copilot-agent/foo.log")
    assert not _is_ignored("evals/foo.log")


def test_non_log_file_under_evals_results_is_not_ignored() -> None:
    assert not _is_ignored("evals/results/issue-192/README.md")
    assert not _is_ignored("evals/results/issue-999/draws/some_case.json")

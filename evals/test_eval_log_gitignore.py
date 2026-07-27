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
   real results live under ``evals/results/issue-NNN/...``) IS ignored, and
   the matching rule comes from this repo's ``.gitignore`` -- not from some
   other rule that happens to also match.
2. A ``.log`` file with the same basename OUTSIDE ``evals/results/`` is NOT
   ignored by this repo's ``.gitignore`` -- guards against a future
   broadening of the rule (e.g. someone "simplifying" it back to a blanket
   ``*.log``) silently hiding logs elsewhere.
3. A non-``.log`` file under ``evals/results/`` (e.g. the structured JSON
   measurement records like ``evals/results/issue-192/``) is NOT ignored by
   this repo's ``.gitignore`` -- guards against a future broadening that
   would swallow tracked evidence.

Uses real (non-existent-on-disk) candidate paths -- ``git check-ignore``
evaluates gitignore rules against a path without requiring the file to
exist, so this stays hermetic and makes no filesystem writes.

**Why ``-v`` and not ``--quiet``.** ``git check-ignore`` also honours a
developer's whole personal ignore stack: ``core.excludesFile`` (e.g.
``~/.config/git/ignore``) and ``.git/info/exclude``. A developer with a
global ``*.log`` rule would get a false-red on the "not ignored" assertions
below for reasons that have nothing to do with this repo's ``.gitignore`` --
and a bare boolean gives no hint why. ``-v`` reports which file and line
matched, so the "not ignored" tests assert specifically that no rule *from
this repo's ``.gitignore``* matches, while tolerating (and explaining) a
match from the developer's personal config. Do NOT reimplement gitignore
glob semantics with ``fnmatch`` or similar to sidestep this -- shelling out
to real git is exactly why the ``**``-matches-zero-directories and
``results-other/`` cases behave correctly; a hand-rolled matcher would get
those wrong.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The ``-v`` source label for a match against this repo's root .gitignore,
# given ``cwd=REPO_ROOT``. Any other value (a nested .gitignore, an absolute
# path to a global excludesFile, or ".git/info/exclude") is *not* this rule.
REPO_GITIGNORE_SOURCE = ".gitignore"


@dataclass(frozen=True)
class CheckIgnoreResult:
    ignored: bool
    source: str | None
    pattern: str | None
    raw: str


def _check_ignore(relative_path: str) -> CheckIgnoreResult:
    """Run ``git check-ignore -v`` and parse the matching source/pattern.

    Exit code 0 == a rule matched (stdout has one line), 1 == no rule
    matched (no stdout), anything else is a real error (e.g. git itself
    failing) and must not be swallowed.

    ``-v`` output format for a match is
    ``<source>:<lineno>:<pattern>\\t<path>`` -- parsed with explicit
    ``partition`` calls on the known delimiters, not a loose regex over the
    whole line.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-v", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git check-ignore exited {result.returncode} for {relative_path!r}: "
            f"stderr={result.stderr!r}"
        )
    if result.returncode == 1:
        return CheckIgnoreResult(ignored=False, source=None, pattern=None, raw="")

    line = result.stdout.rstrip("\n")
    header, _, matched_path = line.partition("\t")
    assert matched_path, f"unexpected git check-ignore -v output: {line!r}"
    source, _, remainder = header.partition(":")
    lineno, _, pattern = remainder.partition(":")
    assert lineno, f"unexpected git check-ignore -v header: {header!r}"
    return CheckIgnoreResult(ignored=True, source=source, pattern=pattern, raw=line)


def _ignored_by_repo_gitignore(relative_path: str) -> bool:
    """Whether ``relative_path`` is ignored specifically by this repo's
    root ``.gitignore`` (as opposed to some other rule in the developer's
    personal ignore stack)."""
    result = _check_ignore(relative_path)
    return result.ignored and result.source == REPO_GITIGNORE_SOURCE


def test_log_under_evals_results_is_ignored() -> None:
    for relative_path in (
        "evals/results/issue163_run.log",
        "evals/results/issue-999/draws/some_case.log",
    ):
        result = _check_ignore(relative_path)
        assert result.ignored, f"expected {relative_path!r} to be ignored"
        assert result.source == REPO_GITIGNORE_SOURCE, (
            f"expected {relative_path!r} to be ignored by "
            f"{REPO_GITIGNORE_SOURCE!r}, but the matching rule was "
            f"{result.raw!r} (source={result.source!r})"
        )


def test_log_outside_evals_results_is_not_ignored() -> None:
    for relative_path in ("services/copilot-agent/foo.log", "evals/foo.log"):
        result = _check_ignore(relative_path)
        assert result.source != REPO_GITIGNORE_SOURCE, (
            f"{relative_path!r} unexpectedly matched this repo's "
            f"{REPO_GITIGNORE_SOURCE!r} (rule: {result.raw!r}) -- the "
            "evals/results/**/*.log rule may have been broadened"
        )
        if result.ignored:
            # Matched, but by something other than this repo's .gitignore
            # (e.g. the developer's global core.excludesFile or
            # .git/info/exclude). That's the developer's own configuration,
            # not a defect in this repo, so it must not fail this test --
            # but make it obvious what happened if anyone goes looking.
            print(
                f"note: {relative_path!r} is ignored by {result.raw!r}, "
                "which is outside this repo's .gitignore (likely the "
                "developer's personal global excludes)"
            )


def test_non_log_file_under_evals_results_is_not_ignored() -> None:
    for relative_path in (
        "evals/results/issue-192/README.md",
        "evals/results/issue-999/draws/some_case.json",
    ):
        result = _check_ignore(relative_path)
        assert result.source != REPO_GITIGNORE_SOURCE, (
            f"{relative_path!r} unexpectedly matched this repo's "
            f"{REPO_GITIGNORE_SOURCE!r} (rule: {result.raw!r}) -- the "
            "evals/results/**/*.log rule may have been broadened to also "
            "swallow non-log files"
        )

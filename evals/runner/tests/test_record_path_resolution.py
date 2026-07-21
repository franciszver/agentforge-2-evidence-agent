"""Regression coverage for #119: ``evals/runner/record.py``'s ``sys.path``
setup must resolve to the LIVE ``app`` package in both layouts it runs
under -- a full monorepo checkout and the ``development-easy-agent-1`` dev
container's flattened layout (``/app`` IS the copilot-agent root, not
``<repo>/services/copilot-agent``) -- never silently fall through to a
stale, pre-built ``app`` copy shadowing it from ``site-packages``.

Two layers:

* ``_agent_root_candidates`` is a pure function of its arguments (no
  module-global/filesystem-location reads), so both layouts are exercised
  directly against synthetic directory trees, without needing an actual
  flattened container.
* the live smoke test proves the import that already happened when this
  test module (and ``runner.record``) loaded actually picked up the
  CURRENT ``app.planner.Planner.run`` -- whose signature gained the
  ``guideline_excerpts`` parameter in #105. A stale pre-#105 shadow copy
  would fail this assertion, so a future regression of this kind fails
  loudly here instead of surfacing only as a runtime ``TypeError`` deep in
  a live record run.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from app.planner import Planner

from runner.record import _agent_root_candidates, _in_flattened_container_layout_for


def test_agent_root_candidates_prefers_monorepo_layout_when_present(tmp_path: Path) -> None:
    """Full monorepo checkout: ``<repo>/services/copilot-agent/app/`` exists;
    ``<repo>`` itself does not directly hold an ``app`` package."""
    repo_root = tmp_path / "repo"
    monorepo_agent_root = repo_root / "services" / "copilot-agent"
    (monorepo_agent_root / "app").mkdir(parents=True)
    (monorepo_agent_root / "app" / "__init__.py").touch()

    candidates = _agent_root_candidates(repo_root, monorepo_agent_root)

    assert candidates[0] == monorepo_agent_root


def test_agent_root_candidates_prefers_flattened_container_layout_when_present(tmp_path: Path) -> None:
    """Flattened dev-container layout: ``/app`` (``repo_root`` here) IS the
    copilot-agent root directly -- the monorepo candidate
    (``repo_root/services/copilot-agent``) does not exist at all."""
    repo_root = tmp_path / "app"
    monorepo_agent_root = repo_root / "services" / "copilot-agent"
    (repo_root / "app").mkdir(parents=True)
    (repo_root / "app" / "__init__.py").touch()

    candidates = _agent_root_candidates(repo_root, monorepo_agent_root)

    assert candidates[0] == repo_root


def test_agent_root_candidates_falls_back_to_monorepo_guess_when_neither_exists(tmp_path: Path) -> None:
    """Neither candidate holds an ``app`` package on disk (e.g. a directory
    tree this test constructs but never populates) -- falls back to the
    monorepo guess, preserving the original behavior for an unrecognized
    layout rather than raising."""
    repo_root = tmp_path / "repo"
    monorepo_agent_root = repo_root / "services" / "copilot-agent"

    candidates = _agent_root_candidates(repo_root, monorepo_agent_root)

    assert candidates[0] == monorepo_agent_root


# --- _in_flattened_container_layout_for (gate review on #143): the fail- --
# closed guard's crux comparison, made directly unit-testable by taking the
# candidate roots as parameters (mirroring _agent_root_candidates itself)
# instead of only being reachable by monkeypatching the whole
# ``_in_flattened_container_layout`` function away. Without this, an
# inverted comparison (``== monorepo_agent_root`` instead of ``==
# repo_root``) would reopen the exact fail-open hole #143 closed while every
# existing test (which monkeypatches the function wholesale) stayed green. -


def test_in_flattened_container_layout_for_true_under_flattened_layout(tmp_path: Path) -> None:
    """Flattened dev-container layout: ``repo_root`` IS the copilot-agent
    root directly -- the monorepo candidate does not exist on disk, so it
    loses the candidate race and the flattened layout is flagged."""
    repo_root = tmp_path / "app"
    monorepo_agent_root = repo_root / "services" / "copilot-agent"
    (repo_root / "app").mkdir(parents=True)
    (repo_root / "app" / "__init__.py").touch()

    assert _in_flattened_container_layout_for(repo_root, monorepo_agent_root) is True


def test_in_flattened_container_layout_for_false_under_monorepo_layout(tmp_path: Path) -> None:
    """Full monorepo/host checkout: the monorepo candidate wins the
    candidate race, so the flattened-container layout is NOT flagged."""
    repo_root = tmp_path / "repo"
    monorepo_agent_root = repo_root / "services" / "copilot-agent"
    (monorepo_agent_root / "app").mkdir(parents=True)
    (monorepo_agent_root / "app" / "__init__.py").touch()

    assert _in_flattened_container_layout_for(repo_root, monorepo_agent_root) is False


def test_in_flattened_container_layout_for_false_when_neither_layout_present(tmp_path: Path) -> None:
    """Neither candidate holds an ``app`` package on disk -- falls back to
    the monorepo guess (same as ``_agent_root_candidates`` itself), so the
    flattened layout is NOT flagged for an unrecognized tree."""
    repo_root = tmp_path / "repo"
    monorepo_agent_root = repo_root / "services" / "copilot-agent"

    assert _in_flattened_container_layout_for(repo_root, monorepo_agent_root) is False


def test_record_module_imports_live_app_package_not_a_stale_shadow() -> None:
    """Proves ``runner.record`` (already imported above, via this test
    module's own ``from app.planner import Planner`` / ``from runner.record
    import ...``) picked up the CURRENT ``app`` package -- not a stale,
    pre-#105 shadow copy. A stale copy's ``Planner.run(self, question)``
    lacks ``guideline_excerpts`` entirely and would fail this assertion."""
    params = inspect.signature(Planner.run).parameters
    assert "guideline_excerpts" in params

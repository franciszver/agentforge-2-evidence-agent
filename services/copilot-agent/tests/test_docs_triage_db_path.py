"""Pins ``docs/TEST_PLAN.md``'s documented triage command (#179) against the
values it must stay true to, so a future change to either side breaks this
test instead of silently leaving stale prose behind.

**Why this exists.** #179's triage workflow is documented, not code -- there
is no runtime path that exercises it. This repo has repeatedly shipped docs
that were true when written and silently drifted (the exact failure mode
#179 itself was filed to fix, after #176's redaction orphaned a workflow
description). Two facts the documented ``docker exec`` command depends on
are each independently mechanical to check without parsing markdown
structure: the trace-store path it queries, and the container/service name
it targets. This test greps both source files as plain text -- no markdown
parser, tolerant of prose changes around them -- and fails loudly if either
drifts out from under the doc.

**What this deliberately does NOT do.** It does not execute the documented
command (that needs a running ``agent`` container -- verified by hand for
this PR, see the PR description) or validate the rest of the doc's prose.
It only pins the two facts most likely to silently rot: the DB path and the
compose service name.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_PLAN = _REPO_ROOT / "docs" / "TEST_PLAN.md"
_COPILOT_COMPOSE = _REPO_ROOT / "docker" / "development-easy" / "docker-compose.copilot.yml"


def _trace_db_path_from_compose() -> str:
    for line in _COPILOT_COMPOSE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("TRACE_DB_PATH:"):
            return stripped.split(":", 1)[1].strip()
    raise AssertionError("TRACE_DB_PATH not found in docker-compose.copilot.yml's agent service")


def test_repo_layout_assumptions_hold() -> None:
    # Guards the two path constants above: if either file moves, the
    # assertions below would otherwise fail with a confusing
    # FileNotFoundError instead of naming the real problem.
    assert _TEST_PLAN.is_file(), f"expected {_TEST_PLAN} to exist"
    assert _COPILOT_COMPOSE.is_file(), f"expected {_COPILOT_COMPOSE} to exist"


def test_documented_triage_db_path_matches_the_compose_override() -> None:
    """The #179 triage command in TEST_PLAN.md queries
    ``/data/traces/traces.db`` -- the compose overlay's TRACE_DB_PATH
    override, NOT ``app.config.Settings.trace_db_path``'s bare
    ``/data/traces.db`` default. If either changes, this must change too."""
    compose_path = _trace_db_path_from_compose()
    test_plan_text = _TEST_PLAN.read_text(encoding="utf-8")

    assert compose_path == "/data/traces/traces.db", (
        "docker-compose.copilot.yml's agent TRACE_DB_PATH changed -- "
        "update the #179 triage command in docs/TEST_PLAN.md to match "
        f"(now {compose_path!r})."
    )
    assert compose_path in test_plan_text, (
        f"docs/TEST_PLAN.md's #179 triage command does not mention {compose_path!r} "
        "-- it has drifted from docker-compose.copilot.yml's agent TRACE_DB_PATH."
    )


def test_documented_triage_container_name_matches_the_compose_service() -> None:
    """The #179 triage command execs into ``development-easy-agent-1`` --
    the ``development-easy_`` compose project's ``agent`` service. If the
    compose service is ever renamed, this must change too."""
    compose_text = _COPILOT_COMPOSE.read_text(encoding="utf-8")
    test_plan_text = _TEST_PLAN.read_text(encoding="utf-8")

    assert "\n  agent:\n" in compose_text, (
        "docker-compose.copilot.yml's agent service was renamed or restructured -- "
        "update the #179 triage command's container name in docs/TEST_PLAN.md to match."
    )
    assert "development-easy-agent-1" in test_plan_text

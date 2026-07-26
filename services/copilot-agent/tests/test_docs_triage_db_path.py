"""Pins ``docs/TEST_PLAN.md``'s documented triage command (#179) -- and the
matching copy in ``tests/Tests/E2e/ClinicalCopilotFeedbackTest.php``'s
docblock (#180) -- against the values they must stay true to, so a future
change to any of the three files breaks this test instead of silently
leaving stale prose behind.

**Why this exists.** #179's triage workflow is documented, not code -- there
is no runtime path that exercises it. This repo has repeatedly shipped docs
that were true when written and silently drifted (the exact failure mode
#179 itself was filed to fix, after #176's redaction orphaned a workflow
description). Facts the documented ``docker exec`` command depends on are
each independently mechanical to check without parsing markdown/PHP
structure: the trace-store path it queries, and the container/service name
it targets. This test greps all three source files as plain text -- no
markdown or PHP parser, tolerant of prose changes around them -- and fails
loudly if any drifts out from under the others.

**Comment/quoting tolerance (review finding).** The compose-side parse
strips a trailing ``# ...`` comment and surrounding quotes before comparing
the ``TRACE_DB_PATH`` value, and the service-name check matches per-line
with a regex rather than a raw multi-line substring. Both are
semantic-preserving edits (adding an inline comment, quoting the value, a
reformatting pass) that must NOT trip this test -- a doc-drift pin that
cries wolf on harmless reformatting gets deleted, which is worse than not
having it.

**What this deliberately does NOT do.** It does not execute the documented
command (that needs a running ``agent`` container -- verified by hand for
this PR, see the PR description) or validate the rest of either doc's
prose. It only pins the facts most likely to silently rot: the DB path and
the container/service name, in both places that state them.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_PLAN = _REPO_ROOT / "docs" / "TEST_PLAN.md"
_COPILOT_COMPOSE = _REPO_ROOT / "docker" / "development-easy" / "docker-compose.copilot.yml"
_FEEDBACK_E2E_TEST = _REPO_ROOT / "tests" / "Tests" / "E2e" / "ClinicalCopilotFeedbackTest.php"

_EXPECTED_TRACE_DB_PATH = "/data/traces/traces.db"
_EXPECTED_CONTAINER_NAME = "development-easy-agent-1"

# Matches a compose service block's key line, e.g. "  agent:" -- optionally
# followed by trailing whitespace or a "# ..." comment, so an appended
# comment or a formatting pass doesn't break the match (review finding).
_AGENT_SERVICE_RE = re.compile(r"^  agent:\s*(#.*)?$", re.MULTILINE)


def _trace_db_path_from_compose() -> str:
    for line in _COPILOT_COMPOSE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("TRACE_DB_PATH:"):
            continue
        value = stripped.split(":", 1)[1]
        # Strip a trailing "# ..." comment (only outside of quotes would be
        # fully correct YAML-wise, but this value is never expected to
        # contain a literal "#", so a plain split is sufficient here).
        value = value.split("#", 1)[0].strip()
        # Strip surrounding quotes -- YAML allows `"..."` / `'...'` around
        # a scalar without changing its value.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value
    raise AssertionError("TRACE_DB_PATH not found in docker-compose.copilot.yml's agent service")


def test_repo_layout_assumptions_hold() -> None:
    # Guards the path constants above: if any file moves, the assertions
    # below would otherwise fail with a confusing FileNotFoundError instead
    # of naming the real problem.
    assert _TEST_PLAN.is_file(), f"expected {_TEST_PLAN} to exist"
    assert _COPILOT_COMPOSE.is_file(), f"expected {_COPILOT_COMPOSE} to exist"
    assert _FEEDBACK_E2E_TEST.is_file(), f"expected {_FEEDBACK_E2E_TEST} to exist"


def test_documented_triage_db_path_matches_the_compose_override() -> None:
    """The #179 triage command in TEST_PLAN.md (and the matching #180
    command in ClinicalCopilotFeedbackTest.php's docblock) queries
    ``/data/traces/traces.db`` -- the compose overlay's TRACE_DB_PATH
    override, NOT ``app.config.Settings.trace_db_path``'s bare
    ``/data/traces.db`` default. If any changes, all must change together."""
    compose_path = _trace_db_path_from_compose()
    test_plan_text = _TEST_PLAN.read_text(encoding="utf-8")
    e2e_test_text = _FEEDBACK_E2E_TEST.read_text(encoding="utf-8")

    assert compose_path == _EXPECTED_TRACE_DB_PATH, (
        "docker-compose.copilot.yml's agent TRACE_DB_PATH changed -- "
        "update the #179/#180 triage commands in docs/TEST_PLAN.md and "
        "ClinicalCopilotFeedbackTest.php's docblock to match "
        f"(now {compose_path!r})."
    )
    assert compose_path in test_plan_text, (
        f"docs/TEST_PLAN.md's #179 triage command does not mention {compose_path!r} "
        "-- it has drifted from docker-compose.copilot.yml's agent TRACE_DB_PATH."
    )
    assert compose_path in e2e_test_text, (
        f"ClinicalCopilotFeedbackTest.php's docblock does not mention {compose_path!r} "
        "-- it has drifted from docker-compose.copilot.yml's agent TRACE_DB_PATH."
    )


def test_documented_triage_container_name_matches_the_compose_service() -> None:
    """The #179/#180 triage commands exec into ``development-easy-agent-1``
    -- the ``development-easy_`` compose project's ``agent`` service. If the
    compose service is ever renamed, both documented copies must change
    too."""
    compose_text = _COPILOT_COMPOSE.read_text(encoding="utf-8")
    test_plan_text = _TEST_PLAN.read_text(encoding="utf-8")
    e2e_test_text = _FEEDBACK_E2E_TEST.read_text(encoding="utf-8")

    assert _AGENT_SERVICE_RE.search(compose_text), (
        "docker-compose.copilot.yml's agent service was renamed or restructured -- "
        "update the #179/#180 triage commands' container name to match."
    )
    assert _EXPECTED_CONTAINER_NAME in test_plan_text
    assert _EXPECTED_CONTAINER_NAME in e2e_test_text


def test_documented_triage_notes_no_sqlite3_cli_in_the_agent_image() -> None:
    """Both documented copies use ``python3 -c "import sqlite3; ..."``, NOT
    a bare ``sqlite3`` CLI invocation -- the agent image (python:3.11-slim)
    has no ``sqlite3`` binary. Pins the rationale text so a future edit
    that "simplifies" the recipe back to a bare CLI form is caught."""
    # PHP docblock wraps mid-phrase across " * "-prefixed lines (e.g. "no
    # `sqlite3`\n * CLI installed"), so compare against whitespace-collapsed,
    # backtick-stripped text rather than a literal substring -- otherwise
    # this pin itself would be one reflow away from a false failure.
    normalize = lambda text: " ".join(text.replace("`", "").split())  # noqa: E731
    test_plan_norm = normalize(_TEST_PLAN.read_text(encoding="utf-8"))
    e2e_test_norm = normalize(_FEEDBACK_E2E_TEST.read_text(encoding="utf-8"))

    assert "no sqlite3 CLI" in test_plan_norm
    assert "no sqlite3" in e2e_test_norm and "CLI" in e2e_test_norm

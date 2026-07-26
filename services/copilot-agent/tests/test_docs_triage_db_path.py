"""Pins ``docs/TEST_PLAN.md``'s documented triage command (#179) -- and the
matching copy in ``tests/Tests/E2e/ClinicalCopilotFeedbackTest.php``'s
docblock (#180) -- against the values they must stay true to, so a future
change to any of the three files breaks this test instead of silently
leaving stale prose behind.

**Why this exists.** #179's triage workflow is documented, not code -- there
is no runtime path that exercises it. This repo has repeatedly shipped docs
that were true when written and silently drifted (the exact failure mode
#179 itself was filed to fix, after #176's redaction orphaned a workflow
description).

**Scoped to the COMMAND, not "anywhere in the file" (review finding,
verified by mutation).** An earlier version of this test asserted each
literal appeared somewhere in the file. Every file states each literal
more than once (an explanatory bullet, an unrelated #119 recipe, prose
elsewhere), so the DOCUMENTED COMMAND could drift to a wrong value --
including silently back to ``app.config.Settings.trace_db_path``'s bare
``/data/traces.db`` default, the exact mistake this doc exists to
prevent -- while every whole-file assertion stayed green. This version
extracts the fenced ```bash block under the "### Triage workflow" heading
(markdown) and the backtick-quoted ``docker exec`` command inside the PHP
docblock, and checks the literals against THOSE substrings specifically.

**Comment/quoting/list-form tolerance (review findings).** The compose-side
parse strips a trailing ``# ...`` comment and surrounding quotes before
comparing the ``TRACE_DB_PATH`` value, accepts either the mapping
(``TRACE_DB_PATH: ...``) or list (``- TRACE_DB_PATH=...``) form of
``environment:``, and is scoped to the ``agent`` service block specifically
(not a whole-file scan, which would misattribute a match from an unrelated
service or emit a wrong "not found in the agent service" error). The
service-name check matches per-line with a regex rather than a raw
multi-line substring. All of these are semantic-preserving edits (adding an
inline comment, quoting the value, switching env-var syntax, a reformatting
pass) that must NOT trip this test -- a doc-drift pin that cries wolf on
harmless reformatting gets deleted, which is worse than not having it.

**What this deliberately does NOT do.** It does not execute the documented
command (that needs a running ``agent`` container -- verified by hand for
this PR, see the PR description) or validate the rest of either doc's
prose. It pins: the DB path and container/service name inside the
documented command specifically (not just present somewhere in the file),
the "no sqlite3 CLI" rationale, and the #179 decision's load-bearing
"agent port never published, copilot_internal-only" premise.
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

_TRIAGE_HEADING = "### Triage workflow: reading the clinician's `feedback_comment` (#179)"

# Matches a compose service block's key line, e.g. "  agent:" -- optionally
# followed by trailing whitespace or a "# ..." comment, so an appended
# comment or a formatting pass doesn't break the match (review finding).
_AGENT_SERVICE_RE = re.compile(r"^  agent:\s*(#.*)?$", re.MULTILINE)

# Any sibling top-level compose key: a 2-space-indented service name (e.g.
# "  ollama:") or a 0-indent top-level section (e.g. "networks:"). Used to
# find where the agent service block ENDS, since agent is not guaranteed to
# stay the last service in the file.
_NEXT_TOP_LEVEL_KEY_RE = re.compile(r"^(?:  [A-Za-z_][\w-]*:|[A-Za-z_][\w-]*:)\s*(#.*)?$", re.MULTILINE)


def _agent_service_block(compose_text: str) -> str:
    """The ``agent:`` service's own lines, bounded so a scan for
    ``TRACE_DB_PATH``/``ports:``/``networks:`` can't accidentally match a
    different service (review finding: the prior version scanned the whole
    file)."""
    start_match = _AGENT_SERVICE_RE.search(compose_text)
    if not start_match:
        raise AssertionError("docker-compose.copilot.yml has no `  agent:` service block")
    end_match = _NEXT_TOP_LEVEL_KEY_RE.search(compose_text, start_match.end())
    end = end_match.start() if end_match else len(compose_text)
    return compose_text[start_match.start() : end]


def _trace_db_path_from_compose() -> str:
    block = _agent_service_block(_COPILOT_COMPOSE.read_text(encoding="utf-8"))
    for line in block.splitlines():
        stripped = line.strip()
        raw_value: str | None = None
        if stripped.startswith("TRACE_DB_PATH:"):
            raw_value = stripped.split(":", 1)[1]
        elif stripped.startswith("- TRACE_DB_PATH="):
            raw_value = stripped[len("- TRACE_DB_PATH=") :]
        if raw_value is None:
            continue
        # Strip a trailing "# ..." comment (only outside of quotes would be
        # fully correct YAML-wise, but this value is never expected to
        # contain a literal "#", so a plain split is sufficient here).
        value = raw_value.split("#", 1)[0].strip()
        # Strip surrounding quotes -- YAML allows `"..."` / `'...'` around
        # a scalar without changing its value.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value
    raise AssertionError("TRACE_DB_PATH not found in the agent service block of docker-compose.copilot.yml")


def _extract_test_plan_command() -> str:
    """The fenced ```bash block under the "### Triage workflow" heading in
    TEST_PLAN.md -- NOT the surrounding explanatory prose, which restates
    the same literals and would let the command itself drift undetected
    (review finding, verified by mutation)."""
    text = _TEST_PLAN.read_text(encoding="utf-8")
    heading_index = text.find(_TRIAGE_HEADING)
    if heading_index == -1:
        raise AssertionError(
            f"docs/TEST_PLAN.md's {_TRIAGE_HEADING!r} heading not found -- "
            "the #179 triage section was renamed, moved, or deleted."
        )
    match = re.search(r"```bash\n(.*?)```", text[heading_index:], re.DOTALL)
    if not match:
        raise AssertionError(
            "no fenced ```bash block found under the #179 Triage workflow heading in docs/TEST_PLAN.md"
        )
    return match.group(1)


def _dedent_php_docblock(text: str) -> str:
    """Strip a PHP docblock's leading ``/**``, ``*/``, and per-line `` * ``
    comment prefixes, returning the remaining content with line breaks
    collapsed to single spaces. Backticks are preserved -- callers that want
    them stripped too (for prose phrase-matching) do that separately; a
    caller extracting a backtick-quoted command needs them intact. Without
    the docblock-prefix stripping this does, a backtick-quoted command that
    line-wraps mid-token (e.g. ``no `sqlite3`\\n * CLI installed``) can't be
    matched as one string, and prose that reflows across lines differently
    produces spurious diffs (review finding)."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^/\*\*$", "", stripped)
        stripped = re.sub(r"^\*/$", "", stripped)
        stripped = re.sub(r"^\*\s?", "", stripped)
        lines.append(stripped)
    return " ".join(" ".join(lines).split())


def _normalize_prose(text: str) -> str:
    """Dedent (if it's a PHP docblock -- a no-op otherwise, since markdown
    has no `` * `` line-comment prefix) and strip backticks, for tolerant
    phrase matching against rationale text rather than an exact literal
    command."""
    return _dedent_php_docblock(text).replace("`", "")


def _extract_e2e_docblock_command() -> str:
    """The backtick-quoted ``docker exec ...`` command inside
    ClinicalCopilotFeedbackTest.php's docblock -- NOT the surrounding prose,
    which restates the same literals (review finding, verified by
    mutation)."""
    normalized = _dedent_php_docblock(_FEEDBACK_E2E_TEST.read_text(encoding="utf-8"))
    match = re.search(r"`(docker exec[^`]*)`", normalized)
    if not match:
        raise AssertionError(
            "no backtick-quoted `docker exec ...` command found in "
            "ClinicalCopilotFeedbackTest.php's docblock"
        )
    return match.group(1)


def test_repo_layout_assumptions_hold() -> None:
    # Guards the path constants above: if any file moves, the assertions
    # below would otherwise fail with a confusing FileNotFoundError instead
    # of naming the real problem.
    assert _TEST_PLAN.is_file(), f"expected {_TEST_PLAN} to exist"
    assert _COPILOT_COMPOSE.is_file(), f"expected {_COPILOT_COMPOSE} to exist"
    assert _FEEDBACK_E2E_TEST.is_file(), f"expected {_FEEDBACK_E2E_TEST} to exist"


def test_documented_triage_db_path_matches_the_compose_override() -> None:
    """The #179 triage COMMAND in TEST_PLAN.md (and the #180 command in
    ClinicalCopilotFeedbackTest.php's docblock) queries
    ``/data/traces/traces.db`` -- the compose overlay's TRACE_DB_PATH
    override, NOT ``app.config.Settings.trace_db_path``'s bare
    ``/data/traces.db`` default. Checked against the extracted command
    text specifically, not "somewhere in the file" -- a whole-file check
    would stay green if the command itself silently regressed to the
    config default while an explanatory bullet elsewhere still named the
    right path (review finding, verified by mutation)."""
    compose_path = _trace_db_path_from_compose()
    test_plan_command = _extract_test_plan_command()
    e2e_command = _extract_e2e_docblock_command()

    assert compose_path == _EXPECTED_TRACE_DB_PATH, (
        "docker-compose.copilot.yml's agent TRACE_DB_PATH changed -- "
        "update the #179/#180 triage commands in docs/TEST_PLAN.md and "
        "ClinicalCopilotFeedbackTest.php's docblock to match "
        f"(now {compose_path!r})."
    )
    assert compose_path in test_plan_command, (
        f"docs/TEST_PLAN.md's #179 triage COMMAND does not use {compose_path!r} "
        "-- it has drifted from docker-compose.copilot.yml's agent TRACE_DB_PATH "
        "(this checks the fenced command itself, not the surrounding prose)."
    )
    assert compose_path in e2e_command, (
        f"ClinicalCopilotFeedbackTest.php's docblock command does not use {compose_path!r} "
        "-- it has drifted from docker-compose.copilot.yml's agent TRACE_DB_PATH "
        "(this checks the backtick-quoted command itself, not the surrounding prose)."
    )


def test_documented_triage_container_name_matches_the_compose_service() -> None:
    """The #179/#180 triage COMMANDS exec into ``development-easy-agent-1``
    -- the ``development-easy_`` compose project's ``agent`` service.
    Checked against the extracted command text specifically (see the
    DB-path test's docstring for why "somewhere in the file" is not
    sufficient -- an unrelated #119 recipe elsewhere in TEST_PLAN.md also
    names an agent container and would mask a wrong name in the actual
    triage command, as would this file's own explanatory bullet)."""
    compose_text = _COPILOT_COMPOSE.read_text(encoding="utf-8")
    test_plan_command = _extract_test_plan_command()
    e2e_command = _extract_e2e_docblock_command()

    assert _AGENT_SERVICE_RE.search(compose_text), (
        "docker-compose.copilot.yml's agent service was renamed or restructured -- "
        "update the #179/#180 triage commands' container name to match."
    )
    assert _EXPECTED_CONTAINER_NAME in test_plan_command, (
        f"docs/TEST_PLAN.md's #179 triage COMMAND does not exec into "
        f"{_EXPECTED_CONTAINER_NAME!r} (checked against the fenced command itself)."
    )
    assert _EXPECTED_CONTAINER_NAME in e2e_command, (
        f"ClinicalCopilotFeedbackTest.php's docblock command does not exec into "
        f"{_EXPECTED_CONTAINER_NAME!r} (checked against the backtick-quoted command itself)."
    )


def test_documented_triage_notes_no_sqlite3_cli_in_the_agent_image() -> None:
    """Both documented copies use ``python3 -c "import sqlite3; ..."``, NOT
    a bare ``sqlite3`` CLI invocation -- the agent image (python:3.11-slim)
    has no ``sqlite3`` binary. Pins the rationale text so a future edit
    that "simplifies" the recipe back to a bare CLI form is caught.

    Uses the same PHP-docblock-prefix-stripping normalizer the command
    extractor uses, plus backtick stripping (review finding: an earlier
    version normalized only whitespace, not the docblock's `` * `` line
    prefixes, so it collapsed to "...has no sqlite3 * CLI installed" --
    which both (a) could never match the intended single phrase, forcing a
    weaker two-substring check that would pass even if the rationale were
    reworded to drop the actual claim, and (b) would itself FAIL on a
    harmless reflow that moved the line break one word earlier, i.e. it
    cried wolf on exactly the reformatting it was meant to tolerate)."""
    test_plan_norm = _normalize_prose(_TEST_PLAN.read_text(encoding="utf-8"))
    e2e_test_norm = _normalize_prose(_FEEDBACK_E2E_TEST.read_text(encoding="utf-8"))

    assert "no sqlite3 CLI" in test_plan_norm
    assert "no sqlite3 CLI" in e2e_test_norm


def test_agent_service_has_no_published_ports_and_stays_internal_only() -> None:
    """The #179 decision's load-bearing premise (docs/TEST_PLAN.md's
    "Decision" paragraph) is that the agent's port is never published to
    the host and the service is reachable only over ``copilot_internal`` --
    that's WHY ``docker exec`` from the host is treated as a strict
    superset of what an authenticated HTTP proxy would grant. Add a
    ``ports:`` mapping here (e.g. for ad-hoc debugging) or a second network
    with outside reach, and that premise silently stops holding while
    nothing else in this test file would notice (review finding)."""
    block = _agent_service_block(_COPILOT_COMPOSE.read_text(encoding="utf-8"))

    assert not re.search(r"^\s*ports:\s*(#.*)?$", block, re.MULTILINE), (
        "docker-compose.copilot.yml's agent service now publishes a ports: "
        "mapping -- this falsifies the #179 decision's 'agent port is never "
        "published' premise (docs/TEST_PLAN.md's Decision section). That "
        "decision needs re-deriving, not just this test fixed."
    )

    networks_match = re.search(r"^    networks:\n((?:^      -.*\n?)+)", block, re.MULTILINE)
    assert networks_match, "agent service's `networks:` block not found in the expected shape"
    networks = [line.strip()[2:] for line in networks_match.group(1).splitlines() if line.strip()]
    assert networks == ["copilot_internal"], (
        f"agent service networks changed to {networks!r} -- if it gained a network "
        "with host/internet reach, the #179 decision's isolation premise needs "
        "re-checking, not just this test updated to match."
    )

"""Pins the two preconditions issue #188's documented accept rests on:
``copilot_per_user_token_enabled`` defaults ``False`` in ``app/config.py``,
and neither compose file sets ``COPILOT_PER_USER_TOKEN_ENABLED`` to turn it
on.

**Why this exists.** #188 measured that an unauthenticated flood of
``/chat`` requests, once the per-user-token flag is ON, drives
``_validate_token`` (``app/chat.py``) to dispatch a real, timeout-bounded
introspection HTTP call via ``run_in_threadpool`` for every request --
because an attacker-chosen token is a guaranteed ``peek_cached`` miss (only
POSITIVE introspection results are cached, per ``app/introspection.py``'s
own docstring). Measured: 40 concurrent unauthenticated requests drove
40/40 concurrent validator calls against anyio's process-wide default
thread limiter (capacity 40, no override anywhere in ``app/`` or
``tests/``) -- watermark == capacity -- with real ``/health`` latency +1.25s
while a pure-async control route was unaffected (0.0000s); at 80 requests
(2x capacity), all queued, zero errors, tail latency ~= 2x timeout.

The OWNER DECISION on #188 is **documented accept, not mitigated** --
recorded in ``docs/ARCHITECTURE.md``'s "Path to Production" section --
specifically because the flag defaults OFF today (flag-off routes to
``_fail_closed_token_validator``/``_dev_permissive_token_validator``,
neither of which does any I/O, so there is no threadpool exposure at all
in the shipped default). That premise is exactly what this test pins.

**What this test does NOT do.** It does not exercise the threadpool
exhaustion itself (that is #188's own measurement, run by hand against a
live stack, not a hermetic unit test) and it does not re-derive the
mitigation options. It exists solely so that if a future change flips the
shipped default or a compose file's override, this test fails LOUDLY and
names the reopened issue -- instead of the accept's premise silently
stopping to hold while docs/ARCHITECTURE.md still describes the old,
now-false state.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PY = _REPO_ROOT / "services" / "copilot-agent" / "app" / "config.py"
_COPILOT_COMPOSE = _REPO_ROOT / "docker" / "development-easy" / "docker-compose.copilot.yml"

_REOPEN_MESSAGE = (
    "issue #188's documented accept assumed this flag ships OFF -- if it is "
    "being turned on, the threadpool starvation mitigation is now a "
    "blocking pre-condition."
)

# Matches the Settings field declaration, e.g.
# "    copilot_per_user_token_enabled: bool = False" -- tolerant of trailing
# whitespace or an inline "# ..." comment, so a harmless reformatting pass
# doesn't trip this test.
_CONFIG_DEFAULT_RE = re.compile(
    r"^\s*copilot_per_user_token_enabled:\s*bool\s*=\s*(\w+)\s*(#.*)?$",
    re.MULTILINE,
)

# Any assignment of the env var inside a compose ``environment:`` block, in
# either mapping (``COPILOT_PER_USER_TOKEN_ENABLED: "true"``) or list
# (``- COPILOT_PER_USER_TOKEN_ENABLED=true``) form -- deliberately NOT
# scoped to a single service, since the accept's premise is that NEITHER
# compose file (nor any service within it) sets the flag, not just that the
# ``agent`` service block happens not to.
_COMPOSE_ENV_SET_RE = re.compile(
    r"^\s*(?:-\s*)?COPILOT_PER_USER_TOKEN_ENABLED\s*[:=]\s*[\"']?(\w+)",
    re.MULTILINE,
)


def test_repo_layout_assumptions_hold() -> None:
    assert _CONFIG_PY.is_file(), f"expected {_CONFIG_PY} to exist"
    assert _COPILOT_COMPOSE.is_file(), f"expected {_COPILOT_COMPOSE} to exist"


def test_copilot_per_user_token_enabled_defaults_false() -> None:
    """#188's accept rests on the flag-off code path -- ``_validate_token``
    never dispatches to the threadpool when ``get_token_validator`` returns
    the fail-closed or dev-permissive stub, since neither exposes
    ``peek_cached`` and neither does I/O. Verified by mutation (flip the
    default to ``True`` locally): fails."""
    text = _CONFIG_PY.read_text(encoding="utf-8")
    match = _CONFIG_DEFAULT_RE.search(text)
    assert match, (
        "app/config.py's `copilot_per_user_token_enabled` field declaration "
        "was not found in the expected `field: bool = <value>` shape -- "
        f"update this test's regex to match. {_REOPEN_MESSAGE}"
    )
    assert match.group(1) == "False", (
        "app/config.py's `copilot_per_user_token_enabled` no longer "
        f"defaults to False (now {match.group(1)!r}). {_REOPEN_MESSAGE}"
    )


def test_no_compose_file_enables_copilot_per_user_token() -> None:
    """The dev compose overlay must not flip the flag on as a side effect of
    an unrelated env-var change -- today it only appears in an explanatory
    comment (see the ``COPILOT_DEV_ACCEPT_ANY_BEARER_TOKEN`` block in
    ``docker-compose.copilot.yml``), never as a live ``environment:``
    assignment."""
    text = _COPILOT_COMPOSE.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = _COMPOSE_ENV_SET_RE.match(stripped)
        if match is None:
            continue
        value = match.group(1).strip().rstrip("\"'").lower()
        assert value in ("false", "0", ""), (
            "docker-compose.copilot.yml sets COPILOT_PER_USER_TOKEN_ENABLED "
            f"to a truthy value ({match.group(1)!r}). {_REOPEN_MESSAGE}"
        )

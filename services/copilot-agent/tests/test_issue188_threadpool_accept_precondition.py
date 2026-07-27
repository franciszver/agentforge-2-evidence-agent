"""Pins the two preconditions issue #188's documented accept rests on:
``copilot_per_user_token_enabled`` defaults ``False`` in ``app/config.py``,
and no compose file in the repo sets ``COPILOT_PER_USER_TOKEN_ENABLED`` to
turn it on.

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

# Every compose file in the repo, not just the one copilot dev overlay --
# a future override file (e.g. a `docker-compose.copilot.override.yml`, or
# a compose file added under `ci/`) that flips the flag ON would otherwise
# silently defeat this guard while the docs still claim "no compose file
# sets it". Globbed fresh on every run, not hardcoded, precisely because
# the whole point is to survive new compose files being added.
_ALL_COMPOSE_FILES = sorted(
    {
        *_REPO_ROOT.glob("docker/**/docker-compose*.yml"),
        *_REPO_ROOT.glob("docker/**/docker-compose*.yaml"),
        *_REPO_ROOT.glob("ci/**/docker-compose*.yml"),
        *_REPO_ROOT.glob("ci/**/docker-compose*.yaml"),
        *_REPO_ROOT.glob("docker-compose*.yml"),
        *_REPO_ROOT.glob("docker-compose*.yaml"),
    }
)

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
# scoped to a single service or a single compose file, since the accept's
# premise is that NO compose file in the repo (nor any service within one)
# sets the flag, not just that the ``agent`` service block in one known
# file happens not to. The captured group is intentionally permissive
# (``\S+``, not ``\w+``) so it also captures a ``${...}`` env-substitution
# reference -- see the fail-closed handling below; we deliberately do NOT
# attempt to resolve substitutions, we just refuse to treat them as proven
# falsy.
_COMPOSE_ENV_SET_RE = re.compile(
    r"^\s*(?:-\s*)?COPILOT_PER_USER_TOKEN_ENABLED\s*[:=]\s*[\"']?(\S+)",
    re.MULTILINE,
)


def test_repo_layout_assumptions_hold() -> None:
    assert _CONFIG_PY.is_file(), f"expected {_CONFIG_PY} to exist"
    assert _COPILOT_COMPOSE.is_file(), f"expected {_COPILOT_COMPOSE} to exist"

    # Presence assertion paired with the absence assertion below: a glob
    # that silently matches zero files is exactly the failure mode this
    # test exists to prevent, so the discovered set must be non-empty and
    # must include the one compose file #188's accept explicitly names.
    assert _ALL_COMPOSE_FILES, (
        "the docker-compose*.y*ml glob under docker/ and ci/ matched zero "
        f"files -- something broke the discovery glob itself. {_REOPEN_MESSAGE}"
    )
    assert _COPILOT_COMPOSE in _ALL_COMPOSE_FILES, (
        f"{_COPILOT_COMPOSE} was not among the {len(_ALL_COMPOSE_FILES)} "
        f"compose files discovered by the glob. {_REOPEN_MESSAGE}"
    )


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
    """No compose file anywhere in the repo -- not just the known dev
    overlay -- may flip the flag on, whether as a direct assignment or an
    unresolved ``${...}`` env-substitution reference (treated fail-closed:
    we can't prove it's falsy, so it counts as set). Today it only appears
    in an explanatory comment (see the ``COPILOT_DEV_ACCEPT_ANY_BEARER_TOKEN``
    block in ``docker-compose.copilot.yml``), never as a live
    ``environment:`` assignment, and no compose file references it via
    ``${...}`` substitution at all. Verified by mutation: adding an
    override file, or a compose file in an unrelated directory, that sets
    the flag truthy fails this test; deleting it passes again."""
    for compose_file in _ALL_COMPOSE_FILES:
        text = compose_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = _COMPOSE_ENV_SET_RE.match(stripped)
            if match is None:
                continue
            raw_value = match.group(1).strip().rstrip("\"'")
            if raw_value.startswith("${"):
                # Fail-closed: an env-substitution reference could resolve
                # to anything at deploy time and we deliberately do not
                # attempt to resolve it here, so we refuse to treat it as
                # proven falsy.
                raise AssertionError(
                    f"{compose_file} sets COPILOT_PER_USER_TOKEN_ENABLED via "
                    f"an unresolved substitution ({match.group(1)!r}) whose "
                    f"value cannot be proven falsy. {_REOPEN_MESSAGE}"
                )
            value = raw_value.lower()
            assert value in ("false", "0", ""), (
                f"{compose_file} sets COPILOT_PER_USER_TOKEN_ENABLED to a "
                f"truthy value ({match.group(1)!r}). {_REOPEN_MESSAGE}"
            )

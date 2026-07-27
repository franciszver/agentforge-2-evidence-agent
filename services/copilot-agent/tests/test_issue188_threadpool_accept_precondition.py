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
It also only covers the in-repo premises: an operator setting
``COPILOT_PER_USER_TOKEN_ENABLED=true`` directly in a deployment
environment (a k8s manifest, a systemd unit, `docker run -e`, a CI
secret) turns the flag on with zero repo change, and no test in this
repo can pin that -- see docs/ARCHITECTURE.md's "Path to Production"
item 2 for that limit stated explicitly.

**What this test does NOT do.** It does not exercise the threadpool
exhaustion itself -- that measurement came from a hermetic in-process
harness (the real FastAPI app object driven over httpx's ASGI transport,
with anyio's real threadpool limiter and a deliberately slow validator
double standing in for a live introspection call; no docker, no live
stack) and it does not re-derive the mitigation options. It exists solely
so that if a future change flips the shipped default or a compose file's
override, this test fails LOUDLY -- naming both this file and
docs/ARCHITECTURE.md's "Path to Production" item 2 section -- instead of
the accept's premise silently stopping to hold while docs/ARCHITECTURE.md
still describes the old, now-false state.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PY = _REPO_ROOT / "services" / "copilot-agent" / "app" / "config.py"
_COPILOT_COMPOSE = _REPO_ROOT / "docker" / "development-easy" / "docker-compose.copilot.yml"

# Directories that are never source-of-truth for this repo's own compose
# files -- vendored/virtualenv/dependency trees can contain arbitrarily many
# *compose*.y*ml files (e.g. inside a packaged dependency's test fixtures)
# that have nothing to do with this repo's own deploy surface.
_EXCLUDED_DIR_NAMES = frozenset({".venv", "node_modules", "vendor", ".git"})


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_DIR_NAMES for part in path.relative_to(_REPO_ROOT).parts)


# Every compose file anywhere in the repo, not just the ones under docker/
# and ci/ -- the Compose Specification's *preferred* canonical filenames
# (`compose.yaml`, `compose.yml`, `compose.override.yaml`, auto-loaded by
# `docker compose` with no `-f`) live at arbitrary paths (e.g.
# `.github/docker/compose.yml`, `ci/inferno/compose.yml`,
# `ci/compose-shared-mailpit/compose.yml`), and a future override file
# anywhere that flips the flag ON would otherwise silently defeat this
# guard while the docs still claim "no compose file sets it". Globbed
# fresh on every run, not hardcoded, precisely because the whole point is
# to survive new compose files being added in new locations.
_ALL_COMPOSE_FILES = sorted(
    p for p in _REPO_ROOT.rglob("*compose*.y*ml") if not _is_excluded(p)
)

_REOPEN_MESSAGE = (
    "issue #188's documented accept assumed this flag ships OFF -- if it is "
    "being turned on, the threadpool starvation mitigation is now a "
    "blocking pre-condition. See "
    "services/copilot-agent/tests/test_issue188_threadpool_accept_precondition.py "
    "and docs/ARCHITECTURE.md's \"Path to Production\" item 2."
)

# Matches the Settings field declaration, e.g.
# "    copilot_per_user_token_enabled: bool = False" -- tolerant of trailing
# whitespace or an inline "# ..." comment, so a harmless reformatting pass
# doesn't trip this test.
_CONFIG_DEFAULT_RE = re.compile(
    r"^\s*copilot_per_user_token_enabled:\s*bool\s*=\s*(\w+)\s*(#.*)?$",
    re.MULTILINE,
)

# Matches the env var's key ANYWHERE on a (comment-stripped) line -- not
# anchored to line start -- because the mainstream quoted/flow-style forms
# put other characters before the key on the same line: a quoted list item
# (``- "COPILOT_PER_USER_TOKEN_ENABLED=true"``), a single-quoted list item
# (``- 'COPILOT_PER_USER_TOKEN_ENABLED=true'``), YAML flow-style list syntax
# (``environment: ["COPILOT_PER_USER_TOKEN_ENABLED=true"]``), and a quoted
# mapping key (``"COPILOT_PER_USER_TOKEN_ENABLED": "true"``) all have a
# leading quote/bracket/dash that a start-anchored match misses entirely --
# previously letting all four forms bypass this guard while turning the flag
# ON. An anchor-free start-of-line dash and an optional quote char between
# the key and the ``[:=]`` operator are both tolerated for the same reason.
# The value capture stops before a comma or closing bracket/brace so flow
# style's trailing ``"]`` doesn't get folded into the captured value; a
# trailing quote character is stripped from the value separately, below,
# the same way the plain-mapping-value form always has been.
#
# Mapping form (``COPILOT_PER_USER_TOKEN_ENABLED: true``), list form
# (``- COPILOT_PER_USER_TOKEN_ENABLED=true``), a bare list entry with no
# operator at all (``- COPILOT_PER_USER_TOKEN_ENABLED``), or an empty
# mapping (``COPILOT_PER_USER_TOKEN_ENABLED:`` with nothing after the
# colon) are also matched -- deliberately NOT scoped to a single service or
# a single compose file, since the accept's premise is that NO compose file
# in the repo (nor any service within one) sets the flag. The ``value``
# group is ``None`` when there is no ``[:=]`` at all (bare list entry) and
# ``""`` when there is an operator but nothing follows (empty mapping) --
# both of those forms mean "inherits from the host environment" in Compose
# semantics, which this guard cannot prove is falsy, so both are handled as
# failures below rather than silently passing.
_COMPOSE_ENV_KEY_RE = re.compile(
    r"COPILOT_PER_USER_TOKEN_ENABLED(?!\w)\s*"
    r"(?:[\"']?\s*(?:[:=]\s*(?P<value>[^,\]\}]*))?)?"
)

# pydantic's bool parsing (see pydantic_core's `str_as_bool`) treats these
# case-insensitively as False; anything else -- including values it would
# reject outright -- cannot be proven falsy, so this guard fails closed on
# everything not in this set.
_PYDANTIC_FALSY_VALUES = frozenset({"false", "0", "no", "off", "n"})

_ENV_FILE_RE = re.compile(r"^env_file\s*:\s*(?P<inline>.*?)\s*(#.*)?$")
_ENV_FILE_ITEM_RE = re.compile(r"^-\s*(?P<path>[^#]+?)\s*(#.*)?$")


def _strip_trailing_comment(line: str) -> str:
    """Strip a trailing ``# ...`` YAML comment, quote-aware -- a ``#``
    inside a single- or double-quoted string is not a comment marker (mirrors
    the quote-awareness YAML itself uses) and is left untouched. Without
    this, a real, correctly-falsy assignment followed by an inline comment
    (e.g. ``COPILOT_PER_USER_TOKEN_ENABLED: "false"  # off by default``) gets
    its comment text folded into the captured value and fails a value that
    is actually fine."""
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _find_env_file_references(text: str) -> list[str]:
    """Return every ``env_file`` path referenced anywhere in the compose
    file -- deliberately NOT scoped to services whose name looks
    agent/copilot-relevant, since a service under any other name (e.g.
    ``app``) can equally reference a file that sets
    ``COPILOT_PER_USER_TOKEN_ENABLED``, and the env-var scan above
    (``_COMPOSE_ENV_KEY_RE``) is already file-wide rather than
    service-scoped for the same reason. Deliberately simple,
    indentation-based block tracking -- this repo's compose files use a flat
    2-space-per-level style. Does not resolve YAML anchors/merges."""
    refs: list[str] = []
    current_env_file_indent: int | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        env_file_match = _ENV_FILE_RE.match(stripped)
        if env_file_match:
            current_env_file_indent = indent
            inline = env_file_match.group("inline").strip().strip("\"'")
            if inline:
                refs.append(inline)
            continue
        if current_env_file_indent is not None and indent > current_env_file_indent:
            item = _ENV_FILE_ITEM_RE.match(stripped)
            if item:
                refs.append(item.group("path").strip().strip("\"'"))
                continue
        current_env_file_indent = None
    return refs


def test_repo_layout_assumptions_hold() -> None:
    assert _CONFIG_PY.is_file(), f"expected {_CONFIG_PY} to exist"
    assert _COPILOT_COMPOSE.is_file(), f"expected {_COPILOT_COMPOSE} to exist"

    # Presence assertion paired with the absence assertion below: a glob
    # that silently matches zero files is exactly the failure mode this
    # test exists to prevent, so the discovered set must be non-empty and
    # must include the one compose file #188's accept explicitly names.
    assert _ALL_COMPOSE_FILES, (
        "the repo-wide *compose*.y*ml scan matched zero files -- something "
        f"broke the discovery glob itself. {_REOPEN_MESSAGE}"
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
    overlay -- may flip the flag on, whether as a direct assignment, an
    unresolved ``${...}`` env-substitution reference, or a bare/empty form
    that inherits from the host environment (all treated fail-closed: we
    can't prove any of them falsy, so they all count as set). Today it only
    appears in an explanatory comment (see the
    ``COPILOT_DEV_ACCEPT_ANY_BEARER_TOKEN`` block in
    ``docker-compose.copilot.yml``), never as a live ``environment:``
    assignment. Verified by mutation: adding an override file, or a compose
    file in an unrelated directory, that sets the flag truthy (or leaves it
    to inherit from the host env) fails this test; deleting it passes
    again."""
    for compose_file in _ALL_COMPOSE_FILES:
        text = compose_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            raw_stripped = line.strip()
            if not raw_stripped or raw_stripped.startswith("#"):
                continue
            stripped = _strip_trailing_comment(raw_stripped).rstrip()
            if not stripped:
                continue
            match = _COMPOSE_ENV_KEY_RE.search(stripped)
            if match is None:
                continue
            value = match.group("value")
            if value is None:
                # Bare list entry, e.g. "- COPILOT_PER_USER_TOKEN_ENABLED",
                # with no "=" at all -- Compose resolves this from the host
                # environment at deploy time. Unprovable, so fail-closed.
                raise AssertionError(
                    f"{compose_file} lists COPILOT_PER_USER_TOKEN_ENABLED "
                    "with no explicit value -- this form inherits from the "
                    "host environment at deploy time and cannot be proven "
                    f"OFF. {_REOPEN_MESSAGE}"
                )
            raw_value = value.strip()
            if raw_value == "":
                # Empty mapping, e.g. "COPILOT_PER_USER_TOKEN_ENABLED:" with
                # nothing after the colon -- same host-inherits semantics.
                raise AssertionError(
                    f"{compose_file} maps COPILOT_PER_USER_TOKEN_ENABLED to "
                    "an empty/null value -- this form inherits from the "
                    "host environment at deploy time and cannot be proven "
                    f"OFF. {_REOPEN_MESSAGE}"
                )
            raw_value = raw_value.strip("\"'")
            if raw_value.startswith("${"):
                # Fail-closed: an env-substitution reference could resolve
                # to anything at deploy time and we deliberately do not
                # attempt to resolve it here, so we refuse to treat it as
                # proven falsy.
                raise AssertionError(
                    f"{compose_file} sets COPILOT_PER_USER_TOKEN_ENABLED via "
                    f"an unresolved substitution ({value!r}) whose value "
                    f"cannot be proven falsy. {_REOPEN_MESSAGE}"
                )
            if raw_value.lower() not in _PYDANTIC_FALSY_VALUES:
                raise AssertionError(
                    f"{compose_file} sets COPILOT_PER_USER_TOKEN_ENABLED to "
                    f"{value!r}, which is not one of pydantic's recognized "
                    f"falsy strings {sorted(_PYDANTIC_FALSY_VALUES)!r} and so "
                    f"cannot be proven OFF. {_REOPEN_MESSAGE}"
                )


def test_no_compose_file_env_file_enables_copilot_per_user_token() -> None:
    """Any service's ``env_file:`` reference -- not just an agent/copilot-
    named one -- is another route to turning the flag on without a live
    ``environment:`` assignment ever appearing in the compose file itself;
    a service named e.g. ``app`` referencing a file that sets the flag
    truthy is exactly as much a bypass as an agent/copilot-named one, so
    every ``env_file`` reference in every compose file is walked, strictly
    fail-closed, mirroring the file-wide (not service-scoped) env-var scan
    above. This deliberately does not trust a referenced file's *absence* of
    the setting as proof of OFF -- if the file cannot be read, that is
    treated the same as "cannot be proven falsy" and fails, naming the
    unreadable path, rather than silently passing. Today no compose file in
    the repo declares ``env_file`` at all, so this test currently has
    nothing to walk."""
    for compose_file in _ALL_COMPOSE_FILES:
        text = compose_file.read_text(encoding="utf-8")
        for raw_ref in _find_env_file_references(text):
            ref_path = (compose_file.parent / raw_ref).resolve()
            if not ref_path.is_file():
                raise AssertionError(
                    f"{compose_file} declares env_file {raw_ref!r}, but "
                    f"{ref_path} could not be read -- whether it sets "
                    "COPILOT_PER_USER_TOKEN_ENABLED cannot be proven, so "
                    f"this cannot be treated as OFF. {_REOPEN_MESSAGE}"
                )
            ref_text = ref_path.read_text(encoding="utf-8")
            for line in ref_text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, _, raw_value = stripped.partition("=")
                if key.strip() != "COPILOT_PER_USER_TOKEN_ENABLED":
                    continue
                value = raw_value.strip().strip("\"'")
                if value.lower() not in _PYDANTIC_FALSY_VALUES:
                    raise AssertionError(
                        f"{ref_path} (referenced via {compose_file}'s "
                        "env_file) sets COPILOT_PER_USER_TOKEN_ENABLED to "
                        f"{raw_value.strip()!r}, which is not a known-falsy "
                        f"value. {_REOPEN_MESSAGE}"
                    )

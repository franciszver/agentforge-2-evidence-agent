"""Content-stamp drift guard for live recording (#140, ``docs/TEST_PLAN.md``
Sec 9).

``development-easy-agent-1`` has zero bind mounts -- its ``app/`` package is
baked into the image at build time, not a live-editable checkout. During
#70/#130, recording a case against that container's live model had to
``docker cp`` current sources in by hand first; forgetting that step meant
recording against silently stale code, corrupting the committed golden eval
set with no signal anything was wrong.

This module has no opinion about *how* the code got into the container --
only whether the code ``evals/runner/record.py`` actually resolved and
imported (see #119's ``_agent_root_candidates``) matches what the recording
protocol expects. The mechanics, end to end (documented in ``docs/
TEST_PLAN.md`` Sec 9):

1. Host-side, before docker-exec'ing into the container, the operator (or a
   wrapper script) computes ``compute_app_stamp`` over
   ``services/copilot-agent/app`` -- the working tree the recording is
   SUPPOSED to represent.
2. That stamp is passed into the container as the ``EXPECTED_APP_STAMP``
   env var.
3. ``record.py``, at startup, computes its OWN stamp over whatever ``app/``
   it actually resolved on ``sys.path`` (the live, in-process code -- not a
   claim about it) and calls :func:`check_code_stamp` before making any live
   model call. A mismatch means the container's baked code has drifted from
   the host tree the operator thinks they're recording against, and refuses
   loudly rather than recording silently-wrong output.

Recording directly on the host (no container in the loop) never sets
``EXPECTED_APP_STAMP`` -- there the live code trivially IS the working tree,
so the check is a deliberate no-op rather than a mandatory-but-vacuous
comparison against itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class CodeStampMismatchError(RuntimeError):
    """Raised by :func:`check_code_stamp` when the in-process app code
    stamp doesn't match the ``EXPECTED_APP_STAMP`` the recording protocol
    supplied -- the container's baked ``app/`` has drifted from the host
    working tree this recording is supposed to represent."""


def compute_app_stamp(app_root: Path) -> str:
    """Deterministic sha256 hex digest over every ``*.py`` file under
    ``app_root`` (an ``app/`` package directory): each file's POSIX-style
    relative path plus its raw bytes, in sorted-path order so the result
    doesn't depend on filesystem iteration order. ``__pycache__`` is
    excluded -- compiled bytecode is a build artifact, not source, and its
    mtime-sensitive contents would make the stamp spuriously flap between
    two checkouts of identical source.
    """
    digest = hashlib.sha256()
    files = sorted(
        (path for path in app_root.rglob("*.py") if "__pycache__" not in path.parts),
        key=lambda path: path.relative_to(app_root).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(app_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def check_code_stamp(local_stamp: str, expected_stamp: str | None) -> None:
    """Fail loudly if ``local_stamp`` (this process's actual, in-use app
    code) doesn't match ``expected_stamp`` (what the recording protocol
    says it should be). A ``None`` ``expected_stamp`` means the check
    wasn't requested at all -- a no-op, not a pass-by-default on a real
    mismatch.
    """
    if expected_stamp is None:
        return
    if local_stamp == expected_stamp:
        return
    raise CodeStampMismatchError(
        "recording refused: in-process app code stamp "
        f"{local_stamp!r} does not match EXPECTED_APP_STAMP {expected_stamp!r} -- "
        "the container's baked app/ code has drifted from the host working tree "
        "this recording protocol expects a live recording to represent (#140). "
        "Remediation: rebuild the agent image so it bakes current sources "
        "(`docker compose build agent && docker compose up -d agent`), or for "
        "quick local iteration, docker cp the current services/copilot-agent/app "
        "tree into the running container before recording again -- see "
        "docs/TEST_PLAN.md Sec 9 for the full protocol."
    )

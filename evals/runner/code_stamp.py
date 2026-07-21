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


# Rewritten in place at RUNTIME by ``app.dashboard_eval_history.append_eval_run``
# (invoked from ``evals/runner/record_run.py`` after every eval run) -- see
# that module's docstring. In a long-lived container this file's on-disk
# content drifts from its committed state as soon as any eval run completes,
# for reasons that have nothing to do with source code drift (gate review on
# #143). Hashing it made the stamp guard cry wolf on the very next recording
# after any prior eval run in the same container -- exactly the kind of
# false-positive abort that trains operators to disable the guard,
# reintroducing #140. Paths are POSIX-style relative to ``app_root``, same
# convention ``compute_app_stamp`` itself sorts/hashes by.
#
# Deliberately NOT included here: app/data/drug_interactions.db and
# app/data/drug_interactions.sha256, even though app/data/drug_interactions.py
# has a ``write_text``/db-build code path -- that path only runs when a
# maintainer manually invokes ``python -m app.data.drug_interactions`` to
# rebuild the committed artifact from source (a source change, checked in
# like any other), never as a side effect of the app serving a request or an
# eval run recording. Those files remain real drift surface.
_RUNTIME_MUTABLE = frozenset(
    {
        "data/eval_history.json",
    }
)


def _is_text(data: bytes) -> bool:
    """Standard NUL-byte heuristic: text files don't contain NUL bytes,
    binary files (sqlite databases, images, ...) reliably do somewhere in
    their first bytes for any non-trivial file. Good enough to decide
    whether normalizing line endings is safe without pulling in a
    full MIME/encoding sniffer."""
    return b"\0" not in data


def _normalized_for_hash(data: bytes) -> bytes:
    """Line-ending-normalize ``data`` for hashing if it looks like a text
    file (CRLF -> LF, then lone CR -> LF), otherwise return it byte-exact.

    Why: this repo's Windows working tree checks out CRLF (autocrlf), but
    the Linux container bakes the LF git blob -- so a byte-exact hash of a
    text file computed on the Windows host (``docs/TEST_PLAN.md`` Sec 9)
    never matches the container's in-process stamp over the identical
    logical source, and EVERY recording aborted with
    ``CodeStampMismatchError`` even when perfectly in sync (gate review on
    #143). Binary files are hashed byte-exact, unnormalized -- a ``\\r\\n``
    byte pair inside a binary asset (e.g. ``drug_interactions.db``) is real
    data, not a line ending, and collapsing it would mask genuine drift in
    exactly the behavioral assets the widened ``app/``-wide coverage exists
    to protect.
    """
    if not _is_text(data):
        return data
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def compute_app_stamp(app_root: Path) -> str:
    """Deterministic sha256 hex digest over every file under ``app_root``
    (an ``app/`` package directory) -- not just ``*.py``: behavioral assets
    the running code loads at import/runtime (e.g. ``app/data/
    drug_interactions.db``, ``reranker_scores*.json``,
    ``retrieval_embeddings.json``) are just as capable of silently drifting
    between the host tree and a stale baked container as source is, and a
    guard that only watches ``*.py`` would report "code matches" while one
    of those assets quietly rotted. Each file's POSIX-style relative path
    plus its raw bytes, in sorted-path order so the result doesn't depend on
    filesystem iteration order.

    Excluded: ``__pycache__`` directories and any ``*.pyc`` file (even one
    sitting outside ``__pycache__``) -- compiled bytecode is a build
    artifact, not source or data, and its mtime-sensitive contents would
    make the stamp spuriously flap between two checkouts of identical
    source. Also excluded: the exact paths in :data:`_RUNTIME_MUTABLE` --
    files the app itself rewrites at runtime, not source or committed data
    (see that constant's docstring for the full rationale and how each entry
    was identified).

    **Line-ending normalization (gate review on #143).** Each TEXT file's
    bytes are normalized (CRLF -> LF, lone CR -> LF; see
    :func:`_normalized_for_hash`) before hashing, so the CRLF Windows host
    checkout and the LF-blob Linux container produce the SAME stamp for
    identical logical source. BINARY files (detected via the NUL-byte
    heuristic) are hashed byte-exact, unnormalized -- real drift in an
    asset like ``drug_interactions.db`` must still be detected, never
    collapsed away as if it were a line-ending difference.
    """
    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in app_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and path.relative_to(app_root).as_posix() not in _RUNTIME_MUTABLE
        ),
        key=lambda path: path.relative_to(app_root).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(app_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_normalized_for_hash(path.read_bytes()))
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

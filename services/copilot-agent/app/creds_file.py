"""Owner-only writer for OAuth client-credential files (#176).

The dev token bridge and the production client registration both persist a
``{"client_id", "client_secret"}`` JSON file. That file holds a real OAuth
``client_secret``, so it must never be group/world-readable. ``write_text``
(used previously) creates the file with default umask permissions
(typically ``0o644`` on POSIX) -- this helper is the single secure-write path
both call sites share.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_creds_securely(path: str, data: dict[str, str]) -> None:
    """Write ``data`` as JSON to ``path`` with owner-only (0o600) permissions.

    ``os.open`` with the ``0o600`` mode sets permissions at *creation* time, but
    that mode is only applied when the file is newly created -- a pre-existing
    file keeps its old (possibly looser) mode -- so an explicit ``os.chmod``
    follows to enforce ``0o600`` regardless of pre-existence. If this call is
    responsible for creating the parent directory, it is created ``0o700``.

    On Windows POSIX modes are a no-op, but every call is harmless there.
    """
    parent = Path(path).parent
    if not parent.exists():
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    # O_NOFOLLOW closes the symlink-follow window on the creds path (a planted
    # symlink could otherwise redirect the write). It exists on Linux/CI;
    # getattr keeps Windows (which lacks it) a no-op.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data))
    os.chmod(path, 0o600)

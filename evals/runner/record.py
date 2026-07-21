"""Record mode (P4.7, ``docs/TEST_PLAN.md`` Sec 9): drives a case through the
REAL pipeline against the LIVE model and commits its ordered Ollama/
llama-server responses as ``evals/recordings/<id>.json``.

Local, opt-in, needs a reachable live model. Two engines are supported,
mirroring ``app.config.Settings.copilot_llm_engine``:

  * ``ollama`` (default) -- Ollama has no published host port on the dev
    stack by design (internal-only network, no egress) -- point
    ``OLLAMA_BASE_URL`` at a bridge (e.g. a disposable ``socat`` container
    publishing the internal ``ollama`` service to the host), the same
    convention ``evals/injection/test_injection.py`` and
    ``services/copilot-agent/tests/test_ollama_client.py`` already use.
  * ``llama_server`` (#58) -- point ``LLAMA_SERVER_BASE_URL`` at a reachable
    ``llama-server`` instance (e.g. a host-published one at
    ``http://127.0.0.1:8080``); uses ``app.llama_server_client.LlamaServerClient``,
    the same production client ``app.chat.get_text_llm_client`` wires up for
    ``copilot_llm_engine=llama_server``.

Usage (from repo root):

    OLLAMA_BASE_URL=http://localhost:11435 python evals/runner/record.py uc2-meds
    OLLAMA_BASE_URL=http://localhost:11435 python evals/runner/record.py --all
    RECORD_ENGINE=llama_server LLAMA_SERVER_BASE_URL=http://127.0.0.1:8080 python evals/runner/record.py --all

Tears down nothing itself -- the bridge (if any) is the caller's to start and
stop.

**Running inside ``development-easy-agent-1`` (or any container whose image
copy of this repo is flattened, i.e. ``/app`` is the copilot-agent root
directly rather than ``<repo>/services/copilot-agent``):** this script's
own ``sys.path`` setup detects both layouts (#119) so it always imports the
LIVE ``app`` package baked into the image, never a stale pre-built copy
under ``site-packages``. That container also has **no bind mounts**
(``docker inspect development-easy-agent-1`` shows ``"Mounts": []``) -- it
is a snapshot of the image, not a live-editable checkout, so any file this
script writes (recordings included) must be copied out explicitly, e.g.:

    docker cp development-easy-agent-1:/app/evals/recordings/<id>.json evals/recordings/

before it will show up on the host to commit.

**Code-stamp drift guard (#140).** Because that container has no bind
mounts, a live recording made without noticing the baked ``app/`` is stale
(exactly what happened during #70/#130) silently corrupts the committed
golden eval set -- no error, just wrong recorded output. Before making any
live model call, ``main()`` calls :func:`verify_code_stamp`, which computes
a content stamp over the actually-IMPORTED ``app`` package (``Path(app.
__file__).parent`` -- provably the code this process has in memory, not a
separately re-resolved guess at it) via ``runner.code_stamp.
compute_app_stamp`` and, if the recording protocol supplied
``EXPECTED_APP_STAMP`` (host-computed over ``services/copilot-agent/app``
before docker-exec'ing in -- see ``docs/TEST_PLAN.md`` Sec 9), refuses
loudly on any mismatch instead of recording against silently-stale code.
The resulting stamp is also written into every new recording's
``code_stamp`` metadata field, so a future audit can tell what code produced
it (old recordings without the field are untouched and still replay exactly
as before).

**Fail CLOSED on a missing stamp under the container layout (gate review,
PR #143).** The first cut of this guard treated an unset ``EXPECTED_APP_STAMP``
as "check not requested" everywhere -- a no-op. That is correct on the
genuine host-monorepo layout (no container, nothing to drift from) but
fails OPEN under the flattened in-container layout: an operator who simply
forgot to pass the env var got silent, unchecked recording against whatever
stale code happened to be baked in -- precisely the bug #140 exists to
prevent. ``verify_code_stamp`` now detects the layout
(:func:`_in_flattened_container_layout`, reusing #119's own resolution) and
raises :class:`MissingExpectedStampError` if the stamp is unset/empty there,
instead of skipping the check.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _EVALS_ROOT.parent
_MONOREPO_AGENT_ROOT = _REPO_ROOT / "services" / "copilot-agent"


def _agent_root_candidates(repo_root: Path, monorepo_agent_root: Path) -> list[Path]:
    """Directories that might hold the live ``app`` package, most-preferred
    first -- covers both layouts this script runs under (#119):

    * full monorepo checkout: ``<repo-root>/services/copilot-agent/app/``
    * the ``development-easy-agent-1`` dev container's flattened layout,
      where ``/app`` (i.e. ``repo_root`` here) IS the copilot-agent root
      directly -- ``repo_root / "services" / "copilot-agent"`` does not
      exist there, so that candidate must not be the only one inserted.

    Whichever candidate actually contains an ``app`` package on disk sorts
    first, so it lands ahead of the other candidate -- and ahead of any
    stale ``app`` copy pre-installed in site-packages -- on ``sys.path``,
    regardless of which layout is active. Pure function of its arguments
    (no module-global reads) so it's directly unit-testable against
    synthetic directory trees -- see
    ``evals/runner/tests/test_record_path_resolution.py``.
    """
    candidates = [monorepo_agent_root, repo_root]
    return sorted(candidates, key=lambda root: not (root / "app" / "__init__.py").is_file())


for _root in reversed(_agent_root_candidates(_REPO_ROOT, _MONOREPO_AGENT_ROOT) + [_EVALS_ROOT]):
    _root_str = str(_root)
    if _root_str not in sys.path:
        sys.path.insert(0, _root_str)

import app  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llama_server_client import LlamaServerClient  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402

from runner.code_stamp import check_code_stamp, compute_app_stamp  # noqa: E402
from runner.loader import discover_case_files, load_case  # noqa: E402
from runner.ollama_replay import OllamaLike, RecordingOllamaClient, recording_path, save_recording  # noqa: E402
from runner.pipeline import run_case  # noqa: E402

_CASES_DIR = _EVALS_ROOT / "cases"
_REGRESSIONS_DIR = _EVALS_ROOT / "regressions"
_RECORDINGS_DIR = _EVALS_ROOT / "recordings"

# Derived from the ACTUALLY-IMPORTED ``app`` module (see ``import app``
# above), not by re-resolving #119's candidates a second time -- this way
# the stamp provably describes the code this process has in memory and is
# about to call, rather than a plausible-but-separately-recomputed guess at
# which candidate sys.path setup picked. #119's own resolution (the loop
# above) is unchanged; only the root the STAMP is computed over changes.
assert app.__file__ is not None, "app package has no __file__ (unexpected namespace package)"
_RESOLVED_APP_ROOT = Path(app.__file__).resolve().parent


def _find_case_file(case_id: str) -> Path:
    for path in discover_case_files(_CASES_DIR, _REGRESSIONS_DIR):
        if load_case(path).id == case_id:
            return path
    raise SystemExit(f"no case with id {case_id!r} under {_CASES_DIR} or {_REGRESSIONS_DIR}")


class MissingExpectedStampError(RuntimeError):
    """Raised by :func:`verify_code_stamp` when running under the flattened
    in-container layout (#119) with ``EXPECTED_APP_STAMP`` unset or empty --
    the gate review that followed #140's initial fix found the original
    behavior (treat an unset env var as "check not requested" -- a no-op)
    fails OPEN exactly in the case that matters most: an operator who simply
    forgot ``docker exec -e EXPECTED_APP_STAMP=...`` got silent, unchecked
    recording against whatever stale code the container happened to have
    baked in, with zero signal anything was wrong. In that layout the stamp
    is now REQUIRED; only the genuine host-monorepo layout (no container,
    no baked-code drift possible) still treats an unset var as a no-op."""


def _in_flattened_container_layout() -> bool:
    """True when this process resolved the live ``app`` package via the
    flattened dev-container layout (#119's ``_agent_root_candidates``
    winning candidate is the repo root itself, not ``services/
    copilot-agent``) -- i.e. exactly the ``development-easy-agent-1``-style
    layout with no bind mounts, where the baked ``app/`` can silently drift
    from the host tree an operator thinks they're recording against.

    The host-monorepo layout (``_MONOREPO_AGENT_ROOT/app/__init__.py``
    exists) is never flagged, whether or not it happens to also win the
    candidate race -- there the live code trivially IS the working tree, so
    ``EXPECTED_APP_STAMP`` staying optional is correct, not a gap.
    """
    return _agent_root_candidates(_REPO_ROOT, _MONOREPO_AGENT_ROOT)[0] == _REPO_ROOT


def verify_code_stamp() -> str:
    """Compute this process's local app-code stamp (#140) and, if the
    recording protocol supplied ``EXPECTED_APP_STAMP`` (host-computed over
    ``services/copilot-agent/app`` before docker-exec'ing into a container
    -- see ``docs/TEST_PLAN.md`` Sec 9), refuse loudly on mismatch before
    any live model call is made. Returns the local stamp so it can be
    stamped into the recording's own metadata (an audit trail even when no
    ``EXPECTED_APP_STAMP`` was supplied, e.g. recording directly on host).

    Fail CLOSED, not open (gate review on #140/PR #143): under the
    flattened in-container layout (:func:`_in_flattened_container_layout`),
    an unset or empty ``EXPECTED_APP_STAMP`` aborts record mode instead of
    silently skipping the check -- see :class:`MissingExpectedStampError`.
    """
    local_stamp = compute_app_stamp(_RESOLVED_APP_ROOT)
    expected_stamp = os.environ.get("EXPECTED_APP_STAMP") or None
    if expected_stamp is None and _in_flattened_container_layout():
        raise MissingExpectedStampError(
            "recording refused: running under the flattened in-container app layout "
            "(#119) with EXPECTED_APP_STAMP unset (or empty) -- this container has no "
            "bind mounts, so its baked app/ code can silently drift from the host tree "
            "you think you're recording against (#140), and the drift check would "
            "otherwise be skipped entirely (fail-open). Set EXPECTED_APP_STAMP to the "
            "host-computed stamp before recording, e.g.:\n"
            "  python -c \"from pathlib import Path; import sys; "
            "sys.path.insert(0, 'evals/runner'); from code_stamp import compute_app_stamp; "
            "print(compute_app_stamp(Path('services/copilot-agent/app')))\"\n"
            "then pass it through: "
            "docker exec -e EXPECTED_APP_STAMP=<stamp from above> -it <container> "
            "python evals/runner/record.py <case-id> -- see docs/TEST_PLAN.md Sec 9 for "
            "the full protocol."
        )
    check_code_stamp(local_stamp, expected_stamp)
    return local_stamp


def _build_live_client(engine: str, ollama_base_url: str) -> OllamaLike:
    """Build the live model client for ``engine`` -- ``"ollama"`` (default)
    or ``"llama_server"`` (#58, same production client as
    ``copilot_llm_engine=llama_server``)."""
    if engine == "llama_server":
        settings = Settings(llama_server_api_timeout_seconds=180.0)
        return LlamaServerClient.from_settings(settings)
    settings = Settings(ollama_base_url=ollama_base_url, ollama_api_timeout_seconds=180.0)
    return OllamaClient.from_settings(settings)


def record_case(case_id: str, ollama_base_url: str, engine: str, code_stamp: str) -> None:
    case = load_case(_find_case_file(case_id))

    recorder = RecordingOllamaClient(_build_live_client(engine, ollama_base_url))  # type: ignore[arg-type]

    result = run_case(case, recorder)

    out_path = recording_path(_RECORDINGS_DIR, case.id)
    save_recording(out_path, recorder.calls, code_stamp=code_stamp)
    tools_dispatched = [call.tool.value for call in result.planner_result.trace]
    print(f"[record] {case.id}: {len(recorder.calls)} call(s) -> {out_path}")
    print(f"[record] {case.id}: tools dispatched = {tools_dispatched}")
    print(f"[record] {case.id}: answer = {result.planner_result.answer!r}")
    if result.verdict_result is not None:
        print(f"[record] {case.id}: verdict = {result.verdict_result.verdict.value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_ids", nargs="*", help="case id(s) to record")
    parser.add_argument(
        "--all", action="store_true", help="record every case under evals/cases/ and evals/regressions/"
    )
    args = parser.parse_args()

    # #140: refuse before any live model call (and before wasting the
    # operator's time/tokens on a doomed run) if the code this process
    # actually resolved and imported has drifted from what the recording
    # protocol expects.
    code_stamp = verify_code_stamp()

    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    engine = os.environ.get("RECORD_ENGINE", "ollama")

    case_ids = (
        [load_case(p).id for p in discover_case_files(_CASES_DIR, _REGRESSIONS_DIR)]
        if args.all
        else args.case_ids
    )
    if not case_ids:
        parser.error("pass one or more case ids, or --all")

    for case_id in case_ids:
        record_case(case_id, ollama_base_url, engine, code_stamp)


if __name__ == "__main__":
    main()

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

from app.config import Settings  # noqa: E402
from app.llama_server_client import LlamaServerClient  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402

from runner.loader import discover_case_files, load_case  # noqa: E402
from runner.ollama_replay import OllamaLike, RecordingOllamaClient, recording_path, save_recording  # noqa: E402
from runner.pipeline import run_case  # noqa: E402

_CASES_DIR = _EVALS_ROOT / "cases"
_REGRESSIONS_DIR = _EVALS_ROOT / "regressions"
_RECORDINGS_DIR = _EVALS_ROOT / "recordings"


def _find_case_file(case_id: str) -> Path:
    for path in discover_case_files(_CASES_DIR, _REGRESSIONS_DIR):
        if load_case(path).id == case_id:
            return path
    raise SystemExit(f"no case with id {case_id!r} under {_CASES_DIR} or {_REGRESSIONS_DIR}")


def _build_live_client(engine: str, ollama_base_url: str) -> OllamaLike:
    """Build the live model client for ``engine`` -- ``"ollama"`` (default)
    or ``"llama_server"`` (#58, same production client as
    ``copilot_llm_engine=llama_server``)."""
    if engine == "llama_server":
        settings = Settings(llama_server_api_timeout_seconds=180.0)
        return LlamaServerClient.from_settings(settings)
    settings = Settings(ollama_base_url=ollama_base_url, ollama_api_timeout_seconds=180.0)
    return OllamaClient.from_settings(settings)


def record_case(case_id: str, ollama_base_url: str, engine: str) -> None:
    case = load_case(_find_case_file(case_id))

    recorder = RecordingOllamaClient(_build_live_client(engine, ollama_base_url))  # type: ignore[arg-type]

    result = run_case(case, recorder)

    out_path = recording_path(_RECORDINGS_DIR, case.id)
    save_recording(out_path, recorder.calls)
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
        record_case(case_id, ollama_base_url, engine)


if __name__ == "__main__":
    main()

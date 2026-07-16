"""Record mode (P4.7, ``docs/TEST_PLAN.md`` Sec 9): drives a case through the
REAL pipeline against the LIVE model and commits its ordered Ollama
responses as ``evals/recordings/<id>.json``.

Local, opt-in, needs a reachable Ollama. Ollama has no published host port on
the dev stack by design (internal-only network, no egress) -- point
``OLLAMA_BASE_URL`` at a bridge (e.g. a disposable ``socat`` container
publishing the internal ``ollama`` service to the host), the same convention
``evals/injection/test_injection.py`` and
``services/copilot-agent/tests/test_ollama_client.py`` already use.

Usage (from repo root):

    OLLAMA_BASE_URL=http://localhost:11435 python evals/runner/record.py uc2-meds
    OLLAMA_BASE_URL=http://localhost:11435 python evals/runner/record.py --all

Tears down nothing itself -- the bridge (if any) is the caller's to start and
stop.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_EVALS_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _EVALS_ROOT.parent / "services" / "copilot-agent"
for _root in (str(_AGENT_ROOT), str(_EVALS_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from app.config import Settings  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402

from runner.loader import discover_case_files, load_case  # noqa: E402
from runner.ollama_replay import RecordingOllamaClient, recording_path, save_recording  # noqa: E402
from runner.pipeline import run_case  # noqa: E402

_CASES_DIR = _EVALS_ROOT / "cases"
_RECORDINGS_DIR = _EVALS_ROOT / "recordings"


def _find_case_file(case_id: str) -> Path:
    for path in discover_case_files(_CASES_DIR):
        if load_case(path).id == case_id:
            return path
    raise SystemExit(f"no case with id {case_id!r} under {_CASES_DIR}")


def record_case(case_id: str, ollama_base_url: str) -> None:
    case = load_case(_find_case_file(case_id))

    settings = Settings(ollama_base_url=ollama_base_url, ollama_api_timeout_seconds=180.0)
    recorder = RecordingOllamaClient(OllamaClient.from_settings(settings))

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
    parser.add_argument("--all", action="store_true", help="record every case under evals/cases/")
    args = parser.parse_args()

    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    case_ids = [load_case(p).id for p in discover_case_files(_CASES_DIR)] if args.all else args.case_ids
    if not case_ids:
        parser.error("pass one or more case ids, or --all")

    for case_id in case_ids:
        record_case(case_id, ollama_base_url)


if __name__ == "__main__":
    main()

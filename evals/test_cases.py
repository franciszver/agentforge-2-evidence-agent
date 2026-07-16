"""The offline eval suite entry point (P4.7).

Every case under ``evals/cases/`` runs in REPLAY mode by default: recorded
model outputs (``evals/recordings/<id>.json``) are fed through the REAL
planner -> extraction -> verification pipeline, deterministically. No
Ollama, no network, no live model -- this is the path CI runs (wiring CI to
invoke it is P5.2; see ``docs/TEST_PLAN.md`` Sec 9).

Two checks per case, run independently so their failures are distinguishable:

  * ``test_case_schema_is_valid`` -- the case file parses and schema-validates.
    Runs even when the case has no recording, so a broken case always fails.
  * ``test_case_replay`` -- the case's recording replays through the real
    pipeline and every assertion passes. A case with no committed recording
    fails here (not skipped -- see ``runner.ollama_replay``'s module
    docstring).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runner.assertions import evaluate_assertions
from runner.loader import discover_case_files, load_case
from runner.ollama_replay import ReplayOllamaClient, load_recording, recording_path
from runner.pipeline import run_case

_CASES_DIR = Path(__file__).parent / "cases"
_RECORDINGS_DIR = Path(__file__).parent / "recordings"

_CASE_FILES = discover_case_files(_CASES_DIR)


@pytest.mark.parametrize("case_file", _CASE_FILES, ids=[p.stem for p in _CASE_FILES])
def test_case_schema_is_valid(case_file: Path) -> None:
    load_case(case_file)


@pytest.mark.parametrize("case_file", _CASE_FILES, ids=[p.stem for p in _CASE_FILES])
def test_case_replay(case_file: Path) -> None:
    case = load_case(case_file)
    calls = load_recording(recording_path(_RECORDINGS_DIR, case.id))
    client = ReplayOllamaClient(calls)

    result = run_case(case, client)

    failures = evaluate_assertions(case, result)
    assert not failures, "\n".join(failures)

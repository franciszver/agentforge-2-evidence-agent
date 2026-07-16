"""Red-first tests for the eval harness's RUNNER MECHANICS (P4.7) -- not
eval content. These prove the harness itself behaves correctly against small
FIXTURE cases under ``evals/runner/tests/fixtures/`` (never collected by the
real suite entry point, ``evals/test_cases.py``, which only scans
``evals/cases/``):

  * a case whose assertions PASS on replay -> the runner reports pass
  * a case whose assertion FAILS on replay -> the runner reports fail (an
    eval failure is a test failure)
  * a case file with malformed YAML syntax -> the loader errors clearly
  * a case file that parses as YAML but fails schema validation (unknown
    category, unknown assertion type) -> the loader errors clearly
  * a case with no committed recording -> replay fails clearly (the decided
    default: FAIL, not skip -- see ``runner.ollama_replay``)
  * a recording whose call sequence doesn't match what the pipeline actually
    requests -> replay fails clearly, not silently
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runner.assertions import evaluate_assertions
from runner.loader import load_case
from runner.ollama_replay import (
    RecordedCall,
    RecordingMismatchError,
    RecordingNotFoundError,
    ReplayOllamaClient,
    load_recording,
    recording_path,
)
from runner.pipeline import run_case
from runner.schema import EvalCaseError

_FIXTURES = Path(__file__).parent / "fixtures"
_CASES = _FIXTURES / "cases"
_RECORDINGS = _FIXTURES / "recordings"


def _run_fixture(case_id: str) -> list[str]:
    case = load_case(_CASES / f"{case_id}.yaml")
    calls = load_recording(recording_path(_RECORDINGS, case.id))
    client = ReplayOllamaClient(calls)
    result = run_case(case, client)
    return evaluate_assertions(case, result)


# --- pass ---------------------------------------------------------------


def test_case_with_passing_assertions_reports_no_failures() -> None:
    failures = _run_fixture("pass")
    assert failures == []


# --- fail (an eval failure is a test failure) ----------------------------


def test_case_with_failing_assertion_reports_the_failure() -> None:
    failures = _run_fixture("fail")
    assert len(failures) == 1
    assert "first_tool_in" in failures[0]
    assert "get_allergies" in failures[0]
    assert "get_medications" in failures[0]


# --- malformed cases fail clearly, at load time --------------------------


def test_malformed_yaml_syntax_raises_a_clear_error() -> None:
    with pytest.raises(EvalCaseError, match="malformed YAML"):
        load_case(_CASES / "malformed-yaml.yaml")


def test_schema_invalid_case_raises_a_clear_error() -> None:
    with pytest.raises(EvalCaseError, match="schema validation failed"):
        load_case(_CASES / "invalid-schema.yaml")


# --- missing recording: FAIL, not skip ------------------------------------


def test_missing_recording_raises_a_clear_error() -> None:
    case = load_case(_CASES / "missing-recording.yaml")
    with pytest.raises(RecordingNotFoundError, match="no recording at"):
        load_recording(recording_path(_RECORDINGS, case.id))


# --- a stale/rotted recording is caught, not silently accepted -----------


def test_recording_sequence_mismatch_raises_a_clear_error() -> None:
    """A recording whose next call doesn't match what the pipeline actually
    requests (e.g. edited out of sync with the case) must not be silently
    replayed as if it were correct."""
    case = load_case(_CASES / "pass.yaml")
    real_calls = load_recording(recording_path(_RECORDINGS, case.id))
    # Corrupt the first call's schema so it no longer matches what the
    # planner's first turn actually requests (PlannerDecision).
    corrupted = [RecordedCall(kind="extract", schema="FinalAnswer", response={"answer": "wrong"})] + list(
        real_calls[1:]
    )
    client = ReplayOllamaClient(corrupted)

    with pytest.raises(RecordingMismatchError, match="recording mismatch"):
        run_case(case, client)

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

import json
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
    save_recording,
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


# --- semantic-support gate (issue #81): the runner must actually wire a
# support judge into citation_present cases' verification pass, not just
# replay a recording that happens to have a pre-baked SemanticSupportJudgement
# call sitting in it unused. Both fixtures share the SAME (question, chart
# data, retrieved chunk) setup -- only the recorded judge verdict differs --
# so a passing "pass" fixture and a failing "downgrade" fixture together prove
# the wiring genuinely branches on the judge's response rather than always
# passing (or always failing) regardless of what it says.


def test_semantic_support_gate_downgrades_a_judge_rejected_citation() -> None:
    failures = _run_fixture("semantic-support-downgrade")
    assert failures, "a judge verdict of not_supported must fail the case, not silently verify"
    assert any("guideline_citation_present" in failure for failure in failures)


def test_semantic_support_gate_leaves_a_judge_supported_citation_verified() -> None:
    failures = _run_fixture("semantic-support-pass")
    assert failures == []


# --- fail (an eval failure is a test failure) ----------------------------


def test_case_with_failing_assertion_reports_the_failure() -> None:
    failures = _run_fixture("fail")
    assert len(failures) == 1
    assert "first_tool_in" in failures[0]
    assert "get_allergies" in failures[0]
    assert "get_medications" in failures[0]


# --- xfail (P4.8): a documented, honest known-failure case is reported as an
# expected failure, not a hard failure -- and the marker is REAL (added
# dynamically from the case's own ``xfail`` field, exactly as
# ``evals/test_cases.py``'s ``test_case_replay`` does), so this test itself
# only passes (shows as ``x``) because its assertions genuinely fail.


def test_xfail_case_is_marked_and_reports_as_xfail(request: pytest.FixtureRequest) -> None:
    case = load_case(_CASES / "xfail-known-failure.yaml")
    assert case.xfail, "fixture must declare a truthy xfail reason for this test to mean anything"
    request.node.add_marker(pytest.mark.xfail(reason=case.xfail, strict=True))

    failures = _run_fixture("xfail-known-failure")
    assert failures, "fixture's assertion must genuinely fail -- an xfail case that secretly passes proves nothing"
    pytest.fail("\n".join(failures))


# --- malformed cases fail clearly, at load time --------------------------


def test_malformed_yaml_syntax_raises_a_clear_error() -> None:
    with pytest.raises(EvalCaseError, match="malformed YAML"):
        load_case(_CASES / "malformed-yaml.yaml")


def test_schema_invalid_case_raises_a_clear_error() -> None:
    with pytest.raises(EvalCaseError, match="schema validation failed"):
        load_case(_CASES / "invalid-schema.yaml")


def test_case_without_failure_mode_raises_validation_error() -> None:
    """A case without a failure_mode must be rejected at load time (policy:
    every case documents why it tests what it does). All 31 cases in evals/cases/
    have failure_mode; this proves the schema enforces it for any new case."""
    with pytest.raises(EvalCaseError, match="schema validation failed"):
        load_case(_CASES / "missing-failure-mode.yaml")


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


# --- recording metadata: code-stamp audit trail (#140) --------------------


def test_save_recording_stamps_code_stamp_into_metadata(tmp_path: Path) -> None:
    """A recording written with a ``code_stamp`` (record.py's job -- see
    ``test_record_code_stamp_guard.py``) must persist it in the artifact so
    a future audit can tell what code produced the recording."""
    out_path = tmp_path / "some-case.json"
    calls = [RecordedCall(kind="chat", schema=None, response="hello")]

    save_recording(out_path, calls, code_stamp="abc123")

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["code_stamp"] == "abc123"
    # The calls themselves must still round-trip untouched.
    assert load_recording(out_path) == calls


def test_recordings_without_a_code_stamp_still_load_and_replay(tmp_path: Path) -> None:
    """Old, already-committed recordings predate #140 and have no
    ``code_stamp`` key at all -- they must keep replaying exactly as before
    (this repo does not rewrite committed recordings), and an audit reading
    them back gets an honest "unknown", not a KeyError."""
    out_path = tmp_path / "pre-140-case.json"
    out_path.write_text(
        json.dumps({"calls": [{"kind": "chat", "schema": None, "response": "hi"}]}),
        encoding="utf-8",
    )

    calls = load_recording(out_path)

    assert calls == [RecordedCall(kind="chat", schema=None, response="hi")]


def test_save_recording_omits_code_stamp_key_when_not_given(tmp_path: Path) -> None:
    """Callers that don't pass ``code_stamp`` (none exist today, but the
    parameter is optional) get the old payload shape exactly -- no stray
    ``null`` key clutter."""
    out_path = tmp_path / "no-stamp-case.json"
    calls = [RecordedCall(kind="chat", schema=None, response="hi")]

    save_recording(out_path, calls)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "code_stamp" not in payload

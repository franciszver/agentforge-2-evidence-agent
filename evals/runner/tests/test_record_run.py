"""Red-first tests for ``evals/runner/record_run.py`` (P4.10): the eval-run
recording mechanism that appends ONE run's aggregate pass/fail/xfail counts
to the committed, agent-packaged history the dashboard chart reads.

Reuses the P4.7 harness self-test fixtures under
``evals/runner/tests/fixtures/`` (never collected by the real suite entry
point, ``evals/test_cases.py``) for the case-level classification tests, plus
one dedicated fixture (``xfail-unexpectedly-passing``) added in this PR to
prove the STALE-xfail accounting decision. ``aggregate_outcomes`` is tested
purely (no I/O) against hand-built outcome lists for the pass-rate/xfail
accounting itself.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from runner.record_run import _case_outcome, aggregate_outcomes, compute_run_result

_FIXTURES = Path(__file__).parent / "fixtures"
_CASES = _FIXTURES / "cases"
_RECORDINGS = _FIXTURES / "recordings"

_PASS_RECORDING = json.loads((_RECORDINGS / "fixture-pass.json").read_text(encoding="utf-8"))
_FAIL_RECORDING = json.loads((_RECORDINGS / "fixture-fail.json").read_text(encoding="utf-8"))


# --- _case_outcome: per-case classification ---------------------------


def test_case_outcome_passing_case_reports_passed() -> None:
    assert _case_outcome(_CASES / "pass.yaml", _RECORDINGS) == "passed"


def test_case_outcome_failing_case_reports_failed() -> None:
    assert _case_outcome(_CASES / "fail.yaml", _RECORDINGS) == "failed"


def test_case_outcome_genuinely_failing_xfail_reports_xfailed() -> None:
    assert _case_outcome(_CASES / "xfail-known-failure.yaml", _RECORDINGS) == "xfailed"


def test_case_outcome_missing_recording_reports_failed() -> None:
    assert _case_outcome(_CASES / "missing-recording.yaml", _RECORDINGS) == "failed"


def test_case_outcome_schema_drifted_recording_reports_failed_not_raises(tmp_path: Path) -> None:
    # A recorded extract payload that passes the call kind/schema-name check
    # but no longer validates against its extraction schema (a drifted
    # recording) is a "recording diverged from the case" -- it must be
    # counted as a failing replay, NOT propagate a pydantic.ValidationError
    # that aborts the whole recording run. Mirrors the RecordingMismatchError
    # handling and pytest's per-case isolation.
    cases_dir = tmp_path / "cases"
    recordings_dir = tmp_path / "recordings"
    cases_dir.mkdir()
    recordings_dir.mkdir()

    (cases_dir / "drift.yaml").write_text(
        "id: drift\n"
        "category: tool_selection\n"
        'question: "What meds is she on?"\n'
        "patient_id: 1\n"
        "tool_data:\n"
        "  get_medications:\n"
        "    items:\n"
        "      - name: Lisinopril\n"
        "        dose: 10mg\n"
        "        route: oral\n"
        "        status: active\n"
        "assertions:\n"
        "  - type: first_tool_in\n"
        "    tools: [get_medications]\n",
        encoding="utf-8",
    )
    # Corrupt the first (PlannerDecision) extract payload so model_validate
    # raises ValidationError -- kind/schema-name still match, only the shape
    # is wrong, so this reaches schema.model_validate rather than tripping the
    # earlier RecordingMismatch guard.
    drifted = copy.deepcopy(_PASS_RECORDING)
    drifted["calls"][0]["response"] = {"bogus": "no longer a valid PlannerDecision"}
    (recordings_dir / "drift.json").write_text(json.dumps(drifted), encoding="utf-8")

    assert _case_outcome(cases_dir / "drift.yaml", recordings_dir) == "failed"


def test_case_outcome_stale_xfail_that_now_passes_reports_failed() -> None:
    # An xfail whose assertions genuinely pass on replay is NOT a pass and
    # NOT a documented known-failure -- it is a stale marker that needs to
    # be removed by a human. Mirrors pytest's strict=True unexpected-pass
    # semantics (docs/TEST_PLAN.md Sec 5).
    assert _case_outcome(_CASES / "xfail-unexpectedly-passing.yaml", _RECORDINGS) == "failed"


# --- aggregate_outcomes: the pure pass-rate/xfail accounting core -----


def test_aggregate_outcomes_empty_history_is_zero_total() -> None:
    point = aggregate_outcomes([], git_sha="0000000", timestamp="2026-07-16T00:00:00+00:00")
    assert point.total == 0
    assert point.passed == 0
    assert point.pass_rate == 0.0


def test_aggregate_outcomes_all_passed_is_pass_rate_one() -> None:
    point = aggregate_outcomes(
        ["passed", "passed", "passed"], git_sha="1111111", timestamp="2026-07-16T00:00:00+00:00"
    )
    assert (point.total, point.passed, point.failed, point.xfailed) == (3, 3, 0, 0)
    assert point.pass_rate == 1.0


def test_aggregate_outcomes_xfail_is_not_counted_as_a_pass() -> None:
    # 2 passed, 1 documented known-failure (xfailed) out of 3 total -- the
    # xfail case must NOT inflate pass_rate to 3/3. It also must not shrink
    # the denominator to 2 (excluding it from total) -- pass_rate stays an
    # honest fraction of everything the suite covers.
    point = aggregate_outcomes(
        ["passed", "passed", "xfailed"], git_sha="2222222", timestamp="2026-07-16T00:00:00+00:00"
    )
    assert (point.total, point.passed, point.failed, point.xfailed) == (3, 2, 0, 1)
    assert point.pass_rate == 2 / 3


def test_aggregate_outcomes_failed_case_lowers_pass_rate() -> None:
    point = aggregate_outcomes(
        ["passed", "failed"], git_sha="3333333", timestamp="2026-07-16T00:00:00+00:00"
    )
    assert (point.total, point.passed, point.failed, point.xfailed) == (2, 1, 1, 0)
    assert point.pass_rate == 0.5


def test_aggregate_outcomes_carries_git_sha_and_timestamp_through() -> None:
    point = aggregate_outcomes(["passed"], git_sha="4444444", timestamp="2026-07-16T09:30:00+00:00")
    assert point.git_sha == "4444444"
    assert point.timestamp == "2026-07-16T09:30:00+00:00"


# --- compute_run_result: discovery + classification + aggregation wired ---


def test_compute_run_result_wires_discovery_through_to_a_run_point(tmp_path: Path) -> None:
    # A dedicated, self-contained pair of cases (pass + fail) under tmp_path
    # so this test doesn't depend on -- or break from -- the harness's other
    # intentionally-malformed self-test fixtures (invalid-schema.yaml,
    # malformed-yaml.yaml) living in the shared fixtures/cases directory.
    cases_dir = tmp_path / "cases"
    recordings_dir = tmp_path / "recordings"
    cases_dir.mkdir()
    recordings_dir.mkdir()

    (cases_dir / "t-pass.yaml").write_text(
        "id: t-pass\n"
        "category: tool_selection\n"
        'question: "What meds is she on?"\n'
        "patient_id: 1\n"
        "tool_data:\n"
        "  get_medications:\n"
        "    items:\n"
        "      - name: Lisinopril\n"
        "        dose: 10mg\n"
        "        route: oral\n"
        "        status: active\n"
        "assertions:\n"
        "  - type: first_tool_in\n"
        "    tools: [get_medications]\n",
        encoding="utf-8",
    )
    (recordings_dir / "t-pass.json").write_text(json.dumps(_PASS_RECORDING), encoding="utf-8")

    (cases_dir / "t-fail.yaml").write_text(
        "id: t-fail\n"
        "category: tool_selection\n"
        'question: "What meds is she on?"\n'
        "patient_id: 1\n"
        "tool_data:\n"
        "  get_medications:\n"
        "    items: []\n"
        "assertions:\n"
        "  - type: first_tool_in\n"
        "    tools: [get_allergies]\n",  # recording dispatches get_medications -> mismatch -> fails
        encoding="utf-8",
    )
    (recordings_dir / "t-fail.json").write_text(json.dumps(_FAIL_RECORDING), encoding="utf-8")

    point = compute_run_result(
        cases_dir=cases_dir,
        regressions_dir=tmp_path / "no-such-regressions-dir",
        recordings_dir=recordings_dir,
        git_sha="5555555",
    )

    assert (point.total, point.passed, point.failed, point.xfailed) == (2, 1, 1, 0)
    assert point.pass_rate == 0.5
    assert point.git_sha == "5555555"

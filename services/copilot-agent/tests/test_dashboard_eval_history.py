"""Red-first tests for the P4.10 eval pass-rate-over-time aggregation
(``app.dashboard_eval_history``).

Pure reader over a committed JSON history file -- no ``TraceStore``, no
network, no live eval run. Mirrors ``test_dashboard_alerts.py``'s style:
hand-built fixture files under ``tmp_path``, one behavior per test.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.dashboard_eval_history import EvalRunPoint, append_eval_run, load_eval_history

_POINT_A = {
    "timestamp": "2026-07-10T12:00:00+00:00",
    "git_sha": "aaa1111",
    "total": 10,
    "passed": 8,
    "failed": 0,
    "xfailed": 2,
    "pass_rate": 0.8,
}
_POINT_B = {
    "timestamp": "2026-07-15T12:00:00+00:00",
    "git_sha": "bbb2222",
    "total": 12,
    "passed": 10,
    "failed": 0,
    "xfailed": 2,
    "pass_rate": 10 / 12,
}


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- empty / missing -------------------------------------------------------


def test_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert load_eval_history(tmp_path / "does-not-exist.json") == []


def test_empty_list_file_returns_empty_list(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    _write(history_path, [])
    assert load_eval_history(history_path) == []


# --- single run --------------------------------------------------------


def test_single_run_returns_one_point(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    _write(history_path, [_POINT_A])

    points = load_eval_history(history_path)

    assert points == [
        EvalRunPoint(
            timestamp="2026-07-10T12:00:00+00:00",
            git_sha="aaa1111",
            total=10,
            passed=8,
            failed=0,
            xfailed=2,
            pass_rate=0.8,
        )
    ]


# --- multiple runs, ordering -------------------------------------------


def test_multiple_runs_are_ordered_by_timestamp_ascending(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    # Written out of order -- the reader must sort, not trust file order.
    _write(history_path, [_POINT_B, _POINT_A])

    points = load_eval_history(history_path)

    assert [p.timestamp for p in points] == [_POINT_A["timestamp"], _POINT_B["timestamp"]]


# --- malformed / missing file: graceful, never raises -------------------


def test_malformed_json_returns_empty_list(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text("{not valid json", encoding="utf-8")

    assert load_eval_history(history_path) == []


def test_non_list_top_level_returns_empty_list(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    _write(history_path, {"oops": "this should be a list"})

    assert load_eval_history(history_path) == []


def test_entry_missing_required_keys_is_skipped_not_fatal(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    broken = {"timestamp": "2026-07-01T00:00:00+00:00", "git_sha": "ccc3333"}  # missing counts
    _write(history_path, [broken, _POINT_A])

    points = load_eval_history(history_path)

    assert [p.git_sha for p in points] == ["aaa1111"]


# --- append_eval_run: the write side used by evals/runner/record_run.py ---


def test_append_eval_run_creates_file_when_absent(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    point = EvalRunPoint(
        timestamp="2026-07-16T00:00:00+00:00",
        git_sha="ddd4444",
        total=5,
        passed=5,
        failed=0,
        xfailed=0,
        pass_rate=1.0,
    )

    append_eval_run(point, history_path)

    assert load_eval_history(history_path) == [point]


def test_append_eval_run_appends_to_existing_history(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    _write(history_path, [_POINT_A])
    new_point = EvalRunPoint(
        timestamp="2026-07-20T00:00:00+00:00",
        git_sha="eee5555",
        total=13,
        passed=11,
        failed=0,
        xfailed=2,
        pass_rate=11 / 13,
    )

    append_eval_run(new_point, history_path)

    points = load_eval_history(history_path)
    assert len(points) == 2
    assert points[-1].git_sha == "eee5555"

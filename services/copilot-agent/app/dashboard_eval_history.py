"""Eval pass-rate-over-time history for the P4.10 dashboard chart.

Reads the committed eval-run history that ``evals/runner/record_run.py``
appends to via :func:`append_eval_run`. The history file lives at an
agent-PACKAGED path -- ``app/data/eval_history.json`` -- because the agent
container ships ``app/``, not ``evals/`` (the same reason
``app/data/drug_interactions.db`` is a committed ``app/data`` artifact
rather than something built at request time; see that module's docstring).
``evals/runner/record_run.py`` runs from the evals/ side of the repo but
writes straight into this file -- no evals/-side copy of the history, one
committed source of truth read by both the dashboard and (eventually) the
P5.3 README results table.

**Pass-rate accounting.** A case marked ``xfail`` (P4.8) is a DOCUMENTED
known failure, not a pass -- counting it as a pass would hide it inside a
green number. So each recorded run's ``pass_rate`` is ``passed / total``
where ``total`` is EVERY case (genuinely-passing + xfailed +
unexpectedly-failed) and ``passed`` counts only cases that passed with no
xfail marker. ``xfailed`` is recorded alongside ``pass_rate`` so the
dashboard/README show both, rather than xfail cases silently inflating (if
counted as passes) or silently deflating the denominator (if excluded from
``total``) as more known-failures get documented over time. See
``evals/runner/record_run.py`` for where this accounting is actually
computed at record time -- this module only reads/writes the committed
result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

EVAL_HISTORY_PATH = Path(__file__).resolve().parent / "data" / "eval_history.json"

_REQUIRED_KEYS = {"timestamp", "git_sha", "total", "passed", "failed", "xfailed", "pass_rate"}


@dataclass(frozen=True)
class EvalRunPoint:
    """One recorded eval run -- one point on the pass-rate-over-time chart."""

    timestamp: str  # ISO 8601
    git_sha: str
    total: int
    passed: int
    failed: int
    xfailed: int
    pass_rate: float  # passed / total, xfailed cases counted as neither


def load_eval_history(history_path: Path = EVAL_HISTORY_PATH) -> list[EvalRunPoint]:
    """Read the committed eval-run history, ordered by timestamp ascending.

    Graceful on a missing or malformed file -- returns ``[]`` rather than
    raising, so a broken/absent history degrades the chart to its "no data"
    state instead of crashing the dashboard. Individual malformed entries
    (missing keys, wrong types) are skipped rather than discarding the
    whole file, so one bad row doesn't erase every good one.
    """
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []

    points: list[EvalRunPoint] = []
    for entry in raw:
        if not isinstance(entry, dict) or not _REQUIRED_KEYS.issubset(entry):
            continue
        try:
            points.append(
                EvalRunPoint(
                    timestamp=str(entry["timestamp"]),
                    git_sha=str(entry["git_sha"]),
                    total=int(entry["total"]),
                    passed=int(entry["passed"]),
                    failed=int(entry["failed"]),
                    xfailed=int(entry["xfailed"]),
                    pass_rate=float(entry["pass_rate"]),
                )
            )
        except (TypeError, ValueError):
            continue
    return sorted(points, key=lambda point: point.timestamp)


def append_eval_run(point: EvalRunPoint, history_path: Path = EVAL_HISTORY_PATH) -> None:
    """Append one recorded run to the committed history file, creating it
    (and its parent directory) if absent. Used by
    ``evals/runner/record_run.py``, never by the dashboard read path."""
    existing: list[dict[str, object]] = []
    if history_path.exists():
        try:
            raw = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                existing = raw
        except (OSError, json.JSONDecodeError):
            existing = []

    existing.append(asdict(point))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

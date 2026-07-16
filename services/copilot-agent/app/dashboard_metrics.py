"""Aggregation queries over the P4.2 trace store, for the P4.5 dashboard.

Pure reader: opens its own short-lived ``sqlite3`` connection against an
injected ``db_path`` (production points at ``Settings.trace_db_path``; every
test points at a ``tmp_path`` file -- see ``docs/TEST_PLAN.md`` Sec 7) and
returns a ``DashboardMetrics`` DTO. No writes, no side effects.

**Latency percentiles** are computed over ``request`` span durations (the
whole-invocation timing already recorded once per ``POST /chat`` call --
``app.chat._stream_chat``'s ``request_start_ts``/``request_end_ts``), using
the nearest-rank method (see ``_percentile``): for N sorted values, the P-th
percentile is the value at 1-based rank ``ceil(P/100 * N)``. Chosen over
linear interpolation for being exactly reproducible with integer arithmetic
and because it always returns an OBSERVED duration rather than an
interpolated one between two observations.

**Retry count.** The trace store has no explicit "retry" column (a retry is
just another dispatch of the same tool). A FAILED tool span is the signal
that a retry was (or should have been) attempted, so ``retry_count`` counts
``tool`` spans with ``status = 'fail'`` -- the number of tool calls that did
NOT succeed on their recorded attempt. ``tool_call_count`` counts ALL tool
spans (successes and failures both), so it and ``retry_count`` are reported
side by side rather than one being a subset presented alone.

**Feedback dedup (#54).** The P4.4 UI can write TWO feedback spans for one
thumbs-down: an immediate ``{thumb: down}`` on click, then a second
``{thumb: down, comment: "..."}`` if the clinician follows up with a comment.
Both spans share ``correlation_id`` and ``span_type = 'feedback'``. Counting
every row would double-count that single clinician action. ``_FEEDBACK_SQL``
dedupes by ``correlation_id`` via a window function, keeping one row per
correlation id -- preferring the row WITH a comment (more informative) and,
among ties, the most recently written row (highest ``id``).
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardMetrics:
    """Aggregate, non-PHI metrics rendered by ``GET /dashboard``.

    Rate/percentile fields are ``None`` (not a fabricated ``0.0``) when there
    is no underlying data to compute them from -- the page renders those as
    an explicit "N/A" rather than a misleading zero.
    """

    request_count: int
    error_rate: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    avg_tokens_per_request: float | None
    tool_call_count: int
    retry_count: int
    verification_pass_rate: float | None
    feedback_up_count: int
    feedback_down_count: int


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    """Nearest-rank percentile over already-sorted ascending values.

    ``None`` for an empty input. Rank is 1-based: ``ceil(percentile/100 * n)``,
    clamped into ``[1, n]`` before indexing.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    rank = math.ceil(percentile / 100 * n)
    rank = max(1, min(n, rank))
    return sorted_values[rank - 1]


_FEEDBACK_DEDUP_SQL = """\
WITH ranked AS (
    SELECT
        feedback_thumb,
        ROW_NUMBER() OVER (
            PARTITION BY correlation_id
            ORDER BY (feedback_comment IS NOT NULL) DESC, id DESC
        ) AS rn
    FROM spans
    WHERE span_type = 'feedback'
)
SELECT feedback_thumb, COUNT(*) FROM ranked WHERE rn = 1 GROUP BY feedback_thumb
"""


def compute_dashboard_metrics(db_path: str) -> DashboardMetrics:
    """Compute all dashboard metrics from the ``spans`` table at ``db_path``.

    Assumes the schema already exists (``TraceStore.__init__`` creates it
    idempotently) -- callers obtain ``db_path`` from an already-constructed
    ``TraceStore`` so this never runs against a schema-less file.
    """
    connection = sqlite3.connect(db_path)
    try:
        request_count = _scalar(connection, "SELECT COUNT(*) FROM spans WHERE span_type = 'request'")
        fail_request_count = _scalar(
            connection, "SELECT COUNT(*) FROM spans WHERE span_type = 'request' AND status = 'fail'"
        )
        error_rate = (fail_request_count / request_count) if request_count else None

        durations = [
            row[0]
            for row in connection.execute(
                "SELECT duration_ms FROM spans WHERE span_type = 'request' ORDER BY duration_ms"
            )
        ]
        p50 = _percentile(durations, 50)
        p95 = _percentile(durations, 95)

        total_tokens = connection.execute(
            "SELECT COALESCE(SUM(COALESCE(tokens_in, 0) + COALESCE(tokens_out, 0)), 0) "
            "FROM spans WHERE span_type = 'llm'"
        ).fetchone()[0]
        avg_tokens_per_request = (total_tokens / request_count) if request_count else None

        tool_call_count = _scalar(connection, "SELECT COUNT(*) FROM spans WHERE span_type = 'tool'")
        retry_count = _scalar(
            connection, "SELECT COUNT(*) FROM spans WHERE span_type = 'tool' AND status = 'fail'"
        )

        verification_total = _scalar(connection, "SELECT COUNT(*) FROM spans WHERE span_type = 'verification'")
        verification_verified = _scalar(
            connection,
            "SELECT COUNT(*) FROM spans WHERE span_type = 'verification' AND verdict = 'verified'",
        )
        verification_pass_rate = (
            (verification_verified / verification_total) if verification_total else None
        )

        feedback_up_count = 0
        feedback_down_count = 0
        for thumb, count in connection.execute(_FEEDBACK_DEDUP_SQL).fetchall():
            if thumb == "up":
                feedback_up_count = count
            elif thumb == "down":
                feedback_down_count = count

        return DashboardMetrics(
            request_count=request_count,
            error_rate=error_rate,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            avg_tokens_per_request=avg_tokens_per_request,
            tool_call_count=tool_call_count,
            retry_count=retry_count,
            verification_pass_rate=verification_pass_rate,
            feedback_up_count=feedback_up_count,
            feedback_down_count=feedback_down_count,
        )
    finally:
        connection.close()


def _scalar(connection: sqlite3.Connection, sql: str) -> int:
    row = connection.execute(sql).fetchone()
    return int(row[0])

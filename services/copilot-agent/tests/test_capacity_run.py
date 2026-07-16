"""Tests for the capacity load-test harness's pure aggregation (P5.1/#60).

Only ``summarize()`` is under test here -- no network, no subprocess, no
``httpx``. The percentile math is hand-verified against linear interpolation
(matches ``numpy``'s default method) in the comments below.
"""

from __future__ import annotations

import pytest

from scripts.capacity_run import RequestResult, summarize


def _result(
    start: float,
    end: float,
    *,
    status: int | None = 200,
    done: bool = True,
    verdict: str | None = None,
    error: str | None = None,
) -> RequestResult:
    return RequestResult(
        start_ts=start,
        end_ts=end,
        status_code=status,
        done_received=done,
        verdict=verdict,
        error=error,
    )


def test_summarize_empty_results():
    summary = summarize([])

    assert summary.count == 0
    assert summary.successes == 0
    assert summary.failures == 0
    assert summary.error_breakdown == {}
    assert summary.p50_latency_s == 0.0
    assert summary.p95_latency_s == 0.0
    assert summary.max_latency_s == 0.0
    assert summary.throughput_rps == 0.0
    assert summary.verdict_histogram == {}


def test_summarize_all_success():
    # latencies (sorted): 1.0, 2.0, 4.0
    # p50: rank=(3-1)*0.5=1.0 -> exactly sorted[1] = 2.0
    # p95: rank=(3-1)*0.95=1.9 -> sorted[1]*0.1 + sorted[2]*0.9 = 0.2+3.6 = 3.8
    # throughput: total wall = max(end)-min(start) = 4.0-0.0 = 4.0; 3 successes / 4.0 = 0.75
    results = [
        _result(0.0, 1.0, verdict="verified"),
        _result(0.0, 4.0, verdict="verified"),
        _result(0.0, 2.0, verdict="partially_verified"),
    ]

    summary = summarize(results)

    assert summary.count == 3
    assert summary.successes == 3
    assert summary.failures == 0
    assert summary.error_breakdown == {}
    assert summary.p50_latency_s == 2.0
    assert summary.p95_latency_s == 3.8
    assert summary.max_latency_s == 4.0
    assert summary.throughput_rps == 0.75
    assert summary.verdict_histogram == {"verified": 2, "partially_verified": 1}


def test_summarize_mixed_success_and_failure():
    # latencies (sorted): 1.0, 2.0, 3.0, 5.0
    # p50: rank=(4-1)*0.5=1.5 -> sorted[1]*0.5 + sorted[2]*0.5 = 1.0+1.5 = 2.5
    # p95: rank=(4-1)*0.95=2.85 -> sorted[2]*0.15 + sorted[3]*0.85 = 0.45+4.25 = 4.7
    # throughput: total wall = 5.0-0.0 = 5.0; 2 successes / 5.0 = 0.4
    results = [
        _result(0.0, 1.0, verdict="verified"),
        _result(0.0, 3.0, verdict="blocked"),
        _result(0.0, 2.0, status=500, done=False, error="http_500"),
        _result(0.0, 5.0, status=None, done=False, error="timeout: read timed out"),
    ]

    summary = summarize(results)

    assert summary.count == 4
    assert summary.successes == 2
    assert summary.failures == 2
    assert summary.error_breakdown == {"http_500": 1, "timeout": 1}
    assert summary.p50_latency_s == 2.5
    assert summary.p95_latency_s == pytest.approx(4.7)
    assert summary.max_latency_s == 5.0
    assert summary.throughput_rps == 0.4
    assert summary.verdict_histogram == {"verified": 1, "blocked": 1}


def test_summarize_single_result():
    # Single-element percentile path: p50/p95/max all collapse to the one value.
    results = [_result(10.0, 12.5, verdict="verified")]

    summary = summarize(results)

    assert summary.count == 1
    assert summary.successes == 1
    assert summary.failures == 0
    assert summary.p50_latency_s == 2.5
    assert summary.p95_latency_s == 2.5
    assert summary.max_latency_s == 2.5
    assert summary.throughput_rps == 1 / 2.5
    assert summary.verdict_histogram == {"verified": 1}


def test_summarize_failure_without_error_message_falls_back_to_status_code():
    # No exception string, but status != 200 and no done frame -> categorized by status.
    results = [_result(0.0, 1.0, status=503, done=False, error=None)]

    summary = summarize(results)

    assert summary.failures == 1
    assert summary.error_breakdown == {"http_503": 1}


def test_summarize_failure_with_no_done_frame_and_ok_status_is_unknown_category():
    # 200 status but the stream never produced a done frame and no exception fired.
    results = [_result(0.0, 1.0, status=200, done=False, error=None)]

    summary = summarize(results)

    assert summary.failures == 1
    assert summary.error_breakdown == {"no_done_frame": 1}

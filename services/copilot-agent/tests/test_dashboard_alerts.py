"""Hermetic tests for the P4.6 alert-threshold logic (app.dashboard_alerts).

Pure unit tests over ``DashboardMetrics`` instances built by hand -- no
``TraceStore``, no I/O, no database. ``evaluate_alerts`` is a pure function
of a DTO plus thresholds, so every boundary can be exercised directly
without seeding spans through the trace store.

Boundary semantics under test (see module docstring for the full rationale):
alerts fire on ``current_value > threshold``, strictly greater-than -- a
metric sitting exactly ON the threshold does NOT fire.
"""

from __future__ import annotations

import pytest

from app.dashboard_alerts import DEFAULT_THRESHOLDS, Alert, AlertThresholds, evaluate_alerts
from app.dashboard_metrics import DashboardMetrics


def _metrics(**overrides: object) -> DashboardMetrics:
    """Baseline metrics with nothing alerting, overridable per test."""
    defaults: dict[str, object] = dict(
        request_count=100,
        error_rate=0.0,
        p50_latency_ms=100.0,
        p95_latency_ms=100.0,
        avg_tokens_per_request=10.0,
        tool_call_count=0,
        retry_count=0,
        verification_pass_rate=1.0,
        feedback_up_count=0,
        feedback_down_count=0,
    )
    defaults.update(overrides)
    return DashboardMetrics(**defaults)  # type: ignore[arg-type]


def _names(alerts: list[Alert]) -> list[str]:
    return [alert.metric for alert in alerts]


# ---------------------------------------------------------------------------
# p95 latency
# ---------------------------------------------------------------------------


def test_p95_latency_just_below_threshold_does_not_fire() -> None:
    metrics = _metrics(p95_latency_ms=DEFAULT_THRESHOLDS.p95_latency_ms - 0.1)
    assert evaluate_alerts(metrics) == []


def test_p95_latency_exactly_at_threshold_does_not_fire() -> None:
    metrics = _metrics(p95_latency_ms=DEFAULT_THRESHOLDS.p95_latency_ms)
    assert evaluate_alerts(metrics) == []


def test_p95_latency_just_above_threshold_fires() -> None:
    metrics = _metrics(p95_latency_ms=DEFAULT_THRESHOLDS.p95_latency_ms + 0.1)
    alerts = evaluate_alerts(metrics)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.metric == "p95 latency"
    assert alert.current_value == DEFAULT_THRESHOLDS.p95_latency_ms + 0.1
    assert alert.threshold == DEFAULT_THRESHOLDS.p95_latency_ms
    assert alert.explanation  # non-empty what-it-means/what-to-check text
    assert "latency" in alert.explanation.lower()


def test_p95_latency_none_does_not_fire() -> None:
    metrics = _metrics(p95_latency_ms=None)
    assert evaluate_alerts(metrics) == []


# ---------------------------------------------------------------------------
# error rate
# ---------------------------------------------------------------------------


def test_error_rate_just_below_threshold_does_not_fire() -> None:
    metrics = _metrics(error_rate=DEFAULT_THRESHOLDS.error_rate - 0.001)
    assert evaluate_alerts(metrics) == []


def test_error_rate_exactly_at_threshold_does_not_fire() -> None:
    metrics = _metrics(error_rate=DEFAULT_THRESHOLDS.error_rate)
    assert evaluate_alerts(metrics) == []


def test_error_rate_just_above_threshold_fires() -> None:
    metrics = _metrics(error_rate=DEFAULT_THRESHOLDS.error_rate + 0.001)
    alerts = evaluate_alerts(metrics)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.metric == "error rate"
    assert alert.current_value == DEFAULT_THRESHOLDS.error_rate + 0.001
    assert alert.threshold == DEFAULT_THRESHOLDS.error_rate
    assert alert.explanation
    assert "error" in alert.explanation.lower()


def test_error_rate_none_does_not_fire() -> None:
    metrics = _metrics(error_rate=None)
    assert evaluate_alerts(metrics) == []


# ---------------------------------------------------------------------------
# tool-failure rate (derived: retry_count / tool_call_count)
# ---------------------------------------------------------------------------


def test_tool_failure_rate_just_below_threshold_does_not_fire() -> None:
    # 19/100 = 0.19 < 0.20 threshold
    metrics = _metrics(tool_call_count=100, retry_count=19)
    assert evaluate_alerts(metrics) == []


def test_tool_failure_rate_exactly_at_threshold_does_not_fire() -> None:
    # 20/100 = 0.20 == threshold
    metrics = _metrics(tool_call_count=100, retry_count=20)
    assert evaluate_alerts(metrics) == []


def test_tool_failure_rate_just_above_threshold_fires() -> None:
    # 21/100 = 0.21 > 0.20 threshold
    metrics = _metrics(tool_call_count=100, retry_count=21)
    alerts = evaluate_alerts(metrics)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.metric == "tool-failure rate"
    assert alert.current_value == 0.21
    assert alert.threshold == DEFAULT_THRESHOLDS.tool_failure_rate
    assert alert.explanation
    assert "tool" in alert.explanation.lower()


def test_tool_failure_rate_zero_tool_calls_does_not_fire() -> None:
    # tool_call_count == 0 -> rate is None (undefined), not 0.0 -- must not
    # fire even though retry_count is also 0. This is the P4.6 "dormant
    # until #149" case: production currently always has tool_call_count == 0.
    metrics = _metrics(tool_call_count=0, retry_count=0)
    assert evaluate_alerts(metrics) == []


# ---------------------------------------------------------------------------
# verification-fail rate (derived: 1 - verification_pass_rate)
# ---------------------------------------------------------------------------


def test_verification_fail_rate_just_below_threshold_does_not_fire() -> None:
    # fail rate = 1 - 0.71 = 0.29 < 0.30 threshold
    metrics = _metrics(verification_pass_rate=0.71)
    assert evaluate_alerts(metrics) == []


def test_verification_fail_rate_exactly_at_threshold_does_not_fire() -> None:
    # fail rate = 1 - 0.70 = 0.30 == threshold
    metrics = _metrics(verification_pass_rate=0.70)
    assert evaluate_alerts(metrics) == []


def test_verification_fail_rate_just_above_threshold_fires() -> None:
    # fail rate = 1 - 0.69 = 0.31 > 0.30 threshold
    metrics = _metrics(verification_pass_rate=0.69)
    alerts = evaluate_alerts(metrics)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.metric == "verification-fail rate"
    assert alert.current_value == pytest.approx(0.31)
    assert alert.threshold == DEFAULT_THRESHOLDS.verification_fail_rate
    assert alert.explanation
    assert "verif" in alert.explanation.lower()


def test_verification_fail_rate_none_does_not_fire() -> None:
    metrics = _metrics(verification_pass_rate=None)
    assert evaluate_alerts(metrics) == []


# ---------------------------------------------------------------------------
# multiple simultaneous alerts
# ---------------------------------------------------------------------------


def test_all_four_alerts_fire_simultaneously_in_fixed_order() -> None:
    metrics = _metrics(
        p95_latency_ms=40_000.0,
        error_rate=0.5,
        tool_call_count=10,
        retry_count=5,
        verification_pass_rate=0.5,
    )
    alerts = evaluate_alerts(metrics)
    assert _names(alerts) == ["p95 latency", "error rate", "tool-failure rate", "verification-fail rate"]


def test_zero_alerts_when_all_metrics_healthy() -> None:
    metrics = _metrics()
    assert evaluate_alerts(metrics) == []


def test_zero_alerts_when_all_metrics_none_or_empty() -> None:
    metrics = _metrics(
        error_rate=None,
        p95_latency_ms=None,
        tool_call_count=0,
        retry_count=0,
        verification_pass_rate=None,
    )
    assert evaluate_alerts(metrics) == []


# ---------------------------------------------------------------------------
# custom thresholds
# ---------------------------------------------------------------------------


def test_custom_thresholds_override_defaults() -> None:
    tight = AlertThresholds(
        p95_latency_ms=1.0, error_rate=0.0, tool_failure_rate=0.0, verification_fail_rate=0.0
    )
    metrics = _metrics(p95_latency_ms=2.0)
    alerts = evaluate_alerts(metrics, thresholds=tight)
    assert _names(alerts) == ["p95 latency"]


def test_alert_is_frozen() -> None:
    alert = Alert(metric="x", current_value=1.0, threshold=0.5, explanation="y", unit="rate")
    try:
        alert.metric = "z"  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised

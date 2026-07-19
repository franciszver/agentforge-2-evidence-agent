"""Alert-threshold evaluation for the P4.5 dashboard (P4.6, extended by
P3G.4/#24).

Pure threshold logic over ``app.dashboard_metrics.DashboardMetrics`` (plus,
as of P3G.4, the eval-run history) -- no I/O, no recomputation of metrics
that ``DashboardMetrics`` already has. Seven alerts, each read straight off
the DTO or derived from fields already on it:

- **p95 latency**: ``metrics.p95_latency_ms`` directly.
- **error rate**: ``metrics.error_rate`` directly.
- **tool-failure rate**: DERIVED as ``retry_count / tool_call_count``. The
  DTO has no failure-rate field, only the raw counts -- see
  ``dashboard_metrics.py``'s ``retry_count`` docstring: a FAILED tool span
  IS the retry signal, so this is "tool calls that did not succeed on their
  recorded attempt, as a share of all tool calls". ``None`` (no alert) when
  ``tool_call_count == 0``. NOTE: in production today ``tool_call_count`` is
  ALWAYS 0 -- ``app.chat`` does not yet emit per-tool spans live (see
  ``app.trace_store.TraceStore.record_tool_span``'s call sites, currently
  none in ``app.chat``; tracked by #149). This alert is exercised here
  against seeded/synthetic metrics and will start evaluating real data once
  #149 lands.
- **verification-fail rate**: DERIVED as ``1 - verification_pass_rate``.
  ``None`` (no alert) when ``verification_pass_rate`` is ``None``.
- **extraction-failure rate** (P3G.4/#24): ``metrics.extraction_failure_rate``
  directly -- always ``None`` today. No ``app.trace_store`` span exists yet
  for document-ingestion extraction (see ``app.ingestion.IngestionResult``'s
  ``failed_pages``), so -- same "dormant until a future issue wires the
  data" posture as tool-failure rate above -- this alert is implemented and
  tested now and will start evaluating real data once a future issue
  computes this field from live spans.
- **retrieval p95 latency** (P3G.4/#24): ``metrics.retrieval_p95_latency_ms``
  directly, likewise always ``None`` today for the same reason -- retrieval
  (``app.retrieval``) has no dedicated span yet.
- **eval-regression** (P3G.4/#24): DERIVED from the pass-rate drop between
  the two most recent points in ``eval_history``
  (``app.dashboard_eval_history.EvalRunPoint``), sorted by timestamp.
  ``None`` (no alert) when fewer than two points are available. Unlike the
  two alerts above, this one IS wired to real, already-committed data --
  ``app.dashboard.dashboard_endpoint`` passes its live ``eval_history``
  straight through.

**Boundary semantics: strictly greater-than.** An alert fires when
``current_value > threshold``, NOT ``>=``. A metric sitting exactly ON the
threshold reads as "at the edge, not yet over it" -- consistent with the
thresholds being phrased as ceilings ("p95 > 30s"), and it keeps a metric
seeded at a clean round number (e.g. exactly 10.0% error rate) from reading
as already-alerting.

**None handling: absence is not evidence of a problem.** A ``None`` metric
(empty store, or tool-failure rate when ``tool_call_count == 0``) never
fires an alert -- there is nothing to alert ON. This mirrors the dashboard's
own "N/A" rendering: no data is not the same as good data.

**All four values are rounded to 9 decimal places** before comparison (and
that rounded value is what ``Alert.current_value`` reports). Every one of
them is itself the result of floating-point division or subtraction
upstream (in ``dashboard_metrics.py`` or in this module's two derived
rates), which can land a hair off an exact decimal boundary purely from
binary float representation (e.g. ``1.0 - 0.7 == 0.30000000000000004``, not
``0.3``). Rounding uniformly, rather than only on the two locally-derived
rates, avoids the same flakiness resurfacing for p95 latency or error rate.
9 decimal places is far finer than any rate the dashboard renders (1 decimal
place) needs, so this only removes float dust -- it never masks a real
difference or a value a clinician would notice.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.dashboard_eval_history import EvalRunPoint
from app.dashboard_metrics import DashboardMetrics

# Demo-tier defaults, not tuned against production traffic (there is none
# yet). Rationale for each ceiling:
#
# - p95 latency > 30_000ms (30s): the plan's Pi-tier "slow" regime for a
#   CPU-bound local Ollama model -- a single occasionally-slow request is
#   expected, but a p95 this high means most requests are crossing it.
# - error rate > 10%: high enough that a handful of one-off failures in a
#   small demo sample won't trip it, low enough to catch a systemic issue
#   (auth, connectivity, a bad deploy) before it looks "normal".
# - tool-failure rate > 20%: one in five tool calls failing points at a
#   specific broken integration (a flaky OpenEMR endpoint, bad args) rather
#   than incidental noise.
# - verification-fail rate > 30%: verification strips/blocks unverifiable
#   claims by design (P3.7) so some non-zero fail rate is normal; 30% is
#   the point where it looks more like a grounding/citation problem than
#   the system doing its job.
# - extraction-failure rate > 15% (P3G.4/#24): a document-ingestion page
#   whose VLM extraction call fails outright contributes nothing to that
#   document's facts (``app.ingestion.IngestionResult.failed_pages``); one
#   occasional failed page in a multi-page document is expected, but more
#   than one in ~seven means the ingestion pipeline itself is degraded
#   rather than hitting isolated bad pages.
# - retrieval p95 latency > 5_000ms (P3G.4/#24): hybrid BM25+dense+rerank
#   retrieval over the guideline corpus (see ``app.retrieval``) is CPU-bound
#   by design on the minimum-spec hardware tier (embeddings/reranker run on
#   CPU to leave the GPU for the resident LLM, per
#   ``docs/W2_ARCHITECTURE.md`` Sec "Reference Hardware & Model Tiers"); 5s
#   is generous headroom over that CPU-bound baseline while still catching
#   a genuinely stuck or thrashing retrieval path.
# - eval-regression > 5% (P3G.4/#24): the same tolerance
#   ``evals/runner/gate.py``'s PR-blocking category gate (#22) already uses
#   for "did this category's pass rate regress" -- kept numerically
#   identical for one consistent "how much eval drift is tolerable"
#   answer across the CI gate and this dashboard alert, which instead
#   compares the two most recent COMMITTED eval runs
#   (``app.dashboard_eval_history``) rather than gating a single PR.
@dataclass(frozen=True)
class AlertThresholds:
    p95_latency_ms: float = 30_000.0
    error_rate: float = 0.10
    tool_failure_rate: float = 0.20
    verification_fail_rate: float = 0.30
    extraction_failure_rate: float = 0.15
    retrieval_p95_latency_ms: float = 5_000.0
    eval_pass_rate_regression: float = 0.05


DEFAULT_THRESHOLDS = AlertThresholds()


@dataclass(frozen=True)
class Alert:
    """One active alert. ``explanation`` is a hardcoded, non-PHI paragraph --
    never built from request data -- safe to render verbatim. ``unit`` tells
    the renderer how to format ``current_value``/``threshold`` (milliseconds
    vs. a percentage) without re-deriving it from ``metric``'s display text."""

    metric: str
    current_value: float
    threshold: float
    explanation: str
    unit: Literal["ms", "rate"]


_P95_LATENCY_EXPLANATION = (
    "p95 response latency has crossed the alerting threshold: at least 1 in 20 "
    "clinician requests are taking longer than expected. Check the local Ollama "
    "model for load or queueing, whether a long conversation history is "
    "inflating generation time, and whether the host is under competing CPU or "
    "memory load. Sustained high p95 erodes clinician trust even when most "
    "requests are fast."
)

_ERROR_RATE_EXPLANATION = (
    "The overall request error rate has crossed the alerting threshold. Check "
    "the OpenEMR API connection for auth or connectivity failures, recent "
    "changes to the request pipeline, and the PHP error log alongside the "
    "agent's own logs. Check /health and /ready before relying on the "
    "assistant for a live session."
)

_TOOL_FAILURE_RATE_EXPLANATION = (
    "The share of tool calls ending in failure has crossed the alerting "
    "threshold. Check whether one specific tool (labs, medications, "
    "encounters, etc.) is failing against a flaky or unreachable OpenEMR "
    "endpoint, whether timeouts are tuned correctly, and whether the planner "
    "is retrying a call that will never succeed instead of surfacing a clear "
    "error to the clinician."
)

_VERIFICATION_FAIL_RATE_EXPLANATION = (
    "The verification-fail rate has crossed the alerting threshold. Some "
    "stripped or blocked claims are expected by design -- verification exists "
    "to catch unsupported statements -- but a rate this high suggests the "
    "model is frequently generating claims it cannot ground in the retrieved "
    "records. Check recent prompt or model changes and look at a sample of "
    "blocked/partially-verified responses for a pattern."
)

_EXTRACTION_FAILURE_RATE_EXPLANATION = (
    "The document-ingestion extraction-failure rate has crossed the alerting "
    "threshold: more pages than expected are failing their VLM extraction "
    "call outright and contributing no facts. Check the VLM's health and "
    "load, whether recently-ingested documents are an unusual scan quality "
    "or format, and whether a specific document type is failing "
    "disproportionately."
)

_RETRIEVAL_LATENCY_EXPLANATION = (
    "Retrieval p95 latency has crossed the alerting threshold: the hybrid "
    "BM25+dense+rerank pipeline over the guideline corpus is taking longer "
    "than expected. Check for CPU contention with the resident LLM on the "
    "minimum-spec hardware tier, a corpus/index that has grown unexpectedly "
    "large, or a stuck/thrashing retrieval call."
)

_EVAL_REGRESSION_EXPLANATION = (
    "Eval pass rate has regressed more than the alerting threshold between "
    "the two most recently recorded runs. Check what changed between those "
    "runs (model, prompt, corpus, verification logic) and compare against "
    "evals/category_baseline.json's per-category detail to find which "
    "category is driving the drop."
)


def _eval_regression(eval_history: Sequence[EvalRunPoint]) -> float | None:
    """Pass-rate drop between the two most recent recorded eval runs, or
    ``None`` when there are fewer than two points to compare (nothing yet to
    regress against). Picks the two most recent by timestamp -- callers are
    not required to pass ``eval_history`` pre-sorted -- without sorting the
    whole history just to find them."""
    if len(eval_history) < 2:
        return None
    current, previous = heapq.nlargest(2, eval_history, key=lambda point: point.timestamp)
    return previous.pass_rate - current.pass_rate


def evaluate_alerts(
    metrics: DashboardMetrics,
    thresholds: AlertThresholds = DEFAULT_THRESHOLDS,
    *,
    eval_history: Sequence[EvalRunPoint] = (),
) -> list[Alert]:
    """Pure function: which alerts are active for ``metrics`` (plus the
    P3G.4/#24 ``eval_history`` input).

    Fixed evaluation order: p95 latency, error rate, tool-failure rate,
    verification-fail rate, extraction-failure rate, retrieval p95 latency,
    eval-regression. See module docstring for boundary (``>``, not ``>=``)
    and ``None`` handling.

    ``metrics.extraction_failure_rate``/``metrics.retrieval_p95_latency_ms``
    are always ``None`` today (see ``DashboardMetrics``'s docstring for
    those fields) because, as of P3G.4, no live span data feeds them yet --
    document ingestion and retrieval do not record a dedicated
    ``app.trace_store`` span the way requests/tools/LLM calls/verification
    do. This mirrors the tool-failure-rate alert's own documented "dormant
    until #149" precedent above: the rule is implemented and tested now, and
    will start evaluating real data once a future issue computes this field
    from live spans. ``eval_history`` (``app.dashboard_eval_history``) IS
    real, already-committed data -- the dashboard passes it live.
    """
    tool_failure_rate = (
        metrics.retry_count / metrics.tool_call_count if metrics.tool_call_count > 0 else None
    )
    verification_fail_rate = (
        1 - metrics.verification_pass_rate if metrics.verification_pass_rate is not None else None
    )
    eval_regression = _eval_regression(eval_history)

    # (metric name, current value, threshold, explanation, unit) -- one row
    # per alert, evaluated in this fixed order. All values get the same
    # 9-decimal rounding before comparison (see module docstring): direct DTO
    # reads (p95, error rate) carry the same float-division risk as the
    # derived rates, so the rounding is applied uniformly rather than only
    # where a boundary test happened to notice it.
    candidates: list[tuple[str, float | None, float, str, Literal["ms", "rate"]]] = [
        ("p95 latency", metrics.p95_latency_ms, thresholds.p95_latency_ms, _P95_LATENCY_EXPLANATION, "ms"),
        ("error rate", metrics.error_rate, thresholds.error_rate, _ERROR_RATE_EXPLANATION, "rate"),
        (
            "tool-failure rate",
            tool_failure_rate,
            thresholds.tool_failure_rate,
            _TOOL_FAILURE_RATE_EXPLANATION,
            "rate",
        ),
        (
            "verification-fail rate",
            verification_fail_rate,
            thresholds.verification_fail_rate,
            _VERIFICATION_FAIL_RATE_EXPLANATION,
            "rate",
        ),
        (
            "extraction-failure rate",
            metrics.extraction_failure_rate,
            thresholds.extraction_failure_rate,
            _EXTRACTION_FAILURE_RATE_EXPLANATION,
            "rate",
        ),
        (
            "retrieval p95 latency",
            metrics.retrieval_p95_latency_ms,
            thresholds.retrieval_p95_latency_ms,
            _RETRIEVAL_LATENCY_EXPLANATION,
            "ms",
        ),
        (
            "eval-regression",
            eval_regression,
            thresholds.eval_pass_rate_regression,
            _EVAL_REGRESSION_EXPLANATION,
            "rate",
        ),
    ]

    alerts: list[Alert] = []
    for metric, value, threshold, explanation, unit in candidates:
        if value is None:
            continue
        rounded_value = round(value, 9)
        if rounded_value > threshold:
            alerts.append(
                Alert(
                    metric=metric,
                    current_value=rounded_value,
                    threshold=threshold,
                    explanation=explanation,
                    unit=unit,
                )
            )
    return alerts

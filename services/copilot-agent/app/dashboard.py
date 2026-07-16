"""``GET /dashboard``: agent-served observability page over the P4.2 trace
store (P4.5).

**No CDN -- the agent has no internet.** Charts are rendered as inline SVG
(``_bar_svg``/``_split_bar_svg`` below), not a vendored/CDN JS charting
library. This was the simpler of the two options the task allowed: Chart.js
would mean committing and serving a ~200KB minified asset for two small
proportion bars, versus a few lines of self-contained SVG with zero moving
parts and no route/StaticFiles wiring. There is no ``http``/``https``/``cdn``
reference anywhere in the emitted HTML -- see
``tests/test_dashboard_page.py::test_dashboard_no_external_network_reference``.

**No PHI, aggregates only.** Every value rendered here comes from
``app.dashboard_metrics.DashboardMetrics`` -- counts, rates, durations,
token counts. The feedback COMMENT text is deliberately never read or
rendered by this module (individual traces + comments are the P4.9 review
queue, a different page); see
``tests/test_dashboard_page.py::test_dashboard_does_not_render_feedback_comment_text``.

**Auth posture: open, matching ``GET /chat`` (P0.6).** Neither page carries
PHI -- the chat SHELL is static markup with no patient data, and this page
is pure aggregate telemetry (no raw args, no tool results, no comments). Both
are served on the internal docker network behind the OpenEMR reverse proxy,
same posture as the rest of the agent's unauthenticated GET surface
(``/health``, ``/ready``). ``POST /chat`` and ``POST /feedback`` differ
because THEY read/write data tied to a real clinician action; a GET of
already-aggregated, non-identifying counts does not carry the same risk.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends
from fastapi.responses import HTMLResponse

from app.chat import get_trace_store
from app.dashboard_alerts import Alert, evaluate_alerts
from app.dashboard_metrics import DashboardMetrics, compute_dashboard_metrics
from app.trace_store import TraceStore

MetricsProvider = Callable[[], DashboardMetrics]


def get_metrics_provider(trace_store: TraceStore = Depends(get_trace_store)) -> MetricsProvider:
    """FastAPI dependency: builds a ``MetricsProvider`` bound to the active
    trace store's db path. Reuses ``get_trace_store`` (same dependency
    ``POST /chat`` and ``POST /feedback`` use) rather than reading
    ``Settings.trace_db_path`` directly, so:

    1. schema creation stays owned by ``TraceStore.__init__`` (no duplicated
       ``CREATE TABLE`` here), and
    2. every test that overrides ``get_trace_store`` (the existing autouse
       isolation fixture in ``tests/conftest.py``, or a per-test override)
       transparently isolates this dependency too, with zero extra plumbing.
    """
    return lambda: compute_dashboard_metrics(trace_store.db_path)


def _fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _fmt_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0f} ms"


def _fmt_tokens(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def _stat_tile(label: str, value: str) -> str:
    return f"""<div class="tile"><div class="tile-label">{label}</div><div class="tile-value">{value}</div></div>"""


def _bar_svg(*, label: str, fraction: float | None, color: str) -> str:
    """One inline SVG proportion bar (0-100%), gray track + colored fill."""
    if fraction is None:
        return f"""<svg width="100%" height="28" role="img" aria-label="{label}: no data">
<rect x="0" y="0" width="100%" height="28" fill="#e0e0e0"></rect>
<text x="8" y="19" fill="#555" font-size="14">No data yet</text>
</svg>"""
    pct = max(0.0, min(1.0, fraction)) * 100
    return f"""<svg width="100%" height="28" role="img" aria-label="{label}: {pct:.1f}%">
<rect x="0" y="0" width="100%" height="28" fill="#e0e0e0"></rect>
<rect x="0" y="0" width="{pct:.1f}%" height="28" fill="{color}"></rect>
<text x="8" y="19" fill="#1a1a1a" font-size="14">{pct:.1f}%</text>
</svg>"""


def _split_bar_svg(*, label: str, up: int, down: int) -> str:
    """One inline SVG two-segment bar: up (green) vs down (red) share of total."""
    total = up + down
    if total == 0:
        return f"""<svg width="100%" height="28" role="img" aria-label="{label}: no data">
<rect x="0" y="0" width="100%" height="28" fill="#e0e0e0"></rect>
<text x="8" y="19" fill="#555" font-size="14">No feedback yet</text>
</svg>"""
    up_pct = up / total * 100
    down_pct = down / total * 100
    return f"""<svg width="100%" height="28" role="img" aria-label="{label}: {up} up, {down} down">
<rect x="0" y="0" width="{up_pct:.1f}%" height="28" fill="#2e7d32"></rect>
<rect x="{up_pct:.1f}%" y="0" width="{down_pct:.1f}%" height="28" fill="#c62828"></rect>
<text x="8" y="19" fill="#fff" font-size="14">{up} up / {down} down</text>
</svg>"""


_STYLE = """\
  * { box-sizing: border-box; }
  .alert-banners { margin-bottom: 0.75rem; }
  .alert-banner {
    background: #fdecea;
    border: 1px solid #c62828;
    border-left-width: 6px;
    border-radius: 6px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    color: #5f1a17;
  }
  .alert-banner-title { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.25rem; }
  .alert-banner-body { font-size: 0.85rem; line-height: 1.35; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: #f5f5f5;
    color: #1a1a1a;
    padding: 1rem;
  }
  header h1 { font-size: 1.1rem; margin: 0 0 1rem 0; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }
  .tile {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 0.75rem;
  }
  .tile-label { font-size: 0.8rem; color: #555; }
  .tile-value { font-size: 1.4rem; font-weight: 600; margin-top: 0.25rem; }
  .chart-section {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 0.75rem;
    margin-top: 0.75rem;
  }
  .chart-section h2 { font-size: 0.9rem; margin: 0 0 0.5rem 0; }

  @media (min-width: 768px) {
    body { display: flex; justify-content: center; }
    .page { width: 100%; max-width: 720px; }
    .grid { grid-template-columns: 1fr 1fr 1fr 1fr; }
  }
"""


def _alert_banner(alert: Alert) -> str:
    """One P4.6 alert banner. ``alert.explanation`` is a hardcoded constant
    (see ``app.dashboard_alerts``) and the numeric values come from the
    metrics DTO -- no user-supplied text is ever interpolated here. Values are
    formatted per the alert's ``unit`` (set once where the alert is
    constructed) -- ms for latency, percentage for a rate."""
    fmt = _fmt_ms if alert.unit == "ms" else _fmt_rate
    return f"""<div class="alert-banner" data-testid="alert-banner" role="alert">
<div class="alert-banner-title">{alert.metric}: {fmt(alert.current_value)} (threshold {fmt(alert.threshold)})</div>
<div class="alert-banner-body">{alert.explanation}</div>
</div>"""


def _alert_banners_section(alerts: list[Alert]) -> str:
    """Zero active alerts renders nothing -- no empty section, no "all
    healthy" clutter on the common case."""
    if not alerts:
        return ""
    return f"""<section class="alert-banners">{"".join(_alert_banner(alert) for alert in alerts)}</section>"""


def render_dashboard_html(metrics: DashboardMetrics) -> str:
    """Render the full dashboard page for ``metrics``. Pure function of the
    DTO -- no I/O, so hermetically testable with any seeded/empty metrics."""
    alert_banners = _alert_banners_section(evaluate_alerts(metrics))
    tiles = "".join(
        [
            _stat_tile("Requests", str(metrics.request_count)),
            _stat_tile("Error rate", _fmt_rate(metrics.error_rate)),
            _stat_tile("p50 latency", _fmt_ms(metrics.p50_latency_ms)),
            _stat_tile("p95 latency", _fmt_ms(metrics.p95_latency_ms)),
            _stat_tile("Tokens / request", _fmt_tokens(metrics.avg_tokens_per_request)),
            _stat_tile("Tool calls", str(metrics.tool_call_count)),
            _stat_tile("Retries", str(metrics.retry_count)),
            _stat_tile("Verification pass rate", _fmt_rate(metrics.verification_pass_rate)),
        ]
    )

    pass_rate_bar = _bar_svg(
        label="Verification pass rate", fraction=metrics.verification_pass_rate, color="#2e7d32"
    )
    feedback_bar = _split_bar_svg(
        label="Feedback", up=metrics.feedback_up_count, down=metrics.feedback_down_count
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical Co-Pilot Dashboard</title>
<style>
{_STYLE}
</style>
</head>
<body>
<div class="page">
<header><h1>Clinical Co-Pilot Dashboard</h1></header>
{alert_banners}
<main data-testid="dashboard-metrics">
<div class="grid">
{tiles}
</div>
<section class="chart-section">
<h2>Verification pass rate</h2>
{pass_rate_bar}
</section>
<section class="chart-section">
<h2>Feedback</h2>
{feedback_bar}
</section>
</main>
</div>
</body>
</html>
"""


def dashboard_endpoint(metrics_provider: MetricsProvider = Depends(get_metrics_provider)) -> HTMLResponse:
    metrics = metrics_provider()
    return HTMLResponse(content=render_dashboard_html(metrics))

"""``GET /review`` + ``GET /review/promote``: the P4.9 review queue and
promote-to-eval action, over the P4.2 trace store / ``app.review_queue``.

**Agent-served, same posture as the P4.5 dashboard.** No CDN (inline
``<script>``/``<style>``, no external references -- see
``tests/test_review_page.py::test_review_queue_no_external_network_reference``),
served directly by this FastAPI app rather than routed through the OpenEMR
module. Where this page differs from the dashboard: the dashboard is
STRICTLY aggregate (never renders a correlation id or the feedback comment,
see ``app.dashboard``'s module docstring) -- this page's entire purpose is
the opposite, showing individual reviewable traces with their comment, so it
deliberately renders per-record fields the dashboard withholds. That is
still not PHI: correlation id, verdict, feedback thumb/comment, and counts
are exactly the non-PHI column set ``app.trace_store`` documents as safe to
persist (see its module docstring).

**HTML-escaping is load-bearing here, not decorative.** Both the
correlation id (attacker-influenceable: an inbound ``X-Correlation-ID``
header is honored verbatim by ``app.correlation.CorrelationIdMiddleware``
and becomes a trace's stored id) and the feedback comment (clinician free
text) are rendered as page content for the first time in this project --
every prior agent-served page only rendered hardcoded strings or numeric
aggregates. Every dynamic string is passed through :func:`_esc` (``html.escape``)
before being embedded in a template; see
``tests/test_review_page.py::test_review_queue_escapes_html_in_correlation_id_and_comment``.

**Auth posture: open, matching ``GET /dashboard`` (P0.6).** Both routes here
are reads: ``GET /review`` renders non-PHI aggregate-per-trace telemetry
(see above), and ``GET /review/promote`` is a pure, side-effect-free
transform of an already-stored, already non-PHI trace into YAML text -- it
writes nothing (see ``app.review_queue``'s module docstring on why the
result is returned, not written to the host repo). Unlike ``POST /feedback``
(gated: a real clinician action that writes a new signal, and open auth
would let anyone spam it), there is no write here to protect and no PHI to
disclose that a GET of the trace itself wouldn't already disclose on this
same open page.

``correlation_id`` is deliberately never interpolated into a raw HTTP
header (e.g. ``Content-Disposition``): as noted above it is
attacker-influenceable, and while ASGI servers generally reject embedded
control characters in header values, the promoted YAML is returned as a
plain response body instead of a header-driven download to avoid depending
on that for safety.
"""

from __future__ import annotations

import html
from collections.abc import Callable

from fastapi import Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.chat import get_trace_store
from app.review_queue import ReviewQueueEntry, generate_regression_case, list_review_queue
from app.trace_store import FeedbackThumb, TraceStore

QueueProvider = Callable[[], list[ReviewQueueEntry]]


def get_queue_provider(trace_store: TraceStore = Depends(get_trace_store)) -> QueueProvider:
    """FastAPI dependency: builds a ``QueueProvider`` bound to the active
    trace store, same wiring as ``app.dashboard.get_metrics_provider`` (see
    that function's docstring for why this reuses ``get_trace_store`` rather
    than reading ``Settings.trace_db_path`` directly)."""
    return lambda: list_review_queue(trace_store)


def _esc(value: str | None) -> str:
    return "" if value is None else html.escape(value)


_STYLE = """\
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: #f5f5f5;
    color: #1a1a1a;
    padding: 1rem;
  }
  header h1 { font-size: 1.1rem; margin: 0 0 1rem 0; }
  .empty { color: #555; font-style: italic; }
  .entry {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 0.75rem;
    margin-bottom: 0.75rem;
  }
  .entry-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .correlation-id { font-family: monospace; font-size: 0.85rem; color: #333; }
  .badge {
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .badge-down { background: #fdecea; color: #c62828; }
  .badge-verdict { background: #fff3e0; color: #e65100; }
  .comment { margin-top: 0.5rem; font-size: 0.9rem; }
  .meta { margin-top: 0.35rem; font-size: 0.8rem; color: #555; }
  .promote-button {
    margin-top: 0.6rem;
    padding: 0.4rem 0.8rem;
    background: #0b5a8a;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
  }
  #promote-output {
    white-space: pre-wrap;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 0.75rem;
    margin-top: 1rem;
    font-family: monospace;
    font-size: 0.8rem;
  }

  @media (min-width: 768px) {
    body { display: flex; justify-content: center; }
    .page { width: 100%; max-width: 720px; }
  }
"""

_SCRIPT = """\
document.querySelectorAll('[data-testid="promote-button"]').forEach(function (button) {
  button.addEventListener('click', function () {
    var correlationId = button.getAttribute('data-correlation-id');
    fetch('/review/promote?correlation_id=' + encodeURIComponent(correlationId))
      .then(function (response) { return response.text(); })
      .then(function (text) {
        var output = document.getElementById('promote-output');
        output.textContent = text;
      });
  });
});
"""


def _entry_card(entry: ReviewQueueEntry) -> str:
    badges = ""
    if entry.feedback_thumb == FeedbackThumb.DOWN:
        badges += '<span class="badge badge-down">thumbs down</span> '
    if entry.has_unverified_verdict:
        badges += f'<span class="badge badge-verdict">verdict: {_esc(entry.verdict)}</span>'

    comment_html = (
        f'<div class="comment">{_esc(entry.feedback_comment)}</div>'
        if entry.feedback_comment
        else ""
    )
    duration = (
        f"{entry.request_duration_ms:.0f} ms" if entry.request_duration_ms is not None else "N/A"
    )

    return f"""<div class="entry" data-testid="review-entry" data-correlation-id="{_esc(entry.correlation_id)}">
<div class="entry-header">
<span class="correlation-id">{_esc(entry.correlation_id)}</span>
<span>{badges}</span>
</div>
{comment_html}
<div class="meta">tool calls: {entry.tool_call_count} &middot; request duration: {duration}</div>
<button class="promote-button" type="button" data-testid="promote-button" data-correlation-id="{_esc(entry.correlation_id)}">Promote to regression case</button>
</div>"""


def render_review_queue_html(entries: list[ReviewQueueEntry]) -> str:
    """Render the full review queue page for ``entries``. Pure function of
    the DTO list -- no I/O, so hermetically testable with any seeded/empty
    queue."""
    body = (
        '<p class="empty">No items in the review queue.</p>'
        if not entries
        else "".join(_entry_card(entry) for entry in entries)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical Co-Pilot Review Queue</title>
<style>
{_STYLE}
</style>
</head>
<body>
<div class="page">
<header><h1>Review Queue</h1></header>
<main data-testid="review-queue">
{body}
</main>
<pre id="promote-output" data-testid="promote-output"></pre>
</div>
<script>
{_SCRIPT}
</script>
</body>
</html>
"""


def review_queue_endpoint(entries_provider: QueueProvider = Depends(get_queue_provider)) -> HTMLResponse:
    return HTMLResponse(content=render_review_queue_html(entries_provider()))


def promote_endpoint(
    correlation_id: str = Query(..., min_length=1),
    trace_store: TraceStore = Depends(get_trace_store),
) -> PlainTextResponse:
    spans = trace_store.get_spans(correlation_id)
    if not spans:
        raise HTTPException(status_code=404, detail="no trace found for that correlation id")
    return PlainTextResponse(content=generate_regression_case(spans), media_type="text/yaml")

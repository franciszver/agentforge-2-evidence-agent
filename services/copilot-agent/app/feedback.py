"""``POST /feedback`` endpoint: thumbs up/down + optional comment on a chat
response (P4.3).

Linkage: the request carries the ``correlation_id`` of the ``/chat`` response
being rated -- the same P4.1 id the client already has from that response's
``X-Correlation-ID`` header (or the SSE ``conversation``/``verification``
frames). Persisting via ``TraceStore.record_feedback_span`` with that SAME
correlation id is the whole linkage mechanism: the resulting feedback span
shares a ``correlation_id`` with the response's request/verification spans,
so ``get_spans(correlation_id)`` (and the P4.5 dashboard / P4.9 review queue
built on it) can join them. No separate foreign key or lookup is needed.

Persistence posture -- HARD FAIL, not best-effort. This is the one place in
the trace-store seam that deliberately differs from
``app.trace_store.record_span_best_effort`` (used by ``app.chat``'s
tool/llm/request/verification spans and ``app.supervisor``'s worker spans):
request/verification spans are passive
telemetry the clinician never asked for, so a write failure there is logged
and swallowed rather than breaking the chat response. Feedback is the
opposite -- a clinician deliberately clicked thumbs up/down (P4.4's UI). If
the write fails and the endpoint reports success anyway, their signal is
silently lost with no way to know it needs retrying. So a
``record_feedback_span`` failure here is surfaced as a 500 (generic detail,
no exception message) so the P4.4 UI can retry or show an error -- it is
never swallowed.

Auth: gated by the SAME bearer-token seam as ``POST /chat``
(``app.chat.get_token_validator`` / ``TokenValidator``), for consistency and
to prevent anonymous feedback spam. Reuses the seam rather than
reimplementing a second one.

Ownership gap (MEDIUM severity -- re-derived by #176, was previously
mis-scored LOW; still deferred with real auth): this endpoint does not
verify the authenticated caller originated ``correlation_id`` -- any valid
bearer can attach feedback, including a free-text comment, to any id it can
guess/observe. The comment field is NOT a no-PHI signal: it is
clinician-typed free text that may incidentally contain PHI (see
``app.trace_store`` module docstring, corrected by #176), and it IS read --
by a human triaging a thumbs-down directly against the trace store, since
no page renders it (#176 redacted it out of ``app.review_page``'s
``/review``, the only UI that ever showed it). So the actual blast radius
is: any valid bearer holding an unguessable correlation id can attach
attacker-controlled free text -- forged as if it came from that trace's
real clinician -- to someone else's trace, and have it read as genuine
during triage. That is worse than "spam of a no-PHI signal with no read
path," the premise this was previously scored against. Binding feedback to
the originating identity still belongs with real token introspection
(``copilot_per_user_token_enabled`` on -- see ``app.chat.get_token_validator``),
where "the authenticated caller" first becomes a meaningful principal to check
ownership against; with that flag off (#168, VULN-0001: the fail-closed
default rejects every token, or -- dev-only -- the
``copilot_dev_accept_any_bearer_token`` stub accepts any) neither validator
carries per-user identity, so such a check would still be theatre today.
Given the corrected severity, this gap should not simply wait for the flag
flip without being tracked on its own -- see issue #180, filed for this
re-derived severity.

PHI note: the comment is user-authored text about the response, not a
patient RECORD value pulled from a tool -- but it may incidentally contain
PHI a clinician typed inline (see ``app.trace_store`` module docstring,
which issue #176 corrected after finding it falsely claimed this field was
non-PHI). It is bounded to ``MAX_COMMENT_LENGTH`` so a request can't persist
an unbounded blob, and it is never rendered on any page (#176).
"""

from __future__ import annotations

import logging
import time

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.chat import (
    TokenValidationError,
    TokenValidator,
    extract_bearer_token,
    get_token_validator,
    get_trace_store,
)
from app.trace_store import FeedbackThumb, TraceStore

_logger = logging.getLogger(__name__)

MAX_COMMENT_LENGTH = 2000


class FeedbackRequest(BaseModel):
    """``POST /feedback`` request body."""

    correlation_id: str = Field(min_length=1)
    thumb: FeedbackThumb
    comment: str | None = Field(default=None, max_length=MAX_COMMENT_LENGTH)


class FeedbackResponse(BaseModel):
    """``POST /feedback`` response body: confirms what was recorded and its linkage."""

    correlation_id: str
    thumb: FeedbackThumb


def feedback_endpoint(
    request: FeedbackRequest,
    authorization: str | None = Header(default=None),
    validator: TokenValidator = Depends(get_token_validator),
    trace_store: TraceStore = Depends(get_trace_store),
) -> FeedbackResponse:
    # Plain `def`, not `async def`: record_feedback_span does blocking
    # sqlite3 I/O. FastAPI runs a sync path-operation function in its
    # worker-thread pool automatically, so the write never blocks the event
    # loop -- same reasoning as app.chat.get_planner_factory's sync dispatch.
    try:
        token = extract_bearer_token(authorization)
        validator(token)
    except TokenValidationError as exc:
        raise HTTPException(status_code=401, detail="invalid or missing token") from exc

    start_ts = time.time()
    try:
        trace_store.record_feedback_span(
            correlation_id=request.correlation_id,
            start_ts=start_ts,
            end_ts=time.time(),
            feedback_thumb=request.thumb,
            feedback_comment=request.comment,
        )
    except Exception as exc:
        # Hard fail (see module docstring): never expose exc's message
        # (may carry a path, e.g. PermissionError on /data) to the caller.
        # "target_correlation_id", not "correlation_id" -- the latter is
        # already stamped on every LogRecord by app.correlation's factory
        # (this /feedback request's OWN id) and setting it again collides;
        # this is the id of the /chat response being rated, a distinct value.
        _logger.error(
            "feedback span write failed",
            extra={"target_correlation_id": request.correlation_id},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="failed to record feedback") from exc

    return FeedbackResponse(correlation_id=request.correlation_id, thumb=request.thumb)

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

Ownership control (#180, extended by #185 -- closes the MEDIUM-severity gap
#176 re-derived; was previously mis-scored LOW and deferred to the flag-on
token-introspection path). Before persisting, the endpoint verifies the
caller matches the principal that originated ``target_correlation_id``'s
trace, via ``TraceStore.caller_owns_trace`` -- see that method's docstring
for the full mixed-regime matrix and ``app.trace_store.hash_owner_token``'s /
``app.openemr_auth.IntrospectionResult.sub``'s docstrings for the two
regimes' full reasoning. In short, TWO regimes, chosen per-row (never mixed
on one row, see ``OwnerKind``):

  * ``copilot_per_user_token_enabled`` ON: ownership binds to OpenEMR's
    signature-verified introspection ``sub`` claim (resolved via
    ``app.chat.get_subject_resolver``, the SAME token-to-subject mapping
    ``/chat`` uses to record it) -- a real per-user principal that survives
    token reissue and a service restart, unlike a token hash (see
    ``hash_owner_token``'s docstring on the restart failure mode it
    inherits from an unset ``trace_args_hash_secret``).
  * Flag OFF (the dev-bridge path): no per-user principal is obtainable even
    in principle -- the dev token's ``username`` claim is HMAC'd with a
    per-session CSRF-derived signing key (``AgentTokenBroker``/
    ``DevAgentToken`` on the PHP side) that this agent never verifies (see
    ``app.chat._user_identity_from_token``'s docstring), so that claim
    cannot be trusted as a principal regardless of what this service
    parses. Ownership instead binds to the raw bearer token itself (hashed,
    never stored raw) -- the panel caches exactly one token per browser
    session and reuses it for both ``/chat`` and ``/feedback`` (see
    ``app.trace_store.hash_owner_token``'s docstring), so "presented the
    same token that started this trace" is a sound proxy for ownership in
    this regime.

A row written under one regime is NEVER claimable under the other -- a flag
flip after the fact does not retroactively upgrade or downgrade an existing
row's ownership, it only changes which regime NEW rows are written under
(see ``caller_owns_trace``'s docstring for the full matrix). A correlation
id with no recorded owner (no ``REQUEST`` span, or one written before #180)
is rejected, not treated as unclaimed -- see ``caller_owns_trace``'s
fail-closed cases. A caller holding a foreign but otherwise-valid token can
still discover a real ``correlation_id`` via the unauthenticated ``GET
/review`` page (#176) -- that page redacts the comment itself but still
lists correlation ids, so ids must be assumed enumerable -- this is exactly
the attempted-forgery case this check rejects.

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
    SubjectResolver,
    TokenValidationError,
    TokenValidator,
    extract_bearer_token,
    get_subject_resolver,
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
    subject_resolver: SubjectResolver = Depends(get_subject_resolver),
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

    # #185: resolve the caller's OpenEMR subject (flag ON, a cache hit off
    # the SAME introspection ``validator`` just performed above) or None
    # (flag OFF). Resolved once and reused for both the ownership check and
    # the FEEDBACK span's own attribution, below.
    subject = subject_resolver(token)

    # #180/#185 ownership check (see module docstring's "Ownership control"
    # section): reject unless this caller matches the principal (subject or
    # token, depending on the ORIGINATING row's own regime) that originated
    # target_correlation_id's trace. Checked after token validation (a 401
    # for a garbage/expired token takes priority) and before the write -- a
    # forged comment must never reach the store.
    if not trace_store.caller_owns_trace(request.correlation_id, token, subject=subject):
        _logger.warning(
            "feedback rejected: caller does not own target_correlation_id",
            extra={"target_correlation_id": request.correlation_id},
        )
        raise HTTPException(status_code=403, detail="caller does not own this correlation id")

    start_ts = time.time()
    try:
        trace_store.record_feedback_span(
            correlation_id=request.correlation_id,
            start_ts=start_ts,
            end_ts=time.time(),
            feedback_thumb=request.thumb,
            feedback_comment=request.comment,
            owner_token=token,
            owner_subject=subject,
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

"""ASGI middleware: reject an oversized request body before it is buffered
or JSON-parsed (issue #173, VULN-0004 follow-up to #167).

``app.chat.ChatRequest.message``'s ``max_length`` (#167) and
``app.feedback.FeedbackRequest.comment``'s ``MAX_COMMENT_LENGTH`` are both
pydantic field bounds -- they fire only AFTER Starlette has already read the
*entire* request body off the wire, buffered it, and handed it to
``json.loads``. A caller that can reach this service's HTTP surface directly
(this app has no production compose overlay; the only live instance is the
dev/demo stack's ``agent`` service, reachable only from the internal,
non-internet-routable ``copilot_internal`` Docker network -- i.e. a caller
that already has a foothold on ``openemr``, ``ollama``, or ``llama-server``,
the same precondition #167 accepted as real and worth fixing) could force
this process to buffer an arbitrarily large body before rejection ever
happens, regardless of how tight the *parsed-field* bound is.

The sanctioned path (browser -> OpenEMR -> this service) is already bounded
well below this middleware's cap by PHP's ``post_max_size = 30M``
(``docker/flex/configs/php8.4/php.ini``, same across PHP 8.2-8.5) and the
forwarded body being built server-side from an already-≤4000-char message --
this middleware exists for the *unsanctioned* path.

Deliberately a BARE ASGI middleware, not ``starlette.middleware.base.
BaseHTTPMiddleware`` -- matching ``app.correlation.CorrelationIdMiddleware``'s
shape for the same reason documented there: ``BaseHTTPMiddleware``
historically hands ``call_next`` off to a separate task, which would disturb
the correlation-id contextvar timing the SSE generator in ``app.chat``
depends on. This middleware sits OUTSIDE (registered after, hence outermost
-- see ``app.main.create_app``, which relies on Starlette applying
``add_middleware`` in reverse-registration order) ``CorrelationIdMiddleware``
specifically so that a request rejected for size never gets a correlation id
minted or logged in the first place.

Checks BOTH:

1. The pre-parse ``Content-Length`` header, when present and parseable --
   rejects before ``receive()`` is ever called at all, so an honestly
   oversized declared body is never even asked for.
2. A running count of the ACTUAL bytes delivered through ``receive()`` --
   catches a caller that lies about (understates or omits) ``Content-Length``
   and then streams more body than declared.

Wraps ONLY ``receive``, never ``send`` -- so this middleware cannot by
construction interfere with a streamed SSE response body
(``app.chat._stream_chat``'s ``StreamingResponse``), which is exactly the
lifetime ``CorrelationIdMiddleware``'s ``send`` wrapper (header injection)
already has to be careful around.

**Traceability tradeoff, deliberate:** because this middleware sits OUTSIDE
``CorrelationIdMiddleware``, a request rejected here gets NO correlation id
at all -- it never reaches the code that would mint one. For genuinely
adversarial traffic (the threat model this exists for) that's the right
call: no reason to spend a mint/log cycle on a request that's being thrown
away anyway. The cost lands on a legitimate but misconfigured caller
instead -- its 413 shows up in ops logs with no correlation id to trace it
by. Accepted; not something this middleware attempts to fix.

The 413 body is built via a real ``starlette.responses.JSONResponse`` --
itself an ASGI callable -- rather than hand-rolled ``send()`` messages, so
it gets correct ``Content-Length`` framing (and any other response
bookkeeping Starlette does) for free. This is the only place this
middleware touches ``send`` at all; it still never *wraps* ``send`` the way
``CorrelationIdMiddleware`` does.

**Why the streaming-counter signal is a real ``starlette.exceptions.
HTTPException``, not a private exception type (found via a full-stack test,
not by inspection):** ``fastapi.routing``'s own request-body-reading code
wraps its ``await request.json()`` call in a broad
``except Exception as e: raise HTTPException(400, "There was an error
parsing the body") from e`` -- with one carve-out immediately above it,
``except HTTPException: raise`` (its own comment: "If a middleware raises
an HTTPException, it should be raised again"). A private exception type
raised from the wrapped ``receive`` gets swallowed by that broad
``except Exception`` and re-surfaces as a generic, wrong-status 400 --
verified by constructing a real oversized streamed request (no
``content-length`` header) against the actual ``/chat`` route and observing
exactly that 400 before this was fixed. Using Starlette's own
``HTTPException(status_code=413, ...)`` instead hits FastAPI's
``except HTTPException: raise`` carve-out, so it survives unmangled and is
then handled correctly by Starlette's ``ExceptionMiddleware`` (which sits
between this middleware and the router) using its own default handler --
which happens to produce the exact same ``{"detail": ...}`` JSON shape
``_reject`` below produces directly. The ``try/except`` in ``__call__``
below is a deliberate belt-and-braces fallback for anything this
middleware wraps that ISN'T a full FastAPI app with that machinery in
front of it (e.g. the toy inner-app fixtures in this module's own unit
tests) -- in the real app, ``ExceptionMiddleware`` handles it first and
this ``except`` never fires at all.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from starlette.types import Message, Receive, Scope, Send

_BODY_TOO_LARGE_DETAIL = "Request body too large"


class BodySizeLimitMiddleware:
    """Reject a request whose body exceeds ``max_bytes``, checked both via
    ``Content-Length`` (pre-parse) and a running received-bytes counter
    (streaming-safe against a lying/absent header). See module docstring."""

    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        *,
        max_bytes: int,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError:
                declared_bytes = None  # malformed header -- fall through to the streaming counter
            if declared_bytes is not None and declared_bytes > self.max_bytes:
                await self._reject(scope, receive, send)
                return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # A real ``starlette.exceptions.HTTPException``, not a
                    # private type -- see module docstring for why: FastAPI's
                    # own body-parsing code re-raises HTTPException unchanged
                    # but converts anything else into a generic 400.
                    raise StarletteHTTPException(status_code=413, detail=_BODY_TOO_LARGE_DETAIL)
            return message

        try:
            await self.app(scope, counting_receive, send)
        except StarletteHTTPException as exc:
            # Only ever our own signal reaches here in practice (see module
            # docstring: a real FastAPI app's ExceptionMiddleware handles
            # this exception -- and any other HTTPException raised deeper in
            # the app -- before it would ever propagate this far out). The
            # status-code check is defense in depth so this middleware can
            # never mistake an unrelated HTTPException for its own signal.
            if exc.status_code != 413 or exc.detail != _BODY_TOO_LARGE_DETAIL:
                raise
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(status_code=413, content={"detail": _BODY_TOO_LARGE_DETAIL})
        await response(scope, receive, send)

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
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.datastructures import Headers
from starlette.types import Message, Receive, Scope, Send


class _BodyTooLarge(Exception):
    """Internal signal only -- raised by the wrapped ``receive`` once the
    running byte count crosses the cap, caught by this same middleware's
    ``__call__`` before it can propagate anywhere else. Never a public
    exception type and never allowed to reach ``ExceptionMiddleware`` or
    ``ServerErrorMiddleware``: both of those sit further from ``send`` than
    this middleware is registered to (see module docstring), so letting it
    escape would either produce a generic 500 (ServerErrorMiddleware) or --
    worse -- go unhandled, since this custom type has no registered
    exception handler for ExceptionMiddleware to match on either."""


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
                await self._reject(send)
                return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, counting_receive, send)
        except _BodyTooLarge:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"detail":"Request body too large"}',
            }
        )

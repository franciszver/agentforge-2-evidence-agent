"""Correlation-id propagation seam (P4.1).

One correlation id is minted per chat invocation (``POST /chat``) and made
readable from anywhere in that invocation's call stack -- log lines, tool
dispatch, LLM calls, verification -- via a single ``contextvars.ContextVar``,
without threading the id through every function signature. This module is
the whole mechanism: mint/bind (``correlation_scope``), read
(``get_correlation_id``), an ASGI middleware that binds it per HTTP request
(``CorrelationIdMiddleware``), and a minimal structured-logging seam
(``configure_logging``) that stamps every ``LogRecord`` with it.

**Reconciling with P2.17.** ``app.chat.Turn`` already carried a per-turn
``correlation_id`` (a local ``uuid.uuid4()`` minted inside ``_stream_chat``).
That is now THIS id -- ``_stream_chat`` reads ``get_correlation_id()`` instead
of minting its own, so there is exactly one id per invocation, not two.

**The SSE-generator lifetime -- the subtle correctness point.** ``POST /chat``
returns a ``StreamingResponse`` wrapping a plain *sync* generator
(``app.chat._stream_chat``). Starlette drives a sync generator's body through
a worker-thread pool: ``starlette.concurrency.iterate_in_threadpool`` calls
``anyio.to_thread.run_sync(next, iterator)`` once PER YIELD, and every one of
those calls takes an independent ``contextvars.copy_context()`` snapshot of
whatever context is live in the calling coroutine/task AT THAT MOMENT.

Two consequences, verified empirically (see ``tests/test_correlation.py``):

  1. A value bound in the coroutine/task that drives the whole request --
     i.e. before ``StreamingResponse`` starts being iterated, such as this
     middleware's ``__call__`` (which stays in ONE coroutine across dependency
     resolution, endpoint invocation, AND response streaming -- no
     ``BaseHTTPMiddleware``-style task hand-off) -- IS visible to every
     later per-yield thread-pool call, because each of those calls copies
     its context fresh from that same live coroutine, which still has the
     value set.
  2. A ``ContextVar.set()`` performed *inside* the generator body, between
     yields, is NOT visible on the next yield. It mutates only that one
     threaded call's private context copy; the copy is discarded when the
     call returns, and the next yield's threadpool call copies fresh from
     the outer coroutine (which was never touched by that mutation).

So the id MUST be bound before the streamed generator starts (this
middleware, or the async endpoint before constructing the
``StreamingResponse``) -- "refreshing" it from inside the sync generator
itself does not work and would silently desync after the first yield.
``CorrelationIdMiddleware`` is therefore a bare ASGI middleware (not
``starlette.middleware.base.BaseHTTPMiddleware``, which historically
decouples ``call_next`` onto a separate task): it wraps
``self.app(scope, receive, send)`` directly in its own coroutine, which is
exactly the coroutine that later drives the streamed body.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import Message, Receive, Scope, Send

CORRELATION_ID_HEADER = "X-Correlation-ID"

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def new_correlation_id() -> str:
    """Mint a fresh correlation id."""
    return str(uuid.uuid4())


def get_correlation_id() -> str:
    """Return the correlation id bound in the current context.

    Lazily mints and binds one if none is set yet, so code invoked outside
    an HTTP request (a direct unit test, a script) still gets a stable,
    non-empty id for the rest of its context rather than an empty string.
    """
    current = _correlation_id.get()
    if current is None:
        current = new_correlation_id()
        _correlation_id.set(current)
    return current


@contextmanager
def correlation_scope(correlation_id: str | None = None) -> Iterator[str]:
    """Bind ``correlation_id`` (minting one if omitted) for the ``with`` block,
    restoring the previous value on exit."""
    value = correlation_id or new_correlation_id()
    token = _correlation_id.set(value)
    try:
        yield value
    finally:
        _correlation_id.reset(token)


class CorrelationIdMiddleware:
    """ASGI middleware: bind one correlation id per HTTP request.

    Reads an inbound ``X-Correlation-ID`` header if present, otherwise mints
    one; binds it to the request's contextvar for the request's ENTIRE
    lifetime (including a streamed SSE response body -- see the module
    docstring); and echoes it back on the response's ``X-Correlation-ID``
    header.
    """

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        inbound = headers.get(CORRELATION_ID_HEADER)
        correlation_id = inbound if inbound else new_correlation_id()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers[CORRELATION_ID_HEADER] = correlation_id
            await send(message)

        with correlation_scope(correlation_id):
            await self.app(scope, receive, send_wrapper)


_logging_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Install the minimal structured-logging seam (idempotent).

    Wraps the current ``logging.LogRecordFactory`` so every ``LogRecord`` --
    from any logger, anywhere in the process -- carries a ``correlation_id``
    attribute read from this module's contextvar at record-creation time.
    This is what makes "appears on every log line" true without an adapter
    or filter per logger: any module can just ``logging.getLogger(__name__)``
    and log normally.

    A record created with no scope bound gets ``"-"`` (never mints a fresh
    id as a side effect of merely logging).
    """
    global _logging_configured
    if _logging_configured:
        return

    base_factory = logging.getLogRecordFactory()

    def _record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = base_factory(*args, **kwargs)
        record.correlation_id = _correlation_id.get() or "-"
        return record

    logging.setLogRecordFactory(_record_factory)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [correlation_id=%(correlation_id)s] %(name)s: %(message)s"
        )
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)

    _logging_configured = True

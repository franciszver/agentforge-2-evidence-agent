"""Hermetic tests for the request-body-size-limit middleware (issue #173).

``ChatRequest.message``'s ``max_length`` (#167) is a pydantic bound -- it
fires only AFTER the entire request body has been read off the wire,
buffered, and JSON-parsed. A caller with a foothold on ``copilot_internal``
posting directly to this service's unpublished port 8000 (the same
precondition #167 accepted as real) could otherwise force the process to
buffer an arbitrarily large body before rejection ever happens.

Two tests, both hermetic (no live stack):

1. ``test_oversized_content_length_rejected_before_endpoint_reached`` --
   proves the pre-parse ``Content-Length`` check: a declared size over the
   cap is rejected with 413 before the request ever reaches routing/
   dependency resolution, let alone the endpoint body. Proven two ways: a
   ``receive`` probe that must never be called, and dependency overrides
   that raise if FastAPI ever tries to resolve them.

2. ``test_streaming_body_over_cap_without_content_length_is_rejected`` --
   proves the actual-bytes-received counter, which is what stops a caller
   that lies about (or omits) ``Content-Length`` entirely. httpx computes
   ``Content-Length`` itself for any ``bytes`` body, so this can't be
   exercised through the normal test client stack -- it constructs the
   ASGI ``scope``/``receive``/``send`` triple by hand (mirroring
   ``tests/test_correlation.py``'s direct-ASGI-call pattern) and feeds the
   middleware multiple ``http.request`` chunks (``more_body=True``) whose
   cumulative size crosses the cap, with no ``content-length`` header at
   all.

Each test also pairs its rejection assertion with a presence assertion
(a body under the cap succeeds) so the test could actually fail if the
middleware were wired backwards or unconditionally rejecting/accepting.

A THIRD test, ``test_streaming_body_over_cap_through_real_chat_route_via_asgi_transport``,
closes a gap the two above leave open: everything above either drives the
middleware directly (no FastAPI request-body-parsing machinery in front of
it) or only exercises the ``Content-Length`` pre-check against the real
app. The streaming-counter path is the genuinely load-bearing one (the
header check is the fast path; the counter is what actually stops a lying
caller), and it had never been proven against the real ``/chat`` route with
real FastAPI body parsing in front of it -- which matters because FastAPI's
own ``fastapi.routing`` body-parsing code wraps ``request.json()`` in a
broad ``except Exception`` that converts anything except a
``starlette.exceptions.HTTPException`` into a generic, wrong-status 400
(see ``app/body_size_limit.py``'s module docstring for the real bug this
full-stack test caught and how it was fixed). This one uses
``httpx.ASGITransport`` directly (bypassing ``starlette.testclient.
TestClient``, which does not support a streamed request body) with an
async-generator body -- httpx sends a generator body chunked, with no
``Content-Length`` header at all, which is exactly the case the streaming
counter (not the header pre-check) has to catch.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from app.body_size_limit import BodySizeLimitMiddleware


# ---------------------------------------------------------------------------
# Unit-level: the middleware in isolation (no FastAPI app), proving both the
# pre-parse header check and the streaming byte counter directly.
# ---------------------------------------------------------------------------


def _make_scope(headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": headers,
        "query_string": b"",
        "client": ("test", 1234),
        "server": ("test", 80),
        "scheme": "http",
    }


def test_middleware_passes_through_non_http_scopes_untouched():
    calls: list[tuple] = []

    async def inner_app(scope, receive, send) -> None:
        calls.append((scope, receive, send))

    middleware = BodySizeLimitMiddleware(inner_app, max_bytes=100)
    scope = {"type": "lifespan"}
    receive = object()
    send = object()

    asyncio.run(middleware(scope, receive, send))

    assert calls == [(scope, receive, send)]  # untouched -- no interception at all


def test_content_length_under_cap_reaches_the_inner_app():
    """Presence pairing for the header pre-check: a declared size UNDER the
    cap must reach the inner app untouched -- proves the check isn't just
    unconditionally rejecting everything."""
    inner_called = False

    async def inner_app(scope, receive, send) -> None:
        nonlocal inner_called
        inner_called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = BodySizeLimitMiddleware(inner_app, max_bytes=100)
    scope = _make_scope([(b"content-length", b"10")])

    async def receive():
        return {"type": "http.request", "body": b"0123456789", "more_body": False}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert inner_called
    assert sent[0]["status"] == 200


def test_content_length_over_cap_rejected_without_ever_calling_receive():
    """The header pre-check must reject BEFORE reading any body bytes --
    ``receive`` must never be invoked."""
    inner_called = False

    async def inner_app(scope, receive, send) -> None:
        nonlocal inner_called
        inner_called = True

    middleware = BodySizeLimitMiddleware(inner_app, max_bytes=100)
    scope = _make_scope([(b"content-length", b"101")])

    receive_called = False

    async def receive():
        nonlocal receive_called
        receive_called = True
        raise AssertionError("receive() must never be called when Content-Length already exceeds the cap")

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert not inner_called
    assert not receive_called
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_streaming_body_over_cap_without_content_length_is_rejected():
    """The genuinely load-bearing test: no ``content-length`` header at all
    (a lying/absent header), body delivered as several ``more_body=True``
    chunks whose CUMULATIVE size crosses the cap. Only the running byte
    counter -- not the header check -- can catch this."""
    inner_reached_send = False

    async def inner_app(scope, receive, send) -> None:
        # A real app would keep calling receive() until more_body is False,
        # then act on the body. If it gets to call a fourth receive() at
        # all past the cap, something is wrong -- but we primarily care
        # that the middleware itself raises/aborts before this app ever
        # gets to send a real response.
        nonlocal inner_reached_send
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        inner_reached_send = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"unreachable"})

    middleware = BodySizeLimitMiddleware(inner_app, max_bytes=20)
    scope = _make_scope([])  # deliberately NO content-length header

    chunks = [b"0" * 8, b"1" * 8, b"2" * 8]  # cumulative 24 bytes > 20-byte cap
    chunk_iter = iter(chunks)

    async def receive():
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunk, "more_body": True}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert not inner_reached_send
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413
    # Presence pairing: the same middleware, same chunk shape, but a cap
    # wide enough to fit the cumulative total lets the request through.
    inner_reached_send_ok = False

    async def inner_app_ok(scope, receive, send) -> None:
        nonlocal inner_reached_send_ok
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        inner_reached_send_ok = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware_ok = BodySizeLimitMiddleware(inner_app_ok, max_bytes=1000)
    chunk_iter_ok = iter(chunks)

    async def receive_ok():
        try:
            chunk = next(chunk_iter_ok)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunk, "more_body": True}

    sent_ok: list[dict] = []

    async def send_ok(message):
        sent_ok.append(message)

    asyncio.run(middleware_ok(scope, receive_ok, send_ok))

    assert inner_reached_send_ok
    assert sent_ok[0]["status"] == 200


# ---------------------------------------------------------------------------
# App-level: the real /chat route, wired through app.main.create_app(), with
# the real (config-sourced) cap -- proves this is actually registered and
# outermost, not just correct in isolation.
# ---------------------------------------------------------------------------


def test_oversized_content_length_rejected_before_endpoint_reached():
    from app.chat import get_planner_factory, get_token_validator
    from app.config import DEFAULT_MAX_REQUEST_BODY_BYTES
    from app.main import app as real_app

    def _never_call_validator(token: str) -> None:
        raise AssertionError("token validator must never run for an oversized body")

    def _never_call_planner_factory(patient_id: int):
        raise AssertionError("planner factory must never run for an oversized body")

    real_app.dependency_overrides[get_token_validator] = lambda: _never_call_validator
    real_app.dependency_overrides[get_planner_factory] = lambda: _never_call_planner_factory
    try:
        from starlette.testclient import TestClient

        client = TestClient(real_app)
        oversized = DEFAULT_MAX_REQUEST_BODY_BYTES + 1
        request = client.build_request(
            "POST",
            "/chat",
            headers={
                "content-length": str(oversized),
                "authorization": "Bearer irrelevant-token",
            },
            json={"message": "hi", "patient_id": 1},
        )
        response = client.send(request)
    finally:
        real_app.dependency_overrides.clear()

    assert response.status_code == 413

    # Presence pairing: a normal, well-under-cap /chat request is untouched
    # by this middleware (still reaches auth and gets the normal 401 for a
    # bad token -- proving the 413 above is size-specific, not a blanket
    # rejection of every /chat request).
    client2 = TestClient(real_app)
    normal_response = client2.post(
        "/chat",
        json={"message": "hi", "patient_id": 1},
        headers={"Authorization": "Bearer bad-token"},
    )
    assert normal_response.status_code != 413


def test_body_size_limit_middleware_is_outermost():
    """Ordering proof: BodySizeLimitMiddleware must run BEFORE
    CorrelationIdMiddleware, so a rejected-for-size request never gets a
    correlation id minted/logged. Inspect Starlette's resolved middleware
    stack order directly rather than relying on behavioural side effects."""
    from app.body_size_limit import BodySizeLimitMiddleware
    from app.correlation import CorrelationIdMiddleware
    from app.main import app as real_app

    classes = [m.cls for m in real_app.user_middleware]
    assert classes.index(BodySizeLimitMiddleware) < classes.index(CorrelationIdMiddleware)


def test_streaming_body_over_cap_through_real_chat_route_via_asgi_transport():
    """Full-stack proof of the streaming counter (not just the header
    pre-check) against the REAL ``/chat`` route, with real FastAPI request-
    body parsing in front of the middleware -- see this module's docstring
    for why the two tests above don't already cover this, and
    ``app/body_size_limit.py``'s module docstring for the real 400-instead-
    of-413 bug this test caught the first time it was written."""
    from app.chat import get_planner_factory, get_token_validator
    from app.config import DEFAULT_MAX_REQUEST_BODY_BYTES
    from app.main import app as real_app

    validator_called = False
    planner_called = False

    def _tracking_validator(token: str) -> None:
        nonlocal validator_called
        validator_called = True

    def _tracking_planner_factory(patient_id: int):
        nonlocal planner_called
        planner_called = True
        raise AssertionError("planner factory must never run for an oversized streamed body")

    async def oversized_chunks() -> AsyncIterator[bytes]:
        # Two chunks whose sum crosses the cap; well-formed JSON isn't
        # required to prove the point -- the counter must reject before
        # JSON parsing (or auth) is ever attempted.
        half = DEFAULT_MAX_REQUEST_BODY_BYTES // 2 + 1000
        yield b"0" * half
        yield b"0" * half

    real_app.dependency_overrides[get_token_validator] = lambda: _tracking_validator
    real_app.dependency_overrides[get_planner_factory] = lambda: _tracking_planner_factory
    try:

        async def _send_oversized() -> httpx.Response:
            transport = httpx.ASGITransport(app=real_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                request = client.build_request(
                    "POST",
                    "/chat",
                    content=oversized_chunks(),
                    headers={
                        "authorization": "Bearer irrelevant-token",
                        "content-type": "application/json",
                    },
                )
                # httpx sends a generator body chunked, with NO
                # content-length header -- exactly the case the streaming
                # counter (not the header pre-check) has to catch.
                assert "content-length" not in request.headers
                return await client.send(request)

        response = asyncio.run(_send_oversized())
    finally:
        real_app.dependency_overrides.clear()

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    assert not validator_called
    assert not planner_called

    # Presence pairing: the identical streamed-chunk mechanism, but under
    # the cap, reaches auth (a bad token still gets a normal 401, proving
    # the 413 above is size-specific, not the streamed-transport mechanism
    # itself being rejected).
    async def under_cap_chunks() -> AsyncIterator[bytes]:
        import json as _json

        yield _json.dumps({"message": "hi", "patient_id": 1}).encode()

    async def _send_under_cap() -> httpx.Response:
        transport = httpx.ASGITransport(app=real_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request = client.build_request(
                "POST",
                "/chat",
                content=under_cap_chunks(),
                headers={
                    "authorization": "Bearer bad-token",
                    "content-type": "application/json",
                },
            )
            assert "content-length" not in request.headers
            return await client.send(request)

    under_cap_response = asyncio.run(_send_under_cap())
    assert under_cap_response.status_code != 413

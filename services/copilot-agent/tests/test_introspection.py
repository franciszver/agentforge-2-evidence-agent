"""Hermetic tests for #124 Phase 4 token introspection.

All HTTP is served by ``httpx.MockTransport`` -- the suite never touches the
network. Covers the low-level ``introspect_token`` primitive (active / inactive
/ malformed / non-2xx / network-error, body client_secret_post auth, token-not-in-URL,
no-token/secret logging), the short-TTL ``TokenIntrospector`` cache (hash-keyed,
network only re-hit when uncached), and the introspection-based validator's
active/inactive/expired/empty mapping to ``TokenValidationError``.

Fake token/secret VALUES are kept tiny + keyword-free so the pre-push
secret-scan (keyword followed by an 8+ char quoted literal) never trips on
this file.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from app.chat import TokenValidationError, build_introspection_validator
from app.introspection import TokenIntrospector
from app.openemr_auth import IntrospectionResult, introspect_token

# Tiny, keyword-free fake values (each quoted literal < 8 chars).
_CID = "cid-1"
_CSEC = "cs-1"
_TOK = "utok-a"


# --- introspect_token primitive -------------------------------------------


def test_introspect_active_returns_active_and_uses_body_client_auth():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["url"] = str(request.url)
        captured["has_auth_header"] = "Authorization" in request.headers
        captured["body"] = request.content.decode()
        return httpx.Response(200, json={"active": True, "exp": 9999999999})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = introspect_token(
            client,
            base_url="https://openemr",
            introspect_path="/oauth2/default/introspect",
            client_id=_CID,
            client_secret=_CSEC,
            token=_TOK,
        )

    assert result == IntrospectionResult(active=True, exp=9999999999)
    assert captured["path"] == "/oauth2/default/introspect"
    # OpenEMR's introspection endpoint reads the client credentials from the
    # POST body (client_secret_post), NOT the Authorization header -- Basic-auth
    # creds are ignored there, yielding a spurious active:false. So the creds
    # must travel in the form body alongside the token.
    body = captured["body"]
    assert isinstance(body, str)
    assert f"client_id={_CID}" in body
    assert f"client_secret={_CSEC}" in body
    assert f"token={_TOK}" in body
    assert captured["has_auth_header"] is False
    # Nothing sensitive travels in the URL/query.
    assert _TOK not in captured["url"]
    assert _CSEC not in captured["url"]
    assert _CID not in captured["url"]


def test_introspect_inactive_returns_invalid():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"active": False})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = introspect_token(
            client,
            base_url="https://openemr",
            introspect_path="/oauth2/default/introspect",
            client_id=_CID,
            client_secret=_CSEC,
            token=_TOK,
        )

    assert result == IntrospectionResult(active=False, exp=None)


def test_introspect_malformed_json_returns_invalid():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = introspect_token(
            client,
            base_url="https://openemr",
            introspect_path="/oauth2/default/introspect",
            client_id=_CID,
            client_secret=_CSEC,
            token=_TOK,
        )

    assert not result.active


def test_introspect_non_2xx_returns_invalid():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"active": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = introspect_token(
            client,
            base_url="https://openemr",
            introspect_path="/oauth2/default/introspect",
            client_id=_CID,
            client_secret=_CSEC,
            token=_TOK,
        )

    assert not result.active


def test_introspect_network_error_returns_invalid():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = introspect_token(
            client,
            base_url="https://openemr",
            introspect_path="/oauth2/default/introspect",
            client_id=_CID,
            client_secret=_CSEC,
            token=_TOK,
        )

    assert not result.active


def test_introspect_never_logs_token_or_secret(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"active": True, "exp": 9999999999})

    with caplog.at_level(logging.DEBUG):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            introspect_token(
                client,
                base_url="https://openemr",
                introspect_path="/oauth2/default/introspect",
                client_id=_CID,
                client_secret=_CSEC,
                token=_TOK,
            )

    assert _TOK not in caplog.text
    assert _CSEC not in caplog.text


# --- TokenIntrospector cache ----------------------------------------------


def _introspector(tmp_path, handler, *, calls: list[int], clock=None) -> TokenIntrospector:
    """Build a ``TokenIntrospector`` with tmp creds + a MockTransport client.

    ``calls`` is appended to on every upstream request so tests can assert
    cache hits skip the network.
    """
    creds = tmp_path / "prod-creds.json"
    creds.write_text(f'{{"client_id": "{_CID}", "client_secret": "{_CSEC}"}}', encoding="utf-8")

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return handler(request)

    def client_factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(counting_handler))

    kwargs: dict[str, object] = {}
    if clock is not None:
        kwargs["clock"] = clock
    return TokenIntrospector(
        base_url="https://openemr",
        introspect_path="/oauth2/default/introspect",
        creds_path=str(creds),
        client_factory=client_factory,
        cache_ttl_seconds=60.0,
        **kwargs,
    )


def test_introspector_caches_active_result_by_token_hash(tmp_path):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"active": True, "exp": 9999999999})

    introspector = _introspector(tmp_path, handler, calls=calls)

    first = introspector.introspect(_TOK)
    second = introspector.introspect(_TOK)

    assert first.active and second.active
    # Second call served from cache -- the network was hit exactly once.
    assert len(calls) == 1
    # The cache key is a hash, never the raw token.
    assert _TOK not in introspector._cache
    import hashlib

    assert hashlib.sha256(_TOK.encode()).hexdigest() in introspector._cache


def test_introspector_does_not_cache_inactive_result(tmp_path):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"active": False})

    introspector = _introspector(tmp_path, handler, calls=calls)

    introspector.introspect(_TOK)
    introspector.introspect(_TOK)

    # Inactive/failed results are NOT cached (a transient failure must not
    # lock a user out for the whole TTL) -- both calls hit the network.
    assert len(calls) == 2


def test_introspector_ttl_cap_expires_cache(tmp_path):
    calls: list[int] = []
    now = {"t": 1000.0}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"active": True, "exp": 9999999999})

    introspector = _introspector(tmp_path, handler, calls=calls, clock=lambda: now["t"])

    introspector.introspect(_TOK)
    assert len(calls) == 1
    now["t"] += 61.0  # past the 60s TTL cap
    introspector.introspect(_TOK)
    assert len(calls) == 2


def test_introspector_cache_deadline_capped_at_exp_when_sooner_than_ttl(tmp_path):
    # exp (1030) is SOONER than now + ttl (1000 + 60 = 1060): the cached entry's
    # deadline must be capped at exp, not ttl. Advancing past exp but within the
    # ttl window forces a re-introspection -- proving the cap is exp, not ttl.
    calls: list[int] = []
    now = {"t": 1000.0}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"active": True, "exp": 1030})

    introspector = _introspector(tmp_path, handler, calls=calls, clock=lambda: now["t"])

    introspector.introspect(_TOK)
    assert len(calls) == 1
    now["t"] = 1025.0  # still before exp -> served from cache
    introspector.introspect(_TOK)
    assert len(calls) == 1
    now["t"] = 1031.0  # past exp (1030) but well within ttl (1060)
    introspector.introspect(_TOK)
    assert len(calls) == 2  # deadline was capped at exp -> re-introspected


def test_introspector_missing_creds_returns_invalid_without_logging_token(tmp_path, caplog):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        return httpx.Response(200, json={"active": True})

    introspector = TokenIntrospector(
        base_url="https://openemr",
        introspect_path="/oauth2/default/introspect",
        creds_path=str(tmp_path / "does-not-exist.json"),
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        cache_ttl_seconds=60.0,
    )

    with caplog.at_level(logging.WARNING):
        result = introspector.introspect(_TOK)

    assert not result.active
    assert len(calls) == 0  # never reached the network
    assert _TOK not in caplog.text


# --- introspection-based validator ----------------------------------------


class _FakeIntrospector:
    def __init__(self, result: IntrospectionResult) -> None:
        self._result = result
        self.seen: list[str] = []

    def introspect(self, token: str) -> IntrospectionResult:
        self.seen.append(token)
        return self._result


def test_validator_accepts_active_token():
    validator = build_introspection_validator(
        _FakeIntrospector(IntrospectionResult(active=True, exp=9999999999)),
        clock=lambda: 1000.0,
    )
    validator(_TOK)  # no raise


def test_validator_rejects_inactive_token():
    validator = build_introspection_validator(
        _FakeIntrospector(IntrospectionResult(active=False, exp=None)),
        clock=lambda: 1000.0,
    )
    with pytest.raises(TokenValidationError):
        validator(_TOK)


def test_validator_rejects_expired_exp_even_if_active():
    validator = build_introspection_validator(
        _FakeIntrospector(IntrospectionResult(active=True, exp=500)),
        clock=lambda: 1000.0,  # now is past exp
    )
    with pytest.raises(TokenValidationError):
        validator(_TOK)


def test_validator_rejects_empty_token_without_introspecting():
    fake = _FakeIntrospector(IntrospectionResult(active=True, exp=9999999999))
    validator = build_introspection_validator(fake, clock=lambda: 1000.0)
    with pytest.raises(TokenValidationError):
        validator("")
    assert fake.seen == []  # short-circuited before any network round-trip

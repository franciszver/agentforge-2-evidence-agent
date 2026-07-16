"""Hermetic tests for the DEV-ONLY real-token bridge (issue #126, finding F4).

All HTTP is served by ``httpx.MockTransport`` -- no network is touched. These
tests pin the token-acquisition + caching behavior the agent relies on to make
authenticated tool calls with a REAL OpenEMR token (not the browser's
DevAgentToken), and the log-safety of the failure paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.dev_token_bridge import DevTokenBridge, DevTokenError

_TOKEN_PATH = "/oauth2/default/token"
_SCOPE = "openid api:oemr user/medication.read"
_SECRET = "super-secret-value"
_PASSWORD = "clinician-password"


def _write_creds(tmp_path: Path, *, client_id: str = "cid-123", client_secret: str = _SECRET) -> str:
    creds_file = tmp_path / "openemr-dev-client.json"
    creds_file.write_text(json.dumps({"client_id": client_id, "client_secret": client_secret}), encoding="utf-8")
    return str(creds_file)


def _make_bridge(
    creds_path: str,
    *,
    handler,
    clock=None,
) -> DevTokenBridge:
    def client_factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return DevTokenBridge(
        base_url="https://openemr",
        token_path=_TOKEN_PATH,
        creds_path=creds_path,
        username="clinician",
        password=_PASSWORD,
        scope=_SCOPE,
        client_factory=client_factory,
        **kwargs,
    )


def _token_handler(access_token: str, *, expires_in: int = 3600, calls: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        assert request.url.path == _TOKEN_PATH
        return httpx.Response(
            200,
            json={
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": expires_in,
                "scope": _SCOPE,
            },
        )

    return handler


def test_get_token_returns_a_real_access_token(tmp_path):
    creds_path = _write_creds(tmp_path)
    bridge = _make_bridge(creds_path, handler=_token_handler("real-token-abc"))

    assert bridge.get_token() == "real-token-abc"


def test_password_grant_uses_creds_file_and_configured_credential(tmp_path):
    creds_path = _write_creds(tmp_path, client_id="cid-xyz")
    calls: list[httpx.Request] = []
    bridge = _make_bridge(creds_path, handler=_token_handler("t", calls=calls))

    bridge.get_token()

    (request,) = calls
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form["grant_type"] == "password"
    assert form["client_id"] == "cid-xyz"
    assert form["client_secret"] == _SECRET
    assert form["username"] == "clinician"
    assert form["password"] == _PASSWORD
    assert form["scope"] == _SCOPE


def test_token_is_cached_within_ttl(tmp_path):
    creds_path = _write_creds(tmp_path)
    calls: list[httpx.Request] = []
    now = [1000.0]
    bridge = _make_bridge(creds_path, handler=_token_handler("t", calls=calls), clock=lambda: now[0])

    first = bridge.get_token()
    now[0] += 100.0  # well within a 3600s TTL
    second = bridge.get_token()

    assert first == second
    assert len(calls) == 1, "a token within TTL must be served from cache, not re-fetched"


def test_token_is_refetched_after_expiry(tmp_path):
    creds_path = _write_creds(tmp_path)
    tokens = iter(["first-token", "second-token"])
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={"access_token": next(tokens), "token_type": "Bearer", "expires_in": 3600},
        )

    now = [1000.0]
    bridge = _make_bridge(creds_path, handler=handler, clock=lambda: now[0])

    first = bridge.get_token()
    now[0] += 3600.0  # past TTL (even accounting for the safety margin)
    second = bridge.get_token()

    assert first == "first-token"
    assert second == "second-token"
    assert len(calls) == 2


def test_missing_creds_file_raises_dev_token_error(tmp_path):
    missing = str(tmp_path / "does-not-exist.json")
    bridge = _make_bridge(missing, handler=_token_handler("t"))

    with pytest.raises(DevTokenError):
        bridge.get_token()


def test_malformed_creds_file_raises_dev_token_error(tmp_path):
    creds_file = tmp_path / "openemr-dev-client.json"
    creds_file.write_text("not json", encoding="utf-8")
    bridge = _make_bridge(str(creds_file), handler=_token_handler("t"))

    with pytest.raises(DevTokenError):
        bridge.get_token()


def test_upstream_failure_raises_dev_token_error_without_leaking_secrets(tmp_path):
    creds_path = _write_creds(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    bridge = _make_bridge(creds_path, handler=handler)

    with pytest.raises(DevTokenError) as excinfo:
        bridge.get_token()

    message = str(excinfo.value)
    assert _SECRET not in message
    assert _PASSWORD not in message

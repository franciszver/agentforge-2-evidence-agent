"""P3G.3: contract tests for the agent's HTTP surface against its OpenAPI spec.

Source of truth: FastAPI's own generated schema (``app.main.app.openapi()``),
NOT a hand-authored document -- FastAPI already derives an accurate schema
from the route signatures and Pydantic models, so re-authoring one by hand
would just be a second copy that silently drifts from the real routes.

Two things are enforced here, fully offline (``TestClient``, no live model,
no external API):

1. **Drift guard** -- the pinned artifact at ``openapi/openapi.json`` (the
   contract other tools/consumers are written against) must exactly match
   the schema the running app generates right now. If a route or model
   changes, this test goes red until someone deliberately regenerates the
   pin (``python scripts/generate_openapi_spec.py``) and reviews the diff --
   spec drift can never land silently.
2. **Response conformance** -- real ``TestClient`` responses for a sample of
   endpoints (including both ``/ready`` states) validate against the
   schema the pinned spec declares for their path/method/status, proving the
   spec describes what the app actually returns, not just what it claims to.
"""

from __future__ import annotations

import functools
import json
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

# `RefResolver` itself warns on import (not just on use) -- see
# `_assert_response_matches_spec`'s docstring note for why it's used anyway.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from jsonschema import RefResolver

from app.chat import get_token_validator, get_trace_store
from app.config import Settings, get_settings
from app.main import app
from app.readiness import get_llama_server_client, get_ollama_client, get_openemr_client
from app.trace_store import TraceStore

_SPEC_PATH = Path(__file__).resolve().parent.parent / "openapi" / "openapi.json"

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@functools.lru_cache(maxsize=1)
def _load_pinned_spec() -> dict[str, Any]:
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200)


def _down_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _fake_client_dependency(handler):
    """Same shape as ``tests/test_ready.py``'s helper of the same name --
    duplicated rather than imported since that module's helper is
    module-private, and the two suites are testing different concerns
    (readiness semantics vs. spec conformance)."""

    async def _override():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as fake_client:
            yield fake_client

    return _override


def _assert_response_matches_spec(
    spec: dict[str, Any], *, path: str, method: str, status: int, body: Any
) -> None:
    schema = spec["paths"][path][method]["responses"][str(status)]["content"]["application/json"]["schema"]
    # RefResolver is deprecated in favor of the `referencing` library, but its
    # replacement has no ergonomic way to validate an inline sub-schema
    # (e.g. the `$ref` on /feedback's 201 response) against `$ref`s that
    # point elsewhere in the SAME already-in-memory document without giving
    # that document a synthetic `$id` first. `from_schema` is built exactly
    # for this case and is not scheduled for removal; deprecation warning is
    # deliberately silenced rather than worked around.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        resolver = RefResolver.from_schema(spec)
        validator = Draft202012Validator(schema, resolver=resolver)
    validator.validate(body)


def test_pinned_openapi_spec_exists() -> None:
    assert _SPEC_PATH.is_file(), (
        f"pinned OpenAPI spec missing at {_SPEC_PATH} -- run "
        "`python scripts/generate_openapi_spec.py` and commit the result"
    )


def test_pinned_spec_matches_live_schema() -> None:
    """Drift guard: the committed spec must equal what the app generates now."""
    live_spec = app.openapi()
    pinned_spec = _load_pinned_spec()
    assert pinned_spec == live_spec, (
        "openapi/openapi.json is stale -- regenerate with "
        "`python scripts/generate_openapi_spec.py` and commit the diff"
    )


def test_health_response_conforms_to_spec() -> None:
    spec = _load_pinned_spec()
    response = client.get("/health")

    assert response.status_code == 200
    _assert_response_matches_spec(spec, path="/health", method="get", status=200, body=response.json())


def test_ready_response_conforms_to_spec_when_ready(tmp_path) -> None:
    spec = _load_pinned_spec()

    app.dependency_overrides[get_openemr_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_ollama_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_llama_server_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_settings] = lambda: Settings(trace_db_path=str(tmp_path / "traces.db"))

    response = client.get("/ready")

    assert response.status_code == 200
    _assert_response_matches_spec(spec, path="/ready", method="get", status=200, body=response.json())


def test_ready_response_conforms_to_spec_when_not_ready(tmp_path) -> None:
    spec = _load_pinned_spec()

    app.dependency_overrides[get_openemr_client] = _fake_client_dependency(_down_handler)
    app.dependency_overrides[get_ollama_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_llama_server_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_settings] = lambda: Settings(trace_db_path=str(tmp_path / "traces.db"))

    response = client.get("/ready")

    assert response.status_code == 503
    _assert_response_matches_spec(spec, path="/ready", method="get", status=503, body=response.json())


def test_feedback_response_conforms_to_spec(tmp_path) -> None:
    spec = _load_pinned_spec()

    def _ok_validator(token: str) -> None:
        return None

    app.dependency_overrides[get_token_validator] = lambda: _ok_validator
    app.dependency_overrides[get_trace_store] = lambda: TraceStore(
        db_path=str(tmp_path / "traces.db"), hash_secret="0" * 32
    )

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-contract", "thumb": "up"},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 201
    _assert_response_matches_spec(spec, path="/feedback", method="post", status=201, body=response.json())

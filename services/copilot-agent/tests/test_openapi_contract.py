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

import copy
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


# #184: keys that are pure presentation/documentation metadata -- FastAPI and
# pydantic derive these from docstrings/field names and their exact rendering
# (wording, presence, ordering) has changed across versions without any
# change to the actual contract. Stripping them never removes a `required`
# entry, a `type`, an `enum` value, a `$ref`, or a path/parameter -- only
# cosmetic text.
#
# CAUTION: these are also legitimate property names. A schema can have a
# field literally named `title`/`description`/`summary`/`example`/`examples`
# (this app already has several -- app/schemas/retrieval.py, app/schemas/
# tools.py, app/quarantine.py, app/planner.py, app/retrieval.py) and a
# response `headers` map can contain a header with one of these names. Only
# strip a key when its PARENT dict is not a `properties` or `headers` map --
# see `_strip_presentation_metadata`.
_PRESENTATION_METADATA_KEYS = frozenset({"description", "title", "summary", "examples", "example"})

# #184 diagnosis: reproduced the standing local failure with an older
# fastapi pin (the diagnosis environment resolved fastapi 0.139.2 vs. fastapi
# 0.124.4 -- pyproject only floored on `fastapi>=0.115` at the time, with no
# lockfile pinning an exact version; both environments ran the same pydantic
# 2.13.4). Issue 213 later added services/copilot-agent/requirements.txt,
# hash-locking the production container build's exact fastapi resolution --
# this diagnosis predates that and is left as historical record of how the
# drift was found.
# and diffed the two schemas key-by-key. The ENTIRE delta -- across all 8
# paths and all 8 component schemas -- was two optional properties on this
# one component: `ValidationError.properties` gained `input`/`ctx` under the
# newer fastapi. `ValidationError` is FastAPI's own built-in
# request-validation-error model, hardcoded as `validation_error_definition`
# in `fastapi/openapi/utils.py`; that definition exists in both versions
# tested, but its `input`/`ctx` properties are present in 0.139.2 and absent
# in 0.124.4 -- no app module defines an
# OpenAPI component schema named `ValidationError`; the component is emitted
# solely by fastapi's own openapi generation. Its `required` list (`loc`,
# `msg`, `type`) is identical across both versions tested, so this is a
# framework rendering detail, not a change to anything this app authored.
# Scoped to this one named schema (not a blanket "ignore extra optional
# properties" rule) so a real optional-property change to one of OUR models
# (ChatRequest, FeedbackRequest, etc.) still fails the drift guard.
_FRAMEWORK_VERSION_DEPENDENT_SCHEMA_PROPERTIES = {
    "ValidationError": frozenset({"input", "ctx"}),
}

# #184: dict keys under these container keys are property/header NAMES, not
# presentation metadata -- even when a name happens to collide with one of
# `_PRESENTATION_METADATA_KEYS` (e.g. a schema property literally named
# `title`).
#
# Known limitation: the exemption is keyed on the PARENT key name alone, so
# a property literally named `properties` or `headers` would have its own
# schema treated as exempt too. That would surface as a loud false-positive
# test failure (a framework rewording of that field's title/description
# stops being stripped), not silent drift-masking. Zero properties with
# either name exist in the pinned spec today.
_METADATA_EXEMPT_CONTAINER_KEYS = frozenset({"properties", "headers"})


def _strip_presentation_metadata(node: Any, *, is_metadata_exempt_container: bool = False) -> Any:
    """Strip presentation-only keys, except where the parent dict is a
    `properties` or `headers` map -- there, every key is a property/header
    NAME (schema-meaningful), never presentation metadata, regardless of
    whether it collides with a metadata key like `title` (#184)."""
    if isinstance(node, dict):
        return {
            key: _strip_presentation_metadata(
                value, is_metadata_exempt_container=key in _METADATA_EXEMPT_CONTAINER_KEYS
            )
            for key, value in node.items()
            if is_metadata_exempt_container or key not in _PRESENTATION_METADATA_KEYS
        }
    if isinstance(node, list):
        return [_strip_presentation_metadata(item) for item in node]
    return node


def _normalize_spec_for_comparison(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OpenAPI schema (dict) down to its structurally meaningful
    contents for drift comparison (#184). See
    ``test_pinned_spec_matches_live_schema``'s docstring for exactly what is
    and is not compared after this normalization."""
    normalized = _strip_presentation_metadata(spec)

    schemas = normalized.get("components", {}).get("schemas", {})
    for schema_name, volatile_keys in _FRAMEWORK_VERSION_DEPENDENT_SCHEMA_PROPERTIES.items():
        properties = schemas.get(schema_name, {}).get("properties")
        if isinstance(properties, dict):
            for key in volatile_keys:
                properties.pop(key, None)

    return normalized


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
    """Drift guard: the committed spec must equal what the app generates now,
    after both sides pass through :func:`_normalize_spec_for_comparison`
    (#184).

    What IS compared (the signal): paths, methods, status codes, parameter
    names/locations/required-ness, and request/response schema structure --
    types, required fields, enum values, nullability, ``$ref`` targets. Any
    difference here is a real contract change and must fail this test.

    What is DELIBERATELY invisible after normalization (not signal): the
    presentation-only keys ``description``/``title``/``summary``/``examples``/
    ``example`` (FastAPI/pydantic render these differently across versions --
    see #184's diagnosis), and the ``input``/``ctx`` debug properties on the
    framework-injected ``ValidationError`` schema (FastAPI-version-dependent,
    not authored by this app -- no app module defines an OpenAPI component
    schema named ``ValidationError``; it is emitted by fastapi's own hardcoded
    ``validation_error_definition`` in ``fastapi/openapi/utils.py``, whose
    ``input``/``ctx`` properties are present in the diagnosis environment's
    fastapi 0.139.2 and absent in its fastapi 0.124.4 (see #184's diagnosis
    comment above the module-level constants for how those two versions were
    resolved, back when there was no repo lockfile pinning an exact fastapi
    version -- issue 213 has since added one for the production container
    build, ``services/copilot-agent/requirements.txt``); its ``required``
    list is untouched across the fastapi versions tested).
    Normalization never touches
    ``required`` lists, so a real field gaining or losing required-ness is
    still caught even on the ValidationError schema.
    """
    live_spec = _normalize_spec_for_comparison(app.openapi())
    pinned_spec = _normalize_spec_for_comparison(_load_pinned_spec())
    assert pinned_spec == live_spec, (
        "openapi/openapi.json has structural drift from the live schema "
        "(after normalizing away presentation-only metadata) -- regenerate "
        "with `python scripts/generate_openapi_spec.py`, review the diff for "
        "a REAL contract change, and commit it"
    )


def _normalized_before_and_after(mutate) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shared setup for the mutation-parametrized tests below: deep-copy the
    live schema, apply one mutation, and normalize both the original and the
    mutated copy for comparison.

    #184: asserts the mutation actually changed the PRE-normalization spec.
    Without this guard a mutate function that happens to set a key to the
    value it already holds (a no-op) would make
    ``test_normalization_tolerates_presentation_only_drift`` pass vacuously
    -- proving nothing about tolerance."""
    live_spec = app.openapi()
    mutated_spec = copy.deepcopy(live_spec)
    mutate(mutated_spec)
    assert mutated_spec != live_spec, "mutation was a no-op -- it did not change the pre-normalization spec"
    return _normalize_spec_for_comparison(live_spec), _normalize_spec_for_comparison(mutated_spec)


def test_normalized_live_schema_contains_load_bearing_paths() -> None:
    """Presence companion to the drift guard above: proves the normalized
    comparison can't pass vacuously (e.g. both sides accidentally emptied by
    an over-eager normalization step). If normalization ever strips down to
    ``{}`` on both sides, this catches it even though the equality assertion
    above would still pass."""
    normalized = _normalize_spec_for_comparison(app.openapi())
    paths = normalized["paths"]
    for expected_path in ("/chat", "/feedback", "/health", "/ready"):
        assert expected_path in paths, f"expected load-bearing path {expected_path!r} missing after normalization"
        assert paths[expected_path], f"path {expected_path!r} normalized to an empty operation map"


@pytest.mark.parametrize(
    ("mutate", "description"),
    [
        (
            lambda spec: spec["paths"].pop("/chat"),
            "removing an entire endpoint",
        ),
        (
            lambda spec: spec["components"]["schemas"]["ChatRequest"]["required"].remove("patient_id"),
            "dropping a required request field",
        ),
        (
            lambda spec: spec["components"]["schemas"]["ChatRequest"]["properties"]["message"].__setitem__(
                "type", "integer"
            ),
            "changing a property's type",
        ),
        (
            lambda spec: spec["components"]["schemas"]["FeedbackThumb"]["enum"].remove("down"),
            "removing an enum value",
        ),
        (
            lambda spec: spec["paths"]["/health"]["get"]["responses"].pop("200"),
            "removing a documented status code",
        ),
        (
            lambda spec: spec["components"]["schemas"]["ChatRequest"]["properties"].pop("conversation_id"),
            "removing an optional (non-required) property from an app-authored schema",
        ),
    ],
)
def test_normalization_still_catches_structural_drift(mutate, description) -> None:
    """Red-first (#184): normalization must strip presentation noise WITHOUT
    hiding real contract drift. Each case here deep-copies the live schema,
    injects one real structural mutation, and asserts the normalized
    comparison still flags it as different from the unmutated normalized
    schema."""
    baseline, mutated = _normalized_before_and_after(mutate)

    assert baseline != mutated, f"normalization hid real structural drift: {description}"


def _toggle_validation_error_property(spec: dict[str, Any], key: str, added_value: dict[str, Any]) -> None:
    """Add ``key`` to ``ValidationError.properties`` if absent, or remove it
    if present -- a real, always-nonzero change regardless of which fastapi
    version is installed locally (#184: this property is present on
    fastapi>=0.125, absent on fastapi<=0.124.x)."""
    properties = spec["components"]["schemas"]["ValidationError"]["properties"]
    if key in properties:
        properties.pop(key)
    else:
        properties[key] = added_value


@pytest.mark.parametrize(
    ("mutate", "description"),
    [
        (
            lambda spec: spec["components"]["schemas"]["ChatRequest"].__setitem__(
                "description", "a totally different description"
            ),
            "changing a schema's description",
        ),
        (
            lambda spec: spec["components"]["schemas"]["ChatRequest"]["properties"]["message"].__setitem__(
                "title", "Some Other Title"
            ),
            "changing a property's title",
        ),
        (
            lambda spec: _toggle_validation_error_property(spec, "input", {"title": "Input"}),
            "toggling presence of FastAPI's version-dependent ValidationError.input property "
            "(present on fastapi>=0.125, absent on fastapi<=0.124.x -- exercised in both directions "
            "so this can't pass by accident regardless of which fastapi is installed locally)",
        ),
        (
            lambda spec: _toggle_validation_error_property(spec, "ctx", {"title": "Context", "type": "object"}),
            "toggling presence of FastAPI's version-dependent ValidationError.ctx property "
            "(present on fastapi>=0.125, absent on fastapi<=0.124.x -- exercised in both directions "
            "so this can't pass by accident regardless of which fastapi is installed locally)",
        ),
    ],
)
def test_normalization_tolerates_presentation_only_drift(mutate, description) -> None:
    """Complement to the mutation-catch test above: proves normalization is
    not so strict that it re-introduces the #184 noise it exists to remove."""
    baseline, mutated = _normalized_before_and_after(mutate)

    assert baseline == mutated, f"normalization failed to absorb presentation-only noise: {description}"


def test_strip_presentation_metadata_preserves_properties_named_like_metadata_keys() -> None:
    """Regression for #184: a schema PROPERTY (or response header) can be
    literally named `title`/`description`/`summary`/`example`/`examples`
    (this app has several -- app/schemas/retrieval.py, app/schemas/tools.py,
    app/quarantine.py, app/planner.py, app/retrieval.py). Those names must
    survive normalization even though the same words are stripped when they
    appear as actual presentation metadata one level up."""
    schema = {
        "title": "SomeModel",  # presentation metadata -- must be stripped
        "description": "docstring content",  # presentation metadata -- must be stripped
        "required": ["title"],
        "properties": {
            "title": {"type": "string", "title": "Title"},
            "description": {"type": "string"},
            "summary": {"type": "string"},
            "example": {"type": "string"},
            "examples": {"type": "array"},
        },
    }
    headers_schema = {"headers": {"title": {"schema": {"type": "string"}}}}

    stripped = _strip_presentation_metadata(schema)
    stripped_headers = _strip_presentation_metadata(headers_schema)

    assert "title" not in stripped, "top-level presentation metadata `title` was not stripped"
    assert "description" not in stripped, "top-level presentation metadata `description` was not stripped"
    assert set(stripped["properties"]) == {"title", "description", "summary", "example", "examples"}, (
        "properties literally named like metadata keys were dropped from the properties map"
    )
    # the nested `title` INSIDE the `title` property is real presentation
    # metadata on that sub-schema and is correctly stripped.
    assert "title" not in stripped["properties"]["title"]
    assert stripped["properties"]["title"]["type"] == "string"
    assert "title" in stripped_headers["headers"], "a header literally named `title` was dropped"

    # removing or re-typing a property that happens to share a name with a
    # metadata key must still be caught as real drift.
    removed = copy.deepcopy(schema)
    removed["properties"].pop("title")
    assert _strip_presentation_metadata(removed) != stripped, "removing the `title` property went undetected"

    retyped = copy.deepcopy(schema)
    retyped["properties"]["title"]["type"] = "integer"
    assert _strip_presentation_metadata(retyped) != stripped, "re-typing the `title` property went undetected"


def test_validation_error_carve_out_is_scoped_to_that_one_schema() -> None:
    """Regression for #184: the `input`/`ctx` tolerance is scoped by name to
    `ValidationError` only. Injecting the same keys into an app-authored
    schema (`ChatRequest`) must NOT be tolerated -- proving the carve-out
    can't silently degrade into a blanket "ignore input/ctx everywhere"
    rule. See this module's mutation-verification report for the manual
    check that a blanket rule makes this test fail."""

    def _inject_into_chat_request(spec: dict[str, Any]) -> None:
        spec["components"]["schemas"]["ChatRequest"]["properties"]["input"] = {"title": "Input"}
        spec["components"]["schemas"]["ChatRequest"]["properties"]["ctx"] = {"title": "Context", "type": "object"}

    baseline, mutated = _normalized_before_and_after(_inject_into_chat_request)

    assert baseline != mutated, (
        "the ValidationError-only carve-out is not scoped -- input/ctx on ChatRequest were tolerated too"
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
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret="0" * 32)
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    # #180: the ownership check requires the correlation id to have been
    # originated by this same bearer token -- seed the REQUEST span a real
    # /chat call would have written, same as tests/test_feedback_endpoint.py.
    trace_store.record_request_span(
        correlation_id="corr-contract", start_ts=0.0, end_ts=0.1, ok=True, owner_token="good-token"
    )

    response = client.post(
        "/feedback",
        json={"correlation_id": "corr-contract", "thumb": "up"},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 201
    _assert_response_matches_spec(spec, path="/feedback", method="post", status=201, body=response.json())

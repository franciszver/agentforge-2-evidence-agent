"""Hermetic tests for the OpenEMR OAuth dev-token flow.

All HTTP is served by ``httpx.MockTransport`` so the suite never touches the
network. Live end-to-end verification lives in ``scripts/verify-oauth-dev.sh``,
outside the test suite.
"""

import json
import os
import stat

import httpx
import pytest

from app.openemr_auth import (
    ClientCredentials,
    OpenEmrAuthError,
    TokenResponse,
    authenticated_get,
    fetch_token_password_grant,
    register_client,
)


def test_register_client_returns_client_id_and_secret():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth2/default/registration"
        sent = json.loads(request.content)
        # Confidential client: OpenEMR requires application_type=private for
        # the user/* scopes this flow uses, and issues a client_secret for it.
        assert sent["application_type"] == "private"
        assert sent["token_endpoint_auth_method"] == "client_secret_post"
        return httpx.Response(
            201,
            json={
                "client_id": "abc123",
                "client_secret": "s3cr3t",
                "registration_access_token": "reg-tok",
                "client_name": "copilot-agent-dev",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        creds = register_client(
            client,
            base_url="https://openemr",
            registration_path="/oauth2/default/registration",
            client_name="copilot-agent-dev",
            redirect_uris=["https://agent.local/callback"],
            scope="openid api:oemr",
        )

    assert isinstance(creds, ClientCredentials)
    assert creds.client_id == "abc123"
    assert creds.client_secret == "s3cr3t"
    assert creds.registration_access_token == "reg-tok"


def test_fetch_token_password_grant_returns_token_with_access_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth2/default/token"
        body = request.content.decode()
        # OpenEMR-specific: password grant + the required user_role param.
        assert "grant_type=password" in body
        assert "user_role=users" in body
        return httpx.Response(
            200,
            json={
                "access_token": "access-tok",
                "refresh_token": "refresh-tok",
                "id_token": "id-tok",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "openid api:oemr",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        token = fetch_token_password_grant(
            client,
            base_url="https://openemr",
            token_path="/oauth2/default/token",
            client_id="cid",
            client_secret="csecret",
            username="admin",
            password="pass",
            scope="openid api:oemr",
        )

    assert isinstance(token, TokenResponse)
    assert token.access_token == "access-tok"
    assert token.refresh_token == "refresh-tok"


def test_fetch_token_bad_credentials_raises_without_leaking_secret_or_password():
    secret = "super-secret-client-value"
    password = "hunter2-user-password"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                # A hostile/echoing upstream might reflect the password here;
                # the raised error must not carry the body through verbatim.
                "error_description": f"bad password {password}",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OpenEmrAuthError) as excinfo:
            fetch_token_password_grant(
                client,
                base_url="https://openemr",
                token_path="/oauth2/default/token",
                client_id="cid",
                client_secret=secret,
                username="admin",
                password=password,
                scope="openid",
            )

    message = str(excinfo.value)
    assert secret not in message
    assert password not in message


def test_authenticated_get_sets_bearer_header_and_returns_200_body():
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        assert request.url.path == "/apis/default/fhir/Patient"
        return httpx.Response(200, json={"resourceType": "Bundle", "total": 0})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = authenticated_get(
            client,
            base_url="https://openemr",
            path="/apis/default/fhir/Patient",
            token="my-access-token",
        )

    assert captured["authorization"] == "Bearer my-access-token"
    assert response.status_code == 200
    assert response.json()["resourceType"] == "Bundle"


# --- #124 Phase 1: production authorization_code client registration -------

# The canonical browser-facing module OAuth callback. Phase 2's authorize/
# callback must match this byte-for-byte (OpenEMR requires exact redirect_uri
# matching). It is the BROWSER host (localhost:9300), not the internal
# ``openemr`` docker alias used for the server-side registration call.
_CANONICAL_REDIRECT_URI = (
    "https://localhost:9300/interface/modules/custom_modules/"
    "oe-module-clinical-copilot/public/oauth-callback.php"
)
# SMART-launch scopes + per-resource read scopes, every one confirmed present
# in OpenEMR's ServerScopeListEntity::getAllSupportedScopesList() (registration
# REJECTS unknown scopes with invalid_scope). The read scopes must be REGISTERED
# (not deferred to authorize time) -- ScopeRepository::finalizeScopes only lets
# a token carry scopes the client registered with. ``user/*.read`` is absent:
# OpenEMR has no wildcard entry.
_RECONCILED_PROD_SCOPES = (
    "openid offline_access launch launch/patient api:oemr api:fhir fhirUser "
    "user/patient.read user/medication.read user/allergy.read "
    "user/medical_problem.read user/encounter.read user/appointment.read "
    "user/vital.read user/procedure.read user/Observation.read"
)


def test_register_prod_client_sends_authz_code_payload():
    """The prod registration path posts the confidential authorization_code
    payload: canonical redirect_uri, reconciled SMART scopes, both
    authorization_code + refresh_token grants, application_type private."""
    from app.config import Settings
    from app.prod_client_registration import register_prod_client

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"client_id": "pc-1", "client_secret": "x"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        creds = register_prod_client(client, settings=Settings())

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["application_type"] == "private"
    assert "authorization_code" in body["grant_types"]
    assert "refresh_token" in body["grant_types"]
    # Confidential prod client must NOT accept the resource-owner password grant.
    assert "password" not in body["grant_types"]
    assert body["response_types"] == ["code"]
    assert body["redirect_uris"] == [_CANONICAL_REDIRECT_URI]
    assert body["scope"] == _RECONCILED_PROD_SCOPES
    # OpenEMR has no wildcard scope entry -- none must be requested (would be
    # rejected with invalid_scope at registration).
    assert "user/*" not in str(body["scope"])
    assert isinstance(creds, ClientCredentials)
    assert creds.client_id == "pc-1"


def test_dev_register_cli_payload_unchanged(tmp_path, monkeypatch):
    """Regression guard: #124 Phase 1 must not alter the DEV bridge's
    registration inputs (internal callback redirect + explicit per-resource
    dev scopes). Drives the real ``_register_cli`` with register_client faked."""
    from app import dev_token_bridge
    from app.openemr_auth import ClientCredentials as _Creds

    captured: dict[str, object] = {}

    def fake_register_client(client, **kwargs):  # noqa: ANN001, ANN003
        captured.update(kwargs)
        return _Creds(client_id="d-1", client_secret="y", registration_access_token="")

    monkeypatch.setattr(dev_token_bridge, "register_client", fake_register_client)
    monkeypatch.setenv("COPILOT_DEV_CLIENT_CREDS_PATH", str(tmp_path / "creds.json"))

    assert dev_token_bridge._register_cli() == 0

    assert captured["client_name"] == "copilot-agent-dev-bridge"
    assert captured["redirect_uris"] == ["https://openemr/oauth2/default/callback"]
    assert captured["scope"] == (
        "openid offline_access api:oemr api:fhir user/patient.read "
        "user/medication.read user/allergy.read user/medical_problem.read "
        "user/encounter.read user/appointment.read user/vital.read "
        "user/procedure.read user/Observation.read"
    )


# --- #176: OAuth client-secret creds file must be owner-only (0o600) --------


def _stub_prod_registration(monkeypatch, tmp_path):
    """Point the prod CLI at a tmp creds file and stub the network call."""
    from app import prod_client_registration as prod
    from app.openemr_auth import ClientCredentials as _Creds

    monkeypatch.setattr(
        prod,
        "register_prod_client",
        lambda client, *, settings: _Creds(
            client_id="pc-1", client_secret="x", registration_access_token=""
        ),
    )
    creds_file = tmp_path / "prod-creds.json"
    monkeypatch.setenv("COPILOT_PROD_CLIENT_CREDS_PATH", str(creds_file))
    return prod, creds_file


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes do not apply on Windows")
def test_prod_creds_file_is_owner_only_posix(tmp_path, monkeypatch):
    """The creds file (holds the OAuth client_secret) is written 0o600."""
    prod, creds_file = _stub_prod_registration(monkeypatch, tmp_path)

    assert prod._register_cli() == 0

    assert creds_file.exists()
    mode = stat.S_IMODE(os.stat(creds_file).st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_prod_creds_write_invokes_secure_chmod(tmp_path, monkeypatch):
    """Platform-independent: the write path enforces 0o600 via os.chmod.

    On Windows POSIX modes are a no-op, so instead of asserting the on-disk
    mode we assert the secure-write primitive (chmod to 0o600) is invoked."""
    prod, creds_file = _stub_prod_registration(monkeypatch, tmp_path)

    chmod_calls: list[tuple[str, int]] = []
    real_chmod = os.chmod

    def spy_chmod(path, mode, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        chmod_calls.append((str(path), mode))
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", spy_chmod)

    assert prod._register_cli() == 0

    assert (str(creds_file), 0o600) in chmod_calls

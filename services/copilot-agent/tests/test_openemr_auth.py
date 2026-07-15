"""Hermetic tests for the OpenEMR OAuth dev-token flow.

All HTTP is served by ``httpx.MockTransport`` so the suite never touches the
network. Live end-to-end verification lives in ``scripts/verify-oauth-dev.sh``,
outside the test suite.
"""

import json

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

"""OpenEMR OAuth2 confidential-client registration and dev token acquisition.

Scope (P0.5): prove the agent can obtain a *user* bearer token and make an
authenticated OpenEMR API call. This is deliberately minimal — it is NOT a
full API client (that is P2.2). It provides:

  * ``register_client``            — OpenEMR dynamic client registration.
  * ``fetch_token_password_grant`` — DEV-ONLY password-grant token acquisition.
  * ``authenticated_get``          — a single bearer-authenticated GET, enough
                                     to prove authorization end-to-end.

Security invariants:
  * ``OpenEmrAuthError`` messages never contain the client secret, the user
    password, or the raw upstream response body (which may echo either). Only
    a fixed operation label, the HTTP status code, and the OAuth ``error``
    *code* (a short, closed-set token such as ``invalid_grant``) are exposed.
  * All HTTP goes through an injected ``httpx.Client`` so tests drive it with
    ``httpx.MockTransport`` and no real network is touched.

Production note (see plan §4.2): production uses the OAuth2
``authorization_code`` grant against an admin-enabled client, NOT the
password grant. The password grant here is a dev-loop shortcut only.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

# OAuth error codes are a closed set defined by RFC 6749 §5.2 plus OpenID
# Connect. They are safe to surface (they carry no user input), unlike the
# free-text ``error_description`` which an upstream may populate by echoing
# request parameters.
_SAFE_OAUTH_ERROR_CODES = frozenset(
    {
        "invalid_request",
        "invalid_client",
        "invalid_grant",
        "unauthorized_client",
        "unsupported_grant_type",
        "invalid_scope",
        "access_denied",
        "server_error",
        "temporarily_unavailable",
    }
)


class OpenEmrAuthError(Exception):
    """Raised when an OpenEMR OAuth operation fails.

    The message is intentionally log-safe: it never embeds secrets, passwords,
    or raw upstream response bodies.
    """


@dataclass(frozen=True)
class ClientCredentials:
    """Credentials returned by OpenEMR dynamic client registration."""

    client_id: str
    client_secret: str
    registration_access_token: str


@dataclass(frozen=True)
class TokenResponse:
    """A user token set returned by the OpenEMR token endpoint."""

    access_token: str
    refresh_token: str | None
    id_token: str | None
    token_type: str
    expires_in: int | None
    scope: str | None


def _safe_error_detail(response: httpx.Response) -> str:
    """Build a log-safe failure detail from an OAuth error response.

    Includes only the HTTP status and, if present and recognised, the OAuth
    ``error`` code. The ``error_description`` and any other body fields are
    deliberately discarded — they may reflect request input (e.g. a bad
    password) verbatim.
    """
    error_code: str | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        candidate = payload.get("error")
        if isinstance(candidate, str) and candidate in _SAFE_OAUTH_ERROR_CODES:
            error_code = candidate
    if error_code is not None:
        return f"status {response.status_code}, error {error_code}"
    return f"status {response.status_code}"


def register_client(
    client: httpx.Client,
    *,
    base_url: str,
    registration_path: str,
    client_name: str,
    redirect_uris: list[str],
    scope: str,
    grant_types: list[str] | None = None,
) -> ClientCredentials:
    """Register a confidential OAuth2 client via OpenEMR dynamic registration.

    ``application_type: "private"`` is what makes OpenEMR treat the client as
    confidential and issue a ``client_secret`` — and it is *required* for the
    ``user/*`` (and ``system/*``) scopes this flow needs. Without it OpenEMR
    rejects the request with ``invalid_client_metadata`` ("system and user
    scopes are only allowed for confidential clients"). The
    ``token_endpoint_auth_method`` is set to ``client_secret_post`` to match.

    ``grant_types`` defaults to the dev set
    (``["password", "refresh_token", "authorization_code"]``) so the dev
    registration payload is unchanged. The production path passes
    ``["authorization_code", "refresh_token"]`` *without* ``password`` — a
    confidential prod client must not accept the resource-owner password grant,
    or a leaked ``client_secret`` plus any clinician credential could mint
    tokens directly, bypassing the authorization_code + consent flow.

    NOTE: OpenEMR registers new clients *disabled*; an admin must enable the
    client before it can obtain tokens (see ``scripts/verify-oauth-dev.sh``).
    """
    url = f"{base_url}{registration_path}"
    request_body = {
        "application_type": "private",
        "client_name": client_name,
        "redirect_uris": redirect_uris,
        "token_endpoint_auth_method": "client_secret_post",
        "grant_types": (
            grant_types
            if grant_types is not None
            else ["password", "refresh_token", "authorization_code"]
        ),
        "response_types": ["code"],
        "scope": scope,
    }
    try:
        response = client.post(url, json=request_body)
    except httpx.HTTPError as exc:
        raise OpenEmrAuthError("client registration request failed") from exc

    if not response.is_success:
        raise OpenEmrAuthError(f"client registration failed: {_safe_error_detail(response)}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenEmrAuthError("client registration returned a non-JSON body") from exc

    client_id = payload.get("client_id")
    client_secret = payload.get("client_secret")
    if not isinstance(client_id, str) or not isinstance(client_secret, str):
        raise OpenEmrAuthError("client registration response missing credentials")

    registration_access_token = payload.get("registration_access_token")
    return ClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        registration_access_token=(
            registration_access_token if isinstance(registration_access_token, str) else ""
        ),
    )


def fetch_token_password_grant(
    client: httpx.Client,
    *,
    base_url: str,
    token_path: str,
    client_id: str,
    client_secret: str,
    username: str,
    password: str,
    scope: str,
    user_role: str = "users",
) -> TokenResponse:
    """DEV-ONLY: obtain a user token via the OAuth2 password grant.

    The password (Resource Owner Password Credentials) grant is a
    development-loop shortcut. Production MUST use the ``authorization_code``
    grant against an admin-enabled client (plan §4.2); do not ship this path.

    ``user_role`` defaults to ``"users"`` — an OpenEMR-specific required
    parameter for the password grant.

    On failure, ``OpenEmrAuthError`` is raised with a log-safe message that
    excludes the secret, password, and raw response body.
    """
    url = f"{base_url}{token_path}"
    form = {
        "grant_type": "password",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
        "username": username,
        "password": password,
        "user_role": user_role,
    }
    try:
        response = client.post(url, data=form)
    except httpx.HTTPError as exc:
        raise OpenEmrAuthError("token request failed") from exc

    if not response.is_success:
        raise OpenEmrAuthError(f"token request failed: {_safe_error_detail(response)}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenEmrAuthError("token response was not valid JSON") from exc

    access_token = payload.get("access_token")
    if not isinstance(access_token, str):
        raise OpenEmrAuthError("token response missing access_token")

    return TokenResponse(
        access_token=access_token,
        refresh_token=_optional_str(payload.get("refresh_token")),
        id_token=_optional_str(payload.get("id_token")),
        token_type=_optional_str(payload.get("token_type")) or "Bearer",
        expires_in=payload.get("expires_in") if isinstance(payload.get("expires_in"), int) else None,
        scope=_optional_str(payload.get("scope")),
    )


def authenticated_get(
    client: httpx.Client,
    *,
    base_url: str,
    path: str,
    token: str,
) -> httpx.Response:
    """Make a single bearer-authenticated GET — enough to prove authorization.

    This is intentionally NOT a full API client (P2.2). The raw
    ``httpx.Response`` is returned so the caller decides how to interpret it.
    """
    url = f"{base_url}{path}"
    return client.get(url, headers={"Authorization": f"Bearer {token}"})


def _optional_str(value: object) -> str | None:
    """Return ``value`` if it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value != "" else None

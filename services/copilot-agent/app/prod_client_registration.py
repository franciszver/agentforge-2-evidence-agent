"""Production OAuth2 ``authorization_code`` client registration (#124 Phase 1).

Registers the confidential client the browser-driven ``authorization_code``
flow (Phase 2) will use. Unlike the dev token bridge (password grant, demo
clinician credential, SQL ``is_enabled=1`` shortcut), this path:

  * targets the browser-facing module OAuth callback as the CANONICAL
    ``redirect_uri`` (``Settings.copilot_prod_client_redirect_uri``),
  * requests the reconciled SMART-on-FHIR scope set
    (``Settings.copilot_prod_client_scopes``), and
  * does NOT enable the client -- an OpenEMR admin must approve/enable it via
    Administration -> Config -> Connectors (the dev SQL ``is_enabled=1``
    shortcut is dev-only). See ``services/copilot-agent/README.md``.

PKCE (S256) is a redirect-time concern handled in Phase 2's authorize/callback
(the ``code_challenge`` parameter travels on the authorize request, not in
client-registration metadata per RFC 7591; OpenEMR's ``CustomAuthCodeGrant``
enables S256 at the grant level). Nothing PKCE-related is declared here.
"""

from __future__ import annotations

import httpx

from app.config import Settings
from app.creds_file import write_creds_securely
from app.openemr_auth import ClientCredentials, register_client

_PROD_CLIENT_NAME = "copilot-agent-prod"


def register_prod_client(client: httpx.Client, *, settings: Settings) -> ClientCredentials:
    """Register the production confidential ``authorization_code`` client.

    Uses the canonical browser-facing module callback ``redirect_uri`` and the
    reconciled SMART scope set from ``settings``. ``register_client`` supplies
    ``application_type: "private"`` (confidential); this path passes
    ``authorization_code`` + ``refresh_token`` grants only (no ``password``).
    """
    return register_client(
        client,
        base_url=settings.openemr_base_url,
        registration_path=settings.openemr_oauth_registration_path,
        client_name=_PROD_CLIENT_NAME,
        redirect_uris=[settings.copilot_prod_client_redirect_uri],
        scope=settings.copilot_prod_client_scopes,
        # No ``password`` grant for the confidential prod client (see
        # register_client): only the browser authorization_code + refresh.
        grant_types=["authorization_code", "refresh_token"],
    )


def _build_http_client(settings: Settings) -> httpx.Client:
    """Build the OpenEMR-facing HTTP client from settings."""
    return httpx.Client(
        verify=settings.openemr_verify_ssl,
        timeout=settings.openemr_api_timeout_seconds,
    )


def _register_cli() -> int:
    """Register the production client and write its credentials.

    Run inside the agent container (only it can reach OpenEMR on the internal
    network) by ``scripts/register-copilot-prod-client.sh``. Writes the
    credentials to ``copilot_prod_client_creds_path`` and prints the
    ``client_id`` (not a secret on its own); the ``client_secret`` is written
    ONLY to the creds file and is never printed.

    The client is registered DISABLED -- an OpenEMR admin must approve/enable
    it before the authorization_code flow works (no SQL shortcut in prod).
    """
    settings = Settings()
    with _build_http_client(settings) as client:
        creds = register_prod_client(client, settings=settings)
    write_creds_securely(
        settings.copilot_prod_client_creds_path,
        {"client_id": creds.client_id, "client_secret": creds.client_secret},
    )
    print(f"CLIENT_ID={creds.client_id}")
    print(
        "Client registered DISABLED. An OpenEMR admin must approve/enable it via "
        "Administration -> Config -> Connectors -> OAuth2 Clients before the "
        "authorization_code flow will issue tokens."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_register_cli())

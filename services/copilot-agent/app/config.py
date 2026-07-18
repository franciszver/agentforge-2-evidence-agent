"""Application configuration loaded from environment variables.

Dev-environment default: the OpenEMR instance on the internal docker
network uses a self-signed certificate, so ``OPENEMR_VERIFY_SSL``
defaults to ``False`` here. Override it via the environment in any
deployment where certificate verification must be enforced.
"""

from __future__ import annotations

import secrets

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the copilot-agent service."""

    openemr_base_url: str = "https://openemr"
    ollama_base_url: str = "http://ollama:11434"
    trace_db_path: str = "/data/traces.db"
    # HMAC key for TraceStore.hash_args (P4.2) -- keeps tool-call args
    # non-reversible even though they are often low-entropy (patient ids,
    # closed-set filter keys, date ranges); an unkeyed hash would let an
    # attacker with read access to traces.db precompute the hash over the
    # candidate space and recover the original args. NO hardcoded default:
    # this repo is public, so any literal default here would be a published
    # key -- defeating the keying entirely. Unset => a strong random key is
    # generated per process (fail-safe). Set TRACE_ARGS_HASH_SECRET in the
    # environment to pin a stable key when args_hash must stay comparable
    # across restarts (e.g. the P4.5 review dashboard).
    trace_args_hash_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    openemr_verify_ssl: bool = False
    # Per-request timeout for calls made by ``OpenEmrClient`` (app/openemr_client.py).
    openemr_api_timeout_seconds: float = 10.0

    # Model served by the internal Ollama instance and per-request timeout /
    # retry policy for ``OllamaClient`` (app/ollama_client.py).
    ollama_model: str = "qwen3:4b"
    ollama_api_timeout_seconds: float = 60.0
    ollama_extract_max_retries: int = 2

    # OAuth2 endpoints on the OpenEMR "default" site. Paths are relative to
    # ``openemr_base_url``.
    openemr_oauth_registration_path: str = "/oauth2/default/registration"
    openemr_oauth_token_path: str = "/oauth2/default/token"
    # RFC 7662 token introspection endpoint (#124 Phase 4). The agent
    # introspects the forwarded per-user bearer here, authenticating as the
    # confidential prod client via HTTP Basic auth.
    openemr_oauth_introspection_path: str = "/oauth2/default/introspect"
    # Superset of scopes requested for the dev token flow: OIDC + refresh,
    # standard + FHIR API, and a FHIR Patient read scope for the proof call.
    openemr_oauth_scopes: str = (
        "openid offline_access api:oemr api:fhir user/patient.read user/Patient.read"
    )

    # DEV-ONLY dev-token bridge (issue #126, finding F4). The agent obtains a
    # REAL OpenEMR user token server-side (dev password grant) for its tool
    # calls, because the browser's DevAgentToken is only an identity assertion,
    # not a real OpenEMR token. The real token never reaches the browser.
    # Identity for ACL is this demo clinician until #124 (production
    # authorization_code) lands. See app/dev_token_bridge.py.
    #
    # Path (inside the agent container) to the confidential-client credentials
    # written by scripts/bootstrap-copilot-dev-client.sh. Lives under the
    # appuser-writable /data dir so the running agent can read it.
    copilot_dev_client_creds_path: str = "/data/openemr-dev-client.json"
    # Demo clinician credential used for the dev password grant (dev defaults;
    # override via env in any non-default dev setup).
    copilot_dev_clinician_username: str = "admin"
    copilot_dev_clinician_password: str = "pass"
    # Resource read scopes the tools need. Only OpenEMR-recognized standard/FHIR
    # scope identifiers (see ServerScopeListEntity::apiScopes) -- e.g. problems
    # are user/medical_problem.read, labs/vitals use the FHIR Observation scope.
    copilot_dev_token_scopes: str = (
        "openid offline_access api:oemr api:fhir user/patient.read "
        "user/medication.read user/allergy.read user/medical_problem.read "
        "user/encounter.read user/appointment.read user/vital.read "
        "user/procedure.read user/Observation.read"
    )

    # Production authorization_code client (#124 Phase 1). Distinct from the dev
    # token bridge above: this client is driven by the browser via the OAuth2
    # authorization_code grant, and an OpenEMR admin (not a dev SQL shortcut)
    # enables it. See app/prod_client_registration.py and the README.
    #
    # CANONICAL redirect_uri -- the single source of truth Phase 2's authorize/
    # callback must match byte-for-byte (OpenEMR requires exact redirect_uri
    # matching). It is the BROWSER-facing host (localhost:9300) and the module's
    # one-file-per-route OAuth callback endpoint, NOT the internal ``openemr``
    # docker alias used for the server-side registration/token calls.
    copilot_prod_client_redirect_uri: str = (
        "https://localhost:9300/interface/modules/custom_modules/"
        "oe-module-clinical-copilot/public/oauth-callback.php"
    )
    # SMART-on-FHIR scopes for the production client. Every scope MUST exist in
    # OpenEMR's ServerScopeListEntity::getAllSupportedScopesList() -- dynamic
    # registration REJECTS the whole request with ``invalid_scope`` on the first
    # unrecognized scope (AuthorizationController::validateScopesAgainstServer-
    # ApprovedScopes). ``user/*.read`` is NOT used -- OpenEMR has no wildcard
    # scope entry, so it would be rejected.
    #
    # The per-resource read scopes are registered here (not deferred to authorize
    # time): ScopeRepository::finalizeScopes only lets a token carry scopes the
    # client REGISTERED with, so an unregistered read scope requested at grant
    # time is silently dropped -> tool calls would have api:oemr/api:fhir but no
    # resource-read authorization. These mirror the known-accepted
    # copilot_dev_token_scopes plus the SMART-launch scopes.
    copilot_prod_client_scopes: str = (
        "openid offline_access launch launch/patient api:oemr api:fhir fhirUser "
        "user/patient.read user/medication.read user/allergy.read "
        "user/medical_problem.read user/encounter.read user/appointment.read "
        "user/vital.read user/procedure.read user/Observation.read"
    )
    # Path (inside the agent container) for the production client credentials
    # written by the prod registration CLI. Distinct file from the dev bridge's.
    copilot_prod_client_creds_path: str = "/data/openemr-prod-client.json"

    # #124 Phase 4 (pivotal): when true, the agent (a) validates the request's
    # forwarded per-user OpenEMR bearer via introspection and (b) uses that same
    # token for every tool call, so OpenEMR enforces per-user ACL end-to-end.
    # Default OFF -- flag off keeps today's dev stub validator + DevTokenBridge
    # demo-clinician token, byte-identical. Flipped on with the module side in
    # Phase 6. Introspection results are cached (hash-keyed) for this many
    # seconds, further capped by each token's own ``exp``.
    copilot_per_user_token_enabled: bool = False
    copilot_introspection_cache_ttl_seconds: float = 60.0


def get_settings() -> Settings:
    """FastAPI dependency returning the current application settings."""
    return Settings()

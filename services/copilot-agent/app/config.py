"""Application configuration loaded from environment variables.

Dev-environment default: the OpenEMR instance on the internal docker
network uses a self-signed certificate, so ``OPENEMR_VERIFY_SSL``
defaults to ``False`` here. Override it via the environment in any
deployment where certificate verification must be enforced.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the copilot-agent service."""

    openemr_base_url: str = "https://openemr"
    ollama_base_url: str = "http://ollama:11434"
    trace_db_path: str = "/data/traces.db"
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
    # Superset of scopes requested for the dev token flow: OIDC + refresh,
    # standard + FHIR API, and a FHIR Patient read scope for the proof call.
    openemr_oauth_scopes: str = (
        "openid offline_access api:oemr api:fhir user/patient.read user/Patient.read"
    )


def get_settings() -> Settings:
    """FastAPI dependency returning the current application settings."""
    return Settings()

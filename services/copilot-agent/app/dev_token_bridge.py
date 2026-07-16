"""DEV-ONLY: real OpenEMR bearer-token bridge for the agent's tool calls.

Finding F4 / issue #126. The browser hands the agent a ``DevAgentToken`` -- an
HMAC identity assertion (see the module's ``TokenBrokerController`` /
``DevAgentToken``), NOT a real OpenEMR token -- so tool calls made with it
auth-fail against the OpenEMR API. This bridge lets the AGENT obtain a REAL
OpenEMR user token server-side (via the dev password grant against a
confidential client scoped for the resource reads) and cache it, so tool calls
actually authenticate. **The real token never reaches the browser.**

Trust boundary (unchanged by this bridge): the browser's ``DevAgentToken``
still gates ``POST /chat`` (the token-validator seam) and still carries the
``pid`` used for patient-context binding (P2.16). This bridge only supplies the
credential the *tools* use to read OpenEMR. Identity for ACL purposes is the
configured demo clinician until #124 (production ``authorization_code``,
per-user tokens) lands; per-user ACL differentiation remains #124.

DEV-ONLY, do NOT ship (the same shortcuts as ``scripts/verify-oauth-dev.sh``):
  * the OAuth2 password grant instead of ``authorization_code`` (#124),
  * a demo clinician credential drawn from config,
  * the confidential client enabled via a dev SQL shortcut (the bootstrap
    script), because OpenEMR registers new clients disabled.

Security invariants:
  * ``DevTokenError`` messages are log-safe: they never embed the client
    secret, the clinician password, or a raw upstream response body
    (``OpenEmrAuthError`` already guarantees this for the wrapped failure).
  * The real access token is held only in-memory (a process-local cache) and
    is returned only to the server-side planner factory -- never serialized to
    the browser.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeGuard

import httpx

from app.config import Settings
from app.creds_file import write_creds_securely
from app.openemr_auth import (
    OpenEmrAuthError,
    TokenResponse,
    fetch_token_password_grant,
    register_client,
)

# Fallback token lifetime when the token endpoint omits ``expires_in``. OpenEMR
# access tokens live ~1h; the safety margin re-fetches slightly early so a
# token never expires mid-request.
_DEFAULT_TOKEN_TTL_SECONDS = 3600
_EXPIRY_SAFETY_MARGIN_SECONDS = 60


class DevTokenError(Exception):
    """Raised when the dev token bridge cannot obtain a real OpenEMR token.

    Log-safe: never embeds the client secret, the clinician password, or a raw
    upstream body.
    """


@dataclass(frozen=True)
class _CachedToken:
    """A cached access token and the wall-clock deadline after which it is
    considered stale and must be re-fetched."""

    access_token: str
    expires_at: float


ClientFactory = Callable[[], httpx.Client]


def _build_http_client(settings: Settings) -> httpx.Client:
    """Build the OpenEMR-facing HTTP client from settings (single construction
    point so TLS/timeout wiring stays consistent across the bridge)."""
    return httpx.Client(
        verify=settings.openemr_verify_ssl,
        timeout=settings.openemr_api_timeout_seconds,
    )


def _is_nonempty_str(value: object) -> TypeGuard[str]:
    """True only for a present, non-empty string (narrows the type for callers)."""
    return isinstance(value, str) and bool(value)


class DevTokenBridge:
    """Obtains and caches a real OpenEMR user token for the agent's tool calls.

    Thread-safe: FastAPI runs the sync ``POST /chat`` dependency chain in a
    worker-thread pool, so concurrent requests may call :meth:`get_token`
    simultaneously; a lock serializes the check-and-fetch so at most one
    upstream token request is in flight and no request reads a torn cache.

    Args:
        base_url: OpenEMR origin (scheme + host).
        token_path: OAuth2 token endpoint path, relative to ``base_url``.
        creds_path: Path to the confidential-client credentials JSON written by
            the bootstrap script (``{"client_id", "client_secret"}``).
        username / password: Demo clinician credential for the password grant.
        scope: Space-separated resource read scopes the tools need.
        client_factory: Builds a fresh ``httpx.Client`` per fetch; injected so
            hermetic tests supply a ``MockTransport``-backed client.
        clock: Time source (seconds); injected for deterministic TTL tests.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token_path: str,
        creds_path: str,
        username: str,
        password: str,
        scope: str,
        client_factory: ClientFactory,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._base_url = base_url
        self._token_path = token_path
        self._creds_path = creds_path
        self._username = username
        self._password = password
        self._scope = scope
        self._client_factory = client_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: _CachedToken | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> DevTokenBridge:
        """Build a production bridge, threading base URL, TLS, timeout, creds
        path, demo credential, and scopes from ``Settings``."""

        def _client_factory() -> httpx.Client:
            return _build_http_client(settings)

        return cls(
            base_url=settings.openemr_base_url,
            token_path=settings.openemr_oauth_token_path,
            creds_path=settings.copilot_dev_client_creds_path,
            username=settings.copilot_dev_clinician_username,
            password=settings.copilot_dev_clinician_password,
            scope=settings.copilot_dev_token_scopes,
            client_factory=_client_factory,
        )

    def get_token(self) -> str:
        """Return a real OpenEMR access token, fetching or refreshing as needed.

        Returns the cached token while it is still within its TTL; otherwise
        fetches a fresh one via the password grant and caches it. Raises
        ``DevTokenError`` (log-safe) if credentials are missing or the upstream
        token request fails.
        """
        with self._lock:
            now = self._clock()
            cached = self._cached
            if cached is not None and now < cached.expires_at:
                return cached.access_token

            token = self._fetch()
            ttl = (
                token.expires_in
                if isinstance(token.expires_in, int) and token.expires_in > 0
                else _DEFAULT_TOKEN_TTL_SECONDS
            )
            expires_at = now + max(ttl - _EXPIRY_SAFETY_MARGIN_SECONDS, 0)
            self._cached = _CachedToken(access_token=token.access_token, expires_at=expires_at)
            return token.access_token

    def _fetch(self) -> TokenResponse:
        client_id, client_secret = self._load_creds()
        try:
            with self._client_factory() as client:
                return fetch_token_password_grant(
                    client,
                    base_url=self._base_url,
                    token_path=self._token_path,
                    client_id=client_id,
                    client_secret=client_secret,
                    username=self._username,
                    password=self._password,
                    scope=self._scope,
                )
        except OpenEmrAuthError as exc:
            # OpenEmrAuthError is already log-safe (no secret/password/body);
            # wrap it in the bridge's error type without embedding its message.
            raise DevTokenError("dev token acquisition failed") from exc

    def _load_creds(self) -> tuple[str, str]:
        try:
            raw = Path(self._creds_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise DevTokenError(
                "dev client credentials not found; run scripts/bootstrap-copilot-dev-client.sh"
            ) from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise DevTokenError("dev client credentials file is not valid JSON") from exc

        if not isinstance(data, dict):
            raise DevTokenError("dev client credentials file missing client_id/client_secret")
        client_id = data.get("client_id")
        client_secret = data.get("client_secret")
        if not _is_nonempty_str(client_id) or not _is_nonempty_str(client_secret):
            raise DevTokenError("dev client credentials file missing client_id/client_secret")
        return client_id, client_secret


def _register_cli() -> int:
    """DEV-ONLY bootstrap step: register the confidential client, write creds.

    Run inside the agent container (only it can reach OpenEMR on the internal
    network) by ``scripts/bootstrap-copilot-dev-client.sh``. Writes the
    credentials to ``copilot_dev_client_creds_path`` and prints the
    ``client_id`` (not a secret on its own) so the script can enable the client
    via the dev SQL shortcut. The client_secret is written ONLY to the creds
    file and is never printed.
    """
    settings = Settings()
    with _build_http_client(settings) as client:
        creds = register_client(
            client,
            base_url=settings.openemr_base_url,
            registration_path=settings.openemr_oauth_registration_path,
            client_name="copilot-agent-dev-bridge",
            redirect_uris=[f"{settings.openemr_base_url}/oauth2/default/callback"],
            scope=settings.copilot_dev_token_scopes,
        )
    write_creds_securely(
        settings.copilot_dev_client_creds_path,
        {"client_id": creds.client_id, "client_secret": creds.client_secret},
    )
    print(f"CLIENT_ID={creds.client_id}")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "register":
        raise SystemExit(_register_cli())
    print("usage: python -m app.dev_token_bridge register", file=sys.stderr)
    raise SystemExit(2)

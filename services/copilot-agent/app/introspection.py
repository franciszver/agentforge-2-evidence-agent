"""Cached OpenEMR token introspection for the agent (#124 Phase 4).

Wraps the low-level :func:`app.openemr_auth.introspect_token` primitive with a
short-TTL, hash-keyed cache so the per-``/chat`` introspection round-trip does
not regress the latency budget. Mirrors ``DevTokenBridge``'s cache-with-lock
pattern (FastAPI runs the sync ``/chat`` dependency chain in a worker-thread
pool, so concurrent requests may introspect simultaneously).

Security invariants:
  * The raw token is NEVER used as a cache key -- the key is a SHA-256 hex
    digest, so a read of process memory / a cache dump never yields a usable
    bearer.
  * The token and the client secret are NEVER logged.
  * Fail-closed: any inability to obtain a positive ``active:true`` result
    (missing/invalid client creds, network error, inactive token) yields
    ``IntrospectionResult(active=False)`` so the caller rejects the request.
    Only positive (active) results are cached -- a transient failure must not
    lock a valid user out for the whole TTL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from app.config import Settings
from app.openemr_auth import IntrospectionResult, introspect_token

_logger = logging.getLogger(__name__)

ClientFactory = Callable[[], httpx.Client]


@dataclass(frozen=True)
class _CachedIntrospection:
    """A cached positive introspection result and its wall-clock deadline."""

    result: IntrospectionResult
    expires_at: float


class TokenIntrospector:
    """Introspects forwarded bearer tokens against OpenEMR, with a TTL cache.

    Args:
        base_url: OpenEMR origin (scheme + host).
        introspect_path: RFC 7662 introspection endpoint path, relative to
            ``base_url``.
        creds_path: Path to the confidential-client credentials JSON (the
            production ``authorization_code`` client) used for Basic auth.
        client_factory: Builds a fresh ``httpx.Client`` per introspection;
            injected so hermetic tests supply a ``MockTransport``-backed client.
        cache_ttl_seconds: Upper bound on how long a positive result is cached
            (further capped by the token's own ``exp`` when present).
        clock: Time source (seconds); injected for deterministic TTL tests.
    """

    def __init__(
        self,
        *,
        base_url: str,
        introspect_path: str,
        creds_path: str,
        client_factory: ClientFactory,
        cache_ttl_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._base_url = base_url
        self._introspect_path = introspect_path
        self._creds_path = creds_path
        self._client_factory = client_factory
        self._cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[str, _CachedIntrospection] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> TokenIntrospector:
        """Build a production introspector from ``Settings`` (prod client creds,
        introspection path, TLS/timeout, and cache TTL)."""

        def _client_factory() -> httpx.Client:
            return httpx.Client(
                verify=settings.openemr_verify_ssl,
                timeout=settings.openemr_api_timeout_seconds,
            )

        return cls(
            base_url=settings.openemr_base_url,
            introspect_path=settings.openemr_oauth_introspection_path,
            creds_path=settings.copilot_prod_client_creds_path,
            client_factory=_client_factory,
            cache_ttl_seconds=settings.copilot_introspection_cache_ttl_seconds,
        )

    def introspect(self, token: str) -> IntrospectionResult:
        """Return the introspection result for ``token``, cached by token-hash.

        Double-checked locking so the (network) introspection round-trip runs
        OUTSIDE the lock: a single cache-miss must not serialize every other
        concurrent ``/chat`` introspection process-wide. The lock is held only
        for the two cache touches -- the fast check and the store.

        (1) lock -> return a still-fresh positive result on a cache hit;
        (2) unlocked -> introspect upstream (two concurrent misses redundantly
            introspecting is fine -- it is idempotent);
        (3) lock -> store, but only for a positive result, with
            ``deadline = min(now + ttl, exp)`` so the cap never outlives the
            token. Fail-closed on any error; inactive results are never cached.
        """
        key = hashlib.sha256(token.encode()).hexdigest()

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and self._clock() < cached.expires_at:
                return cached.result

        result = self._introspect_upstream(token)
        if not result.active:
            return result

        with self._lock:
            deadline = self._clock() + self._cache_ttl_seconds
            if result.exp is not None:
                deadline = min(deadline, float(result.exp))
            self._cache[key] = _CachedIntrospection(result=result, expires_at=deadline)
        return result

    def _introspect_upstream(self, token: str) -> IntrospectionResult:
        creds = self._load_creds()
        if creds is None:
            # Fail-closed misconfiguration: no client creds => cannot
            # introspect => reject. Log-safe (no token, no secret).
            _logger.warning(
                "introspection client credentials unavailable; rejecting token",
                extra={"creds_path": self._creds_path},
            )
            return IntrospectionResult(active=False, exp=None)
        client_id, client_secret = creds
        with self._client_factory() as client:
            return introspect_token(
                client,
                base_url=self._base_url,
                introspect_path=self._introspect_path,
                client_id=client_id,
                client_secret=client_secret,
                token=token,
            )

    def _load_creds(self) -> tuple[str, str] | None:
        """Load ``{client_id, client_secret}`` from the creds file, or ``None``.

        Any problem (missing file, non-JSON, missing/blank fields) returns
        ``None`` -- never raises and never surfaces the secret.
        """
        try:
            data = json.loads(Path(self._creds_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        client_id = data.get("client_id")
        client_secret = data.get("client_secret")
        if isinstance(client_id, str) and client_id and isinstance(client_secret, str) and client_secret:
            return client_id, client_secret
        return None

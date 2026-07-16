"""#124 Phase 5: SMART launch-context patient binding for POST /chat.

The forwarded per-user token can carry a SMART *launch patient* -- the patient
the app was launched for. OpenEMR's RFC 7662 introspection response echoes it
under the ``patient`` key as a FHIR **UUID** string (see
``src/RestControllers/TokenIntrospectionRestController.php``): the token's
stored ``context`` column is copied through verbatim, so no pid/uuid
conversion happens server-side.

This module makes that launch patient the AUTHORITATIVE binding -- a /chat
request whose ``patient_id`` (an internal integer pid) does not map to the
launch UUID is refused before any tool call. It is defense-in-depth atop the
existing agent-side P2.16 conversation-pid binding, not a replacement.

pid -> uuid mapping: the agent resolves ``patient_id`` to its FHIR UUID via the
same forwarded token, using OpenEMR's REST patient roster
(``GET /apis/default/api/patient``, filtered client-side by pid -- the internal
pid is deliberately not a public REST filter, so the roster is fetched and the
record selected here). This mirrors ``app.tools.patient_summary``'s
demographics read and costs one extra authenticated round trip per flag-on
/chat request that carries launch context.

Fail-safe posture: a token WITHOUT launch context (``patient`` absent) is NOT
rejected here -- ``verify`` returns and the request falls back to the P2.16
conversation-pid binding. But once a launch patient IS present, ANY inability
to confirm the match -- a genuine mismatch, the pid absent from the roster, or
the resolve read failing -- rejects the request. The token, the pid, and the
UUIDs are never logged.
"""

from __future__ import annotations

from typing import Protocol

from app.config import Settings
from app.openemr_auth import IntrospectionResult
from app.openemr_client import OpenEmrApiError, OpenEmrClient


class LaunchPatientMismatchError(Exception):
    """The request's ``patient_id`` does not match the token's launch patient.

    Also raised fail-safe when a launch patient IS present but the pid->uuid
    resolution cannot confirm the match (pid absent from the roster, or the
    resolve read failed). The message is log-safe: never a token, pid, or UUID.
    """


class _Introspector(Protocol):
    """What the binder needs from the introspector: ``token -> result``.

    ``app.introspection.TokenIntrospector`` satisfies this; sharing the same
    instance the token validator uses means this introspection is a cache hit
    (no extra introspection round trip -- only the pid->uuid resolve read is
    new).
    """

    def introspect(self, token: str) -> IntrospectionResult: ...


class LaunchPatientBinder:
    """Verifies a request's ``patient_id`` against the token's launch patient."""

    def __init__(self, *, introspector: _Introspector, client: OpenEmrClient) -> None:
        self._introspector = introspector
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings, introspector: _Introspector) -> LaunchPatientBinder:
        """Build a production binder: the shared introspector plus a REST client."""
        return cls(introspector=introspector, client=OpenEmrClient.from_settings(settings))

    def verify(self, token: str, patient_id: int) -> None:
        """Raise ``LaunchPatientMismatchError`` unless ``patient_id`` maps to the
        token's launch patient. A token without launch context is a no-op (the
        caller falls back to the P2.16 conversation-pid binding)."""
        launch_uuid = self._introspector.introspect(token).patient
        if launch_uuid is None:
            return  # no launch context -> fall back to the P2.16 binding
        resolved_uuid = self._resolve_uuid(token, patient_id)
        if resolved_uuid is None or resolved_uuid.casefold() != launch_uuid.casefold():
            # Fail-safe: mismatch, pid-not-found, or an unresolvable read all
            # reject. Log-safe message -- no token, pid, or UUID.
            raise LaunchPatientMismatchError(
                "request patient_id does not match the token launch context"
            )

    def _resolve_uuid(self, token: str, patient_id: int) -> str | None:
        """Resolve the internal ``patient_id`` (pid) to its FHIR UUID via the
        REST patient roster, or ``None`` if it cannot be resolved."""
        try:
            payload = self._client.get_rest("patient", token=token)
        except OpenEmrApiError:
            return None  # fail-safe: cannot resolve -> caller rejects
        records = payload.get("data") if isinstance(payload, dict) else None
        for record in records or []:
            if isinstance(record, dict) and record.get("pid") == patient_id:
                uuid = record.get("uuid")
                return uuid if isinstance(uuid, str) and uuid else None
        return None

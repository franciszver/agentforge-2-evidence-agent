"""Authenticated REST + FHIR client for the OpenEMR API.

Scope (P2.2): the data-access layer tools (P2.3+) will call through. This
module makes no clinical decisions and defines no tool logic — it is
purely "given a user bearer token and a path, make the call and hand back
parsed JSON, or raise a typed error."

Design notes:
  * Synchronous, matching ``app.openemr_auth``'s injectable-``httpx.Client``
    pattern: the client is always passed in, so tests drive it with
    ``httpx.MockTransport`` and no real network is touched. FastAPI runs
    sync path operations/dependencies in a worker thread pool automatically,
    so a sync client here does not block the event loop; callers that need
    async can wrap invocations with ``anyio.to_thread.run_sync``.
  * URL-join hardening (P0.5 carry-forward): ``path`` is always treated as
    relative to the fixed REST/FHIR base under the *configured* host. It is
    validated against a character allowlist (rejecting backslashes, a
    leading ``@``, percent-encoding, control characters, and anything but a
    conservative path alphabet), rejected if it carries a URL scheme or
    authority (absolute / protocol-relative ``//host``), and normalized so
    ``..`` traversal that would climb above the API base is rejected — all
    before any request is made. The final URL is assembled from the
    configured scheme + host, never from anything in ``path``. See
    ``_validate_path`` and ``_build_url``.
  * ``OpenEmrApiError`` messages are log-safe: never the bearer token, never
    a raw response body (which may contain PHI) — only a fixed operation
    label, the error category, and (where relevant) the HTTP status code.
"""

from __future__ import annotations

import posixpath
import re
from enum import Enum
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.config import Settings

_REST_BASE = "/apis/default/api"
_FHIR_BASE = "/apis/default/fhir"

# Conservative allowlist for a relative API path (leading slashes already
# stripped). Permitting only unreserved URL path characters plus ``/`` closes
# the class of authority-injection tricks at the character level — backslash,
# ``@``, ``%``-encoding, and control characters are all excluded by
# construction, so new variants do not each need a bespoke rejection branch.
_ALLOWED_PATH = re.compile(r"[A-Za-z0-9._~/-]*")


class ErrorCategory(str, Enum):
    """Closed set of error categories callers can branch on."""

    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    INVALID_PATH = "invalid_path"
    UNEXPECTED = "unexpected"


class OpenEmrApiError(Exception):
    """Raised when an OpenEMR API request fails.

    ``category`` lets callers branch without parsing the message. The
    message itself is log-safe: it never embeds the bearer token or a raw
    response body.
    """

    def __init__(self, category: ErrorCategory, message: str) -> None:
        self.category = category
        super().__init__(message)


class OpenEmrClient:
    """Authenticated REST + FHIR client for the OpenEMR API.

    Args:
        base_url: Origin of the OpenEMR instance, e.g. ``"https://openemr"``
            (scheme + host; any path/query/fragment is ignored).
        client: An injectable ``httpx.Client`` — hermetic tests inject one
            backed by ``httpx.MockTransport``; production injects one via
            :meth:`from_settings`.
    """

    def __init__(self, *, base_url: str, client: httpx.Client) -> None:
        parsed = urlsplit(base_url.rstrip("/"))
        self._scheme = parsed.scheme
        self._host = parsed.netloc
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenEmrClient:
        """Build a production client, threading base URL, TLS, and timeout.

        Single config entry point: ``verify_ssl`` and the request timeout are
        applied to a fresh ``httpx.Client``, and ``openemr_base_url`` sets the
        origin — so every call site wires config the same way rather than
        assembling the pieces by hand. Hermetic tests bypass this and inject
        their own ``httpx.MockTransport``-backed client via ``__init__``.
        """
        client = httpx.Client(
            verify=settings.openemr_verify_ssl,
            timeout=settings.openemr_api_timeout_seconds,
        )
        return cls(base_url=settings.openemr_base_url, client=client)

    def get_rest(
        self,
        path: str,
        *,
        token: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """GET a path under the REST API base (``/apis/default/api``)."""
        return self._get(_REST_BASE, path, token=token, params=params)

    def get_fhir(
        self,
        path: str,
        *,
        token: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """GET a path under the FHIR API base (``/apis/default/fhir``)."""
        return self._get(_FHIR_BASE, path, token=token, params=params)

    def _get(
        self,
        api_base: str,
        path: str,
        *,
        token: str,
        params: dict[str, str] | None,
    ) -> dict[str, Any] | list[Any]:
        url = self._build_url(api_base, path)
        try:
            response = self._client.get(url, headers={"Authorization": f"Bearer {token}"}, params=params)
        except httpx.TimeoutException as exc:
            raise OpenEmrApiError(ErrorCategory.TIMEOUT, "OpenEMR API request timed out") from exc
        except httpx.HTTPError as exc:
            raise OpenEmrApiError(ErrorCategory.UNEXPECTED, "OpenEMR API request failed") from exc

        if response.status_code == 401:
            raise OpenEmrApiError(ErrorCategory.UNAUTHORIZED, "OpenEMR API request unauthorized (401)")
        if response.status_code == 403:
            raise OpenEmrApiError(ErrorCategory.FORBIDDEN, "OpenEMR API request forbidden (403)")
        if response.status_code == 404:
            raise OpenEmrApiError(ErrorCategory.NOT_FOUND, "OpenEMR API resource not found (404)")
        if not response.is_success:
            raise OpenEmrApiError(
                ErrorCategory.UNEXPECTED,
                f"OpenEMR API request failed (status {response.status_code})",
            )

        try:
            return response.json()
        except ValueError as exc:
            raise OpenEmrApiError(ErrorCategory.UNEXPECTED, "OpenEMR API response was not valid JSON") from exc

    def _build_url(self, api_base: str, path: str) -> str:
        """Join ``api_base`` + ``path`` under the configured host, or raise.

        ``path`` is validated first, then normalized with ``posixpath`` and
        checked to still be under ``api_base`` (rejects ``..`` traversal). The
        final URL is built from the *configured* scheme/host — never from
        anything in ``path`` — so the request host is fixed by construction.
        """
        relative = self._validate_path(path)

        combined = posixpath.normpath(posixpath.join(api_base, relative))
        if combined != api_base and not combined.startswith(api_base + "/"):
            raise OpenEmrApiError(
                ErrorCategory.INVALID_PATH,
                "OpenEMR API request path rejected: escapes API base",
            )

        return urlunsplit((self._scheme, self._host, combined, "", ""))

    @staticmethod
    def _validate_path(path: str) -> str:
        """Validate a relative API path; return it with leading slashes trimmed.

        Rejects any path that carries a URL scheme or authority (absolute or
        protocol-relative ``//host``) or that contains a character outside the
        conservative path allowlist.
        """
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc:
            raise OpenEmrApiError(
                ErrorCategory.INVALID_PATH,
                "OpenEMR API request path rejected: absolute or protocol-relative URL",
            )

        relative = path.lstrip("/")
        if _ALLOWED_PATH.fullmatch(relative) is None:
            raise OpenEmrApiError(
                ErrorCategory.INVALID_PATH,
                "OpenEMR API request path rejected: disallowed characters",
            )
        return relative

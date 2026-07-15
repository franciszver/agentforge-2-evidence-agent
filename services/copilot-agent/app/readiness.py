"""Dependency readiness checks for the copilot-agent service.

The HTTP checks (OpenEMR, Ollama) accept an injected ``httpx.AsyncClient``
so tests can substitute an ``httpx.MockTransport``-backed client and avoid
real network calls. The trace-store check performs a real write/read
against a SQLite database at the configured path, not just a file-exists
check.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import Depends

from app.config import Settings, get_settings

# FHIR CapabilityStatement endpoint: unauthenticated on a stock OpenEMR
# instance, so it doubles as a lightweight reachability probe.
OPENEMR_READY_PATH = "/apis/default/fhir/metadata"
OLLAMA_VERSION_PATH = "/api/version"
HTTP_CHECK_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single dependency readiness check."""

    ok: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregate readiness across all dependency checks."""

    ready: bool
    checks: dict[str, CheckResult]


async def check_openemr(settings: Settings, client: httpx.AsyncClient) -> CheckResult:
    """Check that the OpenEMR FHIR capability endpoint is reachable."""
    url = f"{settings.openemr_base_url}{OPENEMR_READY_PATH}"
    try:
        response = await client.get(url)
    except httpx.HTTPError:
        return CheckResult(ok=False, detail="unreachable")
    if response.is_success:
        return CheckResult(ok=True, detail="reachable")
    return CheckResult(ok=False, detail=f"unexpected status {response.status_code}")


async def check_ollama(settings: Settings, client: httpx.AsyncClient) -> CheckResult:
    """Check that the Ollama API is reachable."""
    url = f"{settings.ollama_base_url}{OLLAMA_VERSION_PATH}"
    try:
        response = await client.get(url)
    except httpx.HTTPError:
        return CheckResult(ok=False, detail="unreachable")
    if response.is_success:
        return CheckResult(ok=True, detail="reachable")
    return CheckResult(ok=False, detail=f"unexpected status {response.status_code}")


def check_trace_store(db_path: str) -> CheckResult:
    """Check that the trace-store SQLite database is writable.

    Performs a real write/read/delete against a throwaway probe table.
    On failure, only the exception class name is returned as the detail
    so the response body never leaks file paths or driver internals.
    """
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS _readiness_probe (id INTEGER PRIMARY KEY, probe TEXT)"
            )
            connection.execute("INSERT INTO _readiness_probe (probe) VALUES ('ok')")
            cursor = connection.execute("SELECT probe FROM _readiness_probe WHERE probe = 'ok'")
            row = cursor.fetchone()
            connection.execute("DELETE FROM _readiness_probe WHERE probe = 'ok'")
            connection.commit()
        finally:
            connection.close()
    except Exception as exc:
        return CheckResult(ok=False, detail=type(exc).__name__)
    if row is None:
        return CheckResult(ok=False, detail="write verification failed")
    return CheckResult(ok=True, detail="writable")


async def get_openemr_client(
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[httpx.AsyncClient]:
    """Production dependency: real OpenEMR HTTP client, closed after the request."""
    async with httpx.AsyncClient(
        verify=settings.openemr_verify_ssl, timeout=HTTP_CHECK_TIMEOUT_SECONDS
    ) as client:
        yield client


async def get_ollama_client(
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[httpx.AsyncClient]:
    """Production dependency: real Ollama HTTP client, closed after the request."""
    async with httpx.AsyncClient(timeout=HTTP_CHECK_TIMEOUT_SECONDS) as client:
        yield client


async def compute_readiness(
    settings: Settings = Depends(get_settings),
    openemr_client: httpx.AsyncClient = Depends(get_openemr_client),
    ollama_client: httpx.AsyncClient = Depends(get_ollama_client),
) -> ReadinessReport:
    """Aggregate all dependency checks into a single readiness report."""
    checks = {
        "openemr": await check_openemr(settings, openemr_client),
        "ollama": await check_ollama(settings, ollama_client),
        "trace_store": check_trace_store(settings.trace_db_path),
    }
    return ReadinessReport(ready=all(result.ok for result in checks.values()), checks=checks)

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
from pydantic import BaseModel

from app.chat import _wants_llama_server
from app.config import Settings, get_settings

# FHIR CapabilityStatement endpoint: unauthenticated on a stock OpenEMR
# instance, so it doubles as a lightweight reachability probe.
OPENEMR_READY_PATH = "/apis/default/fhir/metadata"
OLLAMA_VERSION_PATH = "/api/version"
# llama-server (llama.cpp) built-in health endpoint -- see the
# `llama-server` service's Docker healthcheck in
# docker/development-easy/docker-compose.copilot.yml.
LLAMA_SERVER_HEALTH_PATH = "/health"
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


class ReadyCheckBody(BaseModel):
    """OpenAPI-facing shape of a single ``GET /ready`` check entry.

    Declared separately from ``CheckResult`` (a plain dataclass): this model
    exists only so ``app.main``'s ``responses=`` route metadata can document
    the real ``/ready`` body shape for BOTH the 200 and 503 cases -- the
    handler still builds the body by hand (it returns a ``JSONResponse``,
    not this model, since the status code varies), so this is spec-accuracy
    only and changes no runtime behavior.
    """

    ok: bool
    detail: str


class ReadyResponseBody(BaseModel):
    """OpenAPI-facing shape of the full ``GET /ready`` response body."""

    status: str
    checks: dict[str, ReadyCheckBody]


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


async def check_llama_server(settings: Settings, client: httpx.AsyncClient) -> CheckResult:
    """Check that the llama-server engine is reachable.

    Only wired into ``compute_readiness`` when ``settings.copilot_llm_engine
    == "llama_server"`` (see ``_wants_llama_server``) -- Ollama is checked
    unconditionally regardless of the engine flag, since embeddings and
    vision-based document-ingestion extraction always use it (see
    ``app.config.Settings.copilot_llm_engine``'s docstring).
    """
    url = f"{settings.llama_server_base_url}{LLAMA_SERVER_HEALTH_PATH}"
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


async def get_llama_server_client(
    settings: Settings = Depends(get_settings),
) -> AsyncIterator[httpx.AsyncClient]:
    """Production dependency: real llama-server HTTP client, closed after the request."""
    async with httpx.AsyncClient(timeout=HTTP_CHECK_TIMEOUT_SECONDS) as client:
        yield client


async def compute_readiness(
    settings: Settings = Depends(get_settings),
    openemr_client: httpx.AsyncClient = Depends(get_openemr_client),
    ollama_client: httpx.AsyncClient = Depends(get_ollama_client),
    llama_server_client: httpx.AsyncClient = Depends(get_llama_server_client),
) -> ReadinessReport:
    """Aggregate all dependency checks into a single readiness report.

    Ollama and the trace store are checked unconditionally. llama-server is
    checked only when ``copilot_llm_engine`` actually selects it -- when the
    engine is "ollama" (default), nothing in the request path ever talks to
    llama-server, so probing it would report a fake outage for a dependency
    that isn't in use.
    """
    checks = {
        "openemr": await check_openemr(settings, openemr_client),
        "ollama": await check_ollama(settings, ollama_client),
    }
    if _wants_llama_server(settings):
        checks["llama_server"] = await check_llama_server(settings, llama_server_client)
    checks["trace_store"] = check_trace_store(settings.trace_db_path)
    return ReadinessReport(ready=all(result.ok for result in checks.values()), checks=checks)

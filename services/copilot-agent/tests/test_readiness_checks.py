"""Unit tests for individual readiness check functions, in isolation."""

import asyncio

import httpx

from app.config import Settings
from app.readiness import check_ollama, check_openemr, check_trace_store


def _run_check(handler, check_fn, settings: Settings):
    async def _run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_fn(settings, client)

    return asyncio.run(_run())


def test_check_openemr_ok_on_2xx_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resourceType": "CapabilityStatement"})

    settings = Settings(openemr_base_url="https://openemr")
    result = _run_check(handler, check_openemr, settings)

    assert result.ok is True


def test_check_openemr_not_ok_on_non_2xx_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    settings = Settings(openemr_base_url="https://openemr")
    result = _run_check(handler, check_openemr, settings)

    assert result.ok is False


def test_check_openemr_not_ok_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    settings = Settings(openemr_base_url="https://openemr")
    result = _run_check(handler, check_openemr, settings)

    assert result.ok is False


def test_check_ollama_ok_on_2xx_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"version": "0.1.0"})

    settings = Settings(ollama_base_url="http://ollama:11434")
    result = _run_check(handler, check_ollama, settings)

    assert result.ok is True


def test_check_ollama_not_ok_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    settings = Settings(ollama_base_url="http://ollama:11434")
    result = _run_check(handler, check_ollama, settings)

    assert result.ok is False


def test_check_trace_store_ok_when_writable(tmp_path):
    db_path = tmp_path / "nested" / "traces.db"

    result = check_trace_store(str(db_path))

    assert result.ok is True


def test_check_trace_store_not_ok_when_path_invalid(tmp_path):
    # A path that tries to use a file as a parent directory is invalid.
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("x")
    db_path = blocking_file / "traces.db"

    result = check_trace_store(str(db_path))

    assert result.ok is False
    # Detail must not leak the raw path/error message, only the error class name.
    assert str(db_path) not in result.detail

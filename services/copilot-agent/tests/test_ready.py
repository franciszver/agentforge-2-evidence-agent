"""Tests for the GET /ready endpoint, driven with fake dependencies (no real network)."""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.readiness import get_ollama_client, get_openemr_client

client = TestClient(app)


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200)


def _down_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _fake_client_dependency(handler):
    async def _override():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as fake_client:
            yield fake_client

    return _override


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health_still_returns_200_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_when_all_dependencies_healthy(tmp_path):
    app.dependency_overrides[get_openemr_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_ollama_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_settings] = lambda: Settings(
        trace_db_path=str(tmp_path / "traces.db")
    )

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["openemr"]["ok"] is True
    assert body["checks"]["ollama"]["ok"] is True
    assert body["checks"]["trace_store"]["ok"] is True


def test_ready_returns_503_when_openemr_down(tmp_path):
    app.dependency_overrides[get_openemr_client] = _fake_client_dependency(_down_handler)
    app.dependency_overrides[get_ollama_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_settings] = lambda: Settings(
        trace_db_path=str(tmp_path / "traces.db")
    )

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["openemr"]["ok"] is False
    assert body["checks"]["ollama"]["ok"] is True
    assert body["checks"]["trace_store"]["ok"] is True


def test_ready_returns_503_when_ollama_down(tmp_path):
    app.dependency_overrides[get_openemr_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_ollama_client] = _fake_client_dependency(_down_handler)
    app.dependency_overrides[get_settings] = lambda: Settings(
        trace_db_path=str(tmp_path / "traces.db")
    )

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["ollama"]["ok"] is False


def test_ready_returns_503_when_trace_store_not_writable(tmp_path):
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("x")
    bad_db_path = blocking_file / "traces.db"

    app.dependency_overrides[get_openemr_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_ollama_client] = _fake_client_dependency(_ok_handler)
    app.dependency_overrides[get_settings] = lambda: Settings(trace_db_path=str(bad_db_path))

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["trace_store"]["ok"] is False

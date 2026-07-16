"""Browser viewport tests for ``GET /dashboard`` (Playwright, real browser).

Follows ``tests/test_chat_shell_browser.py`` (P0.6 decision): agent-served
pages are exercised with Playwright, not Panther/Selenium -- the internal
docker network the Selenium grid drives from cannot reach the agent
container directly.

The app runs as a real subprocess (not the in-process ``TestClient``), so the
autouse trace-store isolation fixture in ``tests/conftest.py`` does not apply
here (dependency overrides are in-process only). Isolation instead comes from
pointing the subprocess's ``TRACE_DB_PATH`` env var at a ``tmp_path`` file --
never the configured default (``/data/traces.db``).
"""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import sync_playwright

from app.trace_store import TraceStore

pytestmark = pytest.mark.browser


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def app_base_url(tmp_path: Path) -> Iterator[str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    # Pre-seed a few real spans so the dashboard renders non-zero values --
    # written via TraceStore's public API BEFORE the subprocess starts, to
    # the same file path the subprocess will read (sqlite is file-based, so
    # this cross-process handoff is safe).
    db_path = tmp_path / "traces.db"
    seed_store = TraceStore(db_path=str(db_path), hash_secret=secrets.token_hex(16))
    seed_store.record_request_span(correlation_id="seed-1", start_ts=0.0, end_ts=0.2, ok=True)
    seed_store.record_request_span(correlation_id="seed-2", start_ts=0.0, end_ts=0.1, ok=True)

    env = dict(os.environ)
    env["TRACE_DB_PATH"] = str(db_path)

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
    )

    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("app server did not become healthy in time")

        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


def _assert_dashboard_visible_and_no_overflow(page) -> None:
    for label in ("Requests", "Error rate", "p50", "p95", "Tool calls", "Verification pass rate"):
        assert page.get_by_text(label, exact=False).first.is_visible()

    overflow_ok = page.evaluate("document.body.scrollWidth <= window.innerWidth + 1")
    assert overflow_ok


def test_dashboard_mobile_viewport_renders_real_data(app_base_url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 360, "height": 800})
            page.goto(f"{app_base_url}/dashboard")
            _assert_dashboard_visible_and_no_overflow(page)
            # 2 seeded request spans -> request_count == 2 rendered somewhere.
            assert page.get_by_text("2", exact=True).first.is_visible()
        finally:
            browser.close()


def test_dashboard_desktop_viewport(app_base_url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"{app_base_url}/dashboard")
            _assert_dashboard_visible_and_no_overflow(page)
        finally:
            browser.close()

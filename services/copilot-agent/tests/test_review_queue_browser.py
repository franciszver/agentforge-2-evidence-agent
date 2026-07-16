"""Browser scenario for the P4.9 review queue + promote-to-eval affordance
(Playwright, real browser).

Follows ``tests/test_dashboard_browser.py`` (P0.6 decision): the review
queue is agent-served, not routed through the OpenEMR module, so it is
exercised with Playwright against a real subprocess -- not Panther/Selenium
(which cannot reach the agent container directly) and not the in-process
``TestClient`` (whose dependency overrides don't apply across a subprocess
boundary; isolation instead comes from ``TRACE_DB_PATH`` pointing at a
``tmp_path`` file).

Proves the DoD's timing claim end-to-end: seed a real thumbs-down span, load
``/review``, click the promote button, and read back a valid regression-case
YAML from the page -- all inside one browser session, standing in for "click
-> valid case in under 60 seconds".
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

from app.trace_store import FeedbackThumb, TraceStore

pytestmark = pytest.mark.browser

_SEEDED_CORRELATION_ID = "browser-seed-corr-1"
_SEEDED_COMMENT = "cited a lab value that was never in the record"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def app_base_url(tmp_path: Path) -> Iterator[str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    db_path = tmp_path / "traces.db"
    seed_store = TraceStore(db_path=str(db_path), hash_secret=secrets.token_hex(16))
    seed_store.record_request_span(
        correlation_id=_SEEDED_CORRELATION_ID, start_ts=0.0, end_ts=0.2, ok=True
    )
    seed_store.record_feedback_span(
        correlation_id=_SEEDED_CORRELATION_ID,
        start_ts=0.2,
        end_ts=0.2,
        feedback_thumb=FeedbackThumb.DOWN,
        feedback_comment=_SEEDED_COMMENT,
    )

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


def test_promote_click_produces_a_valid_regression_case_yaml(app_base_url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"{app_base_url}/review")

            assert page.get_by_text(_SEEDED_CORRELATION_ID, exact=False).first.is_visible()
            assert page.get_by_text(_SEEDED_COMMENT, exact=False).first.is_visible()

            promote_button = page.locator(
                f'[data-testid="promote-button"][data-correlation-id="{_SEEDED_CORRELATION_ID}"]'
            )
            assert promote_button.is_visible()

            start = time.monotonic()
            promote_button.click()

            output = page.locator('[data-testid="promote-output"]')
            output.wait_for(state="visible", timeout=5000)
            page.wait_for_function(
                "document.querySelector('[data-testid=\"promote-output\"]').textContent.length > 0"
            )
            elapsed_seconds = time.monotonic() - start

            yaml_text = output.text_content() or ""
            assert "category: regression" in yaml_text
            assert _SEEDED_CORRELATION_ID in yaml_text
            assert _SEEDED_COMMENT in yaml_text
            assert elapsed_seconds < 60
        finally:
            browser.close()


def test_review_mobile_viewport_renders_without_overflow(app_base_url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 360, "height": 800})
            page.goto(f"{app_base_url}/review")
            assert page.get_by_text(_SEEDED_CORRELATION_ID, exact=False).first.is_visible()
            overflow_ok = page.evaluate("document.body.scrollWidth <= window.innerWidth + 1")
            assert overflow_ok
        finally:
            browser.close()

"""Browser viewport tests for the static chat shell (Playwright, real browser)."""

import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def app_base_url() -> Iterator[str]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
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


def _assert_shell_visible_and_no_overflow(page) -> None:
    assert page.locator("[data-testid='chat-stream']").is_visible()
    assert page.locator("[data-testid='chat-input']").is_visible()
    assert page.locator("[data-testid='chat-send']").is_visible()

    overflow_ok = page.evaluate("document.body.scrollWidth <= window.innerWidth + 1")
    assert overflow_ok


def test_chat_shell_mobile_viewport(app_base_url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 360, "height": 800})
            page.goto(f"{app_base_url}/chat")
            _assert_shell_visible_and_no_overflow(page)
        finally:
            browser.close()


def test_chat_shell_desktop_viewport(app_base_url: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"{app_base_url}/chat")
            _assert_shell_visible_and_no_overflow(page)
        finally:
            browser.close()

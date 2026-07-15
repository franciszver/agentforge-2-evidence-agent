"""Hermetic HTML-contract test for the static chat shell (no browser)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_chat_shell_returns_200_html():
    response = client.get("/chat")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_chat_shell_contains_required_elements():
    response = client.get("/chat")
    body = response.text

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in body
    assert 'data-testid="chat-stream"' in body
    assert 'data-testid="chat-input"' in body
    assert 'data-testid="chat-send"' in body

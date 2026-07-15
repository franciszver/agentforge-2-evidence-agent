"""Hermetic tests for the Ollama chat + constrained-extraction client.

All HTTP is served by ``httpx.MockTransport`` so the suite never touches the
network. Live end-to-end verification against the real qwen3:4b model is a
separate ``@pytest.mark.integration`` test, run manually against a proxied
dev-stack Ollama.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.ollama_client import OllamaClient, OllamaError


class _Animal(BaseModel):
    """Small test-only schema for constrained extraction."""

    name: str
    legs: int


def _client(handler, **kwargs) -> OllamaClient:
    return OllamaClient(
        base_url="http://ollama:11434",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def _ndjson(*chunks: dict[str, object]) -> bytes:
    return b"\n".join(json.dumps(chunk).encode() for chunk in chunks) + b"\n"


# --- chat: streaming assembly -----------------------------------------------


def test_chat_assembles_multi_chunk_ndjson_stream_into_full_content():
    body = _ndjson(
        {"message": {"role": "assistant", "content": "Hello"}, "done": False},
        {"message": {"role": "assistant", "content": ", "}, "done": False},
        {"message": {"role": "assistant", "content": "world."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    client = _client(handler)
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "Hello, world."


def test_chat_sends_think_false_stream_true_temperature_zero_and_model():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": "ok"}, "done": True}),
        )

    client = _client(handler, model="qwen3:4b")
    client.chat([{"role": "user", "content": "hi"}])

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["think"] is False
    assert body["stream"] is True
    assert body["model"] == "qwen3:4b"
    assert body["options"]["temperature"] == 0


def test_chat_strips_leaked_thinking_preamble_from_content():
    """Defense against an observed Ollama/qwen3 quirk: even with ``think:
    false``, some Ollama versions still emit the model's reasoning inline in
    ``message.content`` (terminated by a stray ``</think>`` marker) instead
    of suppressing it. The client must return only the real answer.
    """
    body = _ndjson(
        {"message": {"role": "assistant", "content": "some leaked reasoning"}, "done": False},
        {"message": {"role": "assistant", "content": "</think>"}, "done": False},
        {"message": {"role": "assistant", "content": "\n\nhello"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler)
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "hello"


def test_chat_strips_properly_paired_think_tags_too():
    body = _ndjson(
        {"message": {"role": "assistant", "content": "<think>reasoning</think>\n\nhello"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler)
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "hello"


# --- extract: happy path ------------------------------------------------


def test_extract_happy_path_parses_valid_json_into_schema():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson(
                {
                    "message": {"role": "assistant", "content": json.dumps({"name": "dog", "legs": 4})},
                    "done": True,
                }
            ),
        )

    client = _client(handler)
    result = client.extract([{"role": "user", "content": "describe a dog"}], _Animal)

    assert isinstance(result, _Animal)
    assert result.name == "dog"
    assert result.legs == 4

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["format"] == _Animal.model_json_schema()
    assert body["options"]["temperature"] == 0


# --- extract: malformed output retry path -----------------------------------


def test_extract_retries_once_on_malformed_json_then_succeeds():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            content = "not valid json {{{"
        else:
            content = json.dumps({"name": "cat", "legs": 4})
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": content}, "done": True}),
        )

    client = _client(handler, max_retries=2)
    result = client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    assert result.name == "cat"
    assert call_count == 2


def test_extract_raises_ollama_error_after_exhausting_retries_without_leaking_raw_output():
    secret_output = "leaked-phi-like-token-xyz"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"role": "assistant", "content": f"garbage {secret_output}"}, "done": True}
            ),
        )

    client = _client(handler, max_retries=2)
    with pytest.raises(OllamaError) as excinfo:
        client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    message = str(excinfo.value)
    assert secret_output not in message


def test_extract_retries_on_valid_json_that_fails_schema_validation():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # Valid JSON, but "legs" is missing on the first attempt -- fails
        # model_validate even though json.loads succeeds.
        if call_count == 1:
            content = json.dumps({"name": "dog"})
        else:
            content = json.dumps({"name": "dog", "legs": 4})
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": content}, "done": True}),
        )

    client = _client(handler, max_retries=2)
    result = client.extract([{"role": "user", "content": "describe a dog"}], _Animal)

    assert result.legs == 4
    assert call_count == 2


# --- error mapping -----------------------------------------------------


def test_chat_maps_http_500_to_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal stack trace detail")

    client = _client(handler)
    with pytest.raises(OllamaError) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])

    assert "internal stack trace detail" not in str(excinfo.value)


def test_chat_maps_connection_error_to_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_maps_timeout_to_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = _client(handler)
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "hi"}])


def test_from_settings_builds_client_targeting_configured_base_url_and_model():
    settings = Settings(ollama_base_url="http://ollama.example:11434")

    client = OllamaClient.from_settings(settings)

    assert client._base_url == "http://ollama.example:11434"


# --- live integration: real qwen3:4b -----------------------------------
#
# Ollama is internal-only on the dev stack's docker network (no host port
# published). These tests require a bridge -- e.g. a disposable socat proxy
# container publishing the internal ollama service to the host -- pointed to
# via OLLAMA_BASE_URL. Skipped by default (``pytest -m "not integration"``).


@pytest.mark.integration
def test_live_chat_against_real_qwen3_returns_non_thinking_text():
    """think:false must suppress qwen3:4b's default <think>...</think> preamble."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    settings = Settings(ollama_base_url=base_url, ollama_api_timeout_seconds=120.0)
    client = OllamaClient.from_settings(settings)

    result = client.chat([{"role": "user", "content": "Reply with exactly the word: hello"}])

    assert result.strip() != ""
    assert "<think>" not in result
    assert "</think>" not in result


@pytest.mark.integration
def test_live_extract_against_real_qwen3_returns_valid_schema():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    settings = Settings(ollama_base_url=base_url, ollama_api_timeout_seconds=120.0)
    client = OllamaClient.from_settings(settings)

    result = client.extract(
        "Describe a common four-legged pet as JSON with its name and number of legs.",
        _Animal,
    )

    assert isinstance(result, _Animal)
    assert result.legs > 0

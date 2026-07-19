"""Hermetic contract tests for the llama-server chat + constrained-extraction
client (P3.10a -- migrating the answer/extract/reranker roles from Ollama to
a local llama.cpp ``llama-server``, epic #52).

All HTTP is served by ``httpx.MockTransport``; no real network or GPU is
touched. Mirrors ``tests/test_ollama_client.py``'s structure and asserts on
the specific behaviors already vetted in the throwaway measurement harness's
``LlamaServerClient`` (see the P3.10a issue body): a hard ``max_tokens: 1536``
cap on every call, JSON-schema-constrained ``response_format`` on ``extract``,
and defensive ``<think>``-leak stripping.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from app.llama_server_client import LlamaServerClient, LlamaServerError


class _Animal(BaseModel):
    """Small test-only schema for constrained extraction."""

    name: str
    legs: int


def _client(handler, **kwargs) -> LlamaServerClient:
    return LlamaServerClient(
        base_url="http://llama-server:8080",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def _completion(content: str) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    ).encode()


# --- chat -------------------------------------------------------------------


def test_chat_posts_to_openai_compatible_endpoint_with_max_tokens_cap():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_completion("ok"))

    client = _client(handler, model="qwen3-8b")
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert captured["url"] == "http://llama-server:8080/v1/chat/completions"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen3-8b"
    assert body["max_tokens"] == 1536
    assert body["temperature"] == 0
    assert body["stream"] is False


def test_chat_strips_leaked_thinking_preamble_from_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_completion("some leaked reasoning</think>\n\nhello"))

    client = _client(handler)
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "hello"


# --- extract ------------------------------------------------------------


def test_extract_sends_strict_json_schema_response_format_and_max_tokens_cap():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_completion(json.dumps({"name": "dog", "legs": 4})))

    client = _client(handler)
    result = client.extract([{"role": "user", "content": "describe a dog"}], _Animal)

    assert isinstance(result, _Animal)
    assert result.name == "dog"
    assert result.legs == 4

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["max_tokens"] == 1536
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert json_schema["name"] == _Animal.__name__
    assert json_schema["schema"] == _Animal.model_json_schema()
    assert json_schema["strict"] is True


def test_extract_strips_leaked_thinking_preamble_before_parsing_json():
    content = "leaked reasoning</think>\n\n" + json.dumps({"name": "cat", "legs": 4})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_completion(content))

    client = _client(handler)
    result = client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    assert result.name == "cat"
    assert result.legs == 4


def test_extract_retries_on_malformed_json_then_succeeds():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        content = "not valid json {{{" if call_count == 1 else json.dumps({"name": "cat", "legs": 4})
        return httpx.Response(200, content=_completion(content))

    client = _client(handler, max_retries=2)
    result = client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    assert result.name == "cat"
    assert call_count == 2


def test_extract_raises_after_exhausting_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_completion("garbage"))

    client = _client(handler, max_retries=2)
    with pytest.raises(LlamaServerError):
        client.extract([{"role": "user", "content": "describe a cat"}], _Animal)


def test_extract_raises_not_implemented_for_images():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_completion("{}"))

    client = _client(handler)
    with pytest.raises(NotImplementedError):
        client.extract([{"role": "user", "content": "describe"}], _Animal, images=["base64data"])


def test_embed_is_not_implemented():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_completion("{}"))

    client = _client(handler)
    with pytest.raises(NotImplementedError):
        client.embed("some text")

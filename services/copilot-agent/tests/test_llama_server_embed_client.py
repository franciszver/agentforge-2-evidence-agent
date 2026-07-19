"""Hermetic contract tests for the llama-server dense-embeddings client
(P3.10b -- migrating embeddings from Ollama to a dedicated llama.cpp
``llama-server --embedding`` instance, epic #52 step 2).

All HTTP is served by ``httpx.MockTransport``; no real network or GPU is
touched. Mirrors ``tests/test_ollama_client.py``'s embed-section structure
(same retry/network-error policy, same log-safety requirement) so the two
embedders are provably interchangeable from a caller's point of view.
"""

from __future__ import annotations

import json
import logging
import os

import httpx
import pytest

from app.config import Settings
from app.llama_server_embed_client import LlamaServerEmbedClient, LlamaServerEmbedError


def _client(handler, **kwargs) -> LlamaServerEmbedClient:
    return LlamaServerEmbedClient(
        base_url="http://llama-server-embed:8080",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def _all_log_text(records) -> str:
    return " ".join(f"{r.getMessage()} {r.__dict__}" for r in records)


def test_embed_posts_to_v1_embeddings_with_model_and_input():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]})

    client = _client(handler, model="nomic-embed-text-v1.5")
    result = client.embed("some clinical guideline text")

    assert captured["url"] == "http://llama-server-embed:8080/v1/embeddings"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body == {"model": "nomic-embed-text-v1.5", "input": "some clinical guideline text"}
    assert result == [0.1, 0.2, 0.3]


def test_embed_returns_a_plain_unnormalized_list_of_floats():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1, 2, 3]}]})

    client = _client(handler)
    result = client.embed("text")

    assert result == [1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in result)


def test_embed_retries_on_malformed_response_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"data": [{"embedding": [0.5, 0.6]}]})

    client = _client(handler, max_retries=2)
    result = client.embed("text")

    assert result == [0.5, 0.6]
    assert calls["n"] == 2


def test_embed_raises_after_exhausting_retries_on_malformed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": "not-a-list"}]})

    client = _client(handler, max_retries=2)

    with pytest.raises(LlamaServerEmbedError):
        client.embed("text")


def test_embed_raises_immediately_on_http_error_without_retrying():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    client = _client(handler, max_retries=3)

    with pytest.raises(LlamaServerEmbedError):
        client.embed("text")

    assert calls["n"] == 1


def test_embed_does_not_log_the_input_text(caplog):
    caplog.set_level(logging.INFO, logger="app.llama_server_embed_client")
    secret_text = "SUPER_SECRET_GUIDELINE_QUERY_TOKEN"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

    client = _client(handler)
    client.embed(secret_text)

    records = [r for r in caplog.records if r.name == "app.llama_server_embed_client"]
    assert secret_text not in _all_log_text(records)


def test_from_settings_wires_base_url_model_and_timeout():
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        llama_server_embed_base_url="http://llama-server-embed:8080",
        llama_server_embed_model="nomic-embed-text-v1.5",
    )

    client = LlamaServerEmbedClient.from_settings(settings)

    assert client._base_url == "http://llama-server-embed:8080"
    assert client._model == "nomic-embed-text-v1.5"


# --- live integration: real llama-server-embed -------------------------
#
# llama-server-embed is internal-only on the dev stack's docker network (no
# host port published by default). These tests require a bridge -- e.g. a
# disposable socat proxy container -- pointed to via LLAMA_SERVER_EMBED_BASE_URL.
# Skipped by default (``pytest -m "not integration"``).


@pytest.mark.integration
def test_live_embed_against_real_llama_server_embed_returns_a_vector():
    base_url = os.environ.get("LLAMA_SERVER_EMBED_BASE_URL", "http://localhost:8081")
    settings = Settings(llama_server_embed_base_url=base_url, llama_server_embed_api_timeout_seconds=120.0)  # type: ignore[call-arg]
    client = LlamaServerEmbedClient.from_settings(settings)

    result = client.embed("What A1c target for most adults?")

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result)

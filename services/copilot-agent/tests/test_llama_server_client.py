"""Hermetic contract tests for the llama-server chat + constrained-extraction
client (P3.10a -- migrating the answer/extract/reranker roles from Ollama to
a local llama.cpp ``llama-server``, epic #52).

All HTTP is served by ``httpx.MockTransport``; no real network or GPU is
touched. Mirrors ``tests/test_ollama_client.py``'s structure and asserts on
the specific behaviors already vetted in the throwaway measurement harness's
``LlamaServerClient`` (see the P3.10a issue body): a hard ``max_tokens`` cap
on every call (1536 on chat/chat_stream, a larger dedicated cap on extract --
see P5.3, issue #85), JSON-schema-constrained ``response_format`` on
``extract``, and defensive ``<think>``-leak stripping.
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


def _completion(content: str, *, finish_reason: str = "stop") -> bytes:
    return json.dumps(
        {
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
            ]
        }
    ).encode()


def _sse(*chunks: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]
    return b"".join(lines) + b"data: [DONE]\n\n"


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


def test_extract_sends_strict_json_schema_response_format_and_its_own_larger_max_tokens_cap():
    """extract() gets a LARGER max_tokens cap than chat()/chat_stream() (P5.3,
    issue #85) -- see ``_EXTRACT_MAX_TOKENS``'s module-level docstring for the
    sizing rationale. ``chat()``'s cap stays 1536 (see the ``test_chat_*``
    tests above), unchanged from the original runaway-generation measurement."""
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
    assert body["max_tokens"] == 2560
    assert body["max_tokens"] > 1536, "extract() must use a LARGER cap than chat()'s 1536"
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


def test_extract_exhausted_retries_error_carries_finish_reason_and_content_length():
    """P5.3 (issue #85): a validation failure's ``LlamaServerError`` message
    carries non-PHI diagnostic signal (``finish_reason``, truncated-content
    length) so this failure class is diagnosable from logs without a live
    re-probe. ``finish_reason: "length"`` here simulates llama.cpp's own
    truncation signal (hit ``max_tokens`` mid-generation) -- the mechanism
    this issue's token-budget fix targets; the message text is log-safe
    (never the raw model output itself, only its length)."""
    truncated = '{"name": "d'  # deliberately invalid/incomplete JSON

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_completion(truncated, finish_reason="length"))

    client = _client(handler, max_retries=2)
    with pytest.raises(LlamaServerError) as exc_info:
        client.extract([{"role": "user", "content": "describe a dog"}], _Animal)

    message = str(exc_info.value)
    assert "length" in message
    assert str(len(truncated)) in message
    assert truncated not in message, "the raw model output must never appear in a LlamaServerError message"


# --- retry-prompt feedback (issue #93 fix 2/4) ------------------------------


def test_extract_retry_appends_specific_feedback_about_invalid_json():
    """A retry after invalid JSON must not be a blind re-roll of the exact
    same prompt -- the next attempt's message list must carry a NEW user
    turn describing what was wrong, so the model has a genuinely different
    (better-informed) chance of succeeding."""
    captured_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        content = "not valid json {{{" if len(captured_bodies) == 1 else json.dumps({"name": "cat", "legs": 4})
        return httpx.Response(200, content=_completion(content))

    client = _client(handler, max_retries=2)
    result = client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    assert result.name == "cat"
    assert len(captured_bodies) == 2
    first_messages = captured_bodies[0]["messages"]
    second_messages = captured_bodies[1]["messages"]
    assert isinstance(first_messages, list)
    assert isinstance(second_messages, list)
    assert len(second_messages) == len(first_messages) + 1, (
        "the retry must add a feedback turn, not resend the identical prompt"
    )
    feedback = second_messages[-1]
    assert feedback["role"] == "user"
    assert "not valid json" in feedback["content"].lower() or "json" in feedback["content"].lower()


def test_extract_retry_feedback_names_the_offending_field_on_schema_violation():
    """A schema-validation failure (valid JSON, but the wrong shape) must
    surface the SPECIFIC field path that failed -- not a generic "try
    again" -- so the retry can actually target the problem."""
    captured_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        # First attempt: missing the required "legs" field (schema violation,
        # not a JSON-parse failure). Second attempt: valid.
        content = json.dumps({"name": "cat"}) if len(captured_bodies) == 1 else json.dumps({"name": "cat", "legs": 4})
        return httpx.Response(200, content=_completion(content))

    client = _client(handler, max_retries=2)
    result = client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    assert result.name == "cat"
    feedback = captured_bodies[1]["messages"][-1]
    assert feedback["role"] == "user"
    assert "legs" in feedback["content"]


def test_extract_retry_feedback_calls_out_truncation_on_length_finish_reason():
    """A ``finish_reason: "length"`` failure (truncated mid-generation) gets
    feedback asking for a MORE CONCISE response next time, distinct from the
    generic invalid-JSON feedback -- concision is the actual fix for
    truncation, re-sending the same verbose request is not."""
    captured_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        if len(captured_bodies) == 1:
            return httpx.Response(200, content=_completion('{"name": "d', finish_reason="length"))
        return httpx.Response(200, content=_completion(json.dumps({"name": "dog", "legs": 4})))

    client = _client(handler, max_retries=2)
    result = client.extract([{"role": "user", "content": "describe a dog"}], _Animal)

    assert result.name == "dog"
    feedback = captured_bodies[1]["messages"][-1]["content"].lower()
    assert "concise" in feedback or "cut off" in feedback or "token limit" in feedback


def test_extract_never_echoes_raw_model_output_in_retry_feedback():
    """The retry feedback message must never embed the model's actual
    (possibly PHI/corpus-bearing) prior output -- only validation metadata,
    matching this module's existing log-safety discipline."""
    secret_marker = "PHI-MARKER-do-not-echo"
    captured_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        content = secret_marker if len(captured_bodies) == 1 else json.dumps({"name": "cat", "legs": 4})
        return httpx.Response(200, content=_completion(content))

    client = _client(handler, max_retries=2)
    client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    feedback = captured_bodies[1]["messages"][-1]["content"]
    assert secret_marker not in feedback


def test_extract_raises_not_implemented_for_images():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_completion("{}"))

    client = _client(handler)
    with pytest.raises(NotImplementedError):
        client.extract([{"role": "user", "content": "describe"}], _Animal, images=["base64data"])


# --- chat_stream --------------------------------------------------------


def test_chat_stream_yields_deltas_and_strips_leaked_thinking():
    body = _sse(
        {"choices": [{"delta": {"content": "leaked reasoning"}}]},
        {"choices": [{"delta": {"content": "</think>"}}]},
        {"choices": [{"delta": {"content": "\n\nhello"}}]},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler)
    deltas = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert "".join(deltas) == "hello"
    assert client.call_stats[-1].ok is True


def test_chat_stream_closes_the_underlying_response_even_on_early_stop():
    """Regression test (code-review finding): ``chat_stream`` opens the
    response with ``stream=True`` (a held-open connection), so it must be
    closed even when the caller stops iterating before the stream ends --
    otherwise the connection leaks back to the pool never released."""
    body = _sse(
        {"choices": [{"delta": {"content": "hello "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
    )
    closed = {"called": False}

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, content=body)
        original_close = response.close

        def _spy_close() -> None:
            closed["called"] = True
            original_close()

        response.close = _spy_close  # type: ignore[method-assign]
        return response

    client = _client(handler)
    generator = client.chat_stream([{"role": "user", "content": "hi"}])
    next(generator)  # consume only the first delta
    generator.close()  # simulate the caller stopping early (or GC)

    assert closed["called"] is True


def test_embed_is_not_implemented():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_completion("{}"))

    client = _client(handler)
    with pytest.raises(NotImplementedError):
        client.embed("some text")

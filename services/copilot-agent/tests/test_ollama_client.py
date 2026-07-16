"""Hermetic tests for the Ollama chat + constrained-extraction client.

All HTTP is served by ``httpx.MockTransport`` so the suite never touches the
network. Live end-to-end verification against the real qwen3:4b model is a
separate ``@pytest.mark.integration`` test, run manually against a proxied
dev-stack Ollama.
"""

from __future__ import annotations

import json
import logging
import os

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.correlation import _STDLIB_RECORD_ATTRS
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


# --- call_stats: per-call token counts + timing (#149 span emission) --------
#
# ``record_llm_span`` (app.trace_store) needs a model name, token counts, and
# timing per Ollama call, but chat()/extract() only ever returned the
# assembled content/model -- nothing surfaced tokens or timing. These tests
# pin the side-channel ``OllamaClient.call_stats`` list every top-level
# chat()/extract() call appends to, which ``app.planner``/``app.extraction``
# read after the fact to build the spans the dashboard aggregates.


def test_chat_records_call_stats_with_tokens_model_and_timing():
    body = _ndjson(
        {"message": {"role": "assistant", "content": "hello"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": 12,
            "eval_count": 7,
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler, model="qwen3:4b")
    client.chat([{"role": "user", "content": "hi"}])

    assert len(client.call_stats) == 1
    stats = client.call_stats[0]
    assert stats.model == "qwen3:4b"
    assert stats.ok is True
    assert stats.tokens_in == 12
    assert stats.tokens_out == 7
    assert stats.end_ts >= stats.start_ts


def test_chat_records_failed_call_stats_on_http_error_with_no_token_counts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal stack trace detail")

    client = _client(handler)
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "hi"}])

    assert len(client.call_stats) == 1
    assert client.call_stats[0].ok is False
    assert client.call_stats[0].tokens_in is None
    assert client.call_stats[0].tokens_out is None


def test_extract_records_call_stats_with_tokens_on_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(
                {
                    "message": {"role": "assistant", "content": json.dumps({"name": "dog", "legs": 4})},
                    "done": True,
                    "prompt_eval_count": 20,
                    "eval_count": 5,
                }
            ),
        )

    client = _client(handler)
    client.extract([{"role": "user", "content": "describe a dog"}], _Animal)

    assert len(client.call_stats) == 1
    assert client.call_stats[0].ok is True
    assert client.call_stats[0].tokens_in == 20
    assert client.call_stats[0].tokens_out == 5


def test_extract_records_one_call_stats_entry_per_attempt_including_failed_retries():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        content = "not valid json {{{" if call_count == 1 else json.dumps({"name": "cat", "legs": 4})
        return httpx.Response(
            200,
            content=_ndjson(
                {
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 3,
                }
            ),
        )

    client = _client(handler, max_retries=2)
    client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    # One call_stats entry per actual Ollama request -- the failed first
    # attempt AND the succeeding second attempt, not just the final one.
    assert len(client.call_stats) == 2
    assert client.call_stats[0].ok is False
    assert client.call_stats[1].ok is True
    assert client.call_stats[1].tokens_in == 10
    assert client.call_stats[1].tokens_out == 3


def test_extract_http_error_propagates_immediately_without_retry():
    # Contract (see extract() docstring): network/HTTP failures are NOT
    # retried -- they propagate immediately. Recording the failed attempt's
    # call_stats must not turn an HTTP error into a retried one.
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(500, text="internal stack trace detail")

    client = _client(handler, max_retries=2)
    with pytest.raises(OllamaError):
        client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    # Exactly ONE upstream request was made (no retry on HTTP error), and it
    # recorded a single failed call_stats entry with no token counts.
    assert call_count == 1
    assert len(client.call_stats) == 1
    assert client.call_stats[0].ok is False
    assert client.call_stats[0].tokens_in is None
    assert client.call_stats[0].tokens_out is None


# --- failure/retry outcome logging (#144) --------------------------------
#
# ``chat()``/``extract()`` already logged call *starts* (see the "ollama
# chat call"/"ollama extract call" info lines above) but never logged
# *outcomes* -- a correlation trace showed a call began, never whether it
# failed or (for extract's malformed-output path) was retried. These pin
# the symmetric failure/retry log lines, and that no PHI/prompt content
# ever lands in them.
#
# Retry contract reminder (see extract()'s docstring): extract() retries
# malformed/invalid JSON and schema-validation failures, but does NOT retry
# HTTP/network errors -- those propagate immediately. So a retry log line
# is expected only on the malformed-output path, never on the HTTP-error
# path.


def _all_log_text(records: list[logging.LogRecord]) -> str:
    """Flatten every message + extra value across records into one string,
    for a single "no PHI/prompt content anywhere" substring check."""
    parts: list[str] = []
    for record in records:
        parts.append(record.getMessage())
        for key, value in vars(record).items():
            if key not in _STDLIB_RECORD_ATTRS:
                parts.append(str(value))
    return " ".join(parts)


def test_chat_logs_failure_outcome_on_http_error(caplog):
    caplog.set_level(logging.INFO, logger="app.ollama_client")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal stack trace detail")

    client = _client(handler)
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "hi"}])

    records = [r for r in caplog.records if r.name == "app.ollama_client"]
    failures = [r for r in records if r.levelno >= logging.WARNING]
    assert failures, "expected a failure outcome log line from chat()"
    failure = failures[0]
    assert failure.error_type == "OllamaError"
    assert "internal stack trace detail" not in _all_log_text(records)


def test_extract_logs_retry_outcome_on_malformed_json_then_succeeds(caplog):
    caplog.set_level(logging.INFO, logger="app.ollama_client")
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        content = "not valid json {{{" if call_count == 1 else json.dumps({"name": "cat", "legs": 4})
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": content}, "done": True}),
        )

    client = _client(handler, max_retries=2)
    result = client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    assert result.name == "cat"
    records = [r for r in caplog.records if r.name == "app.ollama_client"]
    retry_lines = [r for r in records if getattr(r, "attempt", None) == 1 and r.levelno >= logging.WARNING]
    assert retry_lines, "expected a retry outcome log line on the malformed-output retry path"
    assert retry_lines[0].attempt == 1


def test_extract_does_not_log_retry_on_http_error_path(caplog):
    caplog.set_level(logging.INFO, logger="app.ollama_client")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal stack trace detail")

    client = _client(handler, max_retries=2)
    with pytest.raises(OllamaError):
        client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    records = [r for r in caplog.records if r.name == "app.ollama_client"]
    failures = [r for r in records if r.levelno >= logging.WARNING]
    assert failures, "expected a failure outcome log line from extract()'s HTTP-error path"
    assert failures[0].error_type == "OllamaError"
    assert not any("retry" in r.getMessage().lower() for r in failures)
    assert "internal stack trace detail" not in _all_log_text(records)


def test_extract_logs_failure_after_exhausting_retries_without_leaking_output(caplog):
    caplog.set_level(logging.INFO, logger="app.ollama_client")
    secret_output = "leaked-phi-like-token-xyz"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"role": "assistant", "content": f"garbage {secret_output}"}, "done": True}
            ),
        )

    client = _client(handler, max_retries=2)
    with pytest.raises(OllamaError):
        client.extract([{"role": "user", "content": "describe a cat"}], _Animal)

    records = [r for r in caplog.records if r.name == "app.ollama_client"]
    failures = [r for r in records if r.levelno >= logging.WARNING]
    assert len(failures) == 2  # one retry outcome (attempt 1) + one final failure (attempt 2)
    assert failures[-1].attempt == 2
    assert secret_output not in _all_log_text(records)


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

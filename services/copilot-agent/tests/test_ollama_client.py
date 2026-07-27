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
from app.ollama_client import CHAT_MAX_TOKENS, OllamaClient, OllamaError, is_vision_capable_model


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


def test_chat_defaults_num_predict_to_chat_max_tokens_when_caller_omits_it():
    """Issue #167 Gate 3 follow-up: CHAT_MAX_TOKENS is a memory-safety
    ceiling, not a mere default -- but a caller who passes no ``num_predict``
    at all still gets the cap applied (this is what makes it a bound, not
    just a clamp on an explicit override)."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": "ok"}, "done": True}),
        )

    client = _client(handler)
    client.chat([{"role": "user", "content": "hi"}])

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["options"]["num_predict"] == CHAT_MAX_TOKENS


def test_chat_clamps_a_caller_num_predict_above_the_ceiling():
    """A caller asking for MORE than CHAT_MAX_TOKENS is clamped down to it --
    unlike ``temperature``, this option is a retention bound protecting
    process memory, so a caller must not be able to raise it (issue #167
    Gate 3 follow-up)."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": "ok"}, "done": True}),
        )

    client = _client(handler)
    client.chat([{"role": "user", "content": "hi"}], options={"num_predict": CHAT_MAX_TOKENS + 5000})

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["options"]["num_predict"] == CHAT_MAX_TOKENS


def test_chat_respects_a_caller_num_predict_below_the_ceiling():
    """A caller asking for FEWER tokens than CHAT_MAX_TOKENS is honoured
    as-is -- the ceiling only ever clamps DOWN, never up (issue #167 Gate 3
    follow-up)."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": "ok"}, "done": True}),
        )

    client = _client(handler)
    client.chat([{"role": "user", "content": "hi"}], options={"num_predict": 42})

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["options"]["num_predict"] == 42


@pytest.mark.parametrize(
    "sentinel",
    [
        pytest.param(-1, id="ollama-unlimited-sentinel"),
        pytest.param(-2, id="ollama-fill-context-sentinel"),
        pytest.param(0, id="zero-tokens"),
        pytest.param("1000000", id="non-int-string"),
        pytest.param(12.5, id="non-int-float"),
    ],
)
def test_chat_falls_back_to_cap_for_out_of_range_or_non_int_num_predict(sentinel: object) -> None:
    """Issue #167 Gate 3 re-review finding: a plain ``min(caller, cap)``
    would let Ollama's negative "unlimited" sentinels straight through
    (``-1`` = generate until the model stops on its own; ``-2`` = fill the
    context window -- both numerically SMALLER than CHAT_MAX_TOKENS, so
    ``min()`` picks THEM). ``0`` and any non-int are equally invalid. All
    five must fall back to the cap, not be passed through or raise.
    """
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": "ok"}, "done": True}),
        )

    client = _client(handler)
    client.chat([{"role": "user", "content": "hi"}], options={"num_predict": sentinel})

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["options"]["num_predict"] == CHAT_MAX_TOKENS


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


# --- extract: images param (P3.1 VLM document ingestion) -------------------


def test_extract_attaches_images_to_the_last_message_in_the_request_body():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"role": "assistant", "content": json.dumps({"name": "dog", "legs": 4})}, "done": True}
            ),
        )

    client = _client(handler)
    client.extract(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "describe a dog"}],
        _Animal,
        images=["base64pngdata"],
    )

    body = captured["body"]
    assert isinstance(body, dict)
    messages = body["messages"]
    # Only the LAST message carries images -- the earlier message is untouched.
    assert "images" not in messages[0]
    assert messages[1]["images"] == ["base64pngdata"]
    assert messages[1]["content"] == "describe a dog"


def test_extract_retry_resends_images_on_the_second_attempt():
    call_count = 0
    captured_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        captured_bodies.append(json.loads(request.content))
        content = "not valid json {{{" if call_count == 1 else json.dumps({"name": "cat", "legs": 4})
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": content}, "done": True}),
        )

    client = _client(handler, max_retries=2)
    client.extract([{"role": "user", "content": "describe a cat"}], _Animal, images=["page-1-base64"])

    assert call_count == 2
    for body in captured_bodies:
        assert body["messages"][-1]["images"] == ["page-1-base64"]


def test_extract_without_images_produces_no_images_key_in_body():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"role": "assistant", "content": json.dumps({"name": "dog", "legs": 4})}, "done": True}
            ),
        )

    client = _client(handler)
    client.extract([{"role": "user", "content": "describe a dog"}], _Animal)

    body = captured["body"]
    assert isinstance(body, dict)
    assert "images" not in body["messages"][-1]


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


# --- chat_stream: incremental reasoning deltas (Phase 1 #213) -----------------------
#
# ``chat_stream`` yields the same post-strip content ``chat()`` would return,
# but incrementally instead of assembled -- the reasoning half of the
# planner's two-call final-answer step (P2.9) streams into a UI "thinking"
# zone token-by-token (Phase 1 #213) instead of popping in all at once. The hard
# safety constraint: qwen3:4b's leaked chain-of-thought preamble (see the
# module docstring's "Live-verified quirk" note) must NEVER reach a caller,
# even one token of it -- so nothing is yielded until the ``</think>``
# boundary is resolved (found, with its trailing whitespace consumed) or its
# absence is confirmed (the stream ends with no ``</think>`` ever seen).


def test_chat_stream_buffers_leaked_preamble_without_yielding_any_of_it():
    """The FIRST value the generator ever produces must be post-boundary
    content -- proves nothing from the leaked preamble ever reaches a caller,
    not even a partial token of it."""
    body = _ndjson(
        {"message": {"content": "partial leaked chain of "}, "done": False},
        {"message": {"content": "thought reasoning"}, "done": False},
        {"message": {"content": "</think>"}, "done": False},
        {"message": {"content": "final answer text"}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler)
    deltas = client.chat_stream([{"role": "user", "content": "hi"}])

    first_delta = next(deltas)

    assert first_delta == "final answer text"
    assert "leaked" not in first_delta
    assert "reasoning" not in first_delta


def test_chat_stream_yields_multiple_deltas_incrementally_after_the_boundary():
    """Once the boundary resolves, subsequent pieces stream one at a time
    (not bunched into one final chunk) -- the real incremental behavior a
    typewriter UI needs."""
    body = _ndjson(
        {"message": {"content": "leaked "}, "done": False},
        {"message": {"content": "reasoning</think>"}, "done": False},
        {"message": {"content": "\n\nHello"}, "done": False},
        {"message": {"content": " world"}, "done": False},
        {"message": {"content": "."}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler)
    deltas = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert deltas == ["Hello", " world", "."]
    assert "".join(deltas) == "Hello world."
    assert "leaked" not in "".join(deltas)


def test_chat_stream_matches_chat_output_exactly_when_think_marker_spans_a_chunk_boundary():
    """Parity check: the joined chat_stream() output must equal chat()'s
    return value for the identical script -- including the tricky case where
    the leaked preamble's terminator and the real content's leading
    whitespace are split across different NDJSON chunks (chat()'s
    ``_strip_leaked_thinking`` strips that whitespace via ``\\s*`` on the
    fully-joined string; chat_stream() must reproduce the same result even
    though it resolves the boundary before that whitespace has arrived)."""

    def body() -> bytes:
        return _ndjson(
            {"message": {"content": "some leaked reasoning"}, "done": False},
            {"message": {"content": "</think>"}, "done": False},
            {"message": {"content": "\n\n"}, "done": False},
            {"message": {"content": "hello"}, "done": True},
        )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    chat_client = _client(handler)
    chat_result = chat_client.chat([{"role": "user", "content": "hi"}])

    stream_client = _client(handler)
    stream_result = "".join(stream_client.chat_stream([{"role": "user", "content": "hi"}]))

    assert chat_result == "hello"
    assert stream_result == chat_result


def test_chat_stream_confirms_absence_and_yields_whole_content_at_stream_end():
    """No ``</think>`` marker ever appears -- nothing is yielded until the
    stream ends (absence can only be "confirmed" once there is no more
    stream left that could still contain it), then the whole content is
    yielded as a single final delta."""
    body = _ndjson(
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": ", world."}, "done": True},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler)
    deltas = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert deltas == ["Hello, world."]


def test_chat_stream_sends_think_false_stream_true_and_model_same_as_chat():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ndjson({"message": {"content": "ok"}, "done": True}))

    client = _client(handler, model="qwen3:4b")
    list(client.chat_stream([{"role": "user", "content": "hi"}]))

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["think"] is False
    assert body["stream"] is True
    assert body["model"] == "qwen3:4b"


def test_chat_stream_records_call_stats_with_tokens_model_and_timing():
    body = _ndjson(
        {"message": {"content": "hello"}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 7},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler, model="qwen3:4b")
    list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert len(client.call_stats) == 1
    stats = client.call_stats[0]
    assert stats.model == "qwen3:4b"
    assert stats.ok is True
    assert stats.tokens_in == 12
    assert stats.tokens_out == 7


def test_chat_stream_maps_http_500_to_ollama_error_and_records_failed_call_stats():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal stack trace detail")

    client = _client(handler)
    with pytest.raises(OllamaError) as excinfo:
        list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert "internal stack trace detail" not in str(excinfo.value)
    assert len(client.call_stats) == 1
    assert client.call_stats[0].ok is False


def test_chat_stream_maps_connection_error_to_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    with pytest.raises(OllamaError):
        list(client.chat_stream([{"role": "user", "content": "hi"}]))


def test_chat_does_not_gain_a_chat_stream_call_stats_entry():
    """chat() stays byte-identical/additive-only: calling it must not touch
    chat_stream's machinery or record more than the one call_stats entry it
    always has."""
    body = _ndjson({"message": {"content": "hello"}, "done": True})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _client(handler)
    result = client.chat([{"role": "user", "content": "hi"}])

    assert result == "hello"
    assert len(client.call_stats) == 1


def test_from_settings_builds_client_targeting_configured_base_url_and_model():
    settings = Settings(ollama_base_url="http://ollama.example:11434")

    client = OllamaClient.from_settings(settings)

    assert client._base_url == "http://ollama.example:11434"


# --- call_stats: per-call token counts + timing (Phase 1 #149 span emission) --------
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


# --- embed: dense embeddings for hybrid retrieval (P3.3) --------------------


def test_embed_posts_to_embeddings_path_with_embedding_model_and_prompt():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    client = _client(handler, embedding_model="nomic-embed-text")
    result = client.embed("some clinical guideline text")

    assert captured["url"] == "http://ollama:11434/api/embeddings"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body == {"model": "nomic-embed-text", "prompt": "some clinical guideline text"}
    assert result == [0.1, 0.2, 0.3]


def test_embed_never_touches_chat_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [1.0]})

    client = _client(handler, model="qwen3:4b", embedding_model="nomic-embed-text")
    client.embed("text")

    assert client.call_stats[-1].model == "nomic-embed-text"


def test_embed_retries_on_malformed_response_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"not_embedding": []})
        return httpx.Response(200, json={"embedding": [0.5, 0.6]})

    client = _client(handler, max_retries=2)
    result = client.embed("text")

    assert result == [0.5, 0.6]
    assert calls["n"] == 2


def test_embed_raises_after_exhausting_retries_on_malformed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": "not-a-list"})

    client = _client(handler, max_retries=2)

    with pytest.raises(OllamaError):
        client.embed("text")


def test_embed_raises_immediately_on_http_error_without_retrying():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    client = _client(handler, max_retries=3)

    with pytest.raises(OllamaError):
        client.embed("text")

    assert calls["n"] == 1


def test_embed_does_not_log_the_input_text(caplog):
    caplog.set_level(logging.INFO, logger="app.ollama_client")
    secret_text = "SUPER_SECRET_GUIDELINE_QUERY_TOKEN"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [0.1]})

    client = _client(handler)
    client.embed(secret_text)

    records = [r for r in caplog.records if r.name == "app.ollama_client"]
    assert secret_text not in _all_log_text(records)


# --- is_vision_capable_model: gate-1 finding on #206 -------------------


def test_is_vision_capable_model_rejects_a_text_only_model():
    """Safety property: the heuristic must still reject an unrecognized,
    genuinely text-only model name -- this must not regress while fixing
    the false-rejection cases above."""
    assert is_vision_capable_model("qwen3:4b") is False


# --- is_vision_capable_model: boundary-aware matching (security gate, #204) --
#
# MAJOR finding: the plain substring matcher accepted TEXT-ONLY models whose
# names merely contain "vl"/"vision" as a fragment of an ordinary word
# (``med-supervision-4b``, ``wavlm-base``, ...) -- a false POSITIVE that lets
# a text-only model silently pass the safety check on the default path, with
# no override touched. A second, mirror-image gate finding showed the same
# substring matcher wrongly REJECTING genuine VLM names that don't happen to
# contain a listed marker as a bare fragment in the right spot. Each case
# below is independently named so a failure identifies exactly which model
# name regressed.


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("qwen2.5vl:7b", id="accept-qwen2.5vl-tag"),
        pytest.param("qwen2-vl:7b", id="accept-qwen2-vl-dash-tag"),
        pytest.param("llama3.2-vision", id="accept-llama-vision-no-tag"),
        pytest.param("llama3.2-vision:11b", id="accept-llama-vision-with-tag"),
        pytest.param("llava:13b", id="accept-llava-tag"),
        pytest.param("llava", id="accept-llava-bare"),
        pytest.param("llava-llama3", id="accept-llava-llama3"),
        pytest.param("bakllava", id="accept-bakllava-bare"),
        pytest.param("moondream", id="accept-moondream-bare"),
        pytest.param("pixtral-12b", id="accept-pixtral-12b"),
        pytest.param("pixtral", id="accept-pixtral-bare"),
        pytest.param("minicpm-v", id="accept-minicpm-v-bare"),
        pytest.param("minicpm-v:8b", id="accept-minicpm-v-tag"),
        # Case-insensitivity: is_vision_capable_model() normalizes to
        # lowercase before matching (previously covered only by the
        # now-removed test_is_vision_capable_model_recognizes_minicpm_v).
        pytest.param("MiniCPM-V", id="accept-minicpm-v-mixed-case"),
        pytest.param("minicpm-o", id="accept-minicpm-o-bare"),
        # MINOR-6 (#204 gate-3): trailing digits after the ``vl`` marker,
        # before the next delimiter/end, must not defeat the boundary check
        # -- ``deepseek-vl2``/``qwen2-vl2`` follow the same ``vlN``
        # version-suffix convention as ``deepseek-vl`` (already accepted).
        pytest.param("deepseek-vl2", id="accept-deepseek-vl2-digit-suffix"),
        pytest.param("qwen2-vl2", id="accept-qwen2-vl2-digit-suffix"),
    ],
)
def test_is_vision_capable_model_accepts_genuine_vlm_names(model: str):
    assert is_vision_capable_model(model) is True


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("qwen3:4b", id="reject-qwen3-text-only"),
        pytest.param("med-supervision-4b", id="reject-supervision-near-miss"),
        pytest.param("clinical-provision:2b", id="reject-provision-near-miss"),
        pytest.param("notes-revision-4b", id="reject-revision-near-miss"),
        pytest.param("envision-lite:1b", id="reject-envision-near-miss"),
        pytest.param("wavlm-base", id="reject-wavlm-vl-near-miss"),
        pytest.param("avle-test", id="reject-avle-vl-near-miss"),
        pytest.param("uvloop-helper", id="reject-uvloop-vl-near-miss"),
        pytest.param("llama3:8b", id="reject-llama3-text-only"),
        pytest.param("mistral:7b", id="reject-mistral-text-only"),
        pytest.param("television-model:1b", id="reject-television-vision-near-miss"),
        pytest.param("devlin-base:1b", id="reject-devlin-vl-near-miss"),
        # MODERATE-3 (#204 gate-3): bare "minicpm" wrongly admitted the
        # text-only MiniCPM family -- these are real, non-vision LLMs and
        # must be refused now that the marker is minicpm-v/minicpm-o only.
        pytest.param("minicpm3:4b", id="reject-minicpm3-text-only"),
        pytest.param("minicpm4:8b", id="reject-minicpm4-text-only"),
        pytest.param("minicpm:2b", id="reject-minicpm-2b-text-only"),
        pytest.param("minicpm-2b-sft", id="reject-minicpm-2b-sft-text-only"),
        # MINOR-6 (#204 gate-3): letter-adjacent VL names are structurally
        # indistinguishable from wavlm/avle/uvloop above and stay
        # unrecognized by design -- pinned here so the limit is an explicit
        # test, not folklore.
        pytest.param("internvl2", id="not-recognized-internvl2-letter-adjacent"),
        pytest.param("cogvlm", id="not-recognized-cogvlm-letter-adjacent"),
        pytest.param("smolvlm", id="not-recognized-smolvlm-letter-adjacent"),
        # Multimodal models with no marker this function knows about at
        # all -- not recognized, override required (Settings.
        # copilot_vision_model_capability_check=false).
        pytest.param("gemma3:4b", id="not-recognized-gemma3-no-marker"),
        pytest.param("paligemma:3b", id="not-recognized-paligemma-no-marker"),
        pytest.param("idefics2:8b", id="not-recognized-idefics2-no-marker"),
        pytest.param("fuyu:8b", id="not-recognized-fuyu-no-marker"),
        pytest.param("glm-4v:9b", id="not-recognized-glm-4v-no-marker"),
        pytest.param("llama4:scout", id="not-recognized-llama4-no-marker"),
    ],
)
def test_is_vision_capable_model_rejects_text_only_and_near_miss_names(model: str):
    assert is_vision_capable_model(model) is False


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


@pytest.mark.integration
def test_live_embed_against_real_nomic_embed_text_returns_a_vector():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    settings = Settings(ollama_base_url=base_url, ollama_api_timeout_seconds=120.0)
    client = OllamaClient.from_settings(settings)

    result = client.embed("What A1c target for most adults?")

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, float) for v in result)

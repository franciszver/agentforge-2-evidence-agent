"""Ollama chat client: streaming chat + JSON-schema-constrained extraction.

Scope (P2.7): a thin client for the internal Ollama instance serving
``qwen3:4b`` (the *thinking* variant). Two entry points:

  * ``chat``    — POST ``/api/chat`` with ``stream: true``, assemble the
                  NDJSON chunk stream into the full response text.
  * ``extract`` — POST ``/api/chat`` with ``format`` set to a Pydantic
                  model's JSON schema, so Ollama constrains decoding to
                  valid JSON for that schema, then ``model_validate`` the
                  result. Retries a small, fixed number of times on
                  malformed output before raising.

Design notes:
  * ``think: false`` is set on every request. ``qwen3:4b`` is the thinking
    variant and emits ``thinking`` tokens by default; the agent wants plain
    Instruct-style output, not the chain-of-thought preamble.
  * Live-verified quirk (Ollama 0.12.6 + qwen3:4b): ``think: false`` stops
    Ollama from separating reasoning into ``message.thinking``, but does NOT
    stop the model from generating it -- the reasoning leaks straight into
    ``message.content``, terminated by a stray ``</think>`` marker (often
    with no matching opening tag). ``_strip_leaked_thinking`` defends
    against this by dropping everything up to and including the first
    ``</think>`` marker, so callers only ever see the real answer.
  * ``temperature: 0`` by default (overridable per call via ``options``) —
    deterministic output is what both chat replies and constrained
    extraction want here.
  * Synchronous, matching ``app.openemr_client``'s injectable-``httpx.Client``
    pattern: the client is always passed in, so tests drive it with
    ``httpx.MockTransport`` and no real network is touched.
  * ``OllamaError`` messages are log-safe: never the raw model output, which
    may echo injected or PHI-bearing text from the prompt — only a fixed
    operation label and, where relevant, the HTTP status code.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

_logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_CHAT_PATH = "/api/chat"

# Matches everything up to and including the first "</think>" marker (and any
# whitespace right after it), whether or not a matching "<think>" opening tag
# is present. See the module docstring's "Live-verified quirk" note.
_LEAKED_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class LlmCallStats:
    """Timing + token counts for one completed call to Ollama (P4/#149).

    Appended to ``OllamaClient.call_stats`` for every underlying request the
    client makes -- one per ``chat()`` call, and one per ``extract()``
    *attempt* (a retried extraction is a real, token-consuming call to
    Ollama, so each attempt gets its own entry, not just the final one).
    Callers (``app.planner``, ``app.extraction``) read this side channel
    after the fact to build the ``llm`` spans ``app.trace_store`` persists --
    chosen over changing ``chat``/``extract``'s return types, which would
    touch every call site and the many existing tests asserting on those
    return values.
    """

    model: str
    start_ts: float
    end_ts: float
    ok: bool
    tokens_in: int | None
    tokens_out: int | None


class OllamaError(Exception):
    """Raised when an Ollama request or constrained extraction fails.

    The message is intentionally log-safe: it never embeds raw model output,
    which may contain injected or PHI-bearing text from the prompt.
    """


class OllamaClient:
    """Chat + constrained-extraction client for the internal Ollama instance.

    Args:
        base_url: Origin of the Ollama instance, e.g. ``"http://ollama:11434"``.
        client: An injectable ``httpx.Client`` — hermetic tests inject one
            backed by ``httpx.MockTransport``; production injects one via
            :meth:`from_settings`.
        model: Ollama model name to request, e.g. ``"qwen3:4b"``.
        max_retries: Max attempts :meth:`extract` makes before raising when
            the model's output fails to parse/validate as the target schema
            (this is a total-attempts count, not "retries in addition to
            the first attempt").
    """

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.Client,
        model: str = "qwen3:4b",
        max_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._model = model
        self._max_retries = max_retries
        # Side channel of per-call timing/token stats -- see ``LlmCallStats``.
        # Public (not ``_``-prefixed): ``app.planner``/``app.extraction`` read
        # it after invoking ``chat``/``extract`` to build ``llm`` trace spans.
        self.call_stats: list[LlmCallStats] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaClient:
        """Build a production client, threading base URL, model, timeout, and retries."""
        client = httpx.Client(timeout=settings.ollama_api_timeout_seconds)
        return cls(
            base_url=settings.ollama_base_url,
            client=client,
            model=settings.ollama_model,
            max_retries=settings.ollama_extract_max_retries,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat request and return the assembled response text.

        POSTs with ``stream: true`` and assembles the NDJSON chunk stream's
        ``message.content`` pieces into the full response. Appends one
        ``LlmCallStats`` entry to ``call_stats`` regardless of outcome.
        """
        _logger.info("ollama chat call", extra={"model": self._model})
        body = self._build_body(messages, stream=True, options=options)
        start_ts = time.time()
        try:
            response = self._post_chat(body)
            content, tokens_in, tokens_out = self._assemble_stream(response)
        except OllamaError as exc:
            end_ts = time.time()
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
            )
            _logger.warning(
                "ollama chat call failed",
                extra={
                    "model": self._model,
                    "error_type": type(exc).__name__,
                    "duration_ms": round((end_ts - start_ts) * 1000, 1),
                },
            )
            raise
        self.call_stats.append(
            LlmCallStats(model=self._model, start_ts=start_ts, end_ts=time.time(), ok=True, tokens_in=tokens_in, tokens_out=tokens_out)
        )
        return content

    def extract(
        self,
        prompt_or_messages: str | list[dict[str, str]],
        schema: type[ModelT],
        *,
        options: dict[str, Any] | None = None,
    ) -> ModelT:
        """Extract ``schema`` from the model's response via constrained decoding.

        POSTs with ``format`` set to ``schema.model_json_schema()`` so Ollama
        constrains decoding to valid JSON for that schema, then parses and
        ``model_validate``s the result. If the returned content isn't valid
        JSON, or fails schema validation, retries up to ``max_retries`` total
        attempts before raising ``OllamaError``. Network/HTTP failures (a
        non-2xx status, a timeout, a connection error) are NOT retried here —
        they propagate immediately as ``OllamaError``.
        """
        _logger.info("ollama extract call", extra={"model": self._model, "schema": schema.__name__})
        messages = self._normalize_messages(prompt_or_messages)
        body = self._build_body(
            messages,
            stream=False,
            format=schema.model_json_schema(),
            options=options,
        )

        for attempt in range(1, self._max_retries + 1):
            start_ts = time.time()
            # Network/HTTP failures are NOT retried (see docstring): keep the
            # ``_post_chat`` call OUT of the retry-catch below so an
            # ``OllamaError`` from it propagates immediately, after recording
            # the failed attempt's stats (symmetric with ``chat``).
            try:
                response = self._post_chat(body)
            except OllamaError as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
                )
                _logger.warning(
                    "ollama extract call failed",
                    extra={
                        "model": self._model,
                        "schema": schema.__name__,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "duration_ms": round((end_ts - start_ts) * 1000, 1),
                    },
                )
                raise
            tokens_in: int | None = None
            tokens_out: int | None = None
            try:
                content, tokens_in, tokens_out = self._single_message_content(response)
                payload = json.loads(content)
                result = schema.model_validate(payload)
            except (OllamaError, ValueError, ValidationError) as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=tokens_in, tokens_out=tokens_out)
                )
                will_retry = attempt < self._max_retries
                _logger.warning(
                    "ollama extract call retrying after malformed output"
                    if will_retry
                    else "ollama extract call failed after exhausting retries",
                    extra={
                        "model": self._model,
                        "schema": schema.__name__,
                        "attempt": attempt,
                        "max_retries": self._max_retries,
                        "error_type": type(exc).__name__,
                        "duration_ms": round((end_ts - start_ts) * 1000, 1),
                    },
                )
                continue
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=time.time(), ok=True, tokens_in=tokens_in, tokens_out=tokens_out)
            )
            return result

        raise OllamaError(f"constrained extraction failed after {self._max_retries} attempts")

    def _post_chat(self, body: dict[str, Any]) -> httpx.Response:
        url = f"{self._base_url}{_CHAT_PATH}"
        try:
            response = self._client.post(url, json=body)
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise OllamaError("Ollama request failed") from exc

        if not response.is_success:
            raise OllamaError(f"Ollama request failed (status {response.status_code})")
        return response

    def _build_body(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
        format: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged_options: dict[str, Any] = {"temperature": 0}
        if options:
            merged_options.update(options)

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "options": merged_options,
        }
        if format is not None:
            body["format"] = format
        return body

    @staticmethod
    def _normalize_messages(prompt_or_messages: str | list[dict[str, str]]) -> list[dict[str, str]]:
        if isinstance(prompt_or_messages, str):
            return [{"role": "user", "content": prompt_or_messages}]
        return prompt_or_messages

    @staticmethod
    def _assemble_stream(response: httpx.Response) -> tuple[str, int | None, int | None]:
        """Parse an NDJSON chunk stream and concatenate ``message.content`` pieces.

        Also returns the token counts Ollama reports on the terminal
        (``done: true``) chunk (``prompt_eval_count``/``eval_count``), or
        ``None``/``None`` if that chunk didn't carry them.
        """
        parts: list[str] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError as exc:
                raise OllamaError("Ollama stream contained invalid JSON") from exc
            if not isinstance(chunk, dict):
                continue
            message = chunk.get("message")
            if isinstance(message, dict):
                piece = message.get("content")
                if isinstance(piece, str):
                    parts.append(piece)
            if chunk.get("done") is True:
                tokens_in, tokens_out = OllamaClient._token_counts(chunk)
        return OllamaClient._strip_leaked_thinking("".join(parts)), tokens_in, tokens_out

    @staticmethod
    def _single_message_content(response: httpx.Response) -> tuple[str, int | None, int | None]:
        """Extract ``message.content`` from a non-streamed (single-object) response.

        Also returns ``prompt_eval_count``/``eval_count`` from the same
        response payload.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama response was not valid JSON") from exc

        message = payload.get("message") if isinstance(payload, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaError("Ollama response missing message content")
        tokens_in, tokens_out = OllamaClient._token_counts(payload) if isinstance(payload, dict) else (None, None)
        return OllamaClient._strip_leaked_thinking(content), tokens_in, tokens_out

    @staticmethod
    def _token_counts(payload: dict[str, Any]) -> tuple[int | None, int | None]:
        """Pull ``prompt_eval_count``/``eval_count`` out of an Ollama response
        payload (a streamed ``done: true`` chunk or a non-streamed body) --
        both live at the top level alongside ``message``/``done``."""
        tokens_in = payload.get("prompt_eval_count")
        tokens_out = payload.get("eval_count")
        return (
            tokens_in if isinstance(tokens_in, int) else None,
            tokens_out if isinstance(tokens_out, int) else None,
        )

    @staticmethod
    def _strip_leaked_thinking(content: str) -> str:
        """Drop a leaked chain-of-thought preamble; see module docstring."""
        return _LEAKED_THINK_RE.sub("", content, count=1)

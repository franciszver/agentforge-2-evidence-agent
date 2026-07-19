"""llama-server chat client: constrained extraction + chat for the local
llama.cpp runtime (P3.10a, epic #52 step 1).

Scope: this client serves ONLY the text-generation roles migrated off Ollama
in this step -- planner chat/answer, claim extraction, and the LLM-as-reranker
relevance score. Embeddings (``nomic-embed-text``) and vision-based document-
ingestion extraction stay on ``app.ollama_client.OllamaClient`` (see
``app.chat._build_evidence_workers``); this client's ``embed`` deliberately
raises ``NotImplementedError``.

Wire protocol: llama-server's OpenAI-compatible ``/v1/chat/completions``
endpoint, NOT Ollama's native ``/api/chat`` -- a different JSON shape than
``OllamaClient`` speaks, but this class duck-types the SAME Python call
signature (``chat``, ``chat_stream``, ``extract``, ``embed``, constructor
shape) so callers can swap ``OllamaClient(...)`` for ``LlamaServerClient(...)``
with no other code change (see ``app.chat.get_text_llm_client``).

Design notes:
  * Constrained extraction uses llama-server's
    ``response_format={"type": "json_schema", "json_schema": {"name":
    ..., "schema": ..., "strict": true}}``, which llama.cpp maps internally
    to grammar-constrained decoding -- the same guarantee Ollama's
    ``format=<schema>`` gives: output is always syntactically valid JSON for
    the schema.
  * ``max_tokens: 1536`` is applied to EVERY request (chat and extract), not
    only chat. A JSON schema alone does not bound generation length -- an
    open-ended schema string field can still run away (measured: one
    extraction call ran 305s and hit the request timeout even with a schema
    in effect), so the cap is unconditional.
  * ``<think>`` leak stripping and message normalization directly REUSE
    ``OllamaClient._strip_leaked_thinking``/``_normalize_messages`` (both
    pure, stateless static methods) rather than a second copy of the regex,
    so the two clients can never drift on this shared, security-relevant
    behavior. The production launch flag is ``--reasoning off`` (server-side,
    not a per-request field), which should prevent Qwen3's thinking preamble
    from ever reaching ``message.content`` -- the strip is defense-in-depth
    for the same failure mode ``OllamaClient`` guards against, harmless if
    the server never leaks.
  * ``LlmCallStats`` is also reused directly from ``app.ollama_client``
    (not redefined) -- callers reading ``call_stats`` off either client get
    the same type regardless of which engine served the call.
  * ``temperature: 0`` by default, overridable per call via ``options``
    (merged directly onto the OpenAI-style top-level body).
  * Synchronous, injectable-``httpx.Client`` pattern matching
    ``OllamaClient``/``OpenEmrClient``: hermetic tests drive it with
    ``httpx.MockTransport``, no real network is touched.
  * ``LlamaServerError`` messages are log-safe: never the raw model output.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.ollama_client import LLMEngineError, LlmCallStats, OllamaClient

_logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Hard cap applied to every call (chat and extract) -- see module docstring.
_MAX_TOKENS = 1536


class LlamaServerError(LLMEngineError):
    """Raised when a llama-server request or constrained extraction fails.

    Log-safe: never embeds raw model output (mirrors ``OllamaError``). Shares
    ``LLMEngineError`` as a common base with ``OllamaError`` (#60) so callers
    that must degrade gracefully regardless of the configured
    ``copilot_llm_engine`` can catch one type.
    """


class LlamaServerClient:
    """Chat + constrained-extraction client for a llama.cpp ``llama-server``
    instance, duck-typed against ``app.ollama_client.OllamaClient``.

    Args:
        base_url: e.g. ``"http://llama-server:8080"``.
        client: injectable ``httpx.Client`` -- hermetic tests inject one
            backed by ``httpx.MockTransport``; production injects one via
            :meth:`from_settings`.
        model: model name/id sent in the request body. llama-server running
            a single ``--model`` file ignores this for routing (only one
            model is loaded), but the OpenAI-compatible endpoint still
            requires the field to be present.
        max_retries: total attempts :meth:`extract` makes before raising
            (a total-attempts count, matching ``OllamaClient``).
    """

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.Client,
        model: str = "qwen3-8b",
        max_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._model = model
        self._max_retries = max_retries
        # Side channel of per-call timing/token stats -- see ``LlmCallStats``.
        self.call_stats: list[LlmCallStats] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> LlamaServerClient:
        """Build a production client, threading base URL, model, timeout, and retries."""
        client = httpx.Client(timeout=settings.llama_server_api_timeout_seconds)
        return cls(
            base_url=settings.llama_server_base_url,
            client=client,
            model=settings.llama_server_model,
            max_retries=settings.llama_server_extract_max_retries,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Non-streaming chat call. Returns the assembled response text."""
        _logger.info("llama-server chat call", extra={"model": self._model})
        body = self._build_body(messages, stream=False, options=options)
        start_ts = time.time()
        try:
            response = self._post(_CHAT_COMPLETIONS_PATH, body)
            content, tokens_in, tokens_out = self._single_message_content(response)
        except LlamaServerError as exc:
            end_ts = time.time()
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
            )
            _logger.warning(
                "llama-server chat call failed",
                extra={"model": self._model, "error_type": type(exc).__name__},
            )
            raise
        self.call_stats.append(
            LlmCallStats(model=self._model, start_ts=start_ts, end_ts=time.time(), ok=True, tokens_in=tokens_in, tokens_out=tokens_out)
        )
        return content

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Streaming chat call (SSE ``data: {...}`` lines), yielding text deltas.

        Mirrors ``OllamaClient.chat_stream``'s leaked-``<think>`` buffering
        contract: nothing is yielded until any ``</think>`` boundary is
        resolved (or the stream ends, confirming absence).
        """
        _logger.info("llama-server chat stream call", extra={"model": self._model})
        body = self._build_body(messages, stream=True, options=options)
        start_ts = time.time()
        tokens_in: int | None = None
        tokens_out: int | None = None
        # ``response`` is opened with ``stream=True`` (a held-open connection,
        # unlike the plain buffered ``.post()`` the other methods use), so it
        # MUST be explicitly closed -- otherwise an early-stopped iteration
        # (caller breaks, or this generator is closed/GC'd before exhaustion)
        # leaks the underlying connection back to the pool. ``finally`` runs
        # on every exit path, including ``GeneratorExit``.
        response: httpx.Response | None = None
        try:
            response = self._post(_CHAT_COMPLETIONS_PATH, body, stream=True)
            buffer = ""
            trimming = False
            passthrough = False
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload_str = line[len("data:") :].strip()
                if payload_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload_str)
                except ValueError as exc:
                    raise LlamaServerError("llama-server stream contained invalid JSON") from exc
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                delta = choices[0].get("delta") if choices else None
                piece = delta.get("content") if isinstance(delta, dict) else None
                usage = chunk.get("usage") if isinstance(chunk, dict) else None
                if isinstance(usage, dict):
                    tokens_in = usage.get("prompt_tokens", tokens_in)
                    tokens_out = usage.get("completion_tokens", tokens_out)
                if not isinstance(piece, str) or not piece:
                    continue
                if passthrough:
                    yield piece
                    continue
                if not trimming:
                    scan_from = max(0, len(buffer) - (len("</think>") - 1))
                    buffer += piece
                    idx = buffer.find("</think>", scan_from)
                    if idx == -1:
                        continue
                    piece = buffer[idx + len("</think>") :]
                    buffer = ""
                    trimming = True
                stripped = piece.lstrip()
                if stripped:
                    trimming = False
                    passthrough = True
                    yield stripped
            if not passthrough and not trimming and buffer:
                yield buffer
        except LlamaServerError as exc:
            end_ts = time.time()
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
            )
            _logger.warning(
                "llama-server chat stream call failed",
                extra={"model": self._model, "error_type": type(exc).__name__},
            )
            raise
        except httpx.HTTPError as exc:
            # A network failure mid-stream (e.g. a dropped connection after
            # headers were already received) surfaces as an httpx error, not
            # ``LlamaServerError`` -- ``_post`` only wraps the initial
            # request/connect, not errors raised while iterating the body.
            # Caught here for the same stats-accounting/log-safety guarantee
            # as every other failure path.
            end_ts = time.time()
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
            )
            _logger.warning(
                "llama-server chat stream call failed",
                extra={"model": self._model, "error_type": type(exc).__name__},
            )
            raise LlamaServerError("llama-server stream failed while iterating") from exc
        finally:
            if response is not None:
                response.close()
        self.call_stats.append(
            LlmCallStats(model=self._model, start_ts=start_ts, end_ts=time.time(), ok=True, tokens_in=tokens_in, tokens_out=tokens_out)
        )

    def extract(
        self,
        prompt_or_messages: str | list[dict[str, str]],
        schema: type[ModelT],
        *,
        options: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> ModelT:
        """Extract ``schema`` via llama-server's OpenAI-compatible
        ``response_format={"type": "json_schema", "json_schema": {...}}``,
        which llama.cpp maps internally to grammar-constrained decoding (same
        guarantee as Ollama's ``format=<schema>``: output is always
        syntactically valid JSON for the schema). Retries up to
        ``max_retries`` total attempts on malformed/invalid output;
        network/HTTP failures propagate immediately (same policy as
        ``OllamaClient``).

        ``images`` is accepted for signature parity with ``OllamaClient`` but
        NOT implemented -- vision-based document-ingestion extraction stays
        on Ollama (see module docstring). Raises if a caller ever passes
        non-``None`` images, rather than silently dropping them.
        """
        if images is not None:
            raise NotImplementedError("LlamaServerClient.extract: images not supported -- vision stays on OllamaClient")
        _logger.info("llama-server extract call", extra={"model": self._model, "schema": schema.__name__})
        messages = self._normalize_messages(prompt_or_messages)
        json_schema = schema.model_json_schema()
        body = self._build_body(
            messages,
            stream=False,
            options=options,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": True},
            },
        )

        for attempt in range(1, self._max_retries + 1):
            start_ts = time.time()
            try:
                response = self._post(_CHAT_COMPLETIONS_PATH, body)
            except LlamaServerError as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
                )
                _logger.warning(
                    "llama-server extract call failed",
                    extra={"model": self._model, "schema": schema.__name__, "attempt": attempt, "error_type": type(exc).__name__},
                )
                raise
            tokens_in: int | None = None
            tokens_out: int | None = None
            try:
                content, tokens_in, tokens_out = self._single_message_content(response)
                payload = json.loads(content)
                result = schema.model_validate(payload)
            except (LlamaServerError, ValueError, ValidationError) as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=tokens_in, tokens_out=tokens_out)
                )
                will_retry = attempt < self._max_retries
                _logger.warning(
                    "llama-server extract call retrying after malformed output"
                    if will_retry
                    else "llama-server extract call failed after exhausting retries",
                    extra={
                        "model": self._model,
                        "schema": schema.__name__,
                        "attempt": attempt,
                        "max_retries": self._max_retries,
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=time.time(), ok=True, tokens_in=tokens_in, tokens_out=tokens_out)
            )
            return result

        raise LlamaServerError(f"constrained extraction failed after {self._max_retries} attempts")

    def embed(self, text: str) -> list[float]:
        """Not implemented -- embeddings always stay on ``OllamaClient`` (see
        module docstring)."""
        raise NotImplementedError("LlamaServerClient.embed: embeddings stay on OllamaClient")

    def _post(self, path: str, body: dict[str, Any], *, stream: bool = False) -> httpx.Response:
        url = f"{self._base_url}{path}"
        try:
            if stream:
                request = self._client.build_request("POST", url, json=body)
                response = self._client.send(request, stream=True)
            else:
                response = self._client.post(url, json=body)
        except httpx.TimeoutException as exc:
            raise LlamaServerError("llama-server request timed out") from exc
        except httpx.HTTPError as exc:
            raise LlamaServerError("llama-server request failed") from exc

        if not response.is_success:
            raise LlamaServerError(f"llama-server request failed (status {response.status_code})")
        return response

    def _build_body(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
        options: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "temperature": 0,
            # Hard cap on every call -- see module docstring.
            "max_tokens": _MAX_TOKENS,
        }
        if options:
            # OllamaClient's ``options`` dict is Ollama-shaped (e.g.
            # {"temperature": ...}); pass keys through directly onto the
            # OpenAI-style top-level body, same keys largely apply.
            body.update(options)
        if response_format is not None:
            body["response_format"] = response_format
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    # Message normalization and leaked-<think> stripping are pure, stateless
    # helpers with no Ollama-specific behavior -- reused directly from
    # ``OllamaClient`` (rather than a second copy here) so the two clients
    # can never drift on this shared, security-relevant regex.
    _normalize_messages = staticmethod(OllamaClient._normalize_messages)
    _strip_leaked_thinking = staticmethod(OllamaClient._strip_leaked_thinking)

    @staticmethod
    def _single_message_content(response: httpx.Response) -> tuple[str, int | None, int | None]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlamaServerError("llama-server response was not valid JSON") from exc

        choices = payload.get("choices") if isinstance(payload, dict) else None
        message = choices[0].get("message") if choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LlamaServerError("llama-server response missing message content")
        usage = payload.get("usage") if isinstance(payload, dict) else None
        tokens_in = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        tokens_out = usage.get("completion_tokens") if isinstance(usage, dict) else None
        tokens_in = tokens_in if isinstance(tokens_in, int) else None
        tokens_out = tokens_out if isinstance(tokens_out, int) else None
        return LlamaServerClient._strip_leaked_thinking(content), tokens_in, tokens_out

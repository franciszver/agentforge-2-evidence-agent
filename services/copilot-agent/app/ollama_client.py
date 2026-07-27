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
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings

_logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_CHAT_PATH = "/api/chat"
_EMBEDDINGS_PATH = "/api/embeddings"

# Hard generation cap for chat()/chat_stream() -- engine parity (issue #167
# Gate 3 MAJOR finding). app.llama_server_client hard-caps max_tokens on
# EVERY request it sends (its module docstring's "max_tokens is capped on
# EVERY request" note); this class had no equivalent, so on the
# ``ollama`` engine (``settings.copilot_llm_engine == "ollama"``) chat
# generation ran until the model's own context window filled, and
# app.chat.Turn.answer (the P2.17 audit record, and part of #167's
# ConversationStore memory-bound fix) had no application-level bound on
# that engine. The production default engine is llama_server (see
# app.chat.get_text_llm_client), where this constant already applies via
# LlamaServerClient's own ``_CHAT_MAX_TOKENS`` (now sourced from here, so
# the two engines can't drift on the value) -- so setting this here is a
# NO-OP for the shipped default; it only changes behavior for the
# ``ollama`` engine option.
#
# Deliberately NOT applied as a default to extract(): unlike chat/
# chat_stream, its constrained-JSON output can legitimately need more
# tokens when it echoes retrieved guideline/fact text verbatim (see
# app.llama_server_client's ``_EXTRACT_MAX_TOKENS`` docstring for the same
# reasoning on that engine) -- capping it at the same 1536 here could turn
# real corpus retrieval results into truncated-JSON extraction failures
# that do not exist today. A caller may still pass ``options={"num_predict":
# ...}`` to extract() explicitly if a bound is wanted there.
#
# CEILING, not a default (Gate 3 follow-up): unlike ``temperature`` (a
# behavioural knob a caller is free to override in either direction), this
# is a retention bound protecting process memory -- see the module-level
# rationale above. A caller MAY still ask for FEWER tokens (a smaller, but
# still valid, ``options["num_predict"]`` is honoured as-is), but may never
# ask for more. ``_clamped_chat_options`` accepts a caller value ONLY if it
# is a plain ``int`` in ``[1, CHAT_MAX_TOKENS]``; anything else -- too big,
# non-int, or missing -- falls back to ``CHAT_MAX_TOKENS`` outright (NOT a
# plain ``min(caller_value, CHAT_MAX_TOKENS)``, and NOT clamped to the
# nearest boundary). This matters because Ollama's ``num_predict`` API
# treats certain values as "unlimited" sentinels rather than literal token
# counts: ``-1`` means generate until the model stops on its own (no cap at
# all), ``-2`` means fill the remaining context window. Both are
# numerically SMALLER than ``CHAT_MAX_TOKENS``, so a plain ``min()`` would
# pass them straight through -- the two idiomatic ways to ask for "as many
# as possible" would silently defeat the exact guard this exists to
# enforce. ``0`` (no generation) and any non-int value are equally
# out-of-range and also fall back to the cap rather than erroring or being
# clamped to ``1``.
CHAT_MAX_TOKENS = 1536


def _clamped_chat_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``options`` with a ``num_predict`` ceiling of ``CHAT_MAX_TOKENS``.

    Used by ``chat``/``chat_stream`` only -- see ``CHAT_MAX_TOKENS``'s
    docstring for why an out-of-range caller value (including Ollama's
    negative "unlimited" sentinels, ``0``, or a non-int) falls all the way
    back to ``CHAT_MAX_TOKENS`` rather than being merely defaulted (the way
    ``temperature`` is merged in ``_build_body``) or clamped to the nearest
    boundary via a plain ``min()``/``max()``.
    """
    merged_options: dict[str, Any] = dict(options) if options else {}
    caller_num_predict = merged_options.get("num_predict")
    valid = (
        isinstance(caller_num_predict, int)
        and not isinstance(caller_num_predict, bool)
        and 1 <= caller_num_predict <= CHAT_MAX_TOKENS
    )
    merged_options["num_predict"] = caller_num_predict if valid else CHAT_MAX_TOKENS
    return merged_options


# Matches everything up to and including the first "</think>" marker (and any
# whitespace right after it), whether or not a matching "<think>" opening tag
# is present. See the module docstring's "Live-verified quirk" note.
_LEAKED_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)
# The closing marker itself, for the incremental (streaming) boundary scan in
# ``_stream_deltas`` -- which cannot use ``_LEAKED_THINK_RE`` directly because
# it resolves the boundary before the trailing ``\s*`` has necessarily
# arrived. Kept adjacent so the two stay obviously in sync.
_THINK_CLOSE = "</think>"


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


class LLMEngineError(Exception):
    """Common base for text-LLM-engine failures, shared by ``OllamaError`` and
    ``app.llama_server_client.LlamaServerError`` (#60).

    Callers that must tolerate either configured ``copilot_llm_engine``
    (``ollama`` or ``llama_server``, see ``app.chat.get_text_llm_client``)
    should catch this base rather than one engine's concrete type, so a
    call site does not silently work under one engine and raise a 500 under
    the other.
    """


class OllamaError(LLMEngineError):
    """Raised when an Ollama request or constrained extraction fails.

    The message is intentionally log-safe: it never embeds raw model output,
    which may contain injected or PHI-bearing text from the prompt.
    """


# Issue #204: name-based recognizer for Ollama vision-capable models, used by
# app.supervisor.IntakeExtractorWorker to fail closed instead of silently
# handing an image-bearing document to a text-only model. Name-based rather
# than a live ``/api/show`` capability query (the field docs/DEMO_SCRIPT.md
# recorded, e.g. ``capabilities: ["vision", "completion"]`` for
# ``qwen2.5vl:7b``): keeps the check synchronous, dependency-free, and
# evaluable at construction/dispatch time with no network call -- at the
# cost of recognizing only name patterns already in common use for
# vision-language models, not a per-install, ground-truth capability list.
#
# Gate-1 finding on #206: this list is BEST-EFFORT ONLY, never authoritative
# -- it cannot recognize digest-pinned references (``sha256:...``, no
# human-readable segment), operator-renamed/custom tags, or VLM families not
# yet added here. Do not treat a rejection from this function as proof a
# model isn't vision-capable: the escape hatch for a false rejection is
# ``Settings.copilot_vision_model_capability_check`` (app/config.py), not a
# growing enumeration of markers.
_VISION_MODEL_NAME_MARKERS = ("vl", "vision", "llava", "moondream", "pixtral", "bakllava", "minicpm")


def is_vision_capable_model(model: str) -> bool:
    """Best-effort, name-based check for whether ``model`` is vision-capable.

    Matches Ollama tag substrings used by known vision-language models --
    e.g. ``qwen2.5vl:7b``, ``llama3.2-vision``, ``llava``, ``moondream``,
    ``pixtral``, ``bakllava``, ``minicpm-v`` -- case-insensitively, against
    the full model string (name and tag). Returns ``False`` for anything
    else, including text-only models like ``qwen3:4b``.

    This is NOT a ground-truth capability list (see the module-level
    comment above ``_VISION_MODEL_NAME_MARKERS``): a genuinely vision-
    capable model with an unrecognized name (digest-pinned reference,
    custom re-tag, or a VLM family not yet listed) will get a false
    ``False`` here. Callers needing an escape hatch for that case should
    consult ``Settings.copilot_vision_model_capability_check``, not extend
    this function's matching.
    """
    normalized = model.strip().lower()
    return any(marker in normalized for marker in _VISION_MODEL_NAME_MARKERS)


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
        embedding_model: str = "nomic-embed-text",
        max_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._model = model
        self._embedding_model = embedding_model
        self._max_retries = max_retries
        # Side channel of per-call timing/token stats -- see ``LlmCallStats``.
        # Public (not ``_``-prefixed): ``app.planner``/``app.extraction`` read
        # it after invoking ``chat``/``extract`` to build ``llm`` trace spans.
        self.call_stats: list[LlmCallStats] = []

    @property
    def model(self) -> str:
        """The chat/extraction model this client was built with -- read by
        ``app.supervisor.IntakeExtractorWorker`` (issue #204) to fail closed
        when it is not vision-capable, via ``is_vision_capable_model``."""
        return self._model

    @classmethod
    def from_settings(cls, settings: Settings, *, model: str | None = None) -> OllamaClient:
        """Build a production client, threading base URL, model, timeout, and
        retries. ``model`` overrides ``settings.ollama_model`` when supplied --
        issue #204 uses this to build a SEPARATE client for the vision role
        (``settings.copilot_vision_model``) without duplicating the base_url/
        timeout/retries wiring, and without touching ``ollama_model``'s own
        default or its text-rollback meaning."""
        client = httpx.Client(timeout=settings.ollama_api_timeout_seconds)
        return cls(
            base_url=settings.ollama_base_url,
            client=client,
            model=model if model is not None else settings.ollama_model,
            embedding_model=settings.ollama_embedding_model,
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
        merged_options = _clamped_chat_options(options)
        body = self._build_body(messages, stream=True, options=merged_options)
        start_ts = time.time()
        try:
            response = self._post(_CHAT_PATH, body)
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

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Send a chat request and yield response text deltas as they arrive.

        Same request shape as ``chat()`` (``stream: true``, ``think: false``,
        ``temperature: 0`` unless overridden) -- this is the incremental
        sibling for a caller (P213, ``app.planner``'s two-call final-answer
        step) that wants to render the model's free-text reasoning as it is
        generated instead of waiting for the whole response.

        Leaked-``<think>`` safety (hard constraint): NOTHING is yielded until
        the ``</think>`` boundary is resolved -- see the module docstring's
        "Live-verified quirk" note and ``_strip_leaked_thinking``, which this
        reproduces incrementally instead of via one post-hoc regex sub.
        Concretely: every arriving piece is buffered until the accumulated
        buffer contains ``</think>``, at which point everything up to and
        including it is dropped, any purely-whitespace continuation right
        after it is also swallowed (matching ``\\s*`` in
        ``_LEAKED_THINK_RE``, even when that whitespace lands in a *later*
        chunk than the marker itself), and the first non-whitespace
        remainder is yielded -- every piece after that streams through raw,
        immediately. If ``</think>`` never appears, nothing is yielded until
        the stream ends, at which point the whole buffered content is
        yielded as one final delta: absence is only ever "confirmed" once
        the stream is known to be over, never assumed mid-stream. The joined
        output of every delta this yields is always exactly
        ``_strip_leaked_thinking(<the full response>)`` -- byte-identical to
        what ``chat()`` returns for the same request.

        Appends one ``LlmCallStats`` entry to ``call_stats``, exactly like
        ``chat()`` -- ``chat()`` itself is untouched by this method existing
        (additive only).
        """
        _logger.info("ollama chat stream call", extra={"model": self._model})
        merged_options = _clamped_chat_options(options)
        body = self._build_body(messages, stream=True, options=merged_options)
        start_ts = time.time()
        try:
            response = self._post(_CHAT_PATH, body)
            tokens_in, tokens_out = yield from self._stream_deltas(response)
        except OllamaError as exc:
            end_ts = time.time()
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
            )
            _logger.warning(
                "ollama chat stream call failed",
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

    @staticmethod
    def _iter_content_pieces(
        response: httpx.Response,
    ) -> Generator[str, None, tuple[int | None, int | None]]:
        """Iterate an NDJSON chunk stream, yielding each chunk's non-empty
        ``message.content`` string in order, and returning (via the generator
        protocol) the token counts Ollama reports on the terminal
        (``done: true``) chunk. The single source of truth for parsing the
        chunk stream: both ``_assemble_stream`` (which joins the pieces) and
        ``_stream_deltas`` (which resolves the leaked-``<think>`` boundary
        incrementally) consume this.
        """
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
            piece = message.get("content") if isinstance(message, dict) else None
            if isinstance(piece, str) and piece:
                yield piece
            if chunk.get("done") is True:
                tokens_in, tokens_out = OllamaClient._token_counts(chunk)
        return tokens_in, tokens_out

    @staticmethod
    def _stream_deltas(response: httpx.Response) -> Generator[str, None, tuple[int | None, int | None]]:
        """Parse an NDJSON chunk stream, yielding post-strip content deltas
        as they resolve. See ``chat_stream``'s docstring for the leaked-
        ``<think>`` buffering contract this implements. Returns (via
        ``yield from``'s captured value) the same token counts
        ``_assemble_stream`` returns.
        """
        pieces = OllamaClient._iter_content_pieces(response)
        buffer = ""  # accumulated while still looking for </think>
        trimming = False  # </think> found; swallowing leading whitespace of what follows
        passthrough = False  # boundary resolved; yielding every piece raw from here on
        while True:
            try:
                piece = next(pieces)
            except StopIteration as stop:
                tokens_in, tokens_out = stop.value
                break
            if passthrough:
                yield piece
                continue
            if not trimming:
                # Scan only the tail that a newly-completed marker could span
                # (the last few pre-existing chars plus this piece), not the
                # whole growing buffer -- qwen3:4b's leaked preamble can run
                # to thousands of chars over many chunks, so re-scanning from
                # offset 0 each chunk would be O(n^2).
                scan_from = max(0, len(buffer) - (len(_THINK_CLOSE) - 1))
                buffer += piece
                idx = buffer.find(_THINK_CLOSE, scan_from)
                if idx == -1:
                    continue  # boundary not resolved yet -- keep buffering
                # Everything up to and including </think> is dropped; the
                # remainder falls through to the trimming block below, which
                # swallows the leading whitespace (the ``\s*`` in _LEAKED_THINK_RE).
                piece = buffer[idx + len(_THINK_CLOSE) :]
                buffer = ""
                trimming = True
            stripped = piece.lstrip()
            if stripped:
                trimming = False
                passthrough = True
                yield stripped
            # else: pure whitespace continuation -- keep trimming.
        if not passthrough and not trimming and buffer:
            # </think> never appeared anywhere in the response -- absence is
            # confirmed now that the stream is over, so nothing was stripped.
            yield buffer
        return tokens_in, tokens_out

    def extract(
        self,
        prompt_or_messages: str | list[dict[str, str]],
        schema: type[ModelT],
        *,
        options: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> ModelT:
        """Extract ``schema`` from the model's response via constrained decoding.

        POSTs with ``format`` set to ``schema.model_json_schema()`` so Ollama
        constrains decoding to valid JSON for that schema, then parses and
        ``model_validate``s the result. If the returned content isn't valid
        JSON, or fails schema validation, retries up to ``max_retries`` total
        attempts before raising ``OllamaError``. Network/HTTP failures (a
        non-2xx status, a timeout, a connection error) are NOT retried here —
        they propagate immediately as ``OllamaError``.

        ``images`` (P3.1, document ingestion): a list of base64-encoded page
        images, attached to the LAST message's ``images`` key -- the shape
        Ollama's vision API expects alongside a user message's ``content``.
        Attached on a shallow copy of the messages (each message dict
        copied), never mutating the caller's original list/dicts. ``None``
        (the default) leaves every existing text-only caller byte-identical.
        """
        _logger.info("ollama extract call", extra={"model": self._model, "schema": schema.__name__})
        messages = self._normalize_messages(prompt_or_messages)
        request_messages: list[dict[str, Any]] = [dict(message) for message in messages]
        if images is not None:
            request_messages[-1]["images"] = images
        body = self._build_body(
            request_messages,
            stream=False,
            format=schema.model_json_schema(),
            options=options,
        )

        for attempt in range(1, self._max_retries + 1):
            start_ts = time.time()
            # Network/HTTP failures are NOT retried (see docstring): keep the
            # ``_post`` call OUT of the retry-catch below so an
            # ``OllamaError`` from it propagates immediately, after recording
            # the failed attempt's stats (symmetric with ``chat``).
            try:
                response = self._post(_CHAT_PATH, body)
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

    def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for ``text`` via ``/api/embeddings``.

        Uses ``embedding_model`` (e.g. ``nomic-embed-text``), NOT the chat
        model -- embeddings and chat/extraction are served by different
        Ollama models, so this never touches ``self._model``. Retries up to
        ``max_retries`` total attempts on a malformed response (missing/
        non-list ``embedding``); network/HTTP failures are NOT retried, same
        policy as ``extract``. Log-safe: never logs the input text or the
        returned vector, only the model name and outcome (P3.3, retrieval).
        """
        _logger.info("ollama embed call", extra={"model": self._embedding_model})
        body = {"model": self._embedding_model, "prompt": text}

        for attempt in range(1, self._max_retries + 1):
            start_ts = time.time()
            try:
                response = self._post(_EMBEDDINGS_PATH, body)
            except OllamaError as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._embedding_model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
                )
                _logger.warning(
                    "ollama embed call failed",
                    extra={
                        "model": self._embedding_model,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "duration_ms": round((end_ts - start_ts) * 1000, 1),
                    },
                )
                raise
            try:
                payload = response.json()
                vector = payload.get("embedding") if isinstance(payload, dict) else None
                if not isinstance(vector, list) or not vector or not all(isinstance(v, (int, float)) for v in vector):
                    raise ValueError("Ollama embeddings response missing a valid 'embedding' list")
            except ValueError as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._embedding_model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
                )
                will_retry = attempt < self._max_retries
                _logger.warning(
                    "ollama embed call retrying after malformed response"
                    if will_retry
                    else "ollama embed call failed after exhausting retries",
                    extra={
                        "model": self._embedding_model,
                        "attempt": attempt,
                        "max_retries": self._max_retries,
                        "error_type": type(exc).__name__,
                        "duration_ms": round((end_ts - start_ts) * 1000, 1),
                    },
                )
                continue
            self.call_stats.append(
                LlmCallStats(model=self._embedding_model, start_ts=start_ts, end_ts=time.time(), ok=True, tokens_in=None, tokens_out=None)
            )
            return [float(v) for v in vector]

        raise OllamaError(f"embedding call failed after {self._max_retries} attempts")

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        url = f"{self._base_url}{path}"
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
        pieces = OllamaClient._iter_content_pieces(response)
        while True:
            try:
                parts.append(next(pieces))
            except StopIteration as stop:
                tokens_in, tokens_out = stop.value
                break
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

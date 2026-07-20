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
  * ``max_tokens`` is capped on EVERY request, but ``extract`` uses a LARGER
    cap than ``chat``/``chat_stream`` (P5.3, issue #85). A JSON schema alone
    does not bound generation length -- an open-ended schema string field
    can still run away (measured: one extraction call ran 305s and hit the
    request timeout even with a schema in effect), so a cap is unconditional
    on every call. ``chat``/``chat_stream`` keep the original ``1536`` cap
    (that 305s runaway-generation measurement -- do not relax this path).
    ``extract`` gets its own, larger ``_EXTRACT_MAX_TOKENS`` (see its
    definition below for the sizing rationale): claim extraction's required
    JSON output echoes retrieved guideline-chunk/patient-fact text verbatim
    into ``quote_or_value`` fields (``app.extraction._build_guideline_catalog``
    /``_build_fact_catalog``), which can exceed 1536 tokens once real
    retrieval (unlike the eval harness's tiny hand-authored fixtures) hands
    back multiple full corpus sections -- hitting the cap mid-string then
    fails JSON parsing/schema validation, exhausts retries, and
    ``ClaimExtractor.extract_claims`` fails closed to ``[]`` (see
    ``app.extraction``).
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

# Hard cap applied to chat()/chat_stream() -- see module docstring. Left
# unchanged from the original 305s-runaway-generation measurement.
_CHAT_MAX_TOKENS = 1536

# Hard cap applied to extract() ONLY (P5.3, issue #85) -- larger than
# _CHAT_MAX_TOKENS because the claim-extraction JSON output must echo
# retrieved guideline/fact text verbatim (see module docstring). Sized from
# a measured production decode throughput of ~50 tokens/sec (qwen3-8b-q5 on
# the dev GPU) against ``Settings.llama_server_api_timeout_seconds`` (60s
# default): 2560 tokens / 50 tok/s = ~51s decode, leaving ~9s of headroom
# for connection setup and prompt prefill before the 60s request timeout
# would fire -- large enough to comfortably hold several full corpus
# sections (each ~300-500 chars, ~100-150 tokens) plus the tool-result
# catalog and JSON schema overhead, without trading truncation for a
# request-timeout failure instead (LlamaServerError either way, but a
# timeout gives no finish_reason/content to diagnose from -- see
# ``_single_message_content``'s error detail).
_EXTRACT_MAX_TOKENS = 2560


def _retry_feedback_message(exc: Exception, finish_reason: str | None) -> str:
    """Issue #93 (fix 2/4): a SPECIFIC, actionable feedback message describing
    what went wrong with the previous ``extract`` attempt, appended as a new
    ``user`` turn before the next retry (see :meth:`LlamaServerClient.extract`).

    Deliberately generic to the *shape* of the failure (truncation / schema
    violation / invalid JSON), not to any particular schema's field
    semantics -- this client has no notion of what ``Claim``/``PlannerDecision``
    mean, only what pydantic/JSON told it went wrong. Never echoes the
    model's actual (possibly PHI/corpus-bearing) prior output -- only
    validation metadata (field paths, finish_reason), matching this module's
    existing log-safety discipline (see module docstring, ``LlamaServerError``).
    """
    if finish_reason == "length":
        return (
            "Your previous response was cut off before it finished -- it ran past the "
            "token limit before completing valid JSON. Respond again with the SAME "
            "schema, but be more concise: keep every quoted value/summary short and "
            "complete rather than verbose, so the whole response fits."
        )
    if isinstance(exc, ValidationError):
        field_paths = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        fields = ", ".join(field_paths) if field_paths else "one or more fields"
        return (
            "Your previous response did not match the required schema -- the problem "
            f"field(s): {fields}. Respond again with valid JSON that matches the schema "
            "exactly: every required field must be present and of the correct type, and "
            "every claim must carry at least one citation if it asserts a specific "
            "fact from the provided data."
        )
    return (
        "Your previous response was not valid JSON. Respond again with ONLY valid JSON "
        "matching the schema -- no extra prose, no markdown code fences, no commentary "
        "before or after the JSON object."
    )


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
            content, tokens_in, tokens_out, _finish_reason = self._single_message_content(response)
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
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": True},
        }

        # Issue #93 (fix 2/4): ``retry_messages`` starts as a copy of the
        # caller's original messages and grows by one feedback ``user``
        # turn per failed attempt (see ``_retry_feedback_message`` below) --
        # so attempt 2+ is not a blind re-roll of attempt 1's exact prompt,
        # it tells the model SPECIFICALLY what was wrong (invalid JSON /
        # schema-validation failure naming the offending field path /
        # truncated output) and asks for a fix, giving each retry a genuinely
        # different (and better-informed) chance to succeed.
        retry_messages = list(messages)
        last_finish_reason: str | None = None
        last_content_len: int | None = None
        for attempt in range(1, self._max_retries + 1):
            body = self._build_body(
                retry_messages,
                stream=False,
                options=options,
                response_format=response_format,
                max_tokens=_EXTRACT_MAX_TOKENS,
            )
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
            finish_reason: str | None = None
            content_len: int | None = None
            try:
                content, tokens_in, tokens_out, finish_reason = self._single_message_content(response)
                content_len = len(content)
                payload = json.loads(content)
                result = schema.model_validate(payload)
            except (LlamaServerError, ValueError, ValidationError) as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=tokens_in, tokens_out=tokens_out)
                )
                will_retry = attempt < self._max_retries
                # Diagnostic signal (P5.3, issue #85): finish_reason and
                # content length -- NOT the raw content itself, which may
                # embed retrieved corpus/patient-fact text -- so this failure
                # class (in particular truncation, finish_reason == "length")
                # is diagnosable from logs alone, without a live re-probe.
                # Log-safe: mirrors ``LlamaServerError``'s own "never the raw
                # model output" contract (see module docstring).
                last_finish_reason = finish_reason
                last_content_len = content_len
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
                        "finish_reason": finish_reason,
                        "content_len": content_len,
                    },
                )
                if will_retry:
                    retry_messages = retry_messages + [
                        {"role": "user", "content": _retry_feedback_message(exc, finish_reason)}
                    ]
                continue
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=time.time(), ok=True, tokens_in=tokens_in, tokens_out=tokens_out)
            )
            return result

        raise LlamaServerError(
            f"constrained extraction failed after {self._max_retries} attempts "
            f"(last finish_reason={last_finish_reason!r}, last content_len={last_content_len!r})"
        )

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
        max_tokens: int = _CHAT_MAX_TOKENS,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "temperature": 0,
            # Hard cap on every call -- see module docstring. Callers pass
            # ``_EXTRACT_MAX_TOKENS`` for extract(); every other caller keeps
            # the default ``_CHAT_MAX_TOKENS``.
            "max_tokens": max_tokens,
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
    def _single_message_content(response: httpx.Response) -> tuple[str, int | None, int | None, str | None]:
        """Returns ``(content, tokens_in, tokens_out, finish_reason)``.

        ``finish_reason`` (P5.3, issue #85) is llama.cpp's own signal for why
        generation stopped -- ``"stop"`` (natural end) vs ``"length"`` (hit
        ``max_tokens`` mid-generation, the truncation failure mode this issue
        addresses) vs other engine-specific values. Callers that need to
        diagnose a malformed/truncated extraction thread this through into
        ``LlamaServerError`` (see :meth:`extract`) rather than needing a live
        re-probe to find out which one happened.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlamaServerError("llama-server response was not valid JSON") from exc

        choices = payload.get("choices") if isinstance(payload, dict) else None
        message = choices[0].get("message") if choices else None
        finish_reason = choices[0].get("finish_reason") if choices else None
        finish_reason = finish_reason if isinstance(finish_reason, str) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LlamaServerError("llama-server response missing message content")
        usage = payload.get("usage") if isinstance(payload, dict) else None
        tokens_in = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        tokens_out = usage.get("completion_tokens") if isinstance(usage, dict) else None
        tokens_in = tokens_in if isinstance(tokens_in, int) else None
        tokens_out = tokens_out if isinstance(tokens_out, int) else None
        return LlamaServerClient._strip_leaked_thinking(content), tokens_in, tokens_out, finish_reason

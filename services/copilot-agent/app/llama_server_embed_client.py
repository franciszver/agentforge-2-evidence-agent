"""llama-server dense-embeddings client (P3.10b, epic #52 step 2): migrates
``nomic-embed-text`` embeddings off Ollama onto a SECOND, dedicated llama.cpp
``llama-server`` instance running in ``--embedding`` mode (see
``docker-compose.copilot.yml``'s ``llama-server-embed`` service).

Scope: this client serves ONLY the dense-embedding role
(``app.retrieval.Embedder`` -- hybrid guideline-corpus retrieval). It is a
separate service/client from ``app.llama_server_client.LlamaServerClient``
(chat/extract/rerank, P3.10a) because the two run as distinct containers
with distinct GGUF weights and distinct launch flags (``--embedding``
``--pooling mean`` here vs. plain chat decoding there) -- there is no shared
wire protocol to duck-type against, unlike the OllamaClient/LlamaServerClient
pair.

Vector contract parity with ``OllamaClient.embed`` (P3.10b's critical
constraint, see the corpus retrieval golden-set parity check recorded in
the PR description): both engines serve the SAME nomic-embed-text(-v1.5)
GGUF weights (pinned revision in ``docker-compose.copilot.yml``), so the
returned vectors are the same 768-dim nomic embedding space. Returned as a
plain, UN-normalized ``list[float]`` -- exactly what ``OllamaClient.embed``
returns today. Normalization differences between the two engines (if any)
never change retrieval ranking: ``app.retrieval._cosine_similarity``
divides by each vector's own norm, so cosine similarity is scale-invariant
per vector.

Wire protocol: llama-server's OpenAI-compatible ``/v1/embeddings`` endpoint
(``{"model": ..., "input": <text>}`` -> ``{"data": [{"embedding": [...]}]}``),
NOT Ollama's native ``/api/embeddings`` -- a different JSON shape than
``OllamaClient.embed`` speaks, but this class duck-types the SAME
``embed(text) -> list[float]`` signature (``app.retrieval.Embedder``
Protocol) so callers can swap embedders with no other code change.

Retry/network-error policy mirrors ``OllamaClient.embed`` exactly: up to
``max_retries`` TOTAL attempts on a malformed response (missing/non-list
``embedding``); network/HTTP failures are NOT retried here, they propagate
immediately.

Log-safe: ``LlamaServerEmbedError`` messages never embed raw input text or
the returned vector, only a fixed operation label and, where relevant, the
HTTP status code (mirrors ``OllamaError``/``LlamaServerError``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import Settings
from app.ollama_client import LlmCallStats

_logger = logging.getLogger(__name__)

_EMBEDDINGS_PATH = "/v1/embeddings"


class LlamaServerEmbedError(Exception):
    """Raised when a llama-server embeddings request fails or returns a
    malformed response. Log-safe: never embeds raw input text or the
    returned vector."""


class LlamaServerEmbedClient:
    """Dense-embedding client for a llama.cpp ``llama-server`` instance
    running in ``--embedding`` mode, satisfying ``app.retrieval.Embedder``.

    Args:
        base_url: Origin of the embedding llama-server instance, e.g.
            ``"http://llama-server-embed:8080"``.
        client: An injectable ``httpx.Client`` -- hermetic tests inject one
            backed by ``httpx.MockTransport``; production injects one via
            :meth:`from_settings`.
        model: Model name/id sent in the request body. llama-server running
            a single ``--model`` file ignores this for routing, but the
            OpenAI-compatible endpoint still requires the field present.
        max_retries: Max attempts :meth:`embed` makes before raising when
            the server's response is missing/malformed (a total-attempts
            count, matching ``OllamaClient.embed``).
    """

    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.Client,
        model: str = "nomic-embed-text-v1.5",
        max_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._model = model
        self._max_retries = max_retries
        # Side channel of per-call timing/token stats -- see
        # ``app.ollama_client.LlmCallStats``. Reused directly (not
        # redefined) so callers reading ``call_stats`` off any embedder get
        # the same type regardless of which engine served the call.
        self.call_stats: list[LlmCallStats] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> LlamaServerEmbedClient:
        """Build a production client, threading base URL, model, timeout, and retries."""
        client = httpx.Client(timeout=settings.llama_server_embed_api_timeout_seconds)
        return cls(
            base_url=settings.llama_server_embed_base_url,
            client=client,
            model=settings.llama_server_embed_model,
            max_retries=settings.llama_server_embed_max_retries,
        )

    def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for ``text`` via ``/v1/embeddings``.

        Retries up to ``max_retries`` total attempts on a malformed response
        (missing/non-list ``embedding``); network/HTTP failures are NOT
        retried. Log-safe: never logs the input text or the returned
        vector, only the model name and outcome.
        """
        _logger.info("llama-server embed call", extra={"model": self._model})
        body = {"model": self._model, "input": text}

        for attempt in range(1, self._max_retries + 1):
            start_ts = time.time()
            try:
                response = self._post(_EMBEDDINGS_PATH, body)
            except LlamaServerEmbedError as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
                )
                _logger.warning(
                    "llama-server embed call failed",
                    extra={
                        "model": self._model,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "duration_ms": round((end_ts - start_ts) * 1000, 1),
                    },
                )
                raise
            try:
                payload = response.json()
                data = payload.get("data") if isinstance(payload, dict) else None
                first = data[0] if isinstance(data, list) and data else None
                vector = first.get("embedding") if isinstance(first, dict) else None
                if not isinstance(vector, list) or not vector or not all(isinstance(v, (int, float)) for v in vector):
                    raise ValueError("llama-server embeddings response missing a valid 'embedding' list")
            except ValueError as exc:
                end_ts = time.time()
                self.call_stats.append(
                    LlmCallStats(model=self._model, start_ts=start_ts, end_ts=end_ts, ok=False, tokens_in=None, tokens_out=None)
                )
                will_retry = attempt < self._max_retries
                _logger.warning(
                    "llama-server embed call retrying after malformed response"
                    if will_retry
                    else "llama-server embed call failed after exhausting retries",
                    extra={
                        "model": self._model,
                        "attempt": attempt,
                        "max_retries": self._max_retries,
                        "error_type": type(exc).__name__,
                        "duration_ms": round((end_ts - start_ts) * 1000, 1),
                    },
                )
                continue
            self.call_stats.append(
                LlmCallStats(model=self._model, start_ts=start_ts, end_ts=time.time(), ok=True, tokens_in=None, tokens_out=None)
            )
            return [float(v) for v in vector]

        raise LlamaServerEmbedError(f"embedding call failed after {self._max_retries} attempts")

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        url = f"{self._base_url}{path}"
        try:
            response = self._client.post(url, json=body)
        except httpx.TimeoutException as exc:
            raise LlamaServerEmbedError("llama-server embeddings request timed out") from exc
        except httpx.HTTPError as exc:
            raise LlamaServerEmbedError("llama-server embeddings request failed") from exc

        if not response.is_success:
            raise LlamaServerEmbedError(f"llama-server embeddings request failed (status {response.status_code})")
        return response

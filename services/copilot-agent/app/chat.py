"""SSE ``POST /chat`` endpoint: multi-turn conversation over the planner (P2.10).

Route decision: the static shell page lives at ``GET /chat`` (P0.6) and this
SSE stream lives at ``POST /chat`` -- same path, different HTTP method, which
FastAPI dispatches independently, so both work without a clash. The shell's
``<form>`` posts back to ``/chat``.

Auth: the bearer token is validated through an injectable ``TokenValidator``
seam (``get_token_validator``). The default implementation is a stub that
only checks the token is non-empty -- TODO: replace with real OpenEMR token
introspection. A missing header or a validator rejection both produce a 401
before the planner is ever constructed or invoked.

Multi-turn state: an in-memory ``ConversationStore`` keyed by
``conversation_id``, binding each conversation to the ``patient_id`` it was
created with. Resuming with a ``conversation_id`` bound to a different
``patient_id`` is rejected (409) -- defense-in-depth for the patient-context
binding the planner itself already enforces (see ``app.planner`` module
docstring). This in-memory store is kept as-is (P4.2 does not replace it --
conversation *content* and durable *trace* data are different concerns; see
``app.trace_store``): a durable, queryable ``TraceStore`` (P4.2) is wired in
alongside it and records a **request** span (whole invocation), a
**verification** span (the P3.7 verdict fold), a **tool** span per planner
tool dispatch, and an **llm** span per completed Ollama call (#149), all
keyed by the SAME correlation id ``Turn`` already carries. Tool timing comes
from ``app.planner.ToolCallTrace.start_ts``/``end_ts`` (its ``error`` field
doubles as the tool span's ``error_category`` -- already a closed-set string,
see ``app.planner`` module docstring); LLM timing/tokens come from
``PlannerResult.llm_calls`` and the claim extractor's own ``llm_calls`` (both
``OllamaClient.call_stats`` side channels -- see ``app.ollama_client
.LlmCallStats``). See ``_emit_llm_spans`` and
``.record_feedback_span`` (P4.3's ``/feedback`` endpoint seam, separately
wired).

SSE frame contract (``ChatEvent`` -- the P2.14/P3.8 UI's source of truth):
  * ``conversation`` -- first frame, carries ``{"conversation_id": str,
                         "correlation_id": str}``. ``correlation_id`` is the
                         P4.1 id for THIS turn (``app.correlation.
                         get_correlation_id()``) -- the P4.4 UI's only way to
                         learn it, so a thumbs up/down on this response can be
                         posted to ``POST /feedback`` (P4.3) linked to it.
  * ``tool_call``    -- one per planner tool dispatch, in order, carrying
                         ``{"tool": str, "args": dict, "error": str | None}``.
  * ``reasoning_delta`` -- zero or more, BEFORE the ``answer`` frame: one per
                         incremental piece of the model's free-text reasoning
                         (``app.planner.ReasoningDelta``, P213), carrying
                         ``{"text": str}``. This is UNVERIFIED, provisional
                         text -- the UI renders it into a separate "thinking"
                         surface, NEVER the answer slot. Emitted only on the
                         streaming path (``run_streaming``); the fallback
                         replay path (a planner double implementing only
                         ``run()``) never emits it, same as ``tool_call``.
  * ``answer``        -- the final, VERIFIED answer, ``{"answer": str}`` --
                         always the post-extraction ``FinalAnswer`` text,
                         never reasoning-delta text.
  * ``verification`` -- the P3.8 verification result for this response (verdict
                         badge, citation chips, warning banner). See
                         ``build_verification_payload`` for the payload shape.
                         Populated live by ``app.extraction.run_verification``:
                         the planner's answer is decomposed into cited claims,
                         each re-validated against the RAW records
                         (deterministic, no LLM) and stripped if unverifiable,
                         then folded with the allergy / interaction checks into
                         the whole-answer verdict. An answer with no surviving
                         claims fails closed to ``blocked`` (P3.7).
  * ``done``           -- terminal frame, ``{}``.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol

from fastapi import Depends, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.correlation import get_correlation_id, get_span_id
from app.dev_token_bridge import DevTokenBridge
from app.extraction import (
    ClaimExtractor,
    ClaimExtractorLike,
    apply_recency_notice,
    apply_subject_check,
    clarify_unresolvable_referent,
    cross_patient_refusal_result,
    detect_foreign_patient_reference,
    run_verification,
)
from app.encounter_observability import build_encounter_record
from app.ingestion import LocalIngestionStore
from app.introspection import TokenIntrospector
from app.launch_binding import LaunchPatientBinder, LaunchPatientMismatchError
from app.llama_server_client import LlamaServerClient
from app.llama_server_embed_client import LlamaServerEmbedClient
from app.ollama_client import LlmCallStats, OllamaClient
from app.openemr_auth import IntrospectionResult
from app.openemr_client import OpenEmrClient
from app.planner import Planner, PlannerCompleted, PlannerResult, ReasoningDelta, ToolCallTrace, ToolDispatched
from app.rendering import RenderedAnswer, RenderedClaim
from app.reranking import OllamaRerankScorer, Reranker
from app.retrieval import build_retriever_from_corpus
from app.schemas.ingestion import Citation
from app.schemas.reranking import RerankedChunk
from app.semantic_support import SemanticSupportJudgeLike
from app.supervisor import EvidenceRetrieverWorker, IntakeExtractorWorker, RetrieveSubTask, Supervisor
from app.trace_store import TraceStore, record_span_best_effort
from app.verdict import VerdictResult, to_trace_record

_logger = logging.getLogger(__name__)


class ChatEvent(StrEnum):
    """SSE event names emitted by ``POST /chat``."""

    CONVERSATION = "conversation"
    TOOL_CALL = "tool_call"
    REASONING_DELTA = "reasoning_delta"
    ANSWER = "answer"
    VERIFICATION = "verification"
    DONE = "done"


# DoS guard (issue #167, VULN-0004): the largest ``message`` POST /chat will
# accept before rejecting the request with a 422, mirroring the precedent
# app.feedback.MAX_COMMENT_LENGTH already sets for FeedbackRequest.comment.
# 4000 characters (~1,000 tokens at ~4 chars/token) comfortably covers any
# legitimate clinical question -- including a long, detailed one -- while
# leaving the bulk of the target model's 16k-token context window for the
# planner/system prompt, retrieved evidence chunks, prior conversation
# turns, and the answer itself. This bound applies to the WHOLE request
# message unconditionally; it is independent of, and tighter than, nothing
# -- app.retrieval.MAX_QUERY_CHARS (2000) is a SEPARATE, narrower bound that
# only applies to the derived retrieval query, and only when
# copilot_evidence_retrieval_enabled is true. Rejecting outright (422) is
# deliberate, not truncating: a silently truncated question could change
# clinical meaning without the caller knowing.
MAX_CHAT_MESSAGE_LENGTH = 4000


class ChatRequest(BaseModel):
    """``POST /chat`` request body."""

    message: str = Field(max_length=MAX_CHAT_MESSAGE_LENGTH)
    patient_id: int
    conversation_id: str | None = None


class TokenValidationError(Exception):
    """Raised by a ``TokenValidator`` when a bearer token is invalid."""


class PatientMismatchError(Exception):
    """Raised when a conversation is resumed with a mismatched ``patient_id``.

    Not raised across the FastAPI boundary today (the endpoint maps the
    mismatch directly to a 409); kept as a named type so callers embedding
    ``ConversationStore`` outside the endpoint have a typed error to catch.
    """


class PlannerProtocol(Protocol):
    """What the endpoint needs from a planner: ``run(question) -> PlannerResult``.

    ``app.planner.Planner`` satisfies this; hermetic tests inject a scripted
    fake instead.

    ``guideline_excerpts`` (#105): optional retrieved guideline-corpus chunk
    text, fed into answer composition -- see ``app.planner.Planner.run``'s
    docstring. Defaulted so a test double that ignores it (every existing
    fake planner) stays valid.

    ``document_facts`` (#86): optional patient-scoped ingested document fact
    citations, also fed into answer composition -- see
    ``app.planner.Planner.run``'s docstring. Also defaulted, and
    ``_stream_chat`` only ever passes it as an explicit keyword argument when
    non-empty (mirrors ``app.extraction.run_verification``'s own
    conditional-kwarg convention for ``retrieved_chunks``/``patient_facts``),
    so every existing fake planner (which never receives it) stays valid
    unmodified.
    """

    def run(
        self,
        question: str,
        guideline_excerpts: Sequence[str] | None = None,
        document_facts: Sequence[Citation] | None = None,
    ) -> PlannerResult: ...


TokenValidator = Callable[[str], None]
PlannerFactory = Callable[[int], PlannerProtocol]
# Wall-clock seam for the #153 recency notice: production reads the real UTC
# clock, hermetic tests inject a fixed instant (mirroring the eval harness's
# ``_EVAL_FIXED_NOW``). Returns a tz-AWARE UTC datetime so the recency
# comparison is well-defined against tz-aware OpenEMR/FHIR record dates (see
# ``app.verification._as_aware_utc``).
Clock = Callable[[], datetime]


def _default_token_validator(token: str) -> None:
    """Stub token validator: accepts any non-empty token.

    The flag-OFF default (``copilot_per_user_token_enabled=False``). Replaced by
    the introspection validator when the #124 Phase 4 flag is on.
    """
    if not token:
        raise TokenValidationError("missing bearer token")


class Introspector(Protocol):
    """What :func:`build_introspection_validator` needs: token -> result."""

    def introspect(self, token: str) -> IntrospectionResult: ...


class _IntrospectionValidator:
    """Callable ``TokenValidator`` backed by an ``Introspector``.

    Accepts a token only if introspection reports it ``active`` and (when
    ``exp`` is present) not yet expired. Empty tokens are rejected before any
    introspection round-trip. Every rejection raises ``TokenValidationError``
    -> mapped to 401 by the endpoint, before the planner is built.

    Exposes ``peek_cached`` -- a duck-typed *optional* capability (the same
    pattern as ``Planner.run_streaming``, see this module's docstring) that
    ``_validate_token`` (#185) looks up via ``getattr`` to decide whether a
    call can stay on the event loop (a cache hit) or must be dispatched to
    FastAPI's threadpool (a cache miss, which makes a real HTTP call). A
    validator built over an ``Introspector`` double that has no
    ``peek_cached`` of its own (e.g. a test fake) simply always reports a
    miss -- correct, just not the fast path.
    """

    def __init__(self, introspector: Introspector, clock: Callable[[], float]) -> None:
        self._introspector = introspector
        self._clock = clock

    def __call__(self, token: str) -> None:
        if not token:
            raise TokenValidationError("missing bearer token")
        result = self._introspector.introspect(token)
        if not result.active:
            raise TokenValidationError("token is not active")
        if result.exp is not None and result.exp <= self._clock():
            raise TokenValidationError("token has expired")

    def peek_cached(self, token: str) -> IntrospectionResult | None:
        peek = getattr(self._introspector, "peek_cached", None)
        return peek(token) if peek is not None else None


def build_introspection_validator(
    introspector: Introspector, *, clock: Callable[[], float] = time.time
) -> TokenValidator:
    """Build a ``TokenValidator`` that accepts a token only if introspection
    reports it ``active`` and (when ``exp`` is present) not yet expired.

    Empty tokens are rejected before any introspection round-trip. Every
    rejection raises ``TokenValidationError`` -> mapped to 401 by the endpoint,
    before the planner is built.
    """
    return _IntrospectionValidator(introspector, clock)


async def _validate_token(validator: TokenValidator, token: str) -> None:
    """Invoke ``validator(token)``, dispatching to FastAPI's threadpool only
    when necessary (#185).

    A cache-MISS introspection makes a real, blocking HTTP call and must not
    occupy the event loop inside the ``async`` ``chat_endpoint`` body -- so it
    is run via ``run_in_threadpool``, the same mechanism ``get_planner_factory``
    already relies on for the (sync) planner-factory dependency. A cache-HIT
    (or the flag-off stub, which does no I/O at all) is cheap enough to call
    directly, in-loop -- a threadpool round trip there would be pure overhead
    on the common path.

    ``validator`` optionally exposes ``peek_cached`` (see
    ``_IntrospectionValidator``); its absence (the flag-off stub, or any
    ``TokenValidator`` double without it) means "no fast path available" ->
    call in-loop, byte-identical to before this dispatcher existed.
    """
    peek_cached = getattr(validator, "peek_cached", None)
    if peek_cached is None or peek_cached(token) is not None:
        validator(token)
        return
    await run_in_threadpool(validator, token)


_token_introspector: TokenIntrospector | None = None


def get_token_introspector() -> TokenIntrospector:
    """The process-wide ``TokenIntrospector`` (holds the hash-keyed TTL cache).

    Built lazily and reused so the introspection cache survives across requests.
    """
    global _token_introspector
    if _token_introspector is None:
        _token_introspector = TokenIntrospector.from_settings(get_settings())
    return _token_introspector


def get_token_validator() -> TokenValidator:
    """FastAPI dependency: the active ``TokenValidator``. Override in tests.

    Flag ON (``copilot_per_user_token_enabled``): validates the forwarded
    per-user bearer via OpenEMR introspection. Flag OFF: the non-empty stub,
    byte-identical to today.
    """
    if get_settings().copilot_per_user_token_enabled:
        return build_introspection_validator(get_token_introspector())
    return _default_token_validator


LaunchBindingChecker = Callable[[str, int], None]


def _default_launch_binding_checker(token: str, patient_id: int) -> None:
    """Flag-OFF default: no-op. Keeps /chat byte-identical to today -- the
    token-launch patient binding is a flag-on (#124 Phase 5) layer."""
    return None


_launch_patient_binder: LaunchPatientBinder | None = None


def get_launch_patient_binder() -> LaunchPatientBinder:
    """The process-wide ``LaunchPatientBinder``.

    Built lazily and given the SAME ``TokenIntrospector`` the token validator
    uses, so its introspection is a cache hit -- only the pid->uuid resolve
    read is an extra round trip.
    """
    global _launch_patient_binder
    if _launch_patient_binder is None:
        _launch_patient_binder = LaunchPatientBinder.from_settings(
            get_settings(), get_token_introspector()
        )
    return _launch_patient_binder


def get_launch_binding_checker() -> LaunchBindingChecker:
    """FastAPI dependency: the active launch-patient binding check. Override in tests.

    Flag ON (``copilot_per_user_token_enabled``, #124 Phase 5): verifies the
    token's SMART launch patient matches ``request.patient_id`` before the
    planner runs. Flag OFF: a no-op, so /chat is byte-identical to today.
    """
    if get_settings().copilot_per_user_token_enabled:
        return get_launch_patient_binder().verify
    return _default_launch_binding_checker


_dev_token_bridge: DevTokenBridge | None = None


def get_dev_token_bridge() -> DevTokenBridge:
    """FastAPI dependency: the process-wide ``DevTokenBridge``. Override in tests.

    Built lazily and reused so the real OpenEMR token is cached across
    requests (the bridge holds the in-memory TTL cache).
    """
    global _dev_token_bridge
    if _dev_token_bridge is None:
        _dev_token_bridge = DevTokenBridge.from_settings(get_settings())
    return _dev_token_bridge


_LLAMA_SERVER_ENGINE = "llama_server"


def _wants_llama_server(settings: Settings) -> bool:
    """The single place ``settings.copilot_llm_engine``'s value is compared --
    both ``get_text_llm_client`` and ``_build_evidence_workers`` call this
    rather than each running their own ``== "llama_server"`` check, so a
    future third engine value (or a typo fix) can't drift between the two
    call sites."""
    return settings.copilot_llm_engine == _LLAMA_SERVER_ENGINE


def _wants_llama_server_embed(settings: Settings) -> bool:
    """The single place ``settings.copilot_embed_engine``'s value is compared
    (P3.10b, epic #52 step 2) -- a DEDICATED flag from ``copilot_llm_engine``:
    the embed and answer/extract/reranker roles are independently
    rollback-able, so one flag's value must never be inferred from the
    other's."""
    return settings.copilot_embed_engine == _LLAMA_SERVER_ENGINE


def get_text_llm_client(settings: Settings) -> OllamaClient | LlamaServerClient:
    """Build the client for the text-generation LLM roles -- planner
    chat/extract, claim extraction, and the LLM-as-reranker relevance score
    -- selected by ``settings.copilot_llm_engine`` (P3.10a, epic #52 step 1).

    Embeddings and vision-based document-ingestion extraction are NOT among
    these roles and always use ``OllamaClient`` regardless of this flag --
    see ``_build_evidence_workers``, which constructs its own separate,
    always-Ollama client for those.
    """
    if _wants_llama_server(settings):
        return LlamaServerClient.from_settings(settings)
    return OllamaClient.from_settings(settings)


def _default_planner_factory(token: str) -> PlannerFactory:
    """Build the production planner factory bound to one real OpenEMR ``token``.

    ``token`` is a REAL OpenEMR token obtained server-side by the
    ``DevTokenBridge`` (finding F4 / issue #126) -- NOT the browser's
    ``DevAgentToken`` (an HMAC identity assertion), which never reaches tool
    calls: this factory chain has no access to it at all. The browser token
    still gates the request and carries the pid for patient-context binding
    upstream in ``chat_endpoint``.

    Identity for ACL is the bridge's configured demo clinician until #124
    (production ``authorization_code``, per-user tokens) lands. A tool call
    made with an expired/rejected token still fails per-call (caught as
    ``OpenEmrApiError`` in the planner loop) without crashing the conversation.
    """
    settings = get_settings()

    def factory(patient_id: int) -> PlannerProtocol:
        return Planner(
            ollama_client=get_text_llm_client(settings),
            openemr_client=OpenEmrClient.from_settings(settings),
            token=token,
            patient_id=patient_id,
        )

    return factory


def get_planner_factory(
    authorization: str | None = Header(default=None),
    dev_token_bridge: DevTokenBridge = Depends(get_dev_token_bridge),
) -> PlannerFactory:
    """FastAPI dependency: builds a ``PlannerProtocol`` for a patient_id. Override in tests.

    Flag ON (``copilot_per_user_token_enabled``, #124 Phase 4): the planner is
    bound to the REQUEST's own forwarded bearer, so OpenEMR maps every tool call
    to that user -> per-user ACL. This dependency resolves BEFORE the endpoint
    body validates the token, so a missing/malformed header must NOT raise here
    (that would surface as a 500); it binds an empty token and the body's
    validator then rejects with 401 -- the planner is only ever *run* after
    validation passes, so an unvalidated token never reaches a tool call.

    Flag OFF: byte-identical to today -- the ``DevTokenBridge``'s demo-clinician
    token drives tool calls. The bridge's (potentially blocking, on a cache
    miss) token fetch happens here, in a sync dependency FastAPI runs in its
    worker-thread pool -- not in the ``async`` ``chat_endpoint`` body, so a
    token refresh never blocks the event loop.
    """
    if get_settings().copilot_per_user_token_enabled:
        try:
            token = extract_bearer_token(authorization)
        except TokenValidationError:
            token = ""
        return _default_planner_factory(token)
    return _default_planner_factory(dev_token_bridge.get_token())


def _default_clock() -> datetime:
    """Production wall clock for the #153 recency notice: the real time, UTC
    and tz-aware. Aware (not naive) so the staleness comparison against
    possibly-tz-aware OpenEMR/FHIR record dates never raises ``TypeError``."""
    return datetime.now(timezone.utc)


def get_clock() -> Clock:
    """FastAPI dependency: the wall clock for the recency notice (#153).
    Override in tests to inject a fixed instant for deterministic assertions."""
    return _default_clock


def get_claim_extractor() -> ClaimExtractorLike:
    """FastAPI dependency: the answer->claims extractor. Override in tests.

    Built with ONLY an ``OllamaClient`` -- no tool registry, no OpenEMR
    client, no token -- so the extraction LLM is structurally tool-less (see
    ``app.extraction``'s security-boundary docstring). It is a distinct
    ``OllamaClient`` from the planner's, underscoring that the extractor
    never shares the planner's tool-selecting context.
    """
    return ClaimExtractor(ollama_client=get_text_llm_client(get_settings()))


UNKNOWN_USER = "unknown"


@dataclass
class Turn:
    """One recorded conversation turn: the chart-access audit record P2.17
    requires the agent to keep per turn -- WHO asked (``user``), about WHICH
    patient (``patient_id``), under WHAT ``correlation_id`` -- plus the
    question and answer.

    ``correlation_id`` IS the P4.1 correlation id (``app.correlation``) --
    the same id bound to this request by ``CorrelationIdMiddleware`` and
    readable from every stage of this invocation (log lines, tool dispatch,
    LLM calls, verification), not a second id minted independently here.
    ``user`` is a best-effort identity
    assertion read from the dev bearer token (see ``_user_identity_from_token``
    and the module's ``DevAgentToken``), not a validated principal -- real
    token introspection is the deferred P4.1 work. The durable, DB-backed
    home for these records is P4.2; this dataclass keeps the shape a durable
    store would persist.
    """

    correlation_id: str
    user: str
    patient_id: int
    question: str
    answer: str


def _user_identity_from_token(token: str) -> str:
    """Best-effort user identity for the per-turn audit record.

    The dev bearer token (``DevAgentToken``) is
    ``base64url(payloadJson) . base64url(sig)`` and carries the logged-in
    ``username``/``sub`` claim for exactly this agent-side audit use. We read
    that claim WITHOUT verifying the signature: this is an identity assertion
    for the trace record, not an authorization decision (signature/token
    introspection is the deferred P4.1 work, and the token validator seam
    still gates the request). Returns ``UNKNOWN_USER`` when the token cannot
    be parsed into a payload with a usable identity claim.
    """
    segment = token.split(".", 1)[0]
    padded = segment + "=" * (-len(segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return UNKNOWN_USER
    if not isinstance(payload, dict):
        return UNKNOWN_USER

    username = payload.get("username")
    if isinstance(username, str) and username:
        return username
    sub = payload.get("sub")
    if isinstance(sub, (int, str)) and str(sub):
        return str(sub)
    return UNKNOWN_USER


@dataclass
class Conversation:
    """One multi-turn conversation, bound to the patient it was created for.

    ``patient_name`` (#224 name-binding) is the bound patient's own display
    name, resolved ONCE at creation time (see ``chat_endpoint``'s
    ``_resolve_conversation_patient_name`` call) -- ``None`` when it could
    not be resolved (e.g. an OpenEMR API error, or a planner double with no
    ``resolve_patient_name`` capability). Fed into ``app.extraction
    .detect_foreign_patient_reference``'s named cross-patient signals; a
    ``None`` name simply disables those signals for this conversation,
    falling back to #223's numeric-only detection.

    ``patient_roster`` (#237 roster-based detection) is every OTHER
    patient's display name, resolved LAZILY -- unlike ``patient_name``, NOT
    at conversation-creation time -- see ``_stream_chat``'s
    ``_roster_provider`` closure. ``None`` means "not yet resolved" (the
    common case: most turns never mention another patient by name);
    populated on first use and cached here so a second matching turn in the
    SAME conversation does not pay the round trip again.
    """

    conversation_id: str
    patient_id: int
    patient_name: str | None = None
    patient_roster: list[str] | None = None
    history: list[Turn] = field(default_factory=list)


# Fallback cap for a bare ``ConversationStore()`` constructed outside the
# FastAPI dependency (as every hermetic test in this repo does today) --
# kept in sync with ``Settings.copilot_max_stored_conversations``'s default
# so behaviour is identical whether or not a caller threads settings through.
DEFAULT_MAX_STORED_CONVERSATIONS = 2000


class ConversationStore:
    """In-memory, LRU-bounded conversation store keyed by ``conversation_id``.

    Issue #167 (VULN-0004): an earlier version of this store had no
    eviction, so it retained every conversation for the process lifetime --
    unbounded, attacker-influenced memory growth (any caller can start a new
    conversation). ``max_conversations`` bounds it: once the cap is
    exceeded, the least-recently-used conversation is evicted. "Used" means
    read via ``get`` or appended to via ``append_turn`` -- both move a
    conversation to the most-recently-used end, so an active multi-turn
    conversation is never evicted out from under its own caller as long as
    it stays under the cap; only conversations nobody has touched recently
    are reclaimed.

    A caller resuming with an evicted ``conversation_id`` gets ``get() ->
    None`` -- the endpoint already maps that to a clean 404 (unknown
    conversation_id), not a crash; see ``chat_endpoint``.

    TODO(P4.2): replace with the durable trace store; this is a placeholder
    with the same shape (get / create / append) a DB-backed store would have.
    """

    def __init__(self, max_conversations: int = DEFAULT_MAX_STORED_CONVERSATIONS) -> None:
        if max_conversations <= 0:
            raise ValueError("max_conversations must be positive")
        self._max_conversations = max_conversations
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()

    def get(self, conversation_id: str) -> Conversation | None:
        conversation = self._conversations.get(conversation_id)
        if conversation is not None:
            self._conversations.move_to_end(conversation_id)
        return conversation

    def create(self, patient_id: int, patient_name: str | None = None) -> Conversation:
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()), patient_id=patient_id, patient_name=patient_name
        )
        self._conversations[conversation.conversation_id] = conversation
        self._conversations.move_to_end(conversation.conversation_id)
        while len(self._conversations) > self._max_conversations:
            self._conversations.popitem(last=False)
        return conversation

    def append_turn(self, conversation_id: str, turn: Turn) -> None:
        self._conversations[conversation_id].history.append(turn)
        self._conversations.move_to_end(conversation_id)


_default_store: ConversationStore | None = None


def get_conversation_store(settings: Settings = Depends(get_settings)) -> ConversationStore:
    """FastAPI dependency: the active ``ConversationStore``. Override in tests.

    Built lazily against ``Settings.copilot_max_stored_conversations``, the
    same pattern ``get_trace_store`` uses -- importing this module never
    reads settings, and the cap is operator-tunable via that setting rather
    than a module-global read.
    """
    global _default_store
    if _default_store is None:
        _default_store = ConversationStore(max_conversations=settings.copilot_max_stored_conversations)
    return _default_store


_default_trace_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    """FastAPI dependency: the process-wide ``TraceStore`` (P4.2). Override in tests.

    Built lazily against ``Settings.trace_db_path`` so importing this module
    never touches disk; every test overrides this dependency with a
    ``TraceStore`` pointed at a ``tmp_path`` database instead (see
    ``docs/TEST_PLAN.md`` Sec 7 -- tests never write to the configured path).
    """
    global _default_trace_store
    if _default_trace_store is None:
        settings = get_settings()
        _default_trace_store = TraceStore(
            db_path=settings.trace_db_path, hash_secret=settings.trace_args_hash_secret
        )
    return _default_trace_store


EvidenceRetriever = Callable[[str], list[RerankedChunk]]


def _no_op_evidence_retriever(query: str) -> list[RerankedChunk]:
    """Flag-OFF default: no retrieval, no embedding round trip, no
    ``worker`` span -- evidence retrieval itself is fully flag-gated. (Per-turn
    encounter logging, P3.8, is always-on regardless of this flag -- see
    ``_log_encounter_record`` -- but it only ever logs non-PHI counts/timings,
    never request content, so it carries no observable behavior change for a
    caller of /chat.)"""
    return []


_evidence_workers: tuple[IntakeExtractorWorker, EvidenceRetrieverWorker] | None = None


def _build_evidence_workers(settings: Settings) -> tuple[IntakeExtractorWorker, EvidenceRetrieverWorker]:
    """Construct the P3.5 workers backing /chat's evidence-retrieval path
    (P3.9) from ``settings``. Split out of ``_get_evidence_workers`` (which
    caches the result in a process-wide singleton) so a wiring test can
    exercise construction directly, with a specific ``Settings`` instance,
    without touching that singleton.

    The intake-extractor worker is real (not a stub) -- it shares this same
    ``LocalIngestionStore``/``OllamaClient`` wiring ``app.documents`` already
    uses -- but /chat still never dispatches an ``IngestSubTask``: ingesting
    a NEW document stays a separate concern from a chat turn. Constructing a
    real worker anyway costs nothing (it is never invoked from here) and
    keeps ``Supervisor`` construction uniform. Per-patient fact citations
    (``DocumentFactIndex``) into a *live* chat turn (P3.9's original
    deferral, resolved by P3.9a / issue #46) instead reads a patient's
    ALREADY-ingested facts straight from the same ``LocalIngestionStore`` --
    see ``get_patient_fact_provider`` -- no worker dispatch needed for a
    plain, patient-scoped disk read.

    P3.10a (epic #52 step 1): the intake-extractor worker (vision-based
    document-ingestion extraction) ALWAYS uses ``OllamaClient`` here,
    regardless of either engine flag -- vision has not migrated (that is
    #52c). The reranker's LLM-as-judge relevance score is selectable via
    ``settings.copilot_llm_engine`` (``get_text_llm_client``).

    P3.10b (epic #52 step 2): the embedder (dense-vector retrieval,
    ``nomic-embed-text``) is selectable via its OWN flag,
    ``settings.copilot_embed_engine`` -- defaulting to
    ``LlamaServerEmbedClient`` (a second, dedicated llama-server instance
    running in ``--embedding`` mode) rather than ``OllamaClient``.
    """
    ollama_client = OllamaClient.from_settings(settings)
    embedder = LlamaServerEmbedClient.from_settings(settings) if _wants_llama_server_embed(settings) else ollama_client
    retriever = build_retriever_from_corpus(embedder=embedder)
    # Reuse the same ollama_client for the reranker in the default case
    # instead of paying for a second httpx.Client/connection pool -- only
    # construct a distinct client when the engine flag actually selects
    # llama-server.
    # RESOLVED (issue #99): PR #98's 0.5 floor was calibrated only against
    # app/data/reranker_scores.json (model="qwen3:4b", via Ollama), while
    # the default copilot_llm_engine is "llama_server" (qwen3-8b, see
    # llama_server_model above) -- the PRODUCTION reranker scorer below.
    # scripts/build_reranker_scores.py now supports RERANKER_ENGINE=
    # llama_server to measure the SAME golden (query, chunk) pairs against
    # qwen3-8b (recorded in app/data/reranker_scores_qwen3-8b.json). That
    # measurement raised _EVIDENCE_MIN_RELEVANCE_SCORE from 0.5 to 0.75 --
    # see its docstring for the numbers. tests/test_reranker_calibration.py
    # pins the stamped fixture model to settings.llama_server_model so a
    # future engine/model swap on either side fails loudly instead of
    # silently invalidating the floor again.
    text_llm_client = get_text_llm_client(settings) if _wants_llama_server(settings) else ollama_client
    reranker = Reranker(OllamaRerankScorer(text_llm_client))
    ingestion_store = LocalIngestionStore(settings.copilot_ingestion_base_dir)
    return (
        IntakeExtractorWorker(ollama_client=ollama_client, document_store=ingestion_store, fact_store=ingestion_store),
        EvidenceRetrieverWorker(retriever=retriever, reranker=reranker),
    )


def _get_evidence_workers() -> tuple[IntakeExtractorWorker, EvidenceRetrieverWorker]:
    """The process-wide P3.5 workers backing /chat's evidence-retrieval path
    (P3.9): built once and reused (mirrors ``get_token_introspector``'s own
    lazy-singleton pattern) since ``build_retriever_from_corpus`` parses the
    whole corpus and ``LocalIngestionStore`` is stateless config -- neither
    needs rebuilding per request. The ``Supervisor`` wrapping these (see
    ``get_evidence_retriever``) is instead built FRESH per request, so its
    spans land in THIS request's ``trace_store`` (a per-test tmp_path DB in
    tests) rather than a stale singleton's.
    """
    global _evidence_workers
    if _evidence_workers is None:
        _evidence_workers = _build_evidence_workers(get_settings())
    return _evidence_workers


_EVIDENCE_RETRIEVAL_TOP_K = 3

# Issue #93 (fix 1/4): a relevance-score floor applied to the reranked pool
# BEFORE it is handed to the claim extractor, so a weak/irrelevant chunk that
# merely survived into the top-``_EVIDENCE_RETRIEVAL_TOP_K`` never adds
# extraction-time token pressure or citation-attachment noise for no benefit.
#
# 0.5 was originally chosen (PR #98) by inspecting the score distribution
# recorded in ``app/data/reranker_scores.json`` (5 golden queries, real
# Ollama qwen3:4b pointwise relevance scores): every genuinely-relevant
# chunk scored >=0.87, everything else <=0.35 -- a clean gap with 0.5
# sitting in the middle of it.
#
# Issue #99: that fixture is NOT the model that actually scores relevance in
# production -- ``copilot_llm_engine`` defaults to "llama_server" (qwen3-8b,
# see ``llama_server_model`` in ``app/config.py``), not Ollama's qwen3:4b.
# Re-measuring the SAME golden (query, chunk) pairs against qwen3-8b
# (``RERANKER_ENGINE=llama_server scripts/build_reranker_scores.py``,
# recorded in ``app/data/reranker_scores_qwen3-8b.json``) showed the 0.5
# floor does NOT hold for the production model: qwen3-8b gives partial
# credit far more liberally than qwen3:4b, and critically, TWO of the five
# deliberately-planted lexical distractors (``scripts/reranker_golden_
# distractors.py`` -- chunks a hybrid stage ranks high but that do not
# actually answer the query) scored ABOVE 0.5:
#   * "Can I give ibuprofen with lisinopril?" distractor
#     (renal-function-monitoring#ace-inhibitors-and-arbs) scored 0.65
#     (was 0.21 under qwen3:4b).
#   * "Does warfarin interact with antibiotics?" distractor
#     (statin-monitoring#interaction-caution-cyp3a4-inhibitors) scored 0.62
#     (was 0.0 under qwen3:4b).
# A 0.5 floor would have let both distractors reach the claim extractor as
# "relevant" in production -- exactly the failure mode this filter exists to
# prevent. Across all 5 golden queries, qwen3-8b's genuinely-relevant chunks
# scored 0.85-0.95 and every deliberately-planted distractor scored <=0.65,
# a real (if narrower) gap between 0.65 and 0.85. 0.75 sits in that gap with
# margin on both sides, so it excludes every measured distractor while
# keeping every measured genuine match -- see
# ``tests/test_reranker_calibration.py`` for the regression check pinning
# this invariant to the fixture. ``top_k`` itself is left at 3 (not
# reduced) -- it remains the ceiling on the pool; this floor is what
# actually shrinks what reaches the extractor on the (common) case where
# fewer than ``top_k`` candidates are truly relevant.
_EVIDENCE_MIN_RELEVANCE_SCORE = 0.75


def _filter_by_relevance_score(
    chunks: list[RerankedChunk], min_score: float = _EVIDENCE_MIN_RELEVANCE_SCORE
) -> list[RerankedChunk]:
    """Drop reranked chunks scoring below ``min_score`` (see
    ``_EVIDENCE_MIN_RELEVANCE_SCORE`` for the threshold rationale).

    Deliberately NOT folded into ``app.reranking.Reranker.rerank`` itself --
    that primitive's own tests (``tests/test_reranking.py``) assert exact
    ``top_k`` counts and demoted-not-dropped distractor ordering, both of
    which a hard default filter there would break. This filter is specific
    to the /chat evidence-retrieval boundary (this module's ``_retrieve``
    closure below), which is the only caller that wants "fewer, better"
    rather than "exactly top_k, reordered."

    Logs every drop (score + ``chunk_id`` only -- never chunk text/content,
    matching this module's existing log-safety discipline) so a silent
    "found nothing" isn't indistinguishable from "found it and threw it
    away." The all-dropped case (every retrieved chunk fell below
    ``min_score``, so zero evidence reaches the extractor) is logged at
    WARNING -- that is the exact silent-failure shape a plain per-drop INFO
    line would still bury.
    """
    kept = [chunk for chunk in chunks if chunk.rerank_score >= min_score]
    dropped = [chunk for chunk in chunks if chunk.rerank_score < min_score]
    if dropped:
        _logger.info(
            "evidence relevance filter dropped chunks",
            extra={
                "dropped_count": len(dropped),
                "kept_count": len(kept),
                "min_score": min_score,
                "dropped": [
                    {"chunk_id": chunk.chunk_id, "score": chunk.rerank_score} for chunk in dropped
                ],
            },
        )
    if chunks and not kept:
        _logger.warning(
            "evidence relevance filter dropped all retrieved chunks; "
            "zero evidence will reach the claim extractor",
            extra={"dropped_count": len(dropped), "min_score": min_score},
        )
    return kept


def get_evidence_retriever(
    trace_store: TraceStore = Depends(get_trace_store),
) -> EvidenceRetriever:
    """FastAPI dependency: the active evidence-retrieval callable for /chat's
    P3.9 guideline-citation path. Override in tests.

    Flag ON (``copilot_evidence_retrieval_enabled``): routes the turn's
    question through the P3.5 supervisor's evidence-retriever worker (hybrid
    retrieve + rerank over the PUBLIC guideline corpus), so its handoff is
    traced (a ``worker`` span, parented under a fresh supervisor span) into
    THIS request's ``trace_store`` -- the same one ``_stream_chat`` already
    writes tool/llm/verification spans to, so ``app.encounter_observability
    .build_encounter_record`` picks it up as one more step in the encounter.

    Raises whatever the supervisor/worker raises (``RetrievalError``,
    ``RerankError``, ...) -- NOT caught here. ``_stream_chat`` is the single
    call site that catches it fail-soft (see its own comment), the same
    "caller decides how to recover" discipline ``get_launch_binding_checker``
    already uses for ``LaunchPatientMismatchError``: catching here instead
    would also hide a raising TEST double's failure from that call site.

    Flag OFF (default): ``_no_op_evidence_retriever`` -- no retrieval call, no
    embedding round trip, no worker span. Evidence retrieval itself is
    flag-gated; per-turn encounter logging (P3.8, non-PHI counts only) is
    always-on regardless of this flag -- see ``_log_encounter_record``.
    """
    if not get_settings().copilot_evidence_retrieval_enabled:
        return _no_op_evidence_retriever

    intake_worker, evidence_worker = _get_evidence_workers()
    supervisor = Supervisor(intake_worker=intake_worker, evidence_worker=evidence_worker, trace_store=trace_store)

    def _retrieve(query: str) -> list[RerankedChunk]:
        result = supervisor.handle(RetrieveSubTask(query=query, k=_EVIDENCE_RETRIEVAL_TOP_K))
        chunks: list[RerankedChunk] = result.payload
        return _filter_by_relevance_score(chunks)

    return _retrieve


SupportJudgeProvider = Callable[[], SemanticSupportJudgeLike | None]


def _no_op_support_judge_provider() -> None:
    """Flag-OFF default (issue #47): no judge, no extra LLM call --
    ``run_verification`` treats ``support_judge=None`` as "skip the
    semantic-support gate entirely," same posture as
    ``_no_op_evidence_retriever``/``_no_op_patient_fact_provider``."""
    return None


def get_support_judge_provider(settings: Settings = Depends(get_settings)) -> SupportJudgeProvider:
    """FastAPI dependency: the active semantic-support judge provider for
    /chat's issue #47 gate. Override in tests.

    Flag ON (``copilot_semantic_support_enabled``): the provider builds the
    SAME text-generation client selected by ``copilot_llm_engine``
    (``get_text_llm_client`` -- the production default is
    ``LlamaServerClient``/Qwen3-8B-Q5), so the judge runs on the same engine
    as answer generation/claim extraction. A fresh client per call mirrors
    ``get_claim_extractor``'s own per-request construction.

    Flag OFF (default): ``_no_op_support_judge_provider`` -- ``None`` every
    time, so ``run_verification`` skips the gate with zero added latency."""
    if not settings.copilot_semantic_support_enabled:
        return _no_op_support_judge_provider

    def _provide() -> SemanticSupportJudgeLike:
        return get_text_llm_client(settings)

    return _provide


def get_require_answer_grounding(settings: Settings = Depends(get_settings)) -> bool:
    """FastAPI dependency: whether ``run_verification`` should run the issue
    #153 claim-in-answer grounding gate this request. Override in tests.

    Deterministic, no LLM call -- unlike ``get_support_judge_provider`` above,
    this needs no lazily-constructed client, just the flag's current value.

    Flag ON (``copilot_claim_answer_grounding_enabled``): every claim whose
    citations already passed provenance is additionally re-checked against
    the answer text (``app.answer_grounding.apply_answer_grounding``).

    Flag OFF (default): ``False`` -- ``run_verification`` skips the gate,
    byte-identical to today."""
    return settings.copilot_claim_answer_grounding_enabled


def get_require_tool_call_scoping(settings: Settings = Depends(get_settings)) -> bool:
    """FastAPI dependency: whether ``run_verification`` should run the issue
    #158 per-tool-call scoping gate this request. Override in tests.

    Deterministic, no LLM call -- same posture as
    ``get_require_answer_grounding`` above, just the flag's current value.

    Flag ON (``copilot_extraction_tool_call_scoping_enabled``): the claim
    extractor's citable inputs are narrowed to only the tool calls the
    answer lexically engaged with, and any surviving citation of an
    unengaged call is downgraded (``app.tool_call_scoping``).

    Flag OFF (default): ``False`` -- ``run_verification`` skips both the
    prevention and enforcement halves, byte-identical to today."""
    return settings.copilot_extraction_tool_call_scoping_enabled


PatientFactProvider = Callable[[int], Sequence[Citation]]


def _no_op_patient_fact_provider(patient_id: int) -> list[Citation]:
    """Flag-OFF default: no disk read, ``[]`` every time -- the per-patient
    fact-citation path (P3.9a) is fully flag-gated, same posture as
    ``_no_op_evidence_retriever``."""
    return []


def get_patient_fact_provider(settings: Settings = Depends(get_settings)) -> PatientFactProvider:
    """FastAPI dependency: the active per-patient fact lookup for /chat's
    P3.9a lab/intake-form ``DocumentCitation`` path (issue #46). Override in
    tests.

    Flag ON (``copilot_evidence_retrieval_enabled`` -- the SAME flag as the
    P3.9 guideline-corpus path; this is the other half of "evidence
    retrieval"): reads ``LocalIngestionStore.list_citations_for_patient``,
    which is scoped STRICTLY to the ``patient_id`` it is called with -- see
    that method's docstring for how it fails closed on any malformed/foreign
    data. ``_stream_chat`` calls this ONLY with ``conversation.patient_id``,
    the SAME id ``/chat``'s launch-patient binding (#124 Phase 5, flag-gated,
    ``get_launch_binding_checker``) and P2.16 conversation binding already
    verify before a turn ever reaches this call -- a request that failed
    either of those checks never dispatches the planner, let alone this
    lookup, so this function never has an opportunity to see an
    unauthorized ``patient_id``.

    Flag OFF (default): ``_no_op_patient_fact_provider`` -- no disk read,
    ``[]`` every time, byte-identical to before this dependency existed.
    """
    if not settings.copilot_evidence_retrieval_enabled:
        return _no_op_patient_fact_provider
    store = LocalIngestionStore(settings.copilot_ingestion_base_dir)
    return store.list_citations_for_patient


def _log_encounter_record(
    trace_store: TraceStore, correlation_id: str, retrieved_chunks: Sequence[RerankedChunk]
) -> None:
    """Build and log the P3.8 per-encounter observability record for this
    turn, best-effort -- a build failure must never break an otherwise-
    successful turn. NO PHI: every logged field is a count/timing
    (``app.encounter_observability``'s own structural NO-PHI guarantee --
    see that module's docstring), never a query, answer, or record value.

    ``retrieved_chunks or None``: an empty list (the flag-off default, or a
    flag-on turn where retrieval found nothing / failed fail-soft) is treated
    as "retrieval was not meaningfully attempted" for this summary, same as
    every other caller of ``build_encounter_record`` that has no retrieval
    step at all -- not a fabricated zero-hit count."""
    try:
        record = build_encounter_record(correlation_id, trace_store, retrieval_chunks=retrieved_chunks or None)
    except Exception as exc:
        _logger.warning("encounter record build failed", extra={"error_type": type(exc).__name__})
        return
    _logger.info(
        "encounter record built",
        extra={
            "step_count": len(record.steps),
            "total_tokens_in": record.total_tokens_in,
            "total_tokens_out": record.total_tokens_out,
            "retrieval_hit_count": record.retrieval_hit_count,
        },
    )


_EMPTY_WARNINGS: dict[str, list[object]] = {
    "allergy_conflicts": [],
    "blocking_interactions": [],
    "warning_interactions": [],
}


def _serialize_segments(rendered: RenderedAnswer) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for segment in rendered.segments:
        if isinstance(segment, RenderedClaim):
            segments.append(
                {
                    "type": "claim",
                    "text": segment.text,
                    "citations": [
                        {
                            "tool_call_id": ref.tool_call_id,
                            "record_id": ref.record_id,
                            "field": ref.field,
                            "value": ref.asserted_value,
                        }
                        for ref in segment.source_refs
                    ],
                    # P3.7 citation overlay: the actual DocumentCitation
                    # fields (not just document_citation_count), so the UI
                    # can open the cited source page/quote.
                    "document_citations": [
                        {
                            "source_type": citation.source_type,
                            "source_id": citation.source_id,
                            "page_or_section": citation.page_or_section,
                            "field_or_chunk_id": citation.field_or_chunk_id,
                            "quote_or_value": citation.quote_or_value,
                        }
                        for citation in segment.document_citations
                    ],
                }
            )
        else:  # Notice
            segments.append({"type": "notice", "text": segment.text})
    return segments


def build_verification_payload(
    verdict_result: VerdictResult | None,
    rendered: RenderedAnswer | None,
) -> dict[str, object]:
    """Serialize the verification layer's output into the ``verification`` SSE
    frame payload the P3.8 UI renders (verdict badge, citation chips, warning
    banner).

    ``None`` inputs produce the *pending* payload (``verdict: null``, no
    segments, no warnings), which the UI renders nothing for. The live
    ``_stream_chat`` path now always passes a real ``VerdictResult`` /
    ``RenderedAnswer`` from ``app.extraction.run_verification``; the ``None``
    contract is retained for callers that want an explicit pending frame.
    """
    if verdict_result is None:
        return {
            "verdict": None,
            "segments": [],
            "warnings": dict(_EMPTY_WARNINGS),
        }

    segments = _serialize_segments(rendered) if rendered is not None else []
    return {
        "verdict": verdict_result.verdict.value,
        "segments": segments,
        "warnings": {
            "allergy_conflicts": [
                {
                    "medication_name": conflict.medication_name,
                    "allergy_substance": conflict.allergy_substance,
                }
                for conflict in verdict_result.allergy_conflicts
            ],
            "blocking_interactions": [
                {
                    "drug_a": item.drug_a,
                    "drug_b": item.drug_b,
                    "severity": item.severity.value,
                    "description": item.description,
                }
                for item in verdict_result.blocking_interactions
            ],
            "warning_interactions": [
                {
                    "drug_a": item.drug_a,
                    "drug_b": item.drug_b,
                    "severity": item.severity.value,
                    "description": item.description,
                }
                for item in verdict_result.warning_interactions
            ],
        },
    }


def _sse(event: ChatEvent, data: dict[str, object]) -> str:
    return f"event: {event.value}\ndata: {json.dumps(data)}\n\n"


def _tool_call_frame(trace_store: TraceStore, correlation_id: str, call: ToolCallTrace) -> str:
    """Build the ``tool_call`` SSE frame for one dispatch and record its tool
    span (best-effort), shared by both the streaming path (P2.12, called once
    per ``ToolDispatched`` event AS it arrives) and the fallback replay path
    (called once per ``result.trace`` entry after ``run()`` returns, for a
    planner double that only implements ``run()``)."""

    def _write_tool_span(call: ToolCallTrace = call) -> int:
        return trace_store.record_tool_span(
            correlation_id=correlation_id,
            start_ts=call.start_ts,
            end_ts=call.end_ts,
            ok=call.error is None,
            tool_name=call.tool.value,
            args=call.args,
            error_category=call.error,
            # P3.8: mint this span's own id, parented under whatever span
            # (if any) is already open in this correlation's context --
            # ``None`` today (this call site opens no span of its own), but
            # a real ambient parent once a caller nests tool dispatch under
            # a ``span_scope`` (e.g. a future supervisor-driven /chat path).
            span_id=str(uuid.uuid4()),
            parent_span_id=get_span_id(),
        )

    record_span_best_effort(_logger, "tool_span", _write_tool_span)
    return _sse(ChatEvent.TOOL_CALL, {"tool": call.tool.value, "args": call.args, "error": call.error})


def _emit_llm_spans(trace_store: TraceStore, correlation_id: str, llm_calls: list[LlmCallStats]) -> None:
    """Record one ``llm`` span per completed Ollama call (P4/#149), best-effort.

    ``llm_calls`` comes from ``PlannerResult.llm_calls`` (decision extracts,
    the quarantine summarizer, the two-call finalize) and, separately, the
    claim extractor's own ``llm_calls`` -- both are plain ``LlmCallStats``
    lists, never raw prompts/completions, so nothing PHI-bearing reaches the
    trace store here.
    """
    for llm_call in llm_calls:

        def _write_llm_span(llm_call: LlmCallStats = llm_call) -> int:
            return trace_store.record_llm_span(
                correlation_id=correlation_id,
                start_ts=llm_call.start_ts,
                end_ts=llm_call.end_ts,
                ok=llm_call.ok,
                model=llm_call.model,
                tokens_in=llm_call.tokens_in,
                tokens_out=llm_call.tokens_out,
                # P3.8: same span-tree discipline as _write_tool_span above.
                span_id=str(uuid.uuid4()),
                parent_span_id=get_span_id(),
            )

        record_span_best_effort(_logger, "llm_span", _write_llm_span)


def _stream_chat(
    planner: PlannerProtocol,
    extractor: ClaimExtractorLike,
    conversation: Conversation,
    store: ConversationStore,
    trace_store: TraceStore,
    message: str,
    user: str,
    clock: Clock,
    evidence_retriever: EvidenceRetriever = _no_op_evidence_retriever,
    patient_fact_provider: PatientFactProvider = _no_op_patient_fact_provider,
    support_judge_provider: SupportJudgeProvider = _no_op_support_judge_provider,
    require_answer_grounding: bool = False,
    require_tool_call_scoping: bool = False,
) -> Iterable[str]:
    correlation_id = get_correlation_id()
    request_start_ts = time.time()
    request_ok = True
    _logger.info(
        "chat invocation started",
        extra={"conversation_id": conversation.conversation_id},
    )

    try:
        yield _sse(
            ChatEvent.CONVERSATION,
            {"conversation_id": conversation.conversation_id, "correlation_id": correlation_id},
        )

        # P2.12: prefer the real-time streaming path -- ``run_streaming``
        # yields a ``tool_call`` frame as each tool actually dispatches,
        # instead of replaying the whole trace after the loop finishes.
        # ``getattr`` (not a direct attribute access) so a ``PlannerProtocol``
        # double that only implements ``run()`` (all 8 existing fake-planner
        # tests) falls back to that path untouched, rather than erroring on a
        # missing attribute the protocol never promised. Computed
        # unconditionally (a cheap attribute lookup, no side effect) so it is
        # available below regardless of which branch runs next.
        run_streaming = getattr(planner, "run_streaming", None)

        def _roster_provider() -> list[str]:
            # #237: resolved LAZILY -- this closure is only ever CALLED by
            # detect_foreign_patient_reference when a "switch to <Name>"
            # construction has already matched and isn't the bound patient,
            # so a turn that never uses that construction never pays this
            # round trip. Cached on the conversation so a second matching
            # turn in the SAME conversation reuses it instead of re-fetching.
            if conversation.patient_roster is None:
                resolve_roster = getattr(planner, "resolve_patient_roster", None)
                conversation.patient_roster = resolve_roster() if resolve_roster is not None else []
            return conversation.patient_roster

        # #223 (extended by #224, #237): deterministic PRE-dispatch
        # cross-patient refusal guard, checked BEFORE the planner runs at
        # all. Unlike #194's apply_subject_check below (which only rewrites
        # the answer TEXT after tools have already been dispatched), this
        # short-circuits BEFORE any tool dispatch or model call -- the only
        # way to guarantee a forbidden tool never runs. ``conversation
        # .patient_name`` (resolved once at conversation-creation time, see
        # ``_resolve_conversation_patient_name``) enables the guard's named
        # signals; ``None`` (name-binding unavailable) falls back to #223's
        # numeric-only detection. ``_roster_provider`` enables the #237
        # roster-based "switch to <Name>" signal. See app.extraction
        # .detect_foreign_patient_reference.
        cross_patient_reference_detected = detect_foreign_patient_reference(
            message,
            conversation.patient_id,
            conversation.patient_name,
            roster_provider=_roster_provider,
        )

        # #105: guideline-corpus retrieval now runs BEFORE the planner
        # composes its answer (moved from after -- see the module docstring's
        # "answer" frame note and app.planner's #105 comment for the full
        # mechanism). Previously the planner wrote its answer purely from
        # tool results / its own priors, and a citation was bolted onto that
        # prose after the fact by the claim extractor -- letting the answer's
        # category language (e.g. "elevated blood pressure") drift from the
        # guideline text it ended up citing (e.g. "Stage 2 hypertension").
        # Feeding the retrieved text into the planner's OWN answer-
        # composition call lets it use the guideline's own category name.
        #
        # This is the same fail-soft retrieval call that used to run later in
        # this function (a retrieval/rerank failure must never break an
        # otherwise-working chat turn over chart data unrelated to the
        # guideline corpus); only its POSITION changed, unconditionally, same
        # as before -- including on a cross-patient refusal, where the
        # guideline_excerpts built from it below simply goes unused (the
        # planner never runs in that branch). Retrieval stays unconditional
        # rather than skipped for that branch so this change doesn't also
        # alter P3.8 encounter-observability's retrieval_hit_count / worker
        # span for a refusal turn -- out of scope for this ordering fix.
        try:
            retrieved_chunks = evidence_retriever(message)
        except Exception as exc:
            _logger.warning("evidence retrieval failed", extra={"error_type": type(exc).__name__})
            retrieved_chunks = []
        guideline_excerpts = [chunk.text for chunk in retrieved_chunks]

        # #86: this turn's patient's ingested fact citations, fetched HERE
        # (before the planner runs) so this ONE disk read feeds both answer
        # composition (`document_facts` below) and the post-hoc verification
        # pass further below (`patient_facts`, P3.9a/#46) -- never fetched
        # twice. Fail-soft like evidence_retriever above: logged by type only.
        try:
            patient_facts = patient_fact_provider(conversation.patient_id)
        except Exception as exc:
            _logger.warning("patient fact lookup failed", extra={"error_type": type(exc).__name__})
            patient_facts = []

        # Threaded into the planner call ONLY when non-empty -- the same
        # conditional-kwarg convention `app.extraction.run_verification`
        # already uses for `retrieved_chunks`/`patient_facts` -- so a planner
        # double that predates this parameter (every existing test fake)
        # keeps working unmodified for the (overwhelmingly common) case where
        # this patient has nothing ingested.
        planner_kwargs: dict[str, Sequence[Citation]] = {}
        if patient_facts:
            planner_kwargs["document_facts"] = patient_facts

        if cross_patient_reference_detected:
            result = cross_patient_refusal_result()
        elif run_streaming is not None:
            result = None
            for event in run_streaming(message, guideline_excerpts, **planner_kwargs):
                if isinstance(event, ToolDispatched):
                    yield _tool_call_frame(trace_store, correlation_id, event.trace)
                elif isinstance(event, ReasoningDelta):
                    # Unverified, provisional text -- forwarded as-is into
                    # its own SSE frame so the UI can render it into a
                    # separate "thinking" surface. It must never reach the
                    # ``answer`` frame below, which only ever carries
                    # ``result.answer`` (the post-extraction, verified text).
                    yield _sse(ChatEvent.REASONING_DELTA, {"text": event.text})
                elif isinstance(event, PlannerCompleted):
                    result = event.result
            if result is None:
                raise AssertionError("run_streaming ended without a terminal PlannerCompleted event")  # pragma: no cover
        else:
            result = planner.run(message, guideline_excerpts, **planner_kwargs)
        assert result is not None  # mypy: the elif branch's loop widens result
        # Deterministic cross-patient subject-check (#194, follow-up to
        # #121): a small model can verbally attribute the bound patient's
        # data to a different, unqueried patient the question named/numbered
        # -- this is a model-independent backstop that strips any such
        # reference from the final answer BEFORE it is emitted, regardless of
        # the model's phrasing. See ``app.extraction.apply_subject_check``.
        # Applied BEFORE the recency notice below (not after) so it only ever
        # scans the model's own prose -- never text a later deterministic step
        # appends (e.g. a stale record's literal date), which could otherwise
        # coincidentally collide with a foreign patient number. Safe (a
        # guaranteed no-op) to run unconditionally even when the #223 guard
        # above already fired: ``cross_patient_refusal_result()``'s generic
        # answer never echoes the foreign patient it detected, so this never
        # finds anything to strip.
        result = apply_subject_check(result, question=message, patient_id=conversation.patient_id)
        # Deterministic unresolvable-referent guard (#225): a demonstrative
        # medication reference the question never names ("that new
        # medication") with no prior turn in THIS conversation to anchor it
        # -- a small model is prone to silently guessing the referent rather
        # than asking. Skipped (not called unconditionally like
        # apply_subject_check above) when the #223 guard already fired: a
        # cross-patient refusal is a different, higher-priority question
        # class, and overriding it here would silently discard that refusal
        # even though its own answer text never happens to trip this guard.
        # ``conversation.history`` reflects only PRIOR turns at this point --
        # the current turn is appended further below, after this call -- so
        # "no prior turns" here means genuinely the first message. See
        # ``app.extraction.clarify_unresolvable_referent``.
        if not cross_patient_reference_detected:
            result = clarify_unresolvable_referent(
                result, question=message, has_prior_turns=bool(conversation.history)
            )
        # Deterministic recency notice (#153): append a caveat naming the
        # record's date for any stale record the planner returned this turn,
        # BEFORE the answer is emitted -- so a real user never sees years-old
        # data presented as "current" without its age. No LLM call; a pure
        # function of the planner output + the injected wall clock (tz-aware,
        # so the comparison against possibly-tz-aware record dates is safe --
        # see ``app.verification._as_aware_utc``). Applied here (not deeper in
        # the verification layer) so the notice lands on ``result.answer``,
        # which feeds the answer frame, the verification pipeline, and the
        # stored turn alike. A future cleaner form carries the notice as a
        # structured ``RenderedAnswer``/verdict-warning segment rather than
        # splicing answer text -- deferred (see ``apply_recency_notice``).
        result = apply_recency_notice(result, now=clock())

        if run_streaming is None:
            # Fallback path: the planner double only implements ``run()``, so
            # the whole trace is only available now -- replay it as
            # ``tool_call`` frames (bunched, same as pre-P2.12 behavior).
            for call in result.trace:
                yield _tool_call_frame(trace_store, correlation_id, call)

        _emit_llm_spans(trace_store, correlation_id, result.llm_calls)

        yield _sse(ChatEvent.ANSWER, {"answer": result.answer})

        # P3.9: hybrid-retrieve + rerank guideline-corpus evidence for this
        # turn's question (flag-gated -- see get_evidence_retriever), offered
        # to the extraction pipeline below as extra citable evidence
        # ALONGSIDE the tool-result catalog -- additive; a chart-data-only
        # question that retrieves nothing verifies exactly as before.
        #
        # #105: ``retrieved_chunks`` was already computed ABOVE, before the
        # planner ran (see that comment for why) -- reused here rather than
        # retrieved a second time, so this ordering change costs exactly one
        # retrieval call per turn, same as before, just earlier.

        # P3.9a (issue #46): this turn's bound patient's own extracted
        # lab/intake-form fact citations, offered to the extraction pipeline
        # below alongside the guideline chunks and tool-result catalog.
        # ``patient_facts`` was already computed ABOVE, before the planner
        # ran (see that comment -- #86 folded this fetch into the SAME
        # earlier point ``guideline_excerpts``/``retrieved_chunks`` already
        # use) -- reused here rather than looked up a second time.

        # Run the answer->claims extraction pipeline and populate the verification
        # frame with the REAL verdict / citation chips / warnings for this answer.
        # ``run_verification`` re-validates every extracted claim against the RAW
        # records (deterministic, no LLM) and strips the unverifiable ones, so a
        # miscited or injection-steered claim never reaches the user as fact.
        verification_start_ts = time.time()
        verdict_result, rendered = run_verification(
            extractor,
            result,
            retrieved_chunks=retrieved_chunks,
            patient_facts=patient_facts,
            support_judge=support_judge_provider(),
            require_answer_grounding=require_answer_grounding,
            require_tool_call_scoping=require_tool_call_scoping,
        )
        verification_end_ts = time.time()
        _emit_llm_spans(trace_store, correlation_id, getattr(extractor, "llm_calls", []))
        verdict_trace_record = to_trace_record(verdict_result)
        # Yield the frame before the trace write -- the client shouldn't wait
        # on a disk write for data it already has.
        yield _sse(ChatEvent.VERIFICATION, build_verification_payload(verdict_result, rendered))
        record_span_best_effort(
            _logger,
            "verification_span",
            lambda: trace_store.record_verification_span(
                correlation_id=correlation_id,
                start_ts=verification_start_ts,
                end_ts=verification_end_ts,
                ok=True,
                verdict=verdict_trace_record["verdict"],
                claim_count=verdict_trace_record["total_claim_count"],
                stripped_count=verdict_trace_record["stripped_claim_count"],
            ),
        )

        store.append_turn(
            conversation.conversation_id,
            Turn(
                correlation_id=correlation_id,
                user=user,
                patient_id=conversation.patient_id,
                question=message,
                answer=result.answer,
            ),
        )

        # P3.8: capture this turn's full pipeline (tool/llm/worker/verification
        # spans, all already recorded above) as a per-encounter observability
        # record, best-effort. See ``_log_encounter_record``'s NO-PHI note.
        _log_encounter_record(trace_store, correlation_id, retrieved_chunks)

        yield _sse(ChatEvent.DONE, {})
    except BaseException:
        # BaseException, not Exception: an early client disconnect closes this
        # generator via GeneratorExit (a BaseException, not an Exception), and
        # that case must record ok=False too, not the request's default True.
        request_ok = False
        raise
    finally:
        record_span_best_effort(
            _logger,
            "request_span",
            lambda: trace_store.record_request_span(
                correlation_id=correlation_id,
                start_ts=request_start_ts,
                end_ts=time.time(),
                ok=request_ok,
            ),
        )


async def _resolve_conversation_patient_name(planner: PlannerProtocol) -> str | None:
    """Best-effort resolve the bound patient's display name for a brand-new
    conversation (#224 name-binding), via the planner's OPTIONAL
    ``resolve_patient_name`` capability -- duck-typed via ``getattr``, the
    same pattern ``_stream_chat`` already uses for ``run_streaming``: a
    ``PlannerProtocol`` double that only implements ``run()`` simply has no
    name to offer, and every caller of ``detect_foreign_patient_reference``
    already treats ``None`` as "name-binding unavailable" (falls back to
    #223's numeric-only signal) -- never a hard failure.

    Dispatched to FastAPI's threadpool because a real resolve is a blocking
    HTTP round trip (mirrors ``_validate_token``'s own threadpool dispatch,
    same reason: this runs inside the ``async`` ``chat_endpoint`` body).
    Called ONCE per conversation, at creation time -- never on resume (see
    ``chat_endpoint``), so an established conversation never pays this cost
    again on later turns.
    """
    resolve = getattr(planner, "resolve_patient_name", None)
    if resolve is None:
        return None
    return await run_in_threadpool(resolve)


def extract_bearer_token(authorization: str | None) -> str:
    """Pull the token out of an ``Authorization: Bearer <token>`` header.

    Public (not module-private): ``app.feedback.feedback_endpoint`` reuses
    this exact parsing rather than duplicating it, since both endpoints gate
    on the same bearer-token seam.
    """
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise TokenValidationError("missing bearer token")
    return authorization[len(prefix) :]


async def chat_endpoint(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
    validator: TokenValidator = Depends(get_token_validator),
    launch_binding_checker: LaunchBindingChecker = Depends(get_launch_binding_checker),
    planner_factory: PlannerFactory = Depends(get_planner_factory),
    extractor: ClaimExtractorLike = Depends(get_claim_extractor),
    store: ConversationStore = Depends(get_conversation_store),
    trace_store: TraceStore = Depends(get_trace_store),
    clock: Clock = Depends(get_clock),
    evidence_retriever: EvidenceRetriever = Depends(get_evidence_retriever),
    patient_fact_provider: PatientFactProvider = Depends(get_patient_fact_provider),
    support_judge_provider: SupportJudgeProvider = Depends(get_support_judge_provider),
    require_answer_grounding: bool = Depends(get_require_answer_grounding),
    require_tool_call_scoping: bool = Depends(get_require_tool_call_scoping),
) -> StreamingResponse:
    try:
        token = extract_bearer_token(authorization)
        await _validate_token(validator, token)
    except TokenValidationError as exc:
        raise HTTPException(status_code=401, detail="invalid or missing token") from exc

    # #124 Phase 5: the token's SMART launch patient (when present) is the
    # authoritative binding -- reject a mismatch here, BEFORE the planner (and
    # thus any tool call) is run. A token WITHOUT launch context is not
    # hard-failed; the checker is a no-op and the request falls back to the
    # P2.16 conversation-pid binding below. Flag OFF -> the checker is a no-op.
    try:
        launch_binding_checker(token, request.patient_id)
    except LaunchPatientMismatchError as exc:
        # Log-safe: no token, pid, or UUID -- just that a binding was refused.
        _logger.warning("chat request rejected: token launch-context patient binding mismatch")
        raise HTTPException(
            status_code=403, detail="patient_id is not authorized for this token"
        ) from exc

    user = _user_identity_from_token(token)

    planner = planner_factory(request.patient_id)

    if request.conversation_id:
        conversation = store.get(request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="unknown conversation_id")
        if conversation.patient_id != request.patient_id:
            raise HTTPException(
                status_code=409,
                detail="conversation_id is bound to a different patient_id",
            )
    else:
        # #224: resolve the bound patient's own display name ONCE, at
        # conversation-creation time -- see _resolve_conversation_patient_name.
        patient_name = await _resolve_conversation_patient_name(planner)
        conversation = store.create(request.patient_id, patient_name=patient_name)

    return StreamingResponse(
        _stream_chat(
            planner,
            extractor,
            conversation,
            store,
            trace_store,
            request.message,
            user,
            clock,
            evidence_retriever,
            patient_fact_provider,
            support_judge_provider,
            require_answer_grounding,
            require_tool_call_scoping,
        ),
        media_type="text/event-stream",
    )

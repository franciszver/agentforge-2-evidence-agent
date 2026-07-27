"""SSE ``POST /chat`` endpoint: multi-turn conversation over the planner (P2.10).

Route decision: the static shell page lives at ``GET /chat`` (P0.6) and this
SSE stream lives at ``POST /chat`` -- same path, different HTTP method, which
FastAPI dispatches independently, so both work without a clash. The shell's
``<form>`` posts back to ``/chat``.

Auth: the bearer token is validated through an injectable ``TokenValidator``
seam (``get_token_validator``). With ``copilot_per_user_token_enabled`` off
(the shipped default), validation is FAIL-CLOSED (#168, VULN-0001): every
token is rejected. A dev-only stub that accepts any non-empty token exists
behind an explicit, loudly-logged opt-in
(``copilot_dev_accept_any_bearer_token``) for local development. Real
OpenEMR token introspection is already built and used when
``copilot_per_user_token_enabled`` is on -- see
``build_introspection_validator``. A missing header or a validator
rejection both produce a 401 before the planner is ever constructed or
invoked -- enforced structurally since #177 by ``get_planner_factory``
depending on ``get_authenticated_token`` (see both docstrings): before #177,
``get_planner_factory`` read the raw header itself and FastAPI resolved it
independently of the endpoint body's own check, so the flag-off dev-token
bridge fetch still ran for every unauthenticated request.

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
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Protocol

from fastapi import Depends, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    DEFAULT_MAX_STORED_CONVERSATIONS,
    DEFAULT_MAX_TURNS_PER_CONVERSATION,
    DEFAULT_ROSTER_CACHE_TTL_SECONDS,
    Settings,
    get_settings,
)
from app.correlation import get_correlation_id, get_span_id
from app.dev_token_bridge import DevTokenBridge, DevTokenError
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
from app.tools.patient_summary import RosterEntry
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
# turns, and the answer itself. This bound applies unconditionally to the
# WHOLE request message. It is a SEPARATE, WIDER bound than
# app.retrieval.MAX_QUERY_CHARS (2000): that one only applies to the derived
# retrieval query, and only when copilot_evidence_retrieval_enabled is true;
# this one applies to every /chat request regardless of that flag.
# Rejecting outright (422) is deliberate, not truncating: a silently
# truncated question could change clinical meaning without the caller
# knowing.
MAX_CHAT_MESSAGE_LENGTH = 4000

# Issue #167 Gate 3 MINOR finding: ``conversation_id`` round-trips a value
# THIS service itself minted (``str(uuid.uuid4())`` in
# ``ConversationStore.create``, always exactly 36 chars), so a well-behaved
# caller never sends anything longer -- but nothing stopped an attacker from
# sending an arbitrarily long string here either, ahead of the dict lookup
# it's used for. 64 is generous headroom over the 36-char UUID form (room
# for a future id scheme change without another bump) while still rejecting
# unbounded input outright.
MAX_CONVERSATION_ID_LENGTH = 64


class ChatRequest(BaseModel):
    """``POST /chat`` request body."""

    # min_length=1 (Gate 3 MINOR finding): an empty message still reached
    # the planner and burned a full LLM call/tool-dispatch cycle for
    # nothing -- the PHP chat-panel proxy (ChatProxyRequest::parseMessage)
    # already rejects a blank message before it ever reaches this service,
    # so accepting one here was only reachable by a caller bypassing that
    # proxy, and had no legitimate use either way.
    message: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_LENGTH)
    patient_id: int
    conversation_id: str | None = Field(default=None, max_length=MAX_CONVERSATION_ID_LENGTH)


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


def _fail_closed_token_validator(token: str) -> None:
    """#168 (VULN-0001) fix: the flag-off, dev-flag-off default. Rejects EVERY
    token, valid-looking or not -- with ``copilot_per_user_token_enabled`` off
    and no real introspection wired up, there is no way to actually verify a
    token, so the safe default is to accept none, not to accept all.

    Raises the same ``TokenValidationError`` every other validator raises, so
    ``chat_endpoint``'s except clause maps this to a clean 401 exactly as
    before -- callers cannot distinguish "no per-user auth configured" from
    "bad token" by response shape.
    """
    raise TokenValidationError("token validation is not configured for this deployment")


def _dev_permissive_token_validator(token: str) -> None:
    """DEV-ONLY stub token validator: accepts any non-empty token.

    Only reachable when ``copilot_dev_accept_any_bearer_token`` is explicitly
    set (see ``app.config.Settings`` for why this must never be true outside
    local development) -- restores the pre-#168 behaviour for that opt-in
    case. Logs a loud warning on every use (no PHI, no token value) so the
    permissive path is never silently active.
    """
    if not token:
        raise TokenValidationError("missing bearer token")
    _logger.warning(
        "chat request authenticated via dev-only permissive bearer-token stub "
        "(copilot_dev_accept_any_bearer_token=true) -- any non-empty token is "
        "accepted; this MUST NOT be enabled outside local development",
    )


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
    per-user bearer via OpenEMR introspection.

    Flag OFF (the shipped default): fail-closed as of #168 (VULN-0001) --
    every token is rejected UNLESS ``copilot_dev_accept_any_bearer_token`` is
    also explicitly set, in which case the pre-#168 any-non-empty-token stub
    is used instead (dev-only; see that setting's docstring).
    """
    settings = get_settings()
    if settings.copilot_per_user_token_enabled:
        return build_introspection_validator(get_token_introspector())
    if settings.copilot_dev_accept_any_bearer_token:
        return _dev_permissive_token_validator
    return _fail_closed_token_validator


async def get_authenticated_token(
    authorization: str | None = Header(default=None),
    validator: TokenValidator = Depends(get_token_validator),
) -> str:
    """FastAPI dependency: extract + validate the bearer token, raising the
    401 mapping directly.

    #177: this used to be inline code in ``chat_endpoint``'s body, which runs
    strictly AFTER every ``Depends(...)`` parameter in the signature resolves
    -- including ``get_planner_factory``, whose flag-off branch calls
    ``dev_token_bridge.get_token()`` (an outbound OAuth password-grant against
    OpenEMR). That let an unauthenticated caller (no header at all) trigger a
    real, blocking outbound fetch on every request: FLAG-OFF, that meant
    unauthenticated threadpool starvation, repeated failed password-grants
    against the demo clinician account, and a timing oracle (~2s with a live
    bridge attempt vs ~0.1s without) disclosing whether the dev-token bridge
    is provisioned. (Flag-ON, an unauthenticated flood could already occupy a
    threadpool worker per request via a cache-miss introspection call -- see
    ``_validate_token`` -- because an attacker picks a fresh, never-cached
    token each request, guaranteeing the ``peek_cached`` miss branch and a
    real, timeout-bounded (``openemr_api_timeout_seconds``) introspection
    round trip. This fix does not introduce that exposure and is not a
    regression -- pre-#177, flag-on burned TWO worker slots per unauthenticated
    request (one for introspection, one for the bridge); this fix removes the
    SECOND slot and removes the bridge fetch and its timing oracle entirely on
    both branches. The remaining flag-on introspection-triggered threadpool
    exposure is a pre-existing, separate concern -- pre-introspection rate
    limiting or negative caching there is intentionally NOT attempted by this
    fix and is tracked separately (#188).)

    Pulling the check into its own dependency and making ``get_planner_factory``
    depend on IT (see that function) fixes the ordering structurally: FastAPI
    resolves a dependency's own sub-dependencies before calling its body, so
    ``get_planner_factory``'s body -- and therefore the bridge -- is now
    reached only once this dependency has already returned a validated token.
    A request whose token fails validation never touches the bridge at all --
    EXCEPT under the dev-only ``copilot_dev_accept_any_bearer_token`` flag,
    where "validation" is the permissive stub (any non-empty bearer passes),
    so bridge access is effectively unauthenticated in that one configuration
    too; see ``_dev_permissive_token_validator``.

    Cached per-request by FastAPI's default ``Depends`` behaviour, so
    ``chat_endpoint`` and ``get_planner_factory`` both depending on this
    function costs exactly one validation, not two.

    Raises ``HTTPException(401)`` on any ``TokenValidationError`` -- callers
    cannot distinguish "no per-user auth configured" from "bad token" from
    "missing header" by response shape, same invariant ``_fail_closed_token_
    validator`` (#168) documents.
    """
    try:
        token = extract_bearer_token(authorization)
        await _validate_token(validator, token)
    except TokenValidationError as exc:
        raise HTTPException(status_code=401, detail="invalid or missing token") from exc
    return token


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
    token: str = Depends(get_authenticated_token),
    dev_token_bridge: DevTokenBridge = Depends(get_dev_token_bridge),
) -> PlannerFactory:
    """FastAPI dependency: builds a ``PlannerProtocol`` for a patient_id. Override in tests.

    #177: depends on ``get_authenticated_token`` (not the raw ``Authorization``
    header) precisely so this dependency's body -- and, on the flag-off
    branch, the ``DevTokenBridge`` fetch below -- is unreachable until the
    caller's bearer token has already been extracted AND validated. FastAPI
    resolves a dependency's own sub-dependencies before calling its body, so
    a request whose token fails ``get_authenticated_token`` never gets here
    at all; it 401s there instead. (Under the dev-only
    ``copilot_dev_accept_any_bearer_token`` flag, "fails" means "empty" only
    -- any non-empty bearer passes as "authenticated" and does reach the
    bridge; see ``get_authenticated_token``'s docstring.) Before this fix, this dependency read the
    header directly and FastAPI resolved it independently of (and, in
    practice, before) the endpoint body's own token check -- so an
    unauthenticated caller (no header, or a garbage one) still triggered the
    flag-off branch's outbound OAuth password-grant against OpenEMR on every
    request: unauthenticated threadpool starvation (the fetch runs in
    FastAPI's bounded, app-wide sync-dependency threadpool), repeated failed
    password-grants against the demo clinician account, and a timing oracle
    disclosing whether the bridge is provisioned (see #177 for measurements).

    Flag ON (``copilot_per_user_token_enabled``, #124 Phase 4): the planner is
    bound to the REQUEST's own forwarded bearer -- already extracted and
    validated by ``get_authenticated_token`` -- so OpenEMR maps every tool
    call to that user -> per-user ACL.

    Flag OFF: byte-identical to before -- the ``DevTokenBridge``'s
    demo-clinician token drives tool calls, fetched only now that the caller's
    OWN token has already passed validation. The bridge's (potentially
    blocking, on a cache miss) token fetch happens here, in a sync dependency
    FastAPI runs in its worker-thread pool -- not in the ``async``
    ``chat_endpoint`` body, so a token refresh never blocks the event loop.

    An unprovisioned bridge (missing/invalid dev client creds -- any
    deployment that isn't the local dev stack) still raises ``DevTokenError``,
    caught here and bound to an empty token: the planner factory still gets
    built (this dependency must never raise for an auth problem the caller's
    OWN token already cleared), and the empty token is inert until a tool call
    would use it -- every such call then auth-fails against OpenEMR and is
    caught as ``OpenEmrApiError`` in the planner loop, so the agent answers
    with zero patient evidence instead of crashing the conversation. The
    ``_logger.warning`` below exists so that failure mode is never silent.

    Because the fetch now only ever runs for an ALREADY-authenticated caller,
    an unprovisioned or unreachable bridge can still be amplified by a
    legitimate authenticated caller's request volume (no negative caching or
    backoff on the bridge itself) -- judged out of scope for this fix; see
    #177's PR discussion for the reasoning.
    """
    if get_settings().copilot_per_user_token_enabled:
        return _default_planner_factory(token)
    try:
        bridge_token = dev_token_bridge.get_token()
    except DevTokenError:
        bridge_token = ""
        _logger.warning(
            "dev token bridge unavailable; tool calls for this request will "
            "auth-fail against OpenEMR (answers will cite zero evidence)"
        )
    return _default_planner_factory(bridge_token)


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

    Issue #174: this class used to also carry ``patient_roster`` (#237
    roster-based detection) -- every OTHER patient's display name, resolved
    lazily and cached HERE, on the conversation, for the conversation's
    entire lifetime. That field is gone. Two independent problems with it,
    read together in #174's analysis:

      * Memory: the roster is proportional to total patient count, which no
        operator setting bounds, and every open ``Conversation`` retained
        its OWN full copy -- the dominant term in this service's worst-case
        memory footprint (see ``Settings.copilot_max_turns_per_conversation``
        's docstring, corrected by #174 to stop citing this as a still-open
        gap), and a ~2000x amplification against OpenEMR's patient API for
        any conversation that ever used a "switch to <Name>" construction.
      * Privacy: ``docs/ARCHITECTURE.md``'s conversation-to-pid binding is a
        boundary the agent adds on top of OpenEMR's own auth -- every
        conversation is anchored to the pid the panel was opened on.
        Retaining every OTHER patient's name on that same conversation
        object, for its whole lifetime, undermined that boundary regardless
        of memory, and would have mattered more once ``ConversationStore``'s
        TODO(P4.2) durable, DB-backed store lands.

    The roster is byte-identical across every conversation (nothing in the
    fetch is caller-specific -- see
    ``app.tools.patient_summary.get_patient_roster``), so there was nothing
    conversation-specific here worth keeping: ``app.chat.RosterCache`` now
    serves it from ONE process-wide, TTL'd cache instead, and no
    ``Conversation`` object holds any other patient's name at all.
    ``_stream_chat``'s ``_roster_provider`` closure reads that shared cache
    directly; see its docstring for the resolve-lazily behavior this
    replaces.
    """

    conversation_id: str
    patient_id: int
    patient_name: str | None = None
    history: list[Turn] = field(default_factory=list)


class ConversationStore:
    """In-memory, LRU-bounded conversation store keyed by ``conversation_id``.

    Issue #167 (VULN-0004): an earlier version of this store had no
    eviction, so it retained every conversation for the process lifetime --
    unbounded, attacker-influenced memory growth (any caller can start a new
    conversation). ``max_conversations`` bounds THAT axis: once the cap is
    exceeded, the least-recently-used conversation is evicted. "Used" means
    read via ``get`` or re-registered via ``append_turn`` -- see
    ``append_turn``'s docstring for why a conversation CAN still be evicted
    out from under a caller mid-request, and how that is recovered rather
    than crashing or silently losing the turn.

    A SEPARATE axis (Gate 2 finding on #167): the conversation-count cap
    alone does not stop a single conversation from growing forever -- an
    attacker who reuses ONE ``conversation_id`` and keeps calling ``/chat``
    stays permanently most-recently-used and so is never an eviction
    candidate under the axis above. ``max_turns_per_conversation`` bounds
    THIS axis: ``append_turn`` drops the oldest turn once a conversation's
    own history exceeds the cap. Safe to drop silently (see
    ``Settings.copilot_max_turns_per_conversation``'s docstring for how this
    was verified): ``.history`` is read in exactly one place in the service,
    as a boolean "any prior turns?" signal, never as planner context.

    A caller resuming with an evicted ``conversation_id`` gets ``get() ->
    None`` -- the endpoint already maps that to a clean 404 (unknown
    conversation_id), not a crash; see ``chat_endpoint``.

    Thread safety: this is a shared, mutable, process-wide singleton
    (``_default_store``) served from FastAPI's worker threadpool (``/chat``'s
    ``_stream_chat`` body runs synchronously off the event loop), so every
    mutating operation below holds ``self._lock``. Critical sections are
    kept minimal (dict/list operations only, no I/O) -- this closes the
    whole class of interleaved-mutation races (e.g. two concurrent
    ``append_turn`` calls on the same conversation both reading
    ``len(history)`` before either trims, transiently exceeding the cap),
    not just the specific ones identified during review.

    TODO(P4.2): replace with the durable trace store; this is a placeholder
    with the same shape (get / create / append) a DB-backed store would have.
    """

    def __init__(
        self,
        max_conversations: int = DEFAULT_MAX_STORED_CONVERSATIONS,
        max_turns_per_conversation: int = DEFAULT_MAX_TURNS_PER_CONVERSATION,
    ) -> None:
        if max_conversations <= 0:
            raise ValueError("max_conversations must be positive")
        if max_turns_per_conversation <= 0:
            raise ValueError("max_turns_per_conversation must be positive")
        self._max_conversations = max_conversations
        self._max_turns_per_conversation = max_turns_per_conversation
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is not None:
                self._conversations.move_to_end(conversation_id)
            return conversation

    def create(self, patient_id: int, patient_name: str | None = None) -> Conversation:
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()), patient_id=patient_id, patient_name=patient_name
        )
        with self._lock:
            # No move_to_end here: OrderedDict already inserts a NEW key at
            # the most-recently-used (right/end) position -- calling
            # move_to_end immediately after would be a verified no-op.
            self._conversations[conversation.conversation_id] = conversation
            while len(self._conversations) > self._max_conversations:
                self._conversations.popitem(last=False)
        return conversation

    def append_turn(self, conversation: Conversation, turn: Turn) -> None:
        """Append ``turn`` to ``conversation.history`` and re-register the
        conversation as most-recently-used.

        Takes the live ``Conversation`` object (not a ``conversation_id``
        lookup) deliberately: ``chat_endpoint`` calls ``create``/``get`` on
        the event loop, then does several seconds of planner + verification
        work in a worker thread before ever calling this -- during which
        every OTHER concurrent request's cheap ``create()`` call is a
        chance to evict this conversation as the LRU tail (nothing touches
        it in between). An earlier version of this method looked the
        conversation up by id and either raised ``KeyError`` (if evicted) or
        silently dropped the turn -- either way losing the Turn, which its
        own docstring calls the P2.17 chart-access audit record for a chart
        access that DID happen. Re-inserting the caller's own object instead
        recovers from that eviction: the turn (and the whole conversation)
        is retained, at the cost of counting again against the
        conversation-count cap on the way back in.
        """
        with self._lock:
            conversation.history.append(turn)
            if len(conversation.history) > self._max_turns_per_conversation:
                del conversation.history[: len(conversation.history) - self._max_turns_per_conversation]
            self._conversations[conversation.conversation_id] = conversation
            self._conversations.move_to_end(conversation.conversation_id)
            while len(self._conversations) > self._max_conversations:
                self._conversations.popitem(last=False)


_default_store: ConversationStore | None = None


def get_conversation_store(settings: Settings = Depends(get_settings)) -> ConversationStore:
    """FastAPI dependency: the active ``ConversationStore``. Override in tests.

    Built lazily against ``Settings.copilot_max_stored_conversations`` /
    ``Settings.copilot_max_turns_per_conversation``. Deliberately a
    DIFFERENT shape than ``get_trace_store`` below, which calls
    ``get_settings()`` in its own body rather than taking ``settings`` as an
    injected ``Depends`` parameter: that was the simpler option available
    when this function was written, but injecting ``Settings`` via
    ``Depends`` here is the better pattern -- it makes the settings
    dependency visible in this function's signature (FastAPI's dependency
    graph, not a hidden in-body call), and lets a test override
    ``get_settings`` without needing to know this function reads it. Kept as
    a lazily-built singleton (not rebuilt every request) so the store's
    state persists across requests, matching ``get_trace_store``'s caching.

    CAPS ARE FROZEN AT FIRST USE: ``get_settings()`` returns a fresh
    ``Settings`` on every call (env vars are re-read each time), but this
    function only reads it ONCE -- the first request to ever resolve this
    dependency -- and bakes the two caps into ``_default_store`` for the
    rest of the process's life. Changing
    ``COPILOT_MAX_STORED_CONVERSATIONS``/``COPILOT_MAX_TURNS_PER_CONVERSATION``
    in the environment after the process has started (and served at least
    one ``/chat`` request) has no effect until restart.
    """
    global _default_store
    if _default_store is None:
        _default_store = ConversationStore(
            max_conversations=settings.copilot_max_stored_conversations,
            max_turns_per_conversation=settings.copilot_max_turns_per_conversation,
        )
    return _default_store


class RosterCache:
    """Process-wide, TTL'd cache of OpenEMR's full patient roster (#174).

    Replaces the per-``Conversation`` ``patient_roster`` field this class's
    docstring on ``Conversation`` describes in full -- read that first for
    the memory/privacy analysis behind why a SHARED cache, not a per-object
    one, is the fix. In short: the roster
    (``app.tools.patient_summary.get_patient_roster``) is byte-identical no
    matter which patient is asking, so there is exactly ONE roster worth
    caching process-wide, not one copy per open conversation -- PROVIDED
    every caller sharing that one cached copy is provably the same
    authorization principal (see ``enabled`` below; this is not always
    true).

    ``ttl_seconds`` bounds staleness: the roster feeds a *soft* heuristic
    (routing a "switch to <Name>" construction to a refusal, see
    ``app.extraction.detect_foreign_patient_reference``'s signal 3) -- the
    unconditional numeric-id cross-patient signal is a completely separate
    code path and is unaffected by any staleness here. Worst case, a
    patient added to OpenEMR within the last ``ttl_seconds`` briefly misses
    the name-match signal (falls back to no refusal on that one construction
    until the next refresh) -- an acceptable, bounded trade against paying a
    full roster fetch on every matching turn.

    A SEPARATE staleness case, NOT covered by the TTL analysis above: an
    empty ``fetch()`` result (``[]``) is never cached -- see
    ``get_or_fetch``'s docstring. ``[]`` is ``get_patient_roster``'s
    fail-safe return for ANY OpenEMR API error, indistinguishable here from
    a genuinely empty roster; caching it would let one transient fetch
    failure go dark on signal 3 for every conversation, process-wide, for
    the rest of ``ttl_seconds`` -- a fail-OPEN window strictly worse than
    the pre-#174 per-conversation cache (which only poisoned the one
    conversation that hit the blip). Skipping the cache write on an empty
    result means a failed fetch is retried on every subsequent matching
    turn until one actually succeeds, same cost as having no cache at all.

    ``enabled`` -- Gate 2 (security) finding on #174, CONFIRMED: sharing one
    cached roster across every caller is only safe when every caller is the
    SAME authorization principal. With ``copilot_per_user_token_enabled``
    OFF (the default this class was designed for), every request runs as
    the same dev-bridge demo-clinician identity (see
    ``get_planner_factory``'s docstring) -- a shared roster there is just a
    shared view of that ONE principal's own data. With the flag ON,
    ``get_planner_factory`` binds the planner to the REQUEST's own forwarded
    bearer, so ``Planner.resolve_patient_roster`` executes
    ``GET /apis/default/api/patient`` AS THAT USER -- and
    ``docs/ARCHITECTURE.md``'s "Patient-context binding" section documents
    that OpenEMR's REST authorization is role- and resource-scoped (its
    Sec 124 Phase 4 example: an ``accountant``-role token gets 403 where an
    ``admin`` token gets 200 on the SAME endpoint shape). Sharing one cached
    roster across callers with DIFFERENT tokens in that mode would serve
    caller B a roster fetched under caller A's authorization -- a
    cross-principal authorization-scope leak, and a NEW risk #174's own
    per-conversation fetch never had (each conversation fetched under its
    OWN caller's token). ``enabled=False`` (wired by ``get_roster_cache``
    whenever ``copilot_per_user_token_enabled`` is true) makes
    ``get_or_fetch`` a pure passthrough -- fetch on EVERY call, cache
    nothing -- restoring the pre-#174 per-request-scoped fetch rather than
    silently sharing across principals. Deliberately NOT a per-token cache
    keyed some other way: token rotation would make such a cache grow
    unboundedly, reintroducing the exact unbounded-memory defect #174 exists
    to fix, just keyed by token instead of by conversation. Restoring the
    amplification fix under per-user tokens (keyed by authenticated
    principal instead of bypassed) is tracked as a separate follow-up --
    don't "simplify" this guard away without reading it first.

    Thread safety: shared, mutable, process-wide singleton
    (``_default_roster_cache``), same posture as ``ConversationStore`` above
    -- guarded by ``self._lock``. The fetch itself (an HTTP round trip) runs
    OUTSIDE the lock so concurrent requests are never serialized on network
    I/O, only on the cheap swap of the cached result; a stale-cache race
    (two threads both see an expired entry and both fetch) is accepted --
    last writer wins, and the roster is presumed identical regardless of
    which concurrent fetch produced it. (Not relevant when ``enabled`` is
    ``False``: there is no cached state to race over.)
    """

    def __init__(self, ttl_seconds: float, clock: Clock, *, enabled: bool = True) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._enabled = enabled
        self._lock = threading.Lock()
        self._roster: list[RosterEntry] | None = None
        self._expires_at: datetime | None = None

    def get_or_fetch(self, fetch: Callable[[], list[RosterEntry]]) -> list[RosterEntry]:
        """Return the cached roster if still fresh; otherwise call ``fetch``
        (``Planner.resolve_patient_roster``, an OpenEMR round trip),
        cache the result for ``ttl_seconds``, and return it.

        ``self._enabled is False`` (see the class docstring's
        authorization-scope analysis): always calls ``fetch`` and never
        reads or writes the cache -- every call gets its OWN, freshly
        fetched roster, scoped to whatever principal ``fetch`` itself is
        bound to.

        Gate 2 (security) re-review, CONFIRMED MAJOR: an empty ``fetch()``
        result is NEVER cached (``if roster:`` below). ``get_patient_roster``
        (``app.tools.patient_summary``) returns ``[]`` fail-safe on ANY
        OpenEMR API error -- a timeout, a 403, a 5xx blip -- indistinguishable
        here from a genuinely empty roster. Caching that ``[]`` would turn
        one transient OpenEMR blip, coinciding with the first "switch to
        <Name>" turn after process start, into signal 3
        (``app.extraction.detect_foreign_patient_reference``) going dark for
        EVERY conversation for the rest of ``ttl_seconds`` -- a process-wide
        fail-OPEN window, not just the one conversation that hit the blip
        (pre-#174, a fetch failure poisoned only that one conversation's own
        cache). Leaving ``self._roster``/``self._expires_at`` untouched on an
        empty result means the NEXT call sees the same (already-expired, or
        still-``None``) cache state and retries ``fetch()`` again -- costing
        exactly what calling ``fetch()`` on every matching turn cost before
        this cache existed, until a fetch actually succeeds.
        """
        if not self._enabled:
            return fetch()
        now = self._clock()
        with self._lock:
            if self._roster is not None and self._expires_at is not None and now < self._expires_at:
                return self._roster
        roster = fetch()
        if roster:
            with self._lock:
                self._roster = roster
                self._expires_at = now + timedelta(seconds=self._ttl_seconds)
        return roster


_default_roster_cache: RosterCache | None = None


def get_roster_cache(
    settings: Settings = Depends(get_settings), clock: Clock = Depends(get_clock)
) -> RosterCache:
    """FastAPI dependency: the active ``RosterCache`` (#174). Override in tests.

    Same "frozen at first use" posture as ``get_conversation_store`` above:
    ``settings.copilot_roster_cache_ttl_seconds`` is read once, the first
    time this dependency resolves, and baked into ``_default_roster_cache``
    for the rest of the process's life -- changing
    ``COPILOT_ROSTER_CACHE_TTL_SECONDS`` (or ``COPILOT_PER_USER_TOKEN_ENABLED``)
    in the environment after the process has served at least one ``/chat``
    request has no effect until restart.

    ``enabled=not settings.copilot_per_user_token_enabled`` -- Gate 2
    (security) finding on #174: see ``RosterCache``'s docstring for the full
    authorization-scope analysis. Sharing is only valid when every caller is
    provably the same principal, which is true under the dev-bridge default
    (flag OFF) and NOT true once each request is bound to its own forwarded
    bearer (flag ON, ``get_planner_factory``'s docstring).
    """
    global _default_roster_cache
    if _default_roster_cache is None:
        _default_roster_cache = RosterCache(
            ttl_seconds=settings.copilot_roster_cache_ttl_seconds,
            clock=clock,
            enabled=not settings.copilot_per_user_token_enabled,
        )
    return _default_roster_cache


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


def _no_op_source_ref_relevance_judge_provider() -> None:
    """Flag-OFF default (issue #170): no judge, no extra LLM call --
    ``run_verification`` treats ``source_ref_relevance_judge=None`` as "skip
    the SourceRef-relevance gate entirely," same posture as
    ``_no_op_support_judge_provider`` above."""
    return None


def get_source_ref_relevance_judge_provider(
    settings: Settings = Depends(get_settings),
) -> SupportJudgeProvider:
    """FastAPI dependency: the active SourceRef-relevance judge provider for
    /chat's issue #170 gate. Override in tests.

    Flag ON (``copilot_source_ref_relevance_enabled``): the provider builds
    the SAME text-generation client selected by ``copilot_llm_engine``, same
    posture as ``get_support_judge_provider`` above -- the two gates are
    duck-typed against the identical ``SemanticSupportJudgeLike`` protocol
    (see ``app.source_ref_relevance``'s module docstring).

    Flag OFF (default, and the ONLY state this has ever shipped in --
    MEASUREMENT-gated, see ``app.source_ref_relevance``'s module docstring):
    ``_no_op_source_ref_relevance_judge_provider`` -- ``None`` every time, so
    ``run_verification`` skips the gate with zero added latency."""
    if not settings.copilot_source_ref_relevance_enabled:
        return _no_op_source_ref_relevance_judge_provider

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
    # Keyword-only (#180 review finding): ``message``/``user``/``owner_token``
    # are three adjacent ``str`` parameters, and the sole production call
    # site (``chat_endpoint``) previously passed all fifteen arguments
    # positionally. A ``user``/``owner_token`` transposition there would be
    # invisible to mypy (both are plain ``str``) and to the existing tests
    # (``test_planner_streaming`` already calls by keyword; no ``/feedback``
    # test drives a real ``/chat`` call to catch it) -- the silent result
    # would be ``owner_token_hash`` computed over the DISPLAY username
    # instead of the bearer token (every caller sharing a username would
    # then own each other's traces), plus the raw token landing in
    # ``Turn.user``. Keyword-only turns that transposition into a
    # ``TypeError`` at the one call site instead.
    *,
    planner: PlannerProtocol,
    extractor: ClaimExtractorLike,
    conversation: Conversation,
    store: ConversationStore,
    trace_store: TraceStore,
    message: str,
    user: str,
    owner_token: str,
    clock: Clock,
    roster_cache: RosterCache,
    evidence_retriever: EvidenceRetriever = _no_op_evidence_retriever,
    patient_fact_provider: PatientFactProvider = _no_op_patient_fact_provider,
    support_judge_provider: SupportJudgeProvider = _no_op_support_judge_provider,
    require_answer_grounding: bool = False,
    require_tool_call_scoping: bool = False,
    source_ref_relevance_judge_provider: SupportJudgeProvider = _no_op_source_ref_relevance_judge_provider,
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

        def _roster_provider() -> list[RosterEntry]:
            # #237: resolved LAZILY -- this closure is only ever CALLED by
            # detect_foreign_patient_reference when a "switch to <Name>"
            # construction has already matched and isn't the bound patient,
            # so a turn that never uses that construction never pays this
            # round trip. #174: served from ``roster_cache``, a
            # process-wide, TTL'd cache SHARED across every conversation
            # (not this one conversation's own field) -- see
            # ``Conversation``'s docstring and ``RosterCache``'s docstring
            # for why a shared cache is correct here, not just cheaper. A
            # planner double with no ``resolve_patient_roster`` capability
            # (the pre-#237 default) never reaches the cache at all.
            resolve_roster = getattr(planner, "resolve_patient_roster", None)
            if resolve_roster is None:
                return []
            return roster_cache.get_or_fetch(resolve_roster)

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
            source_ref_relevance_judge=source_ref_relevance_judge_provider(),
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
            conversation,
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
                owner_token=owner_token,
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


# #177: the pre-auth guard below is STRUCTURAL, not positional. It works
# because ``get_planner_factory`` (imported above) itself takes
# ``token: str = Depends(get_authenticated_token)`` as a SUB-dependency --
# FastAPI resolves a dependency's own sub-dependencies before calling its
# body, so ``get_planner_factory``'s body (and the dev-token bridge fetch
# inside it) is unreachable until validation succeeds, regardless of where
# ``planner_factory`` is listed in THIS signature. The other dependencies
# below are auth-safe only in the sense that none of them do outbound I/O at
# resolution time -- they are NOT protected by signature order the way
# ``planner_factory`` is by the sub-dependency link. If ``get_planner_factory``
# is ever "simplified" back to reading the ``Authorization`` header directly
# (undoing the sub-dependency), or a future dependency added here does
# outbound I/O without depending on ``get_authenticated_token`` itself, #177
# reopens silently -- FastAPI will not complain, and only an endpoint-level
# test that resolves the real dependency graph (like
# ``test_chat_endpoint.py::test_unauthenticated_chat_never_touches_dev_token_bridge_transport``)
# will catch it. Do not "fix" this by reordering parameters here -- signature
# order on ``chat_endpoint`` itself has never been what makes this safe.
async def chat_endpoint(
    request: ChatRequest,
    token: str = Depends(get_authenticated_token),
    launch_binding_checker: LaunchBindingChecker = Depends(get_launch_binding_checker),
    planner_factory: PlannerFactory = Depends(get_planner_factory),
    extractor: ClaimExtractorLike = Depends(get_claim_extractor),
    store: ConversationStore = Depends(get_conversation_store),
    trace_store: TraceStore = Depends(get_trace_store),
    clock: Clock = Depends(get_clock),
    roster_cache: RosterCache = Depends(get_roster_cache),
    evidence_retriever: EvidenceRetriever = Depends(get_evidence_retriever),
    patient_fact_provider: PatientFactProvider = Depends(get_patient_fact_provider),
    support_judge_provider: SupportJudgeProvider = Depends(get_support_judge_provider),
    require_answer_grounding: bool = Depends(get_require_answer_grounding),
    require_tool_call_scoping: bool = Depends(get_require_tool_call_scoping),
    source_ref_relevance_judge_provider: SupportJudgeProvider = Depends(get_source_ref_relevance_judge_provider),
) -> StreamingResponse:
    # #177: token extraction + validation now happens in the
    # ``get_authenticated_token`` dependency itself (see its docstring), so
    # ``planner_factory`` above -- which depends on it -- is only ever
    # resolved for a caller whose token already passed. No try/except here
    # any more: a validation failure raises HTTPException(401) inside the
    # dependency and this body never runs.

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
            planner=planner,
            extractor=extractor,
            conversation=conversation,
            store=store,
            trace_store=trace_store,
            message=request.message,
            user=user,
            owner_token=token,
            clock=clock,
            roster_cache=roster_cache,
            evidence_retriever=evidence_retriever,
            patient_fact_provider=patient_fact_provider,
            support_judge_provider=support_judge_provider,
            require_answer_grounding=require_answer_grounding,
            require_tool_call_scoping=require_tool_call_scoping,
            source_ref_relevance_judge_provider=source_ref_relevance_judge_provider,
        ),
        media_type="text/event-stream",
    )

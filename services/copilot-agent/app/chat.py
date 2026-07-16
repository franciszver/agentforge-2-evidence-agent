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
alongside it and records a **request** span (whole invocation) and a
**verification** span (the P3.7 verdict fold) per turn, keyed by the SAME
correlation id ``Turn`` already carries. Per-tool and per-LLM-call spans are
NOT wired live here -- ``app.planner.ToolCallTrace`` and ``OllamaClient``
don't currently expose per-call timestamps/token counts, so emitting those
spans from this module would mean fabricating timing data; see
``app.trace_store.TraceStore.record_tool_span`` /
``.record_llm_span`` (built and tested, just not called from here yet) and
``.record_feedback_span`` (P4.3's ``/feedback`` endpoint seam).

SSE frame contract (``ChatEvent`` -- the P2.14/P3.8 UI's source of truth):
  * ``conversation`` -- first frame, carries ``{"conversation_id": str}``.
  * ``tool_call``    -- one per planner tool dispatch, in order, carrying
                         ``{"tool": str, "args": dict, "error": str | None}``.
  * ``answer``        -- the final answer, ``{"answer": str}``.
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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from fastapi import Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.correlation import get_correlation_id
from app.dev_token_bridge import DevTokenBridge
from app.extraction import ClaimExtractor, ClaimExtractorLike, run_verification
from app.ollama_client import OllamaClient
from app.openemr_client import OpenEmrClient
from app.planner import Planner, PlannerResult
from app.rendering import RenderedAnswer, RenderedClaim
from app.trace_store import TraceStore
from app.verdict import VerdictResult, to_trace_record

_logger = logging.getLogger(__name__)


class ChatEvent(StrEnum):
    """SSE event names emitted by ``POST /chat``."""

    CONVERSATION = "conversation"
    TOOL_CALL = "tool_call"
    ANSWER = "answer"
    VERIFICATION = "verification"
    DONE = "done"


class ChatRequest(BaseModel):
    """``POST /chat`` request body."""

    message: str
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
    """

    def run(self, question: str) -> PlannerResult: ...


TokenValidator = Callable[[str], None]
PlannerFactory = Callable[[int], PlannerProtocol]


def _default_token_validator(token: str) -> None:
    """Stub token validator: accepts any non-empty token.

    TODO: replace with real OpenEMR token introspection (e.g. the resource
    server's token-info endpoint) once that integration exists.
    """
    if not token:
        raise TokenValidationError("missing bearer token")


def get_token_validator() -> TokenValidator:
    """FastAPI dependency: the active ``TokenValidator``. Override in tests."""
    return _default_token_validator


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
            ollama_client=OllamaClient.from_settings(settings),
            openemr_client=OpenEmrClient.from_settings(settings),
            token=token,
            patient_id=patient_id,
        )

    return factory


def get_planner_factory(
    dev_token_bridge: DevTokenBridge = Depends(get_dev_token_bridge),
) -> PlannerFactory:
    """FastAPI dependency: builds a ``PlannerProtocol`` for a patient_id. Override in tests.

    The bridge's (potentially blocking, on a cache miss) token fetch happens
    here, in a sync dependency FastAPI runs in its worker-thread pool -- not in
    the ``async`` ``chat_endpoint`` body, so a token refresh never blocks the
    event loop.
    """
    return _default_planner_factory(dev_token_bridge.get_token())


def get_claim_extractor() -> ClaimExtractorLike:
    """FastAPI dependency: the answer->claims extractor. Override in tests.

    Built with ONLY an ``OllamaClient`` -- no tool registry, no OpenEMR
    client, no token -- so the extraction LLM is structurally tool-less (see
    ``app.extraction``'s security-boundary docstring). It is a distinct
    ``OllamaClient`` from the planner's, underscoring that the extractor
    never shares the planner's tool-selecting context.
    """
    return ClaimExtractor(ollama_client=OllamaClient.from_settings(get_settings()))


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
    """One multi-turn conversation, bound to the patient it was created for."""

    conversation_id: str
    patient_id: int
    history: list[Turn] = field(default_factory=list)


class ConversationStore:
    """In-memory conversation store keyed by ``conversation_id``.

    TODO(P4.2): replace with the durable trace store; this is a placeholder
    with the same shape (get / create / append) a DB-backed store would have.
    """

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def create(self, patient_id: int) -> Conversation:
        conversation = Conversation(conversation_id=str(uuid.uuid4()), patient_id=patient_id)
        self._conversations[conversation.conversation_id] = conversation
        return conversation

    def append_turn(self, conversation_id: str, turn: Turn) -> None:
        self._conversations[conversation_id].history.append(turn)


_default_store = ConversationStore()


def get_conversation_store() -> ConversationStore:
    """FastAPI dependency: the active ``ConversationStore``. Override in tests."""
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


def _record_span_best_effort(operation: str, write: Callable[[], object]) -> None:
    """Emit one trace span, best-effort. Observability must NEVER crash the
    /chat response: a failed span write (a root-owned ``/data`` ->
    ``PermissionError``, a full disk, a locked DB) is logged
    (correlation-tagged by the ``app.correlation`` logging seam; the payload
    carries only the operation label, never PHI) and swallowed, so the
    clinician's answer streams normally. A persistent trace-store failure
    surfaces on ``/ready``, which already gates trace-store writability --
    ``/chat`` degrades gracefully rather than 500ing.

    Catches ``Exception`` (not ``BaseException``): a ``GeneratorExit`` raised
    while a write runs in the ``finally`` below is a client disconnect and
    must keep propagating, not be swallowed here.
    """
    try:
        write()
    except Exception:
        _logger.warning(
            "trace span write failed; continuing without it",
            extra={"operation": operation},
            exc_info=True,
        )


def _stream_chat(
    planner: PlannerProtocol,
    extractor: ClaimExtractorLike,
    conversation: Conversation,
    store: ConversationStore,
    trace_store: TraceStore,
    message: str,
    user: str,
) -> Iterable[str]:
    correlation_id = get_correlation_id()
    request_start_ts = time.time()
    request_ok = True
    _logger.info(
        "chat invocation started",
        extra={"conversation_id": conversation.conversation_id, "patient_id": conversation.patient_id},
    )

    try:
        yield _sse(ChatEvent.CONVERSATION, {"conversation_id": conversation.conversation_id})

        result = planner.run(message)

        for call in result.trace:
            yield _sse(
                ChatEvent.TOOL_CALL,
                {"tool": call.tool.value, "args": call.args, "error": call.error},
            )

        yield _sse(ChatEvent.ANSWER, {"answer": result.answer})

        # Run the answer->claims extraction pipeline and populate the verification
        # frame with the REAL verdict / citation chips / warnings for this answer.
        # ``run_verification`` re-validates every extracted claim against the RAW
        # records (deterministic, no LLM) and strips the unverifiable ones, so a
        # miscited or injection-steered claim never reaches the user as fact.
        verification_start_ts = time.time()
        verdict_result, rendered = run_verification(extractor, result)
        verification_end_ts = time.time()
        verdict_trace_record = to_trace_record(verdict_result)
        # Yield the frame before the trace write -- the client shouldn't wait
        # on a disk write for data it already has.
        yield _sse(ChatEvent.VERIFICATION, build_verification_payload(verdict_result, rendered))
        _record_span_best_effort(
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

        yield _sse(ChatEvent.DONE, {})
    except BaseException:
        # BaseException, not Exception: an early client disconnect closes this
        # generator via GeneratorExit (a BaseException, not an Exception), and
        # that case must record ok=False too, not the request's default True.
        request_ok = False
        raise
    finally:
        _record_span_best_effort(
            "request_span",
            lambda: trace_store.record_request_span(
                correlation_id=correlation_id,
                start_ts=request_start_ts,
                end_ts=time.time(),
                ok=request_ok,
            ),
        )


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
    planner_factory: PlannerFactory = Depends(get_planner_factory),
    extractor: ClaimExtractorLike = Depends(get_claim_extractor),
    store: ConversationStore = Depends(get_conversation_store),
    trace_store: TraceStore = Depends(get_trace_store),
) -> StreamingResponse:
    try:
        token = extract_bearer_token(authorization)
        validator(token)
    except TokenValidationError as exc:
        raise HTTPException(status_code=401, detail="invalid or missing token") from exc

    user = _user_identity_from_token(token)

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
        conversation = store.create(request.patient_id)

    planner = planner_factory(request.patient_id)

    return StreamingResponse(
        _stream_chat(planner, extractor, conversation, store, trace_store, request.message, user),
        media_type="text/event-stream",
    )

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

Multi-turn state: an in-memory ``ConversationStore`` (TODO P4.2: replace with
the durable trace store) keyed by ``conversation_id``, binding each
conversation to the ``patient_id`` it was created with. Resuming with a
``conversation_id`` bound to a different ``patient_id`` is rejected (409) --
defense-in-depth for the patient-context binding the planner itself already
enforces (see ``app.planner`` module docstring).

SSE frame contract (``ChatEvent`` -- the P2.14 UI's source of truth):
  * ``conversation`` -- first frame, carries ``{"conversation_id": str}``.
  * ``tool_call``    -- one per planner tool dispatch, in order, carrying
                         ``{"tool": str, "args": dict, "error": str | None}``.
  * ``answer``        -- the final answer, ``{"answer": str}``.
  * ``done``           -- terminal frame, ``{}``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from fastapi import Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.ollama_client import OllamaClient
from app.openemr_client import OpenEmrClient
from app.planner import Planner, PlannerResult


class ChatEvent(StrEnum):
    """SSE event names emitted by ``POST /chat``."""

    CONVERSATION = "conversation"
    TOOL_CALL = "tool_call"
    ANSWER = "answer"
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


def _bearer_token_or_empty(authorization: str | None) -> str:
    """Extract the bearer token if present, else "" -- never raises.

    Used only to build the production planner factory's closure. The 401
    decision for a missing/invalid token is made by ``validator`` in
    ``chat_endpoint`` (via the raising ``_extract_bearer_token``); that check
    always runs, and always runs first, so a closure built here with an
    empty token is never actually invoked on the reject path.
    """
    prefix = "Bearer "
    if authorization and authorization.startswith(prefix):
        return authorization[len(prefix) :]
    return ""


def _default_planner_factory(token: str) -> PlannerFactory:
    """Build the production planner factory for one request's bearer token.

    Wires real ``OllamaClient``/``OpenEmrClient`` instances from ``Settings``
    (P0.5/P2.2's ``from_settings`` constructors) and passes the request's
    bearer token straight through to ``Planner`` -- the same token flows
    browser -> agent -> OpenEMR API tool calls, per the trust-boundary design
    (plan §5). The token's real-OAuth-token replacement for the current
    dev-only identity assertion is a documented, separate carry-forward
    (see ``TokenBrokerController``); tool calls against OpenEMR made with an
    invalid token fail per-call (caught as ``OpenEmrApiError`` in the planner
    loop) without crashing the conversation.
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


def get_planner_factory(authorization: str | None = Header(default=None)) -> PlannerFactory:
    """FastAPI dependency: builds a ``PlannerProtocol`` for a patient_id. Override in tests."""
    return _default_planner_factory(_bearer_token_or_empty(authorization))


@dataclass
class Conversation:
    """One multi-turn conversation, bound to the patient it was created for."""

    conversation_id: str
    patient_id: int
    history: list[tuple[str, str]] = field(default_factory=list)


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

    def append_turn(self, conversation_id: str, question: str, answer: str) -> None:
        self._conversations[conversation_id].history.append((question, answer))


_default_store = ConversationStore()


def get_conversation_store() -> ConversationStore:
    """FastAPI dependency: the active ``ConversationStore``. Override in tests."""
    return _default_store


def _sse(event: ChatEvent, data: dict[str, object]) -> str:
    return f"event: {event.value}\ndata: {json.dumps(data)}\n\n"


def _stream_chat(
    planner: PlannerProtocol,
    conversation: Conversation,
    store: ConversationStore,
    message: str,
) -> Iterable[str]:
    yield _sse(ChatEvent.CONVERSATION, {"conversation_id": conversation.conversation_id})

    result = planner.run(message)

    for call in result.trace:
        yield _sse(
            ChatEvent.TOOL_CALL,
            {"tool": call.tool.value, "args": call.args, "error": call.error},
        )

    yield _sse(ChatEvent.ANSWER, {"answer": result.answer})

    store.append_turn(conversation.conversation_id, message, result.answer)

    yield _sse(ChatEvent.DONE, {})


def _extract_bearer_token(authorization: str | None) -> str:
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise TokenValidationError("missing bearer token")
    return authorization[len(prefix) :]


async def chat_endpoint(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
    validator: TokenValidator = Depends(get_token_validator),
    planner_factory: PlannerFactory = Depends(get_planner_factory),
    store: ConversationStore = Depends(get_conversation_store),
) -> StreamingResponse:
    try:
        validator(_extract_bearer_token(authorization))
    except TokenValidationError as exc:
        raise HTTPException(status_code=401, detail="invalid or missing token") from exc

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
        _stream_chat(planner, conversation, store, request.message),
        media_type="text/event-stream",
    )

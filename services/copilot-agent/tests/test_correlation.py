"""Hermetic tests for the correlation-id middleware/contextvar seam (P4.1).

Proves propagation of ONE correlation id, minted per chat invocation, across
every pipeline stage: a log line (captured via ``caplog``), the tool-call
trace, an LLM-call site (``OllamaClient``), and the verification event -- plus
contextvar isolation across concurrent/sequential invocations.

The most subtle correctness point (see ``app/correlation.py`` module
docstring) is the SSE streaming lifetime: ``POST /chat`` returns a
``StreamingResponse`` wrapping a plain sync generator (``app.chat._stream_chat``).
Starlette drives that generator's ``next()`` calls through a worker-thread
pool (``iterate_in_threadpool``), one threadpool call PER YIELD, and each call
gets an independent ``contextvars.copy_context()`` snapshot. A contextvar
mutation made *inside* the generator body after a yield is therefore
discarded once that particular threaded call returns -- it does not survive
to the next yield. ``test_middleware_propagates_id_across_multiple_sse_yields``
and ``test_mutating_contextvar_inside_generator_does_not_survive_a_yield``
pin exactly this behaviour.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator

import httpx
import json
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.testclient import TestClient

from app.correlation import (
    CORRELATION_ID_HEADER,
    CorrelationIdMiddleware,
    configure_logging,
    correlation_scope,
    get_correlation_id,
)
from app.ollama_client import OllamaClient


# --------------------------------------------------------------------------
# contextvar mint/scope/isolation
# --------------------------------------------------------------------------


def test_get_correlation_id_lazily_mints_when_unset():
    # No scope bound at all -- get_correlation_id() must mint one itself
    # rather than return empty/None, and the mint must stick for the rest of
    # this context (a second read returns the SAME id, not a fresh one).
    from app.correlation import _correlation_id  # internal, test-only peek

    token = _correlation_id.set(None)  # force the "unset" starting state
    try:
        first = get_correlation_id()
        assert first
        assert get_correlation_id() == first
    finally:
        _correlation_id.reset(token)


def test_correlation_scope_with_no_argument_mints_a_fresh_id():
    with correlation_scope() as first:
        assert first
        assert get_correlation_id() == first  # stable within the same scope


def test_middleware_passes_through_non_http_scopes_untouched():
    import asyncio

    calls: list[tuple] = []

    async def inner_app(scope, receive, send) -> None:
        calls.append((scope, receive, send))

    middleware = CorrelationIdMiddleware(inner_app)
    scope = {"type": "lifespan"}
    receive = object()
    send = object()

    asyncio.run(middleware(scope, receive, send))

    assert calls == [(scope, receive, send)]  # untouched -- no header injection wrapper


def test_correlation_scope_binds_and_restores_previous_value():
    with correlation_scope("outer-id"):
        assert get_correlation_id() == "outer-id"
        with correlation_scope("inner-id"):
            assert get_correlation_id() == "inner-id"
        assert get_correlation_id() == "outer-id"


def test_sequential_invocations_get_distinct_ids():
    with correlation_scope() as first:
        pass
    with correlation_scope() as second:
        pass
    assert first != second


def test_concurrent_invocations_do_not_leak_ids_across_threads():
    seen: dict[str, str] = {}

    def worker(name: str) -> None:
        with correlation_scope(f"id-{name}"):
            time.sleep(0.05)  # widen the window for a race to manifest
            seen[name] = get_correlation_id()

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {"a": "id-a", "b": "id-b", "c": "id-c"}


# --------------------------------------------------------------------------
# logging seam: every log line carries the bound correlation id
# --------------------------------------------------------------------------


def test_log_line_emitted_during_a_scope_carries_its_correlation_id(caplog):
    configure_logging()
    logger = logging.getLogger("app.test_correlation_logging")
    caplog.set_level(logging.INFO, logger="app.test_correlation_logging")

    with correlation_scope("scoped-id-1"):
        logger.info("something happened")

    record = next(r for r in caplog.records if r.message == "something happened")
    assert record.correlation_id == "scoped-id-1"


def test_log_line_outside_any_scope_still_has_a_correlation_id_field(caplog):
    configure_logging()
    logger = logging.getLogger("app.test_correlation_logging_unscoped")
    caplog.set_level(logging.INFO, logger="app.test_correlation_logging_unscoped")

    logger.info("no scope bound")

    record = next(r for r in caplog.records if r.message == "no scope bound")
    assert hasattr(record, "correlation_id")


# --------------------------------------------------------------------------
# the SSE-generator lifetime subtlety
# --------------------------------------------------------------------------


def _build_streaming_app(log_name: str) -> FastAPI:
    configure_logging()
    logger = logging.getLogger(log_name)
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    def gen() -> Iterator[str]:
        logger.info("gen-start")
        yield "chunk1\n"
        logger.info("gen-after-yield1")
        yield "chunk2\n"
        logger.info("gen-after-yield2")
        yield "chunk3\n"

    @app.get("/stream")
    def stream() -> StreamingResponse:
        return StreamingResponse(gen(), media_type="text/plain")

    return app


def test_middleware_propagates_id_across_multiple_sse_yields(caplog):
    log_name = "app.test_sse_propagation"
    caplog.set_level(logging.INFO, logger=log_name)
    app = _build_streaming_app(log_name)
    client = TestClient(app)

    response = client.get("/stream")

    assert response.status_code == 200
    header_id = response.headers[CORRELATION_ID_HEADER]
    assert header_id

    records = [r for r in caplog.records if r.name == log_name]
    assert [r.message for r in records] == ["gen-start", "gen-after-yield1", "gen-after-yield2"]
    # Every log line emitted DURING the streamed generator -- including ones
    # after multiple yields, each dispatched to a fresh worker thread -- must
    # carry the SAME id the middleware minted for this one request.
    assert {r.correlation_id for r in records} == {header_id}


def test_middleware_honors_inbound_header_and_gives_sequential_requests_distinct_ids(caplog):
    log_name = "app.test_sse_propagation_inbound"
    caplog.set_level(logging.INFO, logger=log_name)
    app = _build_streaming_app(log_name)
    client = TestClient(app)

    inbound = client.get("/stream", headers={CORRELATION_ID_HEADER: "caller-supplied-id"})
    assert inbound.headers[CORRELATION_ID_HEADER] == "caller-supplied-id"

    first = client.get("/stream")
    second = client.get("/stream")
    assert first.headers[CORRELATION_ID_HEADER] != second.headers[CORRELATION_ID_HEADER]


def test_mutating_contextvar_inside_generator_does_not_survive_a_yield(caplog):
    """Pins the subtle failure mode: a ``.set()`` performed INSIDE the sync
    generator body between yields is confined to that yield's threaded
    context copy and is lost by the next resume -- proving the id must be
    bound BEFORE the streamed generator starts (middleware/async endpoint),
    not by "refreshing" inside the generator itself."""
    from app.correlation import _correlation_id  # internal, test-only peek

    log_name = "app.test_sse_local_mutation"
    caplog.set_level(logging.INFO, logger=log_name)
    logger = logging.getLogger(log_name)
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    def gen() -> Iterator[str]:
        logger.info("start", extra={})
        yield "chunk1\n"
        # Mutate LOCALLY inside the generator -- this should NOT survive.
        _correlation_id.set("mutated-locally")
        logger.info("mutated")
        yield "chunk2\n"
        logger.info("after-mutation-attempt")
        yield "chunk3\n"

    @app.get("/stream")
    def stream() -> StreamingResponse:
        return StreamingResponse(gen(), media_type="text/plain")

    client = TestClient(app)
    response = client.get("/stream")
    header_id = response.headers[CORRELATION_ID_HEADER]

    records = [r for r in caplog.records if r.name == log_name]
    by_msg = {r.message: r.correlation_id for r in records}
    assert by_msg["start"] == header_id
    assert by_msg["mutated"] == "mutated-locally"  # visible in ITS OWN threaded call
    # But the mutation did not leak forward to the next resume.
    assert by_msg["after-mutation-attempt"] == header_id


# --------------------------------------------------------------------------
# LLM-call site: OllamaClient.chat / .extract
# --------------------------------------------------------------------------


def _ollama_client(handler) -> OllamaClient:
    return OllamaClient(base_url="http://ollama:11434", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_ollama_chat_call_carries_the_active_correlation_id(caplog):
    configure_logging()
    caplog.set_level(logging.INFO, logger="app.ollama_client")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True}).encode() + b"\n"
        return httpx.Response(200, content=body)

    client = _ollama_client(handler)

    with correlation_scope("ollama-chat-id"):
        client.chat([{"role": "user", "content": "hi"}])

    records = [r for r in caplog.records if r.name == "app.ollama_client"]
    assert records, "expected at least one log line from an LLM-call site"
    assert all(r.correlation_id == "ollama-chat-id" for r in records)


def test_ollama_extract_call_carries_the_active_correlation_id(caplog):
    from pydantic import BaseModel

    class _Schema(BaseModel):
        name: str

    configure_logging()
    caplog.set_level(logging.INFO, logger="app.ollama_client")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps({"message": {"role": "assistant", "content": '{"name": "x"}'}}).encode()
        return httpx.Response(200, content=body)

    client = _ollama_client(handler)

    with correlation_scope("ollama-extract-id"):
        client.extract([{"role": "user", "content": "hi"}], _Schema)

    records = [r for r in caplog.records if r.name == "app.ollama_client"]
    assert records
    assert all(r.correlation_id == "ollama-extract-id" for r in records)


# --------------------------------------------------------------------------
# tool-call record: Planner
# --------------------------------------------------------------------------


def test_planner_tool_dispatch_logs_with_the_active_correlation_id(caplog):
    from app.planner import Planner, ToolSpec
    from app.schemas.planner import PlannerAction, PlannerDecision, ToolName
    from app.schemas.tools import GetMedicationsInput, MedicationsOutput

    configure_logging()
    caplog.set_level(logging.INFO, logger="app.planner")

    class _ScriptedOllama:
        def __init__(self) -> None:
            self._decisions = [
                PlannerDecision(
                    action=PlannerAction.CALL_TOOL, tool=ToolName.GET_MEDICATIONS, reason="go"
                ),
                PlannerDecision(action=PlannerAction.ANSWER, reason="done", final_answer="done"),
            ]

        def extract(self, messages, schema):
            from app.quarantine import QuarantineSummary
            from app.schemas.planner import FinalAnswer

            if schema is QuarantineSummary:
                return QuarantineSummary(summary="s")
            if schema is FinalAnswer:
                return FinalAnswer(answer="done")
            return self._decisions.pop(0)

        def chat(self, messages, *, options=None) -> str:
            return "reasoning"

    def fake_get_medications(client, token, patient_id, **kwargs) -> MedicationsOutput:
        return MedicationsOutput(items=[])

    registry = {
        ToolName.GET_MEDICATIONS: ToolSpec(
            description="fake", input_schema=GetMedicationsInput, func=fake_get_medications
        )
    }
    planner = Planner(
        ollama_client=_ScriptedOllama(), openemr_client=object(), token="tok", patient_id=1, registry=registry
    )

    with correlation_scope("planner-tool-id"):
        planner.run("what meds?")

    records = [r for r in caplog.records if r.name == "app.planner"]
    assert records, "expected a tool-call log line from the planner"
    assert all(r.correlation_id == "planner-tool-id" for r in records)


# --------------------------------------------------------------------------
# verification event
# --------------------------------------------------------------------------


def test_run_verification_logs_with_the_active_correlation_id(caplog):
    from app.extraction import run_verification
    from app.planner import PlannerResult

    configure_logging()
    caplog.set_level(logging.INFO, logger="app.extraction")

    class _EmptyExtractor:
        def extract_claims(self, *, answer, tools, raw_results):
            return []

    result = PlannerResult(answer="ok", trace=[], raw_results=[])

    with correlation_scope("verification-id"):
        run_verification(_EmptyExtractor(), result)

    records = [r for r in caplog.records if r.name == "app.extraction"]
    assert records, "expected a verification log line"
    assert all(r.correlation_id == "verification-id" for r in records)


# --------------------------------------------------------------------------
# end-to-end: real /chat endpoint (P2.17 Turn.correlation_id unification)
# --------------------------------------------------------------------------


def test_chat_endpoint_response_header_matches_the_recorded_turn_correlation_id(caplog):
    from app.chat import ConversationStore, get_conversation_store, get_planner_factory, get_token_validator
    from app.main import app as real_app
    from app.planner import PlannerResult

    configure_logging()
    caplog.set_level(logging.INFO, logger="app.chat")

    class _FakePlanner:
        def run(self, question: str) -> PlannerResult:
            return PlannerResult(answer="ok", trace=[], raw_results=[])

    def _ok_validator(token: str) -> None:
        return None

    store = ConversationStore()
    real_app.dependency_overrides[get_token_validator] = lambda: _ok_validator
    real_app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: _FakePlanner())
    real_app.dependency_overrides[get_conversation_store] = lambda: store
    try:
        client = TestClient(real_app)
        response = client.post(
            "/chat",
            json={"message": "hi", "patient_id": 1},
            headers={"Authorization": "Bearer good-token"},
        )
    finally:
        real_app.dependency_overrides.clear()

    assert response.status_code == 200
    header_id = response.headers[CORRELATION_ID_HEADER]

    conversation_id = next(
        json.loads(data)["conversation_id"]
        for name, data in _iter_sse(response.text)
        if name == "conversation"
    )
    conversation = store.get(conversation_id)
    assert conversation is not None
    turn = conversation.history[0]

    # P2.17's per-turn id IS this correlation id now -- not a second, separate
    # uuid4 minted independently.
    assert turn.correlation_id == header_id


def test_chat_endpoint_logs_never_carry_patient_id(caplog):
    """PHI guard (#144): the JSON log formatter renders every ``extra=``
    attribute on a record, so a log call site attaching ``patient_id`` --
    even one that predates this seam -- would now leak it into every
    emitted log line. No log record from a ``/chat`` invocation may carry
    a ``patient_id`` attribute; ``conversation_id``/``correlation_id`` are
    the non-PHI identifiers used for tracing instead."""
    from app.chat import ConversationStore, get_conversation_store, get_planner_factory, get_token_validator
    from app.main import app as real_app
    from app.planner import PlannerResult

    configure_logging()
    caplog.set_level(logging.INFO, logger="app.chat")

    class _FakePlanner:
        def run(self, question: str) -> PlannerResult:
            return PlannerResult(answer="ok", trace=[], raw_results=[])

    def _ok_validator(token: str) -> None:
        return None

    store = ConversationStore()
    real_app.dependency_overrides[get_token_validator] = lambda: _ok_validator
    real_app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: _FakePlanner())
    real_app.dependency_overrides[get_conversation_store] = lambda: store
    try:
        client = TestClient(real_app)
        client.post(
            "/chat",
            json={"message": "hi", "patient_id": 1},
            headers={"Authorization": "Bearer good-token"},
        )
    finally:
        real_app.dependency_overrides.clear()

    records = [r for r in caplog.records if r.name == "app.chat"]
    assert records
    assert not any(hasattr(r, "patient_id") for r in records)


def _iter_sse(text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        events.append((name, "\n".join(data_lines)))
    return events

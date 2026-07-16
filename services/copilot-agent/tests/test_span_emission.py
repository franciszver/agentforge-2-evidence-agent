"""Hermetic tests: ``POST /chat`` emits ``tool`` + ``llm`` spans (#149).

Extends P4.2's request/verification span wiring (``test_chat_trace_emission.py``)
to the ``tool`` and ``llm`` spans P4.5's dashboard aggregates (tool_call_count,
retry_count, avg_tokens_per_request) but that nothing emitted live before this
change -- ``app.planner.ToolCallTrace`` carried no timing and ``OllamaClient``
surfaced no token counts. Everything here is faked (planner, extractor, token
validator); no real Ollama/OpenEMR/network is ever touched.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chat import get_claim_extractor, get_planner_factory, get_token_validator, get_trace_store
from app.main import app
from app.ollama_client import LlmCallStats
from app.planner import PlannerResult, ToolCallTrace
from app.schemas.planner import ToolName
from app.trace_store import SpanStatus, SpanType, TraceStore
from tests.test_chat_endpoint import FakeExtractor, _iter_sse_events

# Derived per run (not a hardcoded literal) so no secret-shaped string is
# committed; stable within a run for any within-test consistency.
_TEST_HASH_KEY = secrets.token_hex(16)


@pytest.fixture(autouse=True)
def _reset_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _override_ok_validator() -> None:
    def _validator(token: str) -> None:
        return None

    app.dependency_overrides[get_token_validator] = lambda: _validator


client = TestClient(app)


class FakePlannerWithLlmCalls:
    """``FakePlanner`` (test_chat_endpoint.py) plus a scripted ``llm_calls``
    list, so tests can drive ``PlannerResult.llm_calls`` -- the side channel
    ``app.planner.Planner`` fills from ``OllamaClient.call_stats``."""

    def __init__(
        self,
        trace: list[ToolCallTrace],
        answer: str,
        llm_calls: list[LlmCallStats] | None = None,
        raw_results: list[dict | None] | None = None,
    ) -> None:
        self._trace = trace
        self._answer = answer
        self._llm_calls = llm_calls or []
        self._raw_results = raw_results or []

    def run(self, question: str) -> PlannerResult:
        return PlannerResult(
            answer=self._answer,
            trace=self._trace,
            raw_results=self._raw_results,
            llm_calls=self._llm_calls,
        )


class FakeExtractorWithLlmCalls(FakeExtractor):
    """``FakeExtractor`` plus a scripted ``llm_calls`` list, mirroring
    ``ClaimExtractor.llm_calls`` (read from its own ``OllamaClient``)."""

    def __init__(self, llm_calls: list[LlmCallStats] | None = None) -> None:
        super().__init__()
        self.llm_calls = llm_calls or []


def _tool_call(
    *,
    args: dict[str, object] | None = None,
    error: str | None = None,
    start_ts: float = 100.0,
    end_ts: float = 100.25,
) -> ToolCallTrace:
    return ToolCallTrace(
        tool=ToolName.GET_MEDICATIONS,
        args=args or {},
        result=None if error else {"count": 1},
        error=error,
        start_ts=start_ts,
        end_ts=end_ts,
    )


def _llm_stats(
    *,
    ok: bool = True,
    tokens_in: int | None = 10,
    tokens_out: int | None = 5,
    start_ts: float = 100.0,
    end_ts: float = 100.2,
    model: str = "qwen3:4b",
) -> LlmCallStats:
    return LlmCallStats(model=model, start_ts=start_ts, end_ts=end_ts, ok=ok, tokens_in=tokens_in, tokens_out=tokens_out)


def _post_chat():
    return client.post(
        "/chat",
        json={"message": "What meds is she on?", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )


def test_stream_chat_emits_a_tool_span_per_dispatched_tool_call(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    fake_planner = FakePlannerWithLlmCalls(trace=[_tool_call()], answer="She is on lisinopril.")

    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractorWithLlmCalls()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = _post_chat()
    assert response.status_code == 200
    correlation_id = response.headers["X-Correlation-ID"]

    spans = trace_store.get_spans(correlation_id)
    tool_spans = [span for span in spans if span.span_type == SpanType.TOOL]

    assert len(tool_spans) == 1
    tool_span = tool_spans[0]
    assert tool_span.tool_name == ToolName.GET_MEDICATIONS.value
    assert tool_span.status == SpanStatus.OK
    assert tool_span.error_category is None
    assert tool_span.duration_ms == pytest.approx(250.0, abs=1.0)
    assert tool_span.args_hash is not None


def test_stream_chat_emits_a_failed_tool_span_with_error_category(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    fake_planner = FakePlannerWithLlmCalls(trace=[_tool_call(error="forbidden")], answer="couldn't retrieve meds")

    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractorWithLlmCalls()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = _post_chat()
    correlation_id = response.headers["X-Correlation-ID"]

    tool_span = next(span for span in trace_store.get_spans(correlation_id) if span.span_type == SpanType.TOOL)
    assert tool_span.status == SpanStatus.FAIL
    assert tool_span.error_category == "forbidden"


def test_stream_chat_emits_llm_spans_from_both_planner_and_extractor(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    planner_llm_calls = [_llm_stats(tokens_in=10, tokens_out=5), _llm_stats(tokens_in=20, tokens_out=8)]
    extractor_llm_calls = [_llm_stats(tokens_in=30, tokens_out=12)]
    fake_planner = FakePlannerWithLlmCalls(trace=[], answer="ok", llm_calls=planner_llm_calls)

    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractorWithLlmCalls(extractor_llm_calls)
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = _post_chat()
    correlation_id = response.headers["X-Correlation-ID"]

    llm_spans = [span for span in trace_store.get_spans(correlation_id) if span.span_type == SpanType.LLM]

    assert len(llm_spans) == 3
    assert {(span.tokens_in, span.tokens_out) for span in llm_spans} == {(10, 5), (20, 8), (30, 12)}
    assert all(span.model == "qwen3:4b" for span in llm_spans)
    assert all(span.status == SpanStatus.OK for span in llm_spans)


def test_stream_chat_emits_a_failed_llm_span(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    fake_planner = FakePlannerWithLlmCalls(trace=[], answer="ok", llm_calls=[_llm_stats(ok=False, tokens_in=None, tokens_out=None)])

    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractorWithLlmCalls()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = _post_chat()
    correlation_id = response.headers["X-Correlation-ID"]

    llm_span = next(span for span in trace_store.get_spans(correlation_id) if span.span_type == SpanType.LLM)
    assert llm_span.status == SpanStatus.FAIL
    assert llm_span.tokens_in is None


def test_no_phi_tool_args_are_hashed_never_stored_raw_in_the_trace_db(tmp_path: Path) -> None:
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)
    sentinel = "PATIENT-SPECIFIC-DISTINCTIVE-MARKER-42"
    fake_planner = FakePlannerWithLlmCalls(trace=[_tool_call(args={"note": sentinel})], answer="ok")

    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractorWithLlmCalls()
    app.dependency_overrides[get_trace_store] = lambda: trace_store

    response = _post_chat()
    assert response.status_code == 200

    raw_bytes = Path(trace_store.db_path).read_bytes()
    assert sentinel.encode() not in raw_bytes


class _RaisingToolAndLlmTraceStore:
    """A trace store whose tool/llm span writes raise -- proves emission is
    best-effort (a write failure must never break the streamed /chat answer),
    same property P4.2 already proved for request/verification spans."""

    def record_request_span(self, **kwargs: object) -> int:
        return 0

    def record_verification_span(self, **kwargs: object) -> int:
        return 0

    def record_tool_span(self, **kwargs: object) -> int:
        raise PermissionError("[Errno 13] Permission denied: '/data'")

    def record_llm_span(self, **kwargs: object) -> int:
        raise PermissionError("[Errno 13] Permission denied: '/data'")


def test_chat_survives_a_tool_and_llm_span_write_failure() -> None:
    fake_planner = FakePlannerWithLlmCalls(
        trace=[_tool_call()], answer="best-effort answer", llm_calls=[_llm_stats()]
    )
    _override_ok_validator()
    app.dependency_overrides[get_planner_factory] = lambda: (lambda patient_id: fake_planner)
    app.dependency_overrides[get_claim_extractor] = lambda: FakeExtractorWithLlmCalls([_llm_stats()])
    app.dependency_overrides[get_trace_store] = lambda: _RaisingToolAndLlmTraceStore()

    response = _post_chat()

    assert response.status_code == 200
    event_names = [name for name, _ in _iter_sse_events(response.text)]
    assert "answer" in event_names
    assert "verification" in event_names
    assert "done" in event_names

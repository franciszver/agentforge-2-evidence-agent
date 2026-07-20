"""Red-first integration test for P3.9: wiring the P3.5 supervisor + P3.6
document-citation verification + P3.8 encounter observability into a REAL
``POST /chat`` turn (`app/chat.py`'s ``get_evidence_retriever`` /
``_log_encounter_record``).

Everything is hermetic: the planner and claim extractor are scripted
doubles (no live Ollama call for the CHAT/EXTRACTION model), but the
evidence-retrieval path exercises the REAL P3.5 ``Supervisor`` ->
``EvidenceRetrieverWorker`` -> ``app.reranking.retrieve_and_rerank`` chain
over the real corpus, using the same recorded query-vector / reranker-score
fixtures ``tests/test_supervisor.py`` / ``tests/test_encounter_observability.py``
already use -- so "the supervisor routed through evidence-retriever" and "the
document citation verifies against the RAW corpus" are both genuinely
exercised, not just asserted from a canned double.

The no-PHI / no-``str(exc)`` test is the load-bearing security case: a
retrieval failure carrying a PHI-shaped message must surface in the log only
as ``error_type``, never the message text.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.chat import (
    ChatEvent,
    _no_op_support_judge_provider,
    get_claim_extractor,
    get_evidence_retriever,
    get_planner_factory,
    get_support_judge_provider,
    get_token_validator,
    get_trace_store,
)
from app.main import app
from app.planner import PlannerResult
from app.reranking import (
    RERANKER_SCORES_PATH,
    RecordedRerankScorer,
    Reranker,
    RerankError,
    load_recorded_reranker_scores,
)
from app.retrieval import build_retriever_from_corpus, recorded_query_vector
from app.schemas.ingestion import DocumentCitation
from app.schemas.verification import Claim
from app.supervisor import EvidenceRetrieverWorker, IntakeExtractorWorker, RetrieveSubTask, Supervisor
from app.trace_store import SpanType, TraceStore
from scripts.retrieval_golden_queries import GOLDEN_QUERIES

_GOLDEN_QUERY, _GOLDEN_CHUNK_ID = GOLDEN_QUERIES[0]  # ("What A1c target for most adults?", "a1c-targets#target-ranges")
_GOLDEN_DOC_ID = "a1c-targets"
_VERBATIM_QUOTE = "A1c target generally below 7%"

_TEST_HASH_KEY = "0" * 32


class _FakePlanner:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def run(self, question: str, guideline_excerpts: object = None) -> PlannerResult:
        return PlannerResult(answer=self._answer, trace=[], raw_results=[])


class _FakeExtractor:
    """Returns one claim citing the real retrieved guideline chunk (P3.9) --
    proves the wiring reaches ``check_document_citation`` against a REAL
    ``CorpusChunkIndex`` built from the SAME chunk, without needing a live
    extraction-model call."""

    def extract_claims(self, *, answer, tools, raw_results, retrieved_chunks=()) -> list[Claim]:
        assert retrieved_chunks, "expected the real supervisor's retrieved chunks to reach the extractor"
        return [
            Claim(
                text="Most adults should target an A1c below 7%.",
                document_citations=[
                    DocumentCitation(
                        source_type="guideline_chunk",
                        source_id=_GOLDEN_DOC_ID,
                        page_or_section="Target Ranges",
                        field_or_chunk_id=_GOLDEN_CHUNK_ID,
                        quote_or_value=_VERBATIM_QUOTE,
                    )
                ],
            )
        ]


def _real_evidence_retriever(trace_store: TraceStore):
    """Wraps a REAL ``Supervisor``/``EvidenceRetrieverWorker`` over the real
    corpus, using recorded (not live) query-vector/reranker-score fixtures --
    so the /chat turn's retrieval genuinely routes through the P3.5
    supervisor chain (traced into ``trace_store``), the same way the
    production ``get_evidence_retriever`` dependency does when the flag is
    on, just without a live Ollama embedding/rerank call."""
    supervisor = Supervisor(
        intake_worker=IntakeExtractorWorker(ollama_client=object(), document_store=object(), fact_store=object()),  # type: ignore[arg-type]
        evidence_worker=EvidenceRetrieverWorker(
            retriever=build_retriever_from_corpus(),
            reranker=Reranker(RecordedRerankScorer(load_recorded_reranker_scores(RERANKER_SCORES_PATH))),
        ),
        trace_store=trace_store,
    )

    def _retrieve(query: str):
        result = supervisor.handle(
            RetrieveSubTask(query=_GOLDEN_QUERY, k=3, query_vector=recorded_query_vector(_GOLDEN_QUERY))
        )
        return result.payload

    return _retrieve


def _failing_evidence_retriever(query: str):
    raise RerankError(f"PHI-MARKER-DO-NOT-LOG: {query}")


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


def _iter_sse_events(text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        events.append((event_name, "\n".join(data_lines)))
    return events


def test_real_chat_turn_produces_a_verified_document_citation_via_the_traced_supervisor(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (
        lambda patient_id: _FakePlanner("Most adults should target an A1c below 7%.")
    )
    app.dependency_overrides[get_claim_extractor] = lambda: _FakeExtractor()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    app.dependency_overrides[get_evidence_retriever] = lambda: _real_evidence_retriever(trace_store)
    # This test's extractor is a scripted double (no live model) -- the
    # issue #47 semantic-support gate needs an LLM-capable judge, which this
    # test has none of, so keep it off here regardless of Settings' default
    # (issue #81) to stay hermetic; the gate itself is covered separately by
    # tests/test_semantic_support.py and tests/test_support_judge_provider.py.
    app.dependency_overrides[get_support_judge_provider] = lambda: _no_op_support_judge_provider

    response = client.post(
        "/chat",
        json={"message": _GOLDEN_QUERY, "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 200
    events = _iter_sse_events(response.text)
    verification_data = next(data for name, data in events if name == ChatEvent.VERIFICATION.value)

    # The claim's document citation verified against the REAL corpus text --
    # not a fabricated/paraphrased quote.
    assert '"guideline_chunk"' in verification_data
    assert _GOLDEN_CHUNK_ID in verification_data
    assert '"verdict":"verified"' in verification_data.replace(" ", "")

    # Nothing was stripped to a Notice: the claim survived, carrying its
    # document citation.
    import json

    payload = json.loads(verification_data)
    (segment,) = payload["segments"]
    assert segment["type"] == "claim"
    (citation,) = segment["document_citations"]
    assert citation["source_type"] == "guideline_chunk"
    assert citation["field_or_chunk_id"] == _GOLDEN_CHUNK_ID

    # The supervisor really dispatched to the evidence-retriever worker --
    # a durable, traced ``worker`` span landed in THIS request's trace store.
    correlation_id = json.loads(next(data for name, data in events if name == "conversation"))["correlation_id"]
    spans = trace_store.get_spans(correlation_id)
    worker_spans = [s for s in spans if s.span_type == SpanType.WORKER]
    assert any(s.worker_name == "evidence-retriever" and s.status.value == "ok" for s in worker_spans)

    # No PHI, no raw exception text anywhere in the response or the logs.
    for record in caplog.records:
        rendered = record.getMessage()
        assert _GOLDEN_QUERY not in rendered
        assert "A1c below 7%" not in rendered


def test_evidence_retrieval_failure_is_fail_soft_and_logs_only_the_error_type(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    trace_store = TraceStore(db_path=str(tmp_path / "traces.db"), hash_secret=_TEST_HASH_KEY)

    app.dependency_overrides[get_token_validator] = lambda: (lambda token: None)
    app.dependency_overrides[get_planner_factory] = lambda: (
        lambda patient_id: _FakePlanner("An ordinary chart-data answer.")
    )
    # A plain no-claims extractor so the failing retriever path is what's
    # under test, not the fake claim's own assertion.
    class _NoClaimsExtractor:
        def extract_claims(self, *, answer, tools, raw_results, retrieved_chunks=()) -> list[Claim]:
            return []

    app.dependency_overrides[get_claim_extractor] = lambda: _NoClaimsExtractor()
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    app.dependency_overrides[get_evidence_retriever] = lambda: _failing_evidence_retriever
    app.dependency_overrides[get_support_judge_provider] = lambda: _no_op_support_judge_provider

    response = client.post(
        "/chat",
        json={"message": "some question about patient-specific-marker-xyz", "patient_id": 1},
        headers={"Authorization": "Bearer good-token"},
    )

    # The turn still completes -- retrieval failure never breaks the chat turn.
    assert response.status_code == 200
    events = _iter_sse_events(response.text)
    assert any(name == ChatEvent.DONE.value for name, _ in events)

    marker = "PHI-MARKER-DO-NOT-LOG"
    assert marker not in response.text
    for record in caplog.records:
        assert marker not in record.getMessage()
        assert marker not in repr(record.__dict__.get("error_type", ""))

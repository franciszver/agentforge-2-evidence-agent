"""Red-first tests for issue #70: ``runner.pipeline.run_case`` must thread a
case's ``patient_facts`` fixture (``runner.schema.PatientFactFixture``) into
BOTH consumers a real turn feeds from one source (#86's ``app.chat``
fetch-once-use-twice convention):

  * ``Planner.run``'s ``document_facts`` kwarg -- the fact's literal quote
    must reach the planner's own answer-composition (reasoning) call, not
    just post-hoc verification (mirrors ``test_guideline_context_wiring.py``
    for ``retrieved_chunks``/``guideline_excerpts``).
  * ``app.extraction.run_verification``'s ``patient_facts`` kwarg -- a claim
    citing the fixture's exact ``(source_id, field_or_chunk_id)`` key must
    verify (VALID), proving the ``DocumentFactIndex`` was actually built from
    it, not silently skipped.
"""

from __future__ import annotations

from pathlib import Path

from app.schemas.verification import Claim, DocumentCitation
from app.verdict import Verdict

from runner.loader import load_case
from runner.pipeline import run_case
from runner.tests.conftest import ReasoningCaptureOllamaClient

_FIXTURES = Path(__file__).parent / "fixtures"
_CASES = _FIXTURES / "cases"


def test_run_case_threads_patient_facts_into_the_reasoning_call() -> None:
    case = load_case(_CASES / "patient-facts-smoke.yaml")
    assert case.patient_facts, "fixture must declare at least one patient fact for this test to mean anything"
    client = ReasoningCaptureOllamaClient(final_answer="Her creatinine was 0.9.")

    run_case(case, client)

    assert client.chat_messages, "the planner never reached its reasoning call"
    fact_quote = case.patient_facts[0].quote_or_value
    reasoning_messages = client.chat_messages[-1]
    joined = " ".join(message["content"] for message in reasoning_messages)
    assert fact_quote in joined, (
        "issue #70: the case's patient_facts fixture quote must reach the "
        "planner's answer-composition (reasoning) call, mirroring #86's "
        "app.chat document_facts wiring"
    )


def test_run_case_threads_patient_facts_into_verification() -> None:
    case = load_case(_CASES / "patient-facts-smoke.yaml")
    client = ReasoningCaptureOllamaClient(
        final_answer="Her creatinine was 0.9.",
        claims=[
            Claim(
                text="Her creatinine was 0.9.",
                source_refs=[],
                document_citations=[
                    DocumentCitation(
                        source_type="lab_pdf",
                        source_id="lab-doc-1",
                        page_or_section="page 2",
                        field_or_chunk_id="lab-doc-1#page-2-row-0",
                        quote_or_value="Creatinine: 0.9",
                    )
                ],
            )
        ],
    )

    result = run_case(case, client)

    assert result.rendered is not None, "case has a verdict assertion, so verification must have run"
    assert result.verdict_result is not None
    assert result.verdict_result.verdict == Verdict.VERIFIED, (
        "issue #70: a claim citing the fixture's own (source_id, field_or_chunk_id) key "
        "must verify -- proving run_verification's DocumentFactIndex was actually built "
        "from case.patient_facts, not silently skipped"
    )

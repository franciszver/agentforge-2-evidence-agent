"""Structural tests for the issue #170 SourceRef-relevance gate
(``app.source_ref_relevance``): a mocked judge double drives the fail-closed
contract and the scope/established-facts rules without any real LLM call.
Mirrors ``tests/test_semantic_support.py``'s style for the sibling gate this
module is adapted from.

This module ships flag-OFF (``Settings.copilot_source_ref_relevance_enabled``
defaults ``False``) pending the issue #170 live re-measurement
(``evals/runner/issue_170_source_ref_relevance_spike.py``) against #130's
pre-registered kill criterion -- these tests exercise the MECHANISM in
isolation; they do not by themselves establish it is safe to enable.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.ollama_client import LLMEngineError
from app.schemas.common import SourceRef
from app.schemas.ingestion import DocumentCitation
from app.schemas.verification import Claim
from app.semantic_support import SemanticSupportJudgement, SupportVerdict
from app.source_ref_relevance import (
    _established_facts_for_source_ref_claim,
    _is_source_ref_only_claim,
    _source_ref_facts,
    apply_source_ref_relevance,
    judge_source_ref_relevance,
)
from app.verification import CitationCheckResult, CitationStatus, ClaimCheckResult, DocumentCitationCheckResult

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _ScriptedJudge:
    """A test double for ``SemanticSupportJudgeLike``: returns one scripted
    ``SemanticSupportJudgement`` per call, in order, or raises a scripted
    ``LLMEngineError``. Records every rendered prompt it was asked about."""

    def __init__(self, responses: list[SemanticSupportJudgement | Exception]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def extract(self, prompt_or_messages: object, schema: type[BaseModel], *, options: object = None) -> BaseModel:
        assert schema is SemanticSupportJudgement
        assert isinstance(prompt_or_messages, list)
        self.prompts.append(prompt_or_messages[-1]["content"])
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _supported(reason: str = "facts are on-topic for the claim") -> SemanticSupportJudgement:
    return SemanticSupportJudgement(verdict=SupportVerdict.SUPPORTED, reason=reason)


def _not_supported(reason: str = "facts are about something else") -> SemanticSupportJudgement:
    return SemanticSupportJudgement(verdict=SupportVerdict.NOT_SUPPORTED, reason=reason)


def _uncertain(reason: str = "ambiguous") -> SemanticSupportJudgement:
    return SemanticSupportJudgement(verdict=SupportVerdict.UNCERTAIN, reason=reason)


def _appointment_status_ref() -> SourceRef:
    return SourceRef(tool_call_id="call_0", record_id="0", field="status", asserted_value="scheduled")


def _guideline_citation() -> DocumentCitation:
    return DocumentCitation(
        source_type="guideline_chunk",
        source_id="hypertension-lifestyle",
        page_or_section="Follow-Up Cadence",
        field_or_chunk_id="hypertension-lifestyle#follow-up-cadence",
        quote_or_value="Recheck blood pressure in 3-6 months.",
    )


def _source_ref_only_claim_result(
    *refs: SourceRef, text: str = "The patient's blood pressure was elevated."
) -> ClaimCheckResult:
    claim = Claim(text=text, source_refs=list(refs))
    return ClaimCheckResult(
        claim=claim,
        citation_results=[CitationCheckResult(source_ref=ref, status=CitationStatus.VALID) for ref in refs],
    )


# ---------------------------------------------------------------------------
# judge_source_ref_relevance
# ---------------------------------------------------------------------------


def test_judge_true_only_on_supported_verdict():
    judge = _ScriptedJudge([_supported()])

    assert judge_source_ref_relevance("claim text", ["field: value"], judge) is True


@pytest.mark.parametrize("response", [_not_supported(), _uncertain()])
def test_judge_false_on_not_supported_or_uncertain(response):
    judge = _ScriptedJudge([response])

    assert judge_source_ref_relevance("claim text", ["field: value"], judge) is False


def test_judge_fails_closed_on_engine_error():
    judge = _ScriptedJudge([LLMEngineError("boom")])

    assert judge_source_ref_relevance("claim text", ["field: value"], judge) is False


def test_judge_never_raises_engine_error():
    judge = _ScriptedJudge([LLMEngineError("boom")])

    judge_source_ref_relevance("claim text", ["field: value"], judge)


def test_judge_prompt_includes_context_block_only_when_facts_given():
    judge = _ScriptedJudge([_supported(), _supported()])

    judge_source_ref_relevance("claim text", ["field: value"], judge, context_facts=None)
    judge_source_ref_relevance("claim text", ["field: value"], judge, context_facts=["other: fact"])

    assert "ESTABLISHED FACTS (already confirmed" not in judge.prompts[0]
    assert "ESTABLISHED FACTS (already confirmed" in judge.prompts[1]
    assert "other: fact" in judge.prompts[1]


# ---------------------------------------------------------------------------
# _is_source_ref_only_claim / _source_ref_facts -- scope helpers
# ---------------------------------------------------------------------------


def test_source_ref_only_claim_is_eligible():
    claim = Claim(text="x", source_refs=[_appointment_status_ref()])
    assert _is_source_ref_only_claim(claim) is True


def test_claim_with_any_document_citation_is_not_eligible():
    claim = Claim(text="x", source_refs=[_appointment_status_ref()], document_citations=[_guideline_citation()])
    assert _is_source_ref_only_claim(claim) is False


def test_uncited_claim_is_not_eligible():
    claim = Claim(text="x")
    assert _is_source_ref_only_claim(claim) is False


def test_source_ref_facts_formats_field_value_pairs():
    ref = SourceRef(tool_call_id="call_0", record_id="0", field="problem_count", asserted_value="0")
    claim = Claim(text="x", source_refs=[ref])
    assert _source_ref_facts(claim) == ["problem_count: 0"]


def test_source_ref_facts_skips_refs_with_no_asserted_value():
    ref = SourceRef(tool_call_id="call_0", record_id="0", field="status", asserted_value=None)
    claim = Claim(text="x", source_refs=[ref])
    assert _source_ref_facts(claim) == []


# ---------------------------------------------------------------------------
# apply_source_ref_relevance -- scope
# ---------------------------------------------------------------------------


def test_supported_source_ref_only_claim_stays_valid():
    claim_result = _source_ref_only_claim_result(_appointment_status_ref())
    judge = _ScriptedJudge([_supported()])

    (result,) = apply_source_ref_relevance([claim_result], judge)

    assert result.passed is True
    assert result.citation_results[0].status is CitationStatus.VALID


def test_unsupported_source_ref_only_claim_is_downgraded_and_fails():
    claim_result = _source_ref_only_claim_result(_appointment_status_ref())
    judge = _ScriptedJudge([_not_supported()])

    (result,) = apply_source_ref_relevance([claim_result], judge)

    assert result.passed is False
    assert result.citation_results[0].status is CitationStatus.NOT_TOPICALLY_RELEVANT


def test_uncertain_verdict_fails_closed():
    claim_result = _source_ref_only_claim_result(_appointment_status_ref())
    judge = _ScriptedJudge([_uncertain()])

    (result,) = apply_source_ref_relevance([claim_result], judge)

    assert result.passed is False
    assert result.citation_results[0].status is CitationStatus.NOT_TOPICALLY_RELEVANT


def test_judge_error_fails_closed():
    claim_result = _source_ref_only_claim_result(_appointment_status_ref())
    judge = _ScriptedJudge([LLMEngineError("timeout")])

    (result,) = apply_source_ref_relevance([claim_result], judge)

    assert result.passed is False
    assert result.citation_results[0].status is CitationStatus.NOT_TOPICALLY_RELEVANT


def test_all_citations_downgraded_when_claim_has_multiple_source_refs():
    date_ref = SourceRef(tool_call_id="call_0", record_id="0", field="date", asserted_value="2014-01-31")
    time_ref = SourceRef(tool_call_id="call_0", record_id="0", field="time", asserted_value="14:30:00")
    claim_result = _source_ref_only_claim_result(date_ref, time_ref)
    judge = _ScriptedJudge([_not_supported()])

    (result,) = apply_source_ref_relevance([claim_result], judge)

    assert result.passed is False
    assert all(c.status is CitationStatus.NOT_TOPICALLY_RELEVANT for c in result.citation_results)


def test_claim_that_already_failed_provenance_is_untouched_no_judge_call():
    ref = _appointment_status_ref()
    failed = CitationCheckResult(source_ref=ref, status=CitationStatus.VALUE_MISMATCH)
    claim = Claim(text="wrong claim", source_refs=[ref])
    claim_result = ClaimCheckResult(claim=claim, citation_results=[failed])
    judge = _ScriptedJudge([])  # a call would raise IndexError

    (result,) = apply_source_ref_relevance([claim_result], judge)

    assert result is claim_result
    assert judge.prompts == []


def test_claim_with_a_document_citation_is_left_entirely_alone_no_judge_call():
    """Scope invariant (#130's ADR): a claim carrying even a PASSING
    DocumentCitation is out of scope for this gate entirely -- not just the
    DocumentCitation half. ``app.semantic_support`` covers that half; the
    two gates are disjoint."""
    ref = _appointment_status_ref()
    guideline = _guideline_citation()
    claim = Claim(text="mixed claim", source_refs=[ref], document_citations=[guideline])
    claim_result = ClaimCheckResult(
        claim=claim,
        citation_results=[
            CitationCheckResult(source_ref=ref, status=CitationStatus.VALID),
            DocumentCitationCheckResult(document_citation=guideline, status=CitationStatus.VALID),
        ],
    )
    judge = _ScriptedJudge([])  # a call would raise IndexError

    (result,) = apply_source_ref_relevance([claim_result], judge)

    assert result is claim_result
    assert judge.prompts == []


def test_order_preserving_across_multiple_claims():
    claim_a = _source_ref_only_claim_result(_appointment_status_ref(), text="claim A")
    claim_b = _source_ref_only_claim_result(_appointment_status_ref(), text="claim B")
    judge = _ScriptedJudge([_supported(), _not_supported()])

    results = apply_source_ref_relevance([claim_a, claim_b], judge)

    assert results[0].claim.text == "claim A"
    assert results[0].passed is True
    assert results[1].claim.text == "claim B"
    assert results[1].passed is False


# ---------------------------------------------------------------------------
# Established-facts context (#111/#128, adapted) and its self-exclusion
# ---------------------------------------------------------------------------


def test_established_facts_include_sibling_source_ref_only_passed_claim():
    sibling_ref = SourceRef(tool_call_id="call_0", record_id="0", field="provider", asserted_value="Billy Smith")
    sibling_claim = Claim(text="Appointment is with Billy Smith.", source_refs=[sibling_ref])
    sibling_result = ClaimCheckResult(
        claim=sibling_claim,
        citation_results=[CitationCheckResult(source_ref=sibling_ref, status=CitationStatus.VALID)],
    )
    target_result = _source_ref_only_claim_result(_appointment_status_ref())

    facts = _established_facts_for_source_ref_claim(target_result, [target_result, sibling_result])

    assert facts == ["provider: Billy Smith"]


def test_established_facts_never_include_the_claim_own_facts_self_exclusion():
    """The one required difference from ``app.semantic_support``'s
    established-facts gathering: the current claim's OWN facts must never
    appear in its own context block -- they are already the primary SOURCE
    FACTS being judged, and re-including them would be circular self-
    corroboration, not independent context."""
    target_result = _source_ref_only_claim_result(_appointment_status_ref())

    facts = _established_facts_for_source_ref_claim(target_result, [target_result])

    assert facts == []


def test_established_facts_exclude_sibling_with_document_citation():
    guideline = _guideline_citation()
    sibling_ref = SourceRef(tool_call_id="call_0", record_id="0", field="provider", asserted_value="Billy Smith")
    sibling_claim = Claim(text="mixed sibling", source_refs=[sibling_ref], document_citations=[guideline])
    sibling_result = ClaimCheckResult(
        claim=sibling_claim,
        citation_results=[
            CitationCheckResult(source_ref=sibling_ref, status=CitationStatus.VALID),
            DocumentCitationCheckResult(document_citation=guideline, status=CitationStatus.VALID),
        ],
    )
    target_result = _source_ref_only_claim_result(_appointment_status_ref())

    facts = _established_facts_for_source_ref_claim(target_result, [target_result, sibling_result])

    assert facts == []


def test_established_facts_exclude_sibling_that_has_not_passed():
    sibling_ref = SourceRef(tool_call_id="call_0", record_id="0", field="provider", asserted_value="Billy Smith")
    sibling_claim = Claim(text="failed sibling", source_refs=[sibling_ref])
    sibling_result = ClaimCheckResult(
        claim=sibling_claim,
        citation_results=[CitationCheckResult(source_ref=sibling_ref, status=CitationStatus.VALUE_MISMATCH)],
    )
    target_result = _source_ref_only_claim_result(_appointment_status_ref())

    facts = _established_facts_for_source_ref_claim(target_result, [target_result, sibling_result])

    assert facts == []


def test_apply_passes_sibling_facts_into_judge_prompt_but_not_own_facts_twice():
    sibling_ref = SourceRef(tool_call_id="call_0", record_id="0", field="provider", asserted_value="Billy Smith")
    sibling_claim = Claim(text="Appointment is with Billy Smith.", source_refs=[sibling_ref])
    sibling_result = ClaimCheckResult(
        claim=sibling_claim,
        citation_results=[CitationCheckResult(source_ref=sibling_ref, status=CitationStatus.VALID)],
    )
    target_result = _source_ref_only_claim_result(_appointment_status_ref())
    judge = _ScriptedJudge([_supported(), _supported()])

    apply_source_ref_relevance([sibling_result, target_result], judge)

    assert len(judge.prompts) == 2
    target_prompt = judge.prompts[1]
    assert "Billy Smith" in target_prompt
    # The claim's own fact ("status: scheduled") appears exactly once, in the
    # SOURCE FACTS section -- never duplicated into ESTABLISHED FACTS.
    assert target_prompt.count("status: scheduled") == 1

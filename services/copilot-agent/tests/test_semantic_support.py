"""Structural tests for the issue #47 semantic-support gate
(``app.semantic_support``): a mocked judge double drives the fail-closed
contract (supported passes, unsupported/uncertain/errored fails) without any
real LLM call. Hermetic and fully deterministic -- mirrors
``tests/test_verification_documents.py``'s style for the underlying checker
this module extends.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.ollama_client import LLMEngineError
from app.schemas.common import SourceRef
from app.schemas.ingestion import DocumentCitation
from app.schemas.verification import Claim
from app.semantic_support import (
    SemanticSupportJudgement,
    SupportVerdict,
    apply_semantic_support,
    judge_support,
)
from app.verification import (
    CitationCheckResult,
    CitationStatus,
    ClaimCheckResult,
    DocumentCitationCheckResult,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _ScriptedJudge:
    """A test double for ``SemanticSupportJudgeLike``: returns one scripted
    ``SemanticSupportJudgement`` per call, in order, or raises a scripted
    ``LLMEngineError``. Records every (claim, quote) pair it was asked about."""

    def __init__(self, responses: list[SemanticSupportJudgement | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def extract(self, prompt_or_messages: object, schema: type[BaseModel], *, options: object = None) -> BaseModel:
        assert schema is SemanticSupportJudgement
        # Record the (claim, quote) the caller embedded in the user message,
        # mirroring how a real judge would receive it.
        assert isinstance(prompt_or_messages, list)
        user_content = prompt_or_messages[-1]["content"]
        self.calls.append((user_content, user_content))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _supported(reason: str = "quote directly states the claim") -> SemanticSupportJudgement:
    return SemanticSupportJudgement(verdict=SupportVerdict.SUPPORTED, reason=reason)


def _not_supported(reason: str = "quote is about something else") -> SemanticSupportJudgement:
    return SemanticSupportJudgement(verdict=SupportVerdict.NOT_SUPPORTED, reason=reason)


def _uncertain(reason: str = "ambiguous") -> SemanticSupportJudgement:
    return SemanticSupportJudgement(verdict=SupportVerdict.UNCERTAIN, reason=reason)


def _guideline_citation(quote: str = "A1c target generally below 7%.") -> DocumentCitation:
    return DocumentCitation(
        source_type="guideline_chunk",
        source_id="a1c-targets",
        page_or_section="Target Ranges",
        field_or_chunk_id="a1c-targets#target-ranges",
        quote_or_value=quote,
    )


def _valid_doc_result(quote: str = "A1c target generally below 7%.") -> DocumentCitationCheckResult:
    return DocumentCitationCheckResult(document_citation=_guideline_citation(quote), status=CitationStatus.VALID)


def _valid_source_ref_result() -> CitationCheckResult:
    ref = SourceRef(tool_call_id="call_0", record_id="0", field="value", asserted_value="7.8")
    return CitationCheckResult(source_ref=ref, status=CitationStatus.VALID)


def _claim_result(*citation_results, text: str = "The A1c is below target.") -> ClaimCheckResult:
    claim = Claim(text=text, document_citations=[_guideline_citation()])
    return ClaimCheckResult(claim=claim, citation_results=list(citation_results))


# ---------------------------------------------------------------------------
# judge_support
# ---------------------------------------------------------------------------


def test_judge_support_true_only_on_supported_verdict():
    judge = _ScriptedJudge([_supported()])

    assert judge_support("claim text", "quote text", judge) is True


@pytest.mark.parametrize("response", [_not_supported(), _uncertain()])
def test_judge_support_false_on_not_supported_or_uncertain(response):
    judge = _ScriptedJudge([response])

    assert judge_support("claim text", "quote text", judge) is False


def test_judge_support_fails_closed_on_engine_error():
    judge = _ScriptedJudge([LLMEngineError("boom")])

    assert judge_support("claim text", "quote text", judge) is False


def test_judge_support_never_raises_engine_error():
    judge = _ScriptedJudge([LLMEngineError("boom")])

    # Must not propagate -- a flaky judge call degrades to "not verified",
    # never crashes an otherwise-working turn.
    judge_support("claim text", "quote text", judge)


# ---------------------------------------------------------------------------
# apply_semantic_support
# ---------------------------------------------------------------------------


def test_supported_document_citation_stays_valid():
    claim_result = _claim_result(_valid_doc_result())
    judge = _ScriptedJudge([_supported()])

    (result,) = apply_semantic_support([claim_result], judge)

    assert result.passed is True
    assert result.citation_results[0].status is CitationStatus.VALID


def test_unsupported_document_citation_is_downgraded_and_claim_fails():
    claim_result = _claim_result(_valid_doc_result())
    judge = _ScriptedJudge([_not_supported()])

    (result,) = apply_semantic_support([claim_result], judge)

    assert result.passed is False
    assert result.citation_results[0].status is CitationStatus.NOT_SEMANTICALLY_SUPPORTED


def test_uncertain_verdict_fails_closed():
    claim_result = _claim_result(_valid_doc_result())
    judge = _ScriptedJudge([_uncertain()])

    (result,) = apply_semantic_support([claim_result], judge)

    assert result.passed is False
    assert result.citation_results[0].status is CitationStatus.NOT_SEMANTICALLY_SUPPORTED


def test_judge_error_fails_closed():
    claim_result = _claim_result(_valid_doc_result())
    judge = _ScriptedJudge([LLMEngineError("timeout")])

    (result,) = apply_semantic_support([claim_result], judge)

    assert result.passed is False
    assert result.citation_results[0].status is CitationStatus.NOT_SEMANTICALLY_SUPPORTED


def test_claim_that_already_failed_provenance_is_untouched_no_judge_call():
    ref = SourceRef(tool_call_id="call_0", record_id="0", field="value", asserted_value="7.8")
    failed_source_ref = CitationCheckResult(source_ref=ref, status=CitationStatus.VALUE_MISMATCH)
    claim = Claim(text="wrong claim", source_refs=[ref])
    claim_result = ClaimCheckResult(claim=claim, citation_results=[failed_source_ref])
    judge = _ScriptedJudge([])  # no responses scripted -- a call would raise IndexError

    (result,) = apply_semantic_support([claim_result], judge)

    assert result is claim_result  # passed through unchanged, byte-identical object
    assert judge.calls == []


def test_source_ref_citations_are_never_judged():
    claim_result = _claim_result(_valid_source_ref_result())
    judge = _ScriptedJudge([])  # no responses scripted -- a call would raise IndexError

    (result,) = apply_semantic_support([claim_result], judge)

    assert result.passed is True
    assert result.citation_results[0].status is CitationStatus.VALID
    assert judge.calls == []


def test_mixed_citations_only_document_citation_is_judged():
    claim_result = _claim_result(_valid_source_ref_result(), _valid_doc_result())
    judge = _ScriptedJudge([_not_supported()])

    (result,) = apply_semantic_support([claim_result], judge)

    assert len(judge.calls) == 1
    assert result.passed is False
    assert result.citation_results[0].status is CitationStatus.VALID  # SourceRef untouched
    assert result.citation_results[1].status is CitationStatus.NOT_SEMANTICALLY_SUPPORTED


def test_already_failed_document_citation_is_not_rejudged():
    failed_doc = DocumentCitationCheckResult(document_citation=_guideline_citation(), status=CitationStatus.QUOTE_NOT_FOUND)
    claim = Claim(text="claim", document_citations=[_guideline_citation()])
    claim_result = ClaimCheckResult(claim=claim, citation_results=[failed_doc])
    judge = _ScriptedJudge([])  # a call would raise IndexError

    (result,) = apply_semantic_support([claim_result], judge)

    assert result is claim_result
    assert judge.calls == []


def test_order_preserving_across_multiple_claims():
    claim_a = _claim_result(_valid_doc_result("quote A"), text="claim A")
    claim_b = _claim_result(_valid_doc_result("quote B"), text="claim B")
    judge = _ScriptedJudge([_supported(), _not_supported()])

    results = apply_semantic_support([claim_a, claim_b], judge)

    assert results[0].claim.text == "claim A"
    assert results[0].passed is True
    assert results[1].claim.text == "claim B"
    assert results[1].passed is False


# ---------------------------------------------------------------------------
# issue #108: duplicate claims citing IDENTICAL evidence must get one
# consistent verdict, not two independently-judged (and potentially
# disagreeing) ones.
# ---------------------------------------------------------------------------


def test_duplicate_claims_citing_identical_evidence_get_one_consistent_verdict():
    """The model sometimes restates one guideline-backed fact as two
    separate claims (e.g. "...above the target range..." / "...is not at
    target.") that both cite the exact same DocumentCitation -- identical
    ``source_type``/``source_id``/``field_or_chunk_id``/``quote_or_value``,
    byte-for-byte. Judging each claim's citation independently risks an
    inconsistent verdict across the two purely from LLM call-to-call
    variance on the paraphrased wording -- observed live as one restatement
    passing while the other is downgraded to ``not_semantically_supported``,
    which strips that claim and holds the whole answer at
    ``partially_verified`` even though both claims cite the identical,
    already-provenance-verified evidence. The fix judges identical evidence
    ONCE and applies that single verdict everywhere it is cited, so
    duplicates can never land on different verdicts."""
    same_quote = "A1c target generally below 7%."
    claim_a = _claim_result(_valid_doc_result(same_quote), text="A1c is above the target range.")
    claim_b = _claim_result(_valid_doc_result(same_quote), text="A1c is not at target.")
    # Only ONE response scripted: a second independent call would raise
    # IndexError, proving the fix collapses the two into a single judge call.
    judge = _ScriptedJudge([_supported()])

    results = apply_semantic_support([claim_a, claim_b], judge)

    assert len(judge.calls) == 1
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[0].citation_results[0].status is CitationStatus.VALID
    assert results[1].citation_results[0].status is CitationStatus.VALID


def test_duplicate_claims_citing_identical_evidence_both_fail_together():
    """The consistency guarantee cuts both ways: if the single shared
    judgement comes back unsupported, EVERY claim citing that identical
    evidence is downgraded together -- never a mix of one passing, one
    failing, for the same byte-identical citation."""
    same_quote = "A1c target generally below 7%."
    claim_a = _claim_result(_valid_doc_result(same_quote), text="A1c is above the target range.")
    claim_b = _claim_result(_valid_doc_result(same_quote), text="A1c is not at target.")
    judge = _ScriptedJudge([_not_supported()])

    results = apply_semantic_support([claim_a, claim_b], judge)

    assert len(judge.calls) == 1
    assert results[0].passed is False
    assert results[1].passed is False
    assert results[0].citation_results[0].status is CitationStatus.NOT_SEMANTICALLY_SUPPORTED
    assert results[1].citation_results[0].status is CitationStatus.NOT_SEMANTICALLY_SUPPORTED


def test_distinct_citations_are_not_conflated_by_dedup():
    """The dedup key is the citation's full identity (source_type,
    source_id, field_or_chunk_id, quote_or_value) -- two claims citing
    DIFFERENT quotes (even from the same source document) are judged
    independently, exactly as before. Guards against an over-broad dedup
    (e.g. keying on source_id alone) silently merging unrelated citations."""
    claim_a = _claim_result(_valid_doc_result("quote A"), text="claim A")
    claim_b = _claim_result(_valid_doc_result("quote B"), text="claim B")
    judge = _ScriptedJudge([_supported(), _not_supported()])

    results = apply_semantic_support([claim_a, claim_b], judge)

    assert len(judge.calls) == 2
    assert results[0].passed is True
    assert results[1].passed is False

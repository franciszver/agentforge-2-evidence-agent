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
    judge_support_full,
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


class _MessageCapturingJudge:
    """Records the EXACT ``messages`` list handed to ``.extract`` (not just
    the user-content string like ``_ScriptedJudge`` does), so a test can
    assert two call sites built byte-identical messages -- the regression
    guard for issue #192 gate-1 finding 1 (see the test below)."""

    def __init__(self, response: SemanticSupportJudgement) -> None:
        self._response = response
        self.messages_seen: list[list[dict[str, str]]] = []

    def extract(self, prompt_or_messages: object, schema: type[BaseModel], *, options: object = None) -> BaseModel:
        assert schema is SemanticSupportJudgement
        assert isinstance(prompt_or_messages, list)
        self.messages_seen.append(prompt_or_messages)
        return self._response


def test_judge_support_full_builds_byte_identical_messages_to_judge_support():
    """Regression guard for issue #192 gate-1 finding 1: the live injection
    battery (``evals/runner/issue_192_injection_battery.py``) used to
    reconstruct ``judge_support``'s message shape by importing this module's
    private ``_SYSTEM_PROMPT``/``_INSTRUCTIONS_TEMPLATE`` and re-assembling
    ``messages`` itself -- so if this module's assembly ever changed shape,
    the battery would silently keep attacking the OLD shape. The fix: the
    battery now calls ``judge_support_full`` directly (the same public seam
    ``judge_support`` itself delegates to), so there is only ONE message-
    assembly code path left. This test proves that path is exercised
    identically regardless of entry point -- ``judge_support`` (bool) and
    ``judge_support_full`` (full judgement) must send byte-for-byte the same
    ``messages`` for the same inputs, including ``context_facts``."""
    claim_text = "The patient's LDL cholesterol was 165 mg/dL, above the target range."
    quote = "Lipid panel results: LDL cholesterol 165 mg/dL. Target LDL below 100 mg/dL."
    context_facts = ["a1c: 6.8%"]

    bool_judge = _MessageCapturingJudge(_supported())
    judge_support(claim_text, quote, bool_judge, context_facts)

    full_judge = _MessageCapturingJudge(_supported())
    judge_support_full(claim_text, quote, full_judge, context_facts)

    assert bool_judge.messages_seen == full_judge.messages_seen
    assert len(bool_judge.messages_seen[0]) == 2  # system + user, same shape both entry points


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





# ---------------------------------------------------------------------------
# issues #111/#128: the judge sees only a single, narrow (claim, quote) pair
# and has no visibility into a claim's OTHER citations, or into sibling
# claims from the same answer that already establish facts the current
# claim's assertion depends on. Both shapes reproduced below with a scripted
# judge that inspects the actual prompt content it was given.
# ---------------------------------------------------------------------------


def _bp_source_refs() -> list[SourceRef]:
    return [
        SourceRef(tool_call_id="call_0", record_id="0", field="blood_pressure_systolic", asserted_value="148"),
        SourceRef(tool_call_id="call_0", record_id="1", field="blood_pressure_diastolic", asserted_value="94"),
    ]


class _InspectingJudge:
    """A test double that records the full rendered prompt text (not just the
    claim/quote) for each call, and returns one scripted verdict per call."""

    def __init__(self, responses: list[SemanticSupportJudgement]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def extract(self, prompt_or_messages: object, schema: type[BaseModel], *, options: object = None) -> BaseModel:
        assert schema is SemanticSupportJudgement
        assert isinstance(prompt_or_messages, list)
        self.prompts.append(prompt_or_messages[-1]["content"])
        return self._responses.pop(0)


def test_issue_111_sibling_citation_on_same_claim_is_surfaced_as_context():
    """#111 shape: ONE claim, TWO citations -- a chart-data SourceRef (148/94
    mmHg, already deterministically re-validated) and a guideline
    DocumentCitation categorizing that reading. The guideline quote alone
    never restates the patient's specific value -- that value lives on the
    claim's OTHER (SourceRef) citation. The judge call for the guideline
    citation must be given that sibling value as established context."""
    guideline = _guideline_citation("Stage 2 hypertension: systolic 140 mmHg or higher OR diastolic 90 mmHg or higher.")
    claim = Claim(
        text="His blood pressure falls into the category of Stage 2 hypertension.",
        source_refs=_bp_source_refs(),
        document_citations=[guideline],
    )
    claim_result = ClaimCheckResult(
        claim=claim,
        citation_results=[
            CitationCheckResult(source_ref=claim.source_refs[0], status=CitationStatus.VALID),
            CitationCheckResult(source_ref=claim.source_refs[1], status=CitationStatus.VALID),
            DocumentCitationCheckResult(document_citation=guideline, status=CitationStatus.VALID),
        ],
    )
    judge = _InspectingJudge([_supported()])

    apply_semantic_support([claim_result], judge)

    assert len(judge.prompts) == 1
    prompt = judge.prompts[0]
    assert "148" in prompt
    assert "94" in prompt


def test_issue_128_sibling_claims_value_is_surfaced_as_context():
    """#128 shape: TWO separate claims -- one citing a chart value (SourceRef,
    "172 mg/dL"), another citing a guideline category chunk that only makes
    sense combined with that value. The second claim's judge call must be
    given the first (already-verified, SourceRef-only) claim's value as
    established context."""
    value_ref = SourceRef(tool_call_id="call_0", record_id="0", field="value", asserted_value="172")
    value_claim = Claim(text="The patient's LDL cholesterol level is 172 mg/dL.", source_refs=[value_ref])
    value_result = ClaimCheckResult(
        claim=value_claim,
        citation_results=[CitationCheckResult(source_ref=value_ref, status=CitationStatus.VALID)],
    )
    guideline = _guideline_citation(
        "LDL cholesterol: optimal below 100 mg/dL; near-optimal 100-129 mg/dL; "
        "borderline-high 130-159 mg/dL; high 160-189 mg/dL; very high 190 mg/dL or above."
    )
    category_claim = Claim(text="The patient's LDL cholesterol level is considered high.", document_citations=[guideline])
    category_result = ClaimCheckResult(
        claim=category_claim,
        citation_results=[DocumentCitationCheckResult(document_citation=guideline, status=CitationStatus.VALID)],
    )
    judge = _InspectingJudge([_supported()])

    apply_semantic_support([value_result, category_result], judge)

    assert len(judge.prompts) == 1
    assert "172" in judge.prompts[0]


def test_sibling_fact_excluded_when_sibling_claim_has_not_passed():
    """Safety invariant: a sibling claim's value is only ever surfaced as
    established context when that sibling has ALREADY fully passed
    (deterministic re-validation, no outstanding LLM judgment). A sibling
    that failed provenance must never leak its (unverified) asserted value
    into another claim's judge context."""
    bad_ref = SourceRef(tool_call_id="call_0", record_id="0", field="value", asserted_value="999")
    bad_claim = Claim(text="The patient's LDL cholesterol level is 999 mg/dL.", source_refs=[bad_ref])
    bad_result = ClaimCheckResult(
        claim=bad_claim,
        citation_results=[CitationCheckResult(source_ref=bad_ref, status=CitationStatus.VALUE_MISMATCH)],
    )
    guideline = _guideline_citation("LDL cholesterol: high 160-189 mg/dL.")
    category_claim = Claim(text="The patient's LDL cholesterol level is considered high.", document_citations=[guideline])
    category_result = ClaimCheckResult(
        claim=category_claim,
        citation_results=[DocumentCitationCheckResult(document_citation=guideline, status=CitationStatus.VALID)],
    )
    judge = _InspectingJudge([_supported()])

    apply_semantic_support([bad_result, category_result], judge)

    assert len(judge.prompts) == 1
    assert "999" not in judge.prompts[0]


def test_sibling_fact_excluded_when_sibling_claim_still_carries_document_citations():
    """Safety invariant: a sibling claim whose OWN citations include a
    DocumentCitation (i.e. its fact is itself only established via an LLM
    judgment, not purely deterministic re-validation) must never be used as
    ground-truth context for another claim -- only claims backed purely by
    already-passed SourceRefs count as established facts, avoiding any
    circularity risk."""
    mixed_ref = SourceRef(tool_call_id="call_0", record_id="0", field="value", asserted_value="172")
    mixed_guideline = _guideline_citation("some unrelated guideline text")
    mixed_claim = Claim(
        text="The patient's LDL cholesterol level is 172 mg/dL, per guideline X.",
        source_refs=[mixed_ref],
        document_citations=[mixed_guideline],
    )
    mixed_result = ClaimCheckResult(
        claim=mixed_claim,
        citation_results=[
            CitationCheckResult(source_ref=mixed_ref, status=CitationStatus.VALID),
            DocumentCitationCheckResult(document_citation=mixed_guideline, status=CitationStatus.VALID),
        ],
    )
    guideline = _guideline_citation("LDL cholesterol: high 160-189 mg/dL.")
    category_claim = Claim(text="The patient's LDL cholesterol level is considered high.", document_citations=[guideline])
    category_result = ClaimCheckResult(
        claim=category_claim,
        citation_results=[DocumentCitationCheckResult(document_citation=guideline, status=CitationStatus.VALID)],
    )
    judge = _InspectingJudge([_supported(), _supported()])

    apply_semantic_support([mixed_result, category_result], judge)

    assert len(judge.prompts) == 2
    category_prompt = judge.prompts[1]
    assert "172" not in category_prompt


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

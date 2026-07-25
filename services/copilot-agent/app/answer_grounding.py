"""Deterministic claim-in-answer grounding gate (issue #153).

**The gap this closes.** ``app.verification.check_source_ref``/``check_claim``
re-validate that a citation's ``(tool_call_id, record_id, field,
asserted_value)`` matches the RAW tool record -- provenance only. Nothing in
that pipeline checks that the resulting ``Claim.text`` corresponds to a
proposition the planner's ``answer`` actually asserts. ``app.extraction``
hands the extractor EVERY field of every tool result (``_build_catalog``),
so a claim citing a real, correctly-valued field the answer never discussed
(e.g. a hallucinated ``respiratory_rate`` claim on an answer that only
discusses weight) still passes provenance re-validation and, with nothing
else wrong, yields a ``VERIFIED`` verdict on an answer that never said that.

**Why not LLM-judge this, mirroring ``app.semantic_support``?** That module
already re-checks a DIFFERENT gap (a real quote paired with an unrelated
claim) via an LLM judge, additively, flag-gated. This gate is deliberately
NOT that: the owner asked for a DETERMINISTIC check here (no LLM call, no
added latency/flakiness on every turn), and the failure mode is different in
kind -- a #47-style semantic-support judge asks "does this EVIDENCE support
this claim", which needs a judgment call; this gate asks "did the answer say
this AT ALL", which is answerable by a much cruder, purely lexical signal:
the claim's own significant words should be substantially present in the
answer's words, because the extractor's whole job is to decompose the
answer's OWN prose into claims -- a claim's text is either drawn from the
answer or it is hallucinated.

**The rule.** Tokenize both ``claim.text`` and ``answer`` into lowercase
alphanumeric words, drop a short, domain-agnostic stopword list plus
single-character tokens, and require that at least ``min_overlap_ratio``
(default 0.5) of the claim's remaining ("significant") tokens also appear in
the answer's token set. This is NOT substring/field-name matching (the
issue's own guidance: clinical answers paraphrase -- "blood pressure" for a
``blood_pressure_systolic`` field -- so matching the FIELD NAME against the
answer would be too brittle). Comparing the claim's own words (which the
extractor is supposed to have drawn FROM the answer) against the answer's
words sidesteps that: a paraphrased claim still shares most of its
significant vocabulary with the sentence it was extracted from, while a
hallucinated claim about a topic the answer never raised shares essentially
none.

**Known failure modes (deliberately accepted, not silently ignored):**

  1. **A short, hallucinated claim can slip through.** With few significant
     tokens, a small hallucinated claim that happens to share a couple of
     common clinical words with the answer (e.g. both mention "mg") could
     clear the 0.5 ratio. The gate trades this residual risk for staying
     purely lexical and cheap; a stricter ratio or a minimum absolute
     overlap count would reduce it at the cost of rejecting more legitimate
     short claims -- a tuning knob left to the eval measurement (issue #153).
  2. **Heavy paraphrase with near-zero shared vocabulary can be rejected.**
     If the extractor restates a claim in wording that shares almost no
     words with the answer (e.g. answer says "BP" and the claim spells out
     "blood pressure" with no other shared token), a legitimate claim can be
     scored ungrounded. This is the flip side of #1 -- a purely lexical
     check cannot tell true paraphrase from true hallucination when the
     vocabulary itself is nearly disjoint; only a semantic (LLM) check could
     close that gap, which is explicitly out of scope here (see above).
  3. **A claim with no significant tokens at all** (e.g. after stripping
     stopwords nothing remains) is treated as ungrounded rather than
     vacuously grounded -- fail-closed, consistent with this trust layer's
     posture elsewhere (``check_source_ref``'s ``NO_ASSERTED_VALUE``,
     ``ClaimCheckResult.passed``'s empty-citation-list case). This should
     never happen for a real extracted claim (every ``Claim.text`` requires
     ``min_length=1`` and is meant to state a fact), but a claim built
     entirely of stopwords/punctuation would otherwise vacuously pass.

**Where a rejected claim ends up.** Mirrors ``app.semantic_support``'s
established shape exactly, rather than inventing a new one: a claim that
fails this gate has EVERY one of its citation results (``SourceRef`` and
``DocumentCitation`` alike) downgraded to the new
``CitationStatus.NOT_GROUNDED_IN_ANSWER``. ``ClaimCheckResult.passed``'s
existing AND-across-citations aggregation (untouched, in
``app.verification``) then automatically fails the whole claim -- no new
claim-level logic, just new CITATION-level results fed into that same
aggregation. Downstream, this means the claim is NOT silently dropped: it
flows through ``app.rendering.render_answer`` exactly like any other failed
claim (stripped from the prose, an explanatory ``Notice`` in its place) and
counts toward ``VerdictResult.stripped_claim_count``/``compute_verdict``'s
citation-state fold (any stripped claim moves the citation axis off
``ALL_VERIFIED``), same as a provenance failure -- so the UI can show
*something* went wrong with that claim rather than have it vanish with no
trace, and a fail-closed verdict (``BLOCKED``, never ``VERIFIED``) follows
the same established path a ``VALUE_MISMATCH`` claim already takes today.

**Scope: only re-checks currently-passing claims.** A claim that already
failed provenance re-validation is passed through completely unchanged --
nothing to re-check, and re-checking it would only obscure why it already
failed (same discipline as ``app.semantic_support.apply_semantic_support``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.verification import (
    AnyCitationCheckResult,
    CitationCheckResult,
    CitationStatus,
    ClaimCheckResult,
    DocumentCitationCheckResult,
)

# Small, domain-agnostic stopword list -- function words that carry no
# claim-identifying content and would otherwise pad the overlap ratio (e.g.
# "is"/"the"/"her" appear in nearly every claim AND every answer regardless
# of topic, which would make an off-topic claim look grounded).
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "and", "or", "but", "if",
        "of", "in", "on", "at", "to", "for", "with", "as", "by", "from",
        "her", "his", "he", "she", "it", "its", "they", "them", "their",
        "that", "this", "these", "those", "not", "no", "which", "who",
        "there", "than", "then", "so", "also", "about", "into", "over",
        "currently", "current",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _significant_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with stopwords and single-character
    tokens removed -- the "content words" of a piece of text."""
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 1 and token not in _STOPWORDS}


def claim_is_grounded_in_answer(claim_text: str, answer: str, *, min_overlap_ratio: float = 0.5) -> bool:
    """Whether ``claim_text`` is lexically grounded in ``answer`` -- see
    module docstring for the rule and its known failure modes.

    Fail-closed: a claim with zero significant tokens (nothing left after
    stopword removal) is never considered grounded, regardless of
    ``answer``."""
    claim_tokens = _significant_tokens(claim_text)
    if not claim_tokens:
        return False
    answer_tokens = _significant_tokens(answer)
    overlap = len(claim_tokens & answer_tokens)
    return (overlap / len(claim_tokens)) >= min_overlap_ratio


def _downgrade_to_ungrounded(citation_result: AnyCitationCheckResult) -> AnyCitationCheckResult:
    """Downgrade one citation result (of either shape) to
    ``NOT_GROUNDED_IN_ANSWER``, preserving the underlying ``SourceRef``/
    ``DocumentCitation`` object so callers/UI can still show what was cited."""
    if isinstance(citation_result, CitationCheckResult):
        return CitationCheckResult(
            source_ref=citation_result.source_ref, status=CitationStatus.NOT_GROUNDED_IN_ANSWER
        )
    return DocumentCitationCheckResult(
        document_citation=citation_result.document_citation, status=CitationStatus.NOT_GROUNDED_IN_ANSWER
    )


def apply_answer_grounding(claim_results: Sequence[ClaimCheckResult], answer: str) -> list[ClaimCheckResult]:
    """Re-check every already-passing claim's text against ``answer``,
    returning a new list of ``ClaimCheckResult`` with any ungrounded claim's
    citation results downgraded to ``NOT_GROUNDED_IN_ANSWER`` (see module
    docstring). Order-preserving, one-to-one with ``claim_results``, same
    shape ``app.verification.check_claims`` already guarantees.

    A claim that already failed provenance re-validation (``.passed`` is
    ``False``) is passed through completely unchanged."""
    results: list[ClaimCheckResult] = []
    for claim_result in claim_results:
        if not claim_result.passed:
            results.append(claim_result)
            continue
        if claim_is_grounded_in_answer(claim_result.claim.text, answer):
            results.append(claim_result)
            continue
        downgraded = [_downgrade_to_ungrounded(citation_result) for citation_result in claim_result.citation_results]
        results.append(ClaimCheckResult(claim=claim_result.claim, citation_results=downgraded))
    return results

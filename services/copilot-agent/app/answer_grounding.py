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
single-character tokens, and require that at least half (see
``_MIN_OVERLAP_RATIO``) of the claim's remaining ("significant") tokens also
appear in the answer's token set. This is NOT substring/field-name matching
(the issue's own guidance: clinical answers paraphrase -- "blood pressure"
for a ``blood_pressure_systolic`` field -- so matching the FIELD NAME against
the answer would be too brittle). Comparing the claim's own words (which the
extractor is supposed to have drawn FROM the answer) against the answer's
words sidesteps that: a paraphrased claim still shares most of its
significant vocabulary with the sentence it was extracted from, while a
hallucinated claim about a topic the answer never raised shares essentially
none.

**Known failure modes -- NOT fit to enable as shipped (issue #153
adversarial review).** These are not residual edge cases; several are the
COMMON case, and this flag must stay OFF until they are addressed:

  1. **Negation/polarity is not handled at all, and the stopword list is
     why.** ``_STOPWORDS`` includes ``"not"`` and ``"no"`` -- dropped as
     "function words" -- so a claim and its direct negation are
     bag-of-words IDENTICAL and always pass:
     claim "Patient is allergic to penicillin." is judged grounded by
     answer "Patient is not allergic to penicillin." (3/3 significant
     tokens overlap). Same failure for "Blood pressure was elevated." vs
     answer "...was not elevated.", and "She is on Metformin." vs answer
     "She was on Metformin but it was discontinued." The gate cannot tell
     an assertion from its opposite.
  2. **#149's original hallucination scenario is not fixed when the
     hallucinated claim reuses the answer's own vocabulary.** With answer
     "Her blood pressure could not be determined from the available vitals
     records, so I cannot report a systolic or diastolic value for today",
     the hallucinated claims "Her systolic blood pressure value for today
     is 148, from the available vitals records." and "Diastolic 90
     today." both score grounded -- they borrow enough of the answer's own
     words to clear the ratio. Only a hallucinated claim in
     vocabulary-disjoint phrasing is actually caught; a fluent model
     restating the answer's own terms around a fabricated number sails
     through.
  3. **Short claims bypass the ratio easily -- this is the common case,
     not a residual risk.** With N significant tokens only ``ceil(N / 2)``
     must match, so at N=2 a single shared common word is enough:
     "Metformin 500 mg." is judged grounded by an unrelated answer
     "Lisinopril 500 mg was ordered." (shares only "500"/"mg" in spirit --
     literally "mg" plus the numeral). Most extracted claims are short
     clinical sentences, so this triggers constantly, not rarely.
  4. **Wrong-record numeric values pass outright.** The gate only checks
     vocabulary overlap, never the asserted number itself: "Her weight is
     250 lb." is judged grounded by answer "Her weight is 220 lb." -- a
     claim asserting the WRONG value for a field the answer did discuss is
     indistinguishable from a correct paraphrase.
  5. **Routine clinical abbreviation causes false rejections of legitimate
     claims**, pushing them into fail-closed ``BLOCKED``: "Blood glucose is
     110." is scored ungrounded against answer "Her sugar is 110.";
     "Her heart rate is 72 bpm." against "HR 72."; "The patient has type 2
     diabetes mellitus." against "Pt has T2DM." Some of these sit at
     exactly the 0.5 boundary, where a single token's presence or absence
     flips the verdict -- ordinary clinical shorthand, not adversarial
     input, gets blocked.

None of the above is a tuning-knob problem solvable by nudging the ratio: a
stricter ratio makes #5 (false rejections) worse while barely touching #1
(negation, which is bag-of-words-invariant) or #4 (wrong values, which the
ratio never looks at). Before this flag is ever flipped ON, at minimum:
polarity/negation handling, an absolute-overlap floor (not just a ratio) to
close the short-claim bypass, and real per-category eval numbers (see
``app/config.py``'s ``copilot_claim_answer_grounding_enabled`` for the
current state of that measurement) are needed. "Did the answer assert this
proposition?" is a semantic question; a bag-of-words rule is likely the
wrong tool for it in its current form.

  6. **A claim with no significant tokens at all** (e.g. after stripping
     stopwords nothing remains) is treated as ungrounded rather than
     vacuously grounded -- fail-closed, consistent with this trust layer's
     posture elsewhere (``check_source_ref``'s ``NO_ASSERTED_VALUE``,
     ``ClaimCheckResult.passed``'s empty-citation-list case). This should
     never happen for a real extracted claim (every ``Claim.text`` requires
     ``min_length=1`` and is meant to state a fact), but a claim built
     entirely of stopwords/punctuation would otherwise vacuously pass. This
     one IS a narrow edge case, unlike 1-5 above.

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

# The minimum fraction of a claim's significant tokens that must also appear
# in the answer's tokens for the claim to count as grounded. Not exposed as a
# parameter: no production path varies it (only tests, to exercise the
# boundary) -- see module docstring for why 0.5 itself is not a safe value to
# tune casually.
_MIN_OVERLAP_RATIO = 0.5


def significant_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with stopwords and single-character
    tokens removed -- the "content words" of a piece of text.

    Public (not module-private): issue #158's ``app.tool_call_scoping``
    reuses this EXACT normalization for its own, differently-scoped lexical
    overlap check (per-tool-call rather than per-claim) -- shared rather than
    copy-pasted so the two gates can never silently drift apart on what
    counts as a "significant word"."""
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 1 and token not in _STOPWORDS}


def claim_is_grounded_in_answer(claim_text: str, answer: str) -> bool:
    """Whether ``claim_text`` is lexically grounded in ``answer`` -- see
    module docstring for the rule and its known failure modes.

    Fail-closed: a claim with zero significant tokens (nothing left after
    stopword removal) is never considered grounded, regardless of
    ``answer``."""
    claim_tokens = significant_tokens(claim_text)
    if not claim_tokens:
        return False
    answer_tokens = significant_tokens(answer)
    overlap = len(claim_tokens & answer_tokens)
    return (overlap / len(claim_tokens)) >= _MIN_OVERLAP_RATIO


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

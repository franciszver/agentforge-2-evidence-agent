"""Claim stripping + notices (P3.3): the trust layer's user-facing payoff.

Consumes ``app.verification.check_claims``'s output -- one ``ClaimCheckResult``
per claim, already re-validated against the conversation's cached tool
results -- and produces the ordered, user-facing rendered answer. NO model
call, NO I/O, NO clock: purely a per-claim keep/strip decision.

**The rule (docs/IMPLEMENTATION_PLAN.md §4.4).** A claim whose citations
*all* passed (``ClaimCheckResult.passed``) is kept, carrying its passing
citations forward (so P3.8's UI can render tappable citation chips). A claim
that failed -- any citation missing, unresolvable, or mismatched -- is
STRIPPED: its original text is discarded entirely and replaced with a
``Notice``. This is the headline trust-but-verify guarantee: a hallucinated
or miscited claim never reaches the user dressed as fact, even partially --
``ClaimCheckResult.passed`` is already AND-across-citations (P3.2), so one
bad citation in an otherwise-correct claim strips the whole claim, not just
the bad part (no partial-credit rendering, which would risk splicing
verified and unverified text back together).

**Output shape.** ``RenderedAnswer.segments`` is an ordered list of
``RenderedClaim | Notice``, one segment per input ``ClaimCheckResult``, in
the same order -- claims are already order-preserving through
``check_claims``, so ordered rendering is the natural, lossless way to
reinsert notices in place of stripped claims (interleaved with surviving
claims, not bucketed separately). This is deliberately a NEW type, not a
reuse of ``app.schemas.verification.VerifiedAnswer``: ``VerifiedAnswer.claims``
is a flat ``list[Claim]`` where every entry must carry >=1 ``SourceRef``
(P3.1's schema-level guarantee) -- there is no way to represent a stripped
claim's notice, or its position relative to the surviving claims, inside
that contract without either violating it (a notice is not a citable
``Claim``) or losing the interleaving (P3.1's own review flagged
``VerifiedAnswer`` as a lossy projection for exactly this reason).
``RenderedAnswer`` is left as plain frozen dataclasses -- matching
``app.verification.ClaimCheckResult`` -- rather than ``pydantic`` models,
because this is an in-process computation result consumed by Python callers
(P3.7's verdict, P3.8's renderer), not a wire schema; nothing here crosses
an HTTP boundary today.

**Notice text.** A single, constant "Not found in record." -- no
differentiation by ``CitationStatus``. The richer statuses (``VALUE_MISMATCH``
vs ``UNKNOWN_RECORD`` vs ...) are available on the input if a future need
for a more specific message arises, but a per-reason notice risks leaking
information about *why* a claim failed (e.g. "value mismatch" implicitly
confirms the field exists and has some other value) for no benefit to the
user, who just needs to know the claim didn't survive verification. Kept
deliberately dumb.

**Seam to P3.7.** P3.7 computes the whole-answer verdict
(``verified`` / ``partially_verified`` / ``blocked``). It can be computed
directly from the ``list[ClaimCheckResult]`` already passed to
``render_answer`` (e.g. ``all(r.passed for r in results)`` /
``any(r.passed for r in results)``), or by tallying ``RenderedAnswer.segments``
via ``isinstance(segment, Notice)`` -- both are equivalent since
``render_answer`` makes exactly one strip/keep decision per result, with no
additional logic. This module does not compute a verdict itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.common import SourceRef
from app.verification import CitationCheckResult, ClaimCheckResult, DocumentCitationCheckResult

_NOT_FOUND_NOTICE_TEXT = "Not found in record."


@dataclass(frozen=True)
class RenderedClaim:
    """A claim whose citations all passed re-validation -- kept verbatim,
    carrying the passing ``SourceRef`` citations forward for P3.8's citation
    chips.

    ``document_citation_count``/``all_citations_verified`` (P3.6 follow-up)
    exist so a claim verified ONLY by ``document_citations`` (legal per
    ``app.schemas.verification.Claim`` -- a claim need not carry any
    ``SourceRef`` at all) is not rendered indistinguishable from a
    genuinely-uncited claim: ``source_refs=[]`` alone cannot tell those two
    cases apart. ``document_citation_count`` is the number of
    ``DocumentCitationCheckResult`` entries this claim's citations included
    (always >=0; a claim reaching ``RenderedClaim`` has already passed, so
    every one of them verified). ``all_citations_verified`` restates the
    claim's ``ClaimCheckResult.passed`` (always ``True`` for a
    ``RenderedClaim`` today, since ``render_answer`` only builds one when
    ``result.passed``) so a consumer has an explicit, self-describing field
    rather than needing to infer verification status from field presence.
    Document-citation CHIPS (rendering the actual citations, not just their
    count) remain the separate, not-yet-built P3.7/P3.8 UI concern -- this
    only stops discarding the fact that verified document evidence exists.
    """

    text: str
    source_refs: list[SourceRef]
    document_citation_count: int
    all_citations_verified: bool


@dataclass(frozen=True)
class Notice:
    """Replaces a claim whose citations did not all pass. Carries no trace
    of the original claim text or which citation(s) failed -- see module
    docstring."""

    text: str = _NOT_FOUND_NOTICE_TEXT


AnswerSegment = RenderedClaim | Notice


@dataclass(frozen=True)
class RenderedAnswer:
    """The user-facing rendered answer: one segment per checked claim, in
    the original order."""

    segments: list[AnswerSegment]


def render_answer(results: list[ClaimCheckResult]) -> RenderedAnswer:
    """Strip failed claims, replacing each with a ``Notice``; keep passed
    claims verbatim with their passing citations. Order-preserving,
    one-to-one with ``results``.

    ``RenderedClaim.source_refs`` only ever carries ``SourceRef`` citations
    -- a passed claim's ``citation_results`` may now also include
    ``DocumentCitationCheckResult`` entries (P3.6's document-citation
    extension to ``app.verification``), which this pre-P3.6 rendering
    contract has no slot for as CHIPS yet; wiring document-citation chips
    into the UI is a separate, not-yet-built P3.7/P3.8 concern (the
    ``VerifiedAnswer``/``RenderedAnswer`` docstrings already flag this
    contract as due for a richer shape). Filtering ``source_refs`` to just
    the ``SourceRef``-shaped results (rather than erroring) keeps a mixed-
    citation claim renderable today without losing its already-supported
    ``SourceRef`` chips -- but the document-citation count is NOT simply
    dropped: ``document_citation_count``/``all_citations_verified`` on
    ``RenderedClaim`` preserve that a claim verified solely (or partly) by
    document evidence is not the same as an uncited claim (see
    ``RenderedClaim``'s docstring).
    """
    segments: list[AnswerSegment] = [
        RenderedClaim(
            text=result.claim.text,
            source_refs=[c.source_ref for c in result.citation_results if isinstance(c, CitationCheckResult)],
            document_citation_count=sum(
                1 for c in result.citation_results if isinstance(c, DocumentCitationCheckResult)
            ),
            all_citations_verified=result.passed,
        )
        if result.passed
        else Notice()
        for result in results
    ]
    return RenderedAnswer(segments=segments)

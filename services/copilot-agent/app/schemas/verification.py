"""Claim-level response contract for the verification layer (P3.1).

The planner's raw output (``app.schemas.planner.FinalAnswer``) is a single
free-text answer with an *optional*, flat list of refs -- that's what a 4B
model can reliably produce via the P2.9 two-call extraction, and it stays
untouched here. It models "the whole answer optionally has some refs," not
"every factual claim carries its own ref."

``VerifiedAnswer`` is a *separate* schema, not an evolution of
``FinalAnswer``. It's what the verification layer (P3.2 citation checker,
P3.3 claim stripping) produces FROM a ``FinalAnswer`` by splitting its prose
into individual claims and validating each one's refs against the cached
tool results for the conversation -- deterministic Python, not model output.
Keeping the two schemas distinct means:

- P2.9's extraction schema/tests are untouched by the verification
  contract landing.
- The "claim without a ref is rejected" rule lives at the boundary that
  actually needs to enforce it (the verification layer's output), not on
  the raw, necessarily-looser model-extraction step.

Every ``Claim`` is *meant* to carry at least one ``SourceRef``/
``DocumentCitation`` -- the headline P3.1 requirement -- but (issue #93,
Option C) that requirement is deliberately NOT enforced here at
construction/parse time. ``VerifiedAnswer.claims`` is a ``list[Claim]``, and
Pydantic validates a list of sub-models all-or-nothing: a per-``Claim``
``model_validator`` that raises on zero citations would fail the WHOLE
``VerifiedAnswer.model_validate`` call the moment ANY one claim in the list
is uncitable -- destroying every co-occurring claim, including ones that
DO carry valid citations, over one unrelated claim's defect. Confirmed live
as the mechanism behind a citation_present eval failure: a deterministic
(temperature=0) extraction model re-emits the same uncitable claim on every
``LlamaServerClient.extract`` retry, so retries cannot recover a different
mix -- the whole answer's claims were lost every time.

The citation-count bar is therefore enforced ONE LAYER DOWN instead, where
its blast radius is naturally scoped to the one claim that fails it:
``app.verification.check_claim`` re-validates each claim's citations
independently, and ``ClaimCheckResult.passed`` is vacuously ``False`` for a
claim with zero citation results (``all([])`` is ``True``, but the ``passed``
property also requires ``bool(citation_results)``) -- exactly like a claim
whose citations all fail re-validation. ``app.rendering.render_answer``
already strips any failed claim to a notice without touching its siblings.
So parsing a zero-citation claim still never lets it reach the user as a
verified fact; it just no longer takes unrelated valid claims down with it.
Non-factual segments (e.g. the "not found in record" notices P3.3 inserts
when a citation fails re-validation) are deliberately NOT modeled here:
they're a P3.3 concern, produced by the checker after claims have already
passed THIS (now looser) parse contract, not raw model output that needs a
schema hook today.

**Document citations (P3.6).** ``document_citations`` is the additive,
document-sourced counterpart to ``source_refs``
(`docs/W2_ARCHITECTURE.md` "Citation Contract") -- a claim built on a
lab/intake-form fact or a hybrid-retrieval guideline chunk carries a
``DocumentCitation`` (``app.schemas.ingestion``) instead of/alongside a
``SourceRef``. Both fields default to an empty list so existing
``source_refs``-only claims are unaffected; "at least one citation of EITHER
shape is present" is the same "claim without a citation is rejected" rule as
before, just widened to cover both citation shapes -- enforced by
``app.verification`` (not a schema-level validator here; see below and the
module docstring's "issue #93" paragraph), which re-validates both kinds of
citation into the SAME ``ClaimCheckResult`` (P3.2's AND-across-citations
aggregation, not forked) -- see that module's docstring for the
document-citation extension.
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import SourceRef, ToolSchemaModel
from app.schemas.ingestion import DocumentCitation


class Claim(ToolSchemaModel):
    """A single factual claim, MEANT to be cited by >=1 ``SourceRef``/
    ``DocumentCitation`` (see module docstring, "Document citations") --
    but that requirement is deliberately NOT enforced here at
    construction/parse time (issue #93, Option C); see the module
    docstring's "issue #93" paragraph for why, and ``has_citation``/
    ``app.verification.check_claim`` for where it actually is enforced.
    """

    text: str = Field(min_length=1)
    source_refs: list[SourceRef] = Field(default_factory=list)
    document_citations: list[DocumentCitation] = Field(default_factory=list)

    @property
    def has_citation(self) -> bool:
        """Whether this claim carries >=1 citation of either shape. A cheap,
        named predicate for the same condition ``app.verification
        .ClaimCheckResult.passed`` already enforces via re-validation
        (``bool(citation_results)``) -- this property doesn't replace that
        enforcement, it just gives call sites (logging, tests) a readable
        way to ask the question without re-deriving it."""
        return bool(self.source_refs or self.document_citations)


class VerifiedAnswer(ToolSchemaModel):
    """The verification layer's response contract: an answer decomposed
    into individually-cited claims.

    ``claims`` may be empty -- e.g. every claim in the source answer failed
    citation and was stripped by P3.3, leaving only non-factual notice text
    that this schema doesn't model. A claim present here may or may not
    carry its own refs -- parsing no longer enforces that (issue #93,
    Option C; see ``Claim``'s docstring) -- an uncited claim still never
    counts as verified, it just doesn't block its siblings from parsing."""

    claims: list[Claim]

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

Every ``Claim`` must carry at least one ``SourceRef``/``DocumentCitation`` --
a claim with zero citations of either shape fails schema validation
(``ValidationError``), which is the headline P3.1 requirement. Non-factual
segments (e.g. the "not found in record" notices P3.3 inserts when a
citation fails re-validation) are deliberately NOT modeled here: they're a
P3.3 concern, produced by the checker after claims have already passed this
contract, not raw model output that needs a schema hook today.

**Document citations (P3.6).** ``document_citations`` is the additive,
document-sourced counterpart to ``source_refs``
(`docs/W2_ARCHITECTURE.md` "Citation Contract") -- a claim built on a
lab/intake-form fact or a hybrid-retrieval guideline chunk carries a
``DocumentCitation`` (``app.schemas.ingestion``) instead of/alongside a
``SourceRef``. Both fields default to an empty list so existing
``source_refs``-only claims are unaffected; a model-level validator (not a
per-field ``min_length``, since either field alone may legitimately be
empty) enforces that at least one citation of EITHER shape is present --
the same "claim without a citation is rejected" rule as before, just
widened to cover both citation shapes. ``app.verification`` re-validates
both kinds of citation into the SAME ``ClaimCheckResult`` (P3.2's
AND-across-citations aggregation, not forked) -- see that module's
docstring for the document-citation extension.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from app.schemas.common import SourceRef, ToolSchemaModel
from app.schemas.ingestion import DocumentCitation


class Claim(ToolSchemaModel):
    """A single factual claim, cited by >=1 ``SourceRef``/``DocumentCitation``
    (see module docstring, "Document citations").
    """

    text: str = Field(min_length=1)
    source_refs: list[SourceRef] = Field(default_factory=list)
    document_citations: list[DocumentCitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_at_least_one_citation(self) -> "Claim":
        if not self.source_refs and not self.document_citations:
            raise ValueError("Claim must carry at least one SourceRef or DocumentCitation")
        return self


class VerifiedAnswer(ToolSchemaModel):
    """The verification layer's response contract: an answer decomposed
    into individually-cited claims.

    ``claims`` may be empty -- e.g. every claim in the source answer failed
    citation and was stripped by P3.3, leaving only non-factual notice text
    that this schema doesn't model. Each claim present, however, must carry
    its own refs (enforced by ``Claim`` above).
    """

    claims: list[Claim]

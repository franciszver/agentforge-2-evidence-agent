"""Document-ingestion schemas (P3.1, `docs/W2_ARCHITECTURE.md` "Schemas").

``Citation``/``LabResultFact``/``ExtractedLabRow``/``LabPageExtraction`` are
the VLM-extraction-side contracts: like every existing tool schema in
``app/schemas/tools.py`` (and ``app/schemas/verification.py``'s
``Claim``/``VerifiedAnswer``, the same category of "constrained-decoding
target" schema), they extend ``ToolSchemaModel`` (frozen + ``extra="forbid"``)
so a malformed VLM extraction fails schema validation instead of silently
coercing bad data. ``DocumentCitation`` is the document-sourced counterpart
to Phase 1's structured-data citation shape and, per the architecture doc,
is a plain ``BaseModel`` (not a tool I/O contract) -- it is what a later
claim built on top of an extracted fact carries, not tool input/output.

**No-fabrication contract.** Any field the model cannot read from the
source document is ``None`` ("not found"), never a guessed value -- this is
the extraction-side equivalent of the verification layer's fail-closed
citation checking. Only ``test`` is required: a lab row with no legible test
name identifies nothing to attach a result to, so it is not a row at all.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ToolSchemaModel


class Citation(ToolSchemaModel):
    """Provenance pointer from an extracted fact back to its source page."""

    source_type: Literal["lab_pdf", "intake_form"]
    source_id: str  # stored source-document identifier
    page_or_section: str  # e.g. "page 2" or "Section: Medications"
    field_or_chunk_id: str  # which extracted field/row this citation backs
    quote_or_value: str  # the literal text/value the model read


class LabFlagCode(StrEnum):
    """The literal abnormal-flag code as PRINTED on a lab report page.

    Deliberately separate from ``app.schemas.common.AbnormalFlag``: that
    enum models the NORMALIZED semantic vocabulary structured tool output
    uses (``HIGH``/``LOW``/``CRITICAL_HIGH``/... from FHIR interpretation
    codes), whereas the VLM here reads a report's own printed single-letter
    code verbatim off the page -- conflating the two would mean silently
    reinterpreting what was actually read as something the model didn't say.
    """

    HIGH = "H"
    LOW = "L"
    ABNORMAL = "A"
    NORMAL = "N"


class ExtractedLabRow(ToolSchemaModel):
    """One VLM-extracted lab row, before a ``Citation`` is attached.

    Every field but ``test`` is ``None`` when the model could not read it --
    see the module docstring's no-fabrication contract.

    ``abnormal_flag`` is deliberately the RAW printed code as a bounded
    string (``max_length=8``), not ``LabFlagCode`` -- constraining it to the
    closed enum at THIS schema level means a single row's flag drift (a
    lowercase ``"h"``, a report printing ``"HIGH"`` instead of the HL7
    single-letter code) fails validation for the ENTIRE page's
    ``LabPageExtraction``, discarding every other row on that page over one
    cosmetic mismatch. Normalization into ``LabFlagCode`` happens
    deterministically downstream, in ``app.ingestion``'s row-to-fact
    assembly, where an unrecognized code degrades to ``None`` for that ONE
    field rather than losing the whole page.
    """

    test: str
    value: str | None
    unit: str | None
    reference_range: str | None
    collection_date: str | None
    abnormal_flag: str | None = Field(max_length=8)


class LabPageExtraction(ToolSchemaModel):
    """The VLM's schema-constrained output for one rendered page."""

    rows: list[ExtractedLabRow]


class LabResultFact(ToolSchemaModel):
    """One extracted lab-report row. Every field but ``test`` and
    ``citation`` is ``None`` when the model could not read it -- see the
    module docstring's no-fabrication contract."""

    test: str
    value: str | None
    unit: str | None
    reference_range: str | None
    collection_date: str | None  # ISO date if legible, else None
    abnormal_flag: LabFlagCode | None
    citation: Citation


class IntakeFormExtraction(ToolSchemaModel):
    """The VLM's schema-constrained output for one rendered intake-form
    page (`docs/W2_ARCHITECTURE.md` "Schemas" -- mirrors ``IntakeFormFact``
    minus ``citation``). A real intake form may span several pages, so any
    section absent from THIS page is the empty/``None`` value for its
    type -- ``demographics={}``, ``medications=[]``, etc. -- never guessed
    from another page. Assembled into ``IntakeFormFact`` (with citation)
    downstream in ``app.ingestion``."""

    demographics: dict[str, str | None]
    chief_concern: str | None
    medications: list[str]
    allergies: list[str]
    family_history: list[str]


class IntakeFormFact(IntakeFormExtraction):
    """One page's worth of extracted intake-form data -- ``IntakeFormExtraction``
    plus the ``Citation`` back to its source page. Per the no-fabrication
    contract, any demographic key/``chief_concern`` the model could not read
    on this page is absent/``None``, and any list section with nothing
    legible on this page is empty -- never guessed from another page."""

    citation: Citation


class DocumentCitation(BaseModel):
    """The citation contract a later claim built on a document-sourced fact
    carries (`docs/W2_ARCHITECTURE.md` "Citation Contract") -- the
    document-sourced counterpart to Phase 1's
    ``{tool_call_id, record_id, field, asserted_value}`` shape. Not a tool
    I/O contract, so it does not extend ``ToolSchemaModel``.

    ``source_type`` additionally allows ``"guideline_chunk"`` (P3.6) beyond
    the extraction-side ``Citation``'s ``"lab_pdf"``/``"intake_form"``: a
    claim can also cite a hybrid-retrieval guideline chunk
    (``app.schemas.retrieval.RetrievedChunk``), which has no VLM-extraction
    step and therefore never appears on the extraction-side ``Citation``
    above. For a ``"guideline_chunk"`` citation, ``source_id`` is the
    corpus doc id, ``field_or_chunk_id`` is the chunk id
    (``<doc_id>#<section-slug>``), and ``quote_or_value`` is the literal
    text the claim quotes from that chunk -- see
    ``app.verification.check_document_citation`` for how each source type is
    re-validated against its RAW source.
    """

    source_type: Literal["lab_pdf", "intake_form", "guideline_chunk"]
    source_id: str
    page_or_section: str
    field_or_chunk_id: str
    # min_length=1 rejects a fully-empty string outright; the model
    # validator below additionally rejects a WHITESPACE-only string (which
    # satisfies min_length=1 but is just as void of asserted content) -- a
    # security-gate finding: a blank quote would otherwise trivially
    # "verify" any guideline_chunk citation (an empty/blank substring is
    # vacuously present in any text), asserting nothing while reading as
    # VALID. See ``app.verification.check_document_citation`` for the
    # checker's OWN independent defensive guard against the same failure
    # mode, for citations that bypass this schema (e.g. via
    # ``model_construct``).
    quote_or_value: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_non_blank_quote(self) -> "DocumentCitation":
        if not self.quote_or_value.strip():
            raise ValueError("quote_or_value must not be blank")
        return self

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

from pydantic import BaseModel

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
    """

    test: str
    value: str | None
    unit: str | None
    reference_range: str | None
    collection_date: str | None
    abnormal_flag: LabFlagCode | None


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


class DocumentCitation(BaseModel):
    """The citation contract a later claim built on a document-sourced fact
    carries (`docs/W2_ARCHITECTURE.md` "Citation Contract") -- the
    document-sourced counterpart to Phase 1's
    ``{tool_call_id, record_id, field, asserted_value}`` shape. Not a tool
    I/O contract, so it does not extend ``ToolSchemaModel``."""

    source_type: Literal["lab_pdf", "intake_form"]
    source_id: str
    page_or_section: str
    field_or_chunk_id: str
    quote_or_value: str

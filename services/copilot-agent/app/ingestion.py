"""``attach_and_extract`` -- document ingestion, ``doc_type="lab_pdf"`` slice
(P3.1, `docs/W2_ARCHITECTURE.md` "Schemas" / "Data Model, Lineage, and Access
Control").

Pipeline: render each PDF page to an image (pypdfium2, in-process, no
external poppler/system dependency) -> schema-constrained VLM extraction per
page via ``OllamaClient.extract`` (the same pattern P2.7 established for the
planner/quarantine LLM, extended here with the ``images`` param it now
accepts) -> deterministic assembly into ``LabResultFact`` + ``Citation`` ->
persisted via the injected ``DocumentStore``/``FactStore``.

**No-fabrication contract.** Per-field ``None`` ("not found") is the ONLY
representation for anything the VLM could not read -- the extraction prompt
instructs this explicitly, and nothing in this module's assembly step ever
substitutes a guessed value for a missing one. A row with no legible test
name is dropped entirely (nothing to attach a result to), never invented.

**Storage honesty (P3.1 scope).** `docs/W2_ARCHITECTURE.md` "Migration
Notes" specifies the source document is stored in OpenEMR's own
document-management facilities and extracted facts are persisted as FHIR
``Observation`` resources through OpenEMR's REST/FHIR API. That OpenEMR-side
plumbing does not exist yet. ``DocumentStore``/``FactStore`` are the seams
that persistence will eventually implement; ``LocalIngestionStore`` below is
an explicit, disclosed LOCAL-DISK PLACEHOLDER for both -- not a FHIR write,
not OpenEMR document storage. Wiring the real OpenEMR/FHIR-backed
implementations is deferred, tracked on issue #13's thread, and is NOT faked
here: callers get an honest local artifact, not a write that only looks like
it reached OpenEMR.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import pypdfium2 as pdfium

from app.ollama_client import OllamaError
from app.schemas.ingestion import Citation, ExtractedLabRow, LabPageExtraction, LabResultFact

_logger = logging.getLogger(__name__)

# Render scale (1.0 == 72 DPI). 2.0 (~144 DPI) is generous for typical
# lab-report font sizes without inflating the base64 image payload past what
# a 7B-class VLM's context window needs for a single page.
_RENDER_SCALE = 2.0

_LAB_EXTRACTION_PROMPT = """\
You are extracting a lab-report page image into structured data. Read ONLY \
what is legible on this page image. For any field you cannot read -- \
blurred, cropped, obscured, or simply absent -- set it to null. NEVER guess \
or infer a value that is not visibly present on the page; a missing or \
illegible field must be reported as null, not as your best guess.

Report one row per lab test result you can identify on this page. Each row \
has:
  - test: the test name (required -- if the test name itself is illegible, \
omit that row entirely rather than guessing a name).
  - value: the result value, or null if illegible/absent.
  - unit: the unit, or null.
  - reference_range: the reference range, or null.
  - collection_date: the collection date in ISO 8601 (YYYY-MM-DD) if \
legible, else null.
  - abnormal_flag: one of "H", "L", "A", "N", or null if no flag is legible.
"""


class _Extractor(Protocol):
    """The one capability ``attach_and_extract`` needs from the VLM client:
    schema-constrained extraction with optional page images. Deliberately
    narrow, mirroring ``app.extraction._Extractor``."""

    def extract(
        self,
        prompt_or_messages: Any,
        schema: type,
        *,
        options: Any = None,
        images: list[str] | None = None,
    ) -> Any: ...


class DocumentStore(Protocol):
    """Where the source PDF/scan itself is persisted. A stopgap for
    OpenEMR's document-management facilities -- see module docstring."""

    def save_source_document(self, patient_id: int, doc_type: str, file_path: Path) -> str:
        """Persist the source document; returns its ``source_id``."""
        ...


class FactStore(Protocol):
    """Where extracted facts are persisted. A stopgap for FHIR
    ``Observation`` writes -- see module docstring."""

    def save_facts(self, patient_id: int, source_id: str, facts: Sequence[LabResultFact]) -> None: ...


@dataclass(frozen=True)
class IngestionResult:
    """The result of one ``attach_and_extract`` call."""

    source_id: str
    facts: list[LabResultFact]


def render_pdf_pages_to_png(file_path: Path) -> list[bytes]:
    """Render every page of the PDF at ``file_path`` to PNG bytes, in page order."""
    pages: list[bytes] = []
    pdf = pdfium.PdfDocument(str(file_path))
    try:
        for page in pdf:
            bitmap = page.render(scale=_RENDER_SCALE)
            buffer = io.BytesIO()
            bitmap.to_pil().save(buffer, format="PNG")
            pages.append(buffer.getvalue())
    finally:
        pdf.close()
    return pages


def _extract_page(ollama_client: _Extractor, *, page_index: int, image_png: bytes) -> LabPageExtraction:
    image_b64 = base64.b64encode(image_png).decode("ascii")
    messages = [{"role": "user", "content": _LAB_EXTRACTION_PROMPT}]
    try:
        result = ollama_client.extract(messages, LabPageExtraction, images=[image_b64])
    except OllamaError:
        _logger.warning("lab pdf page extraction failed", extra={"page_index": page_index})
        return LabPageExtraction(rows=[])
    return result  # type: ignore[no-any-return]


def _quote_for_row(row: ExtractedLabRow) -> str:
    """Deterministic, literal rendering of what the model read for this row.

    Never a new fact -- only a formatted view of already-extracted fields --
    which keeps ``Citation.quote_or_value``'s non-null contract satisfiable
    even when ``value`` itself is ``None`` (not found).
    """
    value_part = row.value if row.value is not None else "(not found)"
    return f"{row.test}: {value_part}"


def _to_lab_result_fact(row: ExtractedLabRow, *, source_id: str, page_index: int) -> LabResultFact:
    citation = Citation(
        source_type="lab_pdf",
        source_id=source_id,
        page_or_section=f"page {page_index + 1}",
        field_or_chunk_id=row.test,
        quote_or_value=_quote_for_row(row),
    )
    return LabResultFact(
        test=row.test,
        value=row.value,
        unit=row.unit,
        reference_range=row.reference_range,
        collection_date=row.collection_date,
        abnormal_flag=row.abnormal_flag,
        citation=citation,
    )


def attach_and_extract(
    patient_id: int,
    file_path: str | Path,
    doc_type: Literal["lab_pdf"],
    *,
    ollama_client: _Extractor,
    document_store: DocumentStore,
    fact_store: FactStore,
) -> IngestionResult:
    """Extract structured ``LabResultFact``s from a lab-report PDF.

    First P3.1 slice: only ``doc_type="lab_pdf"`` is implemented (raises
    ``ValueError`` for anything else -- ``intake_form`` extraction is a
    separate, not-yet-built slice, never silently attempted here). Renders
    every page to an image, runs one schema-constrained VLM extraction call
    per page (``LabPageExtraction``), and deterministically assembles the
    per-row results into ``LabResultFact``s carrying a ``Citation`` back to
    their exact page. Persists the source document and the extracted facts
    via the injected stores (see module docstring's storage-honesty note)
    before returning.
    """
    if doc_type != "lab_pdf":
        raise ValueError(f"attach_and_extract only supports doc_type='lab_pdf' in this slice, got {doc_type!r}")

    path = Path(file_path)
    source_id = document_store.save_source_document(patient_id, doc_type, path)

    facts: list[LabResultFact] = []
    for page_index, image_png in enumerate(render_pdf_pages_to_png(path)):
        extraction = _extract_page(ollama_client, page_index=page_index, image_png=image_png)
        facts.extend(
            _to_lab_result_fact(row, source_id=source_id, page_index=page_index) for row in extraction.rows
        )

    fact_store.save_facts(patient_id, source_id, facts)
    return IngestionResult(source_id=source_id, facts=facts)


class LocalIngestionStore:
    """Local-disk placeholder for OpenEMR document storage + FHIR fact
    persistence (deferred -- see module docstring, issue #13).

    Writes the source document under
    ``<base_dir>/documents/<source_id><suffix>`` and the extracted facts as
    JSON under ``<base_dir>/facts/<source_id>.json``. Satisfies both
    ``DocumentStore`` and ``FactStore`` -- one injectable seam for both
    halves of P3.1's thin storage slice.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        (self._base_dir / "documents").mkdir(parents=True, exist_ok=True)
        (self._base_dir / "facts").mkdir(parents=True, exist_ok=True)

    def save_source_document(self, patient_id: int, doc_type: str, file_path: Path) -> str:
        source_id = uuid.uuid4().hex
        dest = self._base_dir / "documents" / f"{source_id}{file_path.suffix}"
        dest.write_bytes(file_path.read_bytes())
        return source_id

    def save_facts(self, patient_id: int, source_id: str, facts: Sequence[LabResultFact]) -> None:
        dest = self._base_dir / "facts" / f"{source_id}.json"
        payload = {
            "patient_id": patient_id,
            "source_id": source_id,
            "facts": [fact.model_dump(mode="json") for fact in facts],
        }
        dest.write_text(json.dumps(payload, indent=2))

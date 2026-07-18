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
from app.schemas.ingestion import Citation, ExtractedLabRow, LabFlagCode, LabPageExtraction, LabResultFact

_logger = logging.getLogger(__name__)

# Render scale (1.0 == 72 DPI). 2.0 (~144 DPI) is generous for typical
# lab-report font sizes without inflating the base64 image payload past what
# a 7B-class VLM's context window needs for a single page.
_RENDER_SCALE = 2.0

# Defensible bound for a clinical document (lab report, intake form): well
# beyond any real report, but small enough that a malicious or corrupt PDF
# claiming thousands of pages cannot force this module to materialize an
# unbounded in-memory list of rendered page images.
MAX_PAGES = 50

# Per-side page-dimension cap, in PDF points (1/72in). 8000pt (~111in) is far
# beyond any real scanned document; guards against a crafted/corrupt MediaBox
# blowing up render-time memory (a rasterized page's memory cost scales with
# width x height x _RENDER_SCALE^2).
MAX_PAGE_POINTS = 8000.0


class IngestionError(Exception):
    """Raised when a source document cannot be safely parsed/rendered for
    ingestion -- malformed/corrupt input, too many pages, or an oversized
    page dimension. Callers see this ONE stable, log-safe error type
    regardless of the underlying parser failure -- never a raw pdfium
    exception, and never a partially-completed ingestion (see
    ``attach_and_extract``'s validate-then-store ordering)."""

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
    """The result of one ``attach_and_extract`` call.

    ``failed_pages`` (1-based, matching ``Citation.page_or_section``'s own
    "page N" numbering) lists every page whose VLM extraction call failed
    outright -- DISTINCT from a page that was successfully read and
    legitimately contained zero lab rows (that page is simply absent from
    ``failed_pages`` and contributes no facts). **Callers MUST treat a
    non-empty ``failed_pages`` as a PARTIAL extraction**: ``facts`` only
    ever reflects the pages that succeeded, and a caller presenting this
    result to a clinician must surface which pages could not be processed
    rather than silently presenting ``facts`` as complete for the whole
    document.
    """

    source_id: str
    facts: list[LabResultFact]
    pages_total: int
    failed_pages: list[int]


def _open_pdf(file_path: Path) -> pdfium.PdfDocument:
    """Open ``file_path`` as a PDF, translating a malformed/corrupt input into
    ``IngestionError`` instead of letting a raw pdfium exception escape."""
    try:
        return pdfium.PdfDocument(str(file_path))
    except pdfium.PdfiumError as exc:
        raise IngestionError(f"{file_path.name}: could not be parsed as a PDF") from exc


def _validate_pdf(pdf: pdfium.PdfDocument, *, file_path: Path) -> None:
    """Enforce ``MAX_PAGES``/``MAX_PAGE_POINTS`` against an already-open
    document, BEFORE anything is rendered or stored. Raises
    ``IngestionError`` -- a document outside these bounds is rejected
    outright, never clamped/truncated silently."""
    page_count = len(pdf)
    if page_count > MAX_PAGES:
        raise IngestionError(f"{file_path.name}: {page_count} pages exceeds the {MAX_PAGES}-page ingestion limit")

    for page_index in range(page_count):
        page = pdf.get_page(page_index)
        try:
            width, height = page.get_size()
        finally:
            page.close()
        if width > MAX_PAGE_POINTS or height > MAX_PAGE_POINTS:
            raise IngestionError(
                f"{file_path.name}: page {page_index + 1} ({width:.0f}x{height:.0f}pt) "
                f"exceeds the {MAX_PAGE_POINTS:.0f}pt per-side ingestion limit"
            )


def render_pdf_pages_to_png(file_path: Path) -> list[bytes]:
    """Render every page of the PDF at ``file_path`` to PNG bytes, in page
    order. Validates ``MAX_PAGES``/``MAX_PAGE_POINTS`` (see ``_validate_pdf``)
    before rendering anything; raises ``IngestionError`` on a malformed PDF
    or a document outside those bounds."""
    pdf = _open_pdf(file_path)
    try:
        _validate_pdf(pdf, file_path=file_path)
        pages: list[bytes] = []
        for page in pdf:
            bitmap = page.render(scale=_RENDER_SCALE)
            buffer = io.BytesIO()
            bitmap.to_pil().save(buffer, format="PNG")
            pages.append(buffer.getvalue())
        return pages
    finally:
        pdf.close()


def _extract_page(ollama_client: _Extractor, *, page_index: int, image_png: bytes) -> LabPageExtraction | None:
    """Run the schema-constrained VLM extraction call for one page.

    Returns ``None`` -- never an empty ``LabPageExtraction`` -- when the
    call fails outright, so a failed page stays distinguishable from a page
    that was successfully read and legitimately had zero rows (see
    ``IngestionResult.failed_pages``).
    """
    image_b64 = base64.b64encode(image_png).decode("ascii")
    messages = [{"role": "user", "content": _LAB_EXTRACTION_PROMPT}]
    try:
        result = ollama_client.extract(messages, LabPageExtraction, images=[image_b64])
    except OllamaError:
        _logger.warning("lab pdf page extraction failed", extra={"page_index": page_index})
        return None
    return result  # type: ignore[no-any-return]


def _normalize_flag_code(raw: str | None) -> LabFlagCode | None:
    """Case-insensitively normalize a VLM-reported raw flag code into
    ``LabFlagCode``. An unrecognized code (not one of H/L/A/N, any case)
    degrades to ``None`` -- fail-closed per the no-fabrication contract,
    logged at WARNING with ONLY the code token itself (a flag letter/short
    token is not PHI; no other row/page data is logged)."""
    if raw is None:
        return None
    try:
        return LabFlagCode(raw.strip().upper())
    except ValueError:
        _logger.warning("unrecognized lab abnormal-flag code", extra={"flag_code": raw})
        return None


def _quote_for_row(row: ExtractedLabRow, *, normalized_flag: LabFlagCode | None) -> str:
    """Deterministic, literal rendering of what the model read for this row.

    Never a new fact -- only a formatted view of already-extracted fields --
    which keeps ``Citation.quote_or_value``'s non-null contract satisfiable
    even when ``value`` itself is ``None`` (not found). When the row's raw
    flag code failed to normalize (``normalized_flag is None`` but
    ``row.abnormal_flag`` was present), the raw code is appended verbatim --
    the fact's own ``abnormal_flag`` is ``None`` (fail-closed), but the
    citation must still quote what the page actually printed, truthfully.
    """
    value_part = row.value if row.value is not None else "(not found)"
    quote = f"{row.test}: {value_part}"
    if row.abnormal_flag is not None and normalized_flag is None:
        quote += f" [flag: {row.abnormal_flag}]"
    return quote


def _to_lab_result_fact(row: ExtractedLabRow, *, source_id: str, page_index: int) -> LabResultFact:
    normalized_flag = _normalize_flag_code(row.abnormal_flag)
    citation = Citation(
        source_type="lab_pdf",
        source_id=source_id,
        page_or_section=f"page {page_index + 1}",
        field_or_chunk_id=row.test,
        quote_or_value=_quote_for_row(row, normalized_flag=normalized_flag),
    )
    return LabResultFact(
        test=row.test,
        value=row.value,
        unit=row.unit,
        reference_range=row.reference_range,
        collection_date=row.collection_date,
        abnormal_flag=normalized_flag,
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

    **Validate-then-store.** The PDF is opened and checked against
    ``MAX_PAGES``/``MAX_PAGE_POINTS`` (``_validate_pdf``) BEFORE anything is
    written to ``document_store`` -- a malformed or out-of-bounds PDF raises
    ``IngestionError`` with nothing persisted, never an orphaned stored file
    left behind by a parse failure that happens after storage.

    **Partial extraction.** A page whose VLM call fails outright is recorded
    in the returned ``IngestionResult.failed_pages`` (1-based) rather than
    silently contributing zero facts indistinguishable from "this page had
    no lab rows" -- see that field's docstring for the caller obligation.
    """
    if doc_type != "lab_pdf":
        raise ValueError(f"attach_and_extract only supports doc_type='lab_pdf' in this slice, got {doc_type!r}")

    path = Path(file_path)
    pdf = _open_pdf(path)
    try:
        _validate_pdf(pdf, file_path=path)
    finally:
        pdf.close()

    source_id = document_store.save_source_document(patient_id, doc_type, path)

    pages = render_pdf_pages_to_png(path)
    facts: list[LabResultFact] = []
    failed_pages: list[int] = []
    for page_index, image_png in enumerate(pages):
        extraction = _extract_page(ollama_client, page_index=page_index, image_png=image_png)
        if extraction is None:
            failed_pages.append(page_index + 1)
            continue
        facts.extend(
            _to_lab_result_fact(row, source_id=source_id, page_index=page_index) for row in extraction.rows
        )

    fact_store.save_facts(patient_id, source_id, facts)
    return IngestionResult(source_id=source_id, facts=facts, pages_total=len(pages), failed_pages=failed_pages)


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

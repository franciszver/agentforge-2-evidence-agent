"""Generates the synthetic 2-page lab-report PDF fixture for P3.1 document
ingestion (``tests/fixtures/lab_report_synthetic.pdf``).

All data is obviously synthetic: a fake "Test Patient", made-up dates, and
values chosen to be plausible lab numbers rather than copied from any real
report. One field (page 2's Creatinine row's collection date) is
deliberately covered by an opaque black box, simulating a genuinely
unreadable/obscured field on a real scan -- it exercises the "not found"
path: the VLM (or, in hermetic tests, a scripted double standing in for it)
must report that field as ``None`` rather than guess a plausible-looking
date.

Regenerate with (from ``services/copilot-agent/``, using the dev venv which
has ``fpdf2`` installed per ``pyproject.toml``'s ``dev`` extra):

    python scripts/generate_lab_pdf_fixture.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

_OUTPUT_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "lab_report_synthetic.pdf"

# (test, value, unit, reference_range, collection_date, abnormal_flag)
_PAGE_1_ROWS = [
    ("Hemoglobin A1c", "5.4", "%", "4.0-5.6", "2026-06-01", "N"),
    ("Fasting Glucose", "142", "mg/dL", "70-99", "2026-06-01", "H"),
    ("Total Cholesterol", "188", "mg/dL", "<200", "2026-06-01", "N"),
    ("LDL Cholesterol", "112", "mg/dL", "<100", "2026-06-01", "H"),
]
_PAGE_2_ROWS = [
    ("HDL Cholesterol", "58", "mg/dL", "40-60", "2026-06-01", "N"),
    ("Triglycerides", "97", "mg/dL", "<150", "2026-06-01", "N"),
    ("TSH", "2.1", "mIU/L", "0.4-4.0", "2026-06-01", "N"),
    # Creatinine's collection date is covered by an opaque box below -- the
    # deliberately unreadable field.
    ("Creatinine", "0.9", "mg/dL", "0.6-1.3", "2026-06-01", "N"),
]

_ROW_HEIGHT = 8
_COL_WIDTHS = [45, 20, 18, 28, 30, 12]
_HEADERS = ["Test", "Value", "Unit", "Reference Range", "Collection Date", "Flag"]


def _draw_header(pdf: FPDF) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Synthetic Lab Report (test fixture -- not a real patient)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Patient: Test Patient", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "DOB: 1990-01-01 (synthetic)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "MRN: TEST-000000", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def _draw_table(pdf: FPDF, rows: list[tuple[str, str, str, str, str, str]]) -> None:
    pdf.set_font("Helvetica", "B", 9)
    for header, width in zip(_HEADERS, _COL_WIDTHS):
        pdf.cell(width, _ROW_HEIGHT, header, border=1)
    pdf.ln(_ROW_HEIGHT)

    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        for value, width in zip(row, _COL_WIDTHS):
            pdf.cell(width, _ROW_HEIGHT, value, border=1)
        pdf.ln(_ROW_HEIGHT)


def _redact_page_2_creatinine_date(pdf: FPDF) -> None:
    """Draw an opaque black rectangle over page 2's Creatinine row's
    collection-date cell -- the deliberately unreadable field (see module
    docstring)."""
    header_row_top = pdf.t_margin + 10 + 7 * 3 + 4  # title + 3 demographic lines + gap
    date_col_x = pdf.l_margin + sum(_COL_WIDTHS[:4])
    row_top = header_row_top + _ROW_HEIGHT * (1 + 3)  # +1 header row, +3 rows before Creatinine (4th row)
    pdf.set_fill_color(0, 0, 0)
    pdf.rect(date_col_x, row_top, _COL_WIDTHS[4], _ROW_HEIGHT, style="F")


def generate(output_path: Path = _OUTPUT_PATH) -> None:
    pdf = FPDF(orientation="L", unit="mm", format="A4")

    pdf.add_page()
    _draw_header(pdf)
    _draw_table(pdf, _PAGE_1_ROWS)

    pdf.add_page()
    _draw_header(pdf)
    _draw_table(pdf, _PAGE_2_ROWS)
    _redact_page_2_creatinine_date(pdf)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


if __name__ == "__main__":
    generate()

"""Generates the synthetic 2-page intake-form PDF fixture for P3.2 document
ingestion (``tests/fixtures/intake_form_synthetic.pdf``).

Mirrors ``scripts/generate_lab_pdf_fixture.py``'s pattern: all data is
obviously synthetic (a fake "Test Patient", made-up demographics/history),
and one field is deliberately covered by an opaque black box to simulate a
genuinely unreadable/obscured field on a real scan -- here, page 1's DOB
value -- exercising the "not found" path: the VLM (or, in hermetic tests, a
scripted double standing in for it) must report that field as ``None``
rather than guess a plausible-looking date.

Regenerate with (from ``services/copilot-agent/``, using the dev venv which
has ``fpdf2`` installed per ``pyproject.toml``'s ``dev`` extra):

    python scripts/generate_intake_form_fixture.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

_OUTPUT_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "intake_form_synthetic.pdf"

# Pinned (not "now") so regenerating this fixture is byte-stable -- see
# scripts/generate_lab_pdf_fixture.py's identical rationale.
_PINNED_CREATION_DATE = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)

_DEMOGRAPHICS = [
    ("Name", "Test Patient"),
    ("DOB", "1990-01-01"),  # deliberately redacted below -- the unreadable field
    ("Sex", "F"),
    ("MRN", "TEST-000000"),
]
_CHIEF_CONCERN = "Intermittent chest tightness for the past 3 days."
_MEDICATIONS = [
    "Lisinopril 10mg daily",
    "Metformin 500mg twice daily",
    "Atorvastatin 20mg nightly",
]
_ALLERGIES = ["Penicillin", "Shellfish"]
_FAMILY_HISTORY = [
    "Father: hypertension",
    "Mother: type 2 diabetes",
]

_ROW_HEIGHT = 8
_LABEL_WIDTH = 30
_VALUE_WIDTH = 70


def _draw_title(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def _draw_demographics(pdf: FPDF) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Demographics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    for label, value in _DEMOGRAPHICS:
        pdf.cell(_LABEL_WIDTH, _ROW_HEIGHT, f"{label}:", border=1)
        pdf.cell(_VALUE_WIDTH, _ROW_HEIGHT, value, border=1)
        pdf.ln(_ROW_HEIGHT)
    pdf.ln(4)


def _draw_chief_concern(pdf: FPDF) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Chief Concern", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 7, _CHIEF_CONCERN)


def _draw_list_section(pdf: FPDF, title: str, items: list[str]) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    for item in items:
        pdf.cell(0, 7, f"- {item}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)


def _redact_page_1_dob(pdf: FPDF) -> None:
    """Draw an opaque black rectangle over page 1's DOB value cell -- the
    deliberately unreadable field (see module docstring)."""
    title_row_top = pdf.t_margin + 10 + 2  # title + gap
    demographics_header_top = title_row_top + 7  # "Demographics" heading
    dob_row_top = demographics_header_top + _ROW_HEIGHT * 1  # DOB is the 2nd row
    value_col_x = pdf.l_margin + _LABEL_WIDTH
    pdf.set_fill_color(0, 0, 0)
    pdf.rect(value_col_x, dob_row_top, _VALUE_WIDTH, _ROW_HEIGHT, style="F")


def generate(output_path: Path = _OUTPUT_PATH) -> None:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_creation_date(_PINNED_CREATION_DATE)

    pdf.add_page()
    _draw_title(pdf, "Synthetic Patient Intake Form (test fixture -- not a real patient)")
    _draw_demographics(pdf)
    _draw_chief_concern(pdf)
    _redact_page_1_dob(pdf)

    pdf.add_page()
    _draw_title(pdf, "Synthetic Patient Intake Form (test fixture -- not a real patient), page 2")
    _draw_list_section(pdf, "Medications", _MEDICATIONS)
    _draw_list_section(pdf, "Allergies", _ALLERGIES)
    _draw_list_section(pdf, "Family History", _FAMILY_HISTORY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


if __name__ == "__main__":
    generate()

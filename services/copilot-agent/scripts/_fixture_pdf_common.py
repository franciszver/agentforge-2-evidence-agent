"""Shared helpers for the synthetic PDF fixture generators
(``scripts/generate_lab_pdf_fixture.py``, P3.1;
``scripts/generate_intake_form_fixture.py``, P3.2).

Not imported by any ``app/`` code -- these are test-fixture authoring
helpers only, mirroring the ``dev``-extra-only status of ``fpdf2`` itself
(see either generator script's module docstring).
"""

from __future__ import annotations

import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Pinned (not "now") so regenerating a fixture is byte-stable -- fpdf2 stamps
# the PDF's /CreationDate metadata with the real wall-clock time by default,
# which would otherwise make every regeneration produce a spuriously
# different committed file even with identical visible content.
PINNED_CREATION_DATE = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def draw_title(pdf: FPDF, text: str) -> None:
    """Render a bold 14pt title line, then drop to the next line."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def redact_rect(pdf: FPDF, x: float, y: float, width: float, height: float) -> None:
    """Draw an opaque black rectangle over ``(x, y, width, height)`` --
    simulates a genuinely unreadable/obscured field on a real scan, the
    deliberate "not found" case each fixture exercises."""
    pdf.set_fill_color(0, 0, 0)
    pdf.rect(x, y, width, height, style="F")

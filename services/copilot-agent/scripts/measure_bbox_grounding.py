"""Measures qwen2.5vl:7b's pixel-bbox grounding accuracy against a
realistic-scan-degraded fixture set (P3.9c, issue #42).

**Not wired into any test suite or CI job.** This is a maintenance/
measurement tool that requires a live Ollama instance serving
``qwen2.5vl:7b`` (not part of this repo's default model set -- see
``app/config.py``'s ``ollama_model``, ``qwen3:4b``). Run it manually,
by hand, when re-scoring bbox grounding against a candidate VLM.

## What it measures

P3.7 (`app/documents.py`) chose page-level source navigation over drawing
a pixel bounding box on the source image, based on qwen2.5vl:7b's bbox
grounding drifting off-target under scan-realistic degradation. This
script is the expanded, repeatable version of that probe:

1. **Fixture generation.** Builds the lab-report and intake-form
   fixtures' first pages (mirroring ``generate_lab_pdf_fixture.py`` /
   ``generate_intake_form_fixture.py``'s layout exactly), instruments the
   cell-drawing code to record each target field's ground-truth bounding
   box in mm at PDF-draw time, renders to PNG at the same scale
   ``app/ingestion.py`` uses (``pypdfium2``, ``scale=2.0``), then applies
   11 realistic-scan degradation axes (rotation, Gaussian noise,
   salt-and-pepper noise, JPEG recompression, brightness/contrast shift,
   Gaussian blur, a photocopier toner-fade band, and a combo of several)
   to produce 16 total fixture variants.

2. **Ground-truth transform.** Only rotation moves pixel content; the
   forward point-transform used to carry each ground-truth box through a
   rotation is verified independently (see ``rotation_selftest`` below --
   the same self-test used to validate this script's rotation math before
   trusting it for scoring).

3. **VLM probe.** Each field is asked of qwen2.5vl:7b via the same Ollama
   ``POST /api/chat`` surface and the same JSON-schema-constrained-decoding
   mechanism ``OllamaClient.extract()`` uses in production (``format`` set
   to a JSON schema) -- not free-form chat, which produces malformed JSON
   on this model often enough to itself be a confound on a grounding-
   accuracy measurement.

4. **Scoring.** Each response is scored against the ground-truth box by
   IoU (intersection-over-union) and center-in-truth-box (whether the
   predicted box's center point falls inside the true box).

See ``docs/W2_ARCHITECTURE.md``'s "Pixel bbox citation grounding" section
for the recorded results and verdict this script produced.

Usage (from ``services/copilot-agent/``, against a reachable Ollama
instance serving ``qwen2.5vl:7b``, e.g. inside the dev container where
Ollama is reachable at ``http://ollama:11434``):

    python scripts/measure_bbox_grounding.py --out-dir /tmp/bbox_probe

Runs fixture generation, then the VLM probe, then prints a summary and
writes ``manifest.json`` + ``results.json`` to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image, ImageEnhance, ImageFilter
import pypdfium2 as pdfium

from _fixture_pdf_common import PINNED_CREATION_DATE

# Matches app/ingestion.py's _RENDER_SCALE -- the fixtures must be rendered
# at the same scale production ingestion uses, or the probe would not be
# measuring the grounding task the product actually poses to the model.
_RENDER_SCALE = 2.0
_MM_TO_PT = 72.0 / 25.4
_PT_TO_PX = _RENDER_SCALE  # pdfium renders at _RENDER_SCALE px per pt

_OLLAMA_CHAT_PATH = "/api/chat"
_MODEL = "qwen2.5vl:7b"

_BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "x0": {"type": "integer"},
        "y0": {"type": "integer"},
        "x1": {"type": "integer"},
        "y1": {"type": "integer"},
    },
    "required": ["found", "x0", "y0", "x1", "y1"],
}

_PROMPT_TEMPLATE = (
    "This is a scanned document page, {width}x{height} pixels "
    "(x grows right, y grows down, origin top-left corner of the image).\n"
    'Find the field "{label}" on this page and report the pixel bounding '
    "box that TIGHTLY encloses ONLY its VALUE text (not the label itself). "
    "(x0,y0) is the top-left corner and (x1,y1) is the bottom-right corner "
    'of that box, in pixel coordinates of THIS image. Set "found" to true '
    "if you located it."
)


def _mm_box_to_px(x_mm: float, y_mm: float, w_mm: float, h_mm: float) -> list[float]:
    x_pt, y_pt = x_mm * _MM_TO_PT, y_mm * _MM_TO_PT
    w_pt, h_pt = w_mm * _MM_TO_PT, h_mm * _MM_TO_PT
    return [x_pt * _PT_TO_PX, y_pt * _PT_TO_PX, (x_pt + w_pt) * _PT_TO_PX, (y_pt + h_pt) * _PT_TO_PX]


def rotation_selftest() -> None:
    """Verify the forward point-transform formula for PIL's
    ``Image.rotate(angle, expand=True)`` matches PIL's actual pixel
    remapping, before trusting it to carry ground-truth bboxes through a
    rotation degradation. Raises ``AssertionError`` on mismatch."""
    for angle in (1.2, 5, -3, 15):
        w0, h0 = 400, 300
        img = Image.new("RGB", (w0, h0), "white")
        px, py = 300, 80
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                img.putpixel((px + dx, py + dy), (0, 0, 0))
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
        w1, h1 = rotated.size
        pred_x, pred_y = _transform_point_for_rotation(px, py, w0, h0, w1, h1, angle)

        gray = rotated.convert("L")
        pixels = gray.load()
        dark_points = [(x, y) for x in range(w1) for y in range(h1) if pixels[x, y] < 100]
        actual_x = sum(x for x, _ in dark_points) / len(dark_points)
        actual_y = sum(y for _, y in dark_points) / len(dark_points)
        err_x, err_y = pred_x - actual_x, pred_y - actual_y
        assert abs(err_x) < 2.0 and abs(err_y) < 2.0, (
            f"rotation point-transform self-test failed at angle={angle}: "
            f"predicted=({pred_x:.2f},{pred_y:.2f}) actual=({actual_x:.2f},{actual_y:.2f})"
        )


# ---------------------------------------------------------------------------
# Fixture generation: lab-report and intake-form page 1, instrumented to
# record ground-truth field bboxes. Mirrors generate_lab_pdf_fixture.py /
# generate_intake_form_fixture.py's layout exactly (same cell sizes/
# positions) so the measured document matches production ingestion.
# ---------------------------------------------------------------------------
_LAB_PAGE_1_ROWS = [
    ("Hemoglobin A1c", "5.4", "%", "4.0-5.6", "2026-06-01", "N"),
    ("Fasting Glucose", "142", "mg/dL", "70-99", "2026-06-01", "H"),
    ("Total Cholesterol", "188", "mg/dL", "<200", "2026-06-01", "N"),
    ("LDL Cholesterol", "112", "mg/dL", "<100", "2026-06-01", "H"),
]
_LAB_ROW_HEIGHT = 8
_LAB_COL_WIDTHS = [45, 20, 18, 28, 30, 12]
_LAB_HEADERS = ["Test", "Value", "Unit", "Reference Range", "Collection Date", "Flag"]

_INTAKE_DEMOGRAPHICS = [
    ("Name", "Test Patient"),
    ("DOB", "1990-01-01"),  # redacted in the real fixture -- skip as a ground-truth target
    ("Sex", "F"),
    ("MRN", "TEST-000000"),
]
_INTAKE_ROW_HEIGHT = 8
_INTAKE_LABEL_WIDTH = 30
_INTAKE_VALUE_WIDTH = 70


def _build_lab_report_page1() -> tuple[bytes, list[dict[str, Any]]]:
    """Returns (pdf_bytes, fields), fields carrying each row's Value-column
    ground-truth bbox in clean-render pixel space."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_creation_date(PINNED_CREATION_DATE)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Synthetic Lab Report (test fixture -- not a real patient)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Patient: Test Patient", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "DOB: 1990-01-01 (synthetic)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "MRN: TEST-000000", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    fields: list[dict[str, Any]] = []
    pdf.set_font("Helvetica", "B", 9)
    for header, width in zip(_LAB_HEADERS, _LAB_COL_WIDTHS):
        pdf.cell(width, _LAB_ROW_HEIGHT, header, border=1)
    pdf.ln(_LAB_ROW_HEIGHT)

    pdf.set_font("Helvetica", "", 9)
    for row in _LAB_PAGE_1_ROWS:
        row_x0, row_y0 = pdf.get_x(), pdf.get_y()
        col_x = row_x0
        for col_idx, (value, width) in enumerate(zip(row, _LAB_COL_WIDTHS)):
            if col_idx == 1:  # the Value column -- the ground-truth target
                fields.append(
                    {
                        "label": row[0],
                        "true_value": value,
                        "bbox_mm": [col_x, row_y0, width, _LAB_ROW_HEIGHT],
                    }
                )
            pdf.cell(width, _LAB_ROW_HEIGHT, value, border=1)
            col_x += width
        pdf.ln(_LAB_ROW_HEIGHT)

    buf = io.BytesIO()
    pdf.output(buf)
    for f in fields:
        f["bbox_px"] = _mm_box_to_px(*f.pop("bbox_mm"))
    return buf.getvalue(), fields


def _build_intake_form_page1() -> tuple[bytes, list[dict[str, Any]]]:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_creation_date(PINNED_CREATION_DATE)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        0, 10, "Synthetic Patient Intake Form (test fixture -- not a real patient)",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(2)

    fields: list[dict[str, Any]] = []
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Demographics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    for label, value in _INTAKE_DEMOGRAPHICS:
        row_x0, row_y0 = pdf.get_x(), pdf.get_y()
        pdf.cell(_INTAKE_LABEL_WIDTH, _INTAKE_ROW_HEIGHT, f"{label}:", border=1)
        value_x0 = row_x0 + _INTAKE_LABEL_WIDTH
        if label != "DOB":  # DOB is redacted in the real fixture -- skip as a ground-truth target
            fields.append(
                {
                    "label": label,
                    "true_value": value,
                    "bbox_mm": [value_x0, row_y0, _INTAKE_VALUE_WIDTH, _INTAKE_ROW_HEIGHT],
                }
            )
        pdf.cell(_INTAKE_VALUE_WIDTH, _INTAKE_ROW_HEIGHT, value, border=1)
        pdf.ln(_INTAKE_ROW_HEIGHT)
    pdf.ln(4)

    buf = io.BytesIO()
    pdf.output(buf)
    for f in fields:
        f["bbox_px"] = _mm_box_to_px(*f.pop("bbox_mm"))
    return buf.getvalue(), fields


def _render_pdf_page_to_png(pdf_bytes: bytes) -> bytes:
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        page = pdf.get_page(0)
        bitmap = page.render(scale=_RENDER_SCALE)
        buf = io.BytesIO()
        bitmap.to_pil().save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()


# ---------------------------------------------------------------------------
# Degradation axes -- realistic-scan-like transforms. Only rotation moves
# ground-truth pixel coordinates; the rest are pixel-value-only.
# ---------------------------------------------------------------------------
def _transform_point_for_rotation(
    x: float, y: float, w0: float, h0: float, w1: float, h1: float, angle_deg: float
) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    cx0, cy0 = w0 / 2.0, h0 / 2.0
    cx1, cy1 = w1 / 2.0, h1 / 2.0
    xc, yc = x - cx0, y - cy0
    nx = xc * math.cos(theta) + yc * math.sin(theta)
    ny = -xc * math.sin(theta) + yc * math.cos(theta)
    return nx + cx1, ny + cy1


def _rotate_variant(
    img: Image.Image, fields: list[dict[str, Any]], angle_deg: float
) -> tuple[Image.Image, list[dict[str, Any]]]:
    w0, h0 = img.size
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, expand=True, fillcolor="white")
    w1, h1 = rotated.size
    new_fields = []
    for f in fields:
        x0, y0, x1, y1 = f["bbox_px"]
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        tx = [_transform_point_for_rotation(cx, cy, w0, h0, w1, h1, angle_deg) for cx, cy in corners]
        xs, ys = [p[0] for p in tx], [p[1] for p in tx]
        new_fields.append({**f, "bbox_px": [min(xs), min(ys), max(xs), max(ys)]})
    return rotated, new_fields


def _add_gaussian_noise(img: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = img.convert("RGB")
    w, h = img.size
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            pixels[x, y] = (
                min(255, max(0, int(r + rng.gauss(0, sigma)))),
                min(255, max(0, int(g + rng.gauss(0, sigma)))),
                min(255, max(0, int(b + rng.gauss(0, sigma)))),
            )
    return img


def _add_salt_pepper_noise(img: Image.Image, amount: float, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = img.convert("RGB")
    w, h = img.size
    pixels = img.load()
    for _ in range(int(w * h * amount)):
        x, y = rng.randrange(w), rng.randrange(h)
        pixels[x, y] = (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)
    return img


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _photocopy_band(img: Image.Image, seed: int) -> Image.Image:
    """Simulate a horizontal photocopier toner-fade band."""
    img = img.convert("RGB")
    w, h = img.size
    rng = random.Random(seed)
    band_top = rng.randint(int(h * 0.2), int(h * 0.5))
    band_height = int(h * 0.15)
    overlay = Image.new("RGB", (w, band_height), (235, 235, 235))
    faded = Image.blend(img.crop((0, band_top, w, band_top + band_height)), overlay, 0.35)
    img.paste(faded, (0, band_top))
    return img


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


_VARIANTS_LAB: list[tuple[str, Any]] = [
    ("clean", lambda img, f: (img, f)),
    ("rotate_0.5deg", lambda img, f: _rotate_variant(img, f, 0.5)),
    ("rotate_1.2deg", lambda img, f: _rotate_variant(img, f, 1.2)),
    ("rotate_2.0deg", lambda img, f: _rotate_variant(img, f, 2.0)),
    ("rotate_neg1.5deg", lambda img, f: _rotate_variant(img, f, -1.5)),
    ("gaussian_noise", lambda img, f: (_add_gaussian_noise(img, 18, 1), f)),
    ("salt_pepper_noise", lambda img, f: (_add_salt_pepper_noise(img, 0.01, 2), f)),
    ("jpeg_q30", lambda img, f: (_jpeg_roundtrip(img, 30), f)),
    (
        "brightness_contrast",
        lambda img, f: (ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.25)).enhance(1.3), f),
    ),
    ("gaussian_blur", lambda img, f: (img.filter(ImageFilter.GaussianBlur(1.2)), f)),
    ("photocopy_band", lambda img, f: (_photocopy_band(img, 3), f)),
    (
        # Reproduces the original P3.7 finding's combo: rotation + noise + JPEG.
        "combo_scan_realistic",
        lambda img, f: (lambda rimg, rf: (_jpeg_roundtrip(_add_gaussian_noise(rimg, 14, 4), 40), rf))(
            *_rotate_variant(img, f, 1.2)
        ),
    ),
]

_VARIANTS_INTAKE: list[tuple[str, Any]] = [
    ("clean", lambda img, f: (img, f)),
    ("rotate_1.0deg", lambda img, f: _rotate_variant(img, f, 1.0)),
    (
        "combo_scan_realistic",
        lambda img, f: (lambda rimg, rf: (_jpeg_roundtrip(_add_gaussian_noise(rimg, 14, 5), 40), rf))(
            *_rotate_variant(img, f, 1.2)
        ),
    ),
    (
        "blur_brightness",
        lambda img, f: (ImageEnhance.Brightness(img.filter(ImageFilter.GaussianBlur(1.0))).enhance(1.2), f),
    ),
]


def build_fixtures(out_dir: Path) -> list[dict[str, Any]]:
    """Generates the 16 degraded fixture variants + ground-truth manifest,
    writing PNGs under ``out_dir/variants`` and returning the manifest
    (also written to ``out_dir/manifest.json``)."""
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    lab_pdf, lab_fields = _build_lab_report_page1()
    lab_img = Image.open(io.BytesIO(_render_pdf_page_to_png(lab_pdf))).convert("RGB")
    for name, transform in _VARIANTS_LAB:
        variant_img, variant_fields = transform(lab_img.copy(), lab_fields)
        out_path = variants_dir / f"lab_{name}.png"
        out_path.write_bytes(_to_png_bytes(variant_img))
        manifest.append(
            {
                "document": "lab_report",
                "variant": name,
                "file": out_path.name,
                "image_size": list(variant_img.size),
                "fields": variant_fields,
            }
        )

    intake_pdf, intake_fields = _build_intake_form_page1()
    intake_img = Image.open(io.BytesIO(_render_pdf_page_to_png(intake_pdf))).convert("RGB")
    for name, transform in _VARIANTS_INTAKE:
        variant_img, variant_fields = transform(intake_img.copy(), intake_fields)
        out_path = variants_dir / f"intake_{name}.png"
        out_path.write_bytes(_to_png_bytes(variant_img))
        manifest.append(
            {
                "document": "intake_form",
                "variant": name,
                "file": out_path.name,
                "image_size": list(variant_img.size),
                "fields": variant_fields,
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# VLM probe + scoring
# ---------------------------------------------------------------------------
def _ask_bbox(
    client: httpx.Client, ollama_base_url: str, image_b64: str, width: int, height: int, label: str
) -> dict[str, Any]:
    """POSTs to Ollama's ``/api/chat`` with ``format`` set to a JSON schema
    -- the same schema-constrained-decoding mechanism
    ``OllamaClient.extract()`` uses in production -- rather than free-form
    chat, which produces malformed JSON on this model often enough to
    itself be a confound on a grounding-accuracy measurement."""
    prompt = _PROMPT_TEMPLATE.format(width=width, height=height, label=label)
    body = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "format": _BBOX_SCHEMA,
        "options": {"temperature": 0},
    }
    resp = client.post(f"{ollama_base_url}{_OLLAMA_CHAT_PATH}", json=body, timeout=120.0)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        return {
            "found": bool(parsed.get("found", True)),
            "x0": float(parsed["x0"]),
            "y0": float(parsed["y0"]),
            "x1": float(parsed["x1"]),
            "y1": float(parsed["y1"]),
        }
    except (ValueError, KeyError, TypeError):
        return {"_raw": content[:500]}


def _box_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_in_box(pred: dict[str, Any], truth: list[float]) -> bool:
    cx, cy = (pred["x0"] + pred["x1"]) / 2, (pred["y0"] + pred["y1"]) / 2
    return truth[0] <= cx <= truth[2] and truth[1] <= cy <= truth[3]


def run_probe(manifest: list[dict[str, Any]], out_dir: Path, ollama_base_url: str) -> list[dict[str, Any]]:
    """Sends every field-level bbox request in ``manifest`` to the VLM,
    scores each response, writes ``out_dir/results.json``, and returns the
    results."""
    variants_dir = out_dir / "variants"
    results: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for entry in manifest:
            image_b64 = base64.b64encode((variants_dir / entry["file"]).read_bytes()).decode("ascii")
            width, height = entry["image_size"]
            for field in entry["fields"]:
                truth = field["bbox_px"]
                t0 = time.time()
                pred = _ask_bbox(client, ollama_base_url, image_b64, width, height, field["label"])
                elapsed = round(time.time() - t0, 1)
                record: dict[str, Any] = {
                    "document": entry["document"],
                    "variant": entry["variant"],
                    "label": field["label"],
                    "true_value": field["true_value"],
                    "truth_bbox": truth,
                    "pred": pred,
                    "elapsed_s": elapsed,
                }
                if "_raw" not in pred:
                    pred_box = [pred["x0"], pred["y0"], pred["x1"], pred["y1"]]
                    record["iou"] = round(_box_iou(pred_box, truth), 3)
                    record["center_in_truth"] = _center_in_box(pred, truth)
                else:
                    record["iou"] = 0.0
                    record["center_in_truth"] = False
                results.append(record)
                print(
                    f"{entry['document']:12s} {entry['variant']:22s} {field['label']:20s} "
                    f"iou={record['iou']:.2f} center_in={record['center_in_truth']} ({elapsed:.1f}s)",
                    flush=True,
                )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_doc[r["document"]].append(r)

    print(f"\n{len(results)} field-level bbox requests scored:\n")
    for doc, rows in by_doc.items():
        n = len(rows)
        center_hits = sum(1 for r in rows if r["center_in_truth"])
        mean_iou = sum(r["iou"] for r in rows) / n
        max_iou = max(r["iou"] for r in rows)
        print(
            f"  {doc}: n={n} center-in-truth-box={center_hits}/{n} "
            f"({center_hits / n:.0%}) mean_iou={mean_iou:.3f} max_iou={max_iou:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/bbox_probe"), help="Directory for fixtures + results."
    )
    parser.add_argument(
        "--ollama-base-url", default="http://ollama:11434", help="Origin of the Ollama instance to probe."
    )
    parser.add_argument(
        "--skip-selftest", action="store_true", help="Skip the rotation point-transform self-test."
    )
    args = parser.parse_args()

    if not args.skip_selftest:
        rotation_selftest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_fixtures(args.out_dir)
    print(f"Built {len(manifest)} fixture variants -> {args.out_dir}")

    results = run_probe(manifest, args.out_dir, args.ollama_base_url)
    print(f"\nWrote {len(results)} results -> {args.out_dir / 'results.json'}")
    print_summary(results)


if __name__ == "__main__":
    main()

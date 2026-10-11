"""Publication layout for the qualitative viewer's captured PNGs."""

from io import BytesIO
import csv
from math import ceil
from pathlib import Path

from PIL import Image

from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas


# Reading order and labels in drug-qual-example-square.pdf. Keep the internal
# method names intact for image lookup and the per-ligand capture metadata.
COMPARISON_ORDER = (
    "Reference", "AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "TargetDiff",
    "FuncBind", "VoxBind", "VoxBind + Ours",
)

# Title chips take each method's approved palette Color, mixed toward white so the
# black label stays legible (AR's navy is too dark at full strength). Every chip
# on a sheet is as wide as its widest label plus TITLE_PAD, centred on the panel.
PALETTE = Path(__file__).resolve().parents[3] / "figures" / "fig-palette" / "palette-methods.csv"
PALETTE_KEY = {"Reference": "Reference ligand", "VoxBind + Ours": "CoDE"}
TITLE_TINT, TITLE_PAD = 0.4, 10
TITLE_SIZE = 10


def _title_color(method):
    with PALETTE.open(newline="") as handle:
        rows = {row["method"]: row["color_hex"] for row in csv.DictReader(handle)}
    value = rows.get(PALETTE_KEY.get(method, method))
    if value is None:
        return None
    return [1 - TITLE_TINT * (1 - int(value[i:i + 2], 16) / 255) for i in (1, 3, 5)]


def _title_runs(method, pocket_code):
    if method == "Reference":
        return [("Reference ", "Helvetica-Bold"), (pocket_code, "Times-Bold")]
    if method == "VoxBind + Ours":
        return [("CoDE", "Times-Bold"), (" (Ours)", "Helvetica-Bold")]
    return [(method, "Helvetica-Bold")]


def _runs_width(runs, size):
    return sum(stringWidth(text, font, size) for text, font in runs)


def _title(canvas, method, pocket_code, center, baseline, width, chip_width):
    runs = _title_runs(method, pocket_code)
    size = TITLE_SIZE
    text_width = _runs_width(runs, size)
    size *= min(1, (width - 4) / text_width)
    text_width = _runs_width(runs, size)
    color = _title_color(method)
    if color is not None:
        canvas.saveState()
        canvas.setFillColorRGB(*color)
        canvas.roundRect(center - chip_width / 2, baseline - 3.2, chip_width, 12.6, 2.5,
                         stroke=0, fill=1)
        canvas.restoreState()
    text_object = canvas.beginText(center - text_width / 2, baseline)
    for text, font in runs:
        text_object.setFont(font, size)
        text_object.textOut(text)
    canvas.drawText(text_object)


def comparison_pdf(files, metadata, *, image_dpi=400):
    """Build one sheet from this capture's images and server-verified scores.

    The default nine panels reproduce the reference's 324 × 371 pt page.
    Embedded images are downsampled to 400 dpi with alpha preserved; labels
    remain vector text on palette-tinted chips. Source PNGs are untouched. Set image_dpi=None to
    embed the original pixels, or use a higher dpi for enlarged printing.
    Custom method subsets use the same panel size in up to three columns.
    """
    if image_dpi is not None and image_dpi <= 0:
        raise ValueError("image_dpi must be positive or None.")
    records = metadata["methods"]
    methods = [method for method in COMPARISON_ORDER if method in records]
    methods += [method for method in records if method not in COMPARISON_ORDER]
    if not methods:
        raise ValueError("A comparison requires at least one panel.")
    columns = min(3, len(methods))
    rows = ceil(len(methods) / columns)
    margin, gap, header, row_gap = 1, 10, 21, 2
    panel_size = (324 - 2 * margin - 2 * gap) / 3
    width = 2 * margin + columns * panel_size + (columns - 1) * gap
    height = 2 * margin + rows * (panel_size + header) + (rows - 1) * row_gap
    output = BytesIO()
    canvas = Canvas(output, pagesize=(width, height), pageCompression=1)
    canvas.setTitle(f"Pocket {metadata['pocket_code']} — qualitative comparison")
    chip_width = min(panel_size, TITLE_PAD + max(
        _runs_width(_title_runs(method, metadata["pocket_code"]), TITLE_SIZE) for method in methods))
    canvas.setSubject(f"{metadata['energy_key']} ({metadata['energy_unit']}); {metadata['updated_at']}")
    for index, method in enumerate(methods):
        record = records[method]
        row, column = divmod(index, columns)
        x = margin + column * (panel_size + gap)
        top = height - margin - row * (panel_size + header + row_gap)
        y = top - header - panel_size
        canvas.setFillColorRGB(0, 0, 0)
        _title(canvas, method, metadata["pocket_code"], x + panel_size / 2, top - 13, panel_size,
               chip_width)
        if record["status"] == "unavailable":
            canvas.setFillColorRGB(0.45, 0.45, 0.45)
            canvas.setFont("Helvetica", 7)
            lines = simpleSplit(record.get("reason") or "Unavailable", "Helvetica", 7, panel_size - 12)
            for line_index, line in enumerate(lines):
                canvas.drawCentredString(x + panel_size / 2, y + panel_size / 2 - line_index * 9, line)
            continue
        with Image.open(BytesIO(files[record["files"]["png"]])) as source:
            image = source.convert("RGBA")
            if image_dpi is not None:
                pixels = max(1, ceil(panel_size * image_dpi / 72))
                # Pillow < 9.1 exposes resampling filters directly on Image.
                image.thumbnail((pixels, pixels), getattr(Image, "Resampling", Image).LANCZOS)
            canvas.drawImage(ImageReader(image), x, y,
                             width=panel_size, height=panel_size, mask="auto")
        energy = record.get("energy")
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawRightString(x + panel_size - 8, y + 6,
                               f"{energy:.2f}" if energy is not None else "N/A")
    canvas.showPage()
    canvas.save()
    return output.getvalue()

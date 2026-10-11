"""fig-palette: the official method palette sheet (figures/fig-palette/palette-methods.*).

    python figures/draw.py fig-palette-methods

The palette itself (method-palette-approved.json) is read by draw.py; this module only
draws the swatch sheet of it. PAL_ORDER / PAL_EXTRA are the sheet's row order.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

from draw import (
    METHOD_ORDER, METHOD_PALETTE_ROWS, RC, figure, save, write_csv,
)

# ══════════════════════════════════════════════════════════════════════════════
# fig-palette-methods — official Color / Highlight sheet from the shared palette
# ══════════════════════════════════════════════════════════════════════════════
PAL_ORDER = ["Reference ligand", *METHOD_ORDER]
PAL_EXTRA = ["CoE", "DeepDTA", "ProFSA", "Nesso-1", "Ours v2", "FuncBind ft 3.17M",
             "FuncBind ft 8.21M", "FuncBind ft 26.1M"]


@figure("fig-palette-methods", folder="fig-palette")
def draw_palette_methods(out):
    """Official Color and Highlight tones; unused variants have black hatching."""
    rows = METHOD_PALETTE_ROWS
    if not rows:
        raise ValueError("The official palette sheet requires the approved palette, not legacy mode")
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8.0, .42 * len(rows) + 1.55), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_position([.035, .045, .93, .84])
        ax.set_xlim(-2.1, 2.65)
        ax.set_ylim(-len(rows) + .05, 1.1)
        ax.axis("off")
        fig.text(.5, .974, "Method palette", ha="center", fontsize=17, weight="bold")
        fig.text(.5, .944, "Official palette · Color / Highlight", ha="center",
                 fontsize=10, color="#555555")
        for j, (field, title) in enumerate((("color", "Color"), ("highlight", "Highlight"))):
            x = j * 1.35
            ax.text(x + .50, .75, title, ha="center", fontsize=14, weight="bold")
            for i, row in enumerate(rows):
                y = -i - .2
                if j == 0:
                    ax.text(-.18, y, row["label"], ha="right", va="center",
                            fontsize=12, color="#222222")
                hatch = row.get("hatch", "")
                ax.add_patch(matplotlib.patches.Rectangle(
                    (x, y - .17), 1.0, .50, facecolor=row[field],
                    edgecolor=row.get("hatch_color") or "none", hatch=hatch or None, linewidth=0))
                ax.text(x + .50, y - .43, "Black hatch" if hatch else row[field],
                        ha="center", va="center", fontsize=9, color="black")
        with plt.rc_context({"savefig.bbox": "tight", "savefig.pad_inches": .10, "savefig.dpi": 220}):
            save(fig, out, "palette_methods")
    write_csv(out, "palette_methods",
              ["method", "color_hex", "highlight_hex", "hatch", "hatch_color", "legend_label"],
              [[r["method"], r["color"], r["highlight"], r.get("hatch", ""),
                r.get("hatch_color", ""), r["label"]] for r in rows])
    print(f"  {len(rows)} methods · official Color / Highlight · wrote {out}")

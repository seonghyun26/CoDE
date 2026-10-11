"""fig-legend: the 1 figure drawn into figures/fig-legend/.

    python figures/draw.py fig-posecheck-strain-legend

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-posecheck-strain-legend                  -> fig-legend/
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import draw                      # CLI options draw.main() sets at run time
from draw import (
    ALIASES, AXIS_LW, INK, LEGEND_EDGE, PCSZ_LINE_LW, PCSZ_LW_SCALE, PCSZ_RC,
    PCSZ_REF_LW, RB_ORDER, REF_COLOR, REF_LABEL, RF_BOX_AXIS_LW, RF_BOX_LEGEND_FS,
    dash, display, figure, legend, pale, save, use_style,
)

# Shorter swatches with larger text; retain enough length to show dashed styles.
PLEG_HANDLE_LEN = 2.6
PLEG_FONT_SIZE = 13.0
PLEG_HORIZONTAL_LINE_WIDTH = 6.6


def _pleg_method_handles():
    out = []
    for lab in RB_ORDER:
        out.append(Line2D([], [], color=pale(lab), ls=dash(lab),
                          lw=PCSZ_LINE_LW[ALIASES.get(lab, lab)] * PCSZ_LW_SCALE,
                          solid_capstyle="round", label=display(lab)))
    return out


def _pleg_legend(fig, handles, **kw):
    """legend()'s look -- white face, the axis-pen frame, INK text -- with a swatch long
    enough to show a dash."""
    bold = kw.pop("bold", False)
    leg = fig.legend(handles=handles, frameon=True,
                     prop={"size": PLEG_FONT_SIZE, "weight": "bold" if bold else "normal"},
                     handlelength=PLEG_HANDLE_LEN,
                     handletextpad=0.6, labelspacing=kw.pop("labelspacing", 0.3),
                     borderpad=0.4, borderaxespad=0.39,
                     facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0, **kw)
    leg.get_frame().set_linewidth(AXIS_LW)
    leg.set_zorder(7)
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


@figure("fig-posecheck-strain-legend", folder="fig-legend")
def draw_posecheck_strain_legend(out, layout=None):
    """Standalone method key, in one horizontal row or one vertical column."""
    layout = draw.PLEG_LAYOUT if layout is None else layout
    if layout not in ("horizontal", "vertical"):
        raise ValueError("layout must be horizontal or vertical")
    use_style()
    ref = Line2D([], [], color=REF_COLOR, ls=dash(REF_LABEL),
                 lw=PCSZ_REF_LW * PCSZ_LW_SCALE,
                 dash_capstyle="round", label="Reference")
    handles = [ref] + _pleg_method_handles()
    vertical = layout == "vertical"
    if not vertical:
        for handle, label in zip(handles, [REF_LABEL] + list(RB_ORDER)):
            old_width = handle.get_linewidth()
            handle.set_linewidth(PLEG_HORIZONTAL_LINE_WIDTH)
            # Preserve dash lengths when changing horizontal swatch thickness.
            style = dash(label)
            if isinstance(style, tuple):
                offset, pattern = style
                if pattern is not None:
                    ratio = old_width / PLEG_HORIZONTAL_LINE_WIDTH
                    handle.set_linestyle((offset * ratio, tuple(v * ratio for v in pattern)))
    with plt.rc_context(PCSZ_RC):
        fig = plt.figure(figsize=(2.65, 4.2) if vertical else (20.0, 0.75))
        fig.patch.set_facecolor("white")
        if vertical:
            _pleg_legend(fig, handles, loc="center", ncol=1,
                         labelspacing=1.05, columnspacing=1.5, bold=False)
        else:
            # Match the rigid-consistency key's typography, frame and compact row,
            # using line handles so the shared dash patterns remain visible.
            key = legend(fig, handles, loc="center", ncol=len(handles),
                         fontsize=RF_BOX_LEGEND_FS, columnspacing=0.65)
            key.get_frame().set_linewidth(RF_BOX_AXIS_LW)
            fig.canvas.draw()
            bounds = key.get_window_extent(fig.canvas.get_renderer())
            fig.set_size_inches(bounds.width / fig.dpi + 0.12,
                                bounds.height / fig.dpi + 0.12)
        stem = layout
        save(fig, out, stem)
    print(f"  {layout} key: Reference + {len(RB_ORDER)} methods in one "
          + ("column" if vertical else "row"))

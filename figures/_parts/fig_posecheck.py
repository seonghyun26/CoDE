"""fig-posecheck: the 20 figures drawn into figures/fig-posecheck/.

    python figures/draw.py interaction posecheck redock

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-interaction-contacts                     -> fig-posecheck/interaction/
    fig-interaction-hbonds                       -> fig-posecheck/interaction/
    fig-interaction-legend                       -> fig-posecheck/interaction/
    fig-posecheck-legend                         -> fig-posecheck/
    fig-interaction-pair                         -> fig-posecheck/interaction/
    fig-interaction-pair-stacked                 -> fig-posecheck/interaction/
    fig-posecheck-strain-ecdf-by-size            -> fig-posecheck/strain-energy/
    fig-posecheck-clash-violin-by-size           -> fig-posecheck/clash/
    fig-posecheck-clash-box-by-method            -> fig-posecheck/clash/
    fig-posecheck-clash-per-atom-all-methods     -> fig-posecheck/clash/
    fig-posecheck-strain-per-atom                -> fig-posecheck/strain-energy/
    fig-posecheck-clash-per-atom                 -> fig-posecheck/clash/
    fig-posecheck-strain-per-rotbond             -> fig-posecheck/strain-energy/
    fig-posecheck-strain-box-per-rotbond         -> fig-posecheck/strain-energy/
    fig-posecheck-rotbond-distribution-all-methods -> fig-posecheck/strain-energy/
    fig-posecheck-strain-rotbond-all-methods     -> fig-posecheck/strain-energy/
    fig-posecheck-strain-per-atom-all-methods    -> fig-posecheck/strain-energy/
    fig-posecheck-strain-rotbond-grid            -> fig-posecheck/strain-energy/
    fig-redock-ecdf                              -> fig-posecheck/redock/
    fig-redock-median                            -> fig-posecheck/redock/
"""
from __future__ import annotations

import collections
import json
import math
import os
import statistics as st
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator

import draw                      # CLI options draw.main() sets at run time
from draw import (
    ALIASES, ARMS, AXIS, AXIS_LABEL_FS, AXIS_LW, AXIS_WIDTH, BIN_LABELS, CORE, DASH,
    DATA_ALPHA, DIST_FILL, DIST_LW, DOT, E, EDGES, FIG_W, GRID, GRID_LW, HEAT_CMAP,
    INK, LEGACY, LEGEND_EDGE, METHOD_ORDER, MIN_N, MODEL_LW, P79, PANEL_H, PCSZ_AXIS,
    PCSZ_BASELINES, PCSZ_BG, PCSZ_COMPARISON_MARGINS, PCSZ_COMPARISON_TICK_SCALE,
    PCSZ_ECDF_SIZE, PCSZ_INK, PCSZ_KEY_FACE, PCSZ_KEY_KW, PCSZ_LABEL_FS, PCSZ_LINE_LW,
    PCSZ_LOCAL, PCSZ_LW_SCALE, PCSZ_MAIN_ECDF_SIZE, PCSZ_MAIN_STRAIN_SIZE, PCSZ_RC,
    PCSZ_REF_LW, PCSZ_STRAIN_SIZE, PCSZ_YPAD, RB_ORDER, REF_COLOR, REF_LABEL, REF_LW,
    REF_ROOT, REF_WIN, REPO, SOLID, TITLE_SIZE, XTICK_STEP, X_LABEL,
    _pcsz_comparison_layout, _pcsz_ordered, _pcsz_style, arm_handles, arms_for,
    bin_of, by_size, dash, display, figure, fit, furniture, heat_colorbar, legacy,
    legend, model_curve, pale, pose_data, reference_curve, rows_of, save, use_style,
    variants, write_csv, x_range,
)

def core_only(pairs):
    """Just the `core` entry of a variants()-shaped sequence.

    For PoseCheck STRAIN, `all` is not eight methods -- only TargetDiff, VoxBind and CoDE
    carry a strain value at all, because the five published baselines were scored for
    PoseBusters here and their PoseCheck came back from the evaluation server as exports rather than as
    per-molecule rows. So `all` was `core` plus TargetDiff, a third figure that is neither
    the two-arm comparison nor the whole field, and the whole field already has its own
    figures: strain_per_atom_all_methods, strain_rotbond_all_methods and
    strain_box_per_rotbond_all_methods read those exports and draw all eight. Dropping the
    near-duplicate leaves one figure per question instead of three per question."""
    return [p for p in pairs if p[0] == "core"]


# ════════════════════════════════════════════════════════════════════════════════
# fig-interaction-{contacts,hbonds,pair} — ProLIF interaction fingerprints
# ════════════════════════════════════════════════════════════════════════════════
# The VoxBind paper's Fig. 14 form, and the fourth pose-quality measure of this section:
# strain and clashes are fig-posecheck-*, PoseBusters validity is fig-posebusters-*,
# rigid-fragment consistency is fig-consistency-*, and this is what the pose actually DOES
# against the pocket. Same 79 pockets, same arms, same colours.
#
# THE METRIC. An interaction fingerprint records, per pose, which contacts it makes with the
# receptor. PoseCheck runs ProLIF over the protonated pocket and the pose as generated; see
# Harris et al., 2023 (the PoseCheck paper) for the definitions. Four types appear in this
# data:
#
#     VdWContact    van der Waals contact — by far the most common, and mostly a size proxy
#     Hydrophobic   apolar contact
#     HBAcceptor    the ligand accepts a hydrogen bond from the protein
#     HBDonor       the ligand donates one to the protein
#
# MORE IS NOT AUTOMATICALLY BETTER, and this figure is not a leaderboard. A bigger ligand
# makes more of every type, so an arm that generates larger molecules scores higher on all
# four without binding better; the crystal ligands are the reference precisely because they
# are the distribution a real binder draws from, and the useful reading is which arm sits
# CLOSEST to them, not which sits highest. The per-molecule counts are exported so a
# size-matched comparison can be made — the same trap the Vina numbers have already been
# bitten by on this project, on a different metric.
#
# A TYPE A MOLECULE DOES NOT MAKE IS ABSENT FROM ITS FINGERPRINT, NOT ZERO IN IT. Every
# molecule scored by PoseCheck therefore contributes a 0 to every type it did not make, and
# the violins are over all scored molecules, not over the ones that happened to make that
# type. Counting only the latter would turn "Ours makes H-bond donors less often" into "Ours
# makes more of them when it does", which is a different and much smaller claim.
#
# THE POCKET IS THE pocket10 CROP that the target `metrics.json` files carry, the same
# source fig-posecheck-per-atom uses — NOT the whole-receptor scoring in
# `frozenenc_probes/posecheck_full/`. That scoring exists and covers all 79 pockets for all
# three arms, but it has `reference: null`, so it cannot draw the crystal ligands, which are
# the whole point of the comparison. The cost of using the crop was measured over the 21,865
# molecules the two scorings share rather than assumed: per molecule the crop finds 0.16-0.29
# fewer VdWContacts (1.8-3.2 %) and within 0.03 of the same count on all three other types.
# A 10 Å crop around the crystal ligand contains essentially everything within ProLIF's
# interaction cutoffs. Do not, however, read a number here against one from
# fig-posecheck-by-size, which IS whole-receptor.
#
# TWO PANEL FORMS, AND THE SPLIT IS NOT COSMETIC. VdW and hydrophobic contacts run to 30 and
# 12 per pose, so their distribution has a body a KDE can describe and they stay violins. THE
# TWO H-BOND TYPES DO NOT: they run 0-10 and 0-6, roughly half of every arm's molecules sit
# at exactly zero, and a KDE over that invents a smooth hump through counts that do not exist
# and a tail below zero. Those two are drawn as LETTER-VALUE PLOTS instead -- VoxBind's
# Fig. 14(a),(b). Its SVG measures as five nested boxes per column at exactly halving widths
# (21.39, 10.69, 5.35, 2.67, 1.34 pt), each box one quantile depth further out, all of them
# sitting on the axis because their lower letter values are 0. The BOX WIDTH carries the
# depth, not the box colour, which leaves the colour free to stay the method's own.
#
# STYLE. The shared furniture, and the violin construction of fig-posecheck-by-size: KDE
# body, a thick inter-quartile bar, a white median dot. The letter-value panels keep that
# same white median dot rather than boxenplot's median line, so the mark that says "here is
# the middle" does not change between panels of one figure.

# Names this family needs that the shared header does not import. Reached through the
# already-imported `matplotlib` package rather than re-imported, because a part file adds no
# import statements of its own.
IX_RECTANGLE = matplotlib.patches.Rectangle
IX_TO_RGB = matplotlib.colors.to_rgb
IX_FIXED_LOCATOR = matplotlib.ticker.FixedLocator
IX_FIXED_FORMATTER = matplotlib.ticker.FixedFormatter

# key -> panel name -> the form it is drawn in. Ordered by how common the type is, so the
# panel that is mostly a size proxy is read first and the specific ones after it. The two
# H-bond types are letter-value plots and the two contact types are violins; see the banner
# above for why the same family carries both.
IX_TYPES = [
    ("VdWContact",  "van der Waals contacts",         "violin"),
    ("Hydrophobic", "Hydrophobic contacts",           "violin"),
    # SHORT NAMES (2026-09-14). "H-bonds accepted by the ligand" did not survive the axis
    # name going to the ECDF's 16.1 pt: this family draws to a FIXED frame at dpi 220 with no
    # crop to the ink, so the tail of a long name falls off the canvas rather than pushing it
    # wider -- the rendered panels read "...by the liga" and "...by the ligar". The contacts
    # names were short enough to survive, which is why only these two were cut.
    # ONLY THE DISPLAY NAME CHANGES. The keys still drive IX_KINDS, the summary table and the
    # `type` column of interactions.csv, so nothing downstream moves.
    ("HBAcceptor",  "H-bonds acceptor",               "boxen"),
    ("HBDonor",     "H-bonds donor",                  "boxen"),
]
IX_TOTAL = "All interactions"
IX_KINDS = [k for k, _, _ in IX_TYPES] + [IX_TOTAL]
IX_TYPE = {k: (name, form) for k, name, form in IX_TYPES}

# TWO FIGURES, ONE ROW OF TWO PANELS EACH, and the split is the one the forms already make:
# the contacts are violins and the H-bonds are letter-value blocks. Four panels under one
# caption meant a reader comparing arms on H-bonds had two violins in the same eyeful,
# arguing for a different reading of the same data at a different scale. They are separate
# measurements and they are now separate figures.
IX_PARTS = [
    ("contacts", ("VdWContact", "Hydrophobic")),
    ("hbonds",   ("HBAcceptor", "HBDonor")),
]

# DISPLAY NAMES, for this family's figures only. REF_LABEL and ARMS carry the canonical
# spellings the whole section shares; renaming them there would move every other figure in
# it, so they are mapped here at draw time instead. Our arm is labelled \textsc{CoDE} -- the
# LaTeX form, verbatim, because these figures are placed in the paper and typeset there, and
# ALIASES already resolves that spelling so neither the colour nor the drug-design row order
# depends on it. IT RENDERS LITERALLY in the PNG and the PDF; the SVG keeps live text
# (svg.fonttype: none) and is the copy that is typeset.
# Only the crystal ligands get a name of this family's own; everything else goes through the
# shared display(), so a method is spelled here exactly as the ECDF and the clash box spell
# it. The entries that used to live here were r"\textsc{CoDE}" -- LaTeX that nothing in this
# pipeline renders, so the legend printed the six characters \textsc literally. It was
# invisible while the names lived only in a legend nobody re-read; it would have gone
# straight onto the x axis the moment the names moved there.
IX_DISPLAY = {REF_LABEL: "Reference"}
# The method names on the x axis. Smaller than the ECDF's axis-name size: eight of them share
# one panel here, where that figure spends its 16.1 on a single line of axis title.
IX_NAME_FS = 12.5
# WHICH FIGURE CARRIES THE NAMES (2026-09-14). Both figures draw the same methods in the same
# order, so stacked -- contacts over hbonds -- one set of names underneath serves both, and
# printing them twice is the same information twice. Only the bottom figure is named.
# THIS MAKES contacts DEPENDENT ON ITS NEIGHBOUR: on its own, nothing in it says which
# violin is which. If it is ever published alone, put "contacts" back in this tuple.
IX_NAMED_PARTS = ("hbonds",)

# The published baselines, when compute_baseline_interactions.py has written them. Their
# fingerprints are not in posecheck_<Method>.json -- that export carries only strain and
# clashes -- so they are recomputed from the poses in the results bundle, against the same
# pocket10 crop, by an interpreter checked to reproduce the local arms' stored fingerprints
# exactly; the figure-data builder that does it is not part of this repository.
IX_BASELINES = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff"]

# THE ROW ORDER OF 260827/table_drug_design.tex, top to bottom. Every figure and the CSV of
# this family use it, and none of them use a ranking or an order that falls out of how the
# data happened to load. The reader moves between that table and these panels; re-finding a
# method in a different order each time is a cost paid on every glance. The crystal ligands
# lead, exactly as the table's Reference row does. FuncBind is carried even though it has no
# fingerprint here, so adding it later moves nothing.
# FuncBind BEFORE VoxBind (2026-09-15), which is RB_ORDER's sequence and not the drug-design
# table's. The table puts VoxBind first and this family used to follow it, but rotbond and
# the shared posecheck legend both run RB_ORDER, so following the table here made the
# interaction figures the odd ones out in a document that stacks them together. Consistency
# between the FIGURES was chosen over consistency with the table; if the table is ever the
# thing to match, swap these two back and RB_ORDER with them, not this alone.
IX_ORDER = ("Reference ligand", *METHOD_ORDER)

# THE TYPE NAMES THE Y AXIS, and there are no panel titles -- which is the house style, and
# which this family had to break while it was one 2x2 figure: all four panels plotted the
# same quantity in the same unit, so the y axis could not carry the type as well. Split into
# two figures of two panels, each panel's y axis is free again and the titles are gone. The
# names are long for a rotated label, so they take their own size rather than furniture()'s
# 15.5.
IX_Y_SIZE = 13.0

# GEOMETRY. One row of two panels off the shared panel height, plus a strip beneath the
# axes for the legend, and the WIDTH FALLS OUT OF A 4:1 FIGURE ASPECT rather than out of
# FIG_W. Two panels side by side want a landscape frame: at the 3.4:2.4 the shared width
# gave them, eight columns were pressed together while the panel had height to spare above
# every violin.
#
# BOTH VARIANTS ARE THE SAME SIZE, which is what the single-row legend buys. The `all`
# variant's eight names needed two rows at the old width and made its figure 0.3 in taller
# than core's; at this width they fit on one, so core and all are the same frame and sit
# together in a document without a size change between them.
IX_ASPECT = 4.0
IX_ROW_H = PANEL_H * 1.06
IX_LEGEND_H = 0.46                               # inches reserved under the axes
IX_FIG_H = IX_ROW_H + IX_LEGEND_H
IX_FIG_W = IX_ASPECT * IX_FIG_H
IX_LEGEND_SIZE = 12.5
IX_VIOLIN_W = 0.78
IX_BODY_ALPHA = DATA_ALPHA
IX_IQR_LW, IX_MED_MS = 5.0, 5.5
# Head-room above the tallest mark for the per-category mean/median caption. Two lines of
# 10pt over a panel of this height need about 12 % of it, so 1.17 clears the caption with a
# little air and no more: at the 1.34 the old 2x2 figure used, a full-height panel left the
# caption floating a third of the axes above the data it describes.
IX_HEAD = 1.17
IX_HEAD_BARE = 1.06               # no caption to clear, so barely any head-room needed
# ONE ABSOLUTE KDE BANDWIDTH FOR EVERY VIOLIN, in counts, rather than matplotlib's default.
# The default (Scott) scales the bandwidth by n^(-1/5), and the crystal ligands are 79
# molecules against each arm's ~7,800 -- so the reference came out visibly smoother than
# every arm beside it for no reason but its sample size, which on a shape comparison is
# exactly the thing that must not differ. Fixed at 0.45 of a count, each integer still
# reads as its own bump.
IX_KDE_BW = 0.45
# LETTER-VALUE GEOMETRY, off VoxBind's Fig. 14 SVG: five nested boxes whose width halves at
# each step, so the outermost is 1/16 of the innermost. Depth d spans the middle
# 1 - 2^-(d+1) of the distribution -- the quartiles, then the eighths, and so on out to the
# 1.6th and 98.4th percentiles. Five is Fig. 14's count, and it is also all this data can
# carry: the counts are integers running 0-10, so deeper boxes would repeat a bound already
# drawn. The shade ramp goes dark at the middle to pale at the tails, which is boxenplot's,
# but here it is a tint of the METHOD's colour -- the depth is already in the width.
IX_BOXEN_K = 5
# HOW FAR TOWARD WHITE THE OUTERMOST BOX IS MIXED. Fig. 14 goes to 0.85 (#72b6a1 ->
# #eaf4f1), which on this palette left the two narrowest tiers invisible against the white
# ground -- and those tiers ARE the tail this figure is read for, since every arm's median
# is 0 or 1 and the whole difference between them lives above the quartiles. Held short of
# that, and shorter again since the blocks took the violins' alpha: IX_BODY_ALPHA already
# lightens every tier, so a wide ramp on top of it washed the pale end out entirely.
IX_BOXEN_LIGHT = 0.46
IX_BOXEN_EDGE_LW = 0.8            # every block is outlined, in the axis pen


def _interaction_ordered(labels):
    """`labels` in IX_ORDER. Raises on a label the table does not name rather than appending
    it, for the same reason pale() raises: a method that silently drifts to the end of
    every figure is worse than a failed build.

    Spellings resolve through ALIASES first, so the table's order survives a rename -- our
    method answers to "CoDE", "VoxBind + Ours", "Ours v1" and "ours_v1" alike, exactly as
    its colour does."""
    rank = {lab: i for i, lab in enumerate(IX_ORDER)}
    canon = {l: ALIASES.get(l, l) for l in labels}
    unknown = [l for l, c in canon.items() if c not in rank]
    if unknown:
        raise KeyError(f"not in the drug-design table's order: {', '.join(unknown)}")
    return sorted(labels, key=lambda l: rank[canon[l]])


def _interaction_baseline_rows(label):
    """Rows shaped like the pose loader's, from compute_baseline_interactions.py, or None if
    that step has not been run. Only `n` and `ifp` are used by this figure.

    Read off LEGACY directly rather than through legacy(): this input is OPTIONAL -- a
    baseline without a recompute is reported as a missing name at the end of the run, not as
    a failed build, because the figure it would join is complete without it."""
    path = LEGACY / "fig-interaction" / f"interactions_{label}.json"
    if not path.exists():
        return None
    doc = json.load(open(path))
    return [{"n": m["n"], "ifp": m["ifp"]} for m in doc["molecules"]]


def _interaction_counts(rows, kind):
    """Per-molecule count of one interaction type over every molecule with a fingerprint.
    A type the molecule did not make is absent from the dict and counts as 0 — see the
    banner above; this is the line that decides it."""
    if kind == IX_TOTAL:
        return [sum(r["ifp"].values()) for r in rows if r["ifp"] is not None]
    return [r["ifp"].get(kind, 0) for r in rows if r["ifp"] is not None]


def _interaction_display(label):
    """The name this family's figures print for an arm. See IX_DISPLAY."""
    return IX_DISPLAY.get(label) or display(label)


def _interaction_cell_axis(ax, reach):
    """A count axis whose NUMBER SITS IN THE MIDDLE OF ITS BLOCK rather than on the rule
    between two of them. Fig. 14's does the same -- its SVG puts the tick stubs at +0.5,
    +1.5, +2.5 ... of a count off the baseline, never on the integers.

    THE CELLS ARE NUMBERED FROM 0, because 0 is a real count here and the commonest one:
    more than half of most arms' poses donate no hydrogen bond at all. Cell k is the band
    [k, k+1] and its number sits at k+0.5. The dotted rules stay on the integers, where the
    block edges are: a rule through the middle of a block would cut the thing the reader is
    counting. So THE RULES ARE THE COUNTS and a number names its own cell's LOWER edge —
    the topmost number in a column is one less than the count that column reaches.

    Labels stop at the highest count anyone reached, NOT at the panel top: the top is
    head-room for the caption, and numbering empty air both wastes the reader's attention
    and makes the captioned and uncaptioned variants of the same panel carry different
    axes."""
    ks = range(0, int(np.ceil(reach)))
    ax.yaxis.set_major_locator(IX_FIXED_LOCATOR([k + 0.5 for k in ks]))
    ax.yaxis.set_major_formatter(IX_FIXED_FORMATTER([str(k) for k in ks]))
    ax.yaxis.set_minor_locator(IX_FIXED_LOCATOR([float(k) for k in range(1, ks.stop + 1)]))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(False, axis="y", which="major")
    ax.grid(True, axis="y", which="minor", color=GRID, lw=GRID_LW, ls=DOT)


def _interaction_frame(ax, vals, ylabel, top, caption, cells=None):
    """Everything both panel forms share: the limits, the house furniture, the y name and
    the per-category caption."""
    ax.set_ylim(bottom=-0.02 * top, top=top)
    furniture(ax, ylabel=None, xlim=(0.4, len(vals) + 0.6), xloc=None)
    ax.set_ylabel(ylabel, fontsize=IX_Y_SIZE, labelpad=10)
    if cells:
        _interaction_cell_axis(ax, cells)
    else:
        # The violins are counts too, so their ticks are whole ones. The default locator
        # offered 0.0 / 2.5 / 5.0 on the hydrophobic panel the moment the panels grew to
        # full height -- a tick at two and a half contacts, beside a companion panel
        # stepping by whole ones.
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Cleared here and PUT BACK BY THE CALLER (2026-09-14). This used to be the end of it:
    # the legend under the figure was the key, so printing the names under the panels too
    # was the same information twice. With the legend gone the names are the only way to
    # tell one violin from another, so _interaction_ecdf_furniture sets them.
    ax.set_xticks([])
    ax.grid(False, axis="x")          # the x is categorical: a rule per slot is a fence
    # The per-category caption is dropped once there are eight of them: at that width
    # "mean 8.41" is wider than its own slot and the captions overprint each other. The
    # median dot still carries the location, and the numbers are in interactions.csv --
    # an unreadable overlap is worse than a table lookup.
    if caption:
        for i, v in enumerate(vals, start=1):
            ax.text(i, top, f"med {np.median(v):.0f}\nmean {np.mean(v):.2f}", ha="center",
                    va="top", fontsize=10, color=AXIS, linespacing=1.3, zorder=6)


def _interaction_violin_panel(ax, vals, cols, ylabel, caption=True):
    ax.set_facecolor("white")
    parts = ax.violinplot(vals, showextrema=False, widths=IX_VIOLIN_W,
                          bw_method=lambda k: IX_KDE_BW / np.std(k.dataset))
    for body, col in zip(parts["bodies"], cols):
        # THE PAIR FIGURE'S PAINTING (2026-09-14): the outline is the body's OWN colour, which
        # is what _interaction_pair_bodies has always done. Briefly removed earlier the same
        # day and put back so that all four interaction panels are painted by one rule --
        # face a translucent tint, stroke the saturated hue of the same method.
        body.set(facecolor=col, alpha=IX_BODY_ALPHA, edgecolor=col, linewidth=1.2)
    for i, v in enumerate(vals, start=1):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q1, q3, color=INK, lw=IX_IQR_LW, zorder=3)
        ax.plot(i, med, "o", color="white", ms=IX_MED_MS, zorder=4)

    # The top is the largest count anyone actually made, plus head-room for the caption.
    # NOT a percentile: cutting at p99.5 sliced the KDE tails off mid-body and left a row
    # of hard vertical edges along the top of the panel that read as real structure.
    # A count cannot be negative either, but a KDE fitted to a zero-heavy distribution puts
    # a tail below zero anyway; the axis cuts it rather than implying a count of -1 exists.
    top = max(max(v) for v in vals) * (IX_HEAD if caption else IX_HEAD_BARE)
    _interaction_frame(ax, vals, ylabel, top, caption)


def _interaction_letter_values(v, k=IX_BOXEN_K):
    """The k nested (lower, upper) bounds of a letter-value plot: the quartiles, then the
    eighths, and so on, each depth covering half the remaining tail. Depth d is the
    (100/2^(d+2))th and its mirror -- 25/75, 12.5/87.5, 6.25/93.75, 3.1/96.9, 1.6/98.4.

    EVERY BOUND IS AN OBSERVED COUNT (method="nearest"), not numpy's default interpolation
    between the two values a quantile falls between. These are counts: no pose accepts 1.5
    hydrogen bonds, and the default put the crystal ligands' quartiles at 0.5 and 2.5 purely
    because there are 79 of them and the quantile landed between two neighbours. On an axis
    of unit cells that also cut blocks half a count off the rules, so a block edge sat
    exactly where a cell's number sits. Snapped, every edge falls on a count and every block
    is a whole cell.

    The q25/q75 in interactions.csv are NOT snapped -- they stay the conventional
    interpolated quantiles, being a summary statistic rather than something drawn. Read the
    figure against the figure and the table against the table."""
    return [tuple(np.percentile(v, [100 / 2 ** (d + 2), 100 - 100 / 2 ** (d + 2)],
                                method="nearest"))
            for d in range(k)]


def _interaction_letter_bands(v, k=IX_BOXEN_K):
    """The same letter values as (lower, upper, depth) bands that DO NOT OVERLAP: depth 0
    is the quartile box, and each deeper level contributes only the step it adds above and
    below the one inside it.

    Drawn as nested boxes instead -- which is what boxenplot does, and what Fig. 14's SVG
    shows -- every level is a full rectangle from its lower to its upper bound, so the
    narrow pale ones overprint the middle of the wide dark ones. A horizontal slice through
    the column then runs dark at the flanks to pale at the centre, and that gradient encodes
    nothing: the depth is already in the width. Cut into bands, ONE HEIGHT OF THE COLUMN IS
    ONE WIDTH AND ONE SHADE, and the column reads as what it is -- a stack of blocks.

    A band of zero height is dropped rather than drawn, which is why a width can be skipped:
    where two letter values are equal, no height of the column belongs to the shallower of
    them."""
    lv = _interaction_letter_values(v, k)
    bands = [(lv[0][0], lv[0][1], 0)]
    for d in range(1, k):
        bands.append((lv[d - 1][1], lv[d][1], d))     # the step this depth adds above
        bands.append((lv[d][0], lv[d - 1][0], d))     # and below -- empty when both are 0
    return _interaction_cell_split(b for b in bands if b[1] > b[0])


def _interaction_cell_split(bands):
    """Every band cut again at each integer it crosses, so ONE CELL OF THE AXIS IS ONE
    BLOCK — always, including inside a band that spans several counts.

    Without this a depth whose two letter values are three counts apart is drawn as one tall
    rectangle running straight through the rules at the counts between them, and the column
    stops being something you can count: two arms with the same top edge look different
    depending on where their quantiles happened to fall. Cut, every column is the same ladder
    of unit blocks and only the width and the shade of each rung differ. Nothing here rounds:
    the bounds arrive already sitting on counts (see _interaction_letter_values), and this
    only cuts the bands between them."""
    out = []
    for lo, hi, d in bands:
        cuts = [lo] + [float(c) for c in range(int(np.floor(lo)) + 1, int(np.ceil(hi)))
                       if lo < c < hi] + [hi]
        out += [(a, b, d) for a, b in zip(cuts, cuts[1:]) if b > a]
    return out


def _interaction_tint(col, t):
    """`col` mixed t of the way to white."""
    return tuple(c + (1.0 - c) * t for c in IX_TO_RGB(col))


def _interaction_boxen_panel(ax, vals, cols, ylabel, caption=True):
    """VoxBind's Fig. 14(a),(b): nested boxes, one per quantile depth, each half the width
    of the one inside it and all centred on the category — but cut into non-overlapping
    bands, so one height of the column carries one width and one shade and nothing else
    (see _interaction_letter_bands). WHERE THE LOWER LETTER VALUE IS
    ZERO the box sits on the axis — and on both H-bond types it is zero at every depth for
    every arm but one, because at least a quarter of every arm's molecules make none. So the
    column reads as a stack of blocks narrowing upward, each block one step further into the
    tail, which is exactly what a count distribution piled on zero is. (The one exception is
    the crystal ligands' quartile box on HBAcceptor, which starts at 1 rather than 0: only
    25.3 % of them accept none, so its p25 lands on the first molecule that does.)

    A LETTER-VALUE PLOT AND NOT A VIOLIN because these counts are small integers. A KDE has
    to smooth 0-10 into a continuum, which draws a body between the counts rather than on
    them, and the arms then differ in the shape of an artefact. Nested boxes are quantiles
    of the data and nothing else: no bandwidth, no interpolation between counts a molecule
    could not have made, and the zero-heavy left edge stays an edge."""
    ax.set_facecolor("white")
    reach = 0.0
    for i, (v, col) in enumerate(zip(vals, cols), start=1):
        for lo, hi, d in _interaction_letter_bands(v):
            w = IX_VIOLIN_W / 2 ** d  # the innermost band is as wide as a violin body
            # THE VIOLINS' SATURATION, so the two figures of this family read as one set:
            # the alpha goes on the FACE only, as a fourth channel, rather than on the patch.
            #
            # THE OUTLINE STAYS, and it is not decoration: the outline is what makes a block
            # a block. Taken off on 2026-09-14 and PUT BACK the same day, because the column
            # collapsed into one stepped silhouette -- the bands are NON-OVERLAPPING and depth
            # is carried by width and tint, but IX_BOXEN_LIGHT spreads only 0.46 over five
            # depths, so without a rule between them the quartile box and the tail bands
            # stopped being separable.
            # IN THE METHOD'S OWN COLOUR, not the neutral AXIS grey it used to take: that is
            # the rule the pair figure paints by, and all four panels of this family now
            # share it -- face a translucent tint, stroke the saturated hue of the same
            # method.
            #
            # The alpha rides on the FACE as a fourth channel rather than on the patch: a
            # patch alpha would fade the outline with it.
            face = _interaction_tint(col, IX_BOXEN_LIGHT * d / (IX_BOXEN_K - 1)) \
                + (IX_BODY_ALPHA,)
            ax.add_patch(IX_RECTANGLE((i - w / 2, lo), w, hi - lo, zorder=3 + d,
                                      facecolor=face,
                                      edgecolor=col, linewidth=IX_BOXEN_EDGE_LW))
            reach = max(reach, hi)
        # The violins' white median dot, not boxenplot's median line: the two forms sit in
        # one figure and the mark for "here is the middle" must not change between panels.
        # On these types the dot is often ON the axis, which is the finding — more than
        # half of most arms' poses donate no hydrogen bond at all.
        ax.plot(i, np.median(v), "o", color="white", ms=IX_MED_MS, zorder=3 + IX_BOXEN_K)
    top = reach * (IX_HEAD if caption else IX_HEAD_BARE)
    _interaction_frame(ax, vals, ylabel, top, caption, cells=reach)


IX_PANELS = {"violin": _interaction_violin_panel, "boxen": _interaction_boxen_panel}


def _interaction_row_major(n, ncol):
    """Indices that make matplotlib's column-major legend fill read left-to-right."""
    nrow = -(-n // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < n]
    return order


def _interaction_legend(fig, names, cols, form):
    """The method key, under the panels instead of inside one of them — which is what lets
    the category tick labels go (see _interaction_frame). The swatch is drawn the way that
    figure's own marks are: a translucent violin body for the contacts, a solid block in the
    axis pen for the letter values. A legend that does not look like the thing it names is a
    second key to learn."""
    if form == "violin":
        handles = [Patch(facecolor=c, alpha=IX_BODY_ALPHA, edgecolor=c, lw=1.2,
                         label=_interaction_display(n)) for n, c in zip(names, cols)]
    else:
        handles = [Patch(facecolor=IX_TO_RGB(c) + (IX_BODY_ALPHA,), edgecolor=AXIS,
                         lw=IX_BOXEN_EDGE_LW, label=_interaction_display(n))
                   for n, c in zip(names, cols)]
    # ONE ROW, which the 4:1 frame is wide enough for even at eight names. The reordering
    # below is what makes any other count safe: MATPLOTLIB FILLS A MULTI-ROW LEGEND
    # COLUMN-MAJOR, and on the two rows of four this used to need, that turned the
    # drug-design order into Reference, Pocket2Mol, TargetDiff, VoxBind across the top --
    # an order that is no order at all. Re-sequencing the handles so a column-major fill
    # lands them row-major is the only way back; there is no rcParam for it.
    ncol = len(names)
    handles = [handles[i] for i in _interaction_row_major(len(handles), ncol)]
    # The house legend box, as legend() draws it on an axes: opaque white with a rule in the
    # axis pen, square corners. Quiet enough not to compete with the panels, and it gives
    # the key an edge of its own now that it sits in open figure margin.
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.012), ncol=ncol, frameon=True,
                     facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0,
                     borderpad=0.5, fontsize=IX_LEGEND_SIZE, handlelength=1.5,
                     handleheight=1.0, handletextpad=0.6, columnspacing=1.9,
                     labelspacing=0.55)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _interaction_axis_weight(ax, lw=None, title_fs=None, tick_fs=None):
    """The ecdf-by-size-all axis: spine weight, tick type and the y name's size.

    SPLIT OUT OF _interaction_ecdf_furniture (2026-09-15) so the PAIR figure can take the
    same look. That function cannot simply be called there: it also rebuilds the x axis at
    1..n with the names rotated 30 degrees, and the pair figure deliberately keeps its
    names UPRIGHT on each split body's own spine -- calling it would undo the widening that
    was chosen to make upright names fit.

    ONLY THE SPINES, TICKS AND AXIS NAME MOVE. Not _pcsz_style() wholesale: that turns the
    major y grid back on, and the H-bond panels run a deliberate minor-grid arrangement
    (_interaction_cell_axis) that would lose to it.

    PCSZ_* are read at CALL time: they live in a part assembled after this one."""
    # DEFAULTS ARE THE ECDF'S OWN NUMBERS -- _pcsz_style sets spine and tick 1.1, tick labels
    # at 11 and the y name at PCSZ_LABEL_FS, and contacts/hbonds must keep exactly those,
    # since being indistinguishable from ecdf-by-size-all is what this function is for. The
    # three overrides let ONE caller ask for more; only the pair panel does.
    #
    # `tick_fs` REACHES THE Y AXIS ONLY IN PRACTICE, even though tick_params sets both: the
    # pair panel rebuilds its x axis afterwards with set_xticklabels(fontsize=...), so the
    # method names keep IX_PAIR_NAME_FS and this number lands on the y numbers alone.
    lw = 1.1 if lw is None else lw
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PCSZ_AXIS)
        ax.spines[side].set_linewidth(lw)
    ax.tick_params(labelsize=11 if tick_fs is None else tick_fs,
                   direction="out", length=3.5, width=lw, pad=4, colors=PCSZ_AXIS)
    ax.yaxis.label.set_fontsize(PCSZ_LABEL_FS if title_fs is None else title_fs)
    ax.yaxis.label.set_color(PCSZ_AXIS)


def _interaction_ecdf_furniture(ax, names, show_names=True):
    """The strain ECDF's axis weight and type, plus the method names back on the x axis.

    `show_names` is False for the figure that sits ABOVE another carrying the same
    categories -- see IX_NAMED_PARTS.

    ONLY THE SPINES, TICKS AND AXIS NAME MOVE. Not _pcsz_style() wholesale: that turns the
    major y grid back on, and the H-bond panel runs a deliberate minor-grid arrangement
    (_interaction_cell_axis) that would lose. Not PCSZ_RC either -- it carries savefig.dpi
    170 and a crop to the ink, and this family draws at 220 to its own frame.

    PCSZ_* are read at CALL time: they live in a part assembled after this one."""
    _interaction_axis_weight(ax)
    # The violins sit at 1..n, which is where _interaction_frame's xlim is built from.
    ax.set_xticks(range(1, len(names) + 1))
    labels = ax.set_xticklabels([_interaction_display(n) for n in names],
                                fontsize=IX_NAME_FS, rotation=30, ha="right",
                                rotation_mode="anchor")
    if not show_names:
        # DRAWN BUT INVISIBLE, rather than absent. tight_layout measures a text's extent
        # whatever its colour, so colouring the names "none" reserves the exact band of
        # height they would have taken -- which is what keeps this figure's axes the same
        # height as the named one it sits above. Dropping the labels instead let fit() hand
        # that band to the axes, and the two panels came out with their bottom spines 174 px
        # apart inside a canvas they share, so stacked they would have carried different
        # vertical scales for the same categories.
        # The tick MARKS stay: they say where a category is without naming it.
        for t in labels:
            t.set_color("none")


def _interaction_panels(out, part, kinds, rows_by_label, names, variant, caption):
    """One figure: a row of two panels of one form, with the key beneath them.

    `part` is in the filename beside the variant, for the reason the variant is: two
    figures that differ in what they draw must not be able to sit in a folder under one
    name."""
    cols = [pale(n) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(IX_FIG_W, IX_FIG_H), dpi=220)
    fig.patch.set_facecolor("white")
    form = None
    for ax, kind in zip(axes, kinds):
        ylabel, form = IX_TYPE[kind]
        vals = [_interaction_counts(rows_by_label[n], kind) for n in names]
        IX_PANELS[form](ax, vals, cols, ylabel, caption)
        _interaction_ecdf_furniture(ax, names, part in IX_NAMED_PARTS)
    # NO LEGEND, and so no reserved strip: the names are on the x axis now, and the rect that
    # used to hold the key back would leave an empty band under the panels.
    fit(fig, pad=0.5, w_pad=2.2)
    save(fig, out, f"interaction_{part}_{variant}")


# ── the numbers behind the panels ────────────────────────────────────────────────
def _interaction_block(rows):
    out = {"n_molecules": len(rows),
           "n_scored": sum(r["ifp"] is not None for r in rows),
           "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None}
    for kind in IX_KINDS:
        v = _interaction_counts(rows, kind)
        out[kind] = {"mean": round(float(np.mean(v)), 3), "median": float(np.median(v)),
                     "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                     "share_nonzero": round(100 * float(np.mean(np.asarray(v) > 0)), 2)}
    return out


def _interaction_table(p79_rows, refrows, arms, base):
    """Every method's p79 block, in the drug-design table's row order — the order the
    figures draw, so the CSV, the printed table and the panels can be read against each
    other without re-sorting any of them.

    `arms` is what the pose loader says carries a fingerprint over the whole pocket set --
    NOT every entry of ARMS. Since 2026-09-10 the five published baselines are ARMS too,
    staged under exps/baselines_pose/, but they hold PoseBusters and not PoseCheck; blocking
    one here would export a row of nulls that looks measured and lost."""
    blocks = {lab: _interaction_block(p79_rows[key]) for lab, key, _ in arms}
    blocks.update({lab: _interaction_block(base[lab]) for lab in base})
    blocks[REF_LABEL] = _interaction_block(refrows)
    return [(lab, blocks[lab]) for lab in _interaction_ordered(list(blocks))]


def _interaction_part(out, part):
    """One of the two panel figures — core and all — plus the shared summary table.

    THE CSV IS THE WHOLE TABLE, not this part's two types, and it is written into both
    figure folders. It carries the pooled "All interactions" row and the atom counts that
    say why an arm sits high, and a reader checking one panel against the numbers needs
    those whichever half of the split they are looking at."""
    use_style()
    data, p79_rows, refrows = pose_data()
    # ARMS carries every arm this section knows; only some of them have been scored with
    # PoseCheck. arms_for() keeps an arm only where all 79 pockets have a scored molecule,
    # so an arm mid-scoring cannot enter a figure as a complete-looking column.
    scored = arms_for("ifp", data)

    # The published baselines come from the recompute beside the original builder, not from
    # ARMS: their staged sample dirs hold PoseBusters and not PoseCheck. A name that has
    # become a scored arm is taken from the arm, so no method can enter twice.
    live = {lab for lab, _, _ in scored}
    base = {l: r for l in IX_BASELINES if l not in live
            for r in [_interaction_baseline_rows(l)] if r is not None}
    by_label = {lab: p79_rows[key] for lab, key, _ in scored}
    by_label.update(base)
    by_label[REF_LABEL] = refrows

    core = _interaction_ordered([REF_LABEL] + [lab for lab, key, _ in scored
                                               if key in CORE])
    every = _interaction_ordered(list(by_label))
    kinds = dict(IX_PARTS)[part]
    _interaction_panels(out, part, kinds, by_label, core, "core", caption=True)
    _interaction_panels(out, part, kinds, by_label, every, "all", caption=False)
    missing = [l for l in IX_BASELINES if l not in base and l not in live]

    rows = _interaction_table(p79_rows, refrows, scored, base)
    write_csv(out, "interactions",
              ["arm", "type", "mean", "median", "q25", "q75", "share_nonzero_pct",
               "n_scored"],
              [[lab, k, b[k]["mean"], b[k]["median"], b[k]["q25"], b[k]["q75"],
                b[k]["share_nonzero"], b["n_scored"]]
               for lab, b in rows for k in IX_KINDS])

    print(f"  {len(P79)}-pocket set · interaction fingerprint, pocket10 crop · "
          f"mean per molecule\n")
    print(f"  {'arm':16s} {'atoms':>6s} {'mols':>6s} "
          + " ".join(f"{k:>12s}" for k in IX_KINDS))
    for lab, b in rows:
        print(f"  {lab:16s} {b['atoms_mean']:6.1f} {b['n_scored']:6d} "
              + " ".join(f"{b[k]['mean']:12.2f}" for k in IX_KINDS))
    print(f"\n  {'share of molecules making the type at all':>30s}")
    print(f"  {'arm':16s} {'':6s} {'':6s} " + " ".join(f"{k:>12s}" for k in IX_KINDS))
    for lab, b in rows:
        print(f"  {lab:16s} {'':6s} {'':6s} "
              + " ".join(f"{b[k]['share_nonzero']:11.1f}%" for k in IX_KINDS))
    print("\n  Every type scales with ligand size (see the atoms column): read each arm "
          "against\n  the crystal ligands, not against the arm above it.")
    if missing:
        print(f"\n  no interactions_<M>.json for: {', '.join(missing)} — run "
              f"stage_baseline_poses.py then compute_baseline_interactions.py")


@figure("fig-interaction-contacts", folder="fig-posecheck/interaction",
        needs=("metrics.json", "interactions_<Baseline>.json"))
def draw_interaction_contacts(out):
    """van der Waals and hydrophobic contacts per pose, as violins — core and all."""
    _interaction_part(out, "contacts")


@figure("fig-interaction-hbonds", folder="fig-posecheck/interaction",
        needs=("metrics.json", "interactions_<Baseline>.json"))
def draw_interaction_hbonds(out):
    """H-bonds accepted and donated per pose, as letter-value blocks — core and all."""
    _interaction_part(out, "hbonds")


# ════════════════════════════════════════════════════════════════════════════════
# fig-interaction-pair — the same molecule as generated and as redocked
# ════════════════════════════════════════════════════════════════════════════════
# VoxBind's Fig. 14 draws every interaction type twice: the pose AS GENERATED and the same
# molecule REDOCKED. The two figures above draw the first only, because nothing in the
# pipeline keeps a docked conformer. dock_core_poses.py produces the second for the core
# three arms and compute_baseline_interactions.py fingerprints both; this draws the pair.
#
# WHAT THE PAIR IS FOR. Redocking asks a different question from generating: not "what did
# the model put in the pocket" but "what does this molecule do in the pocket when a docking
# program is allowed to place it". A model whose generated poses already sit where Vina would
# put them has little to gain from redocking; one whose poses are merely plausible-looking
# gains a lot. The gap between a method's two columns is that quantity, and it is only
# readable because THE TWO COLUMNS ARE THE SAME MOLECULES: dock_core_poses.py drops a
# molecule from both series when its dock fails, so nothing separates the columns except the
# docking. 8,180 molecules over 78 pockets, from 8,286 docked. The 106 that did not pair
# were the two failures dock_core_poses.py's docstring predicts and nothing else: 105 are
# target_71, whose pocket crop pdb2pqr30 refuses for a chain gap at GLU 866, so it loses
# every arm; the last is VoxBind's `CNNO` at target_96, four heavy atoms whose extent gives
# Vina a box with a zero dimension.
#
# COLOUR STAYS THE METHOD'S, which is where this figure departs from Fig. 14 on purpose.
# Fig. 14 gives each series its own colour ramp (green #72b6a1 -> #eaf4f1 for generated,
# salmon #e99675 -> #fcefea for redocked) and puts the methods on the x axis. It can afford
# that because the series is the only other thing in the panel. Here colour means the method
# in every figure of this section, and a reader who has learned that palette should not have
# to unlearn it for one panel. So the pair takes the two channels Fig. 14 leaves free:
# THE TWO HALVES OF ONE BODY, which puts the comparison inside a single shape, and a HATCH
# on the redocked half.
#
# SPLIT, NOT ADJACENT (2026-09-15). This drew two whole bodies side by side until the split
# violin replaced them. Two bodies made the reader compare across a gap and cost twice the
# ink per method; one body cut down the middle puts the two distributions against a shared
# spine, which is the comparison this figure exists to make. It also doubles the width each
# method's ink gets, because nine slots now hold one body instead of two.
#
# AND THE METHOD NAMES COME BACK TO THE X AXIS. The two figures above drop them because
# eight of them, rotated to fit, cost a third of the height; here they stay upright and the
# FRAME widens to make room, rather than the names tilting to fit the frame. Spending the
# legend on the method as well would leave the pose condition — the whole point of this
# figure — as a footnote in a five-entry key. Each channel gets its own place: the method
# on the axis, the condition in the legend.
#
# Upright names were free when this was three pairs on the siblings' 4:1 frame; at nine they
# cost the width IX_PAIR_ASPECT carries, and that was chosen deliberately over tilting them.
# See the note on IX_PAIR_ASPECT for what the alternative would have been.

# The arms dock_core_poses.py staged, in the drug-design table's row order. These are the
# labels interactions_<Arm>-{gen,docked}.json is written under — the export spelling, not
# the figure's; _interaction_display() maps them at draw time exactly as the siblings do.
# ALL NINE (2026-09-15). This was the core three until the redocking was extended to every
# published baseline: 8,286 docks over 79 pockets, 8,180 of which paired. The five staged
# baselines reach dock_core_poses.py through --staged and TargetDiff through
# --with-targetdiff, so every arm in IX_ORDER now has both a -gen and a -docked scoring.
IX_PAIR_ARMS = ["Reference", *METHOD_ORDER]
IX_SERIES = [("gen", "Generated"), ("docked", "Redocked")]

# The reference's colour is keyed under the canonical label, not under the "Reference" the
# staged JSON is named with.
IX_COLOR_KEY = {"Reference": REF_LABEL}

# GEOMETRY. The panel height and the legend strip are the siblings' exactly, so the figures
# of this family still stack without a vertical size change. THE WIDTH IS NO LONGER THEIRS
# (2026-09-15): nine pairs of upright names do not fit a 4:1 frame.
#
# WHY THIS NUMBER, AND WHY THE NAME FONT MOVES WITH IT. The binding constraint is that one
# method's slot must
# hold its own name AND a gap to its neighbour's, and the slot is the group pitch — so the
# requirement is essentially `n_arms x (widest_label + margin)` and TIGHTENING IX_GROUP_GAP
# BUYS NOTHING, because the span and the pitch shrink together. Measured under this style at
# IX_LEGEND_SIZE, the widest label is VoxBind$_{\sigma=0.9}$ at 1.139 in.
#
# THE MARGIN IS NOT OPTIONAL, which is what a first pass at 5.85 got wrong: sizing the slot
# to the widest label ALONE leaves zero room between neighbours, and the two widest that
# happen to be adjacent -- DecompDiff and VoxBind -- came out 0.034 in apart while every
# other pair had 0.15-0.56 in.
#
# AND THE PITCH DOES NOT GET THE WHOLE WIDENING, which is what the SECOND pass got wrong.
# fit() spends part of every added inch on the margins and the y names, so only about 86 %
# of it reaches the panel: 6.12 was computed to land exactly on a 0.10 in floor and measured
# 0.090. Do not predict this from one render -- sweep the aspect and read the gap back. On
# this frame the worst gap moves ~0.21 in per unit of aspect, and the measured ladder is
# 5.85 -> 0.034, 6.12 -> 0.090, 6.20 -> 0.107, 6.30 -> 0.128. 6.20 is the smallest that
# clears the floor, which is why the figure is no wider than it is.
#
# Measure with the tick labels' own bounding boxes after fit(), never by eye: at this size
# 0.034 in reads as a collision and is not one.
#
# NARROWED TO 0.8x ON REQUEST (2026-09-15): 6.20 -> 4.96, 23.88 in -> 19.10 in. At the font
# of the time that OVERLAPPED -- measured -0.068 in, the labels actually colliding, because
# width is the only lever once the font is fixed and the names are upright. So the font took
# the same 0.8x: 12.5 x 0.8 = 10.0, and the worst gap came back to 0.127 in.
#
# THAT WHOLE CONSTRAINT IS GONE NOW, and the paragraph above is kept only as the record of
# how this number was arrived at. The names were rotated 30 degrees shortly afterwards and
# IX_PAIR_NAME_FS went to 14.0: rotated labels run off their own tick instead of competing
# for a method's pitch, so width no longer gates the font, and widening only ever helps.
#
# THEN 1.03x ON REQUEST: 4.96 -> 5.11, 19.10 in -> 19.68 in. THEN 0.98x: 5.11 -> 5.01,
# 19.68 in -> 19.30 in. That narrowing alone costs the stacked axes 9.009 -> 8.816 in of
# width, before the type sitting on it is counted.
#
# THE COST, recorded so nobody re-derives it: each half violin goes 0.438 in -> 0.342 in,
# under the 0.411 in of the single violin the old adjacent layout drew. That is inherent to
# narrowing the frame, not a sizing mistake like the IX_GROUP_GAP one below it.
# THEN 1.2x ON REQUEST: 5.01 -> 6.01, 19.30 in -> 23.15 in. Widening is the safe direction --
# the names are rotated, so it only ever adds room between them, and it leaves the vertical
# dimension (where the y title's clipping lives) alone.
# THEN 1.03x ON REQUEST (2026-09-15): 6.01 -> 6.1903, 23.15 in -> 23.84 in. Height is
# deliberately NOT touched -- IX_PAIR_TALL stays 1.20 -- so this is width alone.
IX_PAIR_ASPECT = 6.1903
# HEIGHT ONLY, ON TOP OF THE WIDTH ABOVE (2026-09-15). The pair figures are taller than the
# siblings by this much; the WIDTH stays IX_PAIR_ASPECT * IX_FIG_H so the 0.98x narrowing is
# untouched. IX_FIG_H itself must not move: IX_FIG_W is derived from it, so raising it would
# silently widen contacts-all, hbonds-all, both core variants and the legend as well.
#
# WHY 1.20 AND NOT "SLIGHTLY". The y name is rotated, so its LENGTH runs up the figure --
# "van der Waals contacts" is 3.16 in at 19.5 pt against a 2.44 in axes, and it was running
# off the canvas. Growing the frame grows the axes at roughly HALF the rate, so the fix is
# expensive: measured contacts margin at 19.5 pt is 1.00 -> -0.257 in (clipped), 1.10 ->
# -0.064 (still clipped), 1.15 -> +0.032 (seven pixels, breaks on any rounding), 1.20 ->
# +0.128, 1.25 -> +0.224. 1.20 is the smallest that holds.
#
# THE CHEAPER TRADE, if this frame is ever too tall: lower the title with it. At 17.5 pt a
# 1.10 multiplier already gives +0.155 in, and a two-line title clears +0.364 in at no extra
# height at all.
# 1.20 -> 1.35 -> BACK TO 1.20 (2026-09-15). The 1.35 was bought for ONE REASON ONLY: a
# one-line "van der Waals contacts" at 22 pt ran off the canvas at 1.20, measured -0.145 in.
# Wrapping the pair titles to two lines (IX_PAIR_YLABEL) halves the length that runs up the
# figure and repays that debt outright, so the extra height is no longer owed and the frame
# returns to the height it had before.
#
# MEASURED WITH THE WRAP IN PLACE, vertical margin on the contacts standalone -- the panel
# that gates this, since it carries the longest y name: 1.00 -> +0.170, 1.10 -> +0.362,
# 1.20 -> +0.555, 1.35 -> +0.844. Even 1.00 clears now. 1.20 is kept because it is the
# height that was settled on earlier, not because anything below it fails.
#
# WIDTH CANNOT SUBSTITUTE FOR THIS. The title is rotated 90 degrees, so its length runs UP
# the figure; a wider frame buys the y name nothing at all.
IX_PAIR_TALL = 1.20
# The method names' size, PAIR-ONLY and deliberately not IX_LEGEND_SIZE: that constant is
# shared with fig-interaction-legend and _interaction_pair_legend, and retyping it for this
# figure's geometry would silently retype the standalone key as well.
#
# THE SAME SIZE AS THE Y AXIS TITLE, on request (2026-09-15): this must equal
# IX_PAIR_TITLE_FS. It is written as a literal rather than as that name because TITLE_FS is
# defined BELOW this line, and referring to it here would raise at import. CHANGE THE TWO
# TOGETHER -- a title moved on its own silently leaves the method names behind.
#
# The road here: 10.0 while the names were upright, because upright names had to fit side by
# side inside one method's pitch and the frame had been narrowed to 0.8x. Rotating them 30
# degrees removed that constraint entirely -- a diagonal label runs off its own tick instead
# of competing for its neighbour's space -- so the size became free to follow the title.
#
# IT COSTS AXES HEIGHT, and the standalone pays about twice as much because one band of
# names sits on half the height: at the 19.30 in frame, 18.5 gives stacked 8.816 x 3.169 and
# standalone 8.697 x 2.498, 21.0 gives 8.784 x 3.098 and 8.587 x 2.357.
#
# BUT THE CEILING IS THE Y TITLE, NOT THE AXES. The y name is rotated 90 degrees, so its
# LENGTH runs up the figure: "van der Waals contacts" is 3.4 in at 21 pt against a 3.1 in
# axes, and matplotlib centres it on the axes and lets the overflow run off the canvas.
# Measured top margin on the stacked frame: 18.5 -> +0.146 in, 19.5 -> +0.049, 20.0 ->
# +0.004, 20.5 -> -0.043 CLIPPED, 21.0 -> -0.092 CLIPPED. 19.5 is the largest size with a
# margin that survives rounding; 20.0's four thousandths of an inch is one pixel.
#
# CHECK THIS ON THE RENDERED PIXELS. fig.get_tightbbox() reported "no clipping" at 21.0 AND
# at 23.0, both of which cut the title off at row 0 of the raster. Measure the y label's own
# window extent against the figure height, or read the PNG.
#
# DO NOT check these for collisions with get_window_extent: it returns the AXIS-ALIGNED box
# of a rotated label, and neighbouring diagonal labels overlap those boxes while their ink
# passes cleanly between. It reports collisions at sizes that are demonstrably fine.
# 19.5 -> 22.0 (2026-09-15). IT MOVES WITH THE TITLE, per the rule above: the two were
# already EQUAL at 19.5 -- measured on the rendered canvas, not merely equal in the source --
# so raising the title on its own would have left the names behind and broken that rule.
IX_PAIR_NAME_FS = 22.0
# THE PAIR FIGURES' OWN AXIS WEIGHT AND Y NAME SIZE (2026-09-15), heavier than the ecdf's.
# _interaction_axis_weight defaults to the ecdf's exact numbers -- spine and tick 1.1, y name
# PCSZ_LABEL_FS -- and contacts/hbonds MUST keep them, since being indistinguishable from
# ecdf-by-size-all is the whole point of that function. These are passed by the pair panel
# alone so the siblings are untouched.
#
# 2.2 IS THE JSD SUMMARY'S, TAKEN DELIBERATELY (2026-09-15). Weight has to be read against
# the canvas it sits on: this figure is 19.10 in wide, the widest in the set, so at 1.6 it
# was the THINNEST of the three by the measure that matters -- 0.084 pt/in against the JSD
# summary's 0.165 and the clash box's. Matching JSD's absolute 2.2 brings it to 0.115 pt/in,
# most of the way there without going to the full column-scaled 3.15.
#
# The ladder this repo has, for whoever tunes it next: ecdf 1.1, the house default AXIS_LW
# 1.35, the clash box's PCSZ_CBOX_AXIS_LW 1.0 (a narrow canvas, so a small number), the JSD
# summary's 2.2 and the rotatable-bond grid's 2.0. Compare pt/in, not pt.
#
# 18.5 IS SAFE, MEASURED. Raising this family's y name from 13 to 16.1 once truncated the
# H-bond titles to "...by the liga" on the fixed dpi-220 frame, so it was swept before being
# changed: at 16.1/17.5/18.5/19.5/21.0 nothing leaves the frame in either the standalone or
# the stacked figure, and the axes lose 0.06 in of width across that whole range.
IX_PAIR_AXIS_LW = 2.2
# 19.5 -> 22.0 (2026-09-15), ON REQUEST, AND IX_PAIR_NAME_FS MOVES WITH IT. The ceiling is
# the rotated y name's clearance, which is bought either with IX_PAIR_TALL or by wrapping the
# name -- see IX_PAIR_YLABEL, which is what actually paid for this size.
#
# THE OLD WARNING HERE SAID "22.0 clips at the 1.20 frame by 0.145 in". That was measured on
# a ONE-LINE title and is no longer what this figure draws: with the two-line names the same
# frame clears by +0.555 in. Re-run the height sweep before raising this again, and do not
# trust fig.get_tightbbox() for it -- it reported no clipping at sizes that were visibly cut.
IX_PAIR_TITLE_FS = 22.0
# The y numbers. 14.0 is the house default -- furniture()'s labelsize -- rather than a number
# picked by eye, and it replaces the ecdf's 11, which read small once the names and the title
# both went to 18.5. Swept before changing: 11 -> 15 costs the axes 0.6 % of their width and
# nothing at all in height or row gap, so this size is essentially free and can go further.
# 14.0 -> 16.5 (2026-09-15), on request. The sweep recorded above still holds -- the y
# numbers cost axes WIDTH and nothing in height -- so this one does not touch the title's
# vertical clearance, which is the only thing that clips in this figure.
IX_PAIR_TICK_FS = 16.5
# THE GAP BETWEEN THE Y NAME AND ITS AXIS, DOUBLED ON REQUEST (2026-09-15): 10 -> 20 points.
# PAIR-ONLY, and it has to be. _interaction_frame sets labelpad=10 and is called by BOTH
# sibling panel forms as well as this one, so doubling it there would push the y name out on
# contacts-all, hbonds-all and both cores -- six figures that were not asked to move.
#
# IT COSTS AXES WIDTH, NOT CLEARANCE. tight_layout measures the label with its pad, so the
# extra 10 points (0.139 in) is taken out of the panel rather than pushing the title off the
# left edge; the title's own horizontal margin is unchanged. Measure both after changing it.
IX_PAIR_LABELPAD = 20
# A FIXED Y RANGE FOR ONE PANEL, PAIR-ONLY (2026-09-15): kind -> (top, tick step).
# _interaction_frame sizes every panel to its own data (reach * IX_HEAD_BARE), which lets the
# van der Waals violins run to 32 on the strength of a few tails while the mass sits under
# 15. Cutting that panel at 25 spends the height on the part of the distribution a reader is
# comparing. THE TAILS ARE CLIPPED, NOT DROPPED -- the bodies are drawn before this runs, so
# the data is unchanged and only the view is cropped.
#
# ONLY VdWContact IS LISTED, DELIBERATELY. Hydrophobic tops out near 13, so the same 25 would
# leave its upper half empty and squeeze a median of 1-2 into a sliver; it keeps its own
# scale. A kind absent from this map is sized by the frame exactly as before.
# kind -> (ylim top, topmost tick, tick step). THE LIMIT AND THE TOP TICK ARE SEPARATE
# (2026-09-15) and both exist to fix the same thing. Cutting at exactly 25 put the last tick
# ON the top spine, so half of the "25" label sat outside the axes, tight_layout reserved the
# overhang, and the contacts panel came out 0.086 in SHORTER than the H-bond panel it is
# placed next to -- a size difference created purely by a tick label. Raising the limit to 26
# pulls the label inside, and THAT is what fixed it -- the two panels measure the same
# 3.0671 in again. The label is kept; only the limit does the work. If the top tick is ever
# moved back onto the limit, expect the panel to lose height to the overhang again.
IX_PAIR_YCUT = {"VdWContact": (26, 25, 5)}
IX_SPLIT_W = 1.60                # the full split body: left half generated, right redocked
# 0.50, NOT THE 1.15 THE ADJACENT LAYOUT USED (2026-09-15). The split is only worth having
# if the body gets the room the two old columns had between them, and a gap sized for
# separating two whole violins wastes it: at 1.15 each half measured 0.347 in, THINNER than
# the 0.411 in single violin it replaced, because 1.60 of ink in a 2.75 pitch is 58 % of the
# slot where the old pair filled 83 %. At 0.50 each half is 0.437 in and the worst label gap
# is still 0.181 in, comfortably over the 0.10 floor.
#
# THE SLACK GOES INTO THE BODY, NOT INTO A NARROWER FIGURE. Pulling IX_PAIR_ASPECT down
# instead spends it on nothing a reader can see and collapses the labels quickly -- 5.60
# measures a 0.064 in gap, already under the floor.
IX_GROUP_GAP = 0.50              # between one method and the next
# Where each half's median dot and IQR bar sit, as a share of IX_SPLIT_W off the centre
# spine. Far enough in to read as belonging to its own half, not so far that it leaves the
# body at a method whose distribution is narrow.
IX_SPLIT_STAT_OFF = 0.18
IX_HATCH = "////"
IX_HATCH_LW = 3.2  # PoseCheck legend stroke; the pair figure doubles it below.
IX_PAIR_HATCH_LW = 2 * IX_HATCH_LW  # interaction-pair-all: 6.4 pt
IX_PAIR_HATCH = "/"  # Keep 6.4 pt strokes distinct instead of merging into a solid fill.
# Neutral grey for the two condition swatches: the legend here names the POSE, not the
# method, and giving it one of the method colours would say the opposite.
IX_SWATCH = "#9aa0a8"


# THE PAIR FIGURES' Y NAMES, WRAPPED TO TWO LINES (2026-09-15), AND PAIR-ONLY.
# IX_TYPE is read by _interaction_panels as well -- the sibling contacts-all and hbonds-all
# figures and both of their cores -- so wrapping the strings where they are DEFINED would
# re-break the titles on six figures that were never asked to change. This maps the same
# keys to a two-line spelling and is consulted only by _interaction_pair_panel. A kind
# missing here falls back to IX_TYPE's own name, so IX_TYPES stays the single source of
# which kinds exist and adding one cannot raise here.
#
# ALL FOUR BREAK, AND EACH BREAKS BEFORE ITS LAST WORD. The stacked figure puts contacts
# above H-bonds, so titles with unequal line counts would hang at different heights against
# axes aligned to 0.0000 in -- exactly the ragged edge fig.align_ylabels was added to remove.
# Keeping every title two lines is what preserves that alignment.
IX_PAIR_YLABEL = {
    "VdWContact":  "van der Waals\ncontacts",
    "Hydrophobic": "Hydrophobic\ncontacts",
    "HBAcceptor":  "H-bonds\nacceptor",
    "HBDonor":     "H-bonds\ndonor",
}


def _interaction_pair_positions():
    """x for each ARM -- one slot per method, the two conditions split inside it.

    One position per arm, not two (2026-09-15): the conditions are halves of a single body
    now, so they share a centre rather than sitting at their own x."""
    pitch = IX_SPLIT_W + IX_GROUP_GAP
    return [1.0 + i * pitch for i in range(len(IX_PAIR_ARMS))]


def _interaction_pair_load():
    """{(arm, series): rows} from the staged scorings."""
    out = {}
    for arm in IX_PAIR_ARMS:
        for key, _ in IX_SERIES:
            path = legacy("fig-interaction", f"interactions_{arm}-{key}.json")
            out[arm, key] = [{"n": m["n"], "ifp": m["ifp"]}
                             for m in json.load(open(path))["molecules"]]
    return out


def _interaction_pair_check(rows):
    """The figure's one precondition: a method's two columns must be the same molecules.
    Rows are written in pocket-then-row order by one scorer over one staged layout, so
    equal length and an equal heavy-atom sequence is the pairing — the same check
    stage_baseline_poses.py makes against the export."""
    for arm in IX_PAIR_ARMS:
        g, d = rows[arm, "gen"], rows[arm, "docked"]
        if len(g) != len(d) or [r["n"] for r in g] != [r["n"] for r in d]:
            raise SystemExit(
                f"{arm}: the generated and redocked series are not the same molecules "
                f"({len(g)} vs {len(d)} rows) — the gap between the columns would be a "
                f"difference in population, not in pose. Re-stage with dock_core_poses.py")


def _interaction_pair_bodies(ax, vals, cols, xs):
    """One SPLIT violin per method: left half the generated pose, right half the redocked.

    `vals` is still flat and interleaved -- gen then docked for each arm -- so vals[2i] and
    vals[2i+1] are the two halves of the body at xs[i]. `cols` is one colour per ARM.

    HOW THE HALF IS MADE. violinplot has no split, so each half is drawn as a whole violin
    at the same x and then its path is CLIPPED to one side of that x. Clipping the vertices
    rather than re-deriving the KDE keeps both halves on exactly the shared spine, and keeps
    the bandwidth rule identical to the one every other violin in this family uses."""
    for i, (x, col) in enumerate(zip(xs, cols)):
        for half, v in enumerate((vals[2 * i], vals[2 * i + 1])):
            part = ax.violinplot([v], positions=[x], showextrema=False, widths=IX_SPLIT_W,
                                 bw_method=lambda k: IX_KDE_BW / np.std(k.dataset))
            body = part["bodies"][0]
            vtx = body.get_paths()[0].vertices
            lo, hi = (-np.inf, x) if half == 0 else (x, np.inf)
            vtx[:, 0] = np.clip(vtx[:, 0], lo, hi)
            # THE ALPHA IS BAKED INTO THE FACE, NOT SET ON THE ARTIST (2026-09-15). An
            # artist-level alpha applies to the EDGE too, and matplotlib draws a hatch in the
            # EDGE colour -- so the redocked half's slashes were being drawn in the body's own
            # colour at the body's own 0.55 opacity, on top of the fill they sat on, and were
            # effectively invisible. Measured before the fix: face and edge were the identical
            # RGBA (0.604, 0.627, 0.651, 0.55).
            #
            # _interaction_pair_blocks NEVER HAD THIS BUG and is the pattern copied here: it
            # passes an RGBA face and an opaque edgecolor, which is exactly why the H-bond
            # panels showed their slashes while the violins beside them did not.
            body.set(alpha=None, facecolor=IX_TO_RGB(col) + (IX_BODY_ALPHA,), edgecolor=col,
                     linewidth=1.2)
            if half:
                # Matplotlib draws hatches in the edge colour: match the contour exactly.
                body.set_hatch(IX_PAIR_HATCH)
                body.set_hatch_linewidth(IX_PAIR_HATCH_LW)
            # The stats go INSIDE their own half, offset off the spine -- drawn on the
            # centre they would be one mark for two distributions.
            off = (-1 if half == 0 else 1) * IX_SPLIT_STAT_OFF * IX_SPLIT_W
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.vlines(x + off, q1, q3, color=INK, lw=IX_IQR_LW, zorder=3)
            ax.plot(x + off, med, "o", color="white", ms=IX_MED_MS, zorder=4)


def _interaction_pair_blocks(ax, vals, cols, xs):
    """The boxen panel's letter-value stack, SPLIT the way the violins are: each band is
    half its full width, generated on the left of the spine and redocked on the right.

    `vals` is flat and interleaved as in _interaction_pair_bodies; `cols` is one per ARM."""
    reach = 0.0
    for i, (x, col) in enumerate(zip(xs, cols)):
        for half, v in enumerate((vals[2 * i], vals[2 * i + 1])):
            for lo, hi, d in _interaction_letter_bands(v):
                # Half of the full band width, so the two conditions together occupy the
                # same envelope one undivided block used to.
                hw = IX_SPLIT_W / 2 ** d / 2
                x0 = x - hw if half == 0 else x
                face = _interaction_tint(col, IX_BOXEN_LIGHT * d / (IX_BOXEN_K - 1)) \
                    + (IX_BODY_ALPHA,)
                # THE METHOD'S OWN COLOUR, as the violins beside it already use (2026-09-14):
                # _interaction_pair_bodies strokes each body in `col`, so the blocks stroking
                # in the neutral AXIS grey made the two panels of one figure look like two
                # conventions. The face is a tint of the same hue, so the stroke reads as the
                # saturated edge of its own block rather than as a foreign rule.
                # IT ALSO FIXES THE HATCH. matplotlib draws a hatch in the patch's EDGE
                # colour, so the redocked bars were grey here and method-coloured in the
                # violin panel -- the same encoding drawn two ways in one figure.
                ax.add_patch(IX_RECTANGLE((x0, lo), hw, hi - lo, zorder=3 + d,
                                          facecolor=face,
                                          edgecolor=col,
                                          linewidth=IX_BOXEN_EDGE_LW,
                                          hatch_linewidth=IX_PAIR_HATCH_LW,
                                          hatch=IX_PAIR_HATCH if half else None))
                reach = max(reach, hi)
            off = (-1 if half == 0 else 1) * IX_SPLIT_STAT_OFF * IX_SPLIT_W
            ax.plot(x + off, np.median(v), "o", color="white", ms=IX_MED_MS,
                    zorder=3 + IX_BOXEN_K)
    return reach


def _interaction_pair_panel(ax, kind, rows, xs, part=None):
    # The form and the fallback name come from the shared map; the WRAPPED name is this
    # figure's own, because IX_TYPE is the siblings' too. See IX_PAIR_YLABEL.
    ylabel, form = IX_TYPE[kind]
    ylabel = IX_PAIR_YLABEL.get(kind, ylabel)
    # Flat and interleaved: gen then docked for each arm, so the drawing helpers read
    # vals[2i] and vals[2i+1] as the two halves of the body at xs[i]. ONE colour per arm --
    # the halves of a split body are the same method and so the same hue; what tells them
    # apart is the side and the hatch.
    vals = [_interaction_counts(rows[arm, key], kind)
            for arm in IX_PAIR_ARMS for key, _ in IX_SERIES]
    cols = [pale(IX_COLOR_KEY.get(arm, arm)) for arm in IX_PAIR_ARMS]
    ax.set_facecolor("white")

    if form == "violin":
        _interaction_pair_bodies(ax, vals, cols, xs)
        reach = max(max(v) for v in vals)
        cells = None
    else:
        reach = _interaction_pair_blocks(ax, vals, cols, xs)
        cells = reach

    # _interaction_frame lays out the shared furniture, then the x is rebuilt: its limits
    # come from these positions rather than from a count of columns, and the category labels
    # go back on -- one per METHOD, on the split body's own spine. See the banner above.
    _interaction_frame(ax, vals, ylabel, reach * IX_HEAD_BARE, caption=False, cells=cells)
    # The siblings' axis weight and type (2026-09-15), so the three interaction figures
    # carry one look. Applied BEFORE the x axis is rebuilt: it sets a tick labelsize and
    # colour for both axes, and the method names below want their own size and INK.
    _interaction_axis_weight(ax, IX_PAIR_AXIS_LW, IX_PAIR_TITLE_FS, IX_PAIR_TICK_FS)
    # BOTH LINES CENTRED (2026-09-15). The y name is rotated 90 degrees, so a two-line
    # title's lines stack ACROSS the axis rather than along it; on the default the shorter
    # line hangs off one end and the block reads as leaning away from the spine.
    ax.yaxis.label.set_multialignment("center")
    # Set AFTER _interaction_frame, which applies the shared labelpad=10. See IX_PAIR_LABELPAD.
    ax.yaxis.labelpad = IX_PAIR_LABELPAD
    # The fixed range, for the kinds that ask for one. AFTER _interaction_frame, which has
    # already set a data-derived ylim, and keeping that function's -0.02 * top baseline so the
    # zero line sits where it does on every other panel. See IX_PAIR_YCUT.
    cut = IX_PAIR_YCUT.get(kind)
    if cut is not None:
        cut_top, tick_top, cut_step = cut
        ax.set_ylim(-0.02 * cut_top, cut_top)
        # EVERY TICK KEEPS ITS LABEL, 25 INCLUDED. It was blanked briefly to stop it
        # overhanging the top spine; the 26 limit already pulls it inside, so the blanking
        # was redundant and the number is back. See IX_PAIR_YCUT.
        ax.set_yticks(list(range(0, tick_top + 1, cut_step)))
    ax.set_xlim(xs[0] - IX_SPLIT_W, xs[-1] + IX_SPLIT_W)
    ax.set_xticks(xs)
    # ROTATED, AND ONLY ONE PART NAMES THEM (2026-09-15) -- the siblings' rule, IX_NAMED_PARTS,
    # brought over here so the two pair figures can stack. `part` None keeps the names, for a
    # caller drawing one panel on its own.
    labels = ax.set_xticklabels([_interaction_display(IX_COLOR_KEY.get(a, a))
                                 for a in IX_PAIR_ARMS],
                                fontsize=IX_PAIR_NAME_FS, color=INK,
                                rotation=30, ha="right", rotation_mode="anchor")
    if part is not None and part not in IX_NAMED_PARTS:
        # DRAWN BUT INVISIBLE, not absent. tight_layout measures a text's extent whatever its
        # colour, so colouring them "none" reserves the exact band of height they would have
        # taken -- which is what keeps this panel's axes the same height as the named one it
        # sits above. Dropping the labels instead hands that band to the axes and the two
        # rows come out carrying different vertical scales for the same categories.
        for t in labels:
            t.set_color("none")
    ax.tick_params(axis="x", length=0, pad=6)
    return form


def _interaction_pair_legend(fig, form):
    """Two swatches, and they name the POSE. The method is on the x axis (see the banner
    above), so this key carries only the thing the pair is about, drawn the way the panel
    draws it: plain for generated, hatched for redocked, in a neutral grey that cannot be
    mistaken for one of the method colours."""
    edge = IX_SWATCH if form == "violin" else AXIS
    lw = 1.2 if form == "violin" else IX_BOXEN_EDGE_LW
    handles = [Patch(facecolor=IX_TO_RGB(IX_SWATCH) + (IX_BODY_ALPHA,), edgecolor=edge,
                     lw=lw, hatch=(IX_HATCH if key == "docked" else None), label=name)
               for key, name in IX_SERIES]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
                     ncol=len(handles), frameon=True, facecolor="white",
                     edgecolor=LEGEND_EDGE, framealpha=1.0, borderpad=0.5,
                     fontsize=IX_LEGEND_SIZE, handlelength=1.5, handleheight=1.0,
                     handletextpad=0.6, columnspacing=1.9, labelspacing=0.55)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _interaction_pair_panels(out, part, kinds, rows):
    xs = _interaction_pair_positions()
    fig, axes = plt.subplots(1, 2, dpi=220,
                             figsize=(IX_PAIR_ASPECT * IX_FIG_H,
                                      IX_FIG_H * IX_PAIR_TALL))
    fig.patch.set_facecolor("white")
    form = None
    for ax, kind in zip(axes, kinds):
        form = _interaction_pair_panel(ax, kind, rows, xs, part)
    # NO LEGEND, and so no reserved strip (2026-09-14): the condition key moved out to
    # fig-interaction-legend, which names the whole family in one image rather than each
    # figure naming a part of it. The rect that used to hold the key back would leave an
    # empty band under the panels.
    fit(fig, pad=0.5, w_pad=2.2)
    # `all`, not `core` (2026-09-15): the redocking now covers every arm, and the family's
    # rule is that the variant is never only in the content. It was `core` while the
    # published baselines' poses would each have needed their own ~1,000 docks; they have
    # since been docked, so the name would otherwise be a lie about what is in the figure.
    # interaction_<part>_pair_all, not interaction_pair_<part>_all (2026-09-14): the part
    # is what the figure draws and the pair is how it draws it, so the files of one part sort
    # together -- contacts-all, contacts-core, contacts-pair-all, then the hbonds three.
    save(fig, out, f"interaction_{part}_pair_all")


# ══════════════════════════════════════════════════════════════════════════════════
# fig-interaction-legend — the family's key, as its own image
# ══════════════════════════════════════════════════════════════════════════════════
# Every figure in this family has now given up its own key: the methods went onto the x axis
# of the H-bond panel, the contacts panel carries them invisibly so it can stack under it,
# and the pair figure's condition key came off. So nothing names the encoding any more, and
# this draws it once to sit beside the panels.
#
# THREE SECTIONS, SEPARATED BY RULES, because the key carries three kinds of thing and a
# reader should not have to work out which is which: the crystal ligands, which are the
# benchmark and not a method; the eight methods; and the two POSE CONDITIONS, which are not
# methods either and are drawn in a neutral grey for exactly that reason -- giving them a
# method colour would say the opposite of what they mean.
#
# PAINTED BY THE PANELS' OWN RULE: a translucent face under a stroke of the same saturated
# hue, and the redocked swatch hatched as its column is. A key that does not look like the
# thing it names is a second key to learn.
IX_LEG_SW_W = 0.30                # swatch width, as a share of the key's width
IX_LEG_SW_H = 0.52                # swatch height, in row units
IX_LEG_ROW_H = 0.33               # inches per row
IX_LEG_W = 2.7                    # inches


@figure("fig-interaction-legend", folder="fig-posecheck/interaction")
def draw_interaction_legend(out):
    """Reference, a rule, the eight methods, a rule, then generated and redocked."""
    use_style()
    methods = [m for m in IX_ORDER if m != REF_LABEL]
    # THE TEXT IS RESOLVED HERE, not in the drawing loop. _interaction_display() is strict --
    # it defers to the shared display(), which RAISES on a label the palette does not know --
    # and that is right for a method name, where a typo should fail loudly rather than print
    # itself. But the last two rows are POSE CONDITIONS, not methods, so they carry their own
    # text and never reach it.
    rows = ([(_interaction_display(REF_LABEL), pale(REF_LABEL), None)]
            + [(_interaction_display(m), pale(m), None) for m in methods]
            + [("Generated", IX_SWATCH, None), ("Redocked", IX_SWATCH, IX_HATCH)])
    # A rule under the reference and under the last method: the two places the KIND of
    # entry changes, which is the only thing a rule should ever mark here.
    rules = {0, len(methods)}
    with plt.rc_context({"hatch.linewidth": IX_HATCH_LW}):
        fig, ax = plt.subplots(figsize=(IX_LEG_W, IX_LEG_ROW_H * len(rows) + 0.3), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-len(rows) + 0.5, 0.5)
        ax.axis("off")
        for i, (lab, col, hatch) in enumerate(rows):
            y = -i
            ax.add_patch(IX_RECTANGLE((0.05, y - IX_LEG_SW_H / 2), IX_LEG_SW_W,
                                      IX_LEG_SW_H,
                                      facecolor=IX_TO_RGB(col) + (IX_BODY_ALPHA,),
                                      edgecolor=col, linewidth=1.2, hatch=hatch,
                                      zorder=3))
            ax.text(0.05 + IX_LEG_SW_W + 0.07, y, lab,
                    ha="left", va="center", fontsize=IX_LEGEND_SIZE, color=INK)
            if i in rules:
                ax.plot([0.03, 0.97], [y - 0.5, y - 0.5], color=LEGEND_EDGE,
                        lw=AXIS_LW, zorder=2)
        # One frame round the whole key, as the house legend box draws it.
        ax.add_patch(IX_RECTANGLE((0.0, -len(rows) + 0.5), 1.0, len(rows),
                                  facecolor="none", edgecolor=LEGEND_EDGE,
                                  linewidth=AXIS_LW, zorder=1))
        fit(fig, pad=0.3)
        save(fig, out, "interaction_legend")
    print(f"  key: reference · {len(methods)} methods · {len(IX_SERIES)} pose conditions")
    print(f"  wrote {out}")


@figure("fig-posecheck-legend", folder="fig-posecheck")
def draw_posecheck_legend(out):
    """Condition key in the first column, then eight methods in a 2 x 5 grid."""
    use_style()
    methods = [(_interaction_display(m), pale(m), False) for m in METHOD_ORDER]
    entries = ([('Generated', IX_SWATCH, False), *methods[:4],
                ('Relaxed', IX_SWATCH, True), *methods[4:]])
    fig, ax = plt.subplots(figsize=(16.0, 2.2), dpi=220)
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2)
    ax.axis('off')
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.08, top=0.92)
    for i, (label, col, relaxed) in enumerate(entries):
        x, y = i % 5, 1.5 - i // 5
        swatch_x = x + 0.06
        patch = IX_RECTANGLE((swatch_x, y - 0.20), 0.24, 0.40,
                             facecolor=IX_TO_RGB(col) + (IX_BODY_ALPHA,),
                             edgecolor=INK if relaxed else col, linewidth=1.2, zorder=2)
        ax.add_patch(patch)
        if relaxed:
            # Draw the slashes explicitly so the condition remains obvious in PNG,
            # SVG and PDF, including at the small size of a paper legend.
            for start in np.arange(swatch_x - 0.12, swatch_x + 0.24, 0.075):
                line, = ax.plot((start, start + 0.12), (y - 0.20, y + 0.20),
                                color=INK, linewidth=2 * IX_HATCH_LW, zorder=3)
                line.set_clip_path(patch)
        ax.text(x + 0.37, y, label, ha='left', va='center',
                fontsize=TITLE_SIZE, color=INK)
    ax.add_patch(IX_RECTANGLE((0.01, 0.02), 4.98, 1.96,
                              facecolor='none', edgecolor=LEGEND_EDGE,
                              linewidth=AXIS_WIDTH))
    ax.plot((1.0, 1.0), (0.02, 1.98), color=LEGEND_EDGE, linewidth=AXIS_WIDTH)
    save(fig, out, 'posecheck_legend')


def _interaction_pair_row(arm, kind, rows):
    """One CSV line: the type's mean and median under both conditions, and the change."""
    g = np.asarray(_interaction_counts(rows[arm, "gen"], kind), float)
    d = np.asarray(_interaction_counts(rows[arm, "docked"], kind), float)
    return [arm, kind, len(g), round(g.mean(), 3), round(d.mean(), 3),
            round(d.mean() - g.mean(), 3), np.median(g), np.median(d),
            round(100 * (g > 0).mean(), 2), round(100 * (d > 0).mean(), 2)]


@figure("fig-interaction-pair", folder="fig-posecheck/interaction",
        needs=("interactions_<Arm>-gen.json", "interactions_<Arm>-docked.json"))
def draw_interaction_pair(out):
    """Generated vs redocked, paired, for every arm — contacts and H-bonds."""
    use_style()
    rows = _interaction_pair_load()
    _interaction_pair_check(rows)
    for part, kinds in IX_PARTS:
        _interaction_pair_panels(out, part, kinds, rows)

    write_csv(out, "interactions_pair",
              ["arm", "type", "n_molecules", "gen_mean", "docked_mean", "delta_mean",
               "gen_median", "docked_median", "share_nonzero_gen_pct",
               "share_nonzero_docked_pct"],
              [_interaction_pair_row(arm, kind, rows)
               for arm in IX_PAIR_ARMS for kind in IX_KINDS])

    print(f"  paired generated / redocked poses · {len(rows['CoDE', 'gen'])} molecules for "
          f"the model arms, {len(rows['Reference', 'gen'])} crystal ligands\n")
    head = " ".join(f"{k:>21s}" for k in IX_KINDS)
    print(f"  {'arm':12s} {head}")
    for arm in IX_PAIR_ARMS:
        cells = []
        for kind in IX_KINDS:
            g = np.mean(_interaction_counts(rows[arm, "gen"], kind))
            d = np.mean(_interaction_counts(rows[arm, "docked"], kind))
            cells.append(f"{g:7.2f} ->{d:6.2f} {d - g:+5.2f}")
        print(f"  {_interaction_display(IX_COLOR_KEY.get(arm, arm)):12s} "
              + " ".join(cells))
    print("\n  mean per molecule, generated -> redocked and the change. The crystal ligands "
          "are\n  the control: they are already in a measured pose, so redocking should "
          "move them\n  least.")


# ══════════════════════════════════════════════════════════════════════════════════
# fig-interaction-pair-stacked — both parts of the pair figure in one frame
# ══════════════════════════════════════════════════════════════════════════════════
# The two pair figures were always meant to stack: contacts draws its method names in
# colour "none" and H-bonds draws them for both, which is only worth doing if something
# puts one above the other. This is that something -- the same four panels, in one image,
# so the whole comparison is a single figure rather than two that a document has to be
# trusted to keep together and in order.
#
# NOTHING ABOUT THE PANELS CHANGES. It reuses _interaction_pair_panel, so the split bodies,
# the colours, the hatch and the axis weight are the ones the standalone figures draw; only
# the frame is this figure's own. A panel that looked different here would be a second
# convention for the same measurement.
IX_STACK_PAD = 0.06              # inches of air outside the ink, all four sides
IX_STACK_ROW_GAP = 0.10          # inches between the two rows -- "almost none", as asked


def _interaction_stack_layout(fig, axes):
    """Place the stacked grid BY HAND rather than by tight_layout.

    WHY NOT fit(). tight_layout sizes a row to its own content, so the row carrying the
    method names would come out shorter than the one that does not -- and h_pad barely
    touches it: measured 1.18 in of row gap at h_pad 1.4 and still 0.89 in at h_pad 0.0,
    because the gap IS the name band and not padding. Setting the margins directly gives
    both rows exactly the gridspec's height and puts the gap where it was asked for.

    THE INSETS ARE MEASURED, NOT ASSUMED. `left` is the widest y name plus its tick labels
    over all four panels (they differ: the contacts row needed 0.657 in and the H-bond row
    0.559 in, and using the smaller would clip the wider one). `bottom` is what the rotated
    names of the named row need. An inset is how far a panel's decorations overhang its own
    box, so it does not depend on where that panel currently sits -- which is what makes it
    safe to measure before the final placement.

    wspace and hspace are fractions of the AXES size, and the axes size is the thing they
    change, so the fraction that yields a wanted absolute gap has to be solved for:
    gap = s*plot/(n + s)  =>  s = n*gap/(plot - gap)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    flat = [ax for row in axes for ax in row]
    left_in = max(ax.get_position().x0 * W - ax.get_tightbbox(r).x0 / fig.dpi
                  for ax in flat)
    bot_in = max(ax.get_position().y0 * H - ax.get_tightbbox(r).y0 / fig.dpi
                 for ax in axes[-1])
    left, bottom = (left_in + IX_STACK_PAD) / W, (bot_in + IX_STACK_PAD) / H
    right, top = 1.0 - IX_STACK_PAD / W, 1.0 - IX_STACK_PAD / H
    plot_w, plot_h = (right - left) * W, (top - bottom) * H
    # The columns are separated by exactly the room the right column's own y name needs.
    col_gap = left_in + IX_STACK_PAD
    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom,
        wspace=len(axes[0]) * col_gap / (plot_w - col_gap),
        hspace=len(axes) * IX_STACK_ROW_GAP / (plot_h - IX_STACK_ROW_GAP),
    )


@figure("fig-interaction-pair-stacked", folder="fig-posecheck/interaction",
        needs=("interactions_<Arm>-gen.json", "interactions_<Arm>-docked.json"))
def draw_interaction_pair_stacked(out):
    """Generated vs redocked, paired, every arm — contacts above H-bonds in one frame."""
    use_style()
    rows = _interaction_pair_load()
    _interaction_pair_check(rows)
    xs = _interaction_pair_positions()
    # Two rows of the single figure's frame. The height doubles and the width does not: the
    # rows share one x axis of methods, which is the point of stacking them.
    fig, axes = plt.subplots(len(IX_PARTS), 2,
                             figsize=(IX_PAIR_ASPECT * IX_FIG_H,
                                      len(IX_PARTS) * IX_FIG_H * IX_PAIR_TALL),
                             dpi=220)
    fig.patch.set_facecolor("white")
    for row, (part, kinds) in enumerate(IX_PARTS):
        for ax, kind in zip(axes[row], kinds):
            _interaction_pair_panel(ax, kind, rows, xs, part)
    # THE UNNAMED ROW DROPS ITS LABELS OUTRIGHT, which the STANDALONE figures must never do.
    # There the names are drawn in colour "none" so tight_layout still measures them and the
    # two separate figures keep the same axes height. Inside ONE figure the row heights come
    # from the gridspec instead, so that reservation buys nothing and costs the entire band
    # -- 0.889 in of empty space between the rows, measured.
    for ax in axes[0]:
        ax.set_xticklabels([])
    # THE Y NAMES LINE UP ACROSS THE ROWS (2026-09-15). Each one is placed off its own tick
    # labels, and the rows carry different numbers -- two digits on the contacts row, one on
    # the H-bond row -- so left to itself the upper title sits further out than the lower and
    # the two read as a ragged edge even though the AXES are aligned to 0.0000 in. This pins
    # them to a common x. It runs BEFORE the layout because _interaction_stack_layout sizes
    # the left margin from the measured label overhang, which this changes.
    fig.align_ylabels([ax for row in axes for ax in row])
    _interaction_stack_layout(fig, axes)
    save(fig, out, "interaction_pair_all")
    print(f"  {len(IX_PARTS)} x 2 panels · {len(IX_PAIR_ARMS)} arms · "
          f"names on {', '.join(IX_NAMED_PARTS)} only")
# What the crystal ligands are CALLED in this family's key (2026-09-14). The shared REF_LABEL
# ("Reference ligand") still names them everywhere else; only the drawn key is shortened, and
# _pcsz_key splits the key on this name.
PCSZ_REF_NAME = "Reference"

# The five shared size bins plus the pooled one every molecule ALSO lands in.
PCSZ_LABELS = BIN_LABELS + ["all sizes"]
PCSZ_SLUGS = ["le15", "16_20", "21_25", "26_30", "gt30", "all"]
PCSZ_POOLED = len(PCSZ_LABELS) - 1
# The axis starts at 10 kcal/mol. Below that a pose is effectively unstrained, and spending
# decades of width on it pushed the region where the methods separate into the right half.
# Values under the floor are clipped onto it, so the curve enters the axis at its true share.
PCSZ_XFLOOR, PCSZ_XTOP = 1e1, 2e3
# The clash axis is log10(x+1), so it can hold the zero-clash poses; these are the labels put
# back on it in real counts (2026-09-14).
PCSZ_CLASH_TICKS = [0, 1, 10, 100]
PCSZ_ECDF_TICK_SCALE = PCSZ_COMPARISON_TICK_SCALE


def _pcsz_empty():
    return {b: {"strain": [], "clash": []} for b in range(len(PCSZ_LABELS))}


def _pcsz_add(out, n, s, c):
    for b in (bin_of(n), PCSZ_POOLED):
        if s is not None and np.isfinite(s):
            out[b]["strain"].append(float(s))
        if c is not None and np.isfinite(c):
            out[b]["clash"].append(float(c))


def _pcsz_load_baseline(method, keep):
    """One of the five, from its per-molecule export. `keep` is a set of pocket indices."""
    d = json.load(open(legacy("fig-posecheck", f"posecheck_{method}.json"),
                       encoding="utf-8"))
    out = _pcsz_empty()
    for m in d["molecules"]:
        if m["p"] in keep:
            _pcsz_add(out, m["n"], m["s"], m["c"])
    return out, set(d["density79_pockets"])


def _pcsz_metrics(root):
    """Every per-target metrics.json under one run root, in path order."""
    return sorted(str(p) for p in Path(root).glob("target_*/metrics.json"))


def _pcsz_load_local(root, keep, reference=False):
    """One of ours, from per-target metrics.json."""
    out = _pcsz_empty()
    seen = set()
    for path in _pcsz_metrics(root):
        idx = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if idx not in keep:
            continue
        seen.add(idx)
        j = json.load(open(path, encoding="utf-8"))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m or not m.get("n_atoms"):
                continue
            pc = m.get("posecheck") or {}
            _pcsz_add(out, m["n_atoms"], pc.get("strain"), pc.get("clashes"))
    return out, seen


def _pcsz_pockets_of(root):
    return {int(os.path.basename(os.path.dirname(p)).split("_")[1])
            for p in _pcsz_metrics(root)}


def _pcsz_stats(cell):
    s_, c_ = cell["strain"], cell["clash"]
    return {
        "n_strain": len(s_), "n_clash": len(c_),
        "strain_median": round(st.median(s_), 1) if s_ else None,
        "strain_q25": round(float(np.percentile(s_, 25)), 1) if s_ else None,
        "strain_q75": round(float(np.percentile(s_, 75)), 1) if s_ else None,
        "clash_mean": round(float(np.mean(c_)), 2) if c_ else None,
        "clash_median": round(float(np.median(c_)), 1) if c_ else None,
    }


_PCSZ_CACHE = {}


def _pcsz_data():
    """(per-method {bin: {strain, clash}}, the drawn series, the reference, the summary).

    Cached for the life of the process: the two figures below are the same nine series
    read once and drawn twice, and the verification pass alone re-bins every baseline
    molecule five times."""
    if "d" in _PCSZ_CACHE:
        return _PCSZ_CACHE["d"]

    # ── the pocket subsets ────────────────────────────────────────────────────
    probe = json.load(open(legacy("fig-posecheck", "posecheck_AR.json"), encoding="utf-8"))
    density79 = set(probe["density79_pockets"])
    shared = set(density79)
    for _, root, *_ in PCSZ_LOCAL:
        shared &= _pcsz_pockets_of(root)
    dropped = sorted(density79 - shared)
    print(f"  pockets: baselines {len(density79)} · shared with our three {len(shared)}"
          f" · dropped {['target_%02d' % i for i in dropped]}")

    # ── verification: the exports must reproduce the stored per-bin numbers ───
    stored = json.load(open(legacy("fig-posecheck",
                                   "posecheck_baselines_by_atom_range.json"),
                            encoding="utf-8"))
    mismatch = 0
    for key, label, *_ in PCSZ_BASELINES:
        data, _ = _pcsz_load_baseline(key, density79)
        for b in range(len(EDGES) - 1):
            want, got = stored["methods"][label][b], _pcsz_stats(data[b])
            for a, c in (("n", "n_strain"), ("strain_median", "strain_median"),
                         ("clash_mean", "clash_mean")):
                if want.get(a) != got.get(c):
                    mismatch += 1
                    print(f"    MISMATCH {label} bin {PCSZ_LABELS[b]} {a}: "
                          f"stored {want.get(a)} != rebuilt {got.get(c)}")
    if mismatch:
        raise SystemExit(f"loader disagrees with the stored baseline JSON on {mismatch} "
                         f"values -- refusing to plot")
    print("  verification: exports reproduce posecheck_baselines_by_atom_range.json "
          "on 5 methods x 5 bins x 3 statistics, 0 mismatches")

    # ── the data actually plotted, all on the shared pockets ──────────────────
    data_by_label, series = {}, []
    for key, label, lw, ls in PCSZ_BASELINES:
        d79, _ = _pcsz_load_baseline(key, density79)
        dsh, _ = _pcsz_load_baseline(key, shared)
        data_by_label[label] = dsh
        series.append((label, pale(label), lw, ls))
        a, b = _pcsz_stats(d79[PCSZ_POOLED]), _pcsz_stats(dsh[PCSZ_POOLED])
        print(f"    {label:12s} 79 pockets n={a['n_clash']:5d} clash {a['clash_mean']:5.2f} "
              f"strain {a['strain_median']:7.1f}   ->  {len(shared)} pockets "
              f"n={b['n_clash']:5d} clash {b['clash_mean']:5.2f} strain {b['strain_median']:7.1f}")
    for label, root, lw, ls in PCSZ_LOCAL:
        data_by_label[label], _ = _pcsz_load_local(root, shared)
        series.append((label, pale(label), lw, ls))
    ref, _ = _pcsz_load_local(REF_ROOT, shared, reference=True)

    # ── the numbers behind both figures ──────────────────────────────────────
    methods = {label: [_pcsz_stats(data_by_label[label][b])
                       for b in range(len(PCSZ_LABELS))] for label, *_ in series}
    methods[REF_LABEL] = [
        {**_pcsz_stats(ref[b]), "clash_mean": None, "clash_median": None,
         "note": "strain only"} for b in range(len(PCSZ_LABELS))]

    print(f"\n  {'bin':10s} {'method':14s} {'n':>6s} {'strain med':>11s} {'IQR':>19s} "
          f"{'clash mean':>11s} {'med':>5s}")
    for b, lab in enumerate(PCSZ_LABELS):
        for label in methods:
            r = methods[label][b]
            if not r["n_strain"]:
                continue
            iqr = (f"{r['strain_q25']:7.1f}–{r['strain_q75']:<10.1f}"
                   if r["strain_q25"] is not None else " " * 18)
            cm = f"{r['clash_mean']:11.2f}" if r["clash_mean"] is not None else f"{'—':>11s}"
            cd = f"{r['clash_median']:5.1f}" if r["clash_median"] is not None else f"{'—':>5s}"
            print(f"  {lab:10s} {label:14s} {r['n_strain']:6d} {r['strain_median']:11.1f} "
                  f"{iqr} {cm} {cd}")
        print()

    _PCSZ_CACHE["d"] = (data_by_label, series, ref, methods)
    return _PCSZ_CACHE["d"]


PCSZ_KEY_NCOL = 2
# The reference line is anchored on the grid's top edge, so the gap between them is the two
# legends' borderpad back to back. Pull it down by this much of the axes to close half of it.
PCSZ_KEY_GAP = 0.022


def _pcsz_key(fig, ax):
    """The ECDF key: the crystal ligands on a centred line of their own, over the eight methods
    in 4 rows x 2 columns reading across, in one frame in the lower right.

    Built like the Vina per-atom key (_vpa_legend): TWO legends inside ONE rectangle. A legend
    column is as wide as its widest entry and matplotlib cannot span a cell, so the reference
    cannot sit centred over the grid as a cell of it. The grid is placed first, the reference
    legend is centred on it and stacked on top, both frames come off, and a single rectangle is
    drawn round their union. Call AFTER the layout is final: positions are axes fractions of
    the axes as laid out when this runs."""
    pairs = list(zip(*ax.get_legend_handles_labels()))
    ref = [p for p in pairs if p[1] == PCSZ_REF_NAME]
    methods = [p for p in pairs if p[1] != PCSZ_REF_NAME]
    rows = -(-len(methods) // PCSZ_KEY_NCOL)
    # matplotlib fills a legend column by column; interleave so the grid reads across.
    methods = [methods[r * PCSZ_KEY_NCOL + c] for c in range(PCSZ_KEY_NCOL)
               for r in range(rows) if r * PCSZ_KEY_NCOL + c < len(methods)]
    grid = ax.legend(*zip(*methods), loc="lower right", ncol=PCSZ_KEY_NCOL, borderaxespad=0.5,
                     frameon=False, **PCSZ_KEY_KW)
    legs = [grid]
    render = fig.canvas.get_renderer
    if ref:
        ax.add_artist(grid)          # the next ax.legend() would otherwise replace it
        fig.canvas.draw()
        bb = grid.get_window_extent(render()).transformed(ax.transAxes.inverted())
        legs.append(ax.legend(*zip(*ref), loc="lower center", borderaxespad=0, frameon=False,
                              bbox_to_anchor=((bb.x0 + bb.x1) / 2, bb.y1 - PCSZ_KEY_GAP),
                              **PCSZ_KEY_KW))
    fig.canvas.draw()
    boxes = [l.get_window_extent(render()).transformed(ax.transAxes.inverted()) for l in legs]
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    # An AXES artist, not a figure one: a figure-level patch is drawn after the whole axes and
    # would cover the legend text. Translucent white under the text, over the curves.
    ax.add_artist(matplotlib.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes, facecolor=PCSZ_KEY_FACE,
        edgecolor=LEGEND_EDGE, linewidth=0.7, zorder=5, clip_on=False))
    for l in legs:
        l.set_zorder(6)
        for text in l.get_texts():   # identity rides the swatch, not the ink
            text.set_color(PCSZ_INK)


def _pcsz_csv(out, methods):
    write_csv(out, "posecheck_all_by_atom_range",
              ["bin", "method", "n_strain", "strain_median", "strain_q25", "strain_q75",
               "n_clash", "clash_mean", "clash_median"],
              [[lab, label, r["n_strain"], r["strain_median"], r["strain_q25"],
                r["strain_q75"], r["n_clash"], r["clash_mean"], r["clash_median"]]
               for b, lab in enumerate(PCSZ_LABELS)
               for label, r in ((k, v[b]) for k, v in methods.items())])


@figure("fig-posecheck-strain-ecdf-by-size", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_strain_ecdf_by_size(out):
    """UFF strain ECDF, nine methods on one log axis, per heavy-atom bin."""
    data_by_label, series, ref, methods = _pcsz_data()
    # Preserve the shared canvas at export instead of cropping each panel differently.
    with plt.rc_context({**PCSZ_RC, "savefig.bbox": None}):
        for b, (lab, slug) in enumerate(zip(PCSZ_LABELS, PCSZ_SLUGS)):
            size = PCSZ_MAIN_ECDF_SIZE if slug == "all" else PCSZ_STRAIN_SIZE
            fig, ax = plt.subplots(figsize=size)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax)
            for label, colour, lw, ls in series:
                v = np.asarray(data_by_label[label][b]["strain"], dtype=float)
                if v.size < 20:
                    continue
                x = np.sort(np.clip(v, PCSZ_XFLOOR, None))
                # pale(): CoDE in its lighter tint, as in the eight-method strain line figures
                # (2026-09-13). The clash violin below still draws the palette colour.
                # display(): the key carries VoxBind's sigma as a subscript (2026-09-14). It is
                # the ONLY name DISPLAY overrides, so every other entry is unchanged, and the
                # series keeps its plain label everywhere else -- CSV, violin ticks, data keys.
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=pale(label),
                        lw=lw * PCSZ_LW_SCALE, ls=ls,
                        label=display(label), solid_capstyle="round")
            r = np.asarray(ref[b]["strain"], dtype=float)
            if r.size >= 3:
                x = np.sort(np.clip(r, PCSZ_XFLOOR, None))
                ax.plot(x, np.arange(1, x.size + 1) / x.size, color=REF_COLOR,
                        # SOLID, and AR took its dash (2026-09-14): solid was the one style no
                        # baseline had, so it now marks the crystal ligands and our two arms --
                        # the series a reader returns to -- and every baseline carries a dash.
                        lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls="-",
                        label=PCSZ_REF_NAME, solid_capstyle="round")
            ax.set_xscale("log")
            ax.set_xlim(PCSZ_XFLOOR, PCSZ_XTOP)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.1, 0.4, 0.7, 1.0])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
            # "· log scale" dropped (2026-09-14); the decade ticks say it.
            ax.set_xlabel("UFF strain energy", fontsize=PCSZ_LABEL_FS)
            # Two lines (2026-09-14), as on the per-atom panel it pairs with.
            ax.set_ylabel("Cumulative\nprobability", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
            # No title (removed on request 2026-09-13): the size bin is carried by the file
            # name (`ecdf-by-size-<bin>`) and by whatever caption places the figure.
            # The method key is supplied separately from figures/fig-legend.
            _pcsz_comparison_layout(fig, ax, xunit="kcal mol⁻¹",
                                    tick_scale=PCSZ_ECDF_TICK_SCALE)
            save(fig, out, f"strain_ecdf_{slug}")
    _pcsz_csv(out, methods)
    print(f"  wrote {out}")


@figure("fig-posecheck-clash-violin-by-size", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_violin_by_size(out):
    """Steric clashes per pose, eight methods as violins, per heavy-atom bin."""
    data_by_label, series, _ref, methods = _pcsz_data()
    with plt.rc_context(PCSZ_RC):
        for b, (lab, slug) in enumerate(zip(PCSZ_LABELS, PCSZ_SLUGS)):
            # The strain ECDF's canvas (2026-09-14), so the two PoseCheck measurements are the
            # same shape on a page; its rc is already in force for the whole family.
            fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax, grid_axis="y")
            vals, cols, ticks = [], [], []
            for label, colour, lw, ls in series:
                v = data_by_label[label][b]["clash"]
                if len(v) < 20:
                    continue
                vals.append(v)
                # pale(), as every other eight-method figure draws (2026-09-14). `colour` off
                # the series table is the saturated palette entry, which now only the
                # three-arm and docking figures use.
                cols.append(pale(label))
                ticks.append(display(label))
            # KDE fitted on log10(x+1) and the ticks relabelled with real counts: a plain
            # log axis is impossible because a few percent of poses have zero clashes, and
            # symlog would warp only the display while the KDE stayed in data space.
            tf = lambda v: np.log10(np.asarray(v, dtype=float) + 1.0)
            parts = ax.violinplot([tf(v) for v in vals], showextrema=False, widths=0.82)
            for body, colour in zip(parts["bodies"], cols):
                body.set_facecolor(colour)
                body.set_alpha(DATA_ALPHA)
                body.set_edgecolor(colour)
                body.set_linewidth(1.2)
            for i, v in enumerate(vals, start=1):
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.vlines(i, tf(q1), tf(q3), color="#14181f", lw=5, zorder=3)
                ax.plot(i, tf(med), "o", color="white", ms=5.5, zorder=4)
            hi = max(max(v) for v in vals)
            # DECADES ONLY (2026-09-14): zero, then 1, 10, 100. The nine intermediate ticks the
            # axis used to carry were reading aids for numbers that are now in the CSV.
            ticks_at = [t for t in PCSZ_CLASH_TICKS if t <= hi * 1.6]
            ax.set_yticks(tf(ticks_at))
            ax.set_yticklabels([str(t) for t in ticks_at])
            # The headroom the per-violin mean/median strip needed is gone with it.
            ax.set_ylim(tf(0) - 0.04, tf(hi) + 0.06)
            # TURNED (2026-09-14): on the ECDF's narrower canvas eight names laid flat ran
            # into each other -- Pocket2Mol into DiffSBDD into DecompDiff. Anchored at the
            # right so each name ends under its own violin.
            ax.set_xticks(range(1, len(vals) + 1))
            ax.set_xticklabels(ticks, fontsize=9.5, rotation=30, ha="right")
            ax.set_ylabel("Steric clashes\nper pose", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
            # NO TITLE and NO per-violin mean/med strip (2026-09-14), as on the strain ECDF:
            # the bin is in the file name (`clash-violin-by-size-<bin>`) and every number the
            # strip carried is a column of the CSV written beside the figure.
            fig.tight_layout(pad=0.5)
            save(fig, out, f"clash_violin_{slug}")
    _pcsz_csv(out, methods)
    print(f"  wrote {out}")
PCSZ_CBOX_MED_LW = 2.0 * 0.8
# Wider again (2026-09-14), on top of the width the colourbar gave back. TALL is its own
# factor because the x names went up to the y name's size: nine of them at 30 degrees eat
# height that the boxes used to have.
PCSZ_CBOX_WIDE, PCSZ_CBOX_TALL = 1.15, 1.2
# ROTATED (2026-09-14): the methods run DOWN the y axis and the clashes across x. Nine rows
# want height rather than width, and the names stop needing their 30-degree tilt -- upright
# against a y axis is the main thing rotating buys. Kept as its own pair rather than swapping
# the two above, so going back to the upright panel stays one edit.
# Width at 0.6x (2026-09-14), back up from 0.4. The method names sit OUTSIDE the axes but
# INSIDE the figure, and constrained layout fits everything within figsize -- so a cut here
# comes out of the drawing area alone, the names keeping their full width. At 0.4 that left
# the nine boxes about 285 px of a 492 px image to share, with the labels taking 42 % of it,
# and the median spread the panel exists to show (4 against 8) stopped being legible.
PCSZ_CBOX_WIDE_V, PCSZ_CBOX_TALL_V = 0.5, 1.55
# The method names' tilt. A rotated label's horizontal footprint is w*cos(t) + h*sin(t), so
# for names this long against this type size the saving is modest until the angle is steep:
# roughly 5 % at 30 degrees, 17 % at 45, 36 % at 60. Level is easier to read, so this is the
# smallest angle that buys anything rather than the one that buys most.
PCSZ_CBOX_NAME_ROT = 30
# Minor ticks, as a fraction of the frame's weight. They SUBDIVIDE a decade rather than bound
# the panel, and carrying the frame's full 2.0 made the eight of them between each pair of
# decades read as more axis lines instead of as gradations on one.
PCSZ_CBOX_MINOR_LW = 0.5
# WHERE THE VALUE AXIS STOPS (2026-09-14). Left to itself it runs to the largest flier any
# method produced -- DiffSBDD's 333 -- which spends the right third of the panel on a handful
# of points and squeezes all nine boxes into the left half. DiffSBDD is the ONLY method that
# reaches past this: the next largest maximum is TargetDiff's 167, and every other method
# tops out between 26 and 97, so the cap costs one method's extreme tail and nothing else.
PCSZ_CBOX_HI = 200.0
# ONE TICK OF AIR UNDER 10^0 (2026-09-22, on request), in decades. The value axis floor was
# tf(1) exactly, so the bottom decade sat ON the frame: the lower whiskers ran into the
# spine and the 10^0 label touched the corner. A tenth of a decade is ONE of the ten
# gradations between 10^-1 and 10^0 -- enough to lift the decade clear without opening a
# whole empty decade below the boxes. The tick lists are untouched, so nothing new is drawn
# down there: the axis just starts a little lower.
# THE ZERO CLAMP MUST STAY BELOW THIS. Zero-clash poses are placed at 0.5 -- -0.30 decades
# -- and are meant to fall OFF the panel rather than pile on its edge at a count they do not
# have. 0.1 keeps them off it with 0.2 decades to spare; a pad past ~0.30 would pull them
# into view and the figure would be quoting 0 as if it were half a clash.
PCSZ_CBOX_FLOOR_PAD = 0.1
# HOW MUCH WHITE GOES INTO A FILL (2026-09-14). The fills already carry EXACTLY the pale()
# hexes the ECDF and the per-atom panel draw their lines in -- checked on the rendered
# pixels, #EE9190 / #F0A6C0 / #8DCB92 / #C6B46A / #8291E8 / #7FC4D1 match byte for byte
# across the two files. They still read more saturated, and that is an area effect, not a
# palette drift: a filled box is ~100x the ink of a 3 px line of the same colour. So the
# FILL, and only the fill, is mixed toward white. The line palette is untouched, which is
# what keeps a method the same colour everywhere. This is the step the rotatable-bond grid
# gets for free by running its colourmap through pale().
# NO WHITE AT ALL (2026-09-14): the fills are the pale() hexes exactly as the ECDF and the
# per-atom panel draw their lines, so a method is literally one colour across the three
# figures. 0.42 read washed out and 0.2 was still a second colour for the same method; the
# area effect that motivated the blend is worth living with. This stays a knob rather than
# being deleted -- one number brings the tint back if the solid blocks prove too heavy.
PCSZ_CBOX_TINT = 0.0
# The frame's weight. Raised from this family's 1.1 to the rotatable-bond grid's 2.0
# (2026-09-14) because the panel carries the grid's boxes and type at 14 pt, and 1.1 read
# thin under both. The rule between the crystal ligands and the methods takes the SAME
# weight, because it is frame and not data -- 79 poses against ~7,500 is a different
# population, not a ninth method.
#
# DOWN TO 1.0 (2026-09-15), in two steps: 2.0 read too heavy, then the target became the JSD
# summary's proportions.
#
# WEIGHT IS RELATIVE TO THE CANVAS, which is the thing that makes this number look wrong on
# its own. jsd.py already says so -- its 2.2 pt exists because that canvas is far wider and
# gets scaled down in the document. This panel is the opposite extreme: 2.77 in wide against
# the JSD summary's 13.32, so an identical pt reads 4.8x heavier here. At 2.0 this was
# 0.72 pt/in against the JSD summary's 0.165; 1.0 lands at 0.36, roughly halfway, which is
# the correction that was asked for rather than the full column-scaled one (that would have
# meant 0.46 pt, a hairline in the raster).
#
# IT NO LONGER MATCHES RB_GRID_AXIS_LW, still 2.0 -- the two figures were deliberately given
# one frame weight and now differ. If that pairing is wanted back, raise THIS, not that one.
#
# It carries two other things down with it, both intended: the minor ticks (x
# PCSZ_CBOX_MINOR_LW, so 0.50) and the reference/method rule.
PCSZ_CBOX_AXIS_LW = 1.0
# The reference-median rule across the landscape panel: weight, and how far its grey is
# pulled toward black (0 = REF_COLOR itself).
PCSZ_CBOX_REF_LINE_LW = 1.2 * 1.6
PCSZ_CBOX_REF_LINE_DARKEN = 0.45
# The method names, two points off the y name (2026-09-14), then 0.9x on request
# (2026-09-15). Kept as an offset from PCSZ_LABEL_FS with the scale applied after, so the
# relationship to the family size still reads at a glance instead of becoming a bare number.
PCSZ_CBOX_NAME_FS = (PCSZ_LABEL_FS - 2) * 0.9
# The axis title, a little under the family's PCSZ_LABEL_FS (2026-09-15, on request). Scoped
# to this figure rather than lowering PCSZ_LABEL_FS itself, which the strain ECDF, the
# per-atom panels and the rotatable-bond figures all read -- dropping it there would retype
# five figures nobody asked to touch. THIS PANEL HAS ONLY AN X TITLE: it is drawn sideways,
# so the methods are the y tick labels and there is no y name to match.
# The same 0.9x as the method names, so the two move together (2026-09-15). NOT EVERY NUMBER
# IN THIS PANEL SCALES: the decade tick labels take their size from the shared style -- this
# figure's tick_params sets only `width` -- so they are unaffected and would need a constant
# of their own to follow.
PCSZ_CBOX_TITLE_FS = (PCSZ_LABEL_FS - 1.5) * 0.9
# AIR ON EITHER SIDE OF THE RULE (2026-09-14). Boxes are 0.66 wide on a 1.0 pitch, so two
# neighbours are 0.34 apart; at the 2 pt frame weight this was tuned against, the rule ate
# most of that, leaving less air between the reference box and AR's than between the
# reference box and the left spine. The methods slide right by this much so the rule sits in
# a margin of its own.
#
# SIZED FOR A 2 PT RULE AND LEFT ALONE when the rule dropped to 1.0 (2026-09-15). A rule at
# half the weight it was tuned against needs much less margin, so this is now well more
# generous than it has to be -- not wrong, just loose. It is the knob to trim if the
# reference box reads too far from AR's.
PCSZ_CBOX_GAP = 0.5
# The x padding past the outermost box centre, at each end.
PCSZ_CBOX_PAD = 0.7
# HOW FAR THE OUTLIERS SCATTER ACROSS THEIR OWN BAND (2026-09-16), in slot units.
#
# WHY THEY NEEDED IT. bxp puts every flier of a box on ONE y -- measured, each of the nine
# artists had exactly one y value -- and the clash count is an INTEGER, so the hundreds of
# poses sharing a count also share an x. Identical x and identical y means they are drawn
# exactly on top of each other: a method with 754 outliers and one with 311 render as the
# same flat streak, and the density the reader wants to compare is invisible.
#
# WHY ACROSS THE BAND AND NOT ALONG IT. This panel is rotated, so x IS the clash count.
# Spreading along x would move a pose to a count it does not have -- the figure would be
# lying. y carries nothing but "which method", so displacing a point there costs no meaning.
#
# 0.30 IS BOUNDED BY THE SLOT, NOT CHOSEN BY EYE. The boxes are 0.66 wide on a 1.0 pitch, so
# +-0.33 is the whole slot; at 0.30 the clouds keep 0.40 of clear band between neighbours.
PCSZ_CBOX_JITTER = 0.30
# FIXED, so the figure is reproducible. This family is checked by re-rendering and diffing,
# and an unseeded jitter would make every rebuild differ for no reason.
PCSZ_CBOX_JITTER_SEED = 20260916
# ONE SCALE OVER BOTH OF THE ABOVE (2026-09-14), so "tighten the whitespace" stays a single
# number instead of two that drift apart. The box width and the 1.0 pitch are untouched: only
# air moves, so the boxes keep their size and the panel just stops carrying as much empty
# space.
#
# 0.88, UP 1.1x FROM 0.8 ON REQUEST (2026-09-15): the rule's air goes 0.40 -> 0.44 and the
# end padding 0.56 -> 0.62. It moves in the opposite direction to the 0.9x type above, which
# is the point -- smaller glyphs with more room around them.
PCSZ_CBOX_MARGIN = 0.88


def _pcsz_tint(hexv, f=PCSZ_CBOX_TINT):
    """`hexv` mixed f of the way to white, as an rgb triple."""
    r, g, b = (int(hexv[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return tuple(c + (1.0 - c) * f for c in (r, g, b))
# WHERE THE CRYSTAL LIGANDS' CLASHES COME FROM, and why not from REF_ROOT like their strain.
# REF_ROOT's run pose-scored against the 10 A crop, which deletes protein a pose could clash
# with and so under-counts; every method on this axis is scored against the whole receptor.
# This root pose-scored with scope="full", so its reference is on the same footing. Nothing
# was recomputed for it: three independent full-scope runs (this one, reproduction/
# res_test_100 and 260827 base) carry the reference clash for all 79 pockets and agree on
# every one of them, as they must -- counting clashes is deterministic, unlike strain.
PCSZ_REF_CLASH_ROOT = (f"{E}/260908_fusion_default_cv2_scratch_8gpu/samples/"
                       "samples_ep350_test79_n100")


@figure("fig-posecheck-clash-box-by-method", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_box_by_method(out, layout=None):
    """Steric clashes per pose; portrait rows or landscape columns of methods."""
    layout = draw.PCSZ_CBOX_LAYOUT if layout is None else layout
    if layout not in ("horizontal", "vertical"):
        raise ValueError("layout must be horizontal or vertical")
    landscape = layout == "horizontal"
    axis_lw = AXIS_LW if landscape else PCSZ_CBOX_AXIS_LW
    data_by_label, series, _ref, methods = _pcsz_data()
    labels = [label for label, *_ in series
              if len(data_by_label[label][PCSZ_POOLED]["clash"]) >= 20]
    if landscape:
        labels = _pcsz_ordered(labels)
    vals = [np.asarray(data_by_label[l][PCSZ_POOLED]["clash"], dtype=float) for l in labels]
    # THE CRYSTAL LIGANDS LEAD (2026-09-14), from the whole-receptor run named above. They are
    # the benchmark every method is read against, so they are the leftmost box, and n=79 here
    # against ~7,500 a method: one crystal pose per pocket is all there is.
    keep = set(json.load(open(legacy("fig-posecheck", "posecheck_AR.json"),
                              encoding="utf-8"))["density79_pockets"])
    for _, root, *_ in PCSZ_LOCAL:
        keep &= _pcsz_pockets_of(root)
    ref_full, _ = _pcsz_load_local(PCSZ_REF_CLASH_ROOT, keep, reference=True)
    labels = [PCSZ_REF_NAME] + labels
    vals = [np.asarray(ref_full[PCSZ_POOLED]["clash"], dtype=float)] + vals
    meds = [float(np.median(v)) for v in vals]
    # PROPER log10, not log10(x+1) (2026-09-14). The +1 existed so a zero-clash pose had
    # somewhere to sit on the axis -- but the floor is 10^0 now and those poses fall below it
    # either way, so all the +1 still bought was a compressed bottom decade: it doubles 1
    # while leaving 100 untouched. Measured on the drawn panel, 10^0->10^1 spanned 130 px
    # against 169 px for 10^1->10^2, a ratio of 1.30 under tick labels that promise 1.00.
    # Zeros clamp to 0.5, below the floor, and are clipped exactly as before -- so nothing
    # visible changes except that the decades are finally even.
    # THE VIOLIN KEEPS ITS +1 and must: its axis carries a real 0 tick, so a zero-clash pose
    # has a place to land there and the transform is still doing work.
    tf = lambda v: np.log10(np.maximum(np.asarray(v, dtype=float), 0.5))
    rc = dict(PCSZ_RC)
    if landscape:
        rc["savefig.bbox"] = None  # Preserve the exact shared canvas dimensions.
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(PCSZ_STRAIN_SIZE if landscape else
                                        (PCSZ_ECDF_SIZE[0] * PCSZ_CBOX_WIDE_V,
                                         PCSZ_ECDF_SIZE[1] * PCSZ_CBOX_TALL_V)),
                               layout="none" if landscape else "constrained")
        if landscape:
            fig.subplots_adjust(**PCSZ_COMPARISON_MARGINS)
        fig.patch.set_facecolor(PCSZ_BG)
        # The grid follows the VALUE axis, which rotating moved to x. A rule through every
        # category centre is a picket fence rather than a reading aid -- which is the whole
        # reason _pcsz_style takes the axis as an argument.
        _pcsz_style(ax, grid_axis="y" if landscape else "x")
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(axis_lw)
        ax.tick_params(which="major", width=axis_lw)
        if landscape:
            for side in ("left", "bottom"):
                ax.spines[side].set_zorder(6)
            ax.tick_params(labelsize=14)  # Same tick typography as redocking median.
        # THE GRID'S WHISKERS (2026-09-14): 1.5 x IQR on the RAW counts, the way the
        # rotatable-bond grid gets them by handing raw values to a log axes. Handing this one
        # tf(v) instead put the fences 1.5 IQR out in LOG space, which ran AR's whisker to 33
        # where the raw fence is 14 -- the two "same format" figures were quoting different
        # statistics. Quartiles are order statistics and do not care which space they are
        # found in; the fences do. So the box is computed on the counts and only its
        # POSITIONS are transformed for drawing.
        stats = [{k: tf(s[k]) for k in ("med", "q1", "q3", "whislo", "whishi", "fliers")}
                 for s in (matplotlib.cbook.boxplot_stats(v, whis=1.5)[0] for v in vals)]
        # The reference holds position 0; the methods start one full pitch plus the gap out.
        gap = PCSZ_CBOX_GAP * PCSZ_CBOX_MARGIN
        pos = [0.0] + [i + gap for i in range(1, len(stats))]
        bp = ax.bxp(stats, positions=pos, widths=0.66, orientation="vertical" if landscape else "horizontal",
                    showfliers=True, patch_artist=True, manage_ticks=False, zorder=5,
                    # Square diamonds (45-degree rotation), sized for legibility at export resolution.
                    flierprops=dict(marker="D", markersize=1.2, markerfacecolor=INK,
                                    markeredgecolor="none", alpha=DATA_ALPHA))
        for box, label in zip(bp["boxes"], labels):
            # The method's own tint, the one it wears in the violin and the ECDF beside it.
            # pale() is keyed on the palette's "Reference ligand", not on the drawn name.
            box.set(facecolor=_pcsz_tint(REF_COLOR if label == PCSZ_REF_NAME
                                         else pale(label)),
                    edgecolor="none", linewidth=0)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                # THE GRID'S WHISKER INK (2026-09-14), not this family's. The rotatable-bond
                # grid draws whiskers and caps in the house INK -- #514F52, a warm grey --
                # while this family overrides its furniture to pure black; at the same 0.9
                # weight that black read as a harder, heavier line than the grid's. Only the
                # box's own furniture moves: the spines, ticks and names stay black.
                art.set(color=INK, linewidth=0.9)
        for m in bp["medians"]:
            m.set(color=RB_GRID_MED_COLOR, linewidth=PCSZ_CBOX_MED_LW, solid_capstyle="butt")
        # Tens of thousands of flier markers as vector paths make the PDF unplaceable.
        # Scattered across the band before rasterising -- see PCSZ_CBOX_JITTER for why the
        # displacement goes on y and never on x. One generator for the whole panel, seeded,
        # so the figure is identical on every rebuild.
        jrng = np.random.default_rng(PCSZ_CBOX_JITTER_SEED)
        for fl in bp["fliers"]:
            # Jitter only the categorical axis; never change a clash count.
            slots = np.asarray(fl.get_xdata() if landscape else fl.get_ydata(), dtype=float)
            if slots.size:
                shifted = slots + jrng.uniform(-PCSZ_CBOX_JITTER,
                                               PCSZ_CBOX_JITTER, slots.size)
                (fl.set_xdata if landscape else fl.set_ydata)(shifted)
            fl.set_rasterized(True)
        # Capped: fliers past PCSZ_CBOX_HI fall OFF the panel rather than being clipped onto
        # its edge, because piling them on the edge would draw a spike at 200 that no method
        # actually has. The run log below still prints each method's true maximum.
        hi = min(max(float(v.max()) for v in vals), PCSZ_CBOX_HI)
        # THE VALUE AXIS IS X NOW (2026-09-14, rotated): decade names as powers of ten with
        # the log minor ticks between them, as the per-atom panel has them, so the two still
        # read as a pair of log axes -- evenly, now that tf is a true log10. The lowest
        # LABELLED decade is 10^0 (the axis itself stops PCSZ_CBOX_FLOOR_PAD below it), so
        # there is no 0 tick and the poses AT zero (1.6% of TargetDiff's, 10.9% of
        # AR's) sit below the frame, their whisker running off its edge. That those poses are
        # invisible is worth saying in the caption: a tenth of AR's poses clash with nothing
        # at all, which is the best thing about it and the panel cannot show it.
        decades = [t for t in PCSZ_CLASH_TICKS if t and t <= hi * 1.6]
        if landscape:
            # Reference median in the same log10 coordinates as the boxes.
            # 1.6x heavier and darker than the original 1.2 / REF_COLOR (2026-09-24, on request):
            # the reference grey pulled toward black, so it stays the reference's hue.
            ref_line = matplotlib.colors.to_rgb(REF_COLOR)
            ref_line = tuple(c * (1 - PCSZ_CBOX_REF_LINE_DARKEN) for c in ref_line)
            ax.axhline(float(tf(meds[0])), color=ref_line, ls=DOT,
                       lw=PCSZ_CBOX_REF_LINE_LW, zorder=6)
            ax.set_yticks(tf(decades))
            ax.set_yticklabels([f"$10^{{{round(math.log10(t))}}}$" for t in decades])
            ax.set_yticks(tf([k * t for t in decades for k in range(2, 10)
                             if k * t <= hi]), minor=True)
            ax.tick_params(axis="y", which="minor", direction="out", length=2.0,
                           width=axis_lw * PCSZ_CBOX_MINOR_LW, colors=PCSZ_AXIS)
            ax.set_ylim(tf(1) - PCSZ_CBOX_FLOOR_PAD, tf(hi) + 0.06)
            ax.set_xlim(pos[0] - PCSZ_CBOX_PAD * PCSZ_CBOX_MARGIN,
                        pos[-1] + PCSZ_CBOX_PAD * PCSZ_CBOX_MARGIN)
            ax.axvline((pos[0] + pos[1]) / 2.0, color=PCSZ_AXIS,
                       lw=axis_lw, zorder=4)
            ax.set_xticks([])  # Method names are carried by the standalone legend.
            ax.set_ylabel("Steric\nclashes", fontsize=AXIS_LABEL_FS, labelpad=PCSZ_YPAD)
        else:
            ax.set_xticks(tf(decades))
            ax.set_xticklabels([f"$10^{{{round(math.log10(t))}}}$" for t in decades])
            ax.set_xticks(tf([k * t for t in decades for k in range(2, 10) if k * t <= hi]),
                          minor=True)
            ax.tick_params(axis="x", which="minor", direction="out", length=2.0,
                           width=axis_lw * PCSZ_CBOX_MINOR_LW, colors=PCSZ_AXIS)
            ax.set_xlim(tf(1), tf(hi) + 0.06)
            # INVERTED, so position 0 -- the crystal ligands -- is the TOP row. A stack of rows is
            # read downward and matplotlib puts 0 at the bottom, which would have buried the
            # benchmark under the eight methods it exists to be compared against.
            ax.set_ylim(pos[-1] + PCSZ_CBOX_PAD * PCSZ_CBOX_MARGIN,
                        pos[0] - PCSZ_CBOX_PAD * PCSZ_CBOX_MARGIN)
            # The crystal ligands are the benchmark the methods are read against, so a rule
            # separates them rather than leaving them to look like a ninth method. It takes the
            # frame's colour AND weight exactly, with no alpha: at 0.45 the same black rendered as
            # a mid grey that read as a third kind of line -- neither the black spines it belongs
            # with nor the grey whiskers inside the panel.
            ax.axhline((pos[0] + pos[1]) / 2.0, color=PCSZ_AXIS, lw=axis_lw, zorder=4)
            ax.set_yticks(pos)
            # TILTED, ANCHORED WHERE THE NAME ENDS (2026-09-14). rotation_mode="anchor" pins the
            # label's right end to its tick and rotates about that point, so the tick sits
            # exactly where the name stops and each name points at its own row.
            # Measured, this puts a label's bounding-box CENTRE a median 40 px below its tick
            # (54 px for the longest) -- but that is the geometry of an anchored rotation, not a
            # misalignment, because the eye follows a tilted name to its END. Centring the
            # rotated box on the tick instead scores 0 px by that measure and reads WORSE: no
            # part of the name then touches its tick and it floats between two rows.
            ax.set_yticklabels([l if l == PCSZ_REF_NAME else display(l) for l in labels],
                               fontsize=PCSZ_CBOX_NAME_FS, rotation=PCSZ_CBOX_NAME_ROT,
                               ha="right", va="center", rotation_mode="anchor")
            ax.set_xlabel("Steric clashes", fontsize=PCSZ_CBOX_TITLE_FS, labelpad=PCSZ_YPAD)
        if landscape:
            _pcsz_comparison_layout(fig, ax)
        save(fig, out, "clash_box_by_method_horizontal" if landscape else "clash_box_by_method")
    write_csv(out, "clash_box_by_method_horizontal" if landscape else "clash_box_by_method",
              ["method", "n_clash", "clash_mean", "clash_median", "q25", "q75", "zero_frac"],
              [[l, len(v), round(float(v.mean()), 2), round(float(np.median(v)), 1),
                round(float(np.percentile(v, 25)), 1), round(float(np.percentile(v, 75)), 1),
                round(float((v == 0).mean()), 4)]
               for l, v in zip(labels, vals)])
    for l, v, m in zip(labels, vals, meds):
        print(f"    {l:16s} n={len(v):5d}  median {m:4.1f}  mean {v.mean():5.2f}  "
              f"zero-clash {100 * (v == 0).mean():4.1f}%  max {v.max():.0f}")
    print(f"  wrote {out}")


# ── fig-posecheck-clash-per-atom-all-methods ───────────────────────────────────────────
# The strain per-atom all-methods panel's twin, for clashes (2026-09-14).
#
# WHY THE EXISTING clash_per_atom_* FIGURES COULD NOT SIMPLY GAIN FIVE MORE LINES. They draw
# out of the ARMS run trees, and those trees cannot answer this question for eight methods:
#   1. THE FIVE BASELINES HAVE NO PER-MOLECULE POSECHECK THERE AT ALL -- 0 of ~37,000 samples
#      under baselines_pose/* carry posecheck.clashes. Their per-molecule counts exist only in
#      the evaluation server's exports, which is where the violin, the ECDF and the box already read them.
#   2. THE RECEPTOR SCOPE DIFFERS. The ARMS roots pose-scored against the 10 A crop
#      (pose_receptor_scope "crop", or absent, which metrics.py reads as crop); every other
#      clash figure in this family scores against the whole receptor. On the SAME arm and the
#      same 79 pockets the crop reads CoDE at median 5.0 / mean 6.44 / max 39 where the whole
#      receptor reads 6.0 / 7.48 / 97 -- the crop deletes protein the pose could clash with,
#      so it under-counts, and it truncates the tail hardest.
# So this figure takes its five baselines from the exports, its three local arms from
# PCSZ_LOCAL (scope "full") and its reference from PCSZ_REF_CLASH_ROOT, and never from ARMS.
# It keeps the strain twin's windowing (+-RB_WIN atoms, drawn where >= MIN_N molecules pool)
# so the two panels are read the same way, and its house is the ECDF's, like every panel here.
PCSZ_CATOM_STATS = {"median": lambda v: float(np.median(v)),
                    "mean": lambda v: float(np.mean(v))}
# Dash per method, keyed as PCSZ_LINE_LW is: both tables are (.., lw, ls) last-two.
PCSZ_LINE_LS = {**{ALIASES.get(lab, lab): ls for _, lab, _, ls in PCSZ_BASELINES},
                **{ALIASES.get(lab, lab): ls for lab, _, _, ls in PCSZ_LOCAL}}


def _pcsz_atom_rows(root, keep, reference=False):
    """[{n, c}] per molecule for one whole-receptor run tree, over the pockets in `keep`."""
    rows = []
    for path in _pcsz_metrics(root):
        idx = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if idx not in keep:
            continue
        j = json.load(open(path, encoding="utf-8"))
        items = [j.get("reference")] if reference else (j.get("samples") or [])
        for m in items:
            if not m or not m.get("n_atoms"):
                continue
            c = (m.get("posecheck") or {}).get("clashes")
            if c is not None:
                rows.append({"n": m["n_atoms"], "c": float(c)})
    return rows


def _pcsz_atom_keep():
    """The pockets every arm on this axis has: the baselines' 79, narrowed by ours."""
    keep = set(json.load(open(legacy("fig-posecheck", "posecheck_AR.json"),
                              encoding="utf-8"))["density79_pockets"])
    for _, root, *_ in PCSZ_LOCAL:
        keep &= _pcsz_pockets_of(root)
    return keep


@figure("fig-posecheck-clash-per-atom-all-methods", folder="fig-posecheck/clash",
        needs=("posecheck_<Method>.json", "frozenenc_probes/posecheck_full/"))
def draw_posecheck_clash_per_atom_all_methods(out):
    """Steric clashes against ligand size, all eight methods and the crystal ligands."""
    keep = _pcsz_atom_keep()
    series = {}
    for key, label, *_ in PCSZ_BASELINES:
        # The exports' rows are already {n, s, c}; this reads the same file the violin does.
        series[label], _ = _rb_atom_baseline_rows(key, keep)
    for label, root, *_ in PCSZ_LOCAL:
        series[label] = _pcsz_atom_rows(root, keep)
    refrows = _pcsz_atom_rows(PCSZ_REF_CLASH_ROOT, keep, reference=True)

    labels = _pcsz_ordered([label for _, label, *_ in PCSZ_BASELINES]
                           + [lab for lab, *_ in PCSZ_LOCAL])
    per = {lab: by_size(series[lab], "c") for lab in labels}
    ref_per = by_size(refrows, "c")
    xs = list(range(RB_X_LO, RB_X_HI + 1))

    for stat, f in PCSZ_CATOM_STATS.items():
        with plt.rc_context(PCSZ_RC):
            fig, ax = plt.subplots(figsize=PCSZ_ECDF_SIZE)
            fig.patch.set_facecolor(PCSZ_BG)
            _pcsz_style(ax)
            ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR,
                    lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls=SOLID, zorder=4,
                    solid_capstyle="round")
            for lab in labels:
                key = ALIASES.get(lab, lab)
                ax.plot(xs, _rb_curve(per[lab], xs, f), color=pale(lab),
                        lw=PCSZ_LINE_LW[key] * PCSZ_LW_SCALE, ls=PCSZ_LINE_LS[key],
                        zorder=5, solid_capstyle="round")
            ax.set_ylim(bottom=0)
            ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
            ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
            ax.set_xlabel(X_LABEL, fontsize=PCSZ_LABEL_FS)
            ax.set_ylabel(f"Clashes {stat}\nper pose", fontsize=PCSZ_LABEL_FS,
                          labelpad=PCSZ_YPAD)
            fig.tight_layout(pad=0.5)
            save(fig, out, f"clash_per_atom_all_methods_{stat}")

    rows = []
    for lab in labels + [PCSZ_REF_NAME]:
        p = ref_per if lab == PCSZ_REF_NAME else per[lab]
        cur = {s: (reference_curve(p, xs, f) if lab == PCSZ_REF_NAME
                   else _rb_curve(p, xs, f)) for s, f in PCSZ_CATOM_STATS.items()}
        for i, a in enumerate(xs):
            rows.append([lab, a, len(p.get(a, ()))]
                        + ["" if cur[s][i] is None else round(cur[s][i], 2)
                           for s in ("median", "mean")])
    write_csv(out, "clash_per_atom_all_methods",
              ["arm", "heavy_atoms", "n", "clash_median", "clash_mean"], rows)

    print(f"  {len(keep)} pockets · {len(labels)} methods + reference · whole-receptor "
          f"scope · x = {RB_X_LO}-{RB_X_HI} heavy atoms · ±{RB_WIN}-atom window, drawn "
          f"where it pools ≥{MIN_N} molecules")
    heads = (10, 15, 20, 25, 30, 35, 40)
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{a:>6d}" for a in heads) + f" {'drawn':>9s}   (clash median at n)")
    for lab in labels + [PCSZ_REF_NAME]:
        p = ref_per if lab == PCSZ_REF_NAME else per[lab]
        med = (reference_curve(p, xs, PCSZ_CATOM_STATS["median"]) if lab == PCSZ_REF_NAME
               else _rb_curve(p, xs, PCSZ_CATOM_STATS["median"]))
        cells = [(f"{med[xs.index(a)]:6.1f}" if med[xs.index(a)] is not None
                  else f"{'—':>6s}") for a in heads]
        drawn = [a for a, v in zip(xs, med) if v is not None]
        span = f"{drawn[0]}-{drawn[-1]}" if drawn else "—"
        n_mol = sum(len(v) for v in p.values())
        mean_at = np.mean([n for n, v in p.items() for _ in v]) if n_mol else float("nan")
        print(f"  {lab:16s} {n_mol:6d} {mean_at:6.1f} " + " ".join(cells) + f" {span:>9s}")
    print(f"  wrote {out}")


# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-per-atom, fig-posecheck-clash-per-atom, fig-posecheck-ecdf-pair
# ════════════════════════════════════════════════════════════════════════════════
# The size-resolved view of PoseCheck strain and steric clashes: what each arm's poses cost
# at a given ligand size, against the crystal ligand at that size.
#
# MEAN AND MEDIAN ARE SEPARATE FIGURES, which is what the 3-line figures do
# (vina_dock_3line_mean / _median) and here it is not optional. Strain's mean is not a
# location statistic: 6.6% of molecules fail UFF relaxation and land between 1e4 and 1e13,
# and one of those at a thin heavy-atom count moves that count's mean by four decades. The
# two sit three to four decades apart, so overlaid, the mean's spikes cross the whole panel
# and bury the medians -- which are tight, ordered, and the thing worth reading. Drawn
# apart, the contrast is itself the argument for reporting the median: in the mean figure
# the arms are tangled with no order at all, in the median figure they separate cleanly.
#
# The mean figure is still clipped to the bulk -- the 3-line figure's own answer (see its
# `_v3_limits`) -- and the excluded points are NAMED in the run log rather than squashing
# everything else into two decades. Clashes have no such problem and are drawn the same way
# for symmetry.
PC_STATS = (("mean", lambda v: float(np.mean(v))),
            ("median", lambda v: float(np.median(v))))
PC_STRAIN_CLIP = 1e6
PC_STRAIN_UNIT = "\n(kcal mol⁻¹)"
# The ECDF's x floors: strain is drawn on a log axis, so a value at or below zero has no
# place on it and is clipped onto the first decade rather than dropped; clashes are counts
# and floor at 0.
PC_ECDF_FLOOR = {"s": 1e-2, "c": 0}
PC_ECDF_W = 1.52                       # two panels side by side, off the one-panel width


def _pc_ranges():
    """(variant, arms, xs) per variant, plus the line saying what was drawn.

    THE X RANGE IS THE STRAIN RANGE, for the clash figure too. The builder computed it once
    from field "s" and handed the same xs to both, so the two panels register against each
    other count for count. Resolving it against "c" instead would silently shift the clash
    figure: a UFF relaxation that did not converge drops a molecule from the strain counts
    and not from the clash ones, so the two fields do not reach MIN_N at the same sizes."""
    data, p79_rows, _ = pose_data()
    out = []
    for variant, arms in variants("s", data):
        per = {key: by_size(p79_rows[key], "s") for _, key, _ in arms}
        out.append((variant, arms, x_range(per, arms)))
    print(f"  79-pocket set · {len(P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, _, xs in out)
          + f" (counts where every drawn arm has >={MIN_N} molecules)\n")
    return out


def _pc_per_atom_stat(out, field, xs, arms, variant, name, unit, stat, f, *, log,
                      clip=None, stem, legend_loc="upper left", rows=None, ref=None):
    """One statistic, one panel, one file. Which statistic you are looking at is carried by
    the filename and by the y-axis name, exactly as the 3-line figures carry it.

    `rows`/`ref` override where the molecules come from. Strain passes neither and reads
    pose_data() as it always has; CLASHES pass the whole-receptor trees, because the
    pose_data() rows are crop-scored and a crop under-counts clashes."""
    _, p79_rows, refrows = pose_data()
    p79_rows = p79_rows if rows is None else rows
    refrows = refrows if ref is None else ref
    per = {key: by_size(p79_rows[key], field) for _, key, _ in arms}
    ref_per = by_size(refrows, field)
    fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    dropped = []
    ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        y = model_curve(per[key], xs, f)
        if clip:
            over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
            if over:
                dropped.append((lab, over))
        ax.plot(xs, y, color=pale(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if clip:
        ax.set_ylim(top=clip)
    furniture(ax, ylabel=f"{name} {stat}{unit}", xlabel=X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    legend(ax, arm_handles(arms), loc=legend_loc, fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"{stem}_{variant}")
    return dropped


def _pc_block(rows):
    """The pooled numbers the run log reports for one set of molecules. These used to fill
    posecheck_summary.json too; that export belongs to the recompute script, so only what
    the table prints survives here."""
    s = [r["s"] for r in rows if r["s"] is not None]
    c = [r["c"] for r in rows if r["c"] is not None]
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "strain_mean": round(float(np.mean(s)), 1) if s else None,
        "strain_median": round(st.median(s), 1) if s else None,
        "clash_median": round(float(np.median(c)), 1) if c else None,
    }


def _pc_table():
    """Pooled strain and clashes per arm over the 79-pocket set, and the crystal ligand."""
    data, p79_rows, refrows = pose_data()
    print(f"  {'arm':16s} {'atoms':>6s} {'strain med':>11s} {'strain mean':>12s} "
          f"{'clash med':>10s} {'mols':>7s}")
    for lab, key, _ in arms_for("s", data):
        r = _pc_block(p79_rows[key])
        print(f"  {lab:16s} {r['atoms_mean']:6.1f} {r['strain_median']:11.1f} "
              f"{r['strain_mean']:12.3g} {r['clash_median']:10.1f} {r['n_molecules']:7d}")
    r = _pc_block(refrows)
    print(f"  {REF_LABEL:16s} {r['atoms_mean']:6.1f} {r['strain_median']:11.1f} "
          f"{r['strain_mean']:12.3g} {r['clash_median']:10.1f} {r['n_molecules']:7d}")


def _pc_curve_csv(out, xs):
    """The curves themselves, beside the figure, so a reader can check one without
    re-running it. Strain and clashes go in ONE table at ONE xs -- four curves over the
    same molecules at the same heavy-atom counts -- so the strain folder and the clash
    folder each get the whole thing rather than half of it. xs is the `all` variant's
    range, the wider of the two, and `core` is a subset of these rows."""
    data, p79_rows, refrows = pose_data()
    fields = ["strain_mean", "strain_median", "clash_mean", "clash_median"]
    cells = lambda curves, i: [("" if curves[f][i] is None else round(curves[f][i], 3))
                               for f in fields]
    rows = []
    for lab, key, _ in arms_for("s", data):
        s_per, c_per = by_size(p79_rows[key], "s"), by_size(p79_rows[key], "c")
        curves = {"strain_mean": model_curve(s_per, xs, PC_STATS[0][1]),
                  "strain_median": model_curve(s_per, xs, PC_STATS[1][1]),
                  "clash_mean": model_curve(c_per, xs, PC_STATS[0][1]),
                  "clash_median": model_curve(c_per, xs, PC_STATS[1][1])}
        rows += [[lab, a, len(s_per.get(a, ()))] + cells(curves, i)
                 for i, a in enumerate(xs)]
    # The reference's own curves come from the centred +-REF_WIN window, not from a count,
    # so it has no per-count n to report.
    s_ref, c_ref = by_size(refrows, "s"), by_size(refrows, "c")
    curves = {"strain_mean": reference_curve(s_ref, xs, PC_STATS[0][1]),
              "strain_median": reference_curve(s_ref, xs, PC_STATS[1][1]),
              "clash_mean": reference_curve(c_ref, xs, PC_STATS[0][1]),
              "clash_median": reference_curve(c_ref, xs, PC_STATS[1][1])}
    rows += [[REF_LABEL, a, ""] + cells(curves, i) for i, a in enumerate(xs)]
    write_csv(out, "posecheck_per_atom", ["arm", "heavy_atoms", "n"] + fields, rows)


@figure("fig-posecheck-strain-per-atom", folder="fig-posecheck/strain-energy", needs=("metrics.json (posecheck.strain)",))
def draw_posecheck_strain_per_atom(out):
    """PoseCheck strain against heavy-atom count — mean and median, core only.

    No `all`: see core_only(). The eight-method view is strain_per_atom_all_methods."""
    use_style()
    drops, all_xs = {}, None
    for variant, arms, xs in core_only(_pc_ranges()):
        drops[variant] = _pc_per_atom_stat(
            out, "s", xs, arms, variant, "Strain", PC_STRAIN_UNIT, *PC_STATS[0],
            log=True, clip=PC_STRAIN_CLIP, stem="strain_per_atom_mean",
            legend_loc="lower right")
        _pc_per_atom_stat(out, "s", xs, arms, variant, "Strain", PC_STRAIN_UNIT,
                          *PC_STATS[1], log=True, stem="strain_per_atom_median",
                          legend_loc="upper left")
        all_xs = xs
    _pc_table()
    # Named, not hidden: the mean figure clips to the bulk, so say which points that leaves
    # off the panel and how far above they went.
    if drops["core"]:
        print(f"\n  strain_per_atom_mean_core: above the {PC_STRAIN_CLIP:.0e} clip, "
              f"off-panel (a failed UFF relaxation at a thin count moves that count's "
              f"mean):")
        for lab, over in drops["core"]:
            pts = ", ".join(f"{a} atoms {v:.2g}" for a, v in over)
            print(f"    {lab:16s} {len(over):2d} of {len(all_xs):2d}: {pts}")
    _pc_curve_csv(out, all_xs)


@figure("fig-posecheck-clash-per-atom", folder="fig-posecheck/clash",
        needs=("frozenenc_probes/posecheck_full/",))
def draw_posecheck_clash_per_atom(out):
    """PoseCheck steric clashes against heavy-atom count — mean and median, core only.

    WHOLE-RECEPTOR SCOPE (2026-09-14). This used to read pose_data(), i.e. the ARMS run
    trees, and those pose-scored against the 10 A crop: on the same arm over the same 79
    pockets the crop reads CoDE at median 5.0 / mean 6.44 / max 39 where the whole receptor
    reads 6.0 / 7.48 / 97, and VoxBind 4.0 / 5.23 / 42 against 5.0 / 6.23 / 78. A crop
    deletes protein the pose could clash with, so it under-counts and truncates the tail
    hardest. Every other clash figure in this family scores against the whole receptor, so
    this one reads the same posecheck_full trees the violin and the box do.

    No `all`: see core_only(). The five published baselines carry NO per-molecule PoseCheck
    in the ARMS trees at all -- 0 of ~37,000 samples -- so `all` here was only ever core plus
    TargetDiff, which is neither the two-arm comparison nor the whole field. The whole field
    (seven published baselines, ours, and the crystal ligands) is
    clash_per_atom_all_methods, which reads the evaluation server's exports for the five."""
    use_style()
    keep = _pcsz_atom_keep()
    # The run roots are keyed by the ARMS key itself: posecheck_full/<key>.
    rows = {os.path.basename(root): _pcsz_atom_rows(root, keep)
            for _, root, *_ in PCSZ_LOCAL}
    refrows = _pcsz_atom_rows(PCSZ_REF_CLASH_ROOT, keep, reference=True)
    arms = [a for a in ARMS if a[1] in CORE and a[1] in rows]
    per = {key: by_size(rows[key], "c") for _, key, _ in arms}
    # The x range is the CLASH range now, not the strain range _pc_ranges() derives. That
    # sharing existed because both panels read one molecule set; this figure no longer does,
    # so registering the two count for count would be a coincidence, not a property.
    xs = x_range(per, arms)
    for stat, f in PC_STATS:
        _pc_per_atom_stat(out, "c", xs, arms, "core", "Clashes", "", stat, f,
                          log=False, stem=f"clash_per_atom_{stat}",
                          legend_loc="upper left", rows=rows, ref=refrows)
    # Its own CSV: _pc_curve_csv writes strain AND clashes out of pose_data(), whose clash
    # columns are the crop-scored ones this figure just stopped drawing.
    ref_per = by_size(refrows, "c")
    csv_rows = []
    for lab, key, _ in arms:
        cur = {s: model_curve(per[key], xs, fn) for s, fn in PC_STATS}
        csv_rows += [[lab, a, len(per[key].get(a, ()))]
                     + ["" if cur[s][i] is None else round(cur[s][i], 3)
                        for s, _ in PC_STATS] for i, a in enumerate(xs)]
    cur = {s: reference_curve(ref_per, xs, fn) for s, fn in PC_STATS}
    csv_rows += [[REF_LABEL, a, ""] + ["" if cur[s][i] is None else round(cur[s][i], 3)
                                       for s, _ in PC_STATS] for i, a in enumerate(xs)]
    write_csv(out, "clash_per_atom",
              ["arm", "heavy_atoms", "n", "clash_mean", "clash_median"], csv_rows)
    print(f"  {len(keep)} pockets · whole-receptor scope · "
          f"{', '.join(l for l, *_ in arms)} + reference · x = {xs[0]}-{xs[-1]}")




# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-per-rotbond, fig-posecheck-strain-box-per-rotbond,
# fig-posecheck-strain-rotbond-all-methods, fig-posecheck-strain-per-atom-all-methods
# ════════════════════════════════════════════════════════════════════════════════
# STRAIN AGAINST THE NUMBER OF ROTATABLE BONDS, and the eight-method version of the same
# metric on both axes. The VoxBind paper's Fig. 12/13 form.
#
# WHY THE AXIS IS WORTH SWAPPING. Strain is conformational: it is the energy a pose carries
# because its torsions are not where the force field would put them, so the number of
# torsions a molecule HAS is the more direct explanatory variable, and heavy-atom count is
# its proxy. The two are not interchangeable -- a fused polycyclic and a long-chain ligand
# of equal size have very different torsional freedom -- and the ranking of the arms is
# allowed to differ between the two views.
#
# ROTATABLE BONDS ARE COUNTED FROM THE SMILES that `metrics.json` already records, with
# RDKit's default (strict) `CalcNumRotatableBonds`, which excludes amides, terminal bonds
# and ring bonds. It is a topological descriptor, so reading it off the recorded SMILES
# rather than the pose gives the same integer with no risk of a sample-to-SDF index slip.
# Molecules whose SMILES will not parse are dropped and counted in the run log.
#
# THE CRYSTAL REFERENCE WINDOW IS +-1 HERE, NOT the header's +-4. There is one crystal
# ligand per pocket, so a per-count reference curve over 79 ligands would be noise -- that
# is why the window exists at all. But rotatable-bond counts run 0-14 where ligand sizes run
# 5-45, and a +-4 window there spans two thirds of the axis and would flatten the reference
# into a near-constant line. +-1 pools 12-29 ligands per point and the line stops at 9,
# where the window falls under the minimum, rather than being extended into an invented
# value. The originals got this by MUTATING pose_common.REF_WIN at import; in one process
# that draws every figure a module-level override would silently re-window the per-atom
# figures next door, so the narrow window is a local constant and a local curve function.
#
# THE BOXES ARE THE PRIMARY FIGURE AND THE LINES ARE KEPT BESIDE THEM. A median line says
# where an arm sits; it cannot say whether two arms a factor of 1.5 apart are actually
# separated, and on a metric whose distribution spans four decades inside a single bond
# count that is the question. The boxes answer it -- and they show something the lines
# cannot: the arms' inter-quartile ranges overlap almost completely at every count, so the
# median ordering is a shift of a wide distribution, not a separation of two narrow ones.
#
# WHISKERS ARE THE 5TH AND 95TH PERCENTILES AND FLIERS ARE NOT DRAWN. Not a cosmetic
# choice: 4-7% of molecules relax to 1e4-1e13, so Tukey whiskers with fliers would put
# single points nine decades above the boxes and squash every box in the figure into a
# line. The percentile whisker is stated on the y axis, and the tail it leaves out is
# reported as a number rather than drawn. NO MEAN MARKER ON THE BOXES either: within one arm
# and one bond count the mean sits at 1e4-1e11 while the box sits near 1e2, so every marker
# would land far above its own box and drag the axis with it. The mean has its own line
# figure, where being unreadable at least reads as the finding it is.
#
# MEAN AND MEDIAN ARE SEPARATE LINE FIGURES, as everywhere in 260910. THE MEDIAN IS THE
# FIGURE TO READ; the mean is here so that claim can be checked, not as an alternative. AND
# THE PER-ROTBOND MEAN PANEL IS NOT CLIPPED, where the per-atom one is: 15 rotatable-bond
# counts pool 500-1,100 molecules each, so nearly every point catches one of the conformers
# that relax to 1e8-1e13, and a 1e6 clip left the curve as disconnected fragments with most
# of it above the panel. Drawn whole it spans nine decades, sits 6-9 decades above the
# crystal ligands and has no ordering at all -- which is the honest picture of what a mean
# does to this metric, and it is the argument for the median rather than something to hide
# behind an axis limit.
RB_REF_WIN, RB_MIN_REF = 1, 12
RB_X_LABEL = "Number of rotatable bonds in ligand"
RB_XTICK = 1
# Above this a UFF relaxation has effectively failed rather than found a lower conformer
# (the converged bulk sits under ~1e3). Nothing is filtered on it; it only names the tail
# that makes the mean unreadable, so the mean figure can be read for what it is.
RB_TAIL = 1e4
RB_WHIS = (5, 95)
# Strain is floored before boxing, at the value strain_clash_ecdf_pair already floors it to.
# A rigid ligand can relax to ~0, and on a log axis a single 1e-11 at 0 rotatable bonds
# pulled the panel down through fifteen decades and flattened every box in it. Values are
# CLIPPED, not dropped, so the whisker rests on the floor and the count is unchanged.
RB_STRAIN_FLOOR = 1e-2
RB_STATS = {"median": lambda v: float(np.median(v)),
            "mean": lambda v: float(np.mean(v))}
RB_Y_LABEL = "Strain {stat}\n(kcal mol⁻¹)"
# The mean curve sits high and rises; the median curve sits low and rises. Each legend goes
# in the corner its own curves leave empty.
RB_LEGEND_LOC = {"median": "upper left", "mean": "lower right"}
# The three-arm boxes: the ECDF pair's canvas width, and the share of each count's slot left
# as white space between groups.
RB_BOX_WIDE, RB_BOX_GAP = 1.52, 0.28

# ── the eight-method versions ────────────────────────────────────────────────────
# `posecheck_<Method>.json` holds the published baselines' per-molecule strain, and has all
# along -- but those exports carry no SMILES, so rotatable bonds could not be counted from
# them. The molecules themselves are in the results bundle, so the count is recovered by
# joining the two.
#
# WHERE THE ROTATABLE BONDS COME FROM, AND WHY THE JOIN IS SAFE. The baselines' molecules
# live in `results/task2-drugdesign/<M>/samples/meta/` as the TargetDiff meta format: one
# list per test pocket of {mol, smiles, ligand_filename, pred_pos}. `export_posecheck_json.py`
# built `posecheck_<Method>.json` from THE SAME meta, walking each pocket's entries in order
# and skipping the ones whose `mol` is None -- so the two are the same molecules in the same
# order, and `_rb_load_meta` reproduces its loader exactly (base + `_part2` concatenated per
# pocket; `_gap.pt` is present in the bundle but that loader does not use it, so neither
# does this one).
#
# That is an argument, not evidence, so the join is CHECKED rather than trusted: for every
# pocket the heavy-atom sequence recomputed from the meta must equal the `n` sequence in the
# export, position by position, and a mismatch aborts. It matches 100/100 pockets for all
# five methods.
#
# MIXING THE TWO SCORING RUNS IS SAFE FOR STRAIN, AND ONLY FOR STRAIN. The baselines were
# scored against the whole `*_rec.pdb` receptor; the local arms are read from the pocket10
# crop in their target `metrics.json`. Strain is a property of the ligand's own conformer --
# UFF relaxation under a position constraint -- so receptor scope cannot enter it, and
# measuring the same molecules both ways confirms it does not: median |relative difference|
# 0.6-1.0 %, which is the run-to-run noise `num_confs=50` already carries. Do NOT extend
# this to clashes or interactions: those are receptor-dependent and a crop cannot see an
# atom it does not contain.
#
# label, bundle folder, meta stem. DecompDiff's meta is the reference-prior run, which is
# the one export_posecheck_json.py scored. FuncBind's meta reached the bundle on 2026-09-10
# and joins cleanly (100/100 pockets, 9,992 molecules); its shard SDFs under
# `funcbind/artifacts/reproduction/crossdocked/paper_run` were tried as a substitute and
# FAILED this same check -- a different sampling run from the one PoseCheck scored -- so do
# not reach for them again if the meta ever goes missing.
RB_BASELINES = [
    ("AR",         "AR",         "AR"),
    ("Pocket2Mol", "Pocket2Mol", "Pocket2Mol"),
    ("DiffSBDD",   "DiffSBDD",   "DiffSBDD"),
    ("DecompDiff", "DecompDiff", "DecompDiff_ref_prior"),
    ("FuncBind",   "FuncBind",   "FuncBind"),
]
# Thin and dashed: with eight models on one axis hue alone is not enough, and the three
# local arms are the subject while these are context. Same channel split the nine-series
# by-atom-range figures already use.
RB_BASE_LW = 1.6
# The per-method dashes that used to live here are now METHOD_DASH in 00_core.py, read
# through dash(). This table had drifted a whole method out of step with the ECDF family's,
# so the same baseline was drawn with two different patterns depending on the figure.
# The eight-method figures draw every method through pale(): CoDE's lighter tint, the one the
# Vina per-atom family (dock-per-atom-v*) carries, and the palette colour for everyone else.
# The shared #4363D8 was the only saturated hue among nine series (changed 2026-09-13). The
# three-arm figures above keep the full colour: their key comes from arm_handles().
RB_ALL_BOX_GAP = 0.30        # of a bond count's width, left clear between neighbouring groups
# The eight-method box panel is wider and taller than the house figure -- eight boxes have
# to fit inside each of thirteen counts -- and the share panels are a 3x3 block of small ones.
RB_BOX_WIDE_1, RB_BOX_TALL = 1.72, 1.22
# Four columns retain the former per-panel width; eight methods occupy two rows.
RB_DIST_COLS, RB_DIST_WIDE, RB_DIST_TALL = 2, 1.34 * 2 / 3 * 0.9, 0.66
RB_BOX_YLIM = (1e0, 1e5)
# THE AXIS STOPS AT 12 ROTATABLE BONDS, because each INDIVIDUAL count past it is thin and
# stretching the axis to the last molecule anyone made (25, one VoxBind ligand) spent two
# thirds of the width on a handful of boxes.
#
# THE CUMULATIVE TAIL IS NOT NEGLIGIBLE, THOUGH, and the figure must not imply it is: 5.0 %
# of VoxBind's ligands, 3.1 % of DecompDiff's and 2.8 % of DiffSBDD's have more than 12
# rotatable bonds (Pocket2Mol is the outlier at 0.0 %). So every share panel PRINTS its own
# excluded percentage rather than letting the cap pass silently. Raise RB_X_MAX to see them.
RB_X_MAX = 12
# Above this many counts the axis is split over two rows so the boxes stay wide enough to
# read; at RB_X_MAX = 12 it is one row. Raise RB_X_MAX and the split comes back on its own.
RB_SPLIT_ABOVE = 14
# Boxes are drawn further into the tail than the sibling figures' MIN_N=25 allows, so every
# method covers its own full range instead of being clipped to the narrowest one. Ten
# molecules is the floor for a box to carry quartiles at all; the share figure is what tells
# the reader which end of the axis is thin. The crystal ligands are a LINE in the box
# figure, not a ninth box series, so this floor never applies to them -- theirs is
# RB_MIN_REF over the +-RB_REF_WIN window, and it is why their line stops at 9.
RB_GRID_MIN_N = 10
# Floor for the log-scaled share panels; below this a bond count is empty, not rare.
RB_DIST_FLOOR = 0.05
RB_KEY_H = 0.92              # inches of key strip under the line panels; see _rb_lines()
# The summary table bins the axis; the figures keep every count. Half-open, so these are
# {0}, {1,2}, {3,4}, {5,6}, {7,8,9}, {10+}.
RB_BIN_EDGES = [0, 1, 3, 5, 7, 10, 10 ** 6]
RB_BIN_LABELS = ["0", "1–2", "3–4", "5–6", "7–9", "10+"]

# The three arms this section runs locally, addressed by their stable KEYS -- their LABELS
# moved ("VoxBind + Ours" -> "Ours" -> "CoDE") and may move again. They are picked BY KEY
# rather than by walking ARMS: that list grew from three arms to eight on 2026-09-10 when
# the five published baselines were added to it, pointing at `exps/baselines_pose/<m>`,
# which another job is still filling. Those trees carry PoseBusters but no `posecheck` block
# yet, so walking ARMS here would load three methods' worth of rows with `s=None` and
# silently empty every range these builders compute.
#
# WHEN `exps/baselines_pose/` IS COMPLETE the bundle join becomes unnecessary: those trees
# carry SMILES and will carry strain, so the three-arm figures above will cover every method
# through the shared loader alone and these two can be retired.
RB_LOCAL_KEYS = ("targetdiff", "vanilla", "ours_v1")
RB_LOCAL_LABELS = []         # filled by the loaders from ARMS, in RB_LOCAL_KEYS order
RB_RELABEL = {"ours_v1": "CoDE"}
# What the crystal ligands are CALLED in the eight-method keys (2026-09-14), matching the
# strain ECDF. The shared REF_LABEL still names them in the data and in every other family.
RB_REF_NAME = "Reference"

# The per-ATOM eight-method figure. No join is needed there at all: unlike the rotatable-bond
# axis, which had to go back to the results bundle for SMILES, heavy-atom count is already
# in `posecheck_<Method>.json`. Only the p79 pockets are kept, by the export's own index.
RB_ATOM_BASELINES = ["AR", "Pocket2Mol", "DiffSBDD", "DecompDiff", "FuncBind"]
RB_X_LO, RB_X_HI = 5, 44
# DecompDiff PUTS EXACTLY ONE SIZE IN EACH POCKET, AND IT IS THE CRYSTAL LIGAND'S -- checked
# on every run, not assumed. Its reference prior fixes the atom budget per pocket, so on
# that axis DecompDiff is not a distribution but a comb over the 79 reference sizes, with
# exact zeros at 7, 24, 30, 34, 36, 39-41, 43-44 atoms. So EVERY model curve pools a +-1
# atom window: drawn per exact count DecompDiff's line would break ten times inside the axis
# on molecules that were never going to exist. The window is one atom wide, the same
# treatment for all eight methods, and it moves a median of a continuous quantity almost
# nowhere at these counts. The one hole it does not fill -- 40 atoms, where no reference
# sits within +-1 -- is left as a break rather than widened away.
RB_WIN = 1
RB_MEAN_CLIP = 1e6           # the per-atom mean panel's top, as in the three-arm figure

_RB_CHEM = None
_RB_CACHE = {}


def _rb_rdkit():
    """RDKit, imported on first use rather than at the top of draw.py. The chemistry toolkit
    costs seconds to import and is the only thing in this file the other figure families do
    not need, so a run that draws none of these four never pays for it."""
    global _RB_CHEM
    if _RB_CHEM is None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")
        _RB_CHEM = (Chem, rdMolDescriptors)
    return _RB_CHEM


def _rb_rot_bonds(smiles):
    """RDKit's strict rotatable-bond count, or None if the SMILES will not parse. Cached:
    the arms carry ~60k molecules and many repeat."""
    if not smiles:
        return None
    if smiles not in _RB_CACHE:
        chem, desc = _rb_rdkit()
        mol = chem.MolFromSmiles(smiles)
        _RB_CACHE[smiles] = None if mol is None else int(desc.CalcNumRotatableBonds(mol))
    return _RB_CACHE[smiles]


def _rb_attach(rows):
    """Add `rb` to each row in place, and report how many rows could not get one."""
    bad = 0
    for r in rows:
        r["rb"] = _rb_rot_bonds(r.get("smi"))
        bad += r["rb"] is None
    return bad


def _rb_reference_curve(per, xs, f):
    """reference_curve over the NARROW +-RB_REF_WIN window -- see the banner. None where the
    window is too thin to mean anything, which leaves a gap in the line rather than an
    invented value."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= RB_REF_WIN for v in vals]
        out.append(f(p) if len(p) >= RB_MIN_REF else None)
    return out


def _rb_ref_window_n(per, xs):
    """The reference's n is the WINDOWED pool, not the exact-count one: the plotted value
    comes from +-RB_REF_WIN, so an n beside it that counted only the exact bond count would
    read as 4 ligands supporting a point that 16 produced."""
    return [sum(len(v) for x2, v in per.items() if abs(x2 - x) <= RB_REF_WIN) for x in xs]


def _rb_cell(v):
    return "" if v is None else round(v, 3)


def _rb_tail_share(per):
    """% of an arm's molecules whose relaxation ran past RB_TAIL -- the tail that decides its
    mean curve, and the reason the mean curve is not a location statistic."""
    vals = [v for vs in per.values() for v in vs]
    return round(100 * sum(v > RB_TAIL for v in vals) / len(vals), 2) if vals else None


def _rb_tail_share_rows(rows):
    v = [r["s"] for r in rows if r["s"] is not None]
    return round(100 * sum(x > RB_TAIL for x in v) / len(v), 2) if v else None


# ── the three local arms, per rotatable bond ─────────────────────────────────────
def _rb_local():
    """(per-arm {rb: [strain]}, the crystal ligands' own, the (name, arms) variants, dropped).

    THE ARMS ARE THE ONES THAT CARRY STRAIN, not all of ARMS. That list grew to eight on
    2026-09-10 and the five staged baselines hold PoseBusters only, so `variants()` with no
    field would put five all-None curves in `all` and leave x_range intersecting an empty
    set. `variants("s", data)` is the test the per-atom sibling already applies."""
    data, p79_rows, refrows = pose_data()
    bad = sum(_rb_attach(rows) for rows in p79_rows.values()) + _rb_attach(refrows)
    per_arm = {key: by_size(p79_rows[key], "s", key="rb") for _, key, _ in ARMS}
    ref_per = by_size(refrows, "s", key="rb")
    return per_arm, ref_per, variants("s", data), bad


def _rb_strain_panel(out, xs, arms, variant, stat, per_arm, ref_per):
    """One statistic, one panel, one file — the per-atom builder's layout with this axis."""
    f = RB_STATS[stat]
    fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(xs, _rb_reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        y = model_curve(per_arm[key], xs, f)
        ax.plot(xs, y, color=pale(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")

    ax.set_yscale("log")
    furniture(ax, ylabel=RB_Y_LABEL.format(stat=stat), xlabel=RB_X_LABEL,
              xlim=(xs[0] - 0.35, xs[-1] + 0.35), xloc=RB_XTICK)
    legend(ax, arm_handles(arms), loc=RB_LEGEND_LOC[stat], fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_per_rotbond_{stat}_{variant}")


def _rb_strain_boxes(out, xs, arms, variant, per_arm, ref_per):
    """The distribution at each rotatable-bond count, one box per arm, dodged within the
    count. Wider canvas than the line panels (the ECDF pair's width, already in the house)
    because this draws 15 counts x 2-3 arms of boxes on one axis."""
    fig, ax = plt.subplots(figsize=(FIG_W * RB_BOX_WIDE, PANEL_H), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Dodge: n boxes centred on the count, leaving RB_BOX_GAP of the slot as white space so
    # neighbouring counts stay visually separate groups.
    n = len(arms)
    slot = (1.0 - RB_BOX_GAP) / n
    for i, (lab, key, _) in enumerate(arms):
        offs = (i - (n - 1) / 2) * slot
        vals = [np.clip(per_arm[key].get(x, []), RB_STRAIN_FLOOR, None) for x in xs]
        col = pale(lab)
        bp = ax.boxplot(vals, positions=[x + offs for x in xs], widths=slot * 0.86,
                        whis=RB_WHIS, showfliers=False, patch_artist=True, zorder=5,
                        manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=DATA_ALPHA, edgecolor=col, linewidth=1.05)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                art.set(color=col, linewidth=1.05)
        for med in bp["medians"]:
            med.set(color=INK, linewidth=1.5, solid_capstyle="butt")

    ax.plot(xs, _rb_reference_curve(ref_per, xs, RB_STATS["median"]), color=REF_COLOR,
            lw=REF_LW, ls=DASH, zorder=6, dash_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(bottom=RB_STRAIN_FLOOR)
    furniture(ax, ylabel=f"Strain (kcal mol⁻¹)\n{RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct whiskers",
              xlabel=RB_X_LABEL, xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=RB_XTICK)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    legend(ax, arm_handles(arms), loc="upper left", fontsize=11.5)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_box_per_rotbond_{variant}")


def _rb_local_csv(out, xs, arms, per_arm, ref_per):
    """The curves themselves plus the per-count molecule counts, so a reader can check a
    figure without re-running it — and so the rotatable-bond DISTRIBUTION is on the record.
    It is not drawn as a strip (the per-atom figure this mirrors has no strip either), but
    the arms differ in it, and any claim about a curve has to be read against `n`."""
    rows, summary = [], {}
    for lab, key, _ in arms:
        med = model_curve(per_arm[key], xs, RB_STATS["median"])
        mean = model_curve(per_arm[key], xs, RB_STATS["mean"])
        summary[lab] = {"n_scored": sum(len(v) for v in per_arm[key].values()),
                        "strain_gt_1e4": _rb_tail_share(per_arm[key]), "median": med}
        for i, x in enumerate(xs):
            rows.append([lab, x, len(per_arm[key].get(x, ())),
                         _rb_cell(med[i]), _rb_cell(mean[i])])
    med = _rb_reference_curve(ref_per, xs, RB_STATS["median"])
    mean = _rb_reference_curve(ref_per, xs, RB_STATS["mean"])
    summary[REF_LABEL] = {"n_scored": sum(len(v) for v in ref_per.values()),
                          "strain_gt_1e4": _rb_tail_share(ref_per), "median": med}
    for i, x in enumerate(xs):
        rows.append([REF_LABEL, x, _rb_ref_window_n(ref_per, xs)[i],
                     _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_per_rotbond",
              ["arm", "rotatable_bonds", "n", "strain_median", "strain_mean"], rows)
    return summary


def _rb_local_log(xs, summary, ranges, bad):
    print(f"  {len(P79)}-pocket set · "
          + " · ".join(f"{v} x {r[0]}-{r[-1]} rotatable bonds" for v, r in ranges.items())
          + f" (counts where every drawn arm has ≥{MIN_N} molecules)")
    if bad:
        print(f"  {bad} molecules dropped: SMILES would not parse")
    header = [0, 2, 4, 6, 8, 10, 12]
    print("  strain MEDIAN at a given rotatable-bond count, and the tail behind the mean")
    print(f"  {'arm':16s} {'mols':>7s} " + " ".join(f"{'rb=' + str(h):>7s}" for h in header)
          + f" {'>1e4':>7s}")
    for lab, d in summary.items():
        cells = []
        for h in header:
            v = d["median"][xs.index(h)] if h in xs else None
            cells.append(f"{v:7.1f}" if v is not None else f"{'—':>7s}")
        print(f"  {lab:16s} {d['n_scored']:7d} " + " ".join(cells)
              + f" {d['strain_gt_1e4']:6.2f}%")
    print("  The mean figure is drawn unclipped and spans ~9 decades: the tail above is "
          "what puts it there. Read the median.")


@figure("fig-posecheck-strain-per-rotbond", folder="fig-posecheck/strain-energy", needs=("metrics.json (posecheck.strain, smiles)",))
def draw_posecheck_strain_per_rotbond(out):
    """Strain per rotatable bond — mean and median, core only.

    No `all`: see core_only(). The eight-method view is strain_rotbond_all_methods."""
    use_style()
    per_arm, ref_per, vary, bad = _rb_local()
    ranges, all_arms = {}, []
    for variant, arms in core_only(vary):
        xs = x_range({key: per_arm[key] for _, key, _ in arms}, arms)
        ranges[variant] = xs
        all_arms = arms
        for stat in ("median", "mean"):
            _rb_strain_panel(out, xs, arms, variant, stat, per_arm, ref_per)
    summary = _rb_local_csv(out, ranges["core"], all_arms, per_arm, ref_per)
    _rb_local_log(ranges["core"], summary, ranges, bad)


@figure("fig-posecheck-strain-box-per-rotbond", folder="fig-posecheck/strain-energy",
        needs=("metrics.json (posecheck.strain, smiles)",))
def draw_posecheck_strain_box_per_rotbond(out):
    """The strain DISTRIBUTION at each rotatable-bond count, as boxes — core only.

    No `all`: see core_only(). The eight-method view is strain_box_per_rotbond_all_methods."""
    use_style()
    per_arm, ref_per, vary, bad = _rb_local()
    for variant, arms in core_only(vary):
        xs = x_range({key: per_arm[key] for _, key, _ in arms}, arms)
        _rb_strain_boxes(out, xs, arms, variant, per_arm, ref_per)
        print(f"  {variant:4s} {len(arms)} arms x {xs[0]}-{xs[-1]} rotatable bonds "
              f"({RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct whiskers, no fliers)")
    if bad:
        print(f"  {bad} molecules dropped: SMILES would not parse")


# ── all eight methods ────────────────────────────────────────────────────────────
def _rb_bundle():
    """The published baselines' sample bundle, which is where their molecules are."""
    p = REPO / "results" / "task2-drugdesign"
    if not p.is_dir():
        raise FileNotFoundError(f"missing the baseline sample bundle {p}")
    return p


def _rb_load_meta(folder, stem):
    """export_posecheck_json.py's loader, reproduced exactly: base, then `_part2`
    concatenated PER POCKET. `_gap.pt` ships in the bundle and that loader ignores it, so
    including it here would shift every molecule after the split."""
    import torch                    # local, like RDKit above: nothing else here needs it
    d = _rb_bundle() / folder / "samples" / "meta"
    meta = torch.load(d / f"{stem}.pt", weights_only=False)
    part2 = d / f"{stem}_part2.pt"
    if part2.exists():
        meta = [a + b for a, b in zip(meta, torch.load(part2, weights_only=False))]
    return meta


def _rb_baseline_rows(label, folder, stem, keep):
    """Rows shaped like the shared loader's, for the p79 pockets, with the join checked.

    THE CHECK RUNS OVER EVERY POCKET THE EXPORT HOLDS -- all 100 -- while only the p79 ones
    become rows. Checking just the 79 that are drawn would leave the other 21 as evidence
    nobody looked at, and they cost nothing: the meta is already in memory and the export
    already carries them."""
    meta = _rb_load_meta(folder, stem)
    export = json.load(open(legacy("fig-posecheck", f"posecheck_{label}.json")))
    by_pocket = collections.defaultdict(list)
    for m in export["molecules"]:
        by_pocket[m["p"]].append(m)

    rows, unparsed, checked = [], 0, 0
    for p in sorted(by_pocket):
        entries = [e for e in meta[p] if e.get("mol") is not None]
        scored = by_pocket[p]
        mine = [int(e["mol"].GetNumAtoms()) for e in entries[:len(scored)]]
        if mine != [m["n"] for m in scored]:
            raise SystemExit(f"{label}: meta/export heavy-atom sequence differs at pocket "
                             f"{p} -- the join is not valid, refusing to guess")
        checked += 1
        if p not in keep:
            continue
        for e, m in zip(entries, scored):
            rb = _rb_rot_bonds(e.get("smiles"))
            unparsed += rb is None
            rows.append({"n": m["n"], "s": m["s"], "c": m["c"], "rb": rb})
    return rows, unparsed, checked


def _rb_locals_from_arms():
    """(label -> rows) for the three arms we run locally, picked BY KEY out of ARMS, plus
    the crystal-ligand rows. See the banner for why it is by key and not a walk."""
    _, p79_rows, refrows = pose_data()
    by_key = {key: (lab, root) for lab, key, root in ARMS}
    series, local = {}, []
    for key in RB_LOCAL_KEYS:
        if key not in by_key:
            raise SystemExit(f"ARMS no longer defines {key!r}")
        lab = RB_RELABEL.get(key, by_key[key][0])
        series[lab] = p79_rows[key]
        local.append(lab)
    RB_LOCAL_LABELS[:] = local
    return series, refrows


def _rb_load_all():
    """(label -> rows) for every drawn method on the rotatable-bond axis, plus the crystal
    ligands and the join-check counts."""
    series, refrows = _rb_locals_from_arms()
    for rows in series.values():
        _rb_attach(rows)
    _rb_attach(refrows)

    keep = {int(t.split("_")[1]) for t in P79}
    checks = []
    for lab, folder, stem in RB_BASELINES:
        rows, bad, checked = _rb_baseline_rows(lab, folder, stem, keep)
        series[lab] = rows
        checks.append((lab, len(rows), bad, checked))
    return series, refrows, checks


def _rb_order(labels):
    """The drug-design table's row order, ours last. Anything the table does not name is a
    bug rather than something to append quietly, so it raises."""
    unknown = [l for l in labels if l not in RB_ORDER]
    if unknown:
        raise SystemExit(f"not in table_drug_design.tex's order: {unknown}")
    return [l for l in RB_ORDER if l in labels]


def _rb_style_of(label):
    """Weight from provenance, DASH FROM THE SHARED TABLE.

    The weight still keys on RB_LOCAL_LABELS: the three arms we run locally are drawn heavier
    in this family. The dash no longer does. It used to return solid for every local arm,
    which handed TargetDiff -- a published baseline we happen to run ourselves -- the solid
    line that means OURS in every other figure. dash() knows the difference."""
    lw = MODEL_LW if label in RB_LOCAL_LABELS else RB_BASE_LW
    return lw, dash(label)

def _rb_panel_order(labels):
    """Same order as `_rb_order`, with the crystal ligands FIRST -- they are the table's
    first row and the thing every other panel is read against."""
    return [REF_LABEL] + _rb_order(labels)


def _rb_handles(labels, solid=False):
    """`solid` for the box figure: nothing in it is a dashed line, so a dashed swatch in the
    key advertises an encoding the panel does not use. The crystal ligands are SOLID in both
    (2026-09-14), named "Reference" and drawn through display(), so this family's keys read
    like the strain ECDF's and the per-atom panel's -- solid marks the series a reader returns
    to, every baseline carries a dash, and VoxBind shows its sigma as a subscript."""
    h = [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=SOLID, label=RB_REF_NAME)]
    for lab in labels:
        lw, ls = _rb_style_of(lab)
        h.append(Line2D([], [], color=pale(lab), lw=MODEL_LW if solid else lw,
                        ls="-" if solid else ls, label=display(lab)))
    return h


def _rb_line_curve(per_lab, xs, f):
    """The statistic at each exact bond count, null where that count holds fewer than MIN_N
    molecules. THE AXIS IS FIXED AT 0-RB_X_MAX rather than cut back to the counts every
    method can answer: past nine bonds the methods thin out at very different rates --
    Pocket2Mol has 40 ligands at nine and single figures at eleven, CoDE and VoxBind still
    have hundreds -- and the old rule let the emptiest method decide where everyone's line
    stopped. Now each line simply ends where its own method ran out, which is the more
    informative thing to show, and the crystal ligands' dashed line ends earlier still: 79
    of them cannot fill a window at twelve bonds."""
    return [f(per_lab[x]) if len(per_lab.get(x, ())) >= MIN_N else None for x in xs]


def _rb_lines(out, xs, per, ref_per, labels, stat):
    """One statistic, one panel, and the key in a strip of its own beneath it.

    NINE SERIES DO NOT LEAVE A CORNER FREE. Inside the axes this key covered FuncBind's
    spike at six bonds and everything above ~500 kcal/mol on the left half -- the part of
    the figure that carries the finding. Under the panel it covers nothing, and it is the
    same 3x3 block the box figure's key is."""
    f = RB_STATS[stat]
    fig, (ax, key) = plt.subplots(2, 1, figsize=(FIG_W, PANEL_H + RB_KEY_H), dpi=220,
                                  gridspec_kw={"height_ratios": [PANEL_H, RB_KEY_H]})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    key.axis("off")
    # Solid, like the key's swatch and like the ECDF's crystal ligands (2026-09-14). The
    # three-arm core figures above keep the dash -- there the reference is one of three lines,
    # not one of nine, and nothing else in those panels competes for solid.
    ax.plot(xs, _rb_reference_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
            ls=SOLID, zorder=4, solid_capstyle="round")
    for lab in labels:
        lw, ls = _rb_style_of(lab)
        ax.plot(xs, _rb_line_curve(per[lab], xs, f), color=pale(lab), lw=lw, ls=ls,
                zorder=5, solid_capstyle="round")
    ax.set_yscale("log")
    furniture(ax, ylabel=f"Strain {stat}\n(kcal mol⁻¹)", xlabel=RB_X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=RB_XTICK)
    legend(key, _rb_handles(labels), loc="center", ncol=3, fontsize=11)
    fit(fig, pad=0.5)
    save(fig, out, f"strain_rotbond_all_methods_{stat}")


def _rb_axis(labels, refrows, series):
    """The shared x range and per-method rows both figures are drawn over, so the boxes and
    the share panels cannot end up covering different molecules."""
    panels = _rb_panel_order(labels)
    rows_by_lab = {lab: series[lab] for lab in labels}
    rows_by_lab[REF_LABEL] = refrows
    per = {lab: by_size(rows_by_lab[lab], "s", key="rb") for lab in panels}
    dist = {lab: collections.Counter(r["rb"] for r in rows_by_lab[lab]
                                     if r["rb"] is not None) for lab in panels}
    # 0 to RB_X_MAX, the same span the line figures get, so the three are read against one
    # axis BY CONSTRUCTION rather than by today's data happening to agree. Boxes still
    # appear only where a method has RB_GRID_MIN_N molecules, so pinning the axis adds no
    # box; it only stops the axis shrinking when the tail thins.
    xs = list(range(0, RB_X_MAX + 1))
    beyond = {lab: 100 * sum(n for x, n in dist[lab].items() if x > RB_X_MAX)
              / max(sum(dist[lab].values()), 1) for lab in panels}
    return panels, per, dist, xs, beyond


def _rb_all_boxes(out, series, refrows, labels):
    """Strain against rotatable bonds, every method on one axis, dodged inside each count.

    The paper's Fig. 12 shape: one panel of boxplots per number of rotatable bonds, all
    methods together. (The paper packs 151 boxes into one row and colours each box by its
    median strain, spending colour on the value; here colour stays the method key it is in
    every other 260910 figure, and the legend carries it.)

    The crystal ligands stay the dashed grey line rather than becoming a ninth box series:
    79 of them over thirteen bond counts cannot fill a box at every count, and as a line
    they are the same ruler here that they are in every sibling figure."""
    panels, per, dist, xs, beyond = _rb_axis(labels, refrows, series)
    if len(xs) > RB_SPLIT_ABOVE:
        split = len(xs) - len(xs) // 2                 # low row takes the extra count
        bands = [xs[:split], xs[split:]]
    else:
        bands = [xs]
    ref_line = {x: v for x, v in
                zip(xs, _rb_reference_curve(per[REF_LABEL], xs, RB_STATS["median"]))}

    fig, axes = plt.subplots(len(bands), 1,
                             figsize=(FIG_W * RB_BOX_WIDE_1,
                                      PANEL_H * RB_BOX_TALL * len(bands)),
                             dpi=220, squeeze=False)
    fig.patch.set_facecolor("white")
    n = len(labels)
    slot = (1.0 - RB_ALL_BOX_GAP) / n
    for band, ax in zip(bands, axes.ravel()):
        ax.set_facecolor("white")
        for i, lab in enumerate(labels):
            offs = (i - (n - 1) / 2) * slot
            drawn = [x for x in band if x in per[lab] and len(per[lab][x]) >= RB_GRID_MIN_N]
            if not drawn:
                continue
            col = pale(lab)
            bp = ax.boxplot([np.clip(per[lab][x], RB_STRAIN_FLOOR, None) for x in drawn],
                            positions=[x + offs for x in drawn], widths=slot * 0.88,
                            whis=RB_WHIS, showfliers=False, patch_artist=True, zorder=5,
                            manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=col, alpha=DATA_ALPHA, edgecolor=col, linewidth=0.8)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=col, linewidth=0.8)
            for med in bp["medians"]:
                med.set(color=INK, linewidth=1.1, solid_capstyle="butt")
        ax.plot(band, [ref_line.get(x) for x in band], color=REF_COLOR, lw=REF_LW,
                ls=SOLID, zorder=6, solid_capstyle="round")
        ax.set_yscale("log")
        # A FIXED FIVE DECADES, AND THE TAIL IS ALLOWED TO RUN OFF THE TOP. FuncBind's
        # 95th percentile at 6 rotatable bonds reaches ~1e11; autoscaling to it stretched
        # the panel over fourteen decades and pressed every box into the bottom fifth. The
        # window is pinned so the boxes -- which all sit between 1e0 and 1e5 -- stay legible
        # across every rebuild, and the whiskers that leave the top simply leave it. How
        # much tail each method carries is reported as `strain_gt_1e4`, not drawn.
        ax.set_ylim(*RB_BOX_YLIM)
        furniture(ax, ylabel=f"Strain (kcal mol⁻¹)\n{RB_WHIS[0]:g}–{RB_WHIS[1]:g}th pct "
                             "whiskers",
                  xlabel=RB_X_LABEL, xlim=(band[0] - 0.62, band[-1] + 0.62), xloc=1)
        ax.set_xticks(band)
    # Nine entries -- the crystal ligands plus eight methods -- in three columns, so the key
    # is a 3x3 block rather than one long strip across the top of the panel. Lower right:
    # the boxes climb left-to-right, so the empty corner is under the high bond counts,
    # where only the lower whiskers reach.
    legend(axes.ravel()[0], _rb_handles(labels, solid=True), loc="lower right",
           fontsize=10, ncol=3)
    fit(fig, pad=0.5)
    save(fig, out, "strain_box_per_rotbond_all_methods")


def _rb_distribution(out, series, refrows, labels):
    """Method distributions in a 4x2 grid, each with a dotted, unfilled reference.

    Shares use all ligands as the denominator, including the tail beyond the
    displayed 0-12 bond range. The log scale retains the small per-count shares.
    """
    _, _, dist, xs, beyond = _rb_axis(labels, refrows, series)
    panels = _rb_order(labels)
    reference_total = sum(dist[REF_LABEL].values())
    reference_pct = [100 * dist[REF_LABEL].get(x, 0) / max(reference_total, 1) for x in xs]
    ncol = RB_DIST_COLS
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(FIG_W * RB_DIST_WIDE, PANEL_H * RB_DIST_TALL * nrow),
                             dpi=220, sharex=True, squeeze=False)
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    for i, lab in enumerate(panels):
        ax = flat[i]
        ax.set_facecolor("white")
        col = pale(lab)
        total = sum(dist[lab].values())
        pct = [100 * dist[lab].get(x, 0) / max(total, 1) for x in xs]
        ax.step(xs, pct, where="mid", color=col, lw=DIST_LW, zorder=3)
        ax.fill_between(xs, pct, RB_DIST_FLOOR, step="mid", color=col, alpha=DIST_FILL,
                        lw=0, zorder=2)
        # The reference is a comparison line, with no filled area beneath it.
        ax.step(xs, reference_pct, where="mid", color=REF_COLOR, lw=REF_LW,
                ls=DOT, zorder=4)
        ax.set_yscale("log")
        furniture(ax, ylabel=None,
                  xlim=(xs[0] - 0.6, xs[-1] + 0.6), xloc=3)
        ax.set_ylim(bottom=RB_DIST_FLOOR)
        ax.set_title(display(lab), fontsize=12, color=INK, loc="center", pad=4)
        ax.text(0.97, 0.07,
                f">{RB_X_MAX} bonds: {beyond[lab]:.1f}%\n"
                f"(reference: {beyond[REF_LABEL]:.1f}%)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
                linespacing=1.4, color=AXIS, zorder=6)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    fig.supylabel("% of ligands", fontsize=15.5, color=INK)
    fig.supxlabel(RB_X_LABEL, fontsize=15.5, color=INK)
    # Reserve extra space between the shared y title and the left-column ticks.
    fit(fig, pad=0.5, h_pad=0.9, rect=(0.055, 0, 1, 1))
    save(fig, out, "rotbond_distribution_all_methods")
    print(f"  distribution: {nrow}x{ncol} methods; reference >{RB_X_MAX} bonds: "
          f"{beyond[REF_LABEL]:.1f}% ({reference_total} ligands)")


@figure("fig-posecheck-rotbond-distribution-all-methods", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "results/task2-drugdesign/<M>/samples/meta",
               "metrics.json (posecheck.strain, smiles)"))
def draw_posecheck_rotbond_distribution(out):
    """Rotatable-bond distributions, eight methods in 2x4 with a dotted reference overlay."""
    use_style()
    series, refrows, _ = _rb_load_all()
    _rb_distribution(out, series, refrows, _rb_order(series))


def _rb_bin_of(rb):
    return min(int(np.searchsorted(RB_BIN_EDGES, rb, side="right")) - 1,
               len(RB_BIN_LABELS) - 1)


def _rb_all_csv(out, xs, per, ref_per, labels):
    rows = []
    for lab in labels:
        med = _rb_line_curve(per[lab], xs, RB_STATS["median"])
        mean = _rb_line_curve(per[lab], xs, RB_STATS["mean"])
        for i, x in enumerate(xs):
            n = len(per[lab].get(x, ()))
            # the models are drawn per exact count, so their window IS that count
            rows.append([lab, x, n, n, _rb_cell(med[i]), _rb_cell(mean[i])])
    med = _rb_reference_curve(ref_per, xs, RB_STATS["median"])
    mean = _rb_reference_curve(ref_per, xs, RB_STATS["mean"])
    win = _rb_ref_window_n(ref_per, xs)
    for i, x in enumerate(xs):
        rows.append([REF_LABEL, x, len(ref_per.get(x, ())), win[i],
                     _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_rotbond_all_methods",
              ["arm", "rotatable_bonds", "n", "n_window", "strain_median", "strain_mean"],
              rows)


@figure("fig-posecheck-strain-rotbond-all-methods", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "results/task2-drugdesign/<M>/samples/meta",
               "metrics.json (posecheck.strain, smiles)"))
def draw_posecheck_strain_rotbond_all_methods(out):
    """Strain against rotatable bonds, all eight methods: lines, boxes and the share block."""
    use_style()
    series, refrows, checks = _rb_load_all()
    labels = _rb_order(series)
    per = {lab: by_size(series[lab], "s", key="rb") for lab in labels}
    ref_per = by_size(refrows, "s", key="rb")

    # 0 to RB_X_MAX, always -- the same span the box and share figures cover, so the three
    # are read against one axis. Where a method (or the reference) is too thin at a count,
    # its own line stops; the axis does not.
    xs = list(range(0, RB_X_MAX + 1))
    for s in ("median", "mean"):
        _rb_lines(out, xs, per, ref_per, labels, s)
    _rb_all_boxes(out, series, refrows, labels)
    _rb_distribution(out, series, refrows, labels)
    _rb_all_csv(out, xs, per, ref_per, labels)

    for lab, n_rows, bad, checked in checks:
        print(f"  {lab:12s} join checked on {checked} pockets · {n_rows:,} p79 molecules"
              + (f" · {bad} SMILES would not parse" if bad else ""))
    print(f"  {len(P79)} pockets · {len(labels)} methods + reference · "
          f"x = {xs[0]}-{xs[-1]} rotatable bonds "
          f"(each line drawn where its own method has ≥{MIN_N} molecules)")
    for lab in labels:
        drawn = [x for x in xs if len(per[lab].get(x, ())) >= MIN_N]
        # A break INSIDE a method's span is reported rather than collapsed into the
        # endpoints -- "0-12" over a line with a hole in it would be a false summary.
        gaps = [x for x in range(drawn[0], drawn[-1] + 1) if x not in drawn] if drawn else []
        print(f"    {lab:12s} line drawn {drawn[0]}-{drawn[-1]}"
              + (f", broken at {gaps}" if gaps else "") if drawn else
              f"    {lab:12s} nowhere thick enough to draw")
    ref_drawn = [x for x, v in zip(xs, _rb_reference_curve(ref_per, xs, RB_STATS["median"]))
                 if v is not None]
    print(f"    {REF_LABEL:12s} line drawn {ref_drawn[0]}-{ref_drawn[-1]} "
          f"(±{RB_REF_WIN} window, ≥{RB_MIN_REF} ligands)")
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{b:>8s}" for b in RB_BIN_LABELS)
          + f" {'>1e4':>7s}   (strain median by bin)")
    for lab in labels + [REF_LABEL]:
        rows = refrows if lab == REF_LABEL else series[lab]
        cells = [[r["s"] for r in rows if r["s"] is not None and r["rb"] is not None
                  and _rb_bin_of(r["rb"]) == i] for i in range(len(RB_BIN_LABELS))]
        floor = RB_MIN_REF if lab == REF_LABEL else 1
        vals = [f"{np.median(v):8.1f}" if len(v) >= floor else f"{'—':>8s}" for v in cells]
        n_scored = sum(len(v) for v in (ref_per if lab == REF_LABEL else per[lab]).values())
        print(f"  {lab:16s} {n_scored:6d} "
              f"{round(float(np.mean([r['n'] for r in rows])), 2):6.1f} "
              + " ".join(vals) + f" {_rb_tail_share_rows(rows):6.2f}%")


# ── all eight methods, against ligand SIZE ───────────────────────────────────────
def _rb_atom_baseline_rows(label, keep):
    """The export's p79 molecules, as shared-loader-shaped rows."""
    export = json.load(open(legacy("fig-posecheck", f"posecheck_{label}.json")))
    rows = [{"n": m["n"], "s": m["s"], "c": m["c"]}
            for m in export["molecules"] if m["p"] in keep]
    return rows, len(export["molecules"])


def _rb_atom_load_all():
    series, refrows = _rb_locals_from_arms()
    keep = {int(t.split("_")[1]) for t in P79}
    for lab in RB_ATOM_BASELINES:
        series[lab], _ = _rb_atom_baseline_rows(lab, keep)
    return series, refrows


def _rb_check_decompdiff():
    """The banner's claim, re-checked on every run: one size per pocket, and it is the
    crystal ligand's. If DecompDiff's export is ever replaced by the non-reference-prior
    run this stops holding, and the +-1 window stops being something this figure needs."""
    export = json.load(open(legacy("fig-posecheck", "posecheck_DecompDiff.json")))
    ref = {}
    for t in P79:
        r = rows_of(os.path.join(REF_ROOT, t), reference=True)
        if r:
            ref[int(t.split("_")[1])] = r[0]["n"]
    sizes = collections.defaultdict(set)
    for m in export["molecules"]:
        if m["p"] in ref:
            sizes[m["p"]].add(m["n"])
    one = sum(len(v) == 1 for v in sizes.values())
    same = sum(len(v) == 1 and next(iter(v)) == ref[p] for p, v in sizes.items())
    return one, same, len(sizes)


def _rb_pool(per_lab, a):
    """The molecules behind the point at `a`: its +-RB_WIN neighbours, CLIPPED TO THE AXIS.

    The clip is not cosmetic. DecompDiff has no molecule at 43 or 44 heavy atoms and 125 at
    45+, so an unclipped window put a point at 44 computed entirely from molecules the same
    file reports as beyond the axis -- a plotted value representing nothing near where it
    was plotted. Inside the axis the window still interpolates across its comb, which is
    what it is for; at the edges it no longer extrapolates from outside."""
    return [v for n, vals in per_lab.items()
            if abs(n - a) <= RB_WIN and RB_X_LO <= n <= RB_X_HI for v in vals]


def _rb_curve(per_lab, xs, f):
    """The statistic over that window, None where it is too thin to mean anything -- which
    leaves a gap rather than an invented value, the same rule reference_curve applies to the
    crystal ligands."""
    return [f(p) if len(p) >= MIN_N else None
            for p in (_rb_pool(per_lab, a) for a in xs)]


def _rb_atom_lines(out, xs, per, ref_per, labels, stat):
    """One statistic, one panel, NO KEY.

    The key was dropped on request (2026-09-13). It used to be a 3x3 block in a strip of its
    own beneath the panel -- nine series leave no corner of the axes free -- and the strip
    went with it, so the panel is the house single-panel size. The colours and dashes are the
    ones every sibling figure keys (strain_rotbond_all_methods_*), which is where to read them.

    THE MEAN PANEL IS CLIPPED TO THE BULK and the points that leaves off are NAMED in the
    run log: 4-13 % of molecules fail UFF relaxation and land between 1e4 and 1e13, so one
    of them at a thin count carries that count's mean four decades up and an unclipped axis
    spends fourteen decades on it.

    THE REFERENCE KEEPS THE HEADER'S +-4 WINDOW HERE, not the narrow one the rotatable-bond
    figures use: this axis runs 5-44 atoms, which is what that window was set for."""
    f = RB_STATS[stat]
    clip = RB_MEAN_CLIP if stat == "mean" else None
    # DRAWN IN THE STRAIN ECDF'S HOUSE (2026-09-13), so the two strain figures sit side by side
    # as a pair: its canvas (PCSZ_ECDF_SIZE), its rc (12 pt type, black ink, 170 dpi cropped to
    # the ink), its axis furniture and its label size. Read at call time -- the ECDF family is
    # another part of this file. The y name is one line, as the ECDF's are.
    with plt.rc_context({**PCSZ_RC, **({"savefig.bbox": None} if stat == "median" else {})}):
        size = PCSZ_MAIN_STRAIN_SIZE if stat == "median" else PCSZ_ECDF_SIZE
        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor(PCSZ_BG)
        _pcsz_style(ax)
        # Solid here too, matching the ECDF it pairs with (2026-09-14); AR is already dashed
        # on this axis through RB_BASE_DASH.
        ax.plot(xs, reference_curve(ref_per, xs, f), color=REF_COLOR,
                lw=PCSZ_REF_LW * PCSZ_LW_SCALE, ls=SOLID, zorder=4, solid_capstyle="round")
        dropped = []
        for lab in labels:
            y = _rb_curve(per[lab], xs, f)
            if clip:
                over = [(a, v) for a, v in zip(xs, y) if v is not None and v > clip]
                if over:
                    dropped.append((lab, over))
            _, ls = _rb_style_of(lab)
            # The ECDF's weights, not this family's (2026-09-13): the two panels are read as a
            # pair, and our arms were 3.06 here against 3.9-4.4 there. KeyError rather than a
            # default -- a method missing from that table is a rename to fix, not a thin line.
            lw = PCSZ_LINE_LW[ALIASES.get(lab, lab)]
            ax.plot(xs, y, color=pale(lab), lw=lw * PCSZ_LW_SCALE, ls=ls, zorder=5,
                    solid_capstyle="round")
        ax.set_yscale("log")
        if clip:
            ax.set_ylim(top=clip)
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
        ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
        ax.set_xlabel(X_LABEL, fontsize=PCSZ_LABEL_FS)
        # Two lines again (2026-09-14): it was pulled onto one on 2026-09-13, and the unit now
        # goes back under the name, matching the ECDF's y name beside it. The median panel
        # carries the unit inside the frame instead (2026-09-24); the mean keeps it here.
        ax.set_ylabel("Median\nstrain energy" if stat == "median" else
                      f"Strain {stat}\n(kcal mol⁻¹)", fontsize=PCSZ_LABEL_FS,
                      labelpad=PCSZ_YPAD)
        if stat == "median":
            _pcsz_comparison_layout(fig, ax, yunit="kcal mol⁻¹",
                                    tick_scale=PCSZ_ECDF_TICK_SCALE)
        else:
            fig.tight_layout(pad=0.5)
        save(fig, out, f"strain_per_atom_all_methods_{stat}")
    return dropped


def _rb_atom_csv(out, xs, per, ref_per, labels):
    rows = []
    for lab in labels + [REF_LABEL]:
        p = ref_per if lab == REF_LABEL else per[lab]
        med = (reference_curve(p, xs, RB_STATS["median"]) if lab == REF_LABEL
               else _rb_curve(p, xs, RB_STATS["median"]))
        mean = (reference_curve(p, xs, RB_STATS["mean"]) if lab == REF_LABEL
                else _rb_curve(p, xs, RB_STATS["mean"]))
        win = ([sum(len(v) for x, v in p.items() if abs(x - a) <= REF_WIN) for a in xs]
               if lab == REF_LABEL else [len(_rb_pool(p, a)) for a in xs])
        for i, a in enumerate(xs):
            rows.append([lab, a, len(p.get(a, ())), win[i],
                         _rb_cell(med[i]), _rb_cell(mean[i])])
    write_csv(out, "strain_per_atom_all_methods",
              ["arm", "heavy_atoms", "n", "n_window", "strain_median", "strain_mean"], rows)
    return rows


@figure("fig-posecheck-strain-per-atom-all-methods", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "metrics.json (posecheck.strain)"))
def draw_posecheck_strain_per_atom_all_methods(out):
    """Strain against ligand size, all eight methods — mean and median."""
    use_style()
    series, refrows = _rb_atom_load_all()
    labels = _rb_order(series)
    dd = _rb_check_decompdiff()
    per = {lab: by_size(series[lab], "s") for lab in labels}
    ref_per = by_size(refrows, "s")
    xs = list(range(RB_X_LO, RB_X_HI + 1))
    drops = {s: _rb_atom_lines(out, xs, per, ref_per, labels, s) for s in ("median", "mean")}
    _rb_atom_csv(out, xs, per, ref_per, labels)

    print(f"  {len(P79)} pockets · {len(labels)} methods + reference · "
          f"x = {RB_X_LO}-{RB_X_HI} heavy atoms · ±{RB_WIN}-atom window, "
          f"drawn where it pools ≥{MIN_N} molecules")
    print(f"  DecompDiff: {dd[0]}/{dd[2]} pockets hold a single heavy-atom count and "
          f"{dd[1]}/{dd[2]} of them equal the crystal ligand's — it is size-matched to the "
          f"reference by construction")
    heads = (10, 15, 20, 25, 30, 35, 40)
    print(f"  {'method':16s} {'mols':>6s} {'atoms':>6s} "
          + " ".join(f"{a:>7d}" for a in heads)
          + f" {'drawn':>9s} {'>44 at':>7s} {'>1e4':>7s}   (strain median at n atoms)")
    for lab in labels + [REF_LABEL]:
        rows = refrows if lab == REF_LABEL else series[lab]
        p = ref_per if lab == REF_LABEL else per[lab]
        med = (reference_curve(p, xs, RB_STATS["median"]) if lab == REF_LABEL
               else _rb_curve(p, xs, RB_STATS["median"]))
        cells = [(f"{med[xs.index(a)]:7.1f}" if med[xs.index(a)] is not None
                  else f"{'—':>7s}") for a in heads]
        drawn = [a for a, v in zip(xs, med) if v is not None]
        gaps = [a for a in range(drawn[0], drawn[-1] + 1) if a not in drawn] if drawn else []
        span = (f"{drawn[0]}-{drawn[-1]}" + (f"*{len(gaps)}" if gaps else "")) if drawn \
            else "—"
        beyond = round(100 * sum(r["n"] > RB_X_HI for r in rows) / max(len(rows), 1), 2)
        tail = _rb_tail_share_rows(rows)
        print(f"  {lab:16s} {sum(len(v) for v in p.values()):6d} "
              f"{round(float(np.mean([r['n'] for r in rows])), 2):6.1f} "
              + " ".join(cells) + f" {span:>9s} {beyond:6.1f}% "
              + (f"{tail:6.2f}%" if tail is not None else f"{'—':>7s}"))
        if gaps:
            print(f"    * {lab}'s line breaks inside its span at {gaps} heavy atoms — too "
                  f"few molecules in the window there, not a drawing error")
    for lab, over in drops.get("mean", []):
        pts = ", ".join(f"{a} atoms {v:.2g}" for a, v in over)
        print(f"    strain_per_atom_all_methods_mean · off-panel above {RB_MEAN_CLIP:.0e}: "
              f"{lab:12s} {len(over):2d} of {len(xs)}: {pts}")


# ── all eight methods, the paper's small-multiples form ──────────────────────────
# VoxBind Fig. 12 (arXiv 2405.03961, appendix): ONE PANEL PER METHOD, rotatable bonds on x,
# Tukey boxes WITH fliers, and each box FILLED BY ITS OWN MEDIAN on a diverging colormap. The
# dodged figure above answers "which method is lower at this count"; this one answers "how
# does each method's strain climb with torsional freedom", and the colour lets the reader
# compare levels across panels without lining up y values.
#
# WHAT IS KEPT FROM THE PAPER, WHAT IS NOT. Kept: the grid, x = 0-7, Tukey whiskers (1.5 IQR)
# with fliers, log y, a LINEAR colour norm = median. Changed: nine panels (the crystal ligands
# plus eight methods) in 3x3 rather than 2x4, the house fonts and furniture, and the colormap.
#
# THE COLORMAP IS BUILT FROM THE HOUSE PALETTE, not coolwarm: a light version of CoDE's blue at
# the low end running straight to a light version of VoxBind's sand at the high end, so the
# figure reads in the same two hues as every other 260910 figure.
#
# THE NORM IS A FIXED 0-1000 kcal/mol (was 0-800; see RB_GRID_NORM), ticked every 200. Over
# the eight methods' 64 boxes the median is 136 and the 95th percentile 694; only FuncBind at
# six bonds (5,638) lies above 1000, and it saturates onto the top colour with the colorbar's
# arrow saying so (the run log names every saturated box). The price of
# a linear norm is that medians under ~100 -- every crystal-ligand box, and VoxBind/CoDE at low
# counts -- share the darkest blues; their differences are read off the y axis, as in the paper.
#
# THE AXIS STOPS AT 7 AS THE PAPER'S DOES, which leaves a real share of every method off it
# (more than the dodged figure's cut at 12). That share is printed per method and written to
# the CSV rather than implied away; raise RB_GRID_X_MAX to see it.
#
# THE Y WINDOW IS THE DODGED FIGURE'S FIXED FIVE DECADES. Fliers that relaxed past 1e5 leave
# the top, as they do in the paper; how many per box is in the CSV (`n_above_ylim`).
#
# STAGES. The paper draws this twice, on the generated pose (Fig. 12) and after a local
# force-field minimisation (Fig. 13). Only `generated` exists in the data: no per-molecule
# minimised strain has been computed for any method yet. A stage is a field name on the rows,
# so adding the minimised one is a new entry here once the rows carry it.
RB_GRID_X_MAX = 7
RB_GRID_COLS = 3
# The crystal ligands are 79 molecules over eight counts; at the methods' floor of
# RB_GRID_MIN_N most of their boxes would vanish. Three is the least that gives a box a median
# and quartiles that are not the same point, and the CSV carries n for every box.
RB_GRID_REF_MIN_N = 3
# THE PAPER'S OWN SCALE (2026-09-14). VoxBind Fig. 12/13 colours each box by its median on
# matplotlib's coolwarm, and this used to be a house-built lookalike assembled out of the
# SHARED PALETTE: CoDE's soft blue at 0, through both pale tints, to VoxBind's sand at the
# top. That made the magnitude channel collide with the identity one -- a periwinkle box
# meant "low median" here and "CoDE" in every other figure, a sand one "high median" here
# and "VoxBind" there -- and a reader carrying the palette between figures reads it wrong.
# coolwarm keeps what the house version was reaching for, a light middle so the dark median
# bars and the fliers stay legible, while being far more saturated at both ends than any
# pastel in SOFT, so it cannot be mistaken for a method.
#
# IT IS DIVERGING AND THE QUANTITY IS NOT: median strain runs 0-800 with no meaningful
# midpoint at 400, so the white band is an artefact of the map rather than a feature of the
# data. That is the paper's choice and matching it is the point here, but a sequential map
# would be the more honest encoding if we ever stop matching.
# THE RAMP IS THE HOUSE ONE: HEAT_CMAP, light to dark, decided in 00_core because the
# PoseBusters figures key their cells with the same object. It used to be built here and
# read across from there; it moved up on request (2026-09-15) so neither figure owns the
# other's colours. Since 2026-09-23 it runs light grey -> a darkened CoDE highlight rather
# than near-white -> periwinkle, so the boxes here are greyer at the bottom of the scale and
# deeper at the top; the ends are chosen in 00_core and this figure states no opinion on them.
# TOP AT 1000, NOT 800 (2026-09-15, on request, with the two-stop ramp #FAFDFB -> #3C52DD):
# "1000 and over is the end colour". FuncBind at seven bonds (982) now takes its own shade just
# under the tail instead of saturating; FuncBind at six (5,638) is the one box above, and the
# bar's arrow still says so. Ticks stay every 200, so the bar reads 0-1000 in six numbers.
RB_GRID_NORM, RB_GRID_CTICK = (0.0, 1000.0), 200.0
RB_GRID_TALL = 0.9975 * 1.1  # per row, as a share of PANEL_H
# 0.8x the width the three-arm box figures use (2026-09-13). The nine panels keep their type
# and line weights, so pulling the canvas in is what makes the block read fuller; the boxes
# narrow with the axes, the colourbar keeps its share of the figure.
RB_GRID_WIDE = 0.8 * 1.2 * 1.15
# The outer names are a step above the house 15.5 pt, and the y name sits further off its
# tick labels -- on a 3x3 block the house sizes read as small.
RB_GRID_LABEL_FS, RB_GRID_YPAD = 27, 18
# Everything else in the block moves with the names (2026-09-14), so it reads like the strain
# ECDF, which spends 16 pt on a canvas half this wide. Panel names, the tick numbers under and
# beside every panel, and the colourbar's own ticks -- left at the house 14/14/12 they turned
# into fine print as soon as the axis names went to 22.
RB_GRID_TITLE_FS, RB_GRID_TICK_FS = 22, 18
RB_GRID_CBAR_TICK_FS = RB_GRID_TICK_FS   # one size for every number in the figure
# SET AGAINST THE STRAIN ECDF (2026-09-14, user's call): that figure spends 16.1 pt of type
# and a 1.1 pt spine on a 6.95 in canvas; this one is 14.45 in, so matching it at equal placed
# width would mean 33 pt and 2.5, and matching it per PANEL -- a grid panel is 2.8 x 2.6 in
# against the ECDF axes' 6.0 x 2.8 -- would mean 16 pt and 1.1. These are the geometric mean of
# the two, for a block placed wider than the ECDF but not twice as wide.
# THE PANEL TICK NUMBERS ARE SMALLER THAN THE BAR'S (2026-09-14) even though both were set to
# 20: measured off the PNG the two render at the same 48 px, but the bar's are turned a quarter
# turn and a vertical string of digits reads smaller than a horizontal one. They came down to
# 16 to match what the bar LOOKS like, the bar's own numbers followed them down, and both then
# settled at 18 -- one number size for the whole figure, a step under the panel names.
RB_GRID_MED_COLOR = "#323232"   # the median bar, a touch lighter than INK
# The x name's gap above it, DOUBLED (2026-09-13): measured off the drawn PNG it was 27 px at
# 220 dpi, i.e. 8.8 pt from the tick labels; a labelpad of 15 measures back as 17.7 pt, which
# is that doubled (the pad is spent from the axes bbox, so it is ~3 pt more than the gap it
# buys). In points, like the y pad, which is why the name moved off fig.supxlabel and onto an
# axes -- see the call.
RB_GRID_XPAD = 15
# The colourbar's name is the block's THIRD outer name, so it carries the same size as the
# two axis names rather than a step below them (2026-09-13).
RB_GRID_CBAR_FS = RB_GRID_LABEL_FS
# Spines and MAJOR tick marks 1.2x the house weight: nine small panels read as washed out at
# 1.35, and 1.6x was too heavy. Minor ticks (the log decades' 2-9) keep their own weight.
RB_GRID_AXIS_LW = 2.0
# stage -> (row field, y-axis qualifier)
RB_GRID_STAGES = {"generated": ("s", "generated pose")}


def _rb_grid_boxes(rows, field, floor):
    """[(bond count, clipped values)] for every count on the axis that holds `floor` rows."""
    per = by_size(rows, field, key="rb")
    return [(x, np.clip(per[x], RB_STRAIN_FLOOR, None))
            for x in range(0, RB_GRID_X_MAX + 1) if len(per.get(x, ())) >= floor]


def _rb_grid(out, series, refrows, labels, stage):
    field, qualifier = RB_GRID_STAGES[stage]
    panels = _rb_panel_order(labels)
    rows_by_lab = {**{lab: series[lab] for lab in labels}, REF_LABEL: refrows}
    boxes = {lab: _rb_grid_boxes(rows_by_lab[lab], field,
                                 RB_GRID_REF_MIN_N if lab == REF_LABEL else RB_GRID_MIN_N)
             for lab in panels}

    all_meds = [float(np.median(v)) for lab in panels for _, v in boxes[lab]]
    norm = matplotlib.colors.Normalize(*RB_GRID_NORM)
    cmap = HEAT_CMAP
    extend = {(False, False): "neither", (True, False): "min", (False, True): "max",
              (True, True): "both"}[(min(all_meds) < norm.vmin, max(all_meds) > norm.vmax)]

    nrow = -(-len(panels) // RB_GRID_COLS)
    fig, axes = plt.subplots(nrow, RB_GRID_COLS, sharex=True, sharey=True,
                             figsize=(FIG_W * RB_BOX_WIDE_1 * RB_GRID_WIDE,
                                      PANEL_H * RB_GRID_TALL * nrow),
                             dpi=220, squeeze=False, layout="constrained")
    fig.patch.set_facecolor("white")
    flat = axes.ravel()
    for ax, lab in zip(flat, panels):
        ax.set_facecolor("white")
        drawn = boxes[lab]
        if drawn:
            bp = ax.boxplot([v for _, v in drawn], positions=[x for x, _ in drawn],
                            widths=0.72, whis=1.5, showfliers=True, patch_artist=True,
                            manage_ticks=False, zorder=5,
                            flierprops=dict(marker="d", markersize=2.6, markerfacecolor=INK,
                                            markeredgecolor="none", alpha=DATA_ALPHA))
            for box, (_, v) in zip(bp["boxes"], drawn):
                # No outline (2026-09-14): the fill IS the median through the colourmap, and a
                # dark edge on nine panels of eight boxes read as a grid of frames.
                box.set(facecolor=cmap(norm(float(np.median(v)))), edgecolor="none",
                        linewidth=0)
            for part in ("whiskers", "caps"):
                for art in bp[part]:
                    art.set(color=INK, linewidth=0.9)
            for med in bp["medians"]:
                # Up from 1.3, settled at 2.0 (2026-09-14; 2.6 was heavy): with the box outline
                # gone this is the only dark mark on the fill, and it is the number the colour
                # encodes.
                med.set(color=RB_GRID_MED_COLOR, linewidth=2.0, solid_capstyle="butt")
            # Tens of thousands of flier markers as vector paths make the PDF/SVG unplaceable;
            # the points are rasterised inside an otherwise vector figure.
            for fl in bp["fliers"]:
                fl.set_rasterized(True)
        ax.set_yscale("log")
        ax.set_ylim(*RB_BOX_YLIM)
        xlo = -0.6
        furniture(ax, xlim=(xlo, RB_GRID_X_MAX + 0.6), xloc=1)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(RB_GRID_AXIS_LW)
        ax.tick_params(which="major", width=RB_GRID_AXIS_LW)
        ax.tick_params(labelsize=RB_GRID_TICK_FS)   # after furniture(), which sets the house 14
        # EVERY decade keeps its label. At 16 pt the default log locator decided 10^0..10^5 no
        # longer fit and thinned them to every second one, dropping 10^5 -- the axis top -- with
        # them. numticks high enough that the thinning never triggers on this range.
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=99))
        # An unlabelled tick where the axes meet, matching the y axis's 10^0 in that corner:
        # the first bond count sits 0.6 in, so without it the x axis looks cut short.
        counts = list(range(0, RB_GRID_X_MAX + 1))
        ax.set_xticks([xlo] + counts, [""] + [str(x) for x in counts])
        # "Reference", not the shared REF_LABEL, and the gap to the panel doubled (2026-09-14).
        ax.set_title("Reference" if lab == REF_LABEL else display(lab),
                     fontsize=RB_GRID_TITLE_FS, color=INK, pad=10)
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    # Both outer names now sit on an axes rather than on the figure: only an axes label takes a
    # labelpad, and fig.supxlabel has no way to spend points on its gap. The bottom-middle axes
    # is the block's centre with three full rows, so the name also stops being centred on the
    # figure -- which included the colourbar -- and centres on the PANELS instead.
    axes[nrow - 1, RB_GRID_COLS // 2].set_xlabel(RB_X_LABEL, fontsize=RB_GRID_LABEL_FS,
                                                 color=INK, labelpad=RB_GRID_XPAD)
    # The y name goes on the MIDDLE row's left axes rather than fig.supylabel, because only an
    # axes label takes a labelpad -- supylabel sits flush against the tick labels. With three
    # rows the middle axes' centre is the block's centre.
    # TWO LINES (2026-09-14), broken at the comma the sentence already has: the unit stays
    # with the name it belongs to and the stage qualifier gets its own line, which is how the
    # ECDF and the per-atom panel set their y names.
    # THE PAD IS UNCHANGED ON PURPOSE. labelpad is measured to the NEAREST edge of the label,
    # so the 63 px it buys between the tick numbers and the name survives the second line;
    # what grows is the label block itself (~81 px wide to ~170), and under layout=
    # "constrained" that width comes out of the panels rather than out of the canvas.
    axes[nrow // 2, 0].set_ylabel(f"UFF strain energy (kcal mol⁻¹),\n{qualifier}",
                                  fontsize=RB_GRID_LABEL_FS, color=INK,
                                  labelpad=RB_GRID_YPAD)
    # The shared bar (00_core): ramp, shape, pad and the quarter-turned numbers come from
    # there, the point sizes and the tick step are this figure's. `extend` is computed above
    # off the data, so the arrow appears exactly when a median saturates.
    heat_colorbar(fig, axes, norm, "Median strain energy (kcal mol⁻¹)",
                  label_fs=RB_GRID_CBAR_FS, tick_fs=RB_GRID_CBAR_TICK_FS, extend=extend,
                  ticks=np.arange(RB_GRID_NORM[0], RB_GRID_NORM[1] + RB_GRID_CTICK,
                                  RB_GRID_CTICK))
    save(fig, out, f"strain_box_per_rotbond_grid_{stage}")

    csv_rows = []
    for lab in panels:
        for x, v in boxes[lab]:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            csv_rows.append([lab, x, len(v), round(float(med), 3), round(float(q1), 3),
                             round(float(q3), 3), int(np.sum(v > RB_BOX_YLIM[1]))])
    write_csv(out, f"strain_box_per_rotbond_grid_{stage}",
              ["method", "rotatable_bonds", "n", "strain_median", "strain_q25", "strain_q75",
               "n_above_ylim"], csv_rows)

    sat = [(lab, x, round(float(np.median(v)))) for lab in panels for x, v in boxes[lab]
           if not norm.vmin <= float(np.median(v)) <= norm.vmax]
    print(f"  {stage}: {len(panels)} panels · x = 0-{RB_GRID_X_MAX} rotatable bonds · "
          f"colour linear {norm.vmin:g}-{norm.vmax:g} kcal/mol (extend={extend}) · "
          f"saturated boxes {sat}")
    for lab in panels:
        rows = [r for r in rows_by_lab[lab] if r[field] is not None and r["rb"] is not None]
        beyond = 100 * sum(r["rb"] > RB_GRID_X_MAX for r in rows) / max(len(rows), 1)
        top = sum(int(np.sum(v > RB_BOX_YLIM[1])) for _, v in boxes[lab])
        print(f"    {lab:16s} {len(boxes[lab])} boxes · {len(rows):6d} scored · "
              f">{RB_GRID_X_MAX} bonds {beyond:5.1f}% (not drawn) · {top} fliers above "
              f"{RB_BOX_YLIM[1]:.0e}")


@figure("fig-posecheck-strain-rotbond-grid", folder="fig-posecheck/strain-energy",
        needs=("posecheck_<Method>.json", "results/task2-drugdesign/<M>/samples/meta",
               "metrics.json (posecheck.strain, smiles)"))
def draw_posecheck_strain_rotbond_grid(out):
    """Strain per rotatable bond, one panel per method, boxes coloured by median (VoxBind
    Fig. 12 form)."""
    use_style()
    series, refrows, _ = _rb_load_all()
    labels = _rb_order(series)
    for stage in RB_GRID_STAGES:
        _rb_grid(out, series, refrows, labels, stage)
# ════════════════════════════════════════════════════════════════════════════════
# fig-redock-{ecdf,median}
# ════════════════════════════════════════════════════════════════════════════════
# HOW FAR DOES VINA MOVE A POSE IT IS HANDED? Every molecule of this evaluation exists in
# two poses -- as the model generated it, which is exactly the pose `vina_score` scores,
# and as Vina put it back after re-docking it into the same pocket, which is the pose
# behind `vina_dock` -- and the figures here draw the heavy-atom RMSD between them. It is
# the one thing the Vina table cannot say: an affinity reports how good the pose Vina
# FOUND is, never how far it had to move the molecule to find it, so two models whose
# poses are worth the same score can still differ in whether they placed them there.
#
# IN PLACE, NOT SUPERPOSED. The RMSD is computed where the two conformers already sit --
# both are in the receptor's frame -- with molecular symmetry taken into account, so a
# phenyl that came back flipped through its own two-fold axis is not counted as moved.
# Superposing first (`GetBestRMS`) answers a different question -- is this the same
# CONFORMER -- and on these pairs its median is 0.87 A against 3.01 A in place, i.e. it
# would report that redocking barely moved anything.
#
# READ IT AGAINST THE REFERENCE, NEVER AGAINST 2 A. The crystal ligands are redocked by
# the identical protocol and they are the control: whatever they score is this protocol's
# own reproducibility floor, and on these pocket10 crops that floor is 2.69 A with 42.3%
# within 2 A -- far below what a redocking benchmark reports for a prepared full receptor,
# because the receptor is a 10 A crop, protonation is automated, the box is set from the
# ligand's own extent, and a CrossDocked test "reference" pose is not guaranteed to be the
# cognate crystal pose of the receptor it is scored against. So an arm's "% within 2 A" is
# only meaningful as a distance from the reference's, and every variant draws it.
#
# SIZE IS A CONFOUND HERE TOO, and it is checked rather than assumed: the arms do not
# sample the same molecule sizes (AR and Pocket2Mol put half their output under 16 heavy
# atoms, where CoDE's median molecule has 26). CoDE beats VoxBind in all five size bins
# and is the flattest arm in size by a factor of two and a half, but the BASELINE ordering
# does not survive the split -- Pocket2Mol takes the three largest bins. The per-bin table
# is `redock_rmsd_by_size.csv` beside the recompute script; these figures are pooled, and
# pooled means "how reproducible are the poses each arm actually produces".
#
# THE TAIL IS DRAWN, NOT CUT. The ECDF first stopped at 8 A ("p90 is under 8 A for every
# arm") and that hid the one place CoDE is worse: past ~5 A it holds MORE molecules than
# any baseline -- 8.8% beyond 8 A against 2.7-6.5% -- and a linear ECDF squeezes that into
# the last few percent under 100 where no one can read it. So the figure is two panels over
# the full 0-20 A: the ECDF for the body, and the survival curve (share BEYOND x) on a log
# axis for the tail. What the right panel shows, checked against the export:
#   * much of the tail is size -- beyond 8 A is 0-1% under 16 heavy atoms for every arm and
#     9-26% above 30, and half of CoDE's molecules have 26 or more;
#   * what survives the size match is inherited from VoxBind: CoDE and VoxBind are 15.5%
#     and 14.9% beyond 8 A at 26-30 atoms (TargetDiff 4.9, Pocket2Mol 2.1), and their two
#     survival curves lie on top of each other from 5 A on. CoDE's gain is the 1-5 A body;
#   * the tail molecules of those two move their centroid only ~2 A (2.1 and 2.4) -- they
#     come back REVERSED in place -- where TargetDiff's, AR's and Pocket2Mol's shift 4-5 A.
#     Not every baseline is the other kind: DecompDiff's are 2.7 A, close to VoxBind's.
# Past 10 A every arm is under 2.5% and one step of the curve is one molecule.
#
# The RMSDs themselves are measured by a figure-data builder (not part of this repository),
# which needs RDKit and the staged gen/docked pose pairs (the `moleval` env, ~5 s); this
# file only draws what that script exported.
RD_X_MAX = 20.0                  # past every arm's p99.9; FuncBind's 19.5 A is the maximum
RD_TAIL = (4, 6, 8, 10)          # the thresholds tail.csv reports
RD_REF = "Reference"             # the key it is stored under; REF_LABEL is what is drawn


# THE SHARED IDENTITY CHANNELS, NOT LOCAL ONES. Colour is pale() -- the tint every ECDF
# family in this tree draws its arms in -- and line pattern is dash(), the one METHOD_DASH
# table: the baselines each carry their own pattern, VoxBind and CoDE are solid because
# solid means ours, and the crystal ligands are SOLID grey since 2026-09-14, as every ECDF
# draws them. The first version of this file used pale() and a dashed reference, which
# made it the one figure in the section where a reader's key did not transfer.
RD_BASE_LW = 1.6                 # the baselines' weight in the posecheck ECDF family
# Swatch length in font sizes. legend()'s fixed 1.9 is shorter than one period of
# FuncBind's (9, 3) dash, so its swatch reads as a solid line -- i.e. as one of ours. Same
# reasoning, and the same number, as the posecheck strain key's PLEG_HANDLE_LEN.
RD_HANDLE_LEN = 3.6
# THE "% WITHIN 2 A" ON TOP OF EACH BAR (2026-09-22, up from 12 on request). It is the
# panel's second number and the one a reader is meant to compare across arms, so it should
# not be the smallest type in the figure -- which is what it was next to the 11 pt ticks
# standalone and, far more so, inside posecheck-combined, where publication_style.apply()
# raises the ticks to 17.36 pt at the export boundary and deliberately leaves annotations
# where the family put them.
RD_BAR_LABEL_FS = 15
# WHERE THE REFERENCE NOTE SITS, in axes fractions, with its TOP on this line (2026-09-22,
# lifted from 0.98 on request). There is no top spine, so the note may ride above the plot
# area; what bounds it is the CANVAS. This family exports with savefig.bbox=None, so the
# figure keeps PCSZ_COMPARISON_MARGINS exactly: top=0.94 leaves 0.06 of the figure -- about
# 12 pt at this canvas -- above the axes, and exported labels can be clipped if
# a text crosses the canvas edge. At 1.02 the note's top is ~2.7 pt above the frame, a third
# of the room, so the lift is safe in both the standalone and the combined export.
RD_REF_NOTE_Y = 1.02


def _redock_style(lab, core):
    """(colour, pattern, weight, zorder) for one series, from the shared tables."""
    if lab == RD_REF:
        return REF_COLOR, dash(REF_LABEL), REF_LW + .5, 6
    if lab in core:
        return pale(lab), dash(lab), MODEL_LW, 5
    return pale(lab), dash(lab), RD_BASE_LW, 4


def _redock_legend(fig, arms, core, ncol):
    """legend()'s look -- white face, the axis-pen frame, INK text -- with swatches drawn in
    the series' own colour, pattern AND weight, long enough to show a dash."""
    handles = []
    for lab in [RD_REF] + list(arms):
        col, ls, lw, _ = _redock_style(lab, core)
        handles.append(Line2D([], [], color=col, ls=ls, lw=lw, solid_capstyle="round",
                              label=REF_LABEL if lab == RD_REF else display(lab)))
    leg = fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=True,
                     fontsize=12.5, handlelength=RD_HANDLE_LEN, handletextpad=0.55,
                     labelspacing=0.3, borderpad=0.4, borderaxespad=0.39,
                     columnspacing=1.4, facecolor="white", edgecolor=LEGEND_EDGE,
                     framealpha=1.0)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    leg.set_zorder(7)
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


def _redock_rows():
    """{arm: [{p, i, n, rmsd, cen}, ...]} — every paired molecule, as exported."""
    return json.load(open(legacy("fig-redock", "redock_rmsd_per_molecule.json"),
                          encoding="utf-8"))


def _redock_variants(data):
    """(name, arms) per figure variant, core first — labels, in ARMS order, that the
    export actually carries. `core` is this section's comparison: our method against the
    model it modifies, with the crystal ligand as the benchmark."""
    present = [lab for lab, _, _ in ARMS if lab in data]
    core = [lab for lab, key, _ in ARMS if key in CORE and lab in data]
    return (("core", core), ("all", present))


def _redock_q(v, p):
    """The p-quantile by linear interpolation — the same definition the export uses."""
    v = sorted(v)
    k = (len(v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)


def _redock_stats(rows):
    v = [r["rmsd"] for r in rows]
    return {"n": len(v), "median": st.median(v), "mean": sum(v) / len(v),
            "q25": _redock_q(v, .25), "q75": _redock_q(v, .75),
            "within1": 100 * sum(x < 1 for x in v) / len(v),
            "within2": 100 * sum(x < 2 for x in v) / len(v),
            "beyond4": 100 * sum(x > 4 for x in v) / len(v),
            "centroid": st.median([r["cen"] for r in rows])}


def _redock_ecdf(v, xs):
    v = np.sort(np.asarray(v))
    return 100.0 * np.searchsorted(v, xs, side="right") / len(v)


@figure("fig-redock-ecdf", needs=("fig-redock/redock_rmsd_per_molecule.json",))
def draw_redock_ecdf(out):
    """Generated -> redocked RMSD: the ECDF for the body, the log survival curve for the tail."""
    data = _redock_rows()
    core = _redock_variants(data)[0][1]
    xs = np.linspace(0, RD_X_MAX, 800)
    for variant, arms in _redock_variants(data):
        # THE KEY GOES UNDER THE PANELS, IN A RESERVED BAND. The curves sweep corner to
        # corner in both panels and leave no empty corner -- an opaque box anywhere covers
        # either the 2 Å crossing or the tail. A legend anchored to an AXES was tried and is
        # worse: it is wider than its panel, and fit() then shrinks the panel to make room
        # and slices the y label. So the legend belongs to the FIGURE, tight_layout gets a
        # `rect` that keeps the band clear, and the panels keep their width.
        ncol = 3 if len(arms) <= 3 else 5
        band = 0.30 * math.ceil((len(arms) + 1) / ncol) + 0.12
        use_style()
        fig, (body, tail) = plt.subplots(1, 2, figsize=(FIG_W * 1.45, PANEL_H + band))
        for ax, survival in ((body, False), (tail, True)):
            # The 2 Å mark is a threshold, drawn as the SuCOS figure draws its 0.4 one.
            ax.axvline(2.0, color=INK, lw=1.2, ls=DOT, zorder=3)
            for lab in [RD_REF] + list(arms):
                col, ls, lw, z = _redock_style(lab, core)
                y = _redock_ecdf([r["rmsd"] for r in data[lab]], xs)
                if survival:
                    # 0% cannot sit on a log axis. Where an arm has no molecule left the
                    # curve ENDS rather than dropping to the floor, which would draw a
                    # vertical cliff that reads as data.
                    y = 100.0 - y
                    y = np.where(y > 0, y, np.nan)
                # Ours and the model it modifies draw ON TOP of the baselines, and heavier:
                # the curves cross repeatedly and the two the figure is about must not end
                # up under a baseline at a crossing.
                ax.plot(xs, y, color=col, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
            # "Within" / "Beyond" and not "Molecules within RMSD (%)": the longer label is
            # taller than a panel this height and lost its closing "%)" off the top.
            furniture(ax, xlabel="Generated → redocked RMSD (Å)", xlim=(0, RD_X_MAX), xloc=2,
                      ylabel="Beyond RMSD (%)" if survival else "Within RMSD (%)")
            if survival:
                # Down to 0.1%, which is one molecule of a model arm's ~1,014. The reference
                # has 78, so its curve can never go below 1.3% and stops there.
                ax.set_yscale("log")
                ax.set_ylim(0.1, 100)
                ax.yaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            else:
                ax.set_ylim(0, 100)
        fit(fig, rect=(0, band / (PANEL_H + band), 1, 1))
        _redock_legend(fig, arms, core, ncol)
        save(fig, out, f"redock_ecdf_{variant}")
        print(f"  ecdf {variant}: {len(arms)} arms, "
              f"{sum(len(data[a]) for a in arms)} molecules")

    # The tail in numbers -- what the right panel shows, and the size split that says how
    # much of it is size. `beyond8_centroid` is the median centroid shift of the molecules
    # past 8 Å: ~2 Å means they came back reversed in place, 4-5 Å that they moved.
    rows = []
    for lab in [a for a, _, _ in ARMS if a in data] + [RD_REF]:
        recs = data[lab]
        v = np.array([r["rmsd"] for r in recs])
        far = [r for r in recs if r["rmsd"] > 8]
        per_bin = []
        for i in range(len(BIN_LABELS)):
            sel = [r["rmsd"] for r in recs if bin_of(r["n"]) == i]
            per_bin.append(round(100 * sum(x > 8 for x in sel) / len(sel), 2)
                           if len(sel) >= 10 else "")
        rows.append([lab, len(v)] + [round(100 * float(np.mean(v > t)), 2) for t in RD_TAIL]
                    + [round(float(np.percentile(v, q)), 3) for q in (90, 95, 99)]
                    + [round(float(v.max()), 3),
                       round(st.median([r["cen"] for r in far]), 3) if far else ""]
                    + per_bin)
    write_csv(out, "redock_tail",
              ["arm", "n_molecules"] + [f"pct_beyond_{t}A" for t in RD_TAIL]
              + ["rmsd_p90", "rmsd_p95", "rmsd_p99", "rmsd_max", "beyond8_centroid_median"]
              + [f"pct_beyond_8A_heavy_{b}" for b in BIN_LABELS], rows)


@figure("fig-redock-median", needs=("fig-redock/redock_rmsd_per_molecule.json",))
def draw_redock_median(out):
    """Median redocking displacement per arm, against the crystal ligands' own."""
    data = _redock_rows()
    stats = {arm: _redock_stats(rows) for arm, rows in data.items()}
    ref = stats[RD_REF]
    # The bar is the median and the label on it is the share within 2 Å, because those
    # are the two numbers a reader wants and the second is not recoverable from the first:
    # two arms with the same median differ by ten points in it.
    for variant, arms in _redock_variants(data):
        arms = _pcsz_ordered(arms)  # Same method order as the clash panel and standalone key.
        use_style()
        fig, ax = plt.subplots(figsize=PCSZ_STRAIN_SIZE, layout="none")
        xsi = np.arange(len(arms))
        ax.axhline(ref["median"], color=REF_COLOR, lw=REF_LW + .5, ls=dash(REF_LABEL),
                   zorder=4)
        ax.bar(xsi, [stats[a]["median"] for a in arms], width=.68,
               color=[pale(a) for a in arms], zorder=3)
        top = max(stats[a]["median"] for a in arms)
        for x, a in zip(xsi, arms):
            ax.text(x, stats[a]["median"] + top * .02, f"{stats[a]['within2']:.0f}%",
                    ha="center", va="bottom", fontsize=RD_BAR_LABEL_FS, color=INK)
        # Keep the reference note; percentage values remain on the bars without a caption.
        wide = len(arms) > 3
        ax.text(0.99 if wide else 0.01, RD_REF_NOTE_Y,
                f"— {REF_LABEL.lower()}: {ref['median']:.2f} Å", transform=ax.transAxes,
                fontsize=13, color="#666666" if wide else REF_COLOR,
                fontweight="bold" if variant == "all" else "normal",
                ha="right" if wide else "left", va="top")
        ax.set_xticks([])  # Method names are carried by the standalone legend.
        furniture(ax, ylabel="Median\nre-docking RMSD", xloc=None)
        # The bars use zorder=3; keep both visible axis lines above their edges.
        for side in ("left", "bottom"):
            ax.spines[side].set_zorder(6)
        ax.set_ylim(0, 5)
        ax.set_yticks([0, 1.5, 3.0, 4.5])
        ax.grid(False, axis="x")
        _pcsz_comparison_layout(fig, ax, yunit="Å")
        with plt.rc_context({"savefig.bbox": None, "savefig.dpi": 170}):
            save(fig, out, f"redock_median_{variant}")
        print(f"  median {variant}: " + ", ".join(f"{a} {stats[a]['median']:.2f}"
                                                  for a in arms))
    write_csv(out, "redock_summary",
              ["arm", "n_molecules", "rmsd_median", "rmsd_mean", "rmsd_q25", "rmsd_q75",
               "pct_within_1A", "pct_within_2A", "pct_beyond_4A", "centroid_median"],
              [[a, s["n"], round(s["median"], 4), round(s["mean"], 4), round(s["q25"], 4),
                round(s["q75"], 4), round(s["within1"], 2), round(s["within2"], 2),
                round(s["beyond4"], 2), round(s["centroid"], 4)]
               for a, s in ([(lab, stats[lab]) for lab, _, _ in ARMS if lab in stats]
                            + [(RD_REF, ref)])])

"""fig-posebusters: the 9 figures drawn into figures/fig-posebusters/.

    python figures/draw.py posebusters

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-posebusters-valid-per-atom               -> fig-posebusters/
    fig-posebusters-check-failures               -> fig-posebusters/
    fig-posebusters-upset                        -> fig-posebusters/
    fig-posebusters-group-bars                   -> fig-posebusters/
    fig-posebusters-valid-heatmap                -> fig-posebusters/
    fig-posebusters-failure-budget               -> fig-posebusters/
    fig-posebusters-all-checks                   -> fig-posebusters/
    fig-posebusters-sucos-ecdf                   -> fig-posebusters/
    fig-posebusters-sucos-per-atom               -> fig-posebusters/
"""
from __future__ import annotations

import collections
import json
import math
import statistics as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from draw import (
    ARMS, AXIS, AXIS_LW, AXIS_WIDTH, BACKGROUND_ALPHA, BIN_LABELS, DASH, DIST_FILL,
    DIST_LW, FIG_W, GRID, HEAT_CBAR_SHRINK, HEAT_CMAP, HEIGHT_RATIOS, H_PAD, INK,
    MIN_N, MODEL_LW, P79, PANEL_H, REF_COLOR, REF_LABEL, REF_LW, STACK_H, TICK_SIZE,
    X_LABEL, arm_handles, arms_for, bin_of, by_size, display, figure, fit, furniture,
    heat_colorbar, highlight, legacy, legend, model_curve, pale, pose_data,
    reference_curve, rolled, save, share, use_style, variants, write_csv, x_range,
)

def size_distribution(ax, xs, per_arm, ref_per, arms, paint=None):
    """Where each set puts its molecules, as a share of its own -- which is what makes the
    panel above it trustworthy, and is itself a finding, since the arms differ in the sizes
    they generate as much as in per-size pose quality. It counts the SAME molecules the
    panel above plots. `paint` as in arm_handles."""
    for lab, key, _ in arms:
        pct, _ = share(per_arm[key], xs)
        col = (paint or pale)(lab)
        ax.step(xs, pct, where="mid", color=col, lw=DIST_LW, zorder=3)
        ax.fill_between(xs, pct, step="mid", color=col, alpha=DIST_FILL, lw=0, zorder=2)
    raw, _ = share(ref_per, xs)
    ax.plot(xs, rolled(raw, xs), color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
    ax.set_ylim(bottom=0)


# ════════════════════════════════════════════════════════════════════════════════
# fig-posebusters-{valid-per-atom,valid-heatmap,group-bars,check-failures,upset,
#                  all-checks,sucos-ecdf,sucos-per-atom}
# ════════════════════════════════════════════════════════════════════════════════
# SIZE IS THE CONFOUND, SO SIZE IS THE X AXIS. Every PoseBusters check gets harder as the
# molecule grows -- more rings to pucker, more angles to strain, more atoms to reach the
# protein -- and the arms draw different size mixes (24.9 heavy atoms for CoDE against
# TargetDiff's 22.2). A pooled validity rate is therefore partly a report of the size mix,
# the same trap the Vina numbers have, which is why the headline figure is per-atom with
# the size distribution drawn underneath it rather than a single pooled bar.
#
# SuCOS is the one check PoseBusters' `gen` config adds over `dock`: shape overlap x
# pharmacophore-feature overlap against the pocket's crystal ligand. It is NOT folded into
# `valid` -- every other column asks "is this pose physically possible", SuCOS asks "does it
# sit where the crystal ligand sits", and a de novo model is not trying to reproduce the
# crystal ligand. Its per-molecule values are recomputed by
# a figure-data builder (not part of this repository), which needs RDKit and the posebusters
# package (the `moleval` env); this file only draws what that script exported.
#
# THE FOUR CHECKED-IN sucos_*.png WERE RASTERIZED IN THAT `moleval` ENV, and it carries
# FreeType 2.14.3 against `voxbind`'s 2.6.1. Same matplotlib, same data, same code -- but a
# different glyph rasterizer, and tight_layout measures the axes off the text it lays out,
# so a redraw here is sub-pixel-shifted against them over the whole figure. Drawing them
# from `voxbind` is correct and identical to what build_sucos.py itself produces there; it
# just cannot be byte-compared against PNGs made by the other env's FreeType.

# Validity is a proportion, so its per-count curve is smoothed over +-PB_VALID_WIN atoms --
# see model_curve for why the strain and clash curves are not.
PB_VALID_WIN = 2
# A check no method fails above this often is a row of white space in the breakdown: it
# says only that PoseBusters ran it. The full counts stay in the JSON the recompute script
# writes. The filter stays a RATE even though the bars are counts -- it asks "is this row
# informative", and the arms hold different numbers of molecules.
PB_MIN_FAIL_PCT = 0.5
# The check-failure breakdown, split into stacked panels by how often a check fails, each on
# a linear axis scaled to itself (2026-09-14, on request; it was one symlog axis). The median
# rate over the eight arms puts them in three decades:
#   ~10%   bond angles 9.7, min distance 7.0, steric clash 5.7, ring non-flatness 5.4,
#          bond lengths 3.2
#   ~1%    internal energy 1.6, volume overlap 0.36
#   <=0.1% double bond flatness 0.26, aromatic ring flatness 0.02 -- zero for most sets
# Fixed rather than recomputed, so `core` and `all` split the same way; inside a panel the
# checks keep the figure's order (worst first). A check shown but in no panel is an error.
PB_FAIL_PANELS = (
    ("bond_lengths", "bond_angles", "non-aromatic_ring_non-flatness",
     "minimum_distance_to_protein", "internal_steric_clash"),
    ("volume_overlap_with_protein", "internal_energy"),
    ("aromatic_ring_flatness", "double_bond_flatness"),
)
# THE HEAT MAP'S GROUPS. The 20 scored dock-mode checks partitioned into what each one is
# actually asking about, so `valid` can be read as "passes all five" instead of as one
# number. EVERY CHECK IS IN EXACTLY ONE GROUP and the partition is asserted against the
# data: a PoseBusters release that adds a check breaks the build rather than quietly
# dropping it out of the picture, and no check can be double-counted into two groups.
#
# Six of the nine in `Protein clash & overlap` are VACUOUS HERE and pass by construction:
# the receptor every arm is scored against is the pocket10 crop, which carries no HETATM
# and no HOH, so the cofactor and water checks have nothing to measure. They stay in the
# group because the group is a partition of `valid` and dropping them would make the
# columns stop multiplying out to it -- not because they carry information.
#
# `Planarity`, `Ring pucker` and `Self-clash & energy` are not among the four groups this was
# first asked for; they exist because the partition has to be complete, and they are cut this
# way BY WHAT THE CHECK WANTS rather than by where PoseBusters lists it (2026-09-15, on
# request). `aromatic_ring_flatness` and `double_bond_flatness` both fail when something that
# should be FLAT is not, so they are one column; `non-aromatic_ring_non-flatness` is the exact
# opposite test -- `check_nonflat: True` flips the comparison, and it fails when a saturated
# 6-ring comes out PLANAR -- so it stands alone. It is the single worst check for the voxel
# arms (21-23% fail), and it was previously buried in a four-check column called `Strain.`,
# where its 78.9 read as 74.5 and collided in name with the PoseCheck strain-energy figures,
# which measure something else entirely. What is left, `internal_steric_clash` and
# `internal_energy`, is the ligand against ITSELF rather than against the protein -- hence
# `Self-clash & energy`, beside `Protein clash & overlap`.
PB_GROUPS = (
    ("Bond geometry", ("bond_lengths", "bond_angles")),
    ("Planarity", ("aromatic_ring_flatness", "double_bond_flatness")),
    ("Ring pucker", ("non-aromatic_ring_non-flatness",)),
    ("Valence & connectivity", ("sanitization", "all_atoms_connected", "inchi_convertible",
                                "no_radicals")),
    ("Protein clash & overlap", ("minimum_distance_to_protein", "volume_overlap_with_protein",
                                 "protein-ligand_maximum_distance",
                                 "minimum_distance_to_organic_cofactors",
                                 "minimum_distance_to_inorganic_cofactors",
                                 "minimum_distance_to_waters",
                                 "volume_overlap_with_organic_cofactors",
                                 "volume_overlap_with_inorganic_cofactors",
                                 "volume_overlap_with_waters")),
    ("Self-clash & energy", ("internal_steric_clash", "internal_energy")),
)
PB_HEAT_COL0 = "PoseBusters valid"
# What a molecule PoseBusters could not LOAD passes, out of 22, in the counts column. One:
# `mol_cond_loaded` is true of it -- the receptor loaded -- and nothing else about it was ever
# established, `mol_pred_loaded` being false by definition. See the count itself for why those
# molecules are in the population at all.
PB_HEAT_UNLOADED_PASSES = 1
# THE RAMP IS THE DATA'S OWN RANGE: it ends at 100, a rate's ceiling, and starts at the
# WORST CELL DRAWN (2026-09-14, on request) -- FuncBind's 50.1% valid. A 0-100 ramp would
# spend half its range on ground no method stands on and paint the whole map one shade,
# which is the failure mode of a heat map whose numbers all sit near the top.
#
# Both variants take the range of the ALL-ARMS map, so a cell is the same colour in `core`
# as in `all`. Scaling `core` to its own worst cell (CoDE's 67.5) would make our two arms
# look as far apart as the eight are.
PB_HEAT_VMAX = 100.0
# THE COLOURBAR AND THE RAMP ARE THE HOUSE ONES (2026-09-14/15, on request): heat_colorbar()
# and HEAT_CMAP from 00_core, the same pair the per-rotbond strain grid draws. Both used to
# live in the rotbond family and be read across from here at call time; they moved up to the
# shared section so neither figure owns the other's colours. One hue means the ramp needs no
# reversal between the two -- darker is simply more of whatever the bar is named after, more
# strain there and a higher pass rate here.
# AND THE RAMP IS ACTUALLY IT AGAIN (2026-09-23, on request: "pb valid heatmap, pb budget, pb
# check 이것도 다 고쳐줘"). Between 2026-09-15 and then this figure kept a LOCAL white-to-#8291E8
# ramp while the comment above said it used the house one -- the colourbar was the house
# helper, the colours were not. That is now true as written: PB_HEAT_CMAP is gone and the cells
# take HEAT_CMAP, so this map, pb-check, the budget, the group bars, the strain grid and the
# uncertainty matrices are all one ramp, and the light end is the shared #E8F2F7 rather than
# pure white.
# Luminance below which a cell's number is set in white instead of ink, so the text follows
# the background (2026-09-15, on request) rather than being one colour throughout. It reads the
# FILL, never the value, so it is the ramp that decides where the flip lands -- and the ramp
# changed, so the number did (0.55 -> 0.59, 2026-09-23). MEASURED, not chosen: walking the
# shared ramp and comparing WCAG contrast against white and against INK, the two swap places at
# #8495DE, where white is 2.86:1 and ink 2.83:1; below it white wins (5.69:1 against ink's 1.43
# at the deep end), above it ink does (7.13:1 against white's 1.14 at the light end). 0.59 is
# that colour in the same flat 0.2126/0.7152/0.0722 measure the call site computes.
PB_HEAT_WHITE_TEXT_BELOW = 0.59
# Point sizes are NOT copied from the rotbond grid, which is three rows of nine panels on a
# 14.45 in canvas: same bar, this figure's scale. 13/11.5 x 1.3 with everything else.
PB_HEAT_CBAR_FS, PB_HEAT_CBAR_TICK_FS = 20.5, 18
# THE CELL SHAPE, 3 WIDE TO 2 TALL (2026-09-15, on request: square, then "2:3 정도"). It was
# neither before -- at a fixed 8.83 in width a cell came out 1.07 x 0.92 on `core` and
# 1.07 x 0.49 on `all`, so the same number sat in a near-square box in one variant and a
# 2.2:1 letterbox in the other. Now the figure is built FROM the cell, so both variants draw
# the same shape and `all` is simply the taller figure.
# THE WIDTH IS THE HEADING'S (2026-09-15, on request), not a round number, and what sets it
# is the widest ADJACENT PAIR rather than the widest heading: two neighbours each spend half
# their width towards the gutter between them, so "Bond len." (1.116 in at 17 pt) beside
# "Bond ang." (1.201) needs 1.159 plus a gutter, where "Bond ang." alone would only ask for
# 1.20. 1.29 leaves that worst pair 0.13 in of white and every other pair more -- the
# smallest column that still holds all eight headings on ONE line, which is what the short
# forms were for. EVERY MEASUREMENT HERE MOVED WITH THE TYPE when the map's text went up 30%
# (2026-09-15, on request): the column was 0.99 at 13 pt headings, and the gutter scaled with
# it, 0.10 -> 0.13, so the map keeps the same proportions at the larger size rather than
# getting tighter. The height follows from the ratio; "100.0" at 18 pt is 0.715 in, so a
# number still keeps a clear margin in its cell.
PB_HEAT_CELL_W = 1.29
PB_HEAT_CELL_ASPECT = 2 / 3      # cell height / cell width -- what set_aspect() takes
PB_HEAT_CELL_H = PB_HEAT_CELL_W * PB_HEAT_CELL_ASPECT
# The numbers in the cells. 12.5 -> 14 when the cell came in (the number is what the figure is
# read for, so it went the other way), then 14 -> 18 -> 21.5 with the 30% and 20% raises.
PB_HEAT_NUM_FS = 21.5
# The row names, which are not a tick label size any more: they match the headings, and both
# are 13 x 1.3 x 1.2. Held in a constant rather than written at the call because the two have to
# move together -- a map whose columns are named larger than its rows reads as two figures.
PB_HEAT_LABEL_FS = 20.5
# A tick per row and per column (2026-09-15, on request), where the block had none: the
# spines are hidden and the cells are separated by white, so a long row had nothing tying its
# name to it. They sit at the CELL CENTRES, with the labels, not on the cell edges. 4.5 x 1.3
# with the type, so a tick stays the same fraction of the name it points at.
PB_HEAT_TICK_LEN = 7.1
# What the axes does NOT cover, in inches: the row names to its left plus the colourbar and
# its name to the right, and the column headings above it. They size the canvas; set_aspect()
# is what actually holds the cell shape, so an allowance being off costs a margin, never the
# shape. Both moved with the type: 2.50 -> 3.75 over two raises, and the heading strip
# 0.35 -> 0.75 while the headings were stacked, back to 0.55 now they are one line again.
# 3.75 -> 2.60 (2026-09-23, on request: less white on the left). The allowance is what the
# canvas keeps for the row names and the colourbar TOGETHER, and it was set by the fear below
# rather than by a measurement. Measured: the names take 1.85 in and the longest one's left
# edge sits at 0.10 in -- the layout pad -- at every allowance from 3.75 down to 2.10, because
# constrained layout places the labels from the left and spends the surplus on the RIGHT,
# between the block and the bar. 2.60 keeps half an inch of that slack and takes 1.15 in off
# the canvas. Reading the label bboxes back off the drawn figure is what settled it, exactly
# as the note below says it must be.
#
# THE OLD RATIONALE, KEPT because the failure it describes is real: constrained layout
# under-reserves next to a FIXED-ASPECT axes, and the symptom is the longest row name,
# VoxBind(sigma=0.9), starting OFF the left edge of the canvas and losing its V. It has done that twice -- at 2.16 with
# 13 pt names, and again at 2.50 once the map went to eight columns. The spare width cannot
# change the cells: the block's width is what binds, so it becomes margin on the left, which
# is where the names are. Checked by reading the tick labels' own bboxes back off the drawn
# canvas rather than by eye: a clipped label is the one failure that survives a glance at the
# PNG, because the eye reads "VoxBind" from the letters that are left.
PB_HEAT_SIDE, PB_HEAT_HEAD = 2.60, 0.55
# A little air under the bottom row (2026-09-15). The WIDTH is what binds under a fixed
# aspect -- the side allowance above is deliberately generous -- so this does not change the
# cells at all: it lengthens the canvas, and constrained layout centres the block in what is
# left, which puts half of it under the last row. 0.24 buys about 26 px at 220 dpi. The top
# margin looks bigger in the numbers only because the column headings are reserved inside it;
# the white above the headings is the same 12 px the block has below it.
PB_HEAT_FOOT = 0.24
# The blank strip after the first column and the first row, in cell widths (2026-09-15, on
# request: the ink rule alone did not read as a division). A rule marks a boundary; a gap
# makes the block on either side of it a separate object, which is what the first column and
# the first row are.
PB_HEAT_GAP = 0.22
PB_HEAT_WRAP = 12                # characters per line of a column heading
# The headings sit at the row names' size, and the column width above is solved FOR this
# number -- raise one and the other has to move. 13 x 1.3.
PB_HEAT_HEAD_FS = PB_HEAT_LABEL_FS
# Every heading now fits on one line at this width, which is the point of the short forms;
# the wrap is the backstop that keeps a longer one from running into its neighbour instead of
# silently overlapping it.
# What a column is CALLED on the heat map (2026-09-15, on request). The keys are the group
# names, which stay in full everywhere a number is read back -- the CSV headers, the bar
# charts' axis labels, the README table. A heading here only has to say which column you are
# in, and at full length six of them took three lines each and half the figure's height.
PB_HEAT_SHORT = {
    # was `Valid.` (2026-09-23, on request); two lines, since one ran into `Passed (22)`
    "PoseBusters valid": "All\npassed",
    "Bond geometry": "Bond.",
    # The split pair. `Bond` IS DROPPED RATHER THAN THE HEADING STACKED (2026-09-15, on
    # request: one line each). At 20.5 pt "Bond len." and "Bond ang." are 1.35 and 1.45 in,
    # and side by side they ask for a 1.58 in column -- the cell would have to grow right back
    # to the size the type raise was meant to shrink it against. Two lines fixed that and were
    # asked to go; what is left is to drop the word both share. These are the only two columns
    # on the map that could be a length or an angle OF anything else, they sit next to each
    # other, and both are named in full in the CSV and in the bar chart. A newline in a short
    # form would still be taken literally; everything without one goes through PB_HEAT_WRAP.
    "Bond lengths": "Length.",
    "Bond angles": "Angle.",
    # The two ring columns are the pair a reader can mix up, and they are kept apart by the
    # word that distinguishes them: `Planarity.` is what must be flat, `Pucker.` what must
    # not, and they sit side by side so the opposition is visible. `Ring pucker.` in full is
    # 1.118 in -- the widest heading on the map by a quarter inch, and next to `Planarity.`
    # it alone would have pushed every column from 0.99 to 1.08. Named in full in the CSV and
    # in the bar chart that draws the same group.
    "Planarity": "Planarity.",
    "Ring pucker": "Pucker.",
    "Valence & connectivity": "Valence.",
    "Protein clash & overlap": "Clash.",
    # 1.451 in, wider than the 1.29 cell -- deliberately. What has to clear is the GAP to the
    # heading beside it, and `Clash.` is 0.871, so the pair leaves 0.129 in of white, the same
    # gutter every other pair gets. It overhangs its own column by 0.08 in each side, which is
    # empty on the right (the colourbar's own margin) and is the narrow `Clash.` on the left.
    "Self-clash & energy": "Self-clash.",
}
# The groups that get a bar chart of their own (2026-09-15, on request), one file each. The
# others are left out because a bar chart of them says nothing a reader cannot already see:
# `Valence & connectivity` is 100.0 for every set, `Planarity` is 99.5+ for everything except
# AR, and `Self-clash & energy` moves over eight points across the whole field. `Ring pucker`
# takes the slot the four-check `Ring pucker & internal strain` used to hold, and is the same
# bar with the three passengers removed -- it was always the check that moved it. Order is
# the heat map's, so a reader moving between them meets the groups in the same order.
PB_GROUP_BARS = ("Bond geometry", "Protein clash & overlap", "Ring pucker")
PB_GROUP_BAR_H = 0.42            # inches per method row
# The axis runs the full 0-100 even though nothing is below 68. A bar's LENGTH is its value,
# so it has to start at a true zero -- floored at 60 the same numbers would read as a
# fourfold spread. At this range the differences are legible anyway: 70.4 against 100 is a
# third of the axis.
PB_GROUP_BAR_XLIM = (0, 109)
PB_SUCOS_THRESHOLD = 0.4         # gen.yml's own value
PB_BIN_COLS = ["n_molecules", "atoms_mean", "n_posebusters", "pb_valid_rate",
               "pb_valid_rate_size_standardized"]


def _posebusters_rate(vals):
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def _posebusters_std_weights(data, p79_rows):
    """The common size distribution for direct standardization -- every arm pooled.

    The weights come from the arms that are actually IN the comparison, not from every key
    the loader returned: a scoring run in progress leaves partial rows in p79_rows, and
    folding those into the standard population moves every arm's standardized rate."""
    return collections.Counter(
        r["n"] for _, key, _ in arms_for("v", data) for r in p79_rows[key]
        if r["v"] is not None)


def _posebusters_std_rate(rows, weights):
    """Each arm's rate WITHIN every 1-heavy-atom stratum, re-weighted by one common size
    distribution, so what is left is validity at matched size. Strata where the arm holds
    <10 molecules are dropped, not extrapolated. This is the number to compare arms with;
    the crude rate is what the arm actually produced, and the two answer different
    questions."""
    per = by_size(rows, "v")
    num = den = 0.0
    for n, w in weights.items():
        if len(per.get(n, ())) >= 10:
            num += w * float(np.mean(per[n])); den += w
    return 100 * num / den if den else float("nan")


def _posebusters_block(rows, weights):
    v = [r["v"] for r in rows if r["v"] is not None]
    std = _posebusters_std_rate(rows, weights)
    return {
        "n_molecules": len(rows),
        "atoms_mean": round(float(np.mean([r["n"] for r in rows])), 2) if rows else None,
        "n_posebusters": len(v),
        "pb_valid_rate": round(_posebusters_rate(v) / 100, 4) if v else None,
        "pb_valid_rate_size_standardized": round(std / 100, 4) if np.isfinite(std) else None,
    }


# ── figure 1: validity per heavy-atom count, over each arm's size distribution ───
def _posebusters_valid_panel(out, arms, variant, p79_rows, refrows):
    per = {key: by_size(p79_rows[key], "v") for _, key, _ in arms}
    ref_per = by_size(refrows, "v")
    xs = x_range(per, arms)
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    rate100 = lambda v: 100 * float(np.mean(v))
    top.plot(xs, reference_curve(ref_per, xs, rate100), color=REF_COLOR,
             lw=REF_LW, ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        top.plot(xs, model_curve(per[key], xs, rate100, win=PB_VALID_WIN),
                 color=pale(lab), lw=MODEL_LW, zorder=5, solid_capstyle="round")
    furniture(top, ylabel="PoseBusters valid (%)", xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    top.set_ylim(0, 102)
    legend(top, arm_handles(arms, paint=pale), loc="lower left", fontsize=11.5)

    size_distribution(bot, xs, per, ref_per, arms, paint=pale)
    furniture(bot, ylabel="% of ligands", xlabel=X_LABEL,
              xlim=(xs[0] - 0.6, xs[-1] + 0.6))
    fig.align_ylabels((top, bot))
    fit(fig, pad=0.5, h_pad=H_PAD)
    save(fig, out, f"pb_valid_per_atom_{variant}")
    return xs


# ── figure 2: which checks fail ─────────────────────────────────────────────────
def _posebusters_check_failures(data, p79_rows, refrows):
    """{key: {n_mols, counts, rates}} over every arm, plus the crystal ligands."""
    out = {}
    for lab, key, _ in arms_for("v", data):
        rows = [r for r in p79_rows[key] if r["v"] is not None]
        cnt = collections.Counter(c for r in rows for c in r["f"])
        out[key] = {"n_mols": len(rows), "counts": dict(cnt),
                    "rates": {k: 100 * v / len(rows) for k, v in cnt.items()}}
    rrows = [r for r in refrows if r["v"] is not None]
    cnt = collections.Counter(c for r in rrows for c in r["f"])
    out["reference"] = {"n_mols": len(rrows), "counts": dict(cnt),
                        "rates": {k: 100 * v / max(1, len(rrows)) for k, v in cnt.items()}}
    return out


def _posebusters_wrap_check(name):
    """The PoseBusters check name, wrapped instead of abbreviated.

    `non-aromatic_ring_non-flatness` on one line is 30 characters and was eating 2.9 in of
    a 7.6 in figure -- nearly half the width -- as a tick label. Abbreviating it is the
    wrong fix: this check passes when a non-aromatic ring is sufficiently NON-flat (it is
    `check_nonflat: True` in dock.yml, threshold 0.1 A), so failing it means a saturated
    ring came out planar, and every shortening of that name I tried either flipped its
    sense or read as the aromatic check next to it. Wrapping is free and exact.

    textwrap is not one of draw.py's shared imports and this is the only figure that needs
    it, so it is fetched here rather than at the top of the file. Its exact wrapping --
    including breaking on the hyphens -- is what the tick labels are, so this must stay
    textwrap and not a hand-rolled splitter."""
    return "\n".join(__import__("textwrap").wrap(name.replace("_", " "), 18))


def _posebusters_failures_panel(out, arms, variant, fails):
    """Rows are checks, bars are the SHARE of each set's molecules that fail them.

    Rates, not counts, and that is what lets the crystal ligands be an ordinary bar here:
    on a count axis 79 of them against ~7,900 generated molecules put their worst row at 2
    molecules, invisible beside a bar of 1,822, and they had to be drawn as a rate-matched
    marker instead. The price is that a rate hides its denominator -- the reference's 2.5%
    IS those 2 molecules, and it carries about +-1.8 points of binomial noise against the
    arms' +-0.2 -- so its bar alone keeps its n in the key. Counts for every arm and every
    check stay in posebusters_check_failures.json.
    """
    # The reference is one more series, first in every group and first in the key.
    series = [(REF_LABEL, "reference")] + [(lab, key) for lab, key, _ in arms]
    shown = [fails[key] for _, key in series]
    names = sorted({k for g in shown for k in g["counts"]
                    if max(h["rates"].get(k, 0) for h in shown) >= PB_MIN_FAIL_PCT},
                   key=lambda k: -max(g["rates"].get(k, 0) for g in shown))
    # One row of the y axis is 1.0 apart, so the bars of a group must fit inside that:
    # a fixed height works for three series and silently overlaps the neighbouring groups
    # at nine (9 x 0.19 = 1.71), which reads as bars detached from their labels. Derive it.
    h = 0.86 / len(series)
    # THE KEY GOES OUTSIDE THE AXES, centred along the bottom, 3 x 3. There is no empty
    # corner inside:
    # the long bars fill the top and the right, and the in-axes box this used to carry
    # reached far enough left to bury the bottom rows' bars -- `double bond flatness`
    # looked empty while AR, DecompDiff and Pocket2Mol were failing it 3.1, 1.8 and 1.4% of
    # the time. Outside it covers nothing. Three columns only fit because the bars are
    # rates now: an arm's own n no longer has to be in the key for its bar to be readable.
    ncol = 3
    leg_rows = -(-len(series) // ncol)
    # Row pitch: enough that nine thin bars stay readable, without turning a 7.6 in wide
    # figure into a 10 in tall one. Plus the strip the key needs at the bottom.
    # STACKED PANELS, LINEAR X (2026-09-14, on request; it was one symlog axis). The rates run
    # 0.01% to 23%, and on one linear axis everything under ~2% was a stub against FuncBind's
    # 23%; symlog fixed that but made every bar's length a log reading. Three panels -- see
    # PB_FAIL_PANELS -- each give a decade of checks its own linear scale, so a length is a
    # rate again, and a zero is still a bar of nothing. Panel height = its check count, so
    # the bars are the same thickness in every panel; the x scales differ, so each keeps
    # its own tick labels.
    panels = [[k for k in names if k in p] for p in PB_FAIL_PANELS]
    leftover = [k for k in names if not any(k in p for p in PB_FAIL_PANELS)]
    if leftover:
        raise ValueError(f"PoseBusters checks in no PB_FAIL_PANELS panel: {leftover}")
    panels = [p for p in panels if p]
    fig = plt.figure(figsize=(FIG_W, (0.40 if len(arms) <= 3 else 0.62) * len(names)
                              + 1.6 + 0.45 * (len(panels) - 1) + 0.30 * leg_rows + 0.2),
                     dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(len(panels), 1, height_ratios=[len(p) for p in panels])
    for j, panel in enumerate(panels):
        ax = fig.add_subplot(gs[j, 0])
        ax.set_facecolor("white")
        ys = np.arange(len(panel))[::-1]
        for i, (lab, key) in enumerate(series):
            # top of the group downwards, so the key's order IS the order of the bars
            ax.barh(ys + ((len(series) - 1) / 2 - i) * h,
                    [fails[key]["rates"].get(k, 0.0) for k in panel],
                    height=h, color=pale(lab), edgecolor=pale(lab), lw=0.8, zorder=3)
        top = max(g["rates"].get(k, 0.0) for g in shown for k in panel)
        furniture(ax, ylabel=None, xlim=(0, top * 1.08), xloc=None,
                  xlabel="Molecules failing the check (%)" if j == len(panels) - 1 else None)
        ax.grid(False, axis="y")
        for sp in ("left", "bottom"):
            ax.spines[sp].set_zorder(5)      # bars start at 0 and would paint over the axis
        ax.set_yticks(ys)
        ax.set_yticklabels([_posebusters_wrap_check(k) for k in panel], fontsize=12)
        ax.set_ylim(-0.6, len(panel) - 0.4)
    # pale() tints, the eight-method figures' colours (2026-09-14, on request)
    handles = [Patch(facecolor=pale(lab), edgecolor=pale(lab), label=display(lab))
               for lab, key in series]
    fit(fig, pad=0.5)
    # tight_layout does not see a FIGURE legend, so it has just laid the axes out over the
    # strip the key occupies. Place the key, measure what it actually took, and RAISE the
    # axes by that much -- reserving the measured height keeps the tick labels and the x
    # label, which sit BELOW the axes box and would otherwise end up behind an opaque
    # legend; reserving up to its top edge instead would bury them. Columns are dropped
    # if the key overruns the figure, which is cheaper than trusting a width estimate.
    fs = 12.5 if len(arms) <= 3 else 11
    for c in range(ncol, 0, -1):
        leg = legend(fig, handles, loc="lower center", ncol=c, fontsize=fs,
                     bbox_to_anchor=(0.5, 0.008))
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        if (bb.x0 >= 0.003 and bb.x1 <= 0.997) or c == 1:
            break
        leg.remove()
    fig.subplots_adjust(bottom=min(0.6, fig.subplotpars.bottom
                                  + (bb.y1 - bb.y0) + 0.016))
    save(fig, out, f"pb_check_failures_{variant}")


@figure("fig-posebusters-valid-per-atom", needs=("metrics.json (posebusters block)",))
def draw_posebusters_valid_per_atom(out):
    """PoseBusters dock-mode validity against ligand size, over each arm's size mix."""
    data, p79_rows, refrows = pose_data()
    use_style()
    weights = _posebusters_std_weights(data, p79_rows)
    ranges = {}
    for variant, arms in variants("v", data):
        ranges[variant] = _posebusters_valid_panel(out, arms, variant, p79_rows, refrows)

    scored = arms_for("v", data)
    pooled = {lab: _posebusters_block(p79_rows[key], weights) for lab, key, _ in scored}
    ref_block = _posebusters_block(refrows, weights)
    by_bin = {lab: [_posebusters_block([r for r in p79_rows[key] if bin_of(r["n"]) == b],
                                       weights)
                    for b in range(len(BIN_LABELS))] for lab, key, _ in scored}
    by_bin[REF_LABEL] = [_posebusters_block([r for r in refrows if bin_of(r["n"]) == b],
                                            weights)
                         for b in range(len(BIN_LABELS))]

    print(f"  79-pocket set · {len(P79)} pockets · "
          + " · ".join(f"{v} x {xs[0]}-{xs[-1]}" for v, xs in ranges.items())
          + f" (counts where every drawn arm has >={MIN_N} molecules)\n")
    print(f"  {'arm':16s} {'atoms':>6s} {'PB-valid':>9s} {'size-std':>9s} {'mols':>7s}")
    for lab, key, _ in scored:
        r = pooled[lab]
        print(f"  {lab:16s} {r['atoms_mean']:6.1f} {100 * r['pb_valid_rate']:8.1f}% "
              f"{100 * r['pb_valid_rate_size_standardized']:8.1f}% {r['n_molecules']:7d}")
    print(f"  {REF_LABEL:16s} {ref_block['atoms_mean']:6.1f} "
          f"{100 * ref_block['pb_valid_rate']:8.1f}% {'—':>9s} "
          f"{ref_block['n_molecules']:7d}")
    print(f"\n  PB-valid by bin ({' · '.join(BIN_LABELS)}):")
    for lab, key, _ in scored:
        print(f"    {lab:16s} " + "  ".join(
            f"{100 * b['pb_valid_rate']:5.1f}%" if b["pb_valid_rate"] is not None else "    -"
            for b in by_bin[lab]))

    write_csv(out, "posebusters_by_atom_range", ["arm", "bin"] + PB_BIN_COLS,
              [[arm, lab] + [r[c] for c in PB_BIN_COLS]
               for arm, rowset in by_bin.items()
               for lab, r in zip(BIN_LABELS, rowset)])


@figure("fig-posebusters-check-failures", needs=("metrics.json (posebusters block)",))
def draw_posebusters_check_failures(out):
    """Which PoseBusters checks fail, per method, as a share of its own molecules."""
    data, p79_rows, refrows = pose_data()
    use_style()
    fails = _posebusters_check_failures(data, p79_rows, refrows)
    for variant, arms in variants("v", data):
        _posebusters_failures_panel(out, arms, variant, fails)


# ── figure 3: which checks fail TOGETHER ────────────────────────────────────────
# THE CO-FAILURE VIEW OF THE SAME DATA. The breakdown above is a set of MARGINALS: it says
# FuncBind fails bond lengths 8.1% of the time and ring non-flatness 14.1%, and it cannot
# say whether those are the same molecules. The distinction is what `valid` is made of --
# it is all-must-pass, so the validity rate is set by how many molecules carry AT LEAST ONE
# failure, not by how many failures there are. Two arms with identical marginals differ by
# eight points of validity depending on whether their failures stack up on the same
# molecules or spread over different ones.
#
# So a bar here is a molecule's EXACT failed set: `Bond angles + Bond lengths` counts the
# molecules that fail those two AND NOTHING ELSE. The columns are therefore disjoint -- they
# partition each arm, with the empty set (the PoseBusters-valid molecules) left out of the
# picture and reported in the printout and the CSV instead. That is the UpSet reading of
# these sets, and it is the only one available: a Venn diagram of nine sets is not drawable,
# and a stacked bar of overlapping sets double-counts every molecule that fails twice.
#
# WHAT IT SHOWS, ONCE DRAWN: failure is usually a SINGLE check, and how usually is itself the
# thing that separates the arms -- 82% of VoxBind's failing molecules fail exactly one check
# and 79% of CoDE's, against 53% for DiffSBDD and 61% for FuncBind, whose molecules go wrong
# in several ways at once (a fifth of everything they draw fails two checks or more, where
# every other arm is at 4-10%). The tall column is a different check for nearly every method
# -- ring non-flatness for the two voxel arms and FuncBind, bond angles for TargetDiff,
# minimum distance to protein for Pocket2Mol, aromatic ring flatness for AR -- and the
# co-failures that do exist are the ones that share a cause: bond angles with bond lengths
# (one distorted fragment breaks both), minimum distance with volume overlap (one clash
# measured two ways).
PB_UPSET_MAX = 16                # columns drawn, worst first
# ... and nothing rarer than this in EVERY arm. At the cut the `all` map's last column is
# 1.1% and `core`'s is 0.25%: the two variants draw the same NUMBER of sets, from very
# different tails, because eight arms put eight methods' worth of failure modes into one
# ranking while two put two.
PB_UPSET_MIN_PCT = 0.25
# THE CANVAS IS BUILT FROM THE COLUMN, as the heat map is built from its cell. A column has
# to hold one bar per series side by side, so nine arms make a WIDER figure rather than
# thinner bars -- the failure mode of a fixed-width grouped bar chart is that the ninth
# series arrives and every bar becomes a hairline.
PB_UPSET_BAR = 0.072             # inches of one series' bar
PB_UPSET_GUTTER = 0.18           # inches of white between two columns
# A matrix row, in inches. What binds is the ROW NAME, not the dot: the export boundary sets
# every tick label to 17.36 pt, the long check names wrap to two lines (see
# _posebusters_wrap_check), and two lines of that is 0.58 in. Below ~0.66 the names of
# adjacent rows touch, and publication_style's repair pass answers that by growing the whole
# canvas 12% at a time -- so the pitch is set here, once, with the clearance already in it.
PB_UPSET_ROW = 0.72
PB_UPSET_TOP_H = 3.2             # inches of intersection-bar panel
PB_UPSET_MARGIN = 2.1            # inches of marginal (per-check) bar panel
# Slack for the row names, which belong to no panel's width. They used to sit BETWEEN the two
# bar panels; since the marginal moved right (see the panel) this is the figure's left margin,
# and it holds the row names under the set-size panel's own y label -- the wider of the two,
# not their sum, because they are in different rows of the grid.
PB_UPSET_SIDE = 2.6
PB_UPSET_FOOT = 1.45             # the x label under the matrix, plus the canvas margins
PB_UPSET_DOT = 132               # scatter area, pt^2 -- a dot about 0.16 in across
# THE BLANK BETWEEN ONE BLOCK OF SET SIZES AND THE NEXT (2026-09-23). It is what the blocks'
# separate y axes are written in: at the export boundary a number is 17.36 pt, so `2.5` asks
# for about 0.32 in and the axis, its ticks and a little air ask for the rest. The gap is added
# to the canvas rather than taken out of the columns, so a dot keeps its spacing.
PB_UPSET_BLOCK_GAP = 0.62        # inches


def _posebusters_upset_sets(arms, p79_rows, refrows, fails):
    """(the sets drawn, the checks they are made of, {key: {set: %}}, {key: n}).

    A set is the molecule's WHOLE failed list, taken from r["f"] as it is -- never
    intersected with the checks that end up as rows. Truncating it would move a molecule
    that also fails something rare into a column it does not belong to, and the columns
    would stop partitioning the arm; a set containing a rare check simply falls below the
    cut and is not drawn, which is what the coverage line in the printout is for."""
    rows = {key: [r for r in p79_rows[key] if r["v"] is not None] for _, key, _ in arms}
    rows["reference"] = [r for r in refrows if r["v"] is not None]
    rate = {key: {s: 100 * n / len(rs)
                  for s, n in collections.Counter(tuple(r["f"]) for r in rs).items()}
            for key, rs in rows.items() if rs}
    # RANKED BY THE WORST ARM, and the crystal ligands are not in that ranking: there are 79
    # of them, so one molecule is 1.3% and a set that two of them happen to share would
    # outrank a set a 7,900-molecule arm makes 1% of the time. They are still DRAWN -- the
    # benchmark for "how often do real poses fail together" is worth a bar even when the bar
    # is two molecules, and the noise on it is ±1.8 points against the arms' ±0.2.
    keys = [key for _, key, _ in arms]
    worst = lambda s: max(rate[key].get(s, 0.0) for key in keys)
    sets = sorted({s for key in keys for s in rate[key] if s}, key=lambda s: (-worst(s), s))
    sets = [s for s in sets if worst(s) >= PB_UPSET_MIN_PCT][:PB_UPSET_MAX]
    # DRAWN IN BLOCKS OF SET SIZE (2026-09-23, on request: "한개만, 두개, 세개 이렇게 실패한
    # 케이스들끼리 가로 순서 배치"): every single-check set, then every pair, then every triple,
    # and inside a block still the worst arm's share, descending. WHICH sets are drawn is not
    # affected -- that is still the sixteen most frequent, chosen above -- only the order they
    # are drawn in. What the blocks buy is the reading the matrix is for: how much of an arm's
    # failure is ONE thing going wrong, and how much is several at once, is now a distance
    # along the axis rather than a count of dots per column.
    sets = sorted(sets, key=lambda s: (len(s), -worst(s), s))
    # THE ROWS ARE THE ALL-CHECKS TABLE'S ROWS (2026-09-23, on request: "upset의 행들 종류가
    # pb-check-all이랑 일치하지 않아, pb check all 기준으로 만들어줘"). They used to be the checks
    # the DRAWN SETS are made of, ranked by their marginal -- eight rows in a different order
    # from the table's twenty-two, so the same benchmark had two row orders and two row lists,
    # and `double bond flatness`, `inchi convertible` and `protein-ligand maximum distance`
    # were rows of the table and not of this figure at all. Now a row is a row of the table:
    # PB_TAB_GROUPS' own order, which is also the heat map's column order, so a check is in the
    # same place in all three.
    #
    # WHAT IS STILL LEFT OUT, and why it is not a departure from the table: the two `Not
    # scored` rows, because `valid` is the AND over the 20 GRADED checks and a molecule's
    # failed set can never contain a loading column -- there is no column here for them to be
    # a dot in; and the checks NOBODY fails, which the table can afford to draw at 100.0 (six
    # of them grey, as things that cannot fail on a pocket10 crop) and this figure cannot,
    # because an UpSet row of fourteen grey dots beside an empty marginal bar says a check
    # exists, which is the table's job, and costs 0.72 in of a figure that is already 12 in
    # tall. A row with no BLACK dot is kept, though: `inchi convertible` fails 0.1% of
    # DecompDiff and every set it is in is rarer than the cut, so the row is empty in the
    # matrix and its marginal bar is the whole of what it has to say -- which is exactly the
    # reading the marginal panel is for.
    order = [c for _, cs in PB_TAB_GROUPS for c in cs if c not in PB_TAB_LOADING]
    drawn = keys + ["reference"]
    checks = [c for c in order if max(fails[k]["rates"].get(c, 0.0) for k in drawn) > 0]
    used = {c for s in sets for c in s}
    if not used <= set(checks):
        raise KeyError(f"failure set holds a check the table has no row for: "
                       f"{sorted(used - set(checks))} -- PB_TAB_GROUPS is out of date")
    return sets, checks, rate, {key: len(rs) for key, rs in rows.items()}


def _posebusters_upset_panel(out, arms, variant, sets, checks, rate, fails):
    """The three panels of an UpSet: the sets' sizes, the matrix that names them, and the
    per-check marginal the sets are cut out of."""
    series = [(REF_LABEL, "reference")] + [(lab, key) for lab, key, _ in arms]
    n = len(series)
    colw = PB_UPSET_BAR * n + PB_UPSET_GUTTER
    # ONE BLOCK PER SET SIZE: the columns are already ordered by it, so a block is a run of
    # equal len(). The blocks are what carry their own y scale below.
    blocks = []
    for i, s in enumerate(sets):
        if blocks and len(sets[blocks[-1][0]]) == len(s):
            blocks[-1][1] = i + 1
        else:
            blocks.append([i, i + 1])
    counts = [stop - start for start, stop in blocks]
    nb = len(blocks)
    # The gaps are EXTRA canvas, not taken out of the columns: a column is colw wide whether
    # there are two blocks or five, so the dots keep the spacing every other panel is sized
    # against and only the figure gets wider.
    mat_w = colw * len(sets) + PB_UPSET_BLOCK_GAP * (nb - 1)
    mat_h = PB_UPSET_ROW * len(checks)
    fig = plt.figure(figsize=(PB_UPSET_SIDE + PB_UPSET_MARGIN + mat_w,
                              PB_UPSET_TOP_H + mat_h + PB_UPSET_FOOT), dpi=220)
    fig.patch.set_facecolor("white")
    # THE MARGINAL PANEL IS ON THE RIGHT (2026-09-23, on request: "왼쪽에 있는 distribution
    # 오른쪽으로 옮기고, 오른쪽 위에 overall failure 아래 그려주고"). It sat to the LEFT of the
    # matrix, which is where UpSet's own examples put it and where it reads as a mirror of the
    # set-size panel above. Two things paid for that: the row names had to sit BETWEEN the two
    # panels, so every one of them was a measurement tight_layout could get wrong, and the
    # figure opened on a bar chart the reader has no rows for yet -- the names are 2.6 in to
    # its right. Moved, the left edge of the figure is the row names, the matrix begins where
    # the eye does, and the marginal is where the set sizes are: both bar panels down the
    # right-hand side, one over the other, each answering "how much" about the thing the
    # matrix names.
    gs = fig.add_gridspec(2, 2, width_ratios=[mat_w, PB_UPSET_MARGIN],
                          height_ratios=[PB_UPSET_TOP_H, mat_h])
    # NOT sharex/sharey: a shared axis shares its LOCATOR too, and the matrix has to carry
    # ticks the bar panel must not (the row names) and drop ticks the marginal panel keeps.
    # The panels are tied by setting the same limits, which is what sharing was for.
    # ONE AXES PER BLOCK IN EACH ROW, in subgridspecs that share the outer cell and the same
    # width ratios, so a block's bars and its dots are over the same paper before anything is
    # measured -- and are pinned to it again after tight_layout, which moves them apart.
    gs_bar = gs[0, 0].subgridspec(1, nb, width_ratios=counts, wspace=0.0)
    gs_mat = gs[1, 0].subgridspec(1, nb, width_ratios=counts, wspace=0.0)
    bars = [fig.add_subplot(gs_bar[0, j]) for j in range(nb)]
    mats = [fig.add_subplot(gs_mat[0, j]) for j in range(nb)]
    ax_bar, ax_mat = bars[0], mats[0]
    # gs[0, 1] -- over the marginal, beside the set sizes -- is left EMPTY: it is where the
    # key used to be and the key is gone (see below).
    ax_set = fig.add_subplot(gs[1, 1])
    for ax in bars + mats + [ax_set]:
        ax.set_facecolor("white")

    # ── the sets' sizes, one bar per method in each column, ONE SCALE PER BLOCK ──
    # EACH BLOCK IS READ AGAINST ITSELF (2026-09-23, on request: "y축 range도 각각 다르게").
    # On one scale the whole figure is set by the single-check block, whose tallest column is
    # 19.2%; the pairs top out at 5.0 and the triples at 1.9, so two thirds of the columns were
    # drawn in the bottom tenth of the panel and the reader could not rank them at all. A
    # shared scale is the right default when blocks are comparable, and these are not: the
    # question inside a block is which set, and the question across blocks is answered by the
    # blocks themselves. Every block starts at 0 and carries its own numbered axis, so a bar's
    # height is never read across a rule by mistake.
    xs = np.arange(len(sets), dtype=float)
    h = PB_UPSET_BAR / colw          # a column is 1.0 in data units, so a bar is its share
    for j, (start, stop) in enumerate(blocks):
        part = sets[start:stop]
        for i, (lab, key) in enumerate(series):
            bars[j].bar(xs[start:stop] + (i - (n - 1) / 2) * h,
                        [rate[key].get(s, 0.0) for s in part],
                        width=h, color=pale(lab), edgecolor=pale(lab), lw=0.6, zorder=3)
        top = max(rate[key].get(s, 0.0) for _, key in series for s in part)
        # Two lines: at the export boundary's 19.36 pt this name is 4.4 in on one, which is
        # longer than the panel is tall, and a rotated y label that overruns its own axes is
        # simply drawn off the canvas. It is written once, on the leftmost block.
        furniture(bars[j], ylabel="Molecules failing\nexactly this set (%)" if not j else None,
                  xloc=None)
        bars[j].set_xlim(start - 0.5, stop - 0.5)
        bars[j].set_ylim(0, top * 1.08)
        bars[j].grid(False, axis="x")
        bars[j].set_xticks(xs[start:stop])       # a tick per column, aligned with the matrix
        bars[j].tick_params(labelbottom=False)
        for sp in ("left", "bottom"):
            bars[j].spines[sp].set_zorder(5)     # bars start at 0 and paint over the axis

    # ── the matrix: which checks each column is ──
    ys = np.arange(len(checks))[::-1]           # worst check at the top, as in the breakdown
    for j, (start, stop) in enumerate(blocks):
        ax = mats[j]
        for r, y in enumerate(ys):
            if r % 2:
                ax.axhspan(y - 0.5, y + 0.5, color=GRID, alpha=BACKGROUND_ALPHA, zorder=0)
        for i in range(start, stop):
            members = [y for c, y in zip(checks, ys) if c in sets[i]]
            # The rule joins the dots of ONE set, which is what makes a column read as a set
            # rather than as a stack of unrelated marks.
            if len(members) > 1:
                ax.plot([i, i], [min(members), max(members)], color=INK, lw=2.0, zorder=3,
                        solid_capstyle="round")
            ax.scatter([i] * len(checks), ys, s=PB_UPSET_DOT, zorder=4,
                       color=[INK if c in sets[i] else GRID for c in checks])
        ax.set_xlim(start - 0.5, stop - 0.5)
        ax.set_ylim(-0.5, len(checks) - 0.5)
        ax.set_xticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if j:
            # NO BOUNDARY RULE (2026-09-23, on request: "세로선을 없애도 될 것 같아"). There was
            # one -- first an axvline through both panels, then this block's own left spine --
            # and it was worth drawing while the blocks shared a scale and touched. They no
            # longer do: the blank between them and the numbered axis each one carries say
            # where a block ends, and a rule on top of that is a third statement of it.
            ax.set_yticks([])
    ax_mat.set_yticks(ys)
    # The check names EXACTLY as the breakdown writes them, wrapped rather than abbreviated
    # -- see _posebusters_wrap_check for why `non-aromatic ring non-flatness` cannot be
    # shortened without flipping its sense.
    # AT THE EXPORT SIZE, NOT AT furniture()'s. These labels sit BETWEEN two panels rather
    # than outside the figure, so they are the one measurement tight_layout has to get right:
    # laid out at 14 pt and then raised to TICK_SIZE at the export boundary, the longest of
    # them ends up written across the marginal panel's bars. fit()'s overrun correction
    # cannot see that -- it only gives back canvas the drawing ran off the edge of.
    # SPELLED AS THE TABLE SPELLS THEM, wrapped: `_posebusters_check_name` is the all-checks
    # table's own column, so `Non-aromatic ring non-flatness` opens with a capital here too and
    # the two figures name a row the same way. Only the wrap is this figure's -- the table sets
    # a row on one line and has 3.5 in of column to do it in, where this one has the canvas
    # between the edge and the matrix.
    ax_mat.set_yticklabels([_posebusters_wrap_check(_posebusters_check_name(c))
                            for c in checks], fontsize=TICK_SIZE)
    ax_mat.tick_params(length=0, pad=6, colors=INK)
    ax_mat.set_xlabel("Exact set of PoseBusters checks a molecule fails", labelpad=9)

    # ── the marginal: how often each check fails at all ──
    # fails[] IS THE BREAKDOWN FIGURE'S OWN TABLE, passed in rather than recounted, so the
    # row this figure draws and the bar that figure draws cannot disagree. It counts every
    # molecule failing the check, including the ones whose set is too rare to be a column
    # here -- which is exactly what a marginal is, and why it is not the row's columns added
    # up.
    hb = 0.86 / n
    wide = max(fails[key]["rates"].get(c, 0.0) for _, key in series for c in checks)
    for i, (lab, key) in enumerate(series):
        ax_set.barh(ys + ((n - 1) / 2 - i) * hb,
                    [fails[key]["rates"].get(c, 0.0) for c in checks],
                    height=hb, color=pale(lab), edgecolor=pale(lab), lw=0.6, zorder=3)
    furniture(ax_set, xlabel="Molecules failing\nthe check (%)", xloc=None)
    # Ten points apart, spelled out: this panel is 2 in wide and the export boundary sets its
    # numbers at 17.36 pt, where the automatic locator drops to two labels and the reader is
    # left reading bar lengths against a single rule.
    ax_set.xaxis.set_major_locator(MultipleLocator(10))
    # GROWING AWAY FROM THE MATRIX, now that the panel is on its right. It was mirrored
    # (wide -> 0) so that a bar started at the matrix and ran left, which is what an UpSet on
    # the left-hand side has to do; on this side the same trick would hang the bars off the
    # canvas edge and put the numbers hard against the matrix.
    ax_set.set_xlim(0, wide * 1.08)
    ax_set.set_ylim(-0.5, len(checks) - 0.5)
    ax_set.grid(False, axis="y")
    ax_set.set_yticks(ys)
    # The rows are named once, on the matrix; a tick here would be a dash pointing at nothing.
    ax_set.set_yticklabels([])
    ax_set.tick_params(axis="y", length=0)
    for sp in ("left", "bottom"):                # the baseline is the matrix side again
        ax_set.spines[sp].set_zorder(5)

    # NO KEY (2026-09-23, on request: "legend은 없애줘"). Nine series and no key is a real
    # cost, and it is paid once for the deck rather than in every figure: the colours are the
    # house method palette, every other PoseBusters figure carries them labelled -- the
    # breakdown this figure's marginal panel is taken from is the one next to it -- and the
    # bars are in the SAME ORDER in every group here, reference first. What it buys is the
    # corner: the key was a 2.6 x 3.2 in box in the one piece of free canvas an UpSet has, and
    # with the marginal panel moved right that canvas is the figure's opening corner.
    fit(fig, pad=0.5, h_pad=H_PAD)
    # A DOT MUST SIT UNDER ITS BAR. tight_layout places every subplot from its OWN
    # decorations, and the matrix carries the row names while the bar panel carries a y axis,
    # so the two came out of it a sixteenth of an inch apart -- enough, over sixteen columns,
    # that the eye stops trusting which column it is reading. They are one object, so the bar
    # panel is pinned to the matrix's x span (that way round: moving the MATRIX would slide
    # the row names into the marginal panel, whose gap tight_layout has just measured).
    # The blocks are laid out again from the span tight_layout settled on, in inches: a column
    # is the same width in every block and the gaps are what they were asked to be, neither of
    # which survives being left to two subgridspecs' idea of `wspace`.
    span0, span1 = mats[0].get_position().x0, mats[-1].get_position().x1
    gap = PB_UPSET_BLOCK_GAP / fig.get_figwidth()
    drawable = (span1 - span0) - gap * (nb - 1)
    x = span0
    for j, c in enumerate(counts):
        w = drawable * c / len(sets)
        mb, bb = mats[j].get_position(), bars[j].get_position()
        mats[j].set_position([x, mb.y0, w, mb.height])
        bars[j].set_position([x, bb.y0, w, bb.height])
        x += w + gap
    # The x label belongs to the whole matrix, not to the block that carries it. Centred on
    # the span by hand, and held the labelpad's own 9 pt under the axes -- set_label_coords
    # takes axes fractions, so the drop is converted through this block's height in inches.
    p0 = mats[0].get_position()
    mid = (mats[0].get_position().x0 + mats[-1].get_position().x1) / 2
    ax_mat.xaxis.set_label_coords((mid - p0.x0) / p0.width,
                                  -(9 / 72) / (p0.height * fig.get_figheight()))
    save(fig, out, f"pb_upset_{variant}")


@figure("fig-posebusters-upset", needs=("metrics.json (posebusters block)",))
def draw_posebusters_upset(out):
    """Which PoseBusters checks fail TOGETHER — each molecule's exact failed set."""
    data, p79_rows, refrows = pose_data()
    use_style()
    fails = _posebusters_check_failures(data, p79_rows, refrows)
    out_rows = []
    for variant, arms in variants("v", data):
        sets, checks, rate, ns = _posebusters_upset_sets(arms, p79_rows, refrows, fails)
        _posebusters_upset_panel(out, arms, variant, sets, checks, rate, fails)
        series = [(REF_LABEL, "reference")] + [(lab, key) for lab, key, _ in arms]
        seen = len({s for _, key in series for s in rate[key] if s})
        print(f"\n  {variant}: {len(sets)} of {seen} failure sets drawn "
              f"(≥{PB_UPSET_MIN_PCT}% in some arm), over {len(checks)} checks")
        print(f"  {'arm':16s} {'valid':>7s} {'1 check':>8s} {'2+ checks':>10s} "
              f"{'drawn':>7s} {'mols':>7s}")
        for lab, key in series:
            r = rate[key]
            one = sum(v for s, v in r.items() if len(s) == 1)
            many = sum(v for s, v in r.items() if len(s) > 1)
            drawn = sum(r.get(s, 0.0) for s in sets)
            # `drawn` as a share of the arm's FAILING molecules, which is what the columns
            # are a sample of -- against its molecules it would only restate the valid rate.
            print(f"  {lab:16s} {r.get((), 0.0):6.1f}% {one:7.1f}% {many:9.1f}% "
                  f"{100 * drawn / max(1e-9, one + many):6.0f}% {ns[key]:7,d}")
            out_rows.append([variant, lab, ns[key], 0, "(passes every check)",
                             round(r.get((), 0.0), 3)])
            out_rows += [[variant, lab, ns[key], len(s), " + ".join(s),
                          round(r.get(s, 0.0), 3)] for s in sets]
    write_csv(out, "posebusters_upset_sets",
              ["variant", "method", "n_molecules", "n_checks_failed", "checks", "pct"],
              out_rows)


# ── validity heat map ───────────────────────────────────────────────────────────
def _posebusters_group_rates(rows, checked, columns=()):
    """(% passing each group, % valid, n scored) for one set of molecules.

    A GROUP IS PASSED WHEN THE MOLECULE FAILS NOTHING IN IT, which is per-molecule and not
    recoverable from the per-check rates: two checks each failing 5% of molecules are one
    column at 90% if they fail different molecules and at 95% if they fail the same ones.
    That is also why the group columns do not multiply out to the `valid` column.

    `columns` adds further (name, checks) pairs to rate the same way -- the heat map passes
    its split of Bond geometry, which is a narrower cut of the same partition and not a group
    in its own right."""
    scored = [r for r in rows if r["v"] is not None]
    n = len(scored)
    if not n:
        return {}, float("nan"), 0
    seen = {c for r in scored for c in r["f"]}
    if not seen <= checked:
        raise KeyError(f"PoseBusters check outside PB_GROUPS: {sorted(seen - checked)} -- "
                       "add it to a group, or the heat map stops being a partition of `valid`")
    out = {}
    for name, checks in tuple(PB_GROUPS) + tuple(columns or ()):
        want = set(checks)
        out[name] = 100 * sum(not (want & set(r["f"])) for r in scored) / n
    return out, 100 * sum(bool(r["v"]) for r in scored) / n, n


# BOND GEOMETRY IS TWO COLUMNS ON THE MAP (2026-09-15, on request), one per check, because
# the two come apart: TargetDiff passes 98.9% on lengths and 77.5% on angles, Pocket2Mol is
# the other way round (89.6 / 98.0), and the single column averaged that away. PB_GROUPS is
# left alone -- it is the partition `valid` is graded on and what the bar charts and the CSV's
# `group` field mean -- so this is a cut WITHIN one group, listed here and nowhere else.
PB_HEAT_SPLIT = {"Bond geometry": (("Bond lengths", ("bond_lengths",)),
                                   ("Bond angles", ("bond_angles",)))}
PB_HEAT_COLS = tuple(c for name, checks in PB_GROUPS
                     for c in PB_HEAT_SPLIT.get(name, ((name, checks),)))


def _posebusters_counted_checks():
    """(the 20 checks `valid` is graded on, the 22 PoseBusters reports).

    The assert is the heat map's: PB_GROUPS is a PARTITION of `valid`, so a PoseBusters
    release that adds or moves a check breaks the build rather than quietly dropping it out
    of the picture."""
    checked = {c for _, checks in PB_GROUPS for c in checks}
    assert len(checked) == sum(len(cs) for _, cs in PB_GROUPS) == 20
    counted = checked | set(PB_TAB_LOADING)
    assert len(counted) == 22
    return checked, counted


def _posebusters_mean_passed(series, loading):
    """({label: mean checks passed of 22}, {label: molecules that did not load}, 22).

    `series` is [(label, key, rows)]. TWO FIGURES DRAW THIS NUMBER -- the heat map's counts
    column and the summary dots -- so it is computed in one place: two figures that disagree
    about the same quantity are worse than one figure that is wrong.

    THE POPULATION IS THE SCORED MOLECULES PLUS THE ONES THAT COULD NOT BE LOADED
    (2026-09-23, on request), the latter at PB_HEAT_UNLOADED_PASSES each. They are absent
    from metrics.json -- that is WHY they could not be loaded -- so they carry no checks to
    count; what is known about them is that the receptor loaded and the molecule did not, and
    the other twenty were never evaluated, which the count treats as not passed exactly as
    `valid`'s all-must-pass rule does. It is 4 molecules for VoxBind and 8 for CoDE, and it
    moves them 21.62 -> 21.61 and 21.58 -> 21.56; every other arm has none."""
    _, counted = _posebusters_counted_checks()
    passed = {lab: [len(counted) - len(set(r["f"])) for r in rows if r["v"] is not None]
              for lab, _, rows in series}
    unloaded = {lab: loading[key]["not_loaded"] for lab, key, _ in series}
    means = {lab: (sum(v) + PB_HEAT_UNLOADED_PASSES * unloaded[lab]) / (len(v) + unloaded[lab])
             for lab, v in passed.items()}
    return means, unloaded, len(counted)


def _posebusters_heat_norm(grid):
    """One ramp for every PoseBusters figure: floored at the worst cell of the ALL-arms map.

    Both map variants and all three bar charts take it, so a value is the same colour
    wherever it is drawn -- which is the whole reason the bars gave up the method palette."""
    return matplotlib.colors.Normalize(float(grid.min()), PB_HEAT_VMAX)


def _posebusters_heatmap_grid(arms, p79_rows, refrows):
    """(column names, row labels, the rates, n per row) -- the map, before it is drawn."""
    series = [(REF_LABEL, refrows)] + [(lab, p79_rows[key]) for lab, key, _ in arms]
    checked = {c for _, checks in PB_GROUPS for c in checks}
    cols = [PB_HEAT_COL0] + [name for name, _ in PB_HEAT_COLS]
    grid, ns, bars = [], [], {name: [] for name, _ in PB_GROUPS}
    for lab, rows in series:
        groups, valid, n = _posebusters_group_rates(rows, checked, PB_HEAT_COLS)
        grid.append([valid] + [groups[name] for name, _ in PB_HEAT_COLS])
        ns.append(n)
        # The GROUP rates travel alongside the map's columns: Bond geometry is two cells here
        # but stays one bar, and a group's rate is per-molecule, so it cannot be recovered
        # from the two halves after the fact.
        for name, _ in PB_GROUPS:
            bars[name].append(groups[name])
    return (cols, [lab for lab, _ in series], np.array(grid, float), ns,
            {k: np.array(v, float) for k, v in bars.items()})


def _posebusters_heatmap_panel(out, variant, built, norm, means, n_tests):
    cols, labels, grid, ns, bars = built
    series = list(zip(labels, ns))
    # Counts have a neutral background; the shared colour scale remains percentages. The
    # heading carries the DENOMINATOR because it changed once (2026-09-23, 20 -> 22) and a
    # mean of 21.4 means nothing without it, so it is written from the count rather than
    # spelled out beside it.
    cols = [cols[0], f"Passed ({n_tests})"] + cols[1:]
    grid = np.insert(grid, 1, np.nan, axis=1)

    # The canvas is the CELL BLOCK plus its margins, so both variants draw the same square
    # cell and `all` is simply the taller figure. The block is measured in cell widths, the
    # spacer strip included, exactly as the mesh edges below are built.
    # CONSTRAINED LAYOUT, as the rotbond grid uses -- tight_layout does not see a colourbar's
    # label, and fit()'s overrun correction moves the main axes, not the bar's, so the name
    # came out sliced off the right edge of the core map.
    nrow, ncol = grid.shape
    gap = PB_HEAT_GAP
    fig, ax = plt.subplots(
        figsize=(PB_HEAT_CELL_W * (ncol + gap) + PB_HEAT_SIDE,
                 PB_HEAT_CELL_H * (nrow + gap) + PB_HEAT_HEAD + PB_HEAT_FOOT),
        dpi=220, layout="constrained")
    # Constrained layout's own padding, 3 pt by default, which is what the canvas edge leaves
    # round the whole figure. At 3 pt the longest row name ended 9 px off the edge -- drawn in
    # full, but reading as if it had been cut -- and NO amount of extra canvas fixes it: the
    # layout reserves exactly the label's width plus this pad, so the slack lands somewhere
    # else. 0.10 in is the same 3 pt grown with the type.
    fig.get_layout_engine().set(w_pad=0.10, h_pad=0.10)
    fig.patch.set_facecolor("white")
    cmap = HEAT_CMAP
    # A MESH, NOT AN IMAGE, so the gap can be a fraction of a cell. imshow lays cells on one
    # uniform grid, where the only available spacer is a whole empty column; pcolormesh takes
    # the edge coordinates, so PB_HEAT_GAP is inserted into them and the blank strip is
    # exactly as wide as it should be. The spacer cells are NaN and masked, which leaves the
    # figure's own white showing through rather than a painted rectangle.
    xe = np.array([0.0, 1.0, 2.0] + [2.0 + gap + k for k in range(ncol - 1)])
    ye = np.array([0.0, 1.0] + [1.0 + gap + k for k in range(nrow)])
    xc = [0.5, 1.5] + [2.5 + gap + k for k in range(ncol - 2)]
    yc = [0.5] + [1.5 + gap + k for k in range(nrow - 1)]
    mesh = np.insert(np.insert(grid, 1, np.nan, axis=0), 2, np.nan, axis=1)
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(mesh), cmap=cmap, norm=norm,
                  edgecolors="white", linewidth=1.4)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[-1], ye[0])              # first row at the top, as imshow had it
    # A cell is 1 x 1 in DATA units, so the axes aspect is the cell's shape -- and it holds
    # even if the canvas arithmetic above is ever off, which a hand-set figure height would not.
    ax.set_aspect(PB_HEAT_CELL_ASPECT)
    for i in range(nrow):
        for j in range(ncol):
            if j == 1:
                ax.add_patch(plt.Rectangle((1, yc[i] - 0.5), 1, 1,
                                           facecolor="#F2F2F2", edgecolor="white", linewidth=1.4))
                ax.text(xc[j], yc[i], f"{means[labels[i]]:.2f}", ha="center", va="center",
                        fontsize=PB_HEAT_NUM_FS, color=INK, zorder=3)
                continue
            v = grid[i, j]
            # Ink or white off the FILL's own luminance, never off the value: which cells
            # are dark is a property of the ramp, and the ramp is shared with another figure.
            r, g, b, _ = cmap(norm(v))
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < PB_HEAT_WHITE_TEXT_BELOW
            ax.text(xc[j], yc[i], f"{v:.1f}", ha="center", va="center",
                    fontsize=PB_HEAT_NUM_FS, color="white" if dark else INK, zorder=3)
    # A rule after the two summary columns and after the reference row. Both
    # mark a change of KIND, not of degree: column 0 is the aggregate rather than a sixth
    # category -- and the groups do not multiply out to it, see _posebusters_group_rates --
    # while row 0 is the crystal ligands, the benchmark the arms are read against. The rule
    # runs down the middle of the gap, so the two together read as one division.
    ax.axvline(2.0 + gap / 2, color=INK, lw=2.0, zorder=5)
    ax.axhline(1.0 + gap / 2, color=INK, lw=2.0, zorder=5)
    ax.set_xticks(xc)
    # Wrapped at PB_HEAT_WRAP, not at the 18 the check-failure tick labels use: a column here
    # is ~1.2 in wide, and at 18 the group names ran into each other across the header.
    # A short form that already carries a newline is used as written -- textwrap would treat
    # it as whitespace and put the heading back on one line, which is the opposite of the
    # point. The wrap stays as the backstop for everything else.
    heads = [PB_HEAT_SHORT.get(c, c) for c in cols]
    # `Passed (22)` goes to two lines beside `All passed` (2026-09-23): side by side on one
    # line each they ran together across the column gap.
    heads = [h.replace("Passed (", "Passed\n(") for h in heads]
    ax.set_xticklabels([h if "\n" in h else "\n".join(__import__("textwrap").wrap(h, PB_HEAT_WRAP))
                        for h in heads], fontsize=PB_HEAT_HEAD_FS)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(yc)
    # The method name alone (2026-09-15, on request). Every cell is a RATE, so a reader does
    # not need the denominator to compare two of them, and the counts crowded nine already
    # long labels. They are still in posebusters-valid-heatmap.csv, one column per row.
    # "Reference", not the shared REF_LABEL, exactly as the rotbond grid titles its own first
    # panel: with the units gone from the row there is nothing left for "ligand" to
    # disambiguate, and it is the longest label in the column.
    ax.set_yticklabels(["Reference" if lab == REF_LABEL else display(lab)
                        for lab, _ in series], fontsize=PB_HEAT_LABEL_FS)
    ax.tick_params(length=PB_HEAT_TICK_LEN, width=AXIS_LW, colors=INK, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    # The bar's name wraps when the bar is shorter than the name. One line measures 2.13 in
    # at 13 pt and the bar is HEAT_CBAR_SHRINK of the block, so `core`'s three rows give it
    # 1.84 in and `all`'s nine give 5.6: the short map takes two lines ("% of molecules" is
    # 1.38 in), the tall one stays on one. Scaling the type down instead would have put the
    # bar's name below its own tick numbers.
    bar_in = HEAT_CBAR_SHRINK * PB_HEAT_CELL_H * (nrow + gap)
    label = ("% of molecules passing" if bar_in >= 2.13 * PB_HEAT_CBAR_FS / 13
             else "% of molecules\npassing")
    heat_colorbar(fig, ax, norm, label,
                  label_fs=PB_HEAT_CBAR_FS, tick_fs=PB_HEAT_CBAR_TICK_FS, cmap=cmap,
                  ticks=[50, 60, 70, 80, 90, 100])
    save(fig, out, f"pb_valid_heatmap_{variant}")


def _posebusters_group_bar(out, variant, built, name, norm):
    """One group, one bar per method, as the share of its molecules that pass the group.

    PASS, not fail, so a bar reads the same way round as the heat map cell it comes from --
    both come out of one call to _posebusters_heatmap_grid, so a bar and a cell cannot
    disagree. A bar is always the whole GROUP, which for Bond geometry is now two cells. The crystal ligands are the first bar rather than a rule: this is
    a like-for-like quantity here, measured on the same 79 pockets by the same code."""
    cols, labels, grid, ns, bars = built
    vals = bars[name]
    fig, ax = plt.subplots(figsize=(FIG_W * 0.86, PB_GROUP_BAR_H * len(labels) + 1.35),
                           dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ys = np.arange(len(labels))[::-1]
    # THE HOUSE HEAT RAMP, NOT THE METHOD PALETTE (2026-09-15, on request). A bar and the heat
    # map cell it comes from are now the same colour as well as the same number, because both
    # go through HEAT_CMAP under the SAME norm -- the one floored at the worst cell of the
    # whole map, passed in rather than recomputed here so a bar cannot key itself to a
    # different scale than the map does. The method's own colour is given up for that: on a
    # chart with one bar per method the name at the end of every bar already says which method
    # it is, so identity colour was spending the figure's only free channel on a label it
    # already has, while LENGTH and colour now say the same thing twice.
    ax.barh(ys, vals, height=0.66, color=[HEAT_CMAP(norm(v)) for v in vals], zorder=3)
    # THE AXIS LINE GOES OVER THE BARS. A bar starts at x=0, which is exactly where the left
    # spine is drawn, and a spine's default zorder (2.5) is under the bars (3) -- so every bar
    # painted over the axis it is measured from, and the axis appeared to stop wherever the
    # shortest bar was. furniture() cannot fix it for everyone: set_axisbelow() moves the
    # GRID, not the spines, and raising them there would put an axis line over every filled
    # curve in the deck. It is bars-from-zero that have the problem, so it is fixed here.
    for sp in ("left", "bottom"):
        ax.spines[sp].set_zorder(5)
    for y, v in zip(ys, vals):
        ax.text(v + 1.4, y, f"{v:.1f}", va="center", ha="left", fontsize=11.5, color=INK,
                zorder=4)
    # The group's name and the quantity on two lines. On one, "Protein clash & overlap - % of
    # molecules passing" is wider than the figure, and a centred x label that overruns cannot
    # be rescued by fit(): it just loses its right-hand end off the canvas.
    furniture(ax, ylabel=None, xlabel=f"{name}\n% of molecules passing", xloc=None,
              xlim=PB_GROUP_BAR_XLIM)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.grid(False, axis="y")
    ax.set_yticks(ys)
    # The method alone (2026-09-16, on request), as on the heat map: every bar is a RATE, so
    # the denominator is not needed to compare two of them, and it is in the CSV per row.
    ax.set_yticklabels([display(lab) for lab in labels], fontsize=13)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    fit(fig, pad=0.5)
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    save(fig, out, f"pb_group_bar_{slug}_{variant}")


@figure("fig-posebusters-group-bars", needs=("metrics.json (posebusters block)",))
def draw_posebusters_group_bars(out):
    """Each check group on its own, one bar per method -- the heat map's columns, unrolled."""
    data, p79_rows, refrows = pose_data()
    use_style()
    drawn = {variant: _posebusters_heatmap_grid(arms, p79_rows, refrows)
             for variant, arms in variants("v", data)}
    norm = _posebusters_heat_norm(drawn["all"][2])
    for variant, built in drawn.items():
        for name in PB_GROUP_BARS:
            _posebusters_group_bar(out, variant, built, name, norm)
    _, labels, _, _, bars = _posebusters_heatmap_grid(arms_for("v", data), p79_rows, refrows)
    for name in PB_GROUP_BARS:
        vals = bars[name]
        order = np.argsort(vals)[::-1]
        print(f"  {name}: " + ", ".join(f"{labels[i]} {vals[i]:.1f}" for i in order[:3])
              + f"  ...  worst {labels[order[-1]]} {vals[order[-1]]:.1f}")


@figure("fig-posebusters-valid-heatmap",
        needs=("metrics.json (posebusters block)", "loading_checks.json"))
def draw_posebusters_valid_heatmap(out):
    """PoseBusters validity as a method x check-group heat map, with the per-check table."""
    data, p79_rows, refrows = pose_data()
    use_style()
    drawn = {variant: _posebusters_heatmap_grid(arms, p79_rows, refrows)
             for variant, arms in variants("v", data)}
    checked, counted = _posebusters_counted_checks()
    loading = _posebusters_loading_rates()
    series = [(REF_LABEL, "reference", refrows)] + [(lab, key, p79_rows[key])
                                                    for lab, key, _ in arms_for("v", data)]
    means, unloaded, n_counted = _posebusters_mean_passed(series, loading)
    print(f"  Mean tests passed / {n_counted}:", means)
    print("  (the molecules that did not load count as "
          f"{PB_HEAT_UNLOADED_PASSES} of {n_counted}: "
          + ", ".join(f"{lab} {n}" for lab, n in unloaded.items() if n) + ")")
    norm = matplotlib.colors.Normalize(50.0, PB_HEAT_VMAX)
    for variant, built in drawn.items():
        _posebusters_heatmap_panel(out, variant, built, norm, means, n_counted)

    cols, labels, grid, ns, _ = drawn["all"]
    worst = np.unravel_index(np.argmin(grid), grid.shape)
    print(f"  colour ramp {norm.vmin:.1f}-{norm.vmax:.0f}%; worst cell: "
          f"{labels[worst[0]]} / {cols[worst[1]]}")
    # The console table heads its columns with the MAP's short names: cut to nine characters
    # the two halves of Bond geometry both came out "Bond".
    print(f"  {'method':16s} {'n':>7s} "
          + " ".join(f"{PB_HEAT_SHORT.get(c, c).replace(chr(10), ' ')[:9]:>9s}" for c in cols))
    for lab, row, n in zip(labels, grid, ns):
        print(f"  {lab:16s} {n:7,d} " + " ".join(f"{v:8.1f}%" for v in row))
    # `n_molecules` is what the PERCENTAGES are over -- the scored molecules -- and
    # `n_counted_for_mean` is what the counts column is over, which is those plus the ones that
    # did not load. Two numbers because they are two populations; one column would have to pick.
    write_csv(out, "posebusters_valid_heatmap",
              ["method", "n_molecules", f"pct_pass_{cols[0]}", "mean_tests_passed", "n_tests",
               "n_counted_for_mean"] + [f"pct_pass_{c}" for c in cols[1:]],
              [[lab, n, round(float(row[0]), 2), means[lab], n_counted,
                n + unloaded[lab]] + [round(float(v), 2) for v in row[1:]]
               for lab, row, n in zip(labels, grid, ns)])

    # The per-check table the groups are built from -- a group at 99.9% can be one check at
    # 99.9% or four at 100 and one at 99.9, and only this says which.
    checks = [c for _, cs in PB_GROUPS for c in cs]
    group_of = {c: name for name, cs in PB_GROUPS for c in cs}
    fails = _posebusters_check_failures(data, p79_rows, refrows)
    rows = []
    for lab, key, _ in [(REF_LABEL, "reference", None)] + list(arms_for("v", data)):
        g = fails[key]
        rows += [[lab, g["n_mols"], group_of[c], c, g["counts"].get(c, 0),
                  round(100 - g["rates"].get(c, 0.0), 3)] for c in checks]
    write_csv(out, "posebusters_check_pass_rates",
              ["method", "n_molecules", "group", "check", "n_failed", "pct_pass"], rows)


# ── what the validity rate is made of ──────────────────────────────────────────
# A PARTITION OF EACH ARM'S MOLECULES, NOT A SECOND VIEW OF THE HEAT MAP. The map and the
# dot row both answer "how often does this check pass"; neither can answer "what is this
# arm's validity rate made of", because those columns overlap -- a molecule failing bond
# angles AND a clash is in both -- and overlapping shares cannot be stacked or added.
#
# So a molecule is counted ONCE here, by the GROUPS it fails: none (it is valid), exactly one
# (that group's segment), or more than one (the `Two or more` segment). The segments therefore
# sum to 100% of the arm, and the bar reads as a budget: the valid block is the overall level,
# and what follows it says what spent the rest. It is the UpSet figure's per-molecule sets
# aggregated to the map's six groups, which is why the three of them cannot disagree.
#
# WHAT IT SHOWS: the two voxel arms and FuncBind spend most of their budget on ONE group,
# ring pucker, where DiffSBDD and FuncBind spend a fifth of everything on `Two or more` --
# molecules that are wrong in several ways at once. Pocket2Mol's is almost all protein clash.
# A reader can see in one row what the heat map needs eight columns and a comparison for.
PB_BUD_MULTI = "Two or more groups"
PB_BUD_VALID = "PoseBusters valid"
# THE MOLECULES THAT NEVER REACHED THE CHECKS (2026-09-23, on request: "load 고려"). They are
# absent from metrics.json -- that is why they could not be loaded -- so they are added to the
# population here rather than counted from it, and the bar becomes a partition of everything
# the arm produced rather than of everything that was scored. It is 4 molecules for VoxBind
# and 8 for CoDE and nothing anywhere else, so the segment is a hairline where it exists at
# all; it is drawn at its true width and named in the run log, never widened to be seen.
PB_BUD_UNLOADED = "Did not load"
# THE AXIS STARTS AT 45%, NOT 0 (2026-09-23, on request; 40 first, then 45). A stacked bar normally has to start
# at a true zero, because a segment's LENGTH is its share -- and that still holds: every
# segment keeps its width, the cut only takes 40 points off the `valid` block, whose value is
# written at its own end. What the cut buys is 1.7x the width for the failure budget, which is
# the part of the bar this figure exists to show.
PB_BUD_XLIM = (45.0, 100.0)     # 40 -> 45 (2026-09-23): FuncBind's 50.1 is the lowest bar
# THE KEY IS AS WIDE AS THE FIGURE (2026-09-23, on request: "legend 가로 길이 figure 가로랑
# 맞춰주고") -- and at four columns it is the key that sets that width, not the plot. Two rows
# of four at 13.5 pt need about 10 in of labels, more than the 0.5x plot's 8.3, so the canvas
# grows to the key (the export boundary pads to the rendered extent) and the key ends up
# spanning it to within the 0.04 in clip pad. The plot keeps its 4.95 in throughout.
# WHAT WAS TRIED AND IS NOT HERE: `mode="expand"` with a full-width bbox. It does make the box
# exactly the canvas width, but it is a CLAMP, not a fit -- given less width than the labels
# need it overlaps the columns rather than growing the canvas, and `Protein clash & overlap`
# ran into the next swatch. Two columns is the layout that fits inside a 0.5x plot's width;
# four is the layout asked for, and it is what decides the canvas.
PB_BUD_LEG_NCOL = 4              # 8 entries, so two rows of four
# WHAT THE KEY CALLS THE SEGMENTS (2026-09-23, on request): `Valid` rather than `PoseBusters
# valid`, and the two compound names cut to their first half -- `Protein clash`, `Self-clash`.
# This is a DISPLAY map and nothing else: PB_GROUPS still names the groups in full, so the CSV,
# the run log, the group-bar figures and their filenames are untouched, and the cut names never
# reach anything that is read without this figure around it. The halves that go are the ones a
# key does not need to carry -- `overlap` and `energy` are the second check in each group, not
# a second kind of failure, and both groups are keyed by what the reader is looking at.
PB_BUD_LEG_NAMES = {PB_BUD_VALID: "All passed",   # was `Valid` (2026-09-23, on request)
                    "Protein clash & overlap": "Protein clash",
                    "Self-clash & energy": "Self-clash",
                    # `2+ groups`, not `Two or more groups` (2026-09-23, on request: shorter).
                    # It is what the run log has always called this segment, so the figure and
                    # the log now read the same, and the map below is the one place either
                    # looks. `+` is doing the work `or more` did, in two characters.
                    PB_BUD_MULTI: "2+ groups"}
PB_BUD_LEG_FS = 13.5             # 11.5 -> 13.5 (2026-09-23, on request); now the STARTING
                                 # size for the fit below, not the size that lands
PB_BUD_LEG_SPAN = 0.98           # of the figure width, so the key's rule clears the canvas edge
PB_BUD_LEG_FS_MAX = 26.0         # a key cannot be set larger than the plot's own tick labels
# THE KEY CAME BACK UP (2026-09-23, on request: "legend 위로 옮기고"), AND ITS GAP IS NOW IN
# INCHES. It was 0.016 + 0.055 of the FIGURE, and the two variants are 6.58 in and 3.46 in tall,
# so one fraction bought 0.47 in of air under `all` and 0.25 under `core` -- the same number
# meaning two different gaps, which is why the first pass at raising it (0.012) cleared the x
# label on `all` and cut into it on `core`. 0.16 in is measured on the paper and lands the same
# on both, and it is a third of what `all` carried before, which is the raise that was asked
# for. The key itself never moves: it is pinned 0.008 off the canvas bottom, and what changes
# is how much bottom margin the axes gives back to it.
PB_BUD_LEG_GAP_IN = 0.16
# HOW CLOSE THE COLUMNS SIT (2026-09-23, on request: "메소드 사이 여백을 너무 많이 하지는"). The
# alternative was four rows of two, which fits a 0.5x plot by stacking instead of spreading --
# not what was wanted. matplotlib's default columnspacing is 2.0 em, i.e. 27 pt of paper
# between entries at 13.5 pt, three times over; 0.8 em keeps the columns apart without the
# canvas having to carry the gaps, and takes 1.0 in off the figure on its own.
PB_BUD_LEG_COLSPACE = 0.8
# SEGMENT FILLS ARE NOT THE METHOD PALETTE: a segment is a CHECK GROUP, and the palette's hues
# mean methods everywhere else in the deck -- CoDE's blue standing for `Ring pucker` here would
# be read as CoDE. The valid block is off the ramp altogether, near-white, so the ink in a row
# IS its failure budget.
#
# THE RAMP IS THE DECK'S SHARED ONE (2026-09-23, on request: "공용 colorbar 사용해줘"). HEAT_CMAP
# from 00_core -- #E8F2F7 to the line through highlight("CoDE") continued past it, the ramp
# `heat_colorbar` draws on every heat map in the deck -- not a ramp mixed for this figure. Sampling it costs
# nothing here: it spans luminance 0.87 down to 0.14, room enough for the four levels below.
# The steps move with the shared ends, so if 00_core re-tunes the ramp this figure follows.
#
# EVERY GROUP GETS ITS OWN LIGHTNESS, AND THE HATCH IS A SECOND CUE ON TOP (2026-09-23, on
# request: "제일 끝 부분은 빼고 중간 구역들에 해줘, 거기가 구분이 안 돼"). What was there before
# was THREE lightness steps crossed with a hatch, six segments in three pairs -- and a pair is
# two segments of the SAME blue told apart by hatching alone. That works on a wide block and
# fails exactly where this figure spends most of its width: `Planarity` at 0.3% of a bar is a
# hairline, and a hairline has no room to show a hatch, so it read as its unhatched neighbour.
# Now the groups that actually occur are spread across the ramp, one luminance each, and the
# hatch falls on every other one -- so adjacent segments differ in COLOUR, and alternate ones
# differ in colour and texture both.
#
# THE LADDER IS SPENT ON THE SEGMENTS THAT EXIST, not on the six PB_GROUPS names. `Valence &
# connectivity` takes no molecule from any arm, and giving it a step wasted the lightest one on
# a segment nobody ever sees; the live five now take the whole range.
PB_BUD_VALID_FILL = "#F4F5F8"
# The endpoints are LUMINANCES, not ramp positions: the ramp's own lightness falls fastest at
# its light end, so evenly spaced positions are not evenly spaced greys. The light end, 0.70,
# is a step below the near-white valid block at 0.91.
#
# THE DARK END IS CoDE'S OWN LUMINANCE (2026-09-23, on request: "protein clash 쯤에 CoDE
# 색깔이랑 비슷하도록"). Since 00_core now runs the ramp THROUGH highlight("CoDE"), asking for
# the darkest group at that colour's luminance is asking for that group to BE it, exactly --
# and it is, without a hex literal anywhere in this file. It lands on `Self-clash`, the segment
# next to `Protein clash`, and the five groups come out at 0.70 / 0.57 / 0.43 / 0.30 / 0.163,
# which is 0.134 apart where the old ladder was 0.105: this made the bar EASIER to read, not
# harder. PROTEIN CLASH ITSELF CANNOT BE CoDE and that is arithmetic, not a choice: it is the
# fourth of five rungs, so its luminance is 0.25 x (lightest) + 0.75 x (darkest), which at
# CoDE's 0.151 forces a lightest rung of 0.42 (no light blue left in the bar) and a darkest
# below 0.06, i.e. a near-black ramp end and every heat map in the deck going with it.
PB_BUD_GROUP_LUM_HI = 0.70
# On every other group, THE FIRST ONE FIRST (2026-09-23, on request: "빗금은 밑에 있는 항목들만
# 해보자 지금 legend 기준으로"). The key is two rows filled DOWN its columns, so its bottom row is
# the odd entries of the eight -- bond geometry, ring pucker, self-clash, did-not-load -- which
# are groups 0, 2 and 4 of the five here plus the one segment that keeps a hatch at the end.
# Parity is therefore even, not odd, and the hatch now reads as a row of the key rather than as
# a scatter across it.
PB_BUD_GROUP_HATCH = "///"
# THE HATCH IS GREY, NOT WHITE (2026-09-23, on request: "빗금 흰색 말고 회색으로"). matplotlib
# draws a patch's hatch in its EDGE colour, so this is also that segment's outline -- the thin
# separator between it and its neighbours. White was invisible on the two lightest steps, which
# are where a hairline segment most needs a texture; GRID, the house's quiet rule colour, is
# dark enough to show on them and light enough not to close up the darkest ones. Unhatched
# segments keep the white edge: nothing is drawn on them, so the outline is all it would be.
PB_BUD_HATCH_INK = GRID
# THE LAST TWO SEGMENTS ARE PLAIN AT THE RAMP'S DEEP END. `2+ groups` lost its cross hatch on
# the same request: it is the darkest thing in the bar and sits at the end of every row, where
# a heavy texture was doing no work that the colour was not already doing. `Did not load` keeps
# a fine dotted one -- and only because without it the key would show two identical swatches:
# it is 0.1% of two arms and nothing anywhere else, a hairline that is read in the key, not on
# the bar. It is also the one segment that is not a failure of a group at all -- the molecule
# never reached a check -- so a texture that says `not one of these` is not wrong.
PB_BUD_MULTI_STEP = (1.0, None)
PB_BUD_UNLOADED_STEP = (1.0, "...")
PB_BUD_ROW_H = 0.52              # inches per method row
PB_BUD_BAR = 0.62                # of the row pitch
PB_BUD_W = 4.95                  # inches of the axis (9.9 x 0.5, 2026-09-23, on request)
PB_BUD_NUM_FS = 13               # the valid rate, written at the end of its own block


def _posebusters_budget(arms, p79_rows, refrows, loading):
    """(segment names, {label: [share per segment]}, {label: n}) -- each arm partitioned.

    A molecule lands in exactly one segment: `valid` if it fails nothing, its group if every
    check it fails is in one group, `Two or more groups` otherwise. The population is the
    molecules PoseBusters scored, which is the population every percentage in this family is
    over."""
    group_of = {c: name for name, cs in PB_GROUPS for c in cs}
    names = [name for name, _ in PB_GROUPS]
    segments = [PB_BUD_VALID] + names + [PB_BUD_MULTI, PB_BUD_UNLOADED]
    series = [(REF_LABEL, "reference", refrows)] + [(lab, key, p79_rows[key])
                                                    for lab, key, _ in arms]
    out, ns = {}, {}
    for lab, key, rows in series:
        scored = [r for r in rows if r["v"] is not None]
        hit = collections.Counter()
        for r in scored:
            groups = {group_of[c] for c in r["f"]}
            hit[PB_BUD_VALID if not groups else
                (next(iter(groups)) if len(groups) == 1 else PB_BUD_MULTI)] += 1
        hit[PB_BUD_UNLOADED] = loading[key]["not_loaded"]
        n = max(1, len(scored) + hit[PB_BUD_UNLOADED])
        out[lab] = [100 * hit[s] / n for s in segments]
        ns[lab] = n
        # The segments ARE the arm, so they have to add up to it; a molecule counted twice or
        # not at all is a bug this catches rather than a rounding artefact to shrug at.
        assert sum(hit.values()) == n
    return segments, out, ns


def _posebusters_luminance(rgb):
    """Relative luminance, the WCAG one -- sRGB linearised first, which is what makes two
    colours `a step apart` mean the same thing at the light end and at the dark end."""
    f = lambda u: u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4   # noqa: E731
    r, g, b = rgb[:3]
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _posebusters_ramp_at(lum):
    """Where on the shared ramp that luminance sits. Inverted by sampling rather than solved:
    the ramp is whatever 00_core says it is, and a closed form would have to be rewritten every
    time it changes."""
    return min((i / 1000 for i in range(1001)),
               key=lambda t: abs(_posebusters_luminance(HEAT_CMAP(t)) - lum))


def _posebusters_budget_fills(segments, live):
    """{segment: (colour, hatch)} -- the ladder spread over the groups that actually occur.

    `live` decides which groups get a rung, so a group that never takes a molecule cannot
    spend one. Both variants are built from the SAME `live` (the all-arms one), so a segment
    is the same colour in `core` as in `all`."""
    ends = (PB_BUD_VALID, PB_BUD_MULTI, PB_BUD_UNLOADED)
    groups = [s for s in segments if s not in ends]
    drawn = [s for s in groups if s in live]
    hi = PB_BUD_GROUP_LUM_HI
    lo = _posebusters_luminance(matplotlib.colors.to_rgb(highlight("CoDE")))
    span = max(1, len(drawn) - 1)
    fills = {PB_BUD_VALID: (PB_BUD_VALID_FILL, None),
             PB_BUD_MULTI: (HEAT_CMAP(PB_BUD_MULTI_STEP[0]), PB_BUD_MULTI_STEP[1]),
             PB_BUD_UNLOADED: (HEAT_CMAP(PB_BUD_UNLOADED_STEP[0]), PB_BUD_UNLOADED_STEP[1])}
    for i, s in enumerate(drawn):
        t = _posebusters_ramp_at(hi + (lo - hi) * i / span)
        fills[s] = (HEAT_CMAP(t), PB_BUD_GROUP_HATCH if i % 2 == 0 else None)
    # A group with no molecule is never drawn and never keyed; it still needs an entry, because
    # the panel looks every segment up before it knows the share is zero.
    for s in groups:
        fills.setdefault(s, (HEAT_CMAP(1.0), None))
    return fills


def _posebusters_budget_panel(out, variant, segments, shares, ns, fills, drawn_segments):
    labels = list(shares)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(PB_BUD_W + 3.3, PB_BUD_ROW_H * n + 1.9), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ys = np.arange(n)[::-1]                      # the crystal ligands first, as everywhere
    for lab, y in zip(labels, ys):
        left = 0.0
        for seg, v in zip(segments, shares[lab]):
            if v > 0:
                fill, hatch = fills[seg]
                ax.barh(y, v, left=left, height=PB_BUD_BAR, color=fill, hatch=hatch,
                        edgecolor=PB_BUD_HATCH_INK if hatch else "white",
                        linewidth=0.9, zorder=3)
            left += v
        # The headline at the end of the valid block: the number the rest of the row explains.
        # `center_baseline`, not `center` (2026-09-23, on request: the number read high). A
        # text box is as tall as the FONT -- ascender to descender -- and `center` centres that
        # box, so digits, which use none of the descender, come out above the bar's middle by
        # half the descent. `center_baseline` centres the glyphs instead.
        ax.text(shares[lab][0] - 0.8, y, f"{shares[lab][0]:.1f}", ha="right",
                va="center_baseline", fontsize=PB_BUD_NUM_FS, color=INK, zorder=5)
    furniture(ax, xlabel="% of the arm's molecules", xloc=None, xlim=PB_BUD_XLIM)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(False, axis="y")
    # A RULE UNDER THE CRYSTAL LIGANDS, in the x axis's own pen (2026-09-23, on request). The
    # reference is not a ninth method: it is the benchmark the eight are read against, and the
    # heat map rules it off for the same reason. AXIS_WIDTH, not AXIS_LW, because the export
    # boundary takes the spines to 2.64 and a rule meant to look like the axis has to end up
    # there too -- it is not a spine, so nothing would do it for us.
    ax.axhline(n - 1.5, color=AXIS, lw=AXIS_WIDTH, zorder=6)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_zorder(5)
    ax.set_yticks(ys)
    ax.set_yticklabels(["Reference" if lab == REF_LABEL else display(lab) for lab in labels])
    ax.set_ylim(-0.6, n - 0.4)
    handles = [Patch(facecolor=fills[s][0], hatch=fills[s][1], edgecolor="white",
                     label=PB_BUD_LEG_NAMES.get(s, s))
               for s in drawn_segments]
    fit(fig, pad=0.5)
    # THE KEY GOES ABOVE THE PLOT (2026-09-23, on request: "legend 그림 위로 보내줘"). It spent
    # the day under the axes, where it had to be measured against the x label; above there is
    # nothing between it and the top spine -- no title, no top ticks -- so the gap is the plain
    # distance from the key's bottom to the axes, and PB_BUD_LEG_GAP_IN means what it says
    # without the correction pass the bottom placement needed.
    #
    # AND THE ENTRIES READ DOWN, NOT ACROSS (same request: "위 아래 순서로 ... valid 다음 밑에
    # bond geometry가 오게"), which is matplotlib's native column-major fill -- so `handles`
    # goes in as it is and row_major(), which exists to undo exactly that, is not used here.
    # The pairs a column makes are the point: valid over bond geometry, planarity over ring
    # pucker, the two clashes, and the two segments that are not single groups.
    #
    # AND IT IS SIZED TO THE FIGURE'S WIDTH, NOT SET (2026-09-23, on request: "legend 가로 길에
    # 맞게 키워줘"). Cutting `& overlap`, `& energy` and `Two or more groups` left the key
    # narrower than the canvas it used to decide, so the point size is now FITTED: draw at
    # PB_BUD_LEG_FS, measure the box, scale by how far it falls short of PB_BUD_LEG_SPAN, draw
    # again. Everything in a matplotlib key is in font units -- handle, pads, column gap,
    # border -- so its width is very nearly linear in the point size and two passes land within
    # a fraction of a percent; the loop stops early when they do. PB_BUD_LEG_FS is now the
    # STARTING size rather than the final one, and PB_BUD_LEG_FS_MAX is the cap for the day the
    # names get shorter still, so the key cannot grow taller than the plot it keys.
    def _key(fontsize):
        return legend(fig, handles, loc="upper center", ncol=PB_BUD_LEG_NCOL,
                      fontsize=fontsize, bbox_to_anchor=(0.5, 0.992),
                      columnspacing=PB_BUD_LEG_COLSPACE)

    fs, leg = PB_BUD_LEG_FS, _key(PB_BUD_LEG_FS)
    for _ in range(3):
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        scale = PB_BUD_LEG_SPAN / max(1e-6, bb.x1 - bb.x0)
        if abs(scale - 1) < 0.004 or fs >= PB_BUD_LEG_FS_MAX:
            break
        fs, _ = min(PB_BUD_LEG_FS_MAX, fs * scale), leg.remove()
        leg = _key(fs)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    # The axes gives back the key's MEASURED height plus PB_BUD_LEG_GAP_IN, converted here
    # because subplotpars are figure fractions and the gap is a distance on paper.
    fig.subplots_adjust(top=max(0.4, fig.subplotpars.top - (bb.y1 - bb.y0)
                                - PB_BUD_LEG_GAP_IN / fig.get_figheight()))
    # Then measured and corrected once: `fit` leaves its own margin above the axes, so giving
    # back the key's height alone lands short or long of the gap by whatever that margin was.
    fig.canvas.draw()
    key = leg.get_window_extent().transformed(fig.transFigure.inverted())
    off = (key.y0 - ax.get_position().y1) - PB_BUD_LEG_GAP_IN / fig.get_figheight()
    if abs(off) > 1e-4:
        fig.subplots_adjust(top=max(0.4, min(0.995, fig.subplotpars.top + off)))
    save(fig, out, f"pb_budget_{variant}")


@figure("fig-posebusters-failure-budget",
        needs=("metrics.json (posebusters block)", "loading_checks.json"))
def draw_posebusters_failure_budget(out):
    """What each arm's validity rate is made of: one bar per method, summing to 100%."""
    data, p79_rows, refrows = pose_data()
    use_style()
    loading = _posebusters_loading_rates()
    built = {variant: _posebusters_budget(arms, p79_rows, refrows, loading)
             for variant, arms in variants("v", data)}
    segments, all_shares, all_ns = built["all"]
    # A segment that is zero for EVERY arm is a key entry pointing at nothing -- `Valence &
    # connectivity` is 100% everywhere, so no molecule is ever assigned to it. Dropped from the
    # key and named in the run log instead; it keeps its place in the partition either way.
    live = [s for i, s in enumerate(segments)
            if any(shares[i] > 0 for shares in all_shares.values())]
    fills = _posebusters_budget_fills(segments, live)
    for variant, (segs, shares, ns) in built.items():
        _posebusters_budget_panel(out, variant, segs, shares, ns, fills, live)

    empty = [s for s in segments if s not in live]
    print(f"\n  segments with no molecule in any arm: {', '.join(empty) or 'none'}")
    head = [PB_BUD_LEG_NAMES.get(s, s) for s in live]
    print(f"  {'arm':16s} {'n':>7s} " + " ".join(f"{s[:11]:>11s}" for s in head))
    for lab, vals in all_shares.items():
        row = [v for s, v in zip(segments, vals) if s in live]
        print(f"  {lab:16s} {all_ns[lab]:7,d} " + " ".join(f"{v:10.1f}%" for v in row))
    write_csv(out, "posebusters_failure_budget",
              ["method", "n_molecules", "segment", "pct_of_molecules"],
              [[lab, all_ns[lab], s, round(v, 3)]
               for lab, vals in all_shares.items() for s, v in zip(segments, vals)])


# ── every check, one row each ───────────────────────────────────────────────────
# THE WHOLE OF WHAT POSEBUSTERS REPORTS, 22 ROWS, against the heat map's eight columns of
# five. The map above is a partition of `valid` into groups, which is the right shape for
# reading a method; it cannot say WHICH check inside a group moved, and a group at 99.9 can
# be one check at 99.9 or four at 100 beside one at 99.9. That table existed only as
# check-pass-rates.csv. This is it drawn, with the group structure kept as the row blocks so
# the two figures line up: the map's columns are this table's blocks, in the same order.
#
# ELEVEN OF THE 22 ARE 100.0 FOR EVERY ARM, AND THEY ARE NOT ALL THE SAME KIND OF 100.
# Eight of them CANNOT fail here and are drawn in neutral grey rather than at the top of the
# ramp: the six cofactor and water checks have nothing to measure, because the receptor every
# arm is scored against is the pocket10 crop and it carries no HETATM and no HOH, and the two
# loading checks are preconditions of being scored at all -- a molecule PoseBusters could not
# load has no row in the file to count. The other three -- sanitization, all atoms connected,
# no radicals -- are real measurements that every arm happens to pass, and they stay on the
# ramp, because "nobody broke this" and "this could not break" are different findings. The
# eight are asserted to be 100.0, so a run where one of them starts failing stops the build
# instead of painting a real failure grey.
#
# WHAT IS LEFT IS ELEVEN LIVE ROWS, and they say what the group columns compress: every arm's
# worst check is a geometry or a clash one, no arm is bad at more than three of them, and the
# two rows that spread the field widest are `bond angles` (TargetDiff 77.4 against
# Pocket2Mol's 97.9) and `non-aromatic ring non-flatness` (CoDE and FuncBind at 77.5 against
# AR's 99.9). The published baselines are not failing a different SET of checks from ours --
# they trip different ones of the same eleven.
PB_TAB_VALID = "PoseBusters valid"
# The blocks, in the heat map's group order, plus the two checks PoseBusters reports and does
# NOT grade: `n_checks` is 20 in every record, and `valid` is the AND over those 20 (verified
# -- it equals the AND over all 22 as well, because the two extra ones never fail).
PB_TAB_GROUPS = tuple(PB_GROUPS) + (("Not scored", ("mol_pred_loaded", "mol_cond_loaded")),)
# 100.0 BY CONSTRUCTION, not by merit -- see the note above. It is a CAPTION matter and not
# drawn any more (2026-09-23, on request): the grey says the cell is off the ramp, and why
# it is off the ramp is in figures/README.md, in the run log's `(by construction)` marks and
# in all-checks.csv's `cannot_fail_here` column.
#
# THE TWO LOADING CHECKS USED TO BE IN HERE AND ARE NOT ANY MORE (2026-09-23, on request:
# "run them and update"). They were grey on the argument that a molecule PoseBusters could
# not load would not be in metrics.json to be counted -- an assumption, and the wrong one:
# metrics.json carries 60 molecules with no PoseBusters result at all (AR 40, DiffSBDD 20),
# not because they failed to load but because the scoring run TIMED OUT on their chunk. So
# they are measured now, by build_loading_checks.py, over every record of every samples.sdf
# including those 60. What stays grey is only what cannot be measured here at all.
PB_TAB_LOADING = ("mol_pred_loaded", "mol_cond_loaded")
PB_TAB_BY_CONSTRUCTION = frozenset({
    "minimum_distance_to_organic_cofactors", "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters", "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors", "volume_overlap_with_waters"})
# The fill of a cell that is 100.0 BY CONSTRUCTION: the neutral grey the heat map gives its
# counts column, for the same reason -- a cell that is not on the ramp must not look like
# a cell at the top of it.
PB_TAB_FLAT_FILL = "#F2F2F2"
# INCHES PER ROW. 0.34 -> 0.374, a tenth taller (2026-09-23, on request: "세로 칸을 1.1배 정도
# 높게"). Nothing in the row grew with it -- the names are 13 pt and the numbers 14.5 -- so what
# the tenth buys is air: a 14.5 pt number is 0.20 in of ink in a 0.374 in cell, where the row
# pitch of the check names it sits beside is what makes a table of 23 rows scannable. Every
# other vertical measure here is in ROWS (PB_TAB_GAP, PB_TAB_HEAD_H, PB_TAB_FOOT_H), so they
# follow it and the proportions of the table are unchanged.
PB_TAB_ROW_H = 0.374             # inches per row
PB_TAB_GAP = 0.45                # blank between two blocks, in rows
# A TABLE'S TYPE IS THE TABLE'S OWN, not the deck's axis type: this is 23 rows where the heat
# map is nine, and at the export boundary's 17.36 pt the long check names wrap to three lines
# and the figure is 15 in tall before anything is drawn. So the names, the block names, the
# headings and the numbers are TEXT rather than tick labels -- the same reason the heat map
# writes its cells with ax.text -- and the one thing that stays at house size is the
# colourbar, which is the figure's only axis. Nothing here is a tick label, so nothing is
# silently resized on export.
PB_TAB_NAME_FS = 13              # the check names
PB_TAB_GROUP_FS = 13             # the block names, in the left column
PB_TAB_HEAD_FS = 14.5            # the method headings
PB_TAB_NUM_FS = 14.5             # the numbers in the cells
# WHAT THE FIRST TWO COLUMNS ARE CALLED (2026-09-23, on request). They used to be unheaded,
# which left the block names reading as a stray word beside the first row of each block rather
# than as a column of their own.
PB_TAB_HEADS = ("Category", "Specific check")
# WHAT THE SUMMARY ROW IS CALLED where those two columns MERGE (2026-09-23, on request:
# `Total`, then `Overall valid`). The second name is the better one and not only a longer one:
# the row is the `valid` FLAG -- the AND over the 20 graded checks -- and a row called `Total`
# in a table of percentages reads as a sum of the rows under it, which it is not.
PB_TAB_TOTAL = "Overall valid"
# EVERY RULE IS BLACK (2026-09-23, on request, twice). The first pass moved them off the grid's
# grey and onto INK, and INK IS NOT BLACK: #514F52 is a warm near-black, a third of the way to
# white, and at a rule's width that reads as exactly what it is -- grey. So the table's rules
# are #000, and they are the only thing here that is: the text stays on the deck's ink, which
# every other figure in the deck is set in. Two weights, not two colours: the frame -- under
# the headings, under the summary row, under the last row -- is the heavy pen, and the category
# dividers and the column divider are the light one.
PB_TAB_RULE_INK = "#000000"
PB_TAB_RULE_LW, PB_TAB_DIVIDER_LW = 2.0, AXIS_LW
PB_TAB_INSET = 0.10              # inches of white before the block names
PB_TAB_PAD = 0.16                # ... a text column keeps off the cells, and a cell off its number
# The blank between the block names and the check names, with the rule down the middle of it.
# It is a constant and not a by-product: sized from the text alone, the two columns came out
# 0.05 in apart and `Protein clash & overlap` read as the first word of the row beside it.
PB_TAB_GUTTER = 0.34
PB_TAB_HEAD_H = 1.15             # rows of heading strip above the table
PB_TAB_FOOT_H = 0.45             # ... of air under the last row
# Where the rule under the column names sits, in rows above the first row. The headings are
# set on a baseline at -0.30, so this is the gap between their descenders and the rule.
PB_TAB_HEAD_RULE = 0.16
# A RULE PER CATEGORY (2026-09-23, on request), down the middle of the blank between two
# blocks. The blank stays: a rule says where a category ends, the blank makes the block on
# either side of it one object, and the heat map's own gap-plus-rule is the same pairing. The
# rule after `PoseBusters valid` is the INK one the heat map rules its reference row off with,
# because that boundary is a change of KIND -- an aggregate above, the checks it is made of
# below -- while the category rules divide like from like and are drawn in the grid's pen.
PB_TAB_MARGIN = 0.14             # inches of canvas margin
# THE TABLE'S RANGE IS ITS OWN; THE RAMP IS NOT (2026-09-23, on request: "pb valid heatmap, pb
# budget, pb check 이것도 다 고쳐줘"). What this table needed was never a different ramp but a
# different FLOOR: the heat map's 50-100 spends its whole range above the worst CHECK cell,
# 76.9, so twenty rows come out one shade. Floored at 45 the `Overall valid` row starts at the
# ramp's light end at FuncBind's 50.1 and the check rows still separate. So PB_TAB_VMIN stays
# and PB_TAB_CMAP is gone: the colours are HEAT_CMAP, like every other heat figure in the deck.
PB_TAB_VMIN = 45.0
# THE COLOURBAR STANDS TO THE RIGHT OF THE TABLE (2026-09-23, on request: "colorbar 오른쪽으로
# 바꾸어줘"). It lay UNDER it before, and under it the strip was measured in inches of HEIGHT;
# on the right the same three numbers are inches of WIDTH -- the blank between the last cell
# and the bar, the bar itself, and the room its quarter-turned numbers and its name need to
# the right of it. The table is what sets this figure's shape: 23 rows against nine columns is
# a tall block, so the canvas has height to spare and no width at all, and a bar down its side
# is read against the rows it colours rather than a strip the eye has to leave the table for.
# The third number is the widest of the three because a vertical bar's furniture is all beside
# it: the turned numbers (~0.3 in), HEAT_CBAR_LABELPAD, and the rotated name (~0.35 in).
# 0.80 is that measured off the PNG rather than guessed -- at 0.95 the canvas ended on
# 0.17 in of white nothing was standing in.
PB_TAB_CBAR = (0.40, 0.26, 0.80)
# ... and the bar's LENGTH is now a share of the table's height rather than of its width.
PB_TAB_CBAR_LEN = 0.55
def _posebusters_text_in(strings, fontsize):
    """The widest of `strings`, in inches, MEASURED rather than estimated.

    Every column width in this table is what its text needs, and an estimate of 0.55 em per
    character is wrong by a third between `minimum distance to inorganic cofactors` and
    `no radicals`. A throwaway figure is the only way to ask the renderer before the real
    canvas exists, which is what the arithmetic below needs."""
    probe = plt.figure(figsize=(1, 1), dpi=220)
    r = probe.canvas.get_renderer()
    out = 0.0
    for s in strings:
        t = probe.text(0, 0, s, fontsize=fontsize)
        out = max(out, t.get_window_extent(r).width / probe.dpi)
        t.remove()
    plt.close(probe)
    return out


def _posebusters_category_name(name):
    """A block's name for the left column, broken after the ampersand (2026-09-23, on request).

    `Protein clash & overlap` on one line sets a 2.10 in column for a table whose next-longest
    category is 1.16, and the column is as wide as its widest name. Broken it asks for 1.45.
    THE BREAK GOES BEFORE THE `&` (2026-09-23, on request: "&을 다음줄로 넘겨줘"), so the second
    line opens with it -- `Valence` over `& connectivity`, `Protein clash` over `& overlap`. It
    sat after the ampersand first, on the argument that a line opening with `&` reads as a new
    name; carrying it down instead keeps the first line a whole word, which is what is read at
    a glance in a column of category names. Only the setting changes -- the name itself still
    heads the CSV's `group` column and the run log, in full and on one line, so nothing that is
    read back has a newline in it."""
    return name.replace(" & ", "\n& ", 1)


def _posebusters_check_name(check):
    """The PoseBusters key as English, exactly, on one line, opening with a capital.

    Nothing is abbreviated -- see _posebusters_wrap_check for why the ring checks in
    particular cannot be -- and nothing is wrapped: a row is one line tall. The capital is
    the FIRST LETTER ONLY (2026-09-23, on request), never title case: `Non-aromatic ring
    non-flatness` is one name for one check, and capitalising its words would make the two
    halves of it read as two."""
    name = check.replace("_", " ")
    return name[:1].upper() + name[1:]


def _posebusters_pct(v):
    """One decimal, FLOORED. A pass rate of 99.98 that printed as 100.0 would read as a check
    nothing failed, which is the one claim this table exists to let a reader check."""
    return "100.0" if v >= 100 else f"{math.floor(v * 10) / 10:.1f}"


def _posebusters_loading_rates():
    """The two loading columns as MEASURED, from build_loading_checks.py's export.

    That script re-runs PoseBusters' own `Loading` module -- dock.yml's first, the one that
    emits these two columns -- over every record of every samples.sdf. Its denominator is
    therefore every molecule there IS, where the other twenty rows of this table are over the
    molecules PoseBusters managed to score; the two differ by the 60 the scoring run timed
    out on, which is exactly why this row could not be read off metrics.json."""
    j = json.load(open(legacy("fig-posebusters", "loading_checks.json")))
    return {key: {"mol_pred_loaded": 100 * a["mol_pred_loaded"] / max(1, a["records"]),
                  "mol_cond_loaded": 100 * a["mol_cond_loaded"] / max(1, a["records"]),
                  "records": a["records"], "unscored": a["unscored"],
                  # The molecules that did NOT load, which is not the same as the records
                  # metrics.json is missing: TargetDiff is short 511 of them and every one
                  # loaded, so its gap is a filter upstream of this deck, not a failure.
                  "not_loaded": a["records"] - a["mol_pred_loaded"]}
            for key, a in j["arms"].items()}


def _posebusters_all_checks_grid(arms, fails, loading):
    """(columns, {check: [% passing per column]}, n per column) -- every check, every arm.

    Built from the check-failure counts the breakdown figure draws, so a row here and a bar
    there are the same number. `valid` is not one of those counts -- it is the all-must-pass
    AND over the 20 graded checks -- so the caller reads it off the molecules themselves."""
    series = [(REF_LABEL, "reference")] + [(lab, key) for lab, key, _ in arms]
    checks = [c for _, cs in PB_TAB_GROUPS for c in cs]
    if len(set(checks)) != 22:
        raise ValueError(f"PB_TAB_GROUPS holds {len(set(checks))} checks, not the 22 "
                         "PoseBusters reports -- a release has added or renamed one")
    seen = {c for _, key in series for c in fails[key]["counts"]}
    if not seen <= set(checks):
        raise KeyError(f"check outside PB_TAB_GROUPS: {sorted(seen - set(checks))}")
    grid = {c: [100 - fails[key]["rates"].get(c, 0.0) for _, key in series] for c in checks}
    # The loading rows come from the re-run, not from the scored molecules' own checks: the
    # molecules whose loading is in question are the ones metrics.json has no checks for.
    for c in PB_TAB_LOADING:
        grid[c] = [loading[key][c] for _, key in series]
    for c in sorted(PB_TAB_BY_CONSTRUCTION):
        if any(abs(v - 100) > 1e-9 for v in grid[c]):
            raise ValueError(f"{c} is no longer 100% everywhere -- it is drawn grey as a check "
                             "that CANNOT fail on this benchmark; re-read "
                             "PB_TAB_BY_CONSTRUCTION before changing this")
    # Two denominators, and the table says which is which rather than quoting one: `ns` is
    # the molecules PoseBusters scored, `ns_load` every record the loading re-run saw.
    return (series, grid, [fails[key]["n_mols"] for _, key in series],
            [loading[key]["records"] for _, key in series])


def _posebusters_all_checks_panel(out, variant, series, grid, valid, norm):
    """One axes holding the whole table: block names, check names, headings, cells and note.

    THE GEOMETRY IS ARITHMETIC, NOT A NEGOTIATION. Every column is measured, the axes is
    placed in inches with add_axes, and the colourbar is given its own pre-placed cax -- so a
    data unit IS a cell and the text columns are exactly as wide as their text. The first
    version asked constrained layout to do it against a fixed aspect and a space-stealing
    colourbar, and on the two-arm variant the engine gave up and collapsed the axes to 0.7 in.
    """
    labels = [display(lab) if lab != REF_LABEL else "Reference" for lab, _ in series]
    ncol = len(labels)
    blocks = [(None, [PB_TAB_VALID])] + [(name, list(cs)) for name, cs in PB_TAB_GROUPS]
    names = [_posebusters_check_name(c) for _, cs in blocks for c in cs if c != PB_TAB_VALID]

    # Each text column is as wide as the WIDER of its content and its heading, which are set
    # at different sizes -- the heading in the method headings' type, the content in the
    # table's -- so both are measured rather than one assumed to win.
    name_w = max(_posebusters_text_in(names + [PB_TAB_VALID], PB_TAB_NAME_FS),
                 _posebusters_text_in([PB_TAB_HEADS[1]], PB_TAB_HEAD_FS))
    # Measured on the BROKEN names, and get_window_extent gives the widest line of a
    # multi-line text -- which is exactly the column's width.
    group_w = max(_posebusters_text_in([_posebusters_category_name(n) for n, _ in PB_TAB_GROUPS],
                                       PB_TAB_GROUP_FS),
                  _posebusters_text_in([PB_TAB_HEADS[0]], PB_TAB_HEAD_FS))
    # A column holds its number with air on both sides, and its heading has to clear its
    # NEIGHBOUR'S -- the heat map's rule, and here it is worth three inches of canvas. Two
    # neighbours each spend half their width towards the gutter between them, so what binds is
    # the widest adjacent PAIR: VoxBind(sigma=0.9) beside FuncBind asks for 1.13 in, where
    # VoxBind's own 1.32 would have set every one of the nine columns. A heading wider than
    # its column simply overhangs into white that belongs to nobody.
    heads = [_posebusters_text_in([lab], PB_TAB_HEAD_FS) for lab in labels]
    cell_w = max([_posebusters_text_in(["100.0"], PB_TAB_NUM_FS)]
                 + [(a + b) / 2 for a, b in zip(heads, heads[1:])]) + 2 * PB_TAB_PAD

    # Rows, top down, with a blank between blocks. y is in ROWS and grows downwards.
    placed, bounds, y = [], [], 0.0
    for name, checks in blocks:
        top = y
        for c in checks:
            placed.append((c, y))
            y += 1.0
        if name:
            placed.append(((name,), (top + y) / 2))
        bounds.append(y)
        y += PB_TAB_GAP
    y -= PB_TAB_GAP
    bounds.pop()                 # the last block ends at the table's edge, not at a division

    left_in = PB_TAB_INSET + group_w + PB_TAB_GUTTER + name_w + PB_TAB_PAD
    block_w = left_in + ncol * cell_w
    block_h = (y + PB_TAB_HEAD_H + PB_TAB_FOOT_H) * PB_TAB_ROW_H
    # The bar's strip is BESIDE the table, so it costs width and no height at all.
    bar_gap, bar_thick, bar_key = PB_TAB_CBAR
    fig_w = block_w + bar_gap + bar_thick + bar_key + 2 * PB_TAB_MARGIN
    fig_h = block_h + 2 * PB_TAB_MARGIN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([PB_TAB_MARGIN / fig_w, PB_TAB_MARGIN / fig_h,
                       block_w / fig_w, block_h / fig_h])
    ax.set_facecolor("white")
    # One x unit is one cell and one y unit is one row, by construction of the box above.
    ax.set_xlim(-left_in / cell_w, ncol)
    ax.set_ylim(y + PB_TAB_FOOT_H, -PB_TAB_HEAD_H)     # the first row at the top
    x_group = -(left_in - PB_TAB_INSET) / cell_w
    x_rule = -(PB_TAB_PAD + name_w + PB_TAB_GUTTER / 2) / cell_w
    # THE CHECK NAMES ARE LEFT-ALIGNED (2026-09-23, on request), so the column is addressed by
    # its left edge rather than by the cells it used to be set against.
    x_name = -(PB_TAB_PAD + name_w) / cell_w

    for j, lab in enumerate(labels):
        ax.text(j + 0.5, -0.30, lab, ha="center", va="bottom", fontsize=PB_TAB_HEAD_FS,
                color=INK)
    # Each over its own column and aligned with it -- the categories run from the left edge,
    # the check names are set against the cells they belong to.
    ax.text(x_group, -0.30, PB_TAB_HEADS[0], ha="left", va="bottom",
            fontsize=PB_TAB_HEAD_FS, color=INK)
    ax.text(x_name, -0.30, PB_TAB_HEADS[1], ha="left", va="bottom",
            fontsize=PB_TAB_HEAD_FS, color=INK)
    cmap = HEAT_CMAP
    for key, row_y in placed:
        if isinstance(key, tuple):                     # a block name, centred on its rows
            ax.text(x_group, row_y, _posebusters_category_name(key[0]), ha="left",
                    va="center", fontsize=PB_TAB_GROUP_FS, color=INK, linespacing=1.35)
            continue
        is_valid = key == PB_TAB_VALID
        # The summary row has no category of its own, so the two text columns merge and it is
        # written from the left edge of the first of them.
        ax.text(x_group if is_valid else x_name, row_y + 0.5,
                PB_TAB_TOTAL if is_valid else _posebusters_check_name(key),
                ha="left", va="center", fontsize=PB_TAB_NAME_FS, color=INK)
        flat = key in PB_TAB_BY_CONSTRUCTION
        for j, v in enumerate(valid if is_valid else grid[key]):
            face = PB_TAB_FLAT_FILL if flat else cmap(norm(v))
            ax.add_patch(plt.Rectangle((j, row_y), 1, 1, facecolor=face, edgecolor="white",
                                       linewidth=1.4))
            # Ink or white off the FILL's own luminance, never off the value, exactly as the
            # heat map decides it -- which is what keeps a grey cell's number readable too.
            r, g, b = matplotlib.colors.to_rgb(face)
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < PB_HEAT_WHITE_TEXT_BELOW
            ax.text(j + 0.5, row_y + 0.5, _posebusters_pct(v), ha="center", va="center",
                    fontsize=PB_TAB_NUM_FS, color="white" if dark else INK, zorder=3)
    # The summary row is an aggregate of the rest, not a 23rd check: a rule under it, down the
    # middle of the blank, exactly as the heat map rules off its reference row. And a rule
    # down the gutter, so a block of rows reads as one object rather than as a stray word.
    x_left = -left_in / cell_w
    # The frame: under the headings, under the summary row, and under the last row
    # (2026-09-23, on request -- the table used to end on white).
    ax.hlines([-PB_TAB_HEAD_RULE, bounds[0] + PB_TAB_GAP / 2, y], x_left, ncol,
              color=PB_TAB_RULE_INK, lw=PB_TAB_RULE_LW, zorder=5)
    ax.hlines([b + PB_TAB_GAP / 2 for b in bounds[1:]], x_left, ncol,
              color=PB_TAB_RULE_INK, lw=PB_TAB_DIVIDER_LW, zorder=2)
    # The column divider, in TWO segments, because the summary row's cell spans both columns
    # and a rule through a merged cell would un-merge it. Above: the headings, which are two
    # columns and have to look like it. Below: the checks, each under its own category.
    ax.vlines(x_rule, -PB_TAB_HEAD_H + 0.12, -PB_TAB_HEAD_RULE, color=PB_TAB_RULE_INK,
              lw=PB_TAB_DIVIDER_LW, zorder=2)
    ax.vlines(x_rule, bounds[0] + PB_TAB_GAP / 2, y, color=PB_TAB_RULE_INK,
              lw=PB_TAB_DIVIDER_LW, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Centred on the table's own height, not on the canvas: the margins are equal, so the two
    # are the same thing, and writing it against the block keeps the bar tied to the rows it
    # keys if a margin ever moves.
    bar_len = PB_TAB_CBAR_LEN * block_h
    cax = fig.add_axes([(PB_TAB_MARGIN + block_w + bar_gap) / fig_w,
                        (PB_TAB_MARGIN + (block_h - bar_len) / 2) / fig_h,
                        bar_thick / fig_w, bar_len / fig_h])
    heat_colorbar(fig, ax, norm, "% of molecules passing", cax=cax,
                  label_fs=PB_HEAT_CBAR_FS, tick_fs=PB_HEAT_CBAR_TICK_FS, cmap=cmap,
                  ticks=[50, 60, 70, 80, 90, 100])
    save(fig, out, f"pb_all_checks_{variant}")


@figure("fig-posebusters-all-checks",
        needs=("metrics.json (posebusters block)", "loading_checks.json"))
def draw_posebusters_all_checks(out):
    """All 22 PoseBusters checks, one row each, as a pass-rate table per method."""
    data, p79_rows, refrows = pose_data()
    use_style()
    fails = _posebusters_check_failures(data, p79_rows, refrows)
    loading = _posebusters_loading_rates()
    # THE RAMP'S LIGHT END AT 45, ITS DEEP END AT 100 (2026-09-23, on request) -- see
    # PB_TAB_VMIN for why this table floors at 45 where the group heat map floors at its worst
    # cell. The ramp itself is the shared one. Cells are still read against each other through
    # one norm, and both variants take it, so a number is the same colour in `core` as in `all`.
    norm = matplotlib.colors.Normalize(PB_TAB_VMIN, PB_HEAT_VMAX)
    rows = []
    for variant, arms in variants("v", data):
        series, grid, ns, ns_load = _posebusters_all_checks_grid(arms, fails, loading)
        valid = [100 * sum(bool(r["v"]) for r in rs) / max(1, sum(r["v"] is not None
                                                                  for r in rs))
                 for rs in [refrows] + [p79_rows[key] for _, key, _ in arms]]
        _posebusters_all_checks_panel(out, variant, series, grid, valid, norm)
        if variant != "all":
            continue
        group_of = {c: name for name, cs in PB_TAB_GROUPS for c in cs}
        print(f"\n  {'check':40s} " + " ".join(f"{lab[:9]:>9s}" for lab, _ in series))
        print(f"  {PB_TAB_VALID:40s} " + " ".join(f"{_posebusters_pct(v):>8s}%" for v in valid))
        for name, checks in PB_TAB_GROUPS:
            print(f"  {name}")
            for c in checks:
                print(f"    {_posebusters_check_name(c):38s} "
                      + " ".join(f"{_posebusters_pct(v):>8s}%" for v in grid[c])
                      + ("  (by construction)" if c in PB_TAB_BY_CONSTRUCTION else ""))
        live = [c for c in grid if c not in PB_TAB_BY_CONSTRUCTION and min(grid[c]) < 100]
        print(f"\n  {len(live)} of 22 checks ever fail here; {len(PB_TAB_BY_CONSTRUCTION)} "
              f"cannot (no cofactors and no waters in the pocket10 crop), and "
              f"{22 - len(live) - len(PB_TAB_BY_CONSTRUCTION)} could and never do")
        # The two denominators, side by side, because the loading rows have their own: what
        # the scoring run never scored is what those rows exist to cover.
        print(f"\n  {'arm':16s} {'scored':>8s} {'records':>8s} {'unscored':>9s}")
        for (lab, key), n in zip(series, ns):
            l = loading[key]
            print(f"  {lab:16s} {n:8,d} {l['records']:8,d} {l['unscored']:9,d}"
                  + ("  <-- PoseBusters timed out on these" if l["unscored"] else ""))
        for i, (lab, _) in enumerate(series):
            worst = sorted((grid[c][i], c) for c in live)[:2]
            print(f"    {lab:16s} worst: "
                  + ", ".join(f"{_posebusters_check_name(c)} {v:.1f}%" for v, c in worst))
        # n_molecules is the ROW's own denominator: the loading rows were measured over every
        # record, the rest over the molecules that were scored.
        rows = [[lab, ns[i], "", PB_TAB_VALID, False, round(valid[i], 3)]
                for i, (lab, _) in enumerate(series)]
        rows += [[lab, ns_load[i] if c in PB_TAB_LOADING else ns[i], group_of[c], c,
                  c in PB_TAB_BY_CONSTRUCTION, round(grid[c][i], 3)]
                 for i, (lab, _) in enumerate(series) for c in grid]
    write_csv(out, "posebusters_all_checks",
              ["method", "n_molecules", "group", "check", "cannot_fail_here", "pct_pass"],
              rows)


# ── SuCOS ───────────────────────────────────────────────────────────────────────
_PB_SUCOS_CACHE = {}


def _posebusters_sucos_rows():
    """{key: [{t, n, sucos}]}, from the per-molecule export.

    build_sucos.py computes these with `check_sucos(..., sucos_threshold=0.4)` -- exactly
    what gen.yml configures -- and caches them beside itself; it needs RDKit, posebusters
    and the `moleval` env, none of which drawing does. Reading its export is the same code
    path that script itself takes on a warm cache."""
    if not _PB_SUCOS_CACHE:
        for _, key, _ in ARMS:
            path = legacy("fig-posebusters", f"sucos_per_molecule_{key}.json")
            _PB_SUCOS_CACHE[key] = json.load(open(path))["molecules"]
    return _PB_SUCOS_CACHE


def _posebusters_sucos_p79():
    """The same, restricted to the 79-pocket like-for-like set."""
    p79 = set(P79)
    return {key: [r for r in rows if r["t"] in p79]
            for key, rows in _posebusters_sucos_rows().items()}


def _posebusters_sucos_standardized(rows, weights, f):
    """The arm's SuCOS WITHIN each 1-heavy-atom stratum, re-weighted by `weights`. SuCOS
    turns out to be nearly flat in size, so this barely moves -- which is itself worth
    recording, because every other metric in this section is size-confounded."""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["n"]].append(r["sucos"])
    num = den = 0.0
    for n, w in weights.items():
        if len(per.get(n, ())) >= 10:
            num += w * f(per[n]); den += w
    return num / den if den else float("nan")


def _posebusters_sucos_paired(a_rows, b_rows):
    """Per-pocket mean SuCOS, a - b, over the pockets both cover. Paired because the
    pockets differ from each other far more than the arms do: an unpaired comparison of
    two arms is mostly a comparison of which pockets each happened to cover."""
    def by_pocket(rows):
        d = collections.defaultdict(list)
        for r in rows:
            d[r["t"]].append(r["sucos"])
        return {t: float(np.mean(v)) for t, v in d.items()}
    a, b = by_pocket(a_rows), by_pocket(b_rows)
    ts = sorted(set(a) & set(b))
    diff = [a[t] - b[t] for t in ts]
    sd = st.stdev(diff) if len(diff) > 1 else 0.0
    half = 1.96 * sd / max(1, len(diff)) ** 0.5
    return {"n_pockets": len(ts), "mean_diff": round(float(np.mean(diff)), 4),
            "median_diff": round(st.median(diff), 4),
            "ci95": [round(float(np.mean(diff)) - half, 4),
                     round(float(np.mean(diff)) + half, 4)],
            "a_higher_in": sum(d > 0 for d in diff)}


@figure("fig-posebusters-sucos-ecdf", needs=("sucos_per_molecule_<arm>.json",))
def draw_posebusters_sucos_ecdf(out):
    """SuCOS against the pocket's crystal ligand, as a distribution, with gen's 0.4 mark."""
    p79_rows = _posebusters_sucos_p79()
    use_style()
    for variant, arms in variants():
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for lab, key, _ in arms:
            v = np.sort([r["sucos"] for r in p79_rows[key]])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=pale(lab),
                    lw=MODEL_LW, zorder=5, solid_capstyle="round")
        # gen.yml would call everything left of this line invalid.
        ax.axvline(PB_SUCOS_THRESHOLD, color=INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
        # Horizontal and high: rotated against the line it ran straight through the curves.
        ax.text(PB_SUCOS_THRESHOLD + 0.015, 0.97, f"gen threshold {PB_SUCOS_THRESHOLD}",
                ha="left", va="top", fontsize=11, color=INK)
        furniture(ax, ylabel="Cumulative share",
                  xlabel="SuCOS vs the pocket's crystal ligand", xloc=None)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        legend(ax, arm_handles(arms, include_ref=False, paint=pale), loc="upper left",
               fontsize=11.5)
        fit(fig, pad=0.5)
        save(fig, out, f"sucos_ecdf_{variant}")

    # One common size distribution -- every arm pooled -- for direct standardization.
    weights = collections.Counter(r["n"] for rows in p79_rows.values() for r in rows)
    print(f"\n  {'arm':16s} {'mean':>7s} {'median':>7s} {'q25':>7s} {'q75':>7s} "
          f"{'std mean':>9s} {'>=0.4':>8s} {'mols':>7s}")
    rates = {}
    for lab, key, _ in ARMS:
        v = [r["sucos"] for r in p79_rows[key]]
        over = 100 * sum(x >= PB_SUCOS_THRESHOLD for x in v) / len(v)
        rates[lab] = round(over / 100, 4)
        std = round(_posebusters_sucos_standardized(p79_rows[key], weights, np.mean), 4)
        print(f"  {lab:16s} {np.mean(v):7.3f} {st.median(v):7.3f} "
              f"{np.percentile(v, 25):7.3f} {np.percentile(v, 75):7.3f} "
              f"{std:9.3f} {over:7.1f}% {len(v):7d}")

    # The claim this figure exists to support, stated as a paired test rather than a
    # difference of two pooled means.
    print()
    for name, (a, b) in (("CoDE - VoxBind", ("ours_v1", "vanilla")),
                         ("CoDE - TargetDiff", ("ours_v1", "targetdiff")),
                         ("CoDE - DecompDiff", ("ours_v1", "decompdiff"))):
        d = _posebusters_sucos_paired(p79_rows[a], p79_rows[b])
        print(f"    {name:30s} {d['mean_diff']:+.4f} "
              f"(95% CI {d['ci95'][0]:+.4f}..{d['ci95'][1]:+.4f}), "
              f"higher in {d['a_higher_in']}/{d['n_pockets']} pockets")

    # What running config="gen" instead of "dock" would have cost, stated rather than
    # implied: `valid` there is all-must-pass INCLUDING sucos_within_threshold.
    try:
        pb = json.load(open(legacy("fig-posebusters", "posebusters_summary.json")))
        print(f"\n  {'arm':16s} {'dock valid':>11s} {'gen valid (upper bound)':>24s}")
        for lab, key, _ in ARMS:
            # An arm whose PoseBusters run has not finished is simply absent from that
            # summary; skip its row rather than dropping the whole table.
            if lab not in pb.get("arms", {}):
                continue
            dock = pb["arms"][lab]["p79"]["pb_valid_rate"]
            print(f"  {lab:16s} {100 * dock:10.1f}% {100 * dock * rates[lab]:23.1f}%")
        print("    (upper bound: the product assumes the two are independent; the true gen\n"
              "     rate is the share passing BOTH, which cannot exceed either factor)")
    except (OSError, KeyError):
        pass


@figure("fig-posebusters-sucos-per-atom", needs=("sucos_per_molecule_<arm>.json",))
def draw_posebusters_sucos_per_atom(out):
    """SuCOS against ligand size — mean and median, core and all."""
    p79_rows = _posebusters_sucos_p79()
    use_style()
    ranges = {}
    for variant, arms in variants():
        per = {key: by_size(p79_rows[key], "sucos") for _, key, _ in arms}
        xs = x_range(per, arms)
        for stat, f in (("mean", lambda v: float(np.mean(v))),
                        ("median", lambda v: float(np.median(v)))):
            fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            for lab, key, _ in arms:
                ax.plot(xs, model_curve(per[key], xs, f), color=pale(lab),
                        lw=MODEL_LW, zorder=5, solid_capstyle="round")
            ax.axhline(PB_SUCOS_THRESHOLD, color=INK, lw=1.2, ls=(0, (1, 2.6)), zorder=3)
            furniture(ax, ylabel=f"SuCOS {stat}", xlabel=X_LABEL,
                      xlim=(xs[0] - 0.6, xs[-1] + 0.6))
            ax.set_ylim(0, 1)
            legend(ax, arm_handles(arms, include_ref=False, paint=pale), loc="upper left",
                   fontsize=11.5)
            fit(fig, pad=0.5)
            save(fig, out, f"sucos_per_atom_{stat}_{variant}")
        ranges[variant] = xs
    print("  per-atom x range: "
          + " · ".join(f"{v} {xs[0]}-{xs[-1]}" for v, xs in ranges.items()))

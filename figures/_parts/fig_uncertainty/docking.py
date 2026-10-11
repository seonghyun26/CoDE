"""fig-uncertainty: the 7 figures drawn into figures/fig-uncertainty/.

    python figures/draw.py uncertainty

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-uncertainty-bfactor-delta                -> fig-uncertainty/docking/
    fig-uncertainty-occupancy-delta              -> fig-uncertainty/docking/
    fig-uncertainty-correlation-matrix           -> fig-uncertainty/docking/
    fig-uncertainty-bfactor-absolute             -> fig-uncertainty/docking/
    fig-uncertainty-affinity-tercile             -> fig-uncertainty/exploratory/affinity-initial/
    fig-uncertainty-affinity-scatter             -> fig-uncertainty/exploratory/affinity-initial/
    fig-uncertainty-affinity-matrix              -> fig-uncertainty/exploratory/affinity-initial/
"""
from __future__ import annotations

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

from draw import (
    AXIS, AXIS_LW, DASH, DATA_ALPHA, HEAT_CMAP, INK, LEGEND_EDGE, METHOD_ORDER,
    MODEL_LW, REF_COLOR, REF_LW, REPO, SOLID, dash, display, figure, fit, furniture,
    heat_colorbar, ordered_methods, pale, save, use_style, write_csv,
)

# ════════════════════════════════════════════════════════════════════════════════
# fig-uncertainty-bfactor-delta · fig-uncertainty-occupancy-delta
# fig-uncertainty-correlation-matrix · fig-uncertainty-bfactor-absolute
# ════════════════════════════════════════════════════════════════════════════════
# THE QUESTION. CoDE and CoE are conditioned on an X-ray electron-density crop of the
# pocket. If that conditioning is what buys them their gain over VoxBind, the gain should be
# LARGER where the crystal is well ordered and the map is sharp, and smaller where it is
# smeared -- because a smeared map is a conditioning input carrying less information. The
# deposited per-atom measure of that smearing is the B-factor, and the deposited measure of
# how much of the atom is actually there is the occupancy. So: per pocket, the gain against
# the pocket's own crystallography.
#
# WHAT THESE FIGURES DRAW, AND WHY THE CONTROL PANEL IS NOT OPTIONAL. A descriptor can
# correlate with an arm's ABSOLUTE Vina on a pocket simply because it describes how easy the
# pocket is -- open, floppy sites score better for everyone. That is a property of
# CrossDocked, not of a model, and it cancels when the baseline is subtracted. So every
# delta figure here has an absolute counterpart drawn over all nine arms, and a claim is
# only about the conditioning if the delta moves where the absolute is flat across arms.
#
# NOTHING HERE IS COMPUTED HERE. The descriptors, the per-pocket Vina, the Spearman rho, its
# p, its BH q and the partial rho all come out of
# task2-ligand_generation/scripts/95_eval_pocket_uncertainty.sh; this file positions them. The one
# exception is the least-squares line drawn through each cloud, which is a drawing decision
# (it shows the slope the rho is reporting) and is fitted on the points as plotted.
UNC_JSON = REPO / "results" / "task2-drugdesign" / "_shared" / "260916_pocket_uncertainty" \
    / "pocket_uncertainty.json"
UNC_BUILDER = "bash task2-ligand_generation/scripts/95_eval_pocket_uncertainty.sh"

# The arms, in the section order every other `all` figure here uses, so the drawn order and
# the legend agree across the paper. VoxBind is the baseline and therefore has no delta.
UNC_BASELINE = "VoxBind"
UNC_ALL = ordered_methods([m for m in METHOD_ORDER if m != UNC_BASELINE] + ["CoE"])
# `core` is the comparison this section makes: our two density-conditioned arms, in the
# order CoDE then CoE, because CoDE is the method and CoE its coords-only ablation.
UNC_CORE = ["CoDE", "CoE"]

UNC_METRIC = "vina_dock_median"
UNC_METRIC_LABEL = "Vina Dock, pocket median"
# The descriptor each scatter family is keyed on. `site` rather than `pocket` scope: the
# 10 A crop averages in loops the ligand never touches, and the question is about the wall
# the ligand actually meets. The pocket-scope column is in the matrix and the CSV.
UNC_B = "b_site_mean"
UNC_OCC = ["occ_site_min", "occ_site_frac_altloc", "occ_site_frac_partial"]

# GEOMETRY IN INCHES PER PANEL, not one figsize per figure. These families are drawn at
# 2, 3, 4 and 8 panels, and a shared figsize would give the eight-panel version panels a
# third the width of the two-panel one -- the same cloud, read at two scales, in one folder.
# So a panel is always UNC_PANEL_W x UNC_PANEL_H and the figure is however big that makes it.
UNC_PANEL_W, UNC_PANEL_H = 3.55, 3.45
UNC_LABEL_W = 1.25                    # the y axis label and its ticks, once per figure
UNC_ROW_LEAD = 0.55                   # the panel titles

UNC_POINT, UNC_POINT_EDGE = 44, 0.9
UNC_FIT_LW = 2.0
UNC_ZERO_LW = 1.45
# The stats box: the rho the builder computed, its BH q over the 15 descriptors of that
# outcome, and n. Never a bare p -- there are 15 descriptors x 5 outcomes x 8 arms in the
# JSON, so at that many tests one p under 0.05 is the number a pure null predicts.
UNC_BOX = dict(boxstyle="square,pad=0.34", facecolor="white", edgecolor=LEGEND_EDGE,
               linewidth=AXIS_LW)


def _unc_load():
    """The builder's JSON, or a message naming the command that writes it."""
    if not UNC_JSON.exists():
        raise FileNotFoundError(
            f"missing input {UNC_JSON}\n"
            f"    the crystallographic descriptors are not built yet — run:\n"
            f"        {UNC_BUILDER}")
    with open(UNC_JSON, encoding="utf-8") as handle:
        return json.load(handle)


def _unc_labels(data):
    """{descriptor key: (axis label, matrix row label)} as the builder named them."""
    return {key: (label, short) for key, label, short in data["descriptors"]}


def _unc_stat(data, half, arm, outcome, descriptor):
    """One correlation entry, or None where the arm or the descriptor has no data."""
    for entry in data["correlations"].get(half, {}).get(arm, {}).get(outcome, ()):
        if entry["descriptor"] == descriptor:
            return entry
    return None


def _unc_values(data, half):
    """{arm: {target: row}} for whichever half a figure is drawing.

    `deltas` is its own block; `absolute` is simply the per-arm table, since an arm's
    absolute value is what a delta is taken from. Naming the two halves the same way here
    is what lets one draw function serve both."""
    return data["deltas"] if half == "deltas" else data["arms"]


def _unc_xy(data, half, arm, descriptor, outcome=None):
    """(targets, x, y) over the pockets where BOTH the descriptor and the value exist."""
    outcome = outcome or UNC_METRIC
    source = _unc_values(data, half)[arm]
    targets, xs, ys = [], [], []
    for target in data["targets"]:
        x = data["pockets"].get(target, {}).get(descriptor)
        y = source.get(target, {}).get(outcome)
        if x is None or y is None:
            continue
        targets.append(target)
        xs.append(float(x))
        ys.append(float(y))
    return targets, np.asarray(xs), np.asarray(ys)


def _unc_cloud(ax, x, y, label, *, stat=None, fit=True, point=UNC_POINT, corner="lower left",
               bound=False):
    """One arm's pockets, its least-squares line, and the coefficient in the corner.

    The line is drawn even where the correlation is nothing, because a flat line IS the
    result in that case and an empty panel would read as missing data."""
    paint = pale(label)
    ax.axhline(0.0, color=AXIS, lw=UNC_ZERO_LW, ls=DASH, zorder=2)
    ax.scatter(x, y, s=point, facecolor=paint, edgecolor="white",
               linewidth=UNC_POINT_EDGE, alpha=DATA_ALPHA, zorder=4)
    if fit and len(x) >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        span = np.array([x.min(), x.max()])
        ax.plot(span, slope * span + intercept, color=paint, lw=UNC_FIT_LW, ls="-",
                zorder=5, alpha=DATA_ALPHA)
    if stat and stat.get("spearman") is not None:
        # WHICH CORNER is the caller's, not this helper's: the clouds differ in shape between
        # the B-factor figures (spread across x, empty at the bottom left) and the occupancy
        # one (piled against x = 0, empty at the top right), and a box fixed in one place
        # lands on the data in the other.
        va, ha = corner.split()
        # A NULL NEEDS A BOUND, not just a q. "rho = 0.02, q = 0.90" says the data does not
        # show an effect; it does not say the effect is SMALL, and a reader is right to ask
        # whether a real 1 kcal/mol trend is hiding under 79 pockets. The third line is the
        # bootstrap CI on the fitted slope, multiplied by the observed range of the
        # descriptor -- the largest end-to-end effect these data still allow, in kcal/mol.
        extra = ""
        if bound and stat.get("effect_lo") is not None:
            # ONE line, not three: the box sits inside the data area, and a three-line
            # version of this covered half the cloud it is annotating.
            extra = (f"\nrange effect {stat['effect']:+.2f} "
                     f"[{stat['effect_lo']:+.2f}, {stat['effect_hi']:+.2f}]")
        ax.text(0.035 if ha == "left" else 0.965, 0.045 if va == "lower" else 0.955,
                f"ρ = {stat['spearman']:+.2f}   q = {stat['q']:.2f}   n = {stat['n']}" + extra,
                transform=ax.transAxes, fontsize=10.5 if extra else 12.0, color=INK,
                va="bottom" if va == "lower" else "top", ha=ha,
                linespacing=1.45, bbox=UNC_BOX, zorder=8)


# WHAT THE Y AXIS SAYS, ONCE, FOR EVERY PANEL IN THE FAMILY. Vina is negative-is-better, so
# the GAIN is BELOW the zero rule -- the opposite of every other figure in this paper, and
# the single thing most likely to be misread here. It is spelled out on the axis rather than
# left to a caption, and the arm is named by the panel TITLE, which is what lets one label
# serve a row of panels instead of repeating a subtraction four times across it.
# KEPT SHORT ON PURPOSE. A rotated y label longer than the panel is tall overruns the figure
# at BOTH ends and is then silently sliced in the raster -- the full sentence
# ("Vina Dock, pocket median, delta vs VoxBind_sigma=0.9") measures 4.5 in against a 2.6 in
# axis. The metric's aggregation is in the figure's own docstring and in the CSV header; the
# axis carries the quantity, the unit and the direction, which is what has to be read here.
UNC_YLABEL = "Δ Vina Dock (kcal/mol)\nvs VoxBind · lower = gain"


def _unc_frame(cols, rows, *, sharex=True, sharey=True, panel_w=None):
    """A cols x rows grid of equally sized panels, plus room for one y label and the titles."""
    panel_w = panel_w or UNC_PANEL_W
    fig, axes = plt.subplots(rows, cols, sharex=sharex, sharey=sharey,
                             figsize=(UNC_LABEL_W + cols * panel_w,
                                      UNC_ROW_LEAD + rows * UNC_PANEL_H))
    return fig, np.atleast_1d(axes).ravel()


def _unc_edges(flat, n, cols, *, xlabel):
    """Label only the outer panels: the left column carries the y label, the bottom row the
    x label. Repeating either inside the grid costs plot area and says nothing new."""
    for i in range(0, n, cols):
        flat[i].set_ylabel(UNC_YLABEL, fontsize=12.0, labelpad=9)
    for i in range(max(0, n - cols), n):
        flat[i].set_xlabel(xlabel, fontsize=13.0, labelpad=8)
    for ax in flat[n:]:
        ax.set_visible(False)


@figure("fig-uncertainty-bfactor-delta", needs=("_shared/260916_pocket_uncertainty/",), folder="fig-uncertainty/docking")
def draw_uncertainty_bfactor_delta(out):
    """Gain over VoxBind against the pocket's B-factor, per pocket (core and all arms)."""
    use_style()
    data = _unc_load()
    labels = _unc_labels(data)
    xlabel = labels[UNC_B][0]

    for variant, arms, cols, rows in (("core", UNC_CORE, 2, 1),
                                      ("all", UNC_ALL, 4, 2)):
        drawn = [a for a in arms if a in data["correlations"]["deltas"]]
        if not drawn:
            print(f"    bfactor_delta_{variant} skipped: no arm has a delta")
            continue
        cols = min(cols, len(drawn))
        rows = -(-len(drawn) // cols)
        fig, flat = _unc_frame(cols, rows)
        for ax, arm in zip(flat, drawn):
            _, x, y = _unc_xy(data, "deltas", arm, UNC_B)
            _unc_cloud(ax, x, y, arm,
                       stat=_unc_stat(data, "deltas", arm, UNC_METRIC, UNC_B),
                       point=UNC_POINT if variant == "core" else UNC_POINT * 0.62,
                       bound=variant == "core")
            furniture(ax, xloc=20)
            ax.set_title(display(arm), fontsize=14.5, color=INK, pad=7)
        _unc_edges(flat, len(drawn), cols, xlabel=xlabel)
        fit(fig, w_pad=1.1, h_pad=1.5)
        save(fig, out, f"bfactor_delta_dock_{variant}")
        print(f"    bfactor_delta_dock_{variant}: {len(drawn)} arms, "
              f"{len(data['targets'])} pockets")

    # The per-pocket table behind every figure in this folder, written once and from the
    # `all` pass so it carries every arm rather than the two the core panel drew.
    descriptors = [key for key, _, _ in data["descriptors"]]
    header = (["target", "pdb_id", "chain", "cross_docked"] + descriptors
              + [f"{arm}_{UNC_METRIC}" for arm in [UNC_BASELINE] + UNC_ALL
                 if arm in data["arms"]]
              + [f"delta_{arm}_{UNC_METRIC}" for arm in UNC_ALL if arm in data["deltas"]])
    rows_out = []
    for target in data["targets"]:
        pocket = data["pockets"][target]
        row = [target, pocket["pdb_id"], pocket["chain"], int(pocket["cross_docked"])]
        row += [pocket.get(key) for key in descriptors]
        row += [data["arms"][arm].get(target, {}).get(UNC_METRIC)
                for arm in [UNC_BASELINE] + UNC_ALL if arm in data["arms"]]
        row += [data["deltas"][arm].get(target, {}).get(UNC_METRIC)
                for arm in UNC_ALL if arm in data["deltas"]]
        rows_out.append(row)
    write_csv(out, "pocket_uncertainty", header, rows_out)


@figure("fig-uncertainty-occupancy-delta", needs=("_shared/260916_pocket_uncertainty/",), folder="fig-uncertainty/docking")
def draw_uncertainty_occupancy_delta(out):
    """Gain over VoxBind against pocket occupancy — and how few pockets can answer it."""
    use_style()
    data = _unc_load()
    labels = _unc_labels(data)
    drawn = [a for a in UNC_CORE if a in data["correlations"]["deltas"]]

    # THE LEFT PANEL IS THE POINT. Occupancy on this test set is very nearly a constant:
    # almost every deposited site atom is fully occupied and single-conformer, so the
    # scatters to its right are a handful of pockets moving against a wall of tied ones, and
    # a rho computed on that is not a measurement of anything. Drawing the census first is
    # what stops the two panels beside it from being over-read.
    fig, flat = _unc_frame(1 + len(drawn), 1, sharex=False, sharey=False)
    census = flat[0]
    counts, names = [], []
    for key in UNC_OCC:
        values = [data["pockets"][t].get(key) for t in data["targets"]]
        values = [v for v in values if v is not None]
        modal = max(set(values), key=values.count)
        counts.append(sum(1 for v in values if v != modal))
        names.append(labels[key][1])
    # Grey, not an arm's colour: this panel counts POCKETS, a property of CrossDocked, and
    # painting it blue would read as "CoDE" beside two panels where blue means exactly that.
    census.bar(range(len(counts)), counts, color=REF_COLOR, width=0.62, zorder=3)
    for i, c in enumerate(counts):
        census.text(i, c + 0.9, str(c), ha="center", va="bottom", fontsize=12.5, color=INK)
    census.set_xticks(range(len(counts)))
    census.set_xticklabels(names, fontsize=12.0, rotation=18, ha="right")
    furniture(census, xloc=None)
    census.set_ylim(0, max(len(data["targets"]) * 0.32, max(counts) * 1.45))
    census.set_ylabel(f"Pockets differing from\nthe modal value (of {len(data['targets'])})",
                      fontsize=12.5, labelpad=9)
    census.set_title("What occupancy can separate", fontsize=14.5, color=INK, pad=7)

    key = UNC_OCC[1]
    for ax, arm in zip(flat[1:], drawn):
        _, x, y = _unc_xy(data, "deltas", arm, key)
        _unc_cloud(ax, x, y, arm, stat=_unc_stat(data, "deltas", arm, UNC_METRIC, key),
                   corner="upper right", bound=True)
        furniture(ax, xloc=None)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(labels[key][0].replace("Site atoms with an ", "Site atoms with an\n"),
                      fontsize=13.0, labelpad=8)
        ax.set_title(display(arm), fontsize=14.5, color=INK, pad=7)
    # The scatter panels share a y range so the two arms are read against each other, but the
    # census beside them does NOT -- it counts pockets, on its own scale. That is why this
    # figure builds its panels unshared and matches the range by hand.
    lo = min(ax.get_ylim()[0] for ax in flat[1:])
    hi = max(ax.get_ylim()[1] for ax in flat[1:])
    for i, ax in enumerate(flat[1:]):
        ax.set_ylim(lo, hi)
        if i:
            ax.tick_params(labelleft=False)
    flat[1].set_ylabel(UNC_YLABEL, fontsize=12.0, labelpad=9)
    fit(fig, w_pad=1.35)
    save(fig, out, "occupancy_delta_dock_core")
    print(f"    occupancy_delta_dock_core: {names} informative in {counts} pockets")


@figure("fig-uncertainty-correlation-matrix", needs=("_shared/260916_pocket_uncertainty/",), folder="fig-uncertainty/docking")
def draw_uncertainty_correlation_matrix(out):
    """Every descriptor against every arm — as a gain over VoxBind, and in absolute terms."""
    use_style()
    data = _unc_load()
    labels = _unc_labels(data)
    descriptors = [key for key, _, _ in data["descriptors"]]

    # |rho| ON THE HOUSE RAMP, SIGN AS A PRINTED NUMBER. The ramp is one hue, light to dark,
    # and means "more of the quantity the bar is named after" -- so it keys STRENGTH, which
    # is the quantity a reader scans this matrix for. A signed scale would need a diverging
    # ramp, and this paper does not have one; introducing it for one figure would make its
    # white middle mean "rho = 0" here and nothing anywhere else.
    # The two matrices are on DIFFERENT scales on purpose. Nothing in the delta matrix passes
    # |rho| 0.6, and drawing it on the absolute matrix's 0-to-1 ramp would render the whole
    # thing near-white and unreadable; the absolute matrix reaches 0.98 and needs the full
    # range. Each says its own vmax on its colourbar, which is where a reader checks.
    for variant, half, arms, vmax in (("delta", "deltas", UNC_ALL, 0.6),
                                      ("absolute", "absolute",
                                       [UNC_BASELINE] + UNC_ALL, 1.0)):
        drawn = [a for a in arms if data["correlations"][half].get(a)]
        if not drawn:
            print(f"    correlation_matrix_{variant} skipped: nothing in {half}")
            continue
        grid = np.full((len(descriptors), len(drawn)), np.nan)
        marks = {}
        for j, arm in enumerate(drawn):
            for i, key in enumerate(descriptors):
                entry = _unc_stat(data, half, arm, UNC_METRIC, key)
                if entry and entry.get("spearman") is not None:
                    grid[i, j] = entry["spearman"]
                    marks[(i, j)] = entry
        fig, ax = plt.subplots(figsize=(1.02 * len(drawn) + 4.6,
                                        0.46 * len(descriptors) + 2.05))
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
        ax.imshow(np.abs(grid), cmap=HEAT_CMAP, norm=norm, aspect="auto")
        for (i, j), entry in marks.items():
            strong = abs(grid[i, j]) > 0.62 * vmax
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=11.0,
                    color="white" if strong else INK,
                    # Bold marks a coefficient that survives Benjamini-Hochberg over the
                    # 15 descriptors of this column. Nothing else in the matrix is evidence.
                    fontweight="bold" if (entry.get("q") or 1) < 0.05 else "normal")
        ax.set_xticks(range(len(drawn)))
        ax.set_xticklabels([display(a) for a in drawn], fontsize=12.0, rotation=32,
                           ha="right")
        ax.set_yticks(range(len(descriptors)))
        ax.set_yticklabels([labels[k][1] for k in descriptors], fontsize=12.0)
        ax.tick_params(length=0, colors=AXIS, pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(len(drawn) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(descriptors) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=2.1)
        ax.grid(which="major", visible=False)
        ax.set_title("Gain over " + display(UNC_BASELINE) if variant == "delta"
                     else "Absolute " + UNC_METRIC_LABEL,
                     fontsize=14.0, color=INK, pad=9)
        heat_colorbar(fig, ax, norm, "|Spearman ρ| with " + UNC_METRIC_LABEL,
                      label_fs=12.5, tick_fs=11.5,
                      extend="max" if vmax < 1.0 else "neither")
        fit(fig)
        save(fig, out, f"correlation_matrix_{variant}")
        print(f"    correlation_matrix_{variant}: {len(descriptors)} descriptors "
              f"x {len(drawn)} arms")

    # Every coefficient in the JSON, not just the one metric the matrices draw — five
    # outcomes per arm, so the ligand-efficiency column (where a size effect shows up as
    # itself) can be read beside the raw one.
    header = ["half", "arm", "outcome", "descriptor", "n", "n_informative",
              "spearman", "p", "q", "partial_spearman", "pearson", "pearson_p",
              # The effect columns are what turn a null into a bound: the fitted slope, its
              # bootstrap CI, and the same three numbers multiplied by the descriptor's
              # observed range -- the largest end-to-end effect these 79 pockets still allow.
              "slope", "slope_lo", "slope_hi", "span", "effect", "effect_lo", "effect_hi"]
    rows_out = []
    for half in ("deltas", "absolute"):
        for arm in sorted(data["correlations"][half]):
            for outcome, _ in data["outcomes"]:
                for entry in data["correlations"][half][arm].get(outcome, ()):
                    rows_out.append([half, arm, outcome, entry["descriptor"], entry["n"],
                                     entry.get("n_informative"), entry.get("spearman"),
                                     entry.get("p"), entry.get("q"),
                                     entry.get("partial_spearman"), entry.get("pearson"),
                                     entry.get("pearson_p")]
                                    + [entry.get(k) for k in
                                       ("slope", "slope_lo", "slope_hi", "span",
                                        "effect", "effect_lo", "effect_hi")])
    write_csv(out, "correlations", header, rows_out)


@figure("fig-uncertainty-bfactor-absolute", needs=("_shared/260916_pocket_uncertainty/",), folder="fig-uncertainty/docking")
def draw_uncertainty_bfactor_absolute(out):
    """Why the gain shows no B-factor effect: every arm has the SAME weak slope, so it
    cancels in the subtraction."""
    use_style()
    data = _unc_load()
    labels = _unc_labels(data)
    drawn = [a for a in [UNC_BASELINE] + UNC_ALL if a in data["arms"]]

    fig, flat = _unc_frame(2, 1, sharex=False, sharey=False, panel_w=3.9)
    lines, slopes = flat[0], flat[1]

    # LEFT: every arm's least-squares line on one axis, and NO clouds -- nine overlapping
    # clouds of 79 points is a smear, and the shape being shown is that the nine lines are
    # parallel. NO LEGEND either: the right panel is the same nine arms as labelled, coloured
    # bars, so a key here would be a second copy of one that is already in the figure.
    for arm in drawn:
        _, x, y = _unc_xy(data, "absolute", arm, UNC_B)
        if len(x) < 3 or np.ptp(x) == 0:
            continue
        focus = arm in UNC_CORE
        slope, intercept = np.polyfit(x, y, 1)
        span = np.array([x.min(), x.max()])
        lines.plot(span, slope * span + intercept, color=pale(arm),
                   lw=MODEL_LW if focus else REF_LW, ls=SOLID if focus else dash(arm),
                   zorder=5 if focus else 4)
    furniture(lines, xloc=20)
    lines.set_xlabel(labels[UNC_B][0], fontsize=13.0, labelpad=8)
    lines.set_ylabel(f"{UNC_METRIC_LABEL}\n(kcal/mol)", fontsize=12.5, labelpad=9)
    lines.set_title("Every arm, absolute", fontsize=14.5, color=INK, pad=7)

    # RIGHT: the same nine slopes as bars, which is the sentence the left panel is making --
    # they are the same number. A difference between two of these bars is the only thing
    # that could put a B-factor effect into a delta.
    order, values = [], []
    for arm in drawn:
        entry = _unc_stat(data, "absolute", arm, UNC_METRIC, UNC_B)
        if entry and entry.get("spearman") is not None:
            order.append(arm)
            values.append(entry["spearman"])
    slopes.barh(range(len(order)), values, color=[pale(a) for a in order], height=0.66,
                zorder=3)
    slopes.axvline(0.0, color=AXIS, lw=UNC_ZERO_LW, zorder=4)
    slopes.set_yticks(range(len(order)))
    slopes.set_yticklabels([display(a) for a in order], fontsize=12.0)
    slopes.invert_yaxis()
    furniture(slopes, xloc=None)
    slopes.xaxis.set_major_locator(MultipleLocator(0.1))
    slopes.set_xlabel(f"Spearman ρ with {labels[UNC_B][1]}", fontsize=13.0, labelpad=8)
    slopes.set_title("…and the same weak slope", fontsize=14.5, color=INK, pad=7)
    fit(fig, w_pad=1.3)
    save(fig, out, "bfactor_absolute_dock_all")
    print(f"    bfactor_absolute_dock_all: rho from {min(values):+.3f} to {max(values):+.3f} "
          f"over {len(order)} arms")
    write_csv(out, "bfactor_absolute_slopes",
              ["arm", "spearman_rho", "p", "q", "n"],
              [[a, _unc_stat(data, "absolute", a, UNC_METRIC, UNC_B)["spearman"],
                _unc_stat(data, "absolute", a, UNC_METRIC, UNC_B)["p"],
                _unc_stat(data, "absolute", a, UNC_METRIC, UNC_B)["q"],
                _unc_stat(data, "absolute", a, UNC_METRIC, UNC_B)["n"]] for a in order])


# ════════════════════════════════════════════════════════════════════════════════
# fig-uncertainty-affinity-tercile · fig-uncertainty-affinity-scatter
# fig-uncertainty-affinity-matrix
# ════════════════════════════════════════════════════════════════════════════════
# THE SAME QUESTION, WHERE THERE IS ENOUGH DATA TO ANSWER IT. The figures above ask whether the
# GENERATION gain tracks the pocket's crystallography and answer on 79 pockets, one number per
# pocket. The binding-affinity probe predicts one pK per COMPLEX, so every complex carries its
# own error and the difference between two encoders is defined per complex -- 779 paired points
# instead of 79, on the same two input modalities:
#
#     C       coords-only  -- the regression-task twin of the CoE arm
#     C+D+G   + X-ray density and its gradient -- the twin of CoDE
#
# Δ = |error(C+D+G)| − |error(C)| per complex, so NEGATIVE means density helped that complex.
#
# WHY THERE IS A TERCILE FIGURE AND NOT ONLY A SCATTER. The effect here is not monotone: it is
# flat over two thirds of the B-factor range and then turns on, which Spearman barely sees
# (ρ = −0.06) and a permutation test on the top-vs-rest contrast rejects at p = 0.007. A
# scatter of 779 points with a near-flat fit through it would show the ρ and hide the regime,
# so both are drawn and the tercile figure is the one that carries the result.
UNCA_JSON = REPO / "results" / "task1-affinity" / "_shared" / "260916_affinity_uncertainty" \
    / "affinity_uncertainty.json"
UNCA_BUILDER = "bash figures/_parts/fig_uncertainty/96_eval_affinity_uncertainty.sh"
UNCA_ARM = "C+D+G"
UNCA_BASELINE = "C"
# The palette is keyed by GENERATION arm, and these two encoders are those two arms' input
# modalities -- so C takes CoE's teal and C+D+G takes CoDE's blue, and a reader moving between
# the two halves of this folder sees one colour mean one thing. Mapped here rather than as a
# global ALIAS because "C" is too short a key to put in a table shared by every figure.
UNCA_PAINT = {"C": "CoE", "C+D+G": "CoDE"}
UNCA_METRIC = "abs_error"
UNCA_YLABEL = "Δ |pK error| (C+D+G − C)\nnegative = density helped"
# The two descriptors the scatter and the tercile figures are keyed on: the interface B-factor
# (contact-weighted, 1/d²) and the ligand's real-space correlation with its own density.
UNCA_KEYS = ["b_contact_invd2", "lig_rscc"]


def _unca_load():
    if not UNCA_JSON.exists():
        raise FileNotFoundError(
            f"missing input {UNCA_JSON}\n"
            f"    the affinity-probe predictions and descriptors are not built yet — run:\n"
            f"        {UNCA_BUILDER}")
    with open(UNCA_JSON, encoding="utf-8") as handle:
        return json.load(handle)


def _unca_color(label):
    return pale(UNCA_PAINT.get(label, label))


def _unca_xy(data, key, outcome=UNCA_METRIC):
    rows = data["deltas"][UNCA_ARM]
    xs, ys = [], []
    for pid in data["targets"]:
        x = data["complexes"].get(pid, {}).get(key)
        y = rows.get(pid, {}).get(outcome)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


def _unca_stat(data, key, outcome=UNCA_METRIC, half="deltas", arm=UNCA_ARM):
    for entry in data["correlations"][half][arm][outcome]:
        if entry["descriptor"] == key:
            return entry
    return None


@figure("fig-uncertainty-affinity-tercile", needs=("task1-affinity/_shared/260916_affinity_uncertainty/",), folder="fig-uncertainty/exploratory/affinity-initial")
def draw_uncertainty_affinity_tercile(out):
    """Where density helps the affinity probe: Δ|pK error| by tercile of the descriptor."""
    use_style()
    data = _unca_load()
    labels = {k: (lab, short) for k, lab, short in data["descriptors"]}
    terciles = data["terciles"][UNCA_ARM][UNCA_METRIC]
    drawn = [k for k in UNCA_KEYS if terciles.get(k) and terciles[k].get("bins")]

    fig, flat = _unc_frame(len(drawn), 1, sharex=False, sharey=True, panel_w=3.95)
    for ax, key in zip(flat, drawn):
        block = terciles[key]
        bins = [b for b in block["bins"] if b]
        centres = np.arange(len(bins))
        means = [b["mean"] for b in bins]
        # Asymmetric error bars from the bootstrap percentiles, NOT mean ± sd: the CI is what
        # was computed and it is not symmetric, and drawing an sd would claim a normal
        # sampling distribution nothing here established.
        err = np.array([[b["mean"] - b["ci_lo"] for b in bins],
                        [b["ci_hi"] - b["mean"] for b in bins]])
        paint = _unca_color(UNCA_ARM)
        ax.bar(centres, means, color=paint, width=0.62, zorder=3)
        ax.errorbar(centres, means, yerr=err, fmt="none", ecolor=INK, elinewidth=AXIS_LW,
                    capsize=4, capthick=AXIS_LW, zorder=5)
        ax.axhline(0.0, color=AXIS, lw=UNC_ZERO_LW, zorder=4)
        ax.set_xticks(centres)
        ax.set_xticklabels([f"{b['lo_edge']:.3g}–{b['hi_edge']:.3g}\nn={b['n']}" for b in bins],
                           fontsize=10.0)
        furniture(ax, xloc=None)
        ax.set_xlabel(labels[key][0], fontsize=13.0, labelpad=8)
        ax.set_title(labels[key][1], fontsize=14.5, color=INK, pad=7)
        contrast = block.get("contrast")
        if contrast:
            # The pre-specified contrast, printed as the permutation p rather than a q: it is
            # ONE test chosen before looking, not one of the fifteen the matrix screens.
            ax.text(0.5, 0.045,
                    f"top − rest  {contrast['top_minus_rest']:+.3f} "
                    f"[{contrast['ci_lo']:+.3f}, {contrast['ci_hi']:+.3f}]\n"
                    f"permutation p = {contrast['p_permutation']:.3f}",
                    transform=ax.transAxes, fontsize=10.5, color=INK, ha="center",
                    va="bottom", linespacing=1.45, bbox=UNC_BOX, zorder=8)
    flat[0].set_ylabel(UNCA_YLABEL, fontsize=12.0, labelpad=9)
    fit(fig, w_pad=1.35)
    save(fig, out, "affinity_tercile_abs_error")
    print(f"    affinity_tercile_abs_error: {len(drawn)} descriptors, "
          f"{len(data['targets'])} complexes")

    header = ["descriptor", "tercile", "lo_edge", "hi_edge", "n", "mean_delta", "ci_lo", "ci_hi",
              "top_minus_rest", "contrast_ci_lo", "contrast_ci_hi", "p_permutation"]
    rows_out = []
    for key, block in terciles.items():
        if not block or not block.get("bins"):
            continue
        contrast = block.get("contrast") or {}
        for i, b in enumerate(block["bins"]):
            if not b:
                continue
            rows_out.append([key, i + 1, b["lo_edge"], b["hi_edge"], b["n"], b["mean"],
                             b["ci_lo"], b["ci_hi"],
                             contrast.get("top_minus_rest"), contrast.get("ci_lo"),
                             contrast.get("ci_hi"), contrast.get("p_permutation")])
    write_csv(out, "affinity_terciles", header, rows_out)


@figure("fig-uncertainty-affinity-scatter", needs=("task1-affinity/_shared/260916_affinity_uncertainty/",), folder="fig-uncertainty/exploratory/affinity-initial")
def draw_uncertainty_affinity_scatter(out):
    """Per-complex Δ|pK error| against the pocket's B-factor and the ligand's RSCC."""
    use_style()
    data = _unca_load()
    labels = {k: (lab, short) for k, lab, short in data["descriptors"]}

    fig, flat = _unc_frame(len(UNCA_KEYS), 1, sharex=False, sharey=True, panel_w=3.95)
    for ax, key in zip(flat, UNCA_KEYS):
        x, y = _unca_xy(data, key)
        # A THIRD of the point size the 79-pocket figures use, and half the opacity: this is
        # 779 points where those are 79, and at the same size the cloud is a solid block.
        # The PALETTE label, not the arm name: _unc_cloud paints by the shared colour table
        # and "C+D+G" is an input modality, not one of its methods.
        _unc_cloud(ax, x, y, UNCA_PAINT[UNCA_ARM], stat=_unca_stat(data, key),
                   point=UNC_POINT * 0.3, corner="upper right", bound=True)
        for artist in ax.collections:
            artist.set_alpha(DATA_ALPHA)
            artist.set_linewidth(0.0)
        furniture(ax, xloc=None)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(labels[key][0], fontsize=13.0, labelpad=8)
        ax.set_title(labels[key][1], fontsize=14.5, color=INK, pad=7)
    flat[0].set_ylabel(UNCA_YLABEL, fontsize=12.0, labelpad=9)
    fit(fig, w_pad=1.35)
    save(fig, out, "affinity_scatter_abs_error")
    print(f"    affinity_scatter_abs_error: {len(data['targets'])} complexes")

    header = (["pid"] + [k for k, _, _ in data["descriptors"]]
              + [f"{arm}_{f}" for arm in data["arms"] for f in ("y_pred", "abs_error")]
              + ["y_true", "delta_abs_error", "delta_sq_error"])
    rows_out = []
    for pid in data["targets"]:
        row = [pid] + [data["complexes"][pid].get(k) for k, _, _ in data["descriptors"]]
        row += [data["arms"][arm][pid][f] for arm in data["arms"]
                for f in ("y_pred", "abs_error")]
        row += [data["arms"][UNCA_BASELINE][pid]["y_true"],
                data["deltas"][UNCA_ARM][pid]["abs_error"],
                data["deltas"][UNCA_ARM][pid]["sq_error"]]
        rows_out.append(row)
    write_csv(out, "affinity_per_complex", header, rows_out)


@figure("fig-uncertainty-affinity-matrix", needs=("task1-affinity/_shared/260916_affinity_uncertainty/",), folder="fig-uncertainty/exploratory/affinity-initial")
def draw_uncertainty_affinity_matrix(out):
    """Every descriptor against every error column, for the density gain and in absolute terms."""
    use_style()
    data = _unca_load()
    labels = {k: (lab, short) for k, lab, short in data["descriptors"]}
    descriptors = [k for k, _, _ in data["descriptors"]]
    outcomes = [o for o, _ in data["outcomes"]]

    for variant, half, arms, vmax in (("delta", "deltas", [UNCA_ARM], 0.4),
                                      ("absolute", "absolute", [UNCA_BASELINE, UNCA_ARM], 0.4)):
        columns = [(arm, o) for arm in arms for o in outcomes
                   if data["correlations"][half].get(arm, {}).get(o)]
        if not columns:
            continue
        grid = np.full((len(descriptors), len(columns)), np.nan)
        marks = {}
        for j, (arm, o) in enumerate(columns):
            for i, key in enumerate(descriptors):
                entry = _unca_stat(data, key, o, half, arm)
                if entry and entry.get("spearman") is not None:
                    grid[i, j] = entry["spearman"]
                    marks[(i, j)] = entry
        fig, ax = plt.subplots(figsize=(1.35 * len(columns) + 4.8,
                                        0.46 * len(descriptors) + 2.3))
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
        ax.imshow(np.abs(grid), cmap=HEAT_CMAP, norm=norm, aspect="auto")
        for (i, j), entry in marks.items():
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=11.0,
                    color="white" if abs(grid[i, j]) > 0.62 * vmax else INK,
                    fontweight="bold" if (entry.get("q") or 1) < 0.05 else "normal")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([f"{arm}\n{o.replace('_', ' ')}" for arm, o in columns],
                           fontsize=11.0)
        ax.set_yticks(range(len(descriptors)))
        ax.set_yticklabels([labels[k][1] for k in descriptors], fontsize=12.0)
        ax.tick_params(length=0, colors=AXIS, pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(descriptors) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=2.1)
        ax.grid(which="major", visible=False)
        ax.set_title("Density gain (C+D+G − C)" if variant == "delta"
                     else "Absolute per-complex error",
                     fontsize=14.0, color=INK, pad=9)
        heat_colorbar(fig, ax, norm, "|Spearman ρ|", label_fs=12.5, tick_fs=11.5, extend="max")
        fit(fig)
        save(fig, out, f"affinity_correlation_matrix_{variant}")
        print(f"    affinity_correlation_matrix_{variant}: {len(descriptors)} x {len(columns)}")

    header = ["half", "arm", "outcome", "descriptor", "n", "n_informative", "spearman", "p", "q",
              "partial_spearman", "pearson", "pearson_p",
              "slope", "slope_lo", "slope_hi", "span", "effect", "effect_lo", "effect_hi"]
    rows_out = []
    for half in ("deltas", "absolute"):
        for arm in sorted(data["correlations"][half]):
            for o in outcomes:
                for entry in data["correlations"][half][arm].get(o, ()):
                    rows_out.append([half, arm, o, entry["descriptor"], entry["n"],
                                     entry.get("n_informative"), entry.get("spearman"),
                                     entry.get("p"), entry.get("q"),
                                     entry.get("partial_spearman"), entry.get("pearson"),
                                     entry.get("pearson_p")]
                                    + [entry.get(k) for k in
                                       ("slope", "slope_lo", "slope_hi", "span",
                                        "effect", "effect_lo", "effect_hi")])
    write_csv(out, "affinity_correlations", header, rows_out)

"""fig-jsd: the 11 figures drawn into figures/fig-jsd/.

    python figures/draw.py jsd

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-jsd-bond                                 -> fig-jsd/
    fig-jsd-bond-distance                        -> fig-jsd/
    fig-jsd-pair                                 -> fig-jsd/
    fig-jsd-atom-type                            -> fig-jsd/
    fig-jsd-summary                              -> fig-jsd/
    fig-jsd-ring-size                            -> fig-jsd/
    fig-jsd-n-rings                              -> fig-jsd/
    fig-jsd-aromatic                             -> fig-jsd/
    fig-jsd-rings                                -> fig-jsd/
    fig-jsd-legend                               -> fig-jsd/
    fig-jsd-metric-summary                       -> fig-jsd/
"""
from __future__ import annotations

import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator

from draw import (
    ALIASES, ARMS, AXIS, BACKGROUND_ALPHA, BUNDLE, CORE, DASH, DATA_ALPHA, DIST_FILL,
    DIST_LW, FIG_W, GRID_LW, INK, MODEL_LW, PANEL_H, PCSZ_COMPARISON_MARGINS,
    PCSZ_INK, PCSZ_LABEL_FS, PCSZ_STRAIN_SIZE, PCSZ_TRIM_PAD, PCSZ_YPAD, REF_COLOR,
    REF_LABEL, REF_LW, REPO, VPA_KEY_ROW_GAP, _pcsz_ordered, _pcsz_style, display,
    figure, fit, furniture, legend, one_frame, pale, row_major, save, use_style,
    write_csv,
)

# ════════════════════════════════════════════════════════════════════════════════
# fig-jsd-{bond,bond-distance,pair,atom-type,summary,ring-size,n-rings,aromatic,rings}
# ════════════════════════════════════════════════════════════════════════════════
# The distribution metrics of TargetDiff and VoxBind: how closely each arm's bond lengths,
# atom-pair distances, element mix and rings follow CrossDocked ligands.
#
#     bond         bond-distance JSD per bond type          VoxBind Table 2
#     bond-distance  the distance histograms those JSDs compare (was bond-length, 2026-09-15)
#     pair         all-atom (<12 Å) and C-C (<2 Å) pair distances   TargetDiff Fig. 2
#     atom-type    heavy-atom element shares
#     summary      the four headline JSDs side by side
#     ring-size    share of rings by size, 3-9               TargetDiff Table 2
#     n-rings      rings per molecule                        VoxBind Fig. 10, middle
#     aromatic     aromatic share of atoms and of rings      VoxBind Fig. 10, bottom
#     rings        all three, one column per method          VoxBind Fig. 10
#
# THESE ONLY DRAW. task2-ligand_generation/scripts/89_eval_crossdocked_jsd.sh computes everything into one
# JSON; its tool's docstring carries the protocol and the self-check that pins it to the
# published TargetDiff row. What a reader of the figures needs from it:
#
# THE REFERENCE DRAWN IS EVERY CROSSDOCKED LIGAND of the train+test split, each distinct
# molecule weighing one (100,100 files are 8,829 molecules; one is cross-docked into 869
# pockets). The published papers score against the 100 test ligands instead, which are
# too few: a model sampling the training distribution exactly would score C=N JSD 0.52
# against them (16 bonds). Those numbers are still computed -- `jsd_test` in the JSON, the
# CSVs here and the appendix tables of jsd-tables.ipynb -- and so is a pose-weighted
# reference, as the check that the de-duplication does not move anything.
#
# THE ARMS stand on the 79 electron-density pockets (TargetDiff and VoxBind hold 100). JSD
# is scipy's JS DISTANCE, as both papers print it.
#
# EVERY METHOD IS DRAWN IN ITS pale() TINT (2026-09-14, on request), the colours of the
# eight-method strain figures (ecdf-by-size-all), not COLORS' saturated originals.
JSD_JSON = BUNDLE / "_shared" / "260913_crossdocked_jsd" / "crossdocked_jsd.json"
JSD_NEEDS = ("crossdocked_jsd.json (task2-ligand_generation/scripts/89_eval_crossdocked_jsd.sh)",)
# Bond lengths are SCORED in 0.005 Å bins; they are DRAWN four bins at a time, so a
# histogram reads as a shape rather than a comb. The JSD columns still come from the
# scoring bins.
JSD_LEN_MERGE = 4
JSD_CC_XLIM = (1.0, 2.0)        # C-C pairs under 2 Å are bonds; nothing sits below 1.0
JSD_NRINGS_MAX = 8              # rings per molecule: the last bar pools 8 and more
JSD_AROM_MERGE = 2              # 20 scored aromatic-fraction bins drawn as 10
# Share panels (atom type, ring size) split into tiers: >= 50% gets a 0-100 axis, >= 5% a
# middle axis, the rest a small one -- each tier on a linear axis scaled to itself. What is
# measured against those thresholds is the TIER CRITERION, which is the tallest bar of the
# category by default and the reference ligand's own share where a panel passes `tier_by`.
JSD_TIER_PCT = (50.0, 5.0)
JSD_KEY_NCOL = 4                # method columns of the all-arms key: 8 arms = 2 rows x 4
# NO FIGURE IN THIS FAMILY CARRIES A KEY (2026-09-16, on request). atom-type, bond-distance
# and ring-size `all` dropped theirs first; the rest followed, and the key is drawn ONCE as
# fig-jsd-legend, to be placed beside whichever panels a page shows. Set True to put every
# figure's own key back -- the call sites still pass their handles.
JSD_KEY_ON = False
# The standalone key's shape, the rigid-fragment box figure's (rigid-consistent-box-all):
# ONE ROW of filled swatches, the crystal ligands first and named "Reference", in a single
# thin frame. Not this family's old stacked key (reference on its own line over a 2 x 4
# method grid), because a key that stands alone is placed under or beside a panel, where a
# row reads in one pass.
JSD_LEGEND_FS = 17.65           # RF_BOX_LEGEND_FS, the box figure's: 14.0 pt ticks x 14.5/11.5
JSD_LEGEND_W, JSD_LEGEND_H = 19.8, 0.78   # inches; the row is wide and one line tall
# AXIS WEIGHT, AS RATIOS OF THE TICK-LABEL SIZE (2026-09-15, on request). Set on the summary
# first -- 11.5 pt ticks, 16.1 pt axis names (the strain ECDF's PCSZ_LABEL_FS = 11.5 x 1.4) and a
# 2.2 pt axis line -- then carried to every jsd figure as the same PROPORTIONS, so a grid with
# 9.5 pt ticks gets a lighter line and smaller names than a bar chart with 14 pt ones.
JSD_TITLE_PER_TICK = 1.4
JSD_AXIS_LW_PER_TICK = 2.2 / 11.5
# ...UP TO THE SUMMARY'S OWN 16.1 pt. The single-panel bar charts carry 14 pt ticks, and 1.4x
# that is a 19.6 pt name: the two-line "JSD to CrossDocked / ligands" and the ring-size figure's
# then "% of rings, sizes 3-9" (now "% of rings", 2026-09-15, on request) ran past both ends of a
# PANEL_H axis and were cut off at the figure edge.
JSD_TITLE_MAX = 11.5 * 1.4
# The line is capped the same way (2026-09-15, on request: similar proportions, not identical
# weights) -- at 14 pt ticks the ratio gave 2.7 pt, which on a single 7.6 in panel read heavier
# than the summary's 2.2 pt does across its four.
JSD_AXIS_LW_MAX = 2.2
# Ring-size panels, fixed (2026-09-14, on request): 6 | 3, 5, 7 | 4, 8, 9. The 5-ring shares
# its panel with the 3- and 7-ring, which the arms push to 12-30%; 4/8/9 stay under 4% for
# every set, so on a panel of their own they are not stubs under AR's 3-rings. Sizes run in
# ascending order inside a panel, whatever order a tuple lists them in.
JSD_RING_PANELS = (("6",), ("3", "5", "7"), ("4", "8", "9"))
# HATCHING, for colour-blind readers (2026-09-14, trial on request): every grouped-bar chart
# and its key give a method a pattern as well as its pale() tint, drawn in a darker step of
# that tint. The reference stays plain grey and CoDE plain -- ours, and the one solid method
# bar. Summary and rings are not hatched: each bar there already carries the method's name.
# OFF (2026-09-14): tried and set aside on request -- the figures are drawn plain. Set True to
# bring the patterns back.
JSD_HATCH_ON = False
JSD_HATCH = {"AR": "///", "Pocket2Mol": "\\\\\\", "DiffSBDD": "xxx", "DecompDiff": "...",
             "FuncBind": "---", "TargetDiff": "|||", "VoxBind": "ooo"}
JSD_HATCH_LW = 0.6               # points; matplotlib's 1.0 fills a 0.1-inch bar
JSD_HATCH_INK = 0.55             # hatch = the bar's tint times this (toward black)
JSD_SUMMARY = (
    ("Mean bond-distance JSD ↓", lambda j: j["bond_mean"]),
    ("All-atom pair JSD ↓", lambda j: j["pair"]["All_12A"]),
    ("C–C pair JSD ↓", lambda j: j["pair"]["CC_2A"]),
    ("Atom-type JSD ↓", lambda j: j["atom_type"]),
)
# (CSV source label, arm key) for every reference the eval scores against; the first is
# the one the figures draw.
JSD_REFERENCES = (("vs CrossDocked train+test (distinct molecules)", "jsd"),
                  ("vs CrossDocked train+test (pose-weighted)", "jsd_pose_weighted"),
                  ("vs CrossDocked test (100)", "jsd_test"))
_JSD_CACHE = {}


def _jsd_data():
    """(the eval JSON, the ARMS it covers in ARMS order). An arm the JSON lacks is named
    and skipped rather than drawn empty."""
    if "d" not in _JSD_CACHE:
        if not JSD_JSON.exists():
            raise FileNotFoundError(f"missing input {JSD_JSON}\n"
                                    f"  rebuild it with bash task2-ligand_generation/scripts/89_eval_crossdocked_jsd.sh")
        d = json.load(open(JSD_JSON))
        if "reference_test" not in d:
            raise ValueError(f"{JSD_JSON} predates the train+test reference; "
                             f"rerun bash task2-ligand_generation/scripts/89_eval_crossdocked_jsd.sh")
        arms = [a for a in ARMS if a[0] in d["arms"]]
        if not arms:
            raise ValueError(f"{JSD_JSON} holds none of the ARMS labels")
        missing = [a[0] for a in ARMS if a[0] not in d["arms"]]
        pockets = {d["arms"][lab]["n_pockets"] for lab, _, _ in arms}
        ref = d["reference"]
        print(f"  [{os.path.relpath(JSD_JSON, REPO)}]")
        print(f"  reference: CrossDocked train+test, {ref['n_files']:,} ligand files = "
              f"{ref['n_unique_ligands']:,} distinct molecules, each weighing 1 · "
              f"arms over {'/'.join(map(str, sorted(pockets)))} pockets")
        sc = d.get("selfcheck")
        if sc:
            print(f"  selfcheck vs published TargetDiff (test reference): max |Δ| bond JSD "
                  f"{sc['max_abs_diff_bond_jsd']:.4f}, ring % {sc['max_abs_diff_ring_pct']:.2f}"
                  f" -> {'PASS' if sc['pass'] else 'FAIL'}")
        if missing:
            print("  not in the JSON, not drawn: " + ", ".join(missing))
        _JSD_CACHE["d"] = (d, arms)
    return _JSD_CACHE["d"]


def _jsd_ref_key():
    """The reference's legend entry, with its size -- distinct molecules, not files."""
    ref = _jsd_data()[0]["reference"]
    return f"{ref['label']}, {ref['n_unique_ligands']:,} molecules"


def _jsd_ref_sets(d):
    """(CSV label, summary) of every reference whose histograms a CSV should carry."""
    return [(f"{REF_LABEL} (train+test)", d["reference"]),
            (f"{REF_LABEL} (train+test, pose-weighted)", d["reference_pose_weighted"]),
            (f"{REF_LABEL} (test)", d["reference_test"])]


def _jsd_variants(arms):
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


def _jsd_new(w, h):
    fig = plt.figure(figsize=(w, h), dpi=220)
    fig.patch.set_facecolor("white")
    return fig


def _jsd_axis_weight(fig):
    """Axis names and axis lines scaled to each panel's tick labels -- JSD_TITLE_PER_TICK and
    JSD_AXIS_LW_PER_TICK. Call once, after every panel is drawn and BEFORE fit(), since the
    larger names change the layout. The smaller of a panel's two tick sizes sets its line:
    the numeric axis, not a column of method names. Tick LENGTH is left alone -- the bar
    charts set it to zero on their categorical axis. The spines go above the data: every bar
    starts at the baseline and painted over half of the heavier line."""
    for ax in fig.axes:
        size = {a: a.get_major_ticks()[0].label1.get_fontsize()
                for a in (ax.xaxis, ax.yaxis) if a.get_major_ticks()}
        if not size:
            continue
        lw = min(JSD_AXIS_LW_PER_TICK * min(size.values()), JSD_AXIS_LW_MAX)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(lw)
            ax.spines[side].set_zorder(5)
        ax.tick_params(width=lw)
        for a, fs in size.items():
            if a.label.get_text():
                a.label.set_size(min(JSD_TITLE_PER_TICK * fs, JSD_TITLE_MAX))


def _jsd_key_above(fig, handles, ncol, fontsize=11.5):
    """A figure-level key in a strip above every panel. Laid out AFTER fit(), because
    tight_layout does not know about figure legends: the panels are fitted first, then
    pushed down by exactly the overlap the key makes with the tallest one.

    THE REFERENCE TAKES A CENTRED LINE OF ITS OWN over the method grid, one frame round both
    -- the Vina per-atom v3 key's layout. As one more cell of a single-row grid, the
    "Reference ligand (CrossDocked train+test), 8,829 molecules" entry made the key wider
    than the figure and it was cut off at both edges (atom-type, 2026-09-14). The grid reads
    across rows, and it loses a column at a time until it fits the figure's width."""
    _jsd_axis_weight(fig)
    fit(fig, pad=0.5)
    # No handles, no key -- and since 2026-09-16 no figure in the family has one at all
    # (JSD_KEY_ON): the key is drawn once, as fig-jsd-legend. The axis weights and the fit
    # above still apply, which is why every figure still calls this.
    if not handles or not JSD_KEY_ON:
        return
    ref_key = _jsd_ref_key()
    ref = [h for h in handles if h.get_label() == ref_key]
    methods = [h for h in handles if h.get_label() != ref_key]
    inv = fig.transFigure.inverted()
    ncol = max(1, min(ncol, len(methods)))
    while True:
        legs, top = [], 0.995
        for group, cols in ((ref, 1), (methods, ncol)):
            if not group:
                continue
            legs.append(legend(fig, row_major(group, cols), loc="upper center", ncol=cols,
                               fontsize=fontsize, bbox_to_anchor=(0.5, top)))
            fig.canvas.draw()
            top = legs[-1].get_window_extent(fig.canvas.get_renderer()).transformed(inv).y0 \
                - VPA_KEY_ROW_GAP
        rend = fig.canvas.get_renderer()
        widest = max(l.get_window_extent(rend).transformed(inv).width for l in legs)
        if widest <= 0.98 or ncol == 1:
            break
        for l in legs:
            l.remove()
        ncol -= 1
    if len(legs) > 1:
        one_frame(fig, legs)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    y0 = min(l.get_window_extent(rend).transformed(inv).y0 for l in legs)
    y1 = max(ax.get_tightbbox(rend).transformed(inv).y1 for ax in fig.axes)
    if y1 > y0 - 0.012:
        fig.subplots_adjust(top=fig.subplotpars.top - (y1 - y0 + 0.012))


def _jsd_hatch(label, col):
    """(pattern, ink) for a method's bars, or (None, col) where it is drawn plain."""
    pattern = JSD_HATCH.get(ALIASES.get(label, label)) if JSD_HATCH_ON else None
    if not pattern:
        return None, col
    return pattern, tuple(JSD_HATCH_INK * c for c in matplotlib.colors.to_rgb(col))


def _jsd_handles(arms, *, ref=True, patch=False):
    def one(label, col, ls="-", lw=MODEL_LW, key=None):
        if patch:
            hatch, ink = _jsd_hatch(key, col)
            h = Patch(facecolor=col, edgecolor=ink, hatch=hatch, lw=0, label=label)
            h.set_hatch_linewidth(JSD_HATCH_LW)
            return h
        return Line2D([], [], color=col, lw=lw, ls=ls, label=label)
    hs = [one(_jsd_ref_key(), REF_COLOR, DASH, REF_LW, key=REF_LABEL)] if ref else []
    return hs + [one(display(lab), pale(lab), key=lab) for lab, _, _ in arms]


def _jsd_key_cols(n, wide):
    """Method columns of the key. The reference always takes its own line above, so this only
    shapes the methods: the eight arms read 2 x 4 (2026-09-14, on request; they were 5 + 3),
    the same grid as the Vina per-atom v3 and strain keys. _jsd_key_above caps it at the
    method count, so the core pair stays one row."""
    return n if n <= 3 else (JSD_KEY_NCOL if wide else 3)


def _jsd_grouped(ax, groups, series):
    """Grouped bars: one group per x category, one bar per (label, values, colour) in
    `series` order, so the key's order is the bars' order. Bar width is derived from the
    series count -- a fixed width overlaps neighbouring groups at nine series."""
    x = np.arange(len(groups))
    w = 0.84 / len(series)
    for i, (lab, vals, col) in enumerate(series):
        vals = [np.nan if v is None else v for v in vals]
        xs = x + (i - (len(series) - 1) / 2) * w
        # A patch's hatch is drawn in its EDGE colour, so a hatched bar is two passes: the fill
        # with the pattern in its ink and no outline, then the white spacer outline on top.
        hatch, ink = _jsd_hatch(lab, col)
        if hatch:
            for bar in ax.bar(xs, vals, width=w, color=col, edgecolor=ink, hatch=hatch, lw=0,
                              zorder=3):
                bar.set_hatch_linewidth(JSD_HATCH_LW)
            ax.bar(xs, vals, width=w, fill=False, edgecolor="white",
                   lw=0.4 if len(series) > 4 else 0.8, zorder=3)
        else:
            ax.bar(xs, vals, width=w, color=col, edgecolor="white",
                   lw=0.4 if len(series) > 4 else 0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.5, len(groups) - 0.5)


def _jsd_tiered(fig, cats, names, series, ylabel, tier_by=None, panels=None, *, subplot_spec=None):
    """Grouped share bars (in %) on up to three side-by-side LINEAR axes, one per JSD_TIER_PCT
    tier of a category -- so a category at ~1% is not a stub under one at ~70%, and nothing
    needs a log axis. `series` is (label, {category: pct}, colour). The top tier is fixed at
    0-100; the others scale to their own tallest bar. Panel width = category count, so a bar
    is the same width in every panel; the y scales differ, so each keeps its ticks.
    Categories keep their given order inside a panel; panels run tallest tier first.

    WHICH TIER A CATEGORY LANDS IN is decided by `tier_by` -- {category: pct} -- and defaults
    to the category's tallest bar. Passing the reference ligand's own shares instead groups
    the panels by what a real ligand MAKES rather than by what the worst arm happens to do
    with it, which is a statement about the categories and stays put when an arm is added or
    dropped. The y limit is always the real tallest bar, so nothing is ever clipped: a tier
    chosen against the reference can still hold a bar many times the reference's height, and
    that is the finding, not a drawing error.

    `panels` -- a sequence of category tuples -- overrides the tiers outright; a panel whose
    every category reaches the top tier still gets the fixed 0-100 axis.
    `subplot_spec` nests the panels in one row of a larger figure."""
    peak = {c: max((vals[c] or 0) for _, vals, _ in series) for c in cats}
    rank = peak if tier_by is None else {c: tier_by.get(c, 0) or 0 for c in cats}
    hi, mid = JSD_TIER_PCT
    if panels is not None:
        tiers = [[c for c in cats if c in p] for p in panels]
        leftover = [c for c in cats if not any(c in p for p in panels)]
        if leftover:
            raise ValueError(f"categories in no panel: {leftover}")
        tiers = [(0 if all(rank[c] >= hi for c in t) else 1, t) for t in tiers if t]
    else:
        tiers = [[c for c in cats if rank[c] >= hi], [c for c in cats if mid <= rank[c] < hi],
                 [c for c in cats if rank[c] < mid]]
        tiers = [(i, t) for i, t in enumerate(tiers) if t]
    grid = fig.add_gridspec if subplot_spec is None else subplot_spec.subgridspec
    gs = grid(1, len(tiers), width_ratios=[len(t) for _, t in tiers])
    for k, (level, tier) in enumerate(tiers):
        ax = fig.add_subplot(gs[0, k])
        ax.set_facecolor("white")
        _jsd_grouped(ax, [names[c] for c in tier],
                     [(lab, [vals[c] for c in tier], col) for lab, vals, col in series])
        furniture(ax, ylabel=ylabel if k == 0 else None, xloc=None)
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", length=0)
        if level == 0:
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_locator(MultipleLocator(25))
        else:
            ax.set_ylim(0, max(peak[c] for c in tier) * 1.1)
    return [t for _, t in tiers]


def _jsd_round(v, nd=4):
    """A JSON number for a CSV cell. None is a JSD with nothing to compare (an arm with no
    bond of that type) and stays an empty cell rather than becoming 0."""
    return "" if v is None else round(v, nd)


def _jsd_pct(counts):
    counts = np.asarray(counts, float)
    return 100 * counts / counts.sum() if counts.sum() else counts


# ── bond distances ──────────────────────────────────────────────────────────────
@figure("fig-jsd-bond", needs=JSD_NEEDS)
def draw_jsd_bond(out):
    """Bond-distance JSD per bond type against the CrossDocked ligands (VoxBind Table 2)."""
    d, all_arms = _jsd_data()
    use_style()
    types = d["published"]["bond_jsd_columns"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * 1.45, PANEL_H)
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        series = [(lab, [d["arms"][lab]["jsd"]["bond"][t] for t in types]
                   + [d["arms"][lab]["jsd"]["bond_mean"]], pale(lab)) for lab, _, _ in arms]
        _jsd_grouped(ax, types + ["mean"], series)
        top = max(v for _, vals, _ in series for v in vals if v is not None)
        furniture(ax, ylabel="JSD to CrossDocked\nligands ↓", xloc=None)
        # AFTER furniture(), whose tick_params would reset the size
        ax.set_xticklabels(types + ["mean"], fontsize=13)
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")
        ax.axvline(len(types) - 0.5, color=AXIS, lw=GRID_LW, zorder=1)
        ax.set_ylim(0, top * 1.08)
        _jsd_key_above(fig, _jsd_handles(arms, ref=False, patch=True),
                       _jsd_key_cols(len(arms), wide))
        save(fig, out, f"bond_{variant}")

    rows = []
    for source, key in JSD_REFERENCES:
        rows += [[lab, source, d["arms"][lab]["n_mols"]]
                 + [_jsd_round(d["arms"][lab][key]["bond"][t]) for t in types]
                 + [_jsd_round(d["arms"][lab][key]["bond_mean"])] for lab, _, _ in all_arms]
    for lab, s in _jsd_ref_sets(d):
        rows.append([f"{lab}: bonds", "reference bond count (weighted where de-duplicated)",
                     s["n_mols"]] + [s["bond_n"][t] for t in types] + [""])
    for lab, vals in d["published"]["bond_jsd"].items():
        rows.append([lab, "published, vs CrossDocked test (100 pockets)", ""] + vals
                    + [round(float(np.mean(vals)), 4)])
    write_csv(out, "bond_jsd", ["set", "source", "n_mols"] + types + ["mean"], rows)


def _jsd_method_grid(out, d, arms, kinds, *, xlabel, stem, key=True, transpose=False):
    """Rows are methods, columns are what is measured. Every panel is ONE method's
    distribution over the reference's dashed outline, with that panel's JSD in its corner.

    The overlay puts nine curves in one axes, where the arm you are looking for is the one
    you cannot see; here each is read against the reference alone. y is shared along the
    axis a KIND runs, so a method's peak height still compares with the other methods'.

    TRANSPOSE swaps the two (2026-09-16, on request, for pair): methods become the columns
    and kinds the rows, which turns a two-kind figure from a page-tall column strip into a
    wide block -- the shape a paper column wants. The kind's name moves from the column
    head to the row's y name, the method's from the row's y name to the column head, and
    x is then shared ALONG A ROW, because a row is now one kind on one scale.

    `kinds` is [(kind title, set -> (pct, edges), arm -> JSD, xlim, x tick step)]."""
    n, m = len(arms), len(kinds)
    if transpose:
        # rows are shorter than they are wide here, so they get more height than the 1.12
        # a method row takes in the upright form
        fig = _jsd_new(1.95 * n + 1.7, 1.5 * m + 1.5)
        axes = fig.subplots(m, n, sharex="row", sharey="row", squeeze=False)
    else:
        fig = _jsd_new(1.95 * m + 1.7, 1.12 * n + 1.5)
        axes = fig.subplots(n, m, sharex="col", sharey="col", squeeze=False)
    for j, (title, dist, jsd_of, xlim, xstep) in enumerate(kinds):
        ref_pct, _ = dist(d["reference"])
        top = float(ref_pct.max())
        for i, (lab, _, _) in enumerate(arms):
            ax = axes[j][i] if transpose else axes[i][j]
            ax.set_facecolor("white")
            pct, edges = dist(d["arms"][lab])
            top = max(top, float(pct.max()))
            ax.stairs(pct, edges, color=pale(lab), fill=True, alpha=BACKGROUND_ALPHA, lw=0, zorder=2)
            ax.stairs(pct, edges, color=pale(lab), lw=DIST_LW - 0.4, zorder=3)
            ax.stairs(ref_pct, edges, color=REF_COLOR, lw=REF_LW - 0.2, ls=DASH, zorder=4)
            bottom = (j == m - 1) if transpose else (i == n - 1)
            furniture(ax, xloc=xstep, xlim=xlim, xlabel=xlabel if bottom else None)
            ax.tick_params(labelsize=9.5, length=2.5)
            ax.xaxis.label.set_size(11.5)
            v = jsd_of(d["arms"][lab])
            ax.text(0.97, 0.93, "JSD —" if v is None else f"JSD {v:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=DATA_ALPHA))
            if transpose:
                if i == 0:
                    ax.set_ylabel(title, fontsize=11, labelpad=8)
                if j == 0:
                    ax.set_title(display(lab), fontsize=11, color=INK, pad=6)
            else:
                if j == 0:
                    ax.set_ylabel(display(lab), fontsize=12, labelpad=8)
                if i == 0:
                    ax.set_title(title, fontsize=11, color=INK, pad=6)
        # the shared axis a kind runs along: a column upright, a row transposed
        (axes[j][0] if transpose else axes[0][j]).set_ylim(0, top * 1.12)
    # The key names the reference only: the y unit is in each column head, and a longer key
    # outran the two-column pair grid and was cut off at both edges.
    _jsd_key_above(fig, [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH,
                                label=_jsd_ref_key())] if key else None, 1, fontsize=11)
    save(fig, out, stem)


# Renamed from fig-jsd-bond-length / bond-length-* (2026-09-15, on request), to match the
# "bond-distance JSD" of fig-jsd-bond that these histograms feed.
@figure("fig-jsd-bond-distance", needs=JSD_NEEDS)
def draw_jsd_bond_distance(out):
    """The bond-distance histograms behind fig-jsd-bond, one panel per bond type."""
    d, all_arms = _jsd_data()
    use_style()
    types = d["published"]["bond_jsd_columns"]
    bins = np.asarray(d["protocol"]["bond_bins"])
    step = float(bins[1] - bins[0])

    def drawn(counts):
        """Share of ALL of a set's bonds per merged bin. Index 0 and the last index are the
        under/overflow bins (searchsorted over the edges): they count in the denominator,
        so a set with many out-of-range bonds shows less mass, and are not drawn."""
        counts = np.asarray(counts, float)
        inner = counts[1:-1]
        inner = np.concatenate([inner, np.zeros((-len(inner)) % JSD_LEN_MERGE)])
        merged = inner.reshape(-1, JSD_LEN_MERGE).sum(1)
        edges = bins[0] + step * JSD_LEN_MERGE * np.arange(len(merged) + 1)
        # 118 inner bins pad to 120, but the last scoring edge is 1.69 Å: longer bonds are
        # overflow, so the last bar must not claim [1.69, 1.70) it holds nothing from
        edges[-1] = bins[-1]
        return (100 * merged / counts.sum() if counts.sum() else merged), edges

    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        if wide:
            # all arms: a method x bond-type grid (see _jsd_method_grid); core keeps the
            # overlay, where two curves over the reference still read directly
            _jsd_method_grid(out, d, arms, [
                (f"{t}\n% per {step * JSD_LEN_MERGE:.2f} Å",
                 (lambda s, t=t: drawn(s["bond_counts"][t])),
                 (lambda a, t=t: a["jsd"]["bond"][t]), (bins[0], bins[-1]), 0.2) for t in types],
                # no key (2026-09-15, on request); pair-all keeps its reference key
                xlabel="Bond distance (Å)", stem=f"bond_distance_{variant}", key=False)
            continue
        fig = _jsd_new(FIG_W * 1.55, PANEL_H * 1.62)
        axes = fig.subplots(2, 4)
        for i, t in enumerate(types):
            ax = axes.flat[i]
            ax.set_facecolor("white")
            for lab, _, _ in arms:
                pct, edges = drawn(d["arms"][lab]["bond_counts"][t])
                ax.stairs(pct, edges, color=pale(lab), lw=DIST_LW, zorder=3)
                ax.stairs(pct, edges, color=pale(lab), fill=True, alpha=DIST_FILL, lw=0, zorder=2)
            pct, edges = drawn(d["reference"]["bond_counts"][t])
            ax.stairs(pct, edges, color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
            furniture(ax, xloc=0.2, xlim=(bins[0], bins[0] + step * 120),
                      ylabel=f"% of bonds\nper {step * JSD_LEN_MERGE:.2f} Å" if i % 4 == 0 else None,
                      xlabel="Bond distance (Å)" if i >= 4 else None)
            ax.tick_params(labelsize=11.5)
            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)
            ax.set_ylim(bottom=0)
            # a panel LABEL, not a title
            ax.set_title(t, loc="left", fontsize=12.5, color=INK, pad=5)
        _jsd_key_above(fig, _jsd_handles(arms), _jsd_key_cols(len(arms) + 1, True))
        save(fig, out, f"bond_distance_{variant}")

    rows = []
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for t in types:
            c = s["bond_counts"][t]
            total = sum(c)
            lo = [float("-inf")] + bins.tolist()
            hi = bins.tolist() + [float("inf")]
            rows += [[lab, t, round(a, 3), round(b, 3), n, round(100 * n / total, 4) if total else 0]
                     for a, b, n in zip(lo, hi, c)]
    write_csv(out, "bond_distance_hist",["set", "bond_type", "len_lo", "len_hi", "n", "pct"], rows)


# ── pair distances ──────────────────────────────────────────────────────────────
@figure("fig-jsd-pair", needs=JSD_NEEDS)
def draw_jsd_pair(out):
    """All-atom (<12 Å) and C–C (<2 Å) pair-distance distributions (TargetDiff Fig. 2)."""
    d, all_arms = _jsd_data()
    use_style()
    panels = (("All_12A", "All heavy-atom pairs · distance (Å)", None),
              ("CC_2A", "C–C pairs under 2 Å · distance (Å)", JSD_CC_XLIM))

    def dist(key):
        edges = np.asarray(d["protocol"]["pair_bins"][key])
        return lambda s: (_jsd_pct(s["pair_counts"][key])[1:-1], edges)

    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        if wide:
            # all arms: a method x pair-kind grid (see _jsd_method_grid); core keeps the overlay
            step = {k: float(np.diff(d["protocol"]["pair_bins"][k][:2])[0]) for k, _, _ in panels}
            _jsd_method_grid(out, d, arms, [
                # SHORT NAMES (2026-09-16): transposed, a kind's name is a ROW's y name, and
                # "All heavy-atom pairs < 12 Å" is taller than a row -- it was cut off at the
                # top of the figure. The kinds are spelled out in the caption instead.
                (f"All pairs < 12 Å\n% per {step['All_12A']:.2f} Å", dist("All_12A"),
                 lambda a: a["jsd"]["pair"]["All_12A"], (0, 12), 4),
                (f"C–C pairs < 2 Å\n% per {step['CC_2A']:.2f} Å", dist("CC_2A"),
                 lambda a: a["jsd"]["pair"]["CC_2A"], JSD_CC_XLIM, 0.5)],
                # LANDSCAPE (2026-09-16, on request): eight method columns over two kind
                # rows, instead of eight method rows over two kind columns
                xlabel="Distance (Å)", stem=f"pair_{variant}", transpose=True)
            continue
        fig = _jsd_new(FIG_W * 1.45, PANEL_H * 1.05)
        axes = fig.subplots(1, 2)
        for ax, (key, xlabel, xlim) in zip(axes, panels):
            ax.set_facecolor("white")
            edges = np.asarray(d["protocol"]["pair_bins"][key])
            for lab, _, _ in arms:
                ax.stairs(_jsd_pct(d["arms"][lab]["pair_counts"][key])[1:-1], edges,
                          color=pale(lab), lw=DIST_LW, zorder=3)
            ax.stairs(_jsd_pct(d["reference"]["pair_counts"][key])[1:-1], edges,
                      color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
            furniture(ax, xlabel=xlabel, xloc=None, xlim=xlim or (edges[0], edges[-1]),
                      # two lines, as bond-distance's: at the axis-name size _jsd_axis_weight sets,
                      # one line ran past both ends of the panel and lost its unit
                      ylabel=f"% of pairs\nper {edges[1] - edges[0]:.2f} Å")
            ax.tick_params(labelsize=12.5)
            ax.xaxis.label.set_size(13.5)
            ax.yaxis.label.set_size(13.5)
            ax.set_ylim(bottom=0)
        _jsd_key_above(fig, _jsd_handles(arms), _jsd_key_cols(len(arms) + 1, True))
        save(fig, out, f"pair_{variant}")

    rows = []
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for key, _, _ in panels:
            edges = d["protocol"]["pair_bins"][key]
            pct = _jsd_pct(s["pair_counts"][key])
            rows += [[lab, key, round(a, 4), round(b, 4), n, round(float(q), 4)]
                     for a, b, n, q in zip(edges[:-1], edges[1:], s["pair_counts"][key][1:-1], pct[1:-1])]
    write_csv(out, "pair_hist", ["set", "pairs", "dist_lo", "dist_hi", "n", "pct"], rows)


# ── atom types and the headline numbers ─────────────────────────────────────────
@figure("fig-jsd-atom-type", needs=JSD_NEEDS)
def draw_jsd_atom_type(out):
    """Heavy-atom element shares against the CrossDocked ligands: C | N, O | F, P, S, Cl."""
    d, all_arms = _jsd_data()
    use_style()
    elems = d["protocol"]["atom_types"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        # C | N, O | F, P, S, Cl on three linear axes (2026-09-14, on request; it was one log axis)
        series = [(REF_LABEL, {e: 100 * d["reference"]["atom_frac"][e] for e in elems}, REF_COLOR)]
        series += [(lab, {e: 100 * d["arms"][lab]["atom_frac"][e] for e in elems}, pale(lab))
                   for lab, _, _ in arms]
        _jsd_tiered(fig, elems, {e: e for e in elems}, series, "% of heavy atoms")
        # no key on `all` (2026-09-15, on request); `core` keeps its own
        _jsd_key_above(fig, None if wide else _jsd_handles(arms, patch=True),
                       _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"atom_type_{variant}")

    rows = [[lab, s["n_mols"]] + [round(100 * s["atom_frac"][e], 3) for e in elems]
            + [round(100 * s["atom_other_frac"], 3), "", "", ""] for lab, s in _jsd_ref_sets(d)]
    rows += [[lab, d["arms"][lab]["n_mols"]] + [round(100 * d["arms"][lab]["atom_frac"][e], 3) for e in elems]
             + [round(100 * d["arms"][lab]["atom_other_frac"], 3)]
             + [_jsd_round(d["arms"][lab][key]["atom_type"]) for _, key in JSD_REFERENCES]
             for lab, _, _ in all_arms]
    write_csv(out, "atom_type", ["set", "n_mols"] + [f"pct_{e}" for e in elems]
              + ["pct_other_not_scored"] + [f"atom_type_{key}" for _, key in JSD_REFERENCES], rows)


@figure("fig-jsd-summary", needs=JSD_NEEDS)
def draw_jsd_summary(out):
    """Mean bond, all-atom pair, C–C pair and atom-type JSD per arm, side by side."""
    d, all_arms = _jsd_data()
    use_style()
    for variant, arms in _jsd_variants(all_arms):
        fig = _jsd_new(FIG_W * 1.75, 0.5 * len(arms) + 1.55)
        axes = fig.subplots(1, len(JSD_SUMMARY), sharey=True)
        ys = np.arange(len(arms))[::-1]
        for ax, (name, get) in zip(axes, JSD_SUMMARY):
            ax.set_facecolor("white")
            vals = [get(d["arms"][lab]["jsd"]) for lab, _, _ in arms]
            ax.barh(ys, [np.nan if v is None else v for v in vals], height=0.66,
                    color=[pale(lab) for lab, _, _ in arms], zorder=3)
            top = max([v for v in vals if v is not None], default=1.0)
            # every value is labelled: this figure IS the table, drawn. A None is a JSD with
            # nothing to compare and says so, rather than drawing as a zero-length "best"
            for y, v in zip(ys, vals):
                ax.text((v or 0) + top * 0.03, y, "—" if v is None else f"{v:.3f}",
                        va="center", ha="left", fontsize=11.5, color=INK, zorder=4)
            furniture(ax, xlabel=name, xloc=None, xlim=(0, top * 1.38))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=11.5, length=5)
            ax.grid(False, axis="y")
        axes[0].set_yticks(ys)
        axes[0].set_yticklabels([display(lab) for lab, _, _ in arms], fontsize=14.5)
        axes[0].set_ylim(-0.6, len(arms) - 0.4)
        # heavier axes, larger names (2026-09-15, on request): the figure the ratios come from
        _jsd_axis_weight(fig)
        fit(fig, pad=0.5)
        save(fig, out, f"summary_{variant}")
    head = ["set", "n_mols", "n_disconnected_dropped"]
    cols = ("bond_mean", "pair_all_12A", "pair_cc_2A", "atom_type")
    for _, key in JSD_REFERENCES:
        head += [f"{c}_{key}" for c in cols]
    write_csv(out, "summary", head,
              [[lab, d["arms"][lab]["n_mols"], d["arms"][lab]["n_disconnected"]]
               + [_jsd_round(get(d["arms"][lab][key])) for _, key in JSD_REFERENCES for _, get in JSD_SUMMARY]
               for lab, _, _ in all_arms])


# ── rings ───────────────────────────────────────────────────────────────────────
@figure("fig-jsd-ring-size", needs=JSD_NEEDS)
def draw_jsd_ring_size(out):
    """Share of rings by size 3–9 (TargetDiff Table 2), with the table itself as CSV."""
    d, all_arms = _jsd_data()
    use_style()
    sizes = d["published"]["ring_size_pct_columns"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        series = [(REF_LABEL, {s: d["reference"]["ring_size_pct"][s] for s in sizes}, REF_COLOR)]
        series += [(lab, {s: d["arms"][lab]["ring_size_pct"][s] for s in sizes}, pale(lab))
                   for lab, _, _ in arms]
        # Tiered linear axes (2026-09-14, on request; it was one log axis over 0.01-100%).
        # TIERED BY THE CRYSTAL LIGANDS, NOT BY THE TALLEST BAR (2026-09-14, on request): the
        # panels then read 6-ring | 5-ring | the sizes a real ligand barely makes, and 3- and
        # 7-rings sit in that third panel where the reference puts them (1.5% and 0.8%)
        # instead of being lifted into the middle one by AR's 30% 3-rings and TargetDiff's 12%
        # 7-rings. Those two bars are still drawn at full height -- the panel scales to its
        # tallest bar, so the third panel says "sizes the reference avoids, and by how far the
        # arms overshoot them". The grouping no longer moves when an arm is added or dropped:
        # `core` and `all` now split the same way.
        # Then FIXED PANELS (2026-09-14, on request): 6 | 3, 5, 7 | 4, 8, 9 -- see
        # JSD_RING_PANELS. tier_by still decides which panel is the 0-100 one.
        _jsd_tiered(fig, sizes, {s: f"{s}-ring" for s in sizes}, series, "% of rings",
                    tier_by=d["reference"]["ring_size_pct"], panels=JSD_RING_PANELS)
        # no key on `all` (2026-09-15, on request); `core` keeps its own
        _jsd_key_above(fig, None if wide else _jsd_handles(arms, patch=True),
                       _jsd_key_cols(len(arms) + 1, wide))
        save(fig, out, f"ring_size_{variant}")

    head = ["set", "source"] + sizes + ["10+ of all rings"]
    rows = [[lab, "reference"] + [round(s["ring_size_pct"][z], 1) for z in sizes]
            + [round(s["ring_size_pct"]["10+ of all"], 1)] for lab, s in _jsd_ref_sets(d)]
    rows += [[lab, f"this eval ({d['arms'][lab]['n_pockets']} pockets)"]
             + [round(d["arms"][lab]["ring_size_pct"][s], 1) for s in sizes]
             + [round(d["arms"][lab]["ring_size_pct"]["10+ of all"], 1)] for lab, _, _ in all_arms]
    rows += [[lab, "published TargetDiff Table 2 (reference = test)"] + vals + [""]
             for lab, vals in d["published"]["ring_size_pct"].items()]
    write_csv(out, "ring_size_pct", head, rows)


@figure("fig-jsd-n-rings", needs=JSD_NEEDS)
def draw_jsd_n_rings(out):
    """Share of molecules by number of rings; all methods use separately scaled panels."""
    d, all_arms = _jsd_data()
    use_style()
    labels = [str(k) for k in range(JSD_NRINGS_MAX)] + [f"≥{JSD_NRINGS_MAX}"]
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.3 if wide else 1.0), PANEL_H)
        series = [(REF_LABEL, _jsd_ring_rows(d["reference"])[1], REF_COLOR)]
        series += [(lab, _jsd_ring_rows(d["arms"][lab])[1], pale(lab)) for lab, _, _ in arms]
        if wide:
            # Scale the rare ring counts separately, using the same tiers as aromatic.
            _jsd_tiered(fig, labels, {label: label for label in labels},
                        [(lab, dict(zip(labels, vals)), col) for lab, vals, col in series],
                        "% of molecules")
            for ax in fig.axes:
                # Reserve space for a single x label spanning every panel.
                furniture(ax, xlabel=" ", xloc=None)
                ax.grid(False, axis="x")
                ax.tick_params(axis="x", length=0)
        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor("white")
            _jsd_grouped(ax, labels, series)
            furniture(ax, ylabel="% of molecules", xlabel="Rings per molecule", xloc=None)
            ax.grid(False, axis="x")
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0, max(float(np.max(v)) for _, v, _ in series) * 1.08)
        _jsd_key_above(fig, _jsd_handles(arms, patch=True), _jsd_key_cols(len(arms) + 1, wide))
        if wide:
            axes = fig.axes
            ax = axes[len(axes) // 2]
            center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
            ax.xaxis.label.set_text("Rings per molecule")
            ax.xaxis.label.set_x((center - ax.get_position().x0) / ax.get_position().width)
        save(fig, out, f"n_rings_{variant}")

    sets = _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]
    write_csv(out, "n_rings", ["set", "mean_rings_per_molecule"] + [f"pct_{b}" for b in labels],
              [[lab, round(s["mean_n_rings"], 3)] + [round(float(q), 3) for q in _jsd_ring_rows(s)[1]]
               for lab, s in sets])


@figure("fig-jsd-aromatic", needs=JSD_NEEDS)
def draw_jsd_aromatic(out):
    """Aromatic share of each molecule's heavy atoms (VoxBind Fig. 10, bottom row) and of its rings."""
    d, all_arms = _jsd_data()
    use_style()
    labels = [f"{k / 10:.1f}–{(k + 1) / 10:.1f}" for k in range(10)]
    # Two definitions, one row each. VoxBind Fig. 10 plots the aromatic share of a
    # molecule's heavy ATOMS; the share of its RINGS that are aromatic (every ring bond
    # aromatic) says the same thing without the ring-free atoms diluting it, and is only
    # defined for a molecule that has a ring -- so its panel counts those molecules alone.
    panels = (
        ("Aromatic share of a molecule's heavy atoms",
         lambda s: _jsd_ring_rows(s)[2], "mean_arom_atom_frac", "% of molecules"),
        ("Aromatic share of a molecule's rings (molecules with ≥1 ring)",
         lambda s: _jsd_pct(s["arom_ring_frac_counts"]), "mean_arom_ring_frac", "% of molecules\nwith a ring"),
    )
    for variant, arms in _jsd_variants(all_arms):
        wide = variant == "all"
        fig = _jsd_new(FIG_W * (1.6 if wide else 1.0), PANEL_H * 1.9)
        rowspec = (fig.add_gridspec(2, 2, width_ratios=[10, 1.4]) if wide
                   else fig.add_gridspec(2, 1))
        row_labels = []
        for row, (xlabel, get, mean_key, ylabel) in enumerate(panels):
            series = [(REF_LABEL, get(d["reference"]), REF_COLOR)]
            series += [(lab, get(d["arms"][lab]), pale(lab)) for lab, _, _ in arms]
            if wide:
                # Separate large, medium and small bins, as in atom-type and ring-size,
                # so the near-zero aromatic peak does not flatten the other bins.
                start = len(fig.axes)
                _jsd_tiered(fig, labels, {label: label for label in labels},
                            [(lab, dict(zip(labels, vals)), col) for lab, vals, col in series],
                            ylabel, subplot_spec=rowspec[row, 0])
                axes = fig.axes[start:]
                row_labels.append((axes, xlabel))
                # Exact per-molecule means from the source, on a fraction axis rather
                # than the histogram's percentage-of-molecules axis.
                mean_ax = fig.add_subplot(rowspec[row, 1])
                mean_ax.set_facecolor("white")
                means = [(REF_LABEL, [d["reference"][mean_key]], REF_COLOR)]
                means += [(lab, [d["arms"][lab][mean_key]], pale(lab)) for lab, _, _ in arms]
                _jsd_grouped(mean_ax, ["Mean"], means)
                furniture(mean_ax, ylabel="Aromatic share", xlabel=" ", xloc=None)
                mean_ax.set_ylim(0, 1)
                mean_ax.yaxis.set_major_locator(MultipleLocator(0.25))
                mean_ax.grid(False, axis="x")
                mean_ax.tick_params(axis="x", length=0, labelsize=11)
            else:
                ax = fig.add_subplot(rowspec[row])
                ax.set_facecolor("white")
                _jsd_grouped(ax, labels, series)
                ax.set_ylim(0, max(float(np.max(v)) for _, v, _ in series) * 1.08)
                axes = [ax]
            for k, ax in enumerate(axes):
                # Reserve one shared row label before fitting the tiered panels.
                furniture(ax, ylabel=ylabel if k == 0 else None,
                          xlabel=" " if wide else xlabel, xloc=None)
                ax.tick_params(axis="x", length=0, labelsize=11 if wide else 10)
                ax.grid(False, axis="x")
                ax.xaxis.label.set_size(13.5)
        _jsd_key_above(fig, _jsd_handles(arms, patch=True), _jsd_key_cols(len(arms) + 1, wide))
        for axes, xlabel in row_labels:
            ax = axes[len(axes) // 2]
            center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
            ax.xaxis.label.set_text(xlabel)
            ax.xaxis.label.set_x((center - ax.get_position().x0) / ax.get_position().width)
        save(fig, out, f"aromatic_{variant}")

    rows = []
    sets = _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]
    for name, (_, get, mean_key, _) in zip(("atoms", "rings"), panels):
        rows += [[lab, name, _jsd_round(s[mean_key])] + [round(float(q), 3) for q in get(s)]
                 for lab, s in sets]
    write_csv(out, "aromatic", ["set", "aromatic_share_of", "mean"] + [f"pct_{b}" for b in labels], rows)


JSD_RING_ROWS = (
    ([str(k) for k in range(3, 10)] + ["≥10"], "Ring size\n% of rings", (0, 2, 4, 6, 7)),
    ([str(k) for k in range(JSD_NRINGS_MAX)] + [f"≥{JSD_NRINGS_MAX}"],
     "Rings per molecule\n% of molecules", (0, 2, 4, 6, 8)),
    (None, "Aromatic atom fraction\n% of molecules", None),
)


def _jsd_ring_rows(s):
    """The three Fig. 10 histograms of one set, in percent: ring sizes over ALL rings
    (10+ pooled), rings per molecule (8+ pooled), aromatic share of heavy atoms."""
    rs = {int(k): v for k, v in s["ring_size_counts"].items()}
    sizes = [rs.get(k, 0) for k in range(3, 10)] + [sum(v for k, v in rs.items() if k >= 10)]
    nr = {int(k): v for k, v in s["n_rings_counts"].items()}
    counts = [nr.get(k, 0) for k in range(JSD_NRINGS_MAX)] \
        + [sum(v for k, v in nr.items() if k >= JSD_NRINGS_MAX)]
    arom = np.asarray(s["arom_frac_counts"], float).reshape(-1, JSD_AROM_MERGE).sum(1)
    return [_jsd_pct(sizes), _jsd_pct(counts), _jsd_pct(arom)]


@figure("fig-jsd-rings", needs=JSD_NEEDS)
def draw_jsd_rings(out):
    """VoxBind Fig. 10: ring sizes, rings per molecule and aromatic fraction, one column per arm."""
    d, all_arms = _jsd_data()
    use_style()
    ref_rows = _jsd_ring_rows(d["reference"])
    for variant, arms in _jsd_variants(all_arms):
        n = len(arms)
        tick_fs = 10 if n > 2 else 11.5
        fig = _jsd_new(max(FIG_W, 2.05 * n + 1.5), 7.2)
        # The paper's layout: a column per method, the reference as the same dashed outline
        # in every panel, and y shared along a row so columns compare by eye. Column heads
        # name the method -- the one place this family puts text above an axes.
        axes = fig.subplots(3, n, sharey="row", squeeze=False)
        per = {lab: _jsd_ring_rows(d["arms"][lab]) for lab, _, _ in arms}
        for c, (lab, _, _) in enumerate(arms):
            rows = per[lab]
            for r, (labels, ylabel, ticks) in enumerate(JSD_RING_ROWS):
                ax = axes[r][c]
                ax.set_facecolor("white")
                if labels is None:
                    edges = np.linspace(0, 1, len(rows[r]) + 1)
                    x = (edges[:-1] + edges[1:]) / 2
                    width = (edges[1] - edges[0]) * 0.86
                else:
                    x = np.arange(len(rows[r]))
                    edges = np.arange(len(rows[r]) + 1) - 0.5
                    width = 0.78
                ax.bar(x, rows[r], width=width, color=pale(lab), lw=0, zorder=3)
                ax.stairs(ref_rows[r], edges, color=REF_COLOR, lw=REF_LW, ls=DASH,
                          baseline=None, zorder=4)
                furniture(ax, xloc=None, ylabel=ylabel if c == 0 else None)
                ax.grid(False, axis="x")
                ax.tick_params(labelsize=tick_fs)
                ax.yaxis.label.set_size(12)
                if labels is None:
                    ax.set_xlim(0, 1)
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xticklabels(["0", "0.5", "1"])
                else:
                    ax.set_xlim(edges[0], edges[-1])
                    ax.set_xticks(list(ticks))
                    ax.set_xticklabels([labels[i] for i in ticks])
            axes[0][c].set_title(display(lab), fontsize=13, color=INK, pad=7)
        # The row's y range is set ONCE, from every column and the reference. A per-panel
        # set_ylim(bottom=0) freezes the shared top at whatever the columns drawn so far
        # reached, and clipped FuncBind's 83% six-ring bar at AR's 70%.
        for r in range(len(JSD_RING_ROWS)):
            top = max(max(float(per[lab][r].max()) for lab in per), float(ref_rows[r].max()))
            axes[r][0].set_ylim(0, top * 1.08)
        _jsd_key_above(fig, [Line2D([], [], color=REF_COLOR, lw=REF_LW, ls=DASH,
                                    label=_jsd_ref_key())], 1)
        save(fig, out, f"rings_{variant}")

    rows = []
    names = ("ring_size", "rings_per_molecule", "aromatic_fraction")
    for lab, s in _jsd_ref_sets(d) + [(lab, d["arms"][lab]) for lab, _, _ in all_arms]:
        for name, (labels, _, _), pct in zip(names, JSD_RING_ROWS, _jsd_ring_rows(s)):
            if labels is None:
                labels = [f"{i / len(pct):.1f}-{(i + 1) / len(pct):.1f}" for i in range(len(pct))]
            rows += [[lab, name, b, round(float(q), 3)] for b, q in zip(labels, pct)]
    write_csv(out, "rings_hist", ["set", "histogram", "bin", "pct"], rows)


# ── the key, on its own ─────────────────────────────────────────────────────────
@figure("fig-jsd-legend", needs=JSD_NEEDS)
def draw_jsd_legend(out):
    """The key every other jsd figure dropped (2026-09-16, on request), drawn once.

    THE SHAPE IS THE RIGID-FRAGMENT BOX FIGURE'S: one row of filled swatches in a single
    thin frame, the crystal ligands first and named "Reference" -- not this family's old
    stacked key, because a key that stands alone is placed under or beside a panel.

    THE SWATCHES ARE THE PANELS': pale() tints in ARMS order, REF_COLOR grey for the
    reference, flat fills with no edge, exactly as _jsd_grouped draws its bars.

    The canvas starts at JSD_LEGEND_W x JSD_LEGEND_H and is then TRIMMED TO THE KEY, so the
    image is the key and nothing else: fit() cannot do it here, since a figure with no axes
    gives tight_layout nothing to measure."""
    _, all_arms = _jsd_data()
    use_style()
    fig = _jsd_new(JSD_LEGEND_W, JSD_LEGEND_H)
    swatches = [Patch(facecolor=REF_COLOR, lw=0, label="Reference")]
    swatches += [Patch(facecolor=pale(lab), lw=0, label=display(lab)) for lab, _, _ in all_arms]
    key = legend(fig, swatches, loc="center", ncol=len(swatches), fontsize=JSD_LEGEND_FS,
                 columnspacing=0.65)
    key.get_frame().set_linewidth(JSD_AXIS_LW_MAX)
    fig.canvas.draw()
    bb = key.get_window_extent(fig.canvas.get_renderer())
    if bb.width / fig.dpi > JSD_LEGEND_W:
        raise ValueError(f"the key is {bb.width / fig.dpi:.1f} in wide, past the "
                         f"{JSD_LEGEND_W} in canvas: raise JSD_LEGEND_W")
    fig.set_size_inches(bb.width / fig.dpi + 0.12, bb.height / fig.dpi + 0.12)
    print(f"  key: Reference + {len(all_arms)} methods in one row · "
          f"{bb.width / fig.dpi:.2f} x {bb.height / fig.dpi:.2f} in")
    # "jsd_legend", not "legend": the folder's own name is dropped from a file name here, but
    # this one is copied into a shared image folder beside other families' keys.
    save(fig, out, "jsd_legend")


def _jsd_count_distance(left, right):
    """Jensen-Shannon distance with natural logs, matching the stored JSD protocol."""
    p, q = np.asarray(left, float), np.asarray(right, float)
    if p.sum() == 0 or q.sum() == 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    mid = (p + q) / 2
    divergence = sum(float(np.sum(v[v > 0] * np.log(v[v > 0] / mid[v > 0])))
                     for v in (p, q)) / 2
    return float(np.sqrt(max(divergence, 0.0)))


@figure("fig-jsd-metric-summary", needs=JSD_NEEDS)
def draw_jsd_metric_summary(out):
    """Five horizontal panels summarizing method-level JSD."""
    d, arms = _jsd_data()
    labels = _pcsz_ordered([lab for lab, _, _ in arms])
    ref = d["reference"]
    # Include every observed ring size, including sizes above nine; no tail is discarded.
    ring_sizes = sorted(set(ref["ring_size_counts"]).union(
        *(set(d["arms"][lab]["ring_size_counts"]) for lab in labels)), key=int)
    ring_counts = sorted(set(ref["n_rings_counts"]).union(
        *(set(d["arms"][lab]["n_rings_counts"]) for lab in labels)), key=int)
    values = []
    for lab in labels:
        arm = d["arms"][lab]
        ring_jsd = _jsd_count_distance(
            [ref["ring_size_counts"].get(k, 0) for k in ring_sizes],
            [arm["ring_size_counts"].get(k, 0) for k in ring_sizes])
        count_jsd = _jsd_count_distance(
            [ref["n_rings_counts"].get(k, 0) for k in ring_counts],
            [arm["n_rings_counts"].get(k, 0) for k in ring_counts])
        values.append([arm["jsd"]["atom_type"], arm["jsd"]["pair"]["All_12A"],
                       arm["jsd"]["bond_mean"], ring_jsd, count_jsd])
    use_style()
    # Keep original panel sizes; expand the canvas only for inter-panel spacing.
    panel_w, panel_h = PCSZ_STRAIN_SIZE
    margins = PCSZ_COMPARISON_MARGINS
    left_w, right_w = 0.90, 0.14
    original_gap_w = 0.44
    # A BAR IS 1.6x ITS ORIGINAL WIDTH (2026-09-22, on request). A bar fills 0.78 of its
    # unit slot and there are eight slots, so a bar is only ever as wide as the panel lets
    # it be: the panel is what scales, and the bars, the air between them and the numbers
    # riding on top all scale with it. The inter-panel gap is deliberately NOT scaled -- it
    # was set to its own width just above, and this is a bar decision, not a layout one.
    bar_scale = 1.6
    axis_w = (2 * panel_w - left_w - right_w - 4 * original_gap_w) / 5 * bar_scale
    gap_w = original_gap_w * 2.0
    left_w += 0.15  # Clearance for the doubled JSD label padding.
    total_w = left_w + 5 * axis_w + 4 * gap_w + right_w
    # The 1.6x bars stretched the strip to eight times its height, which reads as a ribbon
    # rather than five panels, so the canvas is 1.2x taller (2026-09-22, on request). The
    # margins are fractions of it, so the panels and the air around them grow together; the
    # type does not, because nothing was asked of it.
    height_scale = 1.2
    fig = _jsd_new(total_w, panel_h * height_scale)
    axes = [fig.add_axes(((left_w + i * (axis_w + gap_w)) / total_w,
                          margins["bottom"], axis_w / total_w,
                          margins["top"] - margins["bottom"])) for i in range(5)]
    panel_ticks = ((0.04, 0.08, 0.12), (0.05, 0.10, 0.15),
                   (0.17, 0.34, 0.51),
                   (0.12, 0.24, 0.36), (0.08, 0.16, 0.24))
    xs = np.arange(len(labels))
    for j, (ax, title) in enumerate(zip(axes, ("Atom type", "Atom pair", "Bond length", "Ring size", "Ring count"))):
        vals = [row[j] for row in values]
        ticks = (0.0,) + panel_ticks[j]
        ceiling = float(ticks[-1])
        _pcsz_style(ax, grid_axis="y")
        ax.bar(xs, [np.nan if v is None else v for v in vals], width=0.78,
               color=[pale(lab) for lab in labels], zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, (v or 0) + ceiling * 0.025, "—" if v is None else f"{v:.3f}".removeprefix("0"),
                    ha="center", va="bottom", fontsize=7 * bar_scale, color=PCSZ_INK)
        ax.set_xticks([])
        ax.set_xlabel(title, fontsize=PCSZ_LABEL_FS, color=PCSZ_INK, labelpad=9)
        if j == 0:
            ax.set_ylabel("JSD", fontsize=PCSZ_LABEL_FS, fontweight="normal",
                          color=PCSZ_INK, labelpad=PCSZ_YPAD * 2)
        ax.set_ylim(0, ceiling)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.grid(False, axis="x")
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(1.1)
            ax.spines[side].set_zorder(5)
    # Trim the empty canvas under the panel titles at export (2026-09-24, on request);
    # the panels keep their size, so every other proportion is unchanged.
    fig._publication_trim_bottom = PCSZ_TRIM_PAD
    with plt.rc_context({"savefig.bbox": None, "savefig.dpi": 170}):
        save(fig, out, "metric_summary")
    write_csv(out, "metric_summary",
              ["method", "atom_type_js_distance", "atom_pair_all_12A_js_distance",
               "mean_bond_type_js_distance", "ring_size_js_distance", "ring_count_js_distance"],
              [[lab] + row for lab, row in zip(labels, values)])
    print("  five metrics: 8 methods; atom pairs All_12A; all observed ring sizes; natural-log JS distance")

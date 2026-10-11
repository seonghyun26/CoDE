"""fig-rigid-consistency: the 5 figures drawn into figures/fig-rigid-consistency/.

    python figures/draw.py consistency

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-consistency-rigid-per-size               -> fig-rigid-consistency/
    fig-consistency-rigid-box                    -> fig-rigid-consistency/
    fig-consistency-rigid-box-expanded           -> fig-rigid-consistency/
    fig-consistency-rigid-ecdf                   -> fig-rigid-consistency/
    fig-consistency-rigid-size-distribution      -> fig-rigid-consistency/
"""
from __future__ import annotations

import collections
import json
import math
import os
import statistics as st
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from draw import (
    ARMS, AXIS, AXIS_LW, AXIS_WIDTH, CORE, DASH, DATA_ALPHA, DIST_FILL, DIST_LW,
    FIG_W, GRID_LW, GRID_WIDTH, HEIGHT_RATIOS, H_PAD, INK, MIN_N, MODEL_LW, P79,
    PANEL_H, REF_COLOR, REF_LABEL, REF_LW, REF_ROOT, RF_BOX_AXIS_LW, RF_BOX_LEGEND_FS,
    RF_BOX_TICK_FS, STACK_H, TICK_SIZE, TITLE_SIZE, arm_handles, display, figure, fit,
    furniture, legend, model_curve, pale, save, share, use_style, write_csv, x_range,
)

# ════════════════════════════════════════════════════════════════════════════════
# fig-consistency-rigid-{per-size,box,ecdf} — rigid-fragment RMSD by fragment size
# ════════════════════════════════════════════════════════════════════════════════
# THE METRIC. A rigid fragment has no internal degrees of freedom -- a benzene ring, an
# amide, a fused bicycle each has exactly one correct shape, so a force field has no reason
# to move it. Optimise the molecule with MMFF, cut every rotatable bond, and RMSD each
# fragment against its own optimised self after Kabsch superposition. If it moved, the
# generated geometry was wrong to begin with. Lower is better; the unit is Angstrom.
#
# It is the complement of the two metrics next door. Strain (fig-posecheck) relaxes under a
# 0.1 A position constraint precisely so that local errors wash out and only GLOBAL
# conformational strain is reported -- exactly what this measures instead. PoseBusters
# (fig-posebusters) asks the same local question but only through pass/fail thresholds;
# this is continuous. A model can be good at one and bad at another, and here it is.
#
# WHAT IS ON THE X AXIS IS THE FRAGMENT, NOT THE LIGAND. Every other size-resolved figure
# here plots against heavy atoms in the ligand. This one cannot: the metric has no
# per-molecule value, only per-fragment ones, and fragment size is what drives it (a 2-atom
# fragment is a bond length, a 14-atom fragment is a fused ring system). So the axis, the
# distribution strip and MIN_N all count FRAGMENTS. Do not read a point here against a
# point in fig-posecheck at the same x.
#
# The evaluation itself is task2-ligand_generation/exps/frozenenc_probes/eval_rigid_fragments.py, which
# writes one `rigid_fragment_results.json` per run root and the crystal ligands'
# `rigid_fragment_reference.json` beside ours; this only draws what that wrote.
RF_RESULTS, RF_REFERENCE = "rigid_fragment_results.json", "rigid_fragment_reference.json"

# THE REFERENCE WINDOW IS NARROWER HERE. The shared `rolled` smooths the crystal ligands'
# share over +-REF_WIN=4 atoms because ligand sizes run 5-45; fragment sizes run 2-25 and
# the distribution is spiky (a third of all fragments have exactly 2 atoms, and 6 --
# benzene -- is the next spike), so a +-4 window would pool a bond length with a fused
# bicycle and smear a strip whose whole point is where the spikes are. The reference
# STATISTIC does not use a window at all -- see _rf_ref_curve.
RF_REF_WIN = 1
# The crystal reference is drawn per exact fragment size, where at least this many crystal
# fragments have that size.
RF_MIN_REF_EXACT = 8

RF_X_LABEL = "Number of heavy atoms in rigid fragment"
RF_XTICK = 2
RF_STATS = (("median", lambda v: float(np.median(v))),
            ("mean", lambda v: float(np.mean(v))))
# The box plot uses twelve bins, split into two rows: 3–8 and 9–14+.
# Keep the older coarse bins for the summary tables and historical comparisons only.
#
# `expanded` is the appendix version of `fine`: the same figure with its 14+ group taken
# apart. That group is doing a lot of work in the main figure -- fragment sizes run out to
# 64 heavy atoms, so one box pools a single fused bicycle with a whole polycyclic core and
# reports a mixture rather than a size. There, every size below the tail cut gets its own
# box and only the tail beyond it stays pooled.
#
# THE CUT IS COMPUTED FROM THE ARMS THE FIGURE DRAWS, and so is different in the two
# variants, because they run out in completely different places: VoxBind and CoDE both hold
# at least RF_MIN_DRAW fragments at every size up to 41, while FuncBind is under it from 27
# up and DiffSBDD and DecompDiff join it at 31. One fixed cut would either throw away the
# fifteen sizes the core figure can answer, or fill the all-arms figure with rows where
# most of the methods are missing.
#
# The rule: split sizes while at least RF_EXPANDED_COVER of the drawn arms clear
# RF_MIN_DRAW, and pool from the first size that fails. At three quarters a single thin arm
# may drop out of a bin -- FuncBind at 27, 29 and 30, DecompDiff at 28, AR at 29 -- and its
# box is simply absent there, which the run log names; it is half a panel going missing
# that the rule exists to prevent. Today that puts core at 42+ (40 bins, 7 rows) and all at
# 31+ (29 bins, 5 rows). The crystal ligands stop far earlier either way -- 294 fragments
# over the 79 pockets, 14 of them with 14 atoms or more -- so every row past the second
# carries models alone, as row 1 already does past 10 atoms.
RF_EXPANDED_COVER = 0.75
RF_BINNINGS = {
    "fine": ([(n, n + 1) for n in range(3, 14)] + [(14, 10 ** 6)],
             [str(n) for n in range(3, 14)] + ["14+"]),
    # rigid-consistent-box-all only (2026-09-23, on request): sizes out to 15, then 16+.
    "fine16": ([(n, n + 1) for n in range(3, 16)] + [(16, 10 ** 6)],
               [str(n) for n in range(3, 16)] + ["16+"]),
    "paper": ([(2, 3), (3, 5), (5, 7), (7, 10), (10, 14), (14, 10 ** 6)],
              ["2", "3–4", "5–6", "7–9", "10–13", "14+"]),
}
# What the main box figure exports; `expanded` writes its own table beside it rather than
# adding rows to one the paper's tables already read.
RF_CSV_BINNINGS = ("fine", "paper")
RF_PAPER_BINS = RF_BINNINGS["paper"][0]
# Below this, nothing is drawn at all. A box estimated from three values is unstable,
# it is two points and a line between them, and at a glance it reads exactly like a range
# built from thousands. The 79 crystal ligands hit this at fragment sizes 11-14, where they
# hold two to four fragments; every bin dropped this way is named in the run log.
RF_MIN_DRAW = 5
# The right-hand ECDF panel: where both models leave the crystal ligands behind.
RF_BIG_FRAG = 7
# Where the expanded figure starts splitting sizes -- the first bin of every binning here.
RF_FIRST_SIZE = 3
# Below this many labelled decades on the axis, the 2/3/5 minor ticks get labels too.
RF_MIN_DECADE_LABELS = 3


def _rf_log_y_ticks(ax):
    """A log y axis holding one or two decade labels leaves the reader almost no scale to
    read the curve against -- the mean panel spans 0.02 to 0.7 and shows `10^-1`, alone.
    Where that happens, label the 2/3/5 minor ticks as plain decimals as well.

    The test counts the labels the axis will actually SHOW, not the span of the data: a log
    axis autoscales out to the enclosing decades, so a 0.025-0.72 curve reports a span of
    ~1.7 decades while still carrying two labels."""
    lo, hi = ax.get_ylim()
    if len([t for t in ax.yaxis.get_majorticklocs() if lo <= t <= hi]) >= RF_MIN_DECADE_LABELS:
        return
    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10, subs=(2, 3, 5), numticks=20))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="y", which="minor", labelsize=12.5, colors=AXIS,
                   length=2.5, width=AXIS_LW)


def _rf_ref_curve(per, xs, f):
    """The crystal ligands at each EXACT fragment size, from xs[0] up to the first size
    with fewer than RF_MIN_REF_EXACT of them.

    NOT the shared reference_curve, and the difference matters. That windows the crystal
    ligands over +-REF_WIN because one ligand per pocket is too thin per LIGAND size. The
    same fix applied to fragment sizes was actively misleading here: the distribution
    spikes at 6 atoms (benzene), where the 33 crystal fragments sit at 0.027 A, and a +-1
    window pools that spike with the thin, worse-scoring 5s and 7s and lifts the plotted
    point to 0.041 -- ABOVE both models, reversing the comparison the line exists to make.

    79 crystal ligands give 294 fragments, which cannot support a per-size curve past ~10
    atoms. So the line stops there rather than being smoothed into reaching further, and
    the pooled 7+ comparison is made in the ECDF figure instead, where every fragment
    counts once.
    """
    out = []
    for a in xs:
        v = per.get(a, ())
        if len(v) < RF_MIN_REF_EXACT:
            break
        out.append(f(v))
    return out + [None] * (len(xs) - len(out))


def _rf_rolled(values, xs):
    """The shared `rolled`, over RF_REF_WIN instead of REF_WIN -- see the note on that
    constant. Written out rather than setting the global, because every other figure in
    this file is drawn in the same process and reads ligand sizes, not fragment sizes."""
    index = {a: i for i, a in enumerate(xs)}
    return [st.mean([values[index[n]] for n in xs if abs(n - a) <= RF_REF_WIN])
            for a in xs]


def _rf_contiguous(xs):
    """The leading run of consecutive sizes. x_range keeps every count where all arms clear
    MIN_N, which for LIGAND sizes is a dense range; fragment sizes are not dense out in the
    tail -- 25, 26 and 27 fall under the floor while 28 (a common fused system) clears it --
    and a line drawn from 24 to 28 would cross three counts that were dropped for being too
    thin. Truncating is the same rule the rest of the file follows: no point is invented,
    and nothing is interpolated over a gap."""
    for i, (a, b) in enumerate(zip(xs, xs[1:])):
        if b != a + 1:
            return xs[:i + 1]
    return xs


# ── load ─────────────────────────────────────────────────────────────────────────
def _rf_load(path):
    """{target: {n_mols, n_mmff_failed, fragments:[(size, rmsd), ...]}} for every target
    the run scored. Filtering to the 79 happens in the caller, so the pooled all-pockets
    numbers stay available."""
    with open(path, encoding="utf-8") as fh:
        j = json.load(fh)
    return {t["target"]: t for t in j["per_target"]}


def _rf_pooled(per_target, targets=None):
    """(fragment rows, molecule count, MMFF failure count) over a set of targets."""
    names = sorted(per_target) if targets is None else [t for t in targets
                                                        if t in per_target]
    rows = [tuple(f) for t in names for f in per_target[t]["fragments"]]
    return (rows,
            sum(per_target[t]["n_mols"] for t in names),
            sum(per_target[t]["n_mmff_failed"] for t in names))


def _rf_by_frag_size(rows):
    out = collections.defaultdict(list)
    for n, v in rows:
        out[n].append(v)
    return out


def _rf_scored(root):
    """(per-target rows, connected_only) for a run root, or None if it has not been scored
    over all 79 pockets. An arm mid-evaluation must not become a curve drawn over part of
    the data -- the same rule arms_for applies to the metrics tree."""
    path = os.path.join(root, RF_RESULTS)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        j = json.load(fh)
    per = {t["target"]: t for t in j["per_target"]}
    if any(t not in per for t in P79):
        return None
    return per, bool(j["summary"].get("connected_only", False))


_RF_CACHE = {}


def _rf_data():
    """(arms scored, {key: {size: [rmsd]}}, the reference's, {key: pooled}, ref pooled).

    Cached for the life of the process, like pose_data -- the three rigid-fragment figures
    read the same ~200k fragments over eight run trees."""
    if "d" not in _RF_CACHE:
        t0 = time.time()
        arm_data, mode, arms = {}, {}, []
        for a in ARMS:
            got = _rf_scored(a[2])
            if got is None:
                continue
            arm_data[a[1]], mode[a[0]] = got
            arms.append(a)
        if not arms:
            raise SystemExit(f"no arm has {RF_RESULTS} over all {len(P79)} pockets — run "
                             f"task2-ligand_generation/exps/frozenenc_probes/eval_rigid_fragments.py first")
        # Every arm must have been scored the same way. metrics.py drops molecules that are
        # not one connected component, so an arm scored WITH --connected-only and one
        # scored without are counting different molecules, and a figure drawn across both
        # would compare two different questions without saying so.
        if len(set(mode.values())) > 1:
            raise SystemExit("arms disagree on --connected-only: "
                             + ", ".join(f"{k}={v}" for k, v in mode.items())
                             + " — re-run eval_rigid_fragments.py so they match")
        ref_frags = _rf_pooled(_rf_load(os.path.join(REF_ROOT, RF_REFERENCE)), P79)
        p79_frags = {key: _rf_pooled(arm_data[key], P79) for key in arm_data}
        per = {key: _rf_by_frag_size(p79_frags[key][0]) for key in arm_data}
        ref_per = _rf_by_frag_size(ref_frags[0])
        _RF_CACHE["d"] = (arms, per, ref_per, p79_frags, ref_frags)
        n = sum(len(v[0]) for v in p79_frags.values())
        print(f"  [loaded {n:,} rigid fragments over {len(arms)} arms "
              f"in {time.time() - t0:.1f}s]")
    return _RF_CACHE["d"]


def _rf_variants(arms):
    """(name, arms), core first -- the shared `variants` over the arms that were scored."""
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


# ── figure 1: RMSD per fragment size, over the fragment-size distribution ────────
def _rf_size_distribution(ax, xs, arms, per, ref_per):
    """Where each arm puts its fragments, as a share of its own -- the same panel the
    shared size_distribution draws, with the fill dropped once the figure carries more than
    the core arms. Filled steps read well for two or three series; at nine they stack into
    one opaque mass and the arm you are looking for is the one you cannot see. The
    reference keeps a rolled share, over RF_REF_WIN."""
    fill = len(arms) <= len(CORE)
    for lab, key, _ in arms:
        pct, _ = share(per[key], xs)
        col = pale(lab)
        ax.step(xs, pct, where="mid", color=col, lw=DIST_LW, zorder=3)
        if fill:
            ax.fill_between(xs, pct, step="mid", color=col, alpha=DIST_FILL, lw=0,
                            zorder=2)
    raw, _ = share(ref_per, xs)
    ax.plot(xs, _rf_rolled(raw, xs), color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4)
    ax.set_ylim(bottom=0)


def _rf_per_size_fig(out, xs, arms, variant, stat, f, per, ref_per):
    """One statistic, one panel, one file -- the rule the 3-line figures set. The median is
    the one to read: the RMSD distribution inside a size is heavy-tailed (a handful of
    fragments the force field rebuilds outright sit an order of magnitude above the bulk),
    so the mean tracks that tail rather than the typical fragment. Both are drawn, apart,
    and the filename says which."""
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")

    top.plot(xs, _rf_ref_curve(ref_per, xs, f), color=REF_COLOR, lw=REF_LW,
             ls=DASH, zorder=4, dash_capstyle="round")
    for lab, key, _ in arms:
        top.plot(xs, model_curve(per[key], xs, f), color=pale(lab), lw=MODEL_LW,
                 zorder=5, solid_capstyle="round")
    top.set_yscale("log")
    # Two lines, as the strain figures do it: on one line the label is taller than this
    # panel and runs into the distribution strip's own label.
    furniture(top, ylabel=f"Fragment RMSD {stat}\n(Å)",
              xlim=(xs[0] - 0.4, xs[-1] + 0.4), xloc=RF_XTICK)
    _rf_log_y_ticks(top)
    # Lower right, not the house default upper left: every curve climbs to the right, so
    # upper left is where TargetDiff's rise is and the opaque legend box would cover it.
    legend(top, arm_handles(arms), loc="lower right",
           ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)

    _rf_size_distribution(bot, xs, arms, per, ref_per)
    furniture(bot, ylabel="% of fragments", xlabel=RF_X_LABEL,
              xlim=(xs[0] - 0.4, xs[-1] + 0.4), xloc=RF_XTICK)
    fig.align_ylabels((top, bot))
    fit(fig, pad=0.5, h_pad=H_PAD)
    save(fig, out, f"rigid_per_size_{stat}_{variant}")


# ── the run log's pooled numbers ────────────────────────────────────────────────
# An arm gets a standardised number only if the sizes it clears MIN_N at account for at
# least this much of the common mix. Otherwise the re-weighting is not a correction, it is
# a different question answered on whatever sizes happened to survive.
RF_STD_COVER = 0.90


def _rf_std_coverage(per, std_w):
    """The share of the common size mix this arm can answer at MIN_N."""
    total = sum(std_w.values())
    return sum(w for n, w in std_w.items()
               if len(per.get(n, ())) >= MIN_N) / total if total else 0.0


def _rf_standardised(per, std_w):
    """The arm's median at each fragment size, averaged over the COMMON size mix.

    The pooled median cannot be compared across arms and the numbers here are the reason
    why, not a hypothetical: DecompDiff's pooled median (0.043) sits below Ours (0.047)
    while being worse at every fragment size from 5 atoms up -- it simply draws a different
    mix of sizes, and small fragments score better. This re-weights every arm onto one mix,
    so the summary number answers "how good is this arm's geometry" instead of "what sizes
    does this arm happen to make". It is an average of medians, not a median: a median does
    not re-weight."""
    num = den = 0.0
    for n, w in std_w.items():
        v = per.get(n, ())
        if len(v) >= MIN_N:
            num += w * float(np.median(v))
            den += w
    total = sum(std_w.values())
    # The 79 crystal ligands are the case this guard exists for: they clear MIN_N at four
    # fragment sizes out of nine, and those four carry most of the common mix's weight and
    # are the sizes crystal geometry is best at. A number built from them would read as a
    # like-for-like summary and would not be one.
    return num / den if den and den / total >= RF_STD_COVER else None


def _rf_block(rows, n_mols, n_failed, std_w):
    """One arm's pooled numbers for the run log, over the paper bins."""
    v = [r[1] for r in rows]
    per = _rf_by_frag_size(rows)
    std = _rf_standardised(per, std_w) if rows else None
    binned = {}
    for lo, hi in RF_PAPER_BINS:
        sel = [r[1] for r in rows if lo <= r[0] < hi]
        if sel:
            name = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
            binned[name] = {"n": len(sel), "median": round(st.median(sel), 4),
                            "mean": round(float(np.mean(sel)), 4)}
    return {
        "n_molecules": n_mols,
        "mmff_failure_rate_pct": round(100 * n_failed / n_mols, 2) if n_mols else None,
        "n_fragments": len(rows),
        "rmsd_median": round(st.median(v), 4) if v else None,
        "rmsd_median_size_standardised": None if std is None else round(std, 4),
        "size_standardised_coverage": round(_rf_std_coverage(per, std_w), 3)
        if rows else None,
        "rmsd_mean": round(float(np.mean(v)), 4) if v else None,
        "by_fragment_size": binned,
    }


@figure("fig-consistency-rigid-per-size",
        needs=("rigid_fragment_results.json", "rigid_fragment_reference.json"))
def draw_consistency_rigid_per_size(out):
    """Rigid-fragment RMSD against fragment size, mean and median, each over the arms'
    fragment-size distribution."""
    arms, per, ref_per, p79_frags, ref_frags = _rf_data()
    use_style()
    ranges = {}
    for variant, vs in _rf_variants(arms):
        xs = _rf_contiguous(x_range(per, vs))
        ranges[variant] = xs
        for stat, f in RF_STATS:
            _rf_per_size_fig(out, xs, vs, variant, stat, f, per, ref_per)

    xs = ranges["all"]
    fields = [f"rmsd_{stat}" for stat, _ in RF_STATS]
    curves = {lab: {"n": [len(per[key].get(a, ())) for a in xs],
                    **{f"rmsd_{stat}": model_curve(per[key], xs, f)
                       for stat, f in RF_STATS}}
              for lab, key, _ in arms}
    # The reference is per exact fragment size, null past the first size holding fewer than
    # RF_MIN_REF_EXACT crystal fragments; nothing is windowed.
    curves[REF_LABEL] = {"n": [len(ref_per.get(a, ())) for a in xs],
                         **{f"rmsd_{stat}": _rf_ref_curve(ref_per, xs, f)
                            for stat, f in RF_STATS}}
    write_csv(out, "rigid_fragment_per_size",
              ["arm", "fragment_atoms", "n"] + fields,
              [[arm, a, d["n"][i]]
               + ["" if d[f][i] is None else round(d[f][i], 4) for f in fields]
               for arm, d in curves.items() for i, a in enumerate(xs)])

    # One common fragment-size distribution, every arm pooled, for the size-standardised
    # summary. Sizes where an arm holds fewer than MIN_N fragments are dropped from ITS
    # average rather than extrapolated.
    std_w = collections.Counter(n for key in per for n, v in per[key].items() for _ in v)
    blocks = {lab: _rf_block(*p79_frags[key], std_w) for lab, key, _ in arms}
    blocks[REF_LABEL] = _rf_block(*ref_frags, std_w)

    print(f"  {len(P79)} pockets · "
          + " · ".join(f"{v} x {xs_[0]}-{xs_[-1]}" for v, xs_ in ranges.items())
          + f" (fragment sizes where every drawn arm has >={MIN_N} fragments,\n"
          + "    truncated at the first gap: "
          + ", ".join(f"{v} drops {sorted(set(x_range(per, a)) - set(ranges[v]))}"
                      for v, a in _rf_variants(arms)) + ")\n")
    print(f"  {'arm':16s} {'frags':>7s} {'med':>7s} {'std med':>8s} {'mean':>7s} "
          f"{'mols':>6s} {'MMFF fail':>10s}")
    for lab, r in blocks.items():
        std = r["rmsd_median_size_standardised"]
        print(f"  {lab:16s} {r['n_fragments']:7d} {r['rmsd_median']:7.4f} "
              f"{('%8.4f' % std) if std is not None else '   n/a  '} "
              f"{r['rmsd_mean']:7.4f} {r['n_molecules']:6d} "
              f"{r['mmff_failure_rate_pct']:9.1f}%"
              + ("" if std is not None else
                 f"   (covers {100 * r['size_standardised_coverage']:.0f}% of the mix)"))

    print(f"\n  {'fragment size':>14s} " + "".join(f"{l:>18s}" for l in blocks))
    for name in blocks[REF_LABEL]["by_fragment_size"]:
        cells = ""
        for r in blocks.values():
            b = r["by_fragment_size"].get(name)
            cells += f"{b['median']:10.4f} ({b['n']:>5d})" if b else f"{'':>18s}"
        print(f"  {name:>14s} {cells}")


# ── figure 2: grouped box plots, as in VoxBind Fig. 11b ────────────────────────
# Compute quartiles and Tukey whiskers on the original RMSDs, in Angstroms.
# Outlier points are hidden for readability; all observations enter the statistics.
RF_BOX_WHIS = 1.5
RF_BOX_BINS_PER_ROW = 6
RF_BOX_MAX_FIG_W = 18.0
RF_BOX_YMAX = (0.6, 1.2)
# The expanded figure has no fixed ceilings, because it no longer has a fixed number of
# rows: it takes one per row, the smallest round number clearing that row's quartiles.
# RMSD grows with fragment size, so a single shared limit would flatten the first rows into
# a band at the bottom. Only the longest whiskers are clipped, which is what the fixed
# limits do in the main figure too -- the box statistics are unchanged either way. The
# first row keeps the main figure's 0.6 as a floor, so the two figures start on one scale.
RF_BOX_YMIN = RF_BOX_YMAX[0]
# Variants of the main box figure drawn WITHOUT the method key (2026-09-23, on request):
# `all` goes beside fig-jsd-legend, the same one-row key drawn once on its own.
RF_BOX_KEYLESS = ("all",)
# Per-variant shape of the main box figure: (binning, row ceilings, bins per row, width
# scale); a variant not named here keeps ("fine", RF_BOX_YMAX, RF_BOX_BINS_PER_ROW, 1.0).
# `all` (2026-09-23, on request) runs to 16+ on TWO rows, so seven a row, on a canvas 1.2x
# as wide with the type left at the unwidened canvas's size.
RF_BOX_VARIANT_SHAPE = {"all": ("fine16", (0.6, 1.2), 7, 1.2)}
# A box row is taller than the shared panel height by a tenth. What a reader compares here
# is the box and the median line inside it, and at PANEL_H those sit in the bottom third of
# a row whose upper half is whisker -- the rows carry the same RMSD range either way, so
# the extra tenth goes straight into the part that is being read.
RF_BOX_ROW_H = PANEL_H * 1.1
# THE EXPANDED FIGURES ARE SHAPED DIFFERENTLY, because they carry a different number of
# bins: one per fragment size is 29 bins for `all` and 40 for `core`, and at six a row and
# a full-height row that is a canvas 18 and 25 inches tall -- a figure that can only reach
# a page shrunk to a third, where every box is a hairline and the numeric ticks are gone.
# Ten bins a row and a row four fifths as tall bring both of them to about half that
# height, at the same canvas width, so a box keeps its readable size on the page. Ten fits
# because these labels are one number wide where the main figure's are ranges ("15" against
# "9-10"), and because the rows are drawn over the same bin width either way -- what
# changes is how many of them are on a row, not how wide one is.
RF_EXPANDED_BINS_PER_ROW = 10
RF_EXPANDED_ROW_SCALE = 0.8
# FOUR ROWS, EACH 1.3x AS TALL (2026-09-23, on request): the expanded box plot and its size
# distribution now fix the row COUNT and let the bins per row follow -- ceil(n / 4), so
# `all` and `core` both come out as four rows -- instead of fixing ten a row. The row height
# is RF_EXPANDED_ROW_SCALE x 1.3. RF_EXPANDED_BINS_PER_ROW above is kept as the old shape.
# THREE ROWS (2026-09-23, on request): back from four to three, row height kept at 1.3x.
RF_EXPANDED_ROWS = 3
# Where a variant's pooled tail starts regardless of the coverage rule (2026-09-23, on
# request): `all` would pool at 31+, but Pocket2Mol, TargetDiff, VoxBind and CoDE still
# hold 8-33 fragments at 31 and 32, so it splits those two out and pools at 33+. The arms
# that fall under RF_MIN_DRAW there simply have no box, as the run log lists.
# Pooled at 32+ (2026-09-23, on request): 3..31 plus 32+ is 30 bins = three full rows of 10.
RF_EXPANDED_TAIL = {"all": 32}
# The expanded box plot's canvas widened per variant, the type kept (2026-09-23, on request).
RF_EXPANDED_WIDTH_SCALE = {}          # {"all": 1.1} was tried and reverted
RF_EXPANDED_ROW_H_SCALE = RF_EXPANDED_ROW_SCALE * 1.3


def _rf_expanded_per_row(labels):
    """Bins per row that put `labels` on RF_EXPANDED_ROWS rows."""
    return max(1, math.ceil(len(labels) / RF_EXPANDED_ROWS))
RF_BOX_LINE_LW, RF_BOX_MEDIAN_LW = 1.2, 1.6
RF_BOX_LABEL_FS = RF_BOX_TICK_FS * 1.4

# ...BUT THE TYPE SIZE IS DECIDED AT THE EXPORT BOUNDARY, one absolute size for every
# figure in the tree (publication_style.TICK_SIZE and friends, applied in save()). That is
# right while a figure is about as wide as the ECDF pair those sizes were approved on.
# These are not: nine methods over six bins a row pins the canvas to RF_BOX_MAX_FIG_W, half
# as wide again, so the same 17.36 pt lands two thirds the size on the page as it does in
# fig-jsd/metric-summary or the ECDF pair next door -- the ticks and axis names read small
# in exactly the figures that are hardest to read already.
#
# So a canvas wider than the reference scales the type, the rules and the gaps with it, and
# says so at the boundary (`_publication_style_applied`) rather than being overridden back
# down. A canvas at or below the reference -- every `core` variant, three series wide -- is
# left exactly as the rest of the tree draws it, shared sizes and all. Scaling only ever
# goes up, so no figure here can end up smaller than the house size.
# THE KEY IS NOT SCALED WITH THE REST. Everything else here is a mark inside a panel and
# grows with the panel, but the key is one row of nine method names across the foot of the
# figure: it is already as wide as the canvas at RF_BOX_LEGEND_FS, and at the scaled size
# it runs 1.4 canvases wide, whereupon fit() reserves the overflow and squeezes the panels
# into the middle half of the figure. It stays at the family's own size, which is the size
# it is in every other figure in the family.
RF_STYLE_REF_W = 11.89       # inches: the ECDF pair as publication_style leaves it
RF_TYPE = collections.namedtuple("RF_TYPE", "scale tick title axis grid pad")
RF_PAD = 18.0                # the gap under the x label / beside the y label, in points


def _rf_type(width):
    """The type sizes, rule widths and gaps for a canvas `width` inches across."""
    k = width / RF_STYLE_REF_W
    if k <= 1:
        return RF_TYPE(1.0, RF_BOX_TICK_FS, RF_BOX_LABEL_FS, RF_BOX_AXIS_LW,
                       GRID_LW * 1.2, RF_PAD)
    return RF_TYPE(k, TICK_SIZE * k, TITLE_SIZE * k, AXIS_WIDTH * k,
                   GRID_WIDTH * k, RF_PAD * k)


def _rf_binning(binning):
    """(bins, labels), from a name in RF_BINNINGS or from an already-built pair. The
    expanded figure builds its own per variant, so a binning is no longer always a name --
    which is also why the two lists are checked here rather than trusted: zip() would
    silently drop the tail of whichever one a later edit left shorter."""
    bins, labels = RF_BINNINGS[binning] if isinstance(binning, str) else binning
    if len(bins) != len(labels):
        raise ValueError(f"binning has {len(bins)} bins but {len(labels)} labels")
    return bins, labels


def _rf_expanded_binning(arms, per, tail=None):
    """The per-size bins for `arms`, pooled from the first size RF_EXPANDED_COVER of them
    cannot answer -- see the note on that constant.

    With no arms at all there is nothing to take a cut from, and the coverage test would
    read 0 of 0 as full coverage and split sizes for ever. That is reachable: `core` is
    VoxBind's own two runs, and a tree where only the baselines have been scored leaves it
    empty. One pooled bin is the honest answer there, and it draws the crystal ligands
    alone, as every other figure in this file does when a variant has no arms."""
    n = RF_FIRST_SIZE
    while arms and sum(1 for _, key, _ in arms
                       if len(per[key].get(n, ())) >= RF_MIN_DRAW) \
            >= RF_EXPANDED_COVER * len(arms):
        n += 1
    if tail is not None:
        n = tail
    return ([(a, a + 1) for a in range(RF_FIRST_SIZE, n)] + [(n, 10 ** 6)],
            [str(a) for a in range(RF_FIRST_SIZE, n)] + [f"{n}+"])


# Round numbers to put an axis ceiling on, every one of them divisible by three so that the
# four ticks the rows ask for land on clean values (0.6 -> 0.2, 0.75 -> 0.25, 36 -> 12).
# Rounded, because 6.0 * 10 ** -1 is 0.6000000000000001 and that is what the tick would
# be laid out from -- the same ceiling has to come out of the ladder and out of RF_BOX_YMAX
# as one number.
# 7.5 is deliberately not a rung. It is the one mantissa here whose quarter step (2.5) is
# not itself a round number, and the 9-14 row of the expanded box plot landed on it: ticks
# at 0, 0.25, 0.50, 0.75 in a figure whose other rows step 0.2, 0.4, 0.5 and 0.6. Without
# it that row takes 0.9 and steps 0.3, which is what the row is asked to read.
RF_NICE = sorted(round(m * 10 ** k, 10) for k in range(-3, 4)
                 for m in (1.2, 1.5, 1.8, 2.4, 3.0, 3.6, 4.5, 6.0, 9.0))
RF_CEIL_AIR = 1.05


def _rf_ceiling(v, floor=0.0):
    """The smallest round ceiling that clears `v` with a little air above it. The ladder
    covers RMSDs in Angstroms and shares in percent with room to spare; running off the end
    of it says the caller is passing something else, so it says so rather than raising a
    bare StopIteration out of the generator."""
    over = [c for c in RF_NICE if c >= v * RF_CEIL_AIR]
    if not over:
        raise ValueError(f"no round ceiling for {v} -- above the top of RF_NICE")
    return max(floor, over[0])


def _rf_bin_values(rows, binning="paper"):
    """{bin label: np.array of RMSDs}."""
    bins, labels = _rf_binning(binning)
    return {lab: np.array([r[1] for r in rows if lo <= r[0] < hi])
            for (lo, hi), lab in zip(bins, labels)}


def _rf_box_stats(values):
    """One box's statistics in data space, also exported beside the figure."""
    return matplotlib.cbook.boxplot_stats(values, whis=RF_BOX_WHIS)[0]


RF_BOX_XLABEL = "Rigid fragment size (Number of heavy atoms)"


def _rf_row_axes(ax, labs, ymax, t, xlabel=None, ylabel=None,
                 per_row=RF_BOX_BINS_PER_ROW):
    """The furniture of one row of grouped bins. The box rows and the expanded size
    distribution's rows are the same axis carrying different marks, drawn to the same width
    over the same labels, so that the two figures can be read one above the other."""
    furniture(ax, xloc=None, xlabel=xlabel, ylabel=ylabel)
    ax.grid(False, axis="x")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(t.axis)
        ax.spines[spine].set_zorder(5)
    ax.tick_params(width=t.axis, labelsize=t.tick, length=5 * t.scale)
    ax.xaxis.label.set_size(t.title)
    ax.yaxis.label.set_size(t.title)
    ax.grid(True, axis="y", linewidth=t.grid)
    # Fixed display limits clip whiskers without altering the raw box statistics.
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.linspace(0, ymax, 4))
    # Keep all requested ticks, including bins with no drawable reference box.
    ax.set_xticks(np.arange(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_xlim(-0.58, per_row - 0.42)


def _rf_rows_of(labels, per_row=RF_BOX_BINS_PER_ROW):
    return [labels[i:i + per_row] for i in range(0, len(labels), per_row)]


def _rf_stack_width(n_series, per_row=RF_BOX_BINS_PER_ROW):
    """One bin is the same width in every figure of this family, so the boxes and the size
    distribution line up column by column."""
    return min(RF_BOX_MAX_FIG_W,
               max(FIG_W * 1.52, 2.3 + 0.30 * n_series * per_row))


def _rf_stack_ylabel(fig, axes, ylabel, pad=RF_PAD):
    """Centre a figure-level y label on a stack of rows and clear it of the ticks -- a
    figure-level label needs its own clearance from the wider numeric ticks."""
    ylabel.set_y((axes[0].get_position().y1 + axes[-1].get_position().y0) / 2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_left = min(t.get_window_extent(renderer).x0 for ax in axes
                    for t in ax.get_yticklabels() if t.get_visible())
    clearance = ylabel.get_window_extent(renderer).x1 + pad * fig.dpi / 72 - tick_left
    if clearance > 0:
        fig.subplots_adjust(left=fig.subplotpars.left + clearance / fig.bbox.width)


def _rf_box(out, arms, variant, p79_frags, ref_frags, binning="fine",
            ymaxes=RF_BOX_YMAX, stem="rigid_box", per_row=RF_BOX_BINS_PER_ROW,
            row_h=RF_BOX_ROW_H, show_key=True, width_scale=1.0):
    """Fragment-size groups in rows of `per_row` bins, with one method legend below
    (`show_key=False` drops it; see RF_BOX_KEYLESS).

    `binning` chooses the groups -- "fine" for the main figure's twelve, a built pair for
    the appendix's per-size bins -- and `ymaxes` carries one display ceiling per row, or
    None to take the smallest round one that clears each row's quartiles. `per_row` and
    `row_h` are the appendix figure's flatter rows. Everything else, the box statistics
    included, is the same figure drawn to a different shape."""
    labels = _rf_binning(binning)[1]
    rows_of_labels = _rf_rows_of(labels, per_row)
    series = [(REF_LABEL, REF_COLOR, ref_frags[0])] + \
             [(lab, pale(lab), p79_frags[key][0]) for lab, key, _ in arms]
    per = [(lab, col, _rf_bin_values(rows, binning)) for lab, col, rows in series]
    stats = {lab: {l: _rf_box_stats(v) for l, v in vals.items()
                   if len(v) >= RF_MIN_DRAW} for lab, _, vals in per}
    # Empty bins are named too, not just thin ones: past the second row an arm can simply
    # have no fragment of that size, and a gap a reader cannot tell from a drawing mistake
    # is the thing the log exists to rule out.
    skipped = {lab: [l for l, v in vals.items() if len(v) < RF_MIN_DRAW]
               for lab, _, vals in per}
    if ymaxes is None:
        ymaxes = tuple(_rf_ceiling(max([s["q3"] for by in stats.values()
                                        for l, s in by.items() if l in labs] or [0]),
                                   RF_BOX_YMIN)
                       for labs in rows_of_labels)
    if len(ymaxes) != len(rows_of_labels):
        raise ValueError(f"{len(rows_of_labels)} rows of boxes but {len(ymaxes)} y limits")
    slot = 0.86 / len(series)
    width = _rf_stack_width(len(series), per_row)
    # The main all-method box panel follows the interaction-pair-all axis scale.
    # Keep its existing canvas dimensions; only axis type and strokes change.
    match_pair_style = stem == "rigid_box" and variant == "all"
    t = (RF_TYPE(1.0, TICK_SIZE, TITLE_SIZE, AXIS_WIDTH, GRID_WIDTH, RF_PAD)
         if match_pair_style else _rf_type(width))
    width *= width_scale        # after the type: widening the canvas does not grow the text
    nrows = len(rows_of_labels)
    fig, axes = plt.subplots(nrows, 1, dpi=220,
                             figsize=(width, nrows * row_h + 0.8))
    fig.patch.set_facecolor("white")

    for ax, labs, ymax in zip(axes, rows_of_labels, ymaxes):
        ax.set_facecolor("white")
        idx = np.arange(len(labs))
        for i, (lab, col, _) in enumerate(per):
            pos = idx - 0.43 + slot * (i + 0.5)
            keep = [(x, stats[lab][l]) for x, l in zip(pos, labs) if l in stats[lab]]
            if not keep:
                continue
            ax.bxp([v for _, v in keep], positions=[x for x, _ in keep],
                   widths=slot * 0.82, patch_artist=True, manage_ticks=False,
                   showfliers=False, showmeans=False,
                   boxprops={"facecolor": matplotlib.colors.to_rgba(col, DATA_ALPHA),
                             "edgecolor": INK, "linewidth": 0,
                             "hatch": "///" if lab == REF_LABEL else None},
                   whiskerprops={"color": INK, "linewidth": RF_BOX_LINE_LW},
                   capprops={"color": INK, "linewidth": RF_BOX_LINE_LW},
                   medianprops={"color": INK, "linewidth": RF_BOX_MEDIAN_LW}, zorder=3)

        _rf_row_axes(ax, labs, ymax, t, per_row=per_row,
                     xlabel=RF_BOX_XLABEL if ax is axes[-1] else None)

    ylabel = fig.supylabel("Fragment RMSD (Å)", fontsize=t.title, color=AXIS)
    reserved = 0
    if show_key:
        swatches = [Patch(facecolor=matplotlib.colors.to_rgba(col, DATA_ALPHA), edgecolor=INK,
                          linewidth=0, hatch="///" if lab == REF_LABEL else None,
                          label="Reference" if lab == REF_LABEL else display(lab))
                    for lab, col, _ in per]
        key = legend(fig, swatches, loc="lower center", ncol=len(swatches),
                     fontsize=RF_BOX_LEGEND_FS, bbox_to_anchor=(0.5, 0.012),
                     columnspacing=0.65)
        # No frame on this key. The house rule puts an opaque white box with a rule around a
        # legend so the dotted grid cannot run through its labels; this one sits below the
        # panels, on white, with nothing behind it to be separated from, and the rule then
        # reads as a second axis frame under a figure that has no frames anywhere else.
        key.set_frame_on(False)
        # Reserve the legend plus the label gap above its top edge for the x-axis label.
        fig.canvas.draw()
        reserved = (key.get_window_extent().y1 / fig.dpi + t.pad / 72) / fig.get_figheight()
    fit(fig, pad=0.5, h_pad=H_PAD, rect=(0, reserved, 1, 1))
    _rf_stack_ylabel(fig, axes, ylabel, t.pad)
    if match_pair_style:
        fig._publication_style_applied = True  # avoid export-time canvas growth
    else:
        _rf_keep_type(fig, t)
    save(fig, out, f"{stem}_{variant}")
    return skipped


def _rf_keep_type(fig, t):
    """Tell the export boundary this figure is already styled, where it was drawn to its
    own scale. Only for a scaled canvas: everything else still goes through the shared
    sizes, and through the layout repair that comes with them."""
    if t.scale > 1:
        fig._publication_style_applied = True


def _rf_gone(labels, skipped):
    """The bins an arm has no box in, consecutive ones collapsed to a range. The crystal
    ligands are missing from every bin past 10, and thirty of those listed one at a time
    bury the one model that dropped a single box in the middle of a row."""
    order = {lab: i for i, lab in enumerate(labels)}
    out = []
    for lab, bins in skipped.items():
        if not bins:
            continue
        runs = []
        for b in sorted(bins, key=order.__getitem__):
            if runs and order[b] == order[runs[-1][-1]] + 1:
                runs[-1].append(b)
            else:
                runs.append([b])
        # Every run of two or more becomes a dash: "27, 28" would use the same comma that
        # separates one run from the next, and the reader could not tell which it was.
        out.append(f"{lab} at " + ", ".join(
            r[0] if len(r) == 1 else f"{r[0]}–{r[-1]}" for r in runs))
    return "; ".join(out)


def _rf_bin_csv(out, entries, p79_frags, ref_frags, stem):
    """One row per (binning, arm, bin): the counts, the quartiles and the whiskers the
    boxes were drawn from, so the figure can be checked against the raw RMSDs.

    An entry is (name, binning, arms), because the expanded figure's bins are the drawn
    arms' own -- `expanded_core` and `expanded_all` pool from a different size, and a table
    that did not say which would not be checkable against either figure."""
    rows = []
    for name, binning, arms in entries:
        for lab, key, _ in list(arms) + [(REF_LABEL, None, None)]:
            frags = ref_frags[0] if key is None else p79_frags[key][0]
            for b, v in _rf_bin_values(frags, binning).items():
                if not len(v):
                    continue
                box = _rf_box_stats(v)
                values = (v.min(), box["q1"], box["med"], box["mean"], box["q3"],
                          v.max(), box["whislo"], box["whishi"])
                rows.append([name, lab, b, len(v)]
                            + [round(float(x), 4) for x in values])
    write_csv(out, stem,
              ["binning", "arm", "fragment_size_bin", "n", "min", "q25", "median",
               "mean", "q75", "max", "whisker_low", "whisker_high"], rows)


@figure("fig-consistency-rigid-box",
        needs=("rigid_fragment_results.json", "rigid_fragment_reference.json"))
def draw_consistency_rigid_box(out):
    """Rigid-fragment RMSD box plots: 3–8 / 9–14+, with a bottom legend — core and all."""
    arms, per, ref_per, p79_frags, ref_frags = _rf_data()
    use_style()
    for variant, vs in _rf_variants(arms):
        binning, ymaxes, per_row, wscale = RF_BOX_VARIANT_SHAPE.get(
            variant, ("fine", RF_BOX_YMAX, RF_BOX_BINS_PER_ROW, 1.0))
        skipped = _rf_box(out, vs, variant, p79_frags, ref_frags, binning=binning,
                          ymaxes=ymaxes, per_row=per_row, width_scale=wscale,
                          show_key=variant not in RF_BOX_KEYLESS)
        gone = _rf_gone(RF_BINNINGS[binning][1], skipped)
        if gone:
            print(f"  rigid_box_{variant}: fewer than {RF_MIN_DRAW} fragments, so "
                  f"not drawn: {gone}")

    # Keep the historical summary columns and coarse bins; append the whiskers used
    # by the boxes so the plotted statistics can be checked against the raw RMSDs.
    _rf_bin_csv(out, [(b, b, arms) for b in RF_CSV_BINNINGS], p79_frags, ref_frags,
                "rigid_fragment_by_bin")


@figure("fig-consistency-rigid-box-expanded",
        needs=("rigid_fragment_results.json", "rigid_fragment_reference.json"))
def draw_consistency_rigid_box_expanded(out):
    """The box plots with 14+ taken apart: one box per fragment size, out to where the arms
    the figure draws run out, and one pooled tail — for the appendix."""
    arms, per, ref_per, p79_frags, ref_frags = _rf_data()
    use_style()
    tails, entries = {}, []
    for variant, vs in _rf_variants(arms):
        binning = _rf_expanded_binning(vs, per, RF_EXPANDED_TAIL.get(variant))
        tails[variant] = binning[0][-1][0]
        entries.append((f"expanded_{variant}", binning, vs))
        split = binning[1][:-1]
        print(f"  rigid_box_expanded_{variant}: "
              + ("nothing split by size" if not split else
                 f"one box at size {split[0]}" if len(split) == 1 else
                 f"one box per size {split[0]}–{split[-1]}")
              + f", {tails[variant]}+ pooled — {len(binning[1])} bins "
              f"over {len(_rf_rows_of(binning[1], _rf_expanded_per_row(binning[1])))} rows")
        skipped = _rf_box(out, vs, variant, p79_frags, ref_frags, binning=binning,
                          ymaxes=None, stem="rigid_box_expanded",
                          per_row=_rf_expanded_per_row(binning[1]),
                          row_h=RF_BOX_ROW_H * RF_EXPANDED_ROW_H_SCALE,
                          show_key=False,   # no key on either variant (2026-09-23)
                          width_scale=RF_EXPANDED_WIDTH_SCALE.get(variant, 1.0))
        gone = _rf_gone(binning[1], skipped)
        if gone:
            print(f"  rigid_box_expanded_{variant}: fewer than {RF_MIN_DRAW} fragments, "
                  f"so not drawn: {gone}")

    _rf_bin_csv(out, entries, p79_frags, ref_frags, "rigid_fragment_by_bin_expanded")

    # What the main figure's last box was hiding, in the run log: how far the tail actually
    # reaches, and how little of it there is. One column per cut, because the two variants
    # pool from a different size.
    cuts = sorted(set(tails.values()))
    print(f"\n  {'arm':16s} {'largest':>8s} {'14+':>8s} {'share':>7s}"
          + "".join(f" {str(t) + '+':>8s} {'share':>7s}" for t in cuts))
    for lab, key, _ in list(arms) + [(REF_LABEL, None, None)]:
        frags = ref_frags[0] if key is None else p79_frags[key][0]
        n = len(frags)
        big = sum(1 for a, _ in frags if a >= 14)
        tails_n = [sum(1 for a, _ in frags if a >= t) for t in cuts]
        print(f"  {lab:16s} {max(a for a, _ in frags):8d} {big:8d} "
              f"{100 * big / n:6.2f}%"
              + "".join(f" {t:8d} {100 * t / n:6.2f}%" for t in tails_n))


# ── figure 3: the distribution itself ───────────────────────────────────────────
@figure("fig-consistency-rigid-ecdf",
        needs=("rigid_fragment_results.json", "rigid_fragment_reference.json"))
def draw_consistency_rigid_ecdf(out):
    """Two panels on one shared y, the pose ECDF-pair layout. Left, every fragment; right,
    only the 7+ atom fragments. The split is there because the left panel is dominated by
    the 2- and 3-atom fragments -- more than half of every arm's total -- whose RMSD is a
    bond length and where all four lines sit on top of each other. The right panel is the
    same curve over the fragments that actually carry ring and conjugation geometry."""
    arms, per, ref_per, p79_frags, ref_frags = _rf_data()
    use_style()
    for variant, vs in _rf_variants(arms):
        fig, (left, right) = plt.subplots(1, 2, figsize=(FIG_W * 1.52, PANEL_H),
                                          dpi=220, sharey=True)
        fig.patch.set_facecolor("white")
        for ax in (left, right):
            ax.set_facecolor("white")

        def ecdf(ax, keep):
            def draw(rows, col, lw, ls):
                v = np.sort(np.clip([r[1] for r in rows if keep(r[0])], 1e-4, None))
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=lw, ls=ls,
                        zorder=4 if ls != "-" else 5)
            for lab, key, _ in vs:
                draw(p79_frags[key][0], pale(lab), MODEL_LW, "-")
            draw(ref_frags[0], REF_COLOR, REF_LW, DASH)

        ecdf(left, lambda n: True)
        left.set_xscale("log")
        left.set_xlim(1e-3, 3e0)
        furniture(left, ylabel="Cumulative share", xlabel="Fragment RMSD (Å)", xloc=None)
        left.set_ylim(0, 1.0)
        legend(left, arm_handles(vs), loc="upper left",
               ncol=2 if len(vs) > 4 else 1, fontsize=11.5 if len(vs) <= 4 else 10.0)

        ecdf(right, lambda n: n >= RF_BIG_FRAG)
        right.set_xscale("log")
        right.set_xlim(1e-3, 3e0)
        furniture(right, ylabel=None,
                  xlabel=f"Fragment RMSD (Å), {RF_BIG_FRAG}+ atom fragments", xloc=None)
        fit(fig, pad=0.5)
        save(fig, out, f"rigid_ecdf_pair_{variant}")


# ── standalone fragment-size distribution, paired with the box plots ───────────
def _rf_shares(series, binning):
    """({arm: its share of fragments in each bin, in %}, the rows behind them).

    The denominator is the arm's OWN fragments from RF_FIRST_SIZE up, which is everything
    the bins cover, so the panel answers where a method puts its fragments and not how many
    it made. Every count is in, including the bins below the box plot's RF_MIN_DRAW floor:
    a share is a share at any n, and a bin missing from the boxes because it is thin is
    exactly the bin a reader comes here to size."""
    labels = _rf_binning(binning)[1]
    shares, rows = {}, []
    for lab, _, frags in series:
        counts = np.array([len(v) for v in _rf_bin_values(frags, binning).values()])
        total = int(counts.sum())
        if total == 0:
            raise ValueError(f"{lab}: no fragments with at least {RF_FIRST_SIZE} "
                             "heavy atoms")
        shares[lab] = 100.0 * counts / total
        for name, count, pct in zip(labels, counts, shares[lab]):
            rows.append([lab, name, int(count), total, float(pct)])
    return shares, rows


def _rf_dist_rows(out, arms, variant, p79_frags, ref_frags, binning):
    """The size distribution over the expanded figure's bins, in the same rows at the same
    width, so it reads directly under the boxes.

    Each row carries its own share ceiling, for the reason the box rows do: the 3-atom bin
    holds a third of every method's fragments and the 30+ bins hold tenths of a percent, so
    on one linear axis the entire tail is a flat line along the bottom and the figure says
    nothing about the sizes it was drawn to show. A log axis would show them, but these
    bars are shares -- on a log axis their heights stop being proportional to the numbers
    they stand for, and a reader comparing two bars by eye would be comparing nothing. So
    the axis stays linear and the rows are cut to fit what is in them; the ceiling is
    printed on every row and none of them starts anywhere but zero."""
    # The expanded box plot's layout exactly -- same rows, bins per row, width and type --
    # with every arm's share side by side in each bin and no legend. (A 2 x 4 one-panel-
    # per-method grid, lines then bars on a log axis, was tried on 2026-09-23 and reverted.)
    labels = _rf_binning(binning)[1]
    per_row = _rf_expanded_per_row(labels)
    rows_of_labels = _rf_rows_of(labels, per_row)
    series = [(REF_LABEL, REF_COLOR, ref_frags[0])] + [
        (lab, pale(lab), p79_frags[key][0]) for lab, key, _ in arms]
    shares, rows = _rf_shares(series, binning)
    at = {lab: i for i, lab in enumerate(labels)}
    slot = 0.86 / len(series)
    nrows = len(rows_of_labels)
    width = _rf_stack_width(len(series), per_row)
    t = _rf_type(width)
    fig, axes = plt.subplots(nrows, 1, dpi=220,
                             figsize=(width,
                                      nrows * RF_BOX_ROW_H * RF_EXPANDED_ROW_H_SCALE + 0.8))
    fig.patch.set_facecolor("white")
    for ax, labs in zip(axes, rows_of_labels):
        ax.set_facecolor("white")
        idx = np.arange(len(labs))
        for i, (lab, col, _) in enumerate(series):
            ax.bar(idx - 0.43 + slot * (i + 0.5), [shares[lab][at[l]] for l in labs],
                   width=slot * 0.82, color=matplotlib.colors.to_rgba(col, DATA_ALPHA),
                   edgecolor=INK, linewidth=0,
                   hatch="///" if lab == REF_LABEL else None, zorder=3)
        ymax = _rf_ceiling(max(shares[lab][at[l]] for lab, _, _ in series for l in labs))
        _rf_row_axes(ax, labs, ymax, t, per_row=per_row,
                     xlabel=RF_BOX_XLABEL if ax is axes[-1] else None)
    ylabel = fig.supylabel("Share of fragments (%)", fontsize=t.title, color=AXIS)
    fit(fig, pad=0.5, h_pad=H_PAD)
    _rf_stack_ylabel(fig, axes, ylabel, t.pad)
    _rf_keep_type(fig, t)
    save(fig, out, f"rigid_size_distribution_expanded_{variant}")
    return rows


@figure("fig-consistency-rigid-size-distribution",
        needs=("rigid_fragment_results.json", "rigid_fragment_reference.json"))
def draw_consistency_rigid_size_distribution(out):
    """Fragment-size shares, without a legend: the main box plot's 3–13 / 14+ bins in one
    row, and the expanded figure's per-size bins in its own rows beside it."""
    arms, per, _, p79_frags, ref_frags = _rf_data()
    use_style()
    labels = RF_BINNINGS["fine"][1]
    series = [(REF_LABEL, REF_COLOR, ref_frags[0])] + [
        (lab, pale(lab), p79_frags[key][0]) for lab, key, _ in arms]
    shares, rows = _rf_shares(series, "fine")

    for variant, vs in _rf_variants(arms):
        drawn = [(REF_LABEL, REF_COLOR)] + [(lab, pale(lab)) for lab, _, _ in vs]
        t = _rf_type(RF_BOX_MAX_FIG_W)
        fig, ax = plt.subplots(figsize=(RF_BOX_MAX_FIG_W, PANEL_H + 0.8), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        idx = np.arange(len(labels))
        slot = 0.86 / len(drawn)
        for i, (lab, col) in enumerate(drawn):
            ax.bar(idx - 0.43 + slot * (i + 0.5), shares[lab], width=slot * 0.82,
                   color=matplotlib.colors.to_rgba(col, DATA_ALPHA), edgecolor=INK,
                   linewidth=0, hatch="///" if lab == REF_LABEL else None, zorder=3)
        furniture(ax, xloc=None, ylabel="Share of fragments (%)",
                  xlabel="Rigid fragment size (Number of heavy atoms)")
        ax.grid(False, axis="x")
        ax.set_xticks(idx, labels)
        ax.set_xlim(-0.58, len(labels) - 0.42)
        ax.set_ylim(bottom=0)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(t.axis)
        ax.tick_params(width=t.axis, labelsize=t.tick, length=5 * t.scale)
        ax.xaxis.label.set_size(t.title)
        ax.yaxis.label.set_size(t.title)
        fit(fig, pad=0.5)
        _rf_keep_type(fig, t)
        save(fig, out, f"rigid_size_distribution_{variant}")
    write_csv(out, "rigid_size_distribution",
              ["arm", "fragment_size_bin", "n", "n_fragments_ge3", "share_pct"], rows)

    # The same distribution over the expanded figure's bins, one file per variant because
    # the two cut the tail in a different place.
    expanded = []
    for variant, vs in _rf_variants(arms):
        binning = _rf_expanded_binning(vs, per, RF_EXPANDED_TAIL.get(variant))
        expanded += [[f"expanded_{variant}"] + r for r in
                     _rf_dist_rows(out, vs, variant, p79_frags, ref_frags, binning)]
    write_csv(out, "rigid_size_distribution_expanded",
              ["binning", "arm", "fragment_size_bin", "n", "n_fragments_ge3",
               "share_pct"], expanded)

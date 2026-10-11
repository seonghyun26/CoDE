"""fig-vina: the 5 figures drawn into figures/fig-vina/.

    python figures/draw.py baseline vina

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-vina-per-pocket-dock                     -> fig-vina/per-pocket/
    fig-vina-per-pocket-score                    -> fig-vina/per-pocket/
    fig-vina-per-pocket-min                      -> fig-vina/per-pocket/
    fig-baseline-dock-per-pocket                 -> fig-vina/baseline/
    fig-vina-per-atom                            -> fig-vina/per-atomnum/
"""
from __future__ import annotations

import collections
import csv
import json
import os
import statistics as st
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from draw import (
    AXIS, AXIS_LABEL_FS, AXIS_LW, BASEDRUG, BUNDLE, DASH, DATA_ALPHA, DIST_FILL,
    DIST_LW, DOT, E, FIG_W, GRID, GRID_LW, HEIGHT_RATIOS, H_PAD, INK, LEGEND_EDGE,
    MIN_N, MODEL_LW, REF_COLOR, REF_LABEL, REF_LW, REF_WIN, REPO, STACK_H,
    VPA_KEY_ROW_GAP, XTICK_STEP, X_LABEL, display, figure, filename, fit, furniture,
    highlight, legend, one_frame, ordered_methods, pale, rolled, row_major, save,
    share, use_style, write_csv,
)

def plot_width(fig, ax):
    """Width of one axes' plot area, in inches -- what has to match across figures."""
    return ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
VINA_PER_MOLECULE = "eval/vina_docking/per_molecule.csv"

# (palette label, bundle folder), in the section's arm order so the drawn order, the zorder
# and the legend all agree with every other `all` figure here. VoxBind is the published
# sigma=0.9 reproduction and CoDE is ours; the Fusion arms and the base-ep350 variants are
# an ablation, a different comparison, and are deliberately not part of "all methods".
VINA_BUNDLE_ARMS = [
    ("AR",         "AR",              None),
    ("Pocket2Mol", "Pocket2Mol",      None),
    ("DiffSBDD",   "DiffSBDD",        None),
    ("DecompDiff", "DecompDiff",      None),
    ("FuncBind",   "FuncBind",        None),
    # Docked here, not on the evaluation server: results/task2-drugdesign/TargetDiff/eval/vina_docking is
    # run_docking_eval.py -> TargetDiff's own VinaDockingTask, whole receptor,
    # exhaustiveness 32, the same 79 pockets -- the baseline protocol, so it sits on this
    # axis exactly as the five above do. It was the one method the per-atom family drew and
    # this one did not (2026-09-16).
    ("TargetDiff", "TargetDiff",      None),
    # THIS ARM'S BUNDLE CSV IS THE WRONG PROTOCOL. `VoxBind-vanilla` is
    # exps/reproduction/samples/res_test_100, Vina-scored by metrics.py against the
    # pocket10 CROP (its results.json records dock_receptor_scope: null), and a crop and a
    # whole receptor do not share an axis -- the same crystal ligand docks to -7.18 one way
    # and -7.31 the other. The third field names a run tree to read INSTEAD when the
    # bundle's scope is not `full`: _vanilla_ep923 is the same published model under the
    # baseline protocol, and is what the per-atom family already draws, so the two Vina
    # families now show the same VoxBind. Delete it once res_test_100's re-dock lands in
    # the bundle and the CSV itself is full/exh32.
    ("VoxBind",    "VoxBind-vanilla", f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    ("CoDE",       "VoxBind-Ours",    None),
]

VINA_BUNDLE_ARMS = ordered_methods(VINA_BUNDLE_ARMS, key=lambda row: row[0])
VINA_WANT_SCOPE = "full"


def vina_rows(folder, *, density79=True):
    """Per-molecule Vina for one bundle method, or None if that method has no CSV.

    `density79` keeps only the 79 pockets with usable deposited density -- the set every
    figure in this section is drawn over. The flag is in the file, so the subset is the
    bundle's own definition and not a pocket list re-derived here."""
    path = BUNDLE / folder / VINA_PER_MOLECULE
    if not path.exists():
        return None
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if density79 and r.get("in_density79") != "1":
                continue
            try:
                row = {"target": r["target"], "n": int(r["n_atoms"])}
            except (KeyError, ValueError):
                continue
            for f in ("vina_score", "vina_min", "vina_dock"):
                v = r.get(f)
                row[f] = float(v) if v not in (None, "", "None") else None
            out.append(row)
    return out


def vina_engine(folder):
    """The engine string the bundle records for one method's Vina, for the run log.

    THE FIGURES MIX ENGINES AND MUST SAY SO. The five published baselines and CoDE are
    TargetDiff's VinaDockingTask over the whole receptor; the Fusion and base arms went
    through the earlier task-2 evaluator; VoxBind-vanilla's older numbers were scored against
    the pocket10 CROP, which is not the same axis at all -- the same crystal ligand docks
    to -7.31 one way and -7.18 the other. A figure that pools them is only honest if the
    census is printed beside it."""
    path = BUNDLE / folder / "eval/vina_docking/results.json"
    if not path.exists():
        return None
    try:
        pr = json.load(open(path, encoding="utf-8")).get("protocol", {})
    except (json.JSONDecodeError, OSError):
        return None
    return f"{pr.get('engine')} / scope={pr.get('dock_receptor_scope')}"


def vina_run_rows(root, field="vina_dock"):
    """Per-molecule rows from a run's own eval_docking_results_full79.json, shaped like
    vina_rows(). Used where the bundle CSV is not the protocol this figure draws."""
    path = os.path.join(root, "eval_docking_results_full79.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            per_target = json.load(handle)["per_target"]
    except (json.JSONDecodeError, OSError, KeyError):
        return None
    out = []
    for t in per_target:
        for m in t.get("per_mol") or []:
            if not isinstance(m, dict) or not m.get("n_atoms"):
                continue
            row = {"target": t["target"], "n": int(m["n_atoms"])}
            for f in ("vina_score", "vina_min", "vina_dock"):
                v = m.get(f)
                row[f] = float(v) if v is not None else None
            out.append(row)
    return out or None


def vina_bundle(*, density79=True):
    """[(label, folder, rows)] for every arm this figure can draw, in section order, with
    the engine census printed. An arm that has no data is named, so "not drawn" never looks
    like "not measured"; an arm whose bundle CSV is the wrong protocol is read from the run
    tree its entry names, and the substitution is printed."""
    out, engines = [], collections.Counter()
    for label, folder, run_root in VINA_BUNDLE_ARMS:
        rows = vina_rows(folder, density79=density79)
        eng = vina_engine(folder) or "unrecorded"
        if run_root and f"scope={VINA_WANT_SCOPE}" not in eng:
            run_rows = vina_run_rows(run_root)
            if run_rows:
                print(f"    {label}: bundle CSV is {eng.split('scope=')[-1]} scope; "
                      f"reading {os.path.relpath(run_root, REPO)} instead "
                      f"(baseline protocol)")
                rows = run_rows
                eng = f"run_docking_eval.py baseline protocol / scope={VINA_WANT_SCOPE}"
            else:
                print(f"    {label}: bundle CSV is the wrong protocol and "
                      f"{os.path.relpath(run_root, REPO)} has no full79 results")
        if rows is None:
            print(f"    {label} skipped: no {folder}/{VINA_PER_MOLECULE}")
            continue
        if not rows:
            print(f"    {label} skipped: {folder}/{VINA_PER_MOLECULE} has no scored rows")
            continue
        engines[eng] += 1
        out.append((label, folder, rows))
    for eng, n in engines.most_common():
        print(f"    engine: {eng}  ({n} arm{'s' if n > 1 else ''})")
    # SCOPE is what breaks the axis, not the implementation. A crop and a whole receptor
    # put the same crystal ligand 0.13 kcal/mol apart, so mixing them is a real error;
    # TargetDiff's VinaDockingTask and run_docking_eval.py at the same scope and the same
    # exhaustiveness are the same protocol by construction, which is the whole point of
    # 73_dock_baseline_protocol_79.sh. So the warning fires on scope, and the engine census
    # above is printed either way for the reader to judge.
    scopes = {e.split("scope=")[-1] for e in engines}
    if len(scopes) > 1:
        print(f"    NOTE: more than one receptor scope in this figure ({', '.join(sorted(scopes))})"
              " — the arms are NOT on one axis; see each method's eval/vina_docking/results.json")
    return out


# ════════════════════════════════════════════════════════════════════════════════
# fig-vina-per-pocket-{dock,score,min} — every method's per-pocket Vina, ranked
# ════════════════════════════════════════════════════════════════════════════════
# x is the pockets sorted by the FOCUS arm's own value for that metric and statistic, so
# its points climb monotonically and everything else is read against them. Every series is
# a point cloud, not a curve: consecutive pockets are a RANKING, not a sequence, and a
# connecting line would imply a continuity that is not there. x is ticked at ranks
# 1/20/40/60/last while its rules fall every 10, and neither is a pocket id -- rank 1 is
# whichever pocket that metric ranks first -- so the exported CSV maps every rank to its
# target.
#
# THE ARM LIST IS DATA. It was three hardcoded series ("3line") until 2026-09-13; it is now
# VPP_ARMS, and an arm whose per-pocket file is absent is SKIPPED WITH ITS PATH NAMED
# rather than silently missing. So the day a method's per-pocket results land, it appears
# here with no code change -- which is the whole reason the figure stopped being called
# "3line". What each one needs is `<root>/<VPP_BASENAME>` with a `per_target` list whose
# entries carry `target` and a `per_mol` list of {vina_score, vina_min, vina_dock}.
#
# PROTOCOL. The published-baseline protocol only -- whole `*_rec.pdb` receptor,
# exhaustiveness 32, the 79 density pockets -- so these clouds sit on the same axis as the
# AR/Pocket2Mol/DiffSBDD/DecompDiff/FuncBind table. The older crop protocol
# (`*_pocket10.pdb`, exhaustiveness 16, 78 pockets) is deliberately NOT plotted: the same
# crystal reference docks to -7.31 one way and -7.18 the other, so the two cannot share an
# axis. `eval_docking_results_full.json` is a third thing again (full receptor,
# exhaustiveness 16); only the `_full79` file is the baseline protocol.
VPP_BASENAME = "eval_docking_results_full79.json"
VPP_PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

# label -> run root. Order is the drawing order, so the focus arm lands on top; an entry
# with no VPP_BASENAME under it is reported and dropped. The five published baselines and
# the two arms scored on the evaluation server are listed so the skip message names the exact file that has to
# appear -- "not plotted" and "not measured" must never look the same.
VPP_ARMS = [
    ("AR",         f"{E}/baselines_pose/ar"),
    ("Pocket2Mol", f"{E}/baselines_pose/pocket2mol"),
    ("DiffSBDD",   f"{E}/baselines_pose/diffsbdd"),
    ("DecompDiff", f"{E}/baselines_pose/decompdiff"),
    ("FuncBind",   f"{E}/baselines_pose/funcbind"),
    ("TargetDiff", str(BASEDRUG / "eval" / "targetdiff")),
    ("VoxBind",    f"{E}/reproduction/samples/res_test_100"),
    ("VoxBind",    f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    ("CoDE",       f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
]

VPP_ARMS = ordered_methods(VPP_ARMS, key=lambda row: row[0])
# The arm the ranking is built from, and the one drawn last and largest.
VPP_FOCUS = "CoDE"
# `core` is the comparison this section makes -- our method against the model it modifies,
# with the crystal ligand as the benchmark -- and `all` adds the published baselines. The
# same split every other figure here carries, and for the same reason: two figures that
# differ in which arms they draw must not be able to share a name.
VPP_CORE = ("VoxBind", "CoDE")

# metric key -> (per-molecule field, per-pocket reference field, y-axis label)
VPP_METRICS = {
    "dock":  ("vina_dock",  "ref_vina_dock",  "Vina Dock"),
    "score": ("vina_score", "ref_vina_score", "Vina Score"),
    "min":   ("vina_min",   "ref_vina_min",   "Vina Min"),
}
VPP_STATS = ("mean", "median")

# Ours is drawn in its SOFT tint (the shared table above), not the flat #4363D8: against the
# sand at 79 small markers the saturated blue reads as the louder series, and the ordering --
# not saturation -- is what is supposed to be doing the arguing.
# Shape is a second identity channel, so the series survive greyscale and colour blindness.
# matplotlib's `markersize` is a DIAMETER, and at one diameter a triangle carries only 41%
# of a circle's ink -- two series would read as two different weights -- so every non-round
# marker is area-matched by VPP_SHAPE_SCALE and the focus arm keeps the 3.89 that sets it
# slightly ahead. Anything not named here takes the next marker in VPP_MARKER_CYCLE, which
# is what lets a new method arrive without a palette decision.
VPP_BASE_SIZE = 3.31
VPP_FOCUS_SIZE = 3.89
# AND BIGGER AGAIN WHERE THE PANEL IS CROWDED (asked for 2026-09-16). VPP_FOCUS_SIZE's
# margin over the base is enough to pick ours out of one other cloud; against eight, with
# the rest already dropped to VPP_FADE, it is not -- the same reason the fade exists, and
# the same panels it applies to. `core` keeps the plain 3.89.
VPP_FOCUS_BOOST = 1.4
VPP_MARKERS = {"VoxBind": "^", "CoDE": "o"}
VPP_MARKER_CYCLE = ("s", "D", "v", "P", "X", "<", ">", "*")
VPP_SHAPE_SCALE = {"o": 1.0, "^": 1.555, "v": 1.555, "s": 1.13, "D": 1.35,
                   "P": 1.2, "X": 1.2, "<": 1.555, ">": 1.555, "*": 1.7}
# THE CRYSTAL LIGAND IS NOT DRAWN IN THIS FAMILY AT ALL (asked for 2026-09-22). It was
# already off the crowded `all` panels; `core` now matches, so every per-pocket panel is
# model clouds only and the two variants cannot differ in what a mark on them means. The
# value is still read from each run and still written to the CSV beside every rank -- not
# plotted and not measured stay different things -- which is why VPP_METRICS keeps its
# ref_field and _vpp_reference_root() still runs. VPP_REF_LW went with the line.
VPP_X_LABEL = "Target pocket number"
VPP_XTICKS = (1, 20, 40, 60)          # plus the last rank, appended at draw time
# y EVERY 3 kcal/mol, ON THE GRID THROUGH -4 (asked for 2026-09-22). The automatic locator
# put a label on every integer -- ten or eleven of them down a 2.8 in panel, which is a
# ruler, not a scale, and at this type size the numbers were the busiest thing on the
# figure. Three is the step at which Dock reads -13 / -10 / -7 / -4; the anchor is what
# makes those the four rather than -12 / -9 / -6 / -3, and it is a constant so the three
# metrics label the same values wherever their spans overlap.
VPP_Y_STEP, VPP_Y_ANCHOR = 3, -4
# WIDE/TALL scale the 7.25 x 2.77 base figure. The paired figure is SIZED FROM the
# standalone one so one of its panels is exactly the same shape -- moving between the two
# is never silently a change of aspect ratio.
# WIDENED BY VPP_WIDEN (asked for 2026-09-22). 79 ranks across 5.2 in of plotting area is
# about fifteen markers to the inch per cloud, and the first thing that costs is the run of
# the focus arm's climb -- which is what the ordering exists to show. The height is
# untouched, so this IS a change of aspect ratio, taken by the whole family at once rather
# than by one metric: the pair figures derive their width from the same constant and stay
# panel-matched to the standalone ones.
VPP_WIDEN = 1.1
VPP_FIG_W, VPP_FIG_H = 7.25 * 0.88 * VPP_WIDEN, 2.77 * 1.155
# `all` PUTS ITS KEY UNDER THE PANEL (2026-09-16). Eight clouds over 79 ranks leave no
# empty corner: in the lower right the key sat on top of the best-ranked pockets, which is
# the half of the figure the ordering is built to show. So the `all` figures grow by a
# strip of their own and put the key in it -- ONE ROW, in drawing order (it led with the
# reference until the reference stopped being drawn, 2026-09-22). `core` is two clouds and keeps its key in the corner, where it costs no
# height and covers nothing.
#
# THE TYPE IS SIZED TO THE FIGURE RATHER THAN CHOSEN. The eight methods on one line want
# about 9.8 in at VPP_KEY_FS; the pair figure has it, the standalone (7.0 in) does not, so
# _vpp_key_below() steps the size down until the row fits with VPP_KEY_MARGIN to spare --
# it measures and prints what it settled on, which is why no number here has to be right
# for the figure to be. That is the trade a one-line key makes at this width: the
# standalone's key lands near VPP_KEY_MIN_FS, which is half the tick labels' 14 pt.
# The stacked pair is drawn WIDER than the figures it stacks (asked for 2026-09-16). Its
# two panels are the only ones here read as a time series rather than a ranking of 79
# named pockets -- one above the other, the same x under both -- and at the standalone
# width the run of CoDE's climb is compressed into the same 5.2 in that has to hold 632
# markers. So this pair alone breaks the panel-width match the rest of the family keeps,
# and its run-log line prints the factor rather than a delta so the break is on the record.
VPP_STACK_WIDE = 1.6
VPP_KEY_H = 0.45             # first guess at the strip; _vpp_fit_strip() trues it up
VPP_KEY_FS = 11.5            # the key's type where the row fits at it
VPP_KEY_MIN_FS = 6.0         # and the size below which it is too small to be a key at all
VPP_KEY_MARGIN = 0.10        # inches the row keeps clear of the figure's edges, each side
# Between the x label and the key. It was 1 px at the first try (2026-09-16): the strip is
# allocated as a gridspec RATIO, and tight_layout hands out what is left after the panel's
# decorations, so a strip asked for in inches does not arrive in inches. _vpp_fit_strip()
# measures what arrived and trues the figure up instead of guessing again.
VPP_KEY_PAD = 0.14
# EVERY ARM BUT OURS IS DRAWN THROUGH. At eight clouds the panel is 632 markers and the
# ones underneath were simply lost; at this alpha the pile-ups read as density instead, and
# CoDE stays opaque so the series the ranking is built from is still the one on top. The
# key's swatches stay at full strength -- in the panel a marker has to sit in a crowd, in
# the key it has to name a colour. `core` is two clouds that barely overlap and keeps its
# flat ink.
VPP_FADE = DATA_ALPHA


def _vpp_reference_root():
    """A run that records the crystal ligand's own docked value per pocket.

    The bundle's per-molecule CSVs are GENERATED molecules only -- there is no reference
    row in them -- so the grey benchmark still comes from a run's own
    eval_docking_results_full79.json, which carries ref_vina_* beside each target."""
    for _, root in (("CoDE", f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
                    ("VoxBind", f"{E}/_vanilla_ep923/samples/full_eval_ep923")):
        if os.path.exists(os.path.join(root, VPP_BASENAME)):
            return root
    return None


def _vpp_per_target(rows, metric_field):
    """{target: {"mean", "median", "n"}} from per-molecule rows."""
    by = collections.defaultdict(list)
    for r in rows:
        if r.get(metric_field) is not None:
            by[r["target"]].append(r[metric_field])
    return {t: {"mean": st.mean(v), "median": st.median(v), "n": len(v)}
            for t, v in by.items() if v}


def _vpp_slug(label):
    """A label as a CSV column prefix: lower case, no spaces."""
    return label.lower().replace(" ", "_").replace("+", "plus")


def _vpp_marker(label, index):
    """A method's marker and the size it is drawn at.

    THE MARKER IS THE METHOD'S, not its position in whatever set of arms one figure happens
    to draw. fig-baseline-* puts a single baseline against ours on each panel, so off a
    per-figure index every baseline would come out as the cycle's FIRST marker while the
    `all` panel gives each of them a different one -- the same method, two shapes, which is
    exactly what the shape channel exists to prevent. The index is therefore the method's
    place in VINA_BUNDLE_ARMS wherever it has one -- the table the colours and the drawing
    order already come from -- and the caller's only where it does not. On the `all` panel
    the two agree arm for arm, which is why this changes no existing figure."""
    order = [lab for lab, _, _ in VINA_BUNDLE_ARMS]
    index = order.index(label) if label in order else index
    m = VPP_MARKERS.get(label) or VPP_MARKER_CYCLE[index % len(VPP_MARKER_CYCLE)]
    base = VPP_FOCUS_SIZE if label == VPP_FOCUS else VPP_BASE_SIZE
    return m, base * VPP_SHAPE_SCALE.get(m, 1.0)


def _vpp_load(root):
    """{target: {metric: {"mean", "median", "ref", "n"}}} from one run's docking results."""
    with open(os.path.join(root, VPP_BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    out = {}
    for t in per_target:
        row = {}
        for key, (field, ref_field, _) in VPP_METRICS.items():
            vals = [m[field] for m in (t.get("per_mol") or [])
                    if isinstance(m, dict) and m.get(field) is not None]
            if not vals:
                continue
            row[key] = {"mean": st.mean(vals), "median": st.median(vals),
                        "ref": t.get(ref_field), "n": len(vals)}
        if row:
            out[t["target"]] = row
    return out


def _vpp_reference(data, targets, metric):
    """Per-pocket crystal-ligand value, averaged over the arms that report it. Only Dock is
    meaningfully noisy: score_only and minimize are deterministic and the runs agree
    exactly, while Dock re-searches unseeded and can disagree by ~0.7 kcal/mol on a pocket."""
    ref, worst = {}, (0.0, None)
    for t in targets:
        vals = [d[t][metric]["ref"] for d in data.values()
                if t in d and d[t].get(metric, {}).get("ref") is not None]
        ref[t] = st.mean(vals) if vals else None
        if len(vals) > 1 and max(vals) - min(vals) > worst[0]:
            worst = (max(vals) - min(vals), t)
    return ref, worst


def _vpp_limits(*serieses):
    """Clip to the bulk instead of squashing 79 pockets for a couple of runaway values; the
    caller names the excluded points, the image does not."""
    flat = sorted(v for ys in serieses for v in ys if v is not None)
    pct = lambda q: flat[int(round(q * (len(flat) - 1)))]
    return pct(0.01) - 0.45, pct(0.99) + 0.55


def _vpp_key(target, handles, *, loc, ncol, fontsize, **kw):
    """This family's key, wherever it goes: in the panel's corner, or stacked in the strip
    below it. `handles` None lets the axes collect its own, which is what the in-panel key
    has always done; the strip passes them because it splits them over two legends.

    `target` is an Axes or the Figure. The spacing is tighter than the shared legend()'s --
    these keys carry markers, not line samples, so a long handle is mostly empty space."""
    leg = target.legend(**({} if handles is None else {"handles": handles}),
                        loc=loc, ncol=ncol, frameon=True, fontsize=fontsize,
                        handlelength=1.4, handletextpad=0.4, labelspacing=0.3,
                        columnspacing=1.0, borderpad=0.4, borderaxespad=0.39,
                        facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0, **kw)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    leg.set_zorder(6)
    for text in leg.get_texts():      # identity rides the swatch, not the ink
        text.set_color(INK)
    # Swatches at twice the plotted size: in the cloud a marker only has to be findable, in
    # the key it has to be identifiable, and at 3-5 pt a triangle and a circle are the same
    # dot. Legend handles are copies, so this does not touch data.
    #
    # TWICE, OR AS MUCH AS THE HANDLE BOX HOLDS. The box is handlelength x the type, so at
    # the strip key's small size a doubled triangle would be wider than its own cell and
    # would print over the label beside it. One factor for every swatch, never a per-marker
    # clamp: the shapes are area-matched to each other (VPP_SHAPE_SCALE) and a clamp would
    # quietly undo that for the big ones only.
    drawn = [h for h in (getattr(leg, "legend_handles", None) or leg.legendHandles)
             if h.get_markersize() and h.get_marker() not in (None, "None", "", " ")]
    grow = min(2.0, 1.4 * fontsize / max((h.get_markersize() for h in drawn), default=1.0))
    for handle in getattr(leg, "legend_handles", None) or leg.legendHandles:
        if handle.get_markersize():
            handle.set_markersize(handle.get_markersize() * grow)
    return leg


def _vpp_draw(ax, order, clouds, ylim, *, legend_on, ylabel, crowded=False, y_step=None,
              paint=None, marker_scale=1.0, y_anchor=None):
    """One panel: one point cloud per arm over a dotted grid. The focus arm is drawn last,
    larger and opaque, because it is the series the ordering is built from and the one
    being read.

    NO REFERENCE ON ANY PANEL (the crowded `all` 2026-09-16, `core` too 2026-09-22). One
    crystal-ligand value per pocket, under an ordering set by another series, is a jagged
    line the full height of the panel: against eight clouds it was a ninth series crossing
    all of them and the thing the eye caught first, and against two it still reads as a
    series rather than as the benchmark it is meant to be. The value itself stays in the
    CSV beside every rank. So `crowded` now says only how many clouds share the panel --
    the fade and the focus boost -- and decides nothing about a reference."""
    x = list(range(1, len(order) + 1))
    lo, hi = ylim
    for i, (label, ys) in enumerate(clouds):
        marker, size = _vpp_marker(label, i)
        # `paint` as in size_distribution(): the approved sheet's other column. The
        # per-pocket figures take Color; fig-baseline-* takes Highlight, which is the same
        # hue a step down in lightness and reads better at 79 small markers a panel.
        colour = (paint or pale)(label)
        focus = label == VPP_FOCUS
        # `marker_scale` is the family's, not the series' -- it multiplies every arm by the
        # same number, so the shapes stay area-matched to each other (VPP_SHAPE_SCALE) and
        # only the ink per point changes. What it may be is set by the x pitch; see
        # VBL_MARKER_SCALE, which is the one caller that moves it.
        ax.plot(x, ys, ls="none", marker=marker, markersize=size * marker_scale * (
                    VPP_FOCUS_BOOST if focus and crowded else 1.0),
                color=colour, markerfacecolor=colour, markeredgewidth=0,
                zorder=4 if focus else 3, label=display(label),
                alpha=None if focus or not crowded else VPP_FADE)

    ax.set_xlabel(VPP_X_LABEL, fontsize=15.5, labelpad=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS, labelpad=10)
    ax.set_xlim(0.2, len(order) + 0.8)
    ax.set_ylim(lo, hi)
    # y is kcal/mol and the panel is short, so the default locator lands on halves: two
    # extra glyphs per label for precision the eye cannot use at this size. The step and
    # the grid it sits on are pinned instead (VPP_Y_STEP, VPP_Y_ANCHOR) -- a locator that
    # picks its own step gives the same panel a different number of labels each time the
    # data moves it by half a unit.
    #
    # `y_step` is the one thing about it a caller may set, and only to another step on the
    # SAME grid: fig-baseline-* holds all seven of its panels to one span, and a span that
    # narrow takes 2 without becoming the ruler this pinning exists to avoid.
    ax.yaxis.set_major_locator(MultipleLocator(
        y_step or VPP_Y_STEP,
        offset=VPP_Y_ANCHOR if y_anchor is None else y_anchor))
    ax.set_xticks([t for t in VPP_XTICKS if t < len(order)] + [len(order)])
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=AXIS_LW, pad=4,
                   colors=AXIS)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=DOT)
    # The x rules are plain vlines rather than a minor grid: a minor tick is silently
    # dropped wherever a major tick already sits, which left 20/40/60 without a rule.
    for xv in range(10, len(order) + 1, 10):
        ax.axvline(xv, color=GRID, lw=GRID_LW, ls=DOT, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(AXIS_LW)

    if legend_on:
        # One column while the key is short, two once it would run past the panel -- a
        # nine-row key in the corner of a 2.2 in panel covers the data it explains.
        _vpp_key(ax, None, loc="lower right", ncol=1 if len(clouds) <= 3 else 2,
                 fontsize=12.5)


def _vpp_key_below(fig, key_ax, ax, handles=None):
    """The `all` key, in the blank strip under the panel: one row of methods, in drawing
    order.

    `handles` is for a figure whose panels do not each hold every series: the combined
    baseline sheet draws one baseline per row, so its key is collected across the rows and
    handed in. Left None, the key is whatever `ax` drew, which is what a single panel wants.

    The size is the largest of VPP_KEY_FS and below at which the row still clears the
    figure's edges by VPP_KEY_MARGIN -- a key wider than the figure is cropped in the
    raster with no error at all, so this is measured, stepped down and then printed beside
    the figure it belongs to rather than left to a constant that goes stale the day a
    method joins.

    Called AFTER fit(): the row is centred on the FIGURE and sits at the bottom of the
    STRIP, so both have to be where they will finally be before it is placed."""
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()   # the drawing order, methods only
    # x is the figure's centre, y the strip's: centred on the panel alone the key sits a
    # y-label's width off the middle of the image, and the strip is only what decides how
    # far up it goes.
    place = matplotlib.transforms.blended_transform_factory(
        fig.transFigure, key_ax.transAxes)
    target = fig.get_size_inches()[0] - 2 * VPP_KEY_MARGIN
    fs, leg, width = VPP_KEY_FS, None, None
    while True:
        if leg is not None:
            leg.remove()
        leg = _vpp_key(fig, handles, loc="lower center", ncol=len(handles), fontsize=fs,
                       bbox_to_anchor=(0.5, 0.0), bbox_transform=place)
        leg.set_zorder(7)
        for handle in getattr(leg, "legend_handles", None) or leg.legendHandles:
            handle.set_alpha(1.0)     # the panel fades its markers; the key names them
        fig.canvas.draw()
        width = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        if width <= target or fs <= VPP_KEY_MIN_FS:
            break
        # A legend's width is its type size times a constant -- the handles, the padding and
        # the columns are all in font units -- so one proportional step lands within a
        # quarter point of the answer; the loop is what catches the rest.
        fs = max(VPP_KEY_MIN_FS, min(fs - 0.25, round(fs * target / width * 4) / 4))
    over = "" if width <= target else f"  OVER THE EDGE by {width - target:.2f} in"
    print(f"      key {width:.2f} in wide at {fs:g} pt, one row of {len(handles)}{over}")
    return leg


def _vpp_fit_strip(fig, panels, key_ax, leg, refit, want=None, pad=None):
    """Give the key's strip exactly the key plus VPP_KEY_PAD, and every panel row back the
    height it was laid out at.

    The strip is a gridspec row, so what it gets is a SHARE of what the panels' decorations
    leave -- ask for 0.92 in under a 3.2 in panel and 0.76 arrives, which is how the key
    came to sit 1 px under the x label. Measuring is the only way to know the share, so:
    measure the rows as laid out, re-cut the ratios in those inches with the strip at what
    the key actually needs, and change the FIGURE's height by the difference -- which leaves
    every margin where it was. One entry per ROW, so the stacked pair is the same two lines
    of arithmetic as a single panel.

    `want` is the panel height to hold every row to, and is how the stacked pair ends up
    with the standalone figure's panel and not merely its own two of equal size: a taller
    figure amortises the top and bottom margins over two rows and hands each of them ~0.04
    in more than a single-panel figure gets. Left None, each row keeps the height it was
    laid out at."""
    for _ in range(4):
        fig.canvas.draw()
        w, h = fig.get_size_inches()
        by_row = {a.get_subplotspec().rowspan.start: a for a in panels}
        heights = [by_row[r].get_position().height * h for r in sorted(by_row)]
        targets = [want if want is not None else v for v in heights]
        strip = key_ax.get_position().height * h
        # `pad` overrides the gap over the key. One panel's x label to one key is what
        # VPP_KEY_PAD was set against; a sheet of twelve panels puts a lot more ink between
        # the reader and that key, and it needs more air to read as being under all of them
        # rather than under the bottom row.
        need = leg.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi \
            + (VPP_KEY_PAD if pad is None else pad)
        grow = sum(t - v for t, v in zip(targets, heights)) + need - strip
        if abs(grow) < 0.005 and abs(need - strip) < 0.01:
            break
        panels[0].get_gridspec().set_height_ratios(targets + [need])
        fig.set_size_inches(w, h + grow)
        refit()


def _vpp_panel(below, *, rows=1, cols=1, width=None):
    """The figure, its panels laid out `rows` x `cols`, and a blank strip under them when
    the key goes below. Returns (fig, [panel axes], key axes or None), the panels in reading
    order and sharing one y scale.

    Every panel starts at VPP_FIG_H whatever the arrangement, so a `core` figure, an `all`
    figure and either half of a pair are the same panel -- and at the default width they are
    the same panel WIDTH too, which is what lets the stacked pair skip the width hunt the
    side-by-side one needs."""
    w = width if width is not None else VPP_FIG_W
    ratios = [VPP_FIG_H] * rows + ([VPP_KEY_H] if below else [])
    fig = plt.figure(figsize=(w, sum(ratios)), dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(len(ratios), cols, height_ratios=ratios)
    axes = []
    for i in range(rows * cols):
        ax = fig.add_subplot(gs[i // cols, i % cols], sharey=axes[0] if axes else None)
        ax.set_facecolor("white")
        axes.append(ax)
    key = None
    if below:
        key = fig.add_subplot(gs[-1, :])
        key.axis("off")
    return fig, axes, key


def _vpp_outside(order, curves, ylim):
    """Points the y-clip leaves off the panel, as (label, target, value)."""
    lo, hi = ylim
    return [(label, order[i], v) for label, ys in curves
            for i, v in enumerate(ys) if v is not None and not lo <= v <= hi]


def _vpp_pair_width(target_w, build):
    """Figure width the paired figure needs for ONE panel to be `target_w` wide. Lays the
    pair out once at a trial width, measures what the furniture actually costs at this type
    size, and adds that back around two panels of the requested width -- a constant here
    goes stale silently when the type size or the tick-label width moves."""
    trial = 12.0
    fig, axes = build(trial)
    furn = trial - 2 * plot_width(fig, axes[0])
    plt.close(fig)
    return 2 * target_w + furn


def _vina_per_pocket(metric, out):
    use_style()
    field, ref_field, label = VPP_METRICS[metric]
    print(f"  {VPP_PROTOCOL}")
    available = vina_bundle(density79=True)
    if not available:
        raise SystemExit("no method has a per-molecule Vina CSV in the results bundle")
    data = {lab: _vpp_per_target(rows, field) for lab, _, rows in available}
    if VPP_FOCUS not in data:
        raise SystemExit(f"the focus arm {VPP_FOCUS} has no {field}; the ranking this "
                         "figure is built on cannot be formed")

    ref, ref_root = {}, _vpp_reference_root()
    if ref_root:
        with open(os.path.join(ref_root, VPP_BASENAME), encoding="utf-8") as handle:
            per_target = json.load(handle)["per_target"]
        ref = {t["target"]: t.get(ref_field) for t in per_target}
        print(f"    reference: {os.path.relpath(ref_root, REPO)}/{VPP_BASENAME}"
              "  (CSV column and the line below only — no panel draws it)")
    else:
        print("    reference: none available — the CSV's reference column is empty")

    for variant, labels in (("core", [l for l, _, _ in available if l in VPP_CORE]),
                            ("all", [l for l, _, _ in available])):
        missing = [l for l in VPP_CORE if l not in labels]
        if missing:
            raise SystemExit(f"`core` is {' + '.join(VPP_CORE)}; {', '.join(missing)} "
                             "has no per-molecule Vina CSV, so the comparison this figure "
                             "exists to make cannot be drawn")
        _vpp_variant(metric, out, label, field, data, labels, ref, variant)


def _vpp_variant(metric, out, label, field, data, labels, ref, variant):
    """One arm set of one metric: mean, median and the pair."""
    targets = sorted(set.intersection(*(set(data[l]) for l in labels)))
    print(f"  {label} · {variant}: {' + '.join(labels)} · {len(targets)} pockets")

    # WHAT `all` AND `core` DIFFER BY, besides the arms: eight clouds have nowhere inside
    # the panel to put a key and no room for a ninth series either, two clouds have both.
    # See VPP_KEY_H, VPP_FADE, VPP_FOCUS_BOOST and the reference in _vpp_draw().
    below = variant == "all"

    curves = {}
    for stat in VPP_STATS:
        order = sorted(targets, key=lambda t: data[VPP_FOCUS][t][stat])
        clouds = [(lab, [data[lab][t][stat] for t in order]) for lab in labels]
        curves[stat] = (order, clouds)

    # The span is set by WHAT IS ON THE PANEL, and no panel draws the reference any more,
    # so neither variant is scaled to a line it does not show.
    spans = {s2: _vpp_limits(*[ys for _, ys in c[1]]) for s2, c in curves.items()}
    # The paired figure shares one y scale, so its panels are comparable vertically.
    shared = (min(v[0] for v in spans.values()), max(v[1] for v in spans.values()))
    plot_w = None                    # set by the first standalone panel, matched below

    for stat, (order, clouds) in curves.items():
        fig, (ax,), key = _vpp_panel(below)
        _vpp_draw(ax, order, clouds, spans[stat], legend_on=not below,
                  ylabel=f"{label} {stat}", crowded=below)
        refit = lambda: fit(fig, pad=0.5)
        refit()
        if key is not None:
            _vpp_fit_strip(fig, [ax], key, _vpp_key_below(fig, key, ax), refit)
        plot_w = plot_width(fig, ax)
        plot_h = ax.get_position().height * fig.get_size_inches()[1]
        stem = f"vina_{metric}_per_pocket_{stat}_{variant}"
        save(fig, out, stem)

        pooled = {}
        refs = [ref[t] for t in order if ref.get(t) is not None]
        if refs:
            # The run log keeps the benchmark the panel stopped drawing -- both variants,
            # because both CSVs still carry the column it is the mean of.
            pooled[REF_LABEL] = st.mean(refs)
        pooled.update({lab: st.mean(ys) for lab, ys in clouds})
        print(f"    {stat:6s} " + "  ".join(f"{k} {v:+.2f}" for k, v in pooled.items()))
        for lab, target, value in _vpp_outside(order, clouds, spans[stat]):
            print(f"      outside the plotted y-range: {lab} @ {target} = {value:+.2f}")

        header = ["rank", "target", "reference"]
        for lab in labels:
            header += [f"n_{_vpp_slug(lab)}", f"{_vpp_slug(lab)}_{stat}"]
        rows = []
        for i, t in enumerate(order, 1):
            r = ref.get(t)
            row = [i, t, f"{r:.3f}" if r is not None else ""]
            for lab in labels:
                row += [data[lab][t]["n"], f"{data[lab][t][stat]:.3f}"]
            rows.append(row)
        write_csv(out, stem, header, rows)

    # `all` STACKS ITS PAIR, side by side is `core`'s (asked for 2026-09-16). Two panels on
    # one row have to be squeezed to half a figure each or the figure has to grow to twice
    # the width -- which is what the side-by-side pair does, and at eight clouds over 79
    # ranks that width is then two crowded panels the reader cannot put next to the
    # standalone figures at all. Stacked, each panel IS the standalone panel, at the
    # standalone width, and the one-line key underneath spans both.
    stacked = below
    def build_pair(width):
        fig, axes, key = _vpp_panel(below, rows=2 if stacked else 1,
                                    cols=1 if stacked else 2, width=width)
        for i, (stat, (order, clouds)) in enumerate(curves.items()):
            _vpp_draw(axes[i], order, clouds, shared,
                      legend_on=not below and i == 0, ylabel=f"{label} {stat}", crowded=below)
        # sharey suppresses a follower panel's tick labels wherever it is not in the first
        # column; put the numbers back so each panel reads on its own -- its y name says
        # mean vs median, so it has to.
        axes[1].tick_params(labelleft=True, labelsize=14)
        refit = lambda: fit(fig, pad=0.5, **({} if stacked else {"w_pad": 1.54}))
        refit()
        if key is not None:
            # One key under BOTH panels -- the two are the same arms at two statistics, and
            # a second copy of a nine-entry key would be the widest thing in the figure
            # saying nothing new.
            _vpp_fit_strip(fig, axes, key, _vpp_key_below(fig, key, axes[0]), refit,
                           want=plot_h if stacked else None)
        return fig, axes

    # Stacked, one panel is the whole width and needs no hunting for it; side by side, the
    # width is whatever makes ONE panel as wide as the standalone figure's.
    fig, axes = build_pair(VPP_FIG_W * VPP_STACK_WIDE if stacked
                           else _vpp_pair_width(plot_w, build_pair))
    got = plot_width(fig, axes[0])
    save(fig, out, f"vina_{metric}_per_pocket_pair_{variant}")
    against = (f"x{got / plot_w:.2f}, widened on purpose" if stacked
               else f"{got - plot_w:+.4f}")
    print(f"    pair   shared y {shared[0]:.2f} .. {shared[1]:.2f}   "
          f"panel {got:.3f} in vs standalone {plot_w:.3f} in ({against})")


@figure("fig-vina-per-pocket-dock", needs=(VPP_BASENAME,),
        folder="fig-vina/per-pocket")
def draw_vina_per_pocket_dock(out):
    """Vina Dock per pocket, every arm that has per-pocket results — mean, median, pair."""
    _vina_per_pocket("dock", out)


@figure("fig-vina-per-pocket-score", needs=(VPP_BASENAME,),
        folder="fig-vina/per-pocket")
def draw_vina_per_pocket_score(out):
    """Vina Score per pocket, every arm that has per-pocket results — mean, median, pair."""
    _vina_per_pocket("score", out)


@figure("fig-vina-per-pocket-min", needs=(VPP_BASENAME,),
        folder="fig-vina/per-pocket")
def draw_vina_per_pocket_min(out):
    """Vina Min per pocket, every arm that has per-pocket results — mean, median, pair."""
    _vina_per_pocket("min", out)
# ════════════════════════════════════════════════════════════════════════════════
# fig-baseline-dock-per-pocket — CoDE against ONE baseline per figure
# ════════════════════════════════════════════════════════════════════════════════
# fig-vina's per-pocket family draws either two clouds (`core`: ours against the model it
# modifies) or all eight (`all`). This folder is the case between them, asked for
# 2026-09-22: the `core` PANEL, redrawn once per baseline, so CoDE is read against one
# method at a time instead of out of a crowd of eight where the only thing a reader can
# really follow is our own climb.
#
# It IS the `core` panel -- same ranking (pockets sorted by CoDE's own value for that
# statistic), same clouds-and-not-curves, same in-panel key, same colours, same markers,
# same CSV written beside every figure -- and differs in exactly two things, both asked for:
#
#   * y is ticked every VBL_Y_STEP kcal/mol rather than the family's VPP_Y_STEP of 3, and
#   * EVERY PANEL IN THE FOLDER SHARES ONE y RANGE. CoDE is one of the two clouds in all
#     seven figures; with a span fitted to each pair, ours would be a different shape in
#     every file, and the one thing these figures are flipped between to do -- compare the
#     gap against one baseline with the gap against the next -- would be read off a moving
#     scale. So the span is taken once over every arm and both statistics (_vbl_span) and
#     handed to all fourteen panels.
#
# The VoxBind pair is the existing `core` figure, so it KEEPS THAT NAME rather than arriving
# as a second copy of it under a new one: `dock-per-pocket-{mean,median}-core`, the same
# stem fig-vina/ uses, in this folder, on this folder's axis.
VBL_METRIC = "dock"
# y ON THREES, ANCHORED AT -3 (asked for 2026-09-22): -12 / -9 / -6 / -3 over this folder's
# span. It was 2 on the -4 grid, which is six labels down a panel this short. The family's
# own step is 3 as well (VPP_Y_STEP) but on the -4 anchor -- these two grids cannot both be
# right for one span, and the one asked for here is the one this folder uses.
VBL_Y_STEP, VBL_Y_ANCHOR = 3, -3
# A baseline's file name is its slug, except the one whose pair already has a name.
VBL_STEM_NAME = {"VoxBind": "core"}
VBL_STATS_FILE = "stats.txt"
# MARKERS IN THE HIGHLIGHT TONE (asked for 2026-09-22), the approved sheet's second column
# rather than its Color one -- same hue, a step darker. A panel here is 158 markers of two
# methods and nothing else, with no line or band to carry the identity, so the tone that
# survives being 3 pt across is the one to use. It is still the shared table and not a local
# hex: highlight() is in 00_core.py beside color() and pale(), and every figure in this
# folder takes it, so the folder is one palette even though it is not fig-vina/per-pocket's.
VBL_PAINT = highlight
# HOW MUCH BIGGER THE MARKERS MAY GET (asked for 2026-09-22) is not a taste question here.
# Two points of ONE series that are one rank apart sit exactly one x pitch apart, and the
# focus arm's ranking makes them near-equal in y as well, so the pitch is the distance that
# decides whether a cloud reads as points or as a line of touching blobs. It is 5.84 pt in
# the individual panels and 6.28 pt in the wider combined sheet; the narrower one governs,
# since the sizes are shared. The family's largest marker is the area-matched triangle at
# VPP_BASE_SIZE x 1.555 = 5.15 pt, which leaves x1.135 before it IS the pitch -- so this is
# that, with a hair in hand: at x1.12 the triangles come to 5.76 pt against a 5.84 pt pitch
# and touch rather than merge, and every other shape keeps its ratio to them. Measured off
# the real spacing rather than picked; the run log prints what it comes to.
VBL_MARKER_SCALE = 1.12
# ── who wins each pocket, as the panel's own background ──────────────────────────
# One full-height band per rank (asked for 2026-09-22), tinted OURS where CoDE's statistic
# is the lower one and the BASELINE's where it is not. The ranking already sorts the pockets
# by CoDE's value, so the bands turn the per-pocket verdict -- which the reader otherwise
# has to make 79 times by eye, comparing two marks that are often a millimetre apart -- into
# a block of colour, and the count in the corner is the same thing as a number.
#
# The wash is the COLOR column and the marks are the HIGHLIGHT one: same hue, two steps of
# lightness, so a band can never be mistaken for a mark of the method it belongs to. At this
# alpha it sits under the dotted grid and reads as paper, not as data.
VBL_BAND_ALPHA = 0.22
# The count, bottom right, in the panel's own corner: "CoDE better in 84% of pockets". It is
# the same number stats.txt tabulates -- strictly lower, ties to neither -- so the figure and
# the table cannot drift. Where a panel keeps its key in that corner (the individual
# figures), the line sits directly above it rather than moving the key.
VBL_NOTE_FS = 12.5
VBL_NOTE_PAD = 0.035
# THE COMBINED SHEET TAKES NEITHER THE BANDS NOR THE NAME (asked for 2026-09-22). Twelve
# banded panels on one page is a wall of tint that the six single figures deliver one at a
# time, and the count in each corner already carries the verdict the bands colour in. The
# individual figures keep them: there the band IS the reason to open that one file.
VBL_COMBINED_BANDS = False
# And its key gets air over it. VPP_KEY_PAD is the gap a single panel's x label wants; under
# six rows the same gap reads as the bottom row's key rather than the sheet's.
VBL_COMBINED_KEY_PAD = 0.40

# ── the combined sheet ───────────────────────────────────────────────────────────
# The same panels again, all on one page: one ROW per method, median in the left column and
# mean in the right (asked for 2026-09-22). The individual files are for reading one
# comparison; this is for reading the six against each other, which is a different question
# and needs them stacked on one axis rather than in one folder.
#
# ITS ROWS ARE THE PUBLISHED BASELINES, six of them, and not the `core` pair: VoxBind is the
# model ours MODIFIES rather than a method ours is benchmarked against, and mixing the two
# on one sheet would put a +0.3 row beside six +1.x ones as though they answered the same
# question. Its figure keeps its own name in this folder.
#
# THE ORDER IS THE DRUG-DESIGN TABLE'S -- METHOD_ORDER, the one sequence every bar, box and
# legend in the paper already uses -- so a reader moving between this sheet and the table
# finds the methods in the same places.
VBL_COMBINED_SKIP = (VPP_CORE[0],)          # VoxBind: the `core` pair, not a row here
VBL_COMBINED_COLS = ("median", "mean")      # left, right
# ONLY THE BOTTOM ROW CARRIES x. Twelve panels over one ranking is the same 79 pockets
# twelve times; repeating "Target pocket number" and its numbers under every one of them
# spends a fifth of the sheet's height restating what the bottom row already says. The tick
# MARKS and the vertical rules stay in every panel -- they are how a value is located --
# and only the numbers and the axis name come off.
VBL_COMBINED_XLABEL_ROW = -1
# ... AND THE y NAME IS WRITTEN ONCE, OVER THE COLUMN (asked for 2026-09-22). What a column
# is -- median or mean -- is one fact about six panels, and as a y-axis name it was set six
# times down each side, which is the widest repeated thing on the sheet and pushes both
# columns right by the height of the type. As a heading it is stated where the eye starts,
# once per column. The tick NUMBERS stay on every panel: they are how a value is read, and
# the heading names the quantity they are in. This is the one place in the family with type
# over a panel -- elsewhere the y name does the job, and it still does in the single figures.
VBL_COMBINED_HEAD_FS = AXIS_LABEL_FS
VBL_COMBINED_HEAD_PAD = 9
# ONE KEY, UNDER THE WHOLE SHEET (asked for 2026-09-22). In the panels it was six two-line
# keys, one per row, each naming that row's method and repeating CoDE -- seven entries said
# twelve times, and each one sitting over the corner of the data it explains. Under the
# sheet it is one row of seven, the same strip the crowded `all` figure uses and the same
# measured step-down of its type, so nothing in a panel is covered any more. The handles are
# collected ACROSS the rows, since no single panel here holds every series.
# The sheet is then 1.1x as wide and 0.95x as tall as the panels alone would make it (asked
# for the same day): the extra width goes to the ranking -- 79 pockets across a column -- and
# the height comes off the panels, which the key below no longer has to share.
VBL_COMBINED_W_SCALE = 1.1
VBL_COMBINED_H_SCALE = 0.95


def _vbl_stem(stat, baseline):
    """The file name for one pair at one statistic, as the user reads it: the metric, the
    statistic, the baseline it is against."""
    slug = VBL_STEM_NAME.get(baseline) or _vpp_slug(baseline)
    return f"{VBL_METRIC}_per_pocket_{stat}_{slug}"


def _vbl_span(data, labels):
    """The y range every panel in the folder is drawn on: _vpp_limits' percentile clip, but
    taken ONCE over every arm at both statistics instead of per panel. Each arm contributes
    its own pockets once, so the clip is not weighted by how many pairs a method appears in
    -- ours is in all seven of them."""
    return _vpp_limits(*[[data[lab][t][stat] for t in sorted(data[lab])]
                         for lab in labels for stat in VPP_STATS])


def _vbl_bands(ax, order, clouds, baseline):
    """The per-rank verdict, as the panel's background. Drawn under everything, and stretched
    to the axes' edges at the two ends so the shading has no white margin the data does not
    have."""
    (_, theirs), (_, ours) = clouds
    lo, hi = ax.get_xlim()
    for i, (a, b) in enumerate(zip(ours, theirs), start=1):
        ax.axvspan(max(lo, i - 0.5), min(hi, i + 0.5),
                   color=pale(VPP_FOCUS if a < b else baseline),
                   alpha=VBL_BAND_ALPHA, lw=0, zorder=-2)


def _vbl_note(fig, ax, pct, who=None):
    """`CoDE better in NN% of pockets`, in the bottom right corner -- above the key where the
    panel has one, since the key is already there and naming the arms is its job.

    `who` is None on the combined sheet (asked for 2026-09-22): ours is one of the two
    clouds in all twelve of its panels and is named once in the key under them, so putting
    its name in twelve corners as well says nothing the reader does not already have."""
    y = 0.025
    leg = ax.get_legend()
    if leg is not None:
        fig.canvas.draw()
        box = leg.get_window_extent(fig.canvas.get_renderer())
        y = ax.transAxes.inverted().transform((0, box.y1))[1] + VBL_NOTE_PAD
    return ax.text(0.985, y, f"{who + ' ' if who else ''}better in {pct:.0f}% of pockets",
                   transform=ax.transAxes, ha="right", va="bottom",
                   fontsize=VBL_NOTE_FS, color=INK, zorder=7)


def _vbl_pair(out, data, ref, baseline, ylabel, span):
    """One baseline: the mean panel, the median panel, a CSV beside each, and the two rows
    they contribute to stats.txt."""
    labels = [baseline, VPP_FOCUS]        # drawing order — the focus arm goes on top
    targets = sorted(set(data[baseline]) & set(data[VPP_FOCUS]))
    if not targets:
        print(f"  {baseline} skipped: no pocket has both arms")
        return []
    print(f"  {ylabel} · {baseline}: {' + '.join(labels)} · {len(targets)} pockets")

    rows_out = []
    for stat in VPP_STATS:
        order = sorted(targets, key=lambda t: data[VPP_FOCUS][t][stat])
        clouds = [(lab, [data[lab][t][stat] for t in order]) for lab in labels]

        fig, (ax,), _ = _vpp_panel(False)
        _vpp_draw(ax, order, clouds, span, legend_on=True, ylabel=f"{ylabel} {stat}",
                  y_step=VBL_Y_STEP, y_anchor=VBL_Y_ANCHOR, paint=VBL_PAINT,
                  marker_scale=VBL_MARKER_SCALE)
        _vbl_bands(ax, order, clouds, baseline)
        ours = [data[VPP_FOCUS][t][stat] for t in order]
        theirs = [data[baseline][t][stat] for t in order]
        lower = sum(a < b for a, b in zip(ours, theirs))
        fit(fig, pad=0.5)
        _vbl_note(fig, ax, 100.0 * lower / len(order), who=display(VPP_FOCUS))
        stem = _vbl_stem(stat, baseline)
        save(fig, out, stem)

        # WHO IS LOWER, POCKET BY POCKET. Vina is an energy, so lower is the stronger
        # predicted affinity; a tie is not a win and is counted on its own line rather than
        # folded into either side.
        ties = sum(a == b for a, b in zip(ours, theirs))
        delta = st.mean(a - b for a, b in zip(ours, theirs))
        pct = 100.0 * lower / len(order)
        print(f"    {stat:6s} CoDE lower in {lower}/{len(order)} pockets ({pct:.1f}%)  "
              f"mean CoDE-{baseline} {delta:+.2f}")
        for lab, target, value in _vpp_outside(order, clouds, span):
            print(f"      outside the plotted y-range: {lab} @ {target} = {value:+.2f}")
        rows_out.append({"baseline": baseline, "figure": filename(stem), "stat": stat,
                         "n": len(order), "lower": lower, "ties": ties, "pct": pct,
                         "delta": delta})

        header = ["rank", "target", "reference"]
        for lab in labels:
            header += [f"n_{_vpp_slug(lab)}", f"{_vpp_slug(lab)}_{stat}"]
        rows = []
        for i, t in enumerate(order, 1):
            r = ref.get(t)
            row = [i, t, f"{r:.3f}" if r is not None else ""]
            for lab in labels:
                row += [data[lab][t]["n"], f"{data[lab][t][stat]:.3f}"]
            rows.append(row)
        write_csv(out, stem, header, rows)
    return rows_out


def _vbl_stats(out, rows, ylabel, span, sheet=None):
    """Every pair's count in ONE file, because the question the folder is for -- on what
    share of the pockets is ours ahead of THIS one -- is only answered by the seven answers
    read together. A number in a figure's caption cannot be compared with the next figure's
    without both figures open; a table can."""
    path = Path(out) / VBL_STATS_FILE
    w = max(len(r["baseline"]) for r in rows)
    f = max(len(r["figure"]) for r in rows)
    text = [
        f"{ylabel} per pocket — CoDE against one baseline at a time",
        VPP_PROTOCOL,
        f"figures/draw.py fig-baseline-dock-per-pocket   ·   {time.strftime('%Y-%m-%d')}",
        "",
        "`CoDE lower` counts the pockets where CoDE's per-pocket statistic is MORE NEGATIVE",
        "than the baseline's — the stronger predicted affinity. `n` is per pair: the pockets",
        "are the intersection of the two arms, so a method that is missing a pocket shortens",
        "its own row and nobody else's. `mean Δ` is the mean over those pockets of",
        "(CoDE − baseline), so a negative Δ is ours ahead. A tie is counted as neither.",
        f"All panels share one y range, {span[0]:.2f} .. {span[1]:.2f}, ticked every"
        f" {VBL_Y_STEP} kcal/mol on the {VBL_Y_ANCHOR} grid. Each panel is banded per pocket"
        " — ours where CoDE is lower, the baseline's where it is not — and carries the same"
        " count as its row below, in its bottom right corner.",]
    if sheet:
        text += [f"`{sheet}` is the same panels on one sheet: one row per published"
                 " baseline in the drug-design table's order, median left and mean right.",]
    text += [
        "",
        f"{'baseline':<{w}}  {'figure':<{f}}  {'stat':<6} {'n':>3} {'CoDE lower':>10} "
        f"{'%':>6} {'ties':>5} {'mean Δ':>8}",
    ]
    for r in rows:
        text.append(f"{r['baseline']:<{w}}  {r['figure']:<{f}}  {r['stat']:<6} {r['n']:>3} "
                    f"{r['lower']:>10} {r['pct']:>6.1f} {r['ties']:>5} {r['delta']:>+8.2f}")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"  stats -> {os.path.relpath(path, REPO)}  ({len(rows)} rows)")


def _vbl_combined(out, data, baselines, ylabel, span):
    """Every published baseline on one sheet: one row per method, median left, mean right.

    Each cell is the standalone panel -- same width, same height, same span -- so a row IS
    the pair of figures it sits beside on disk, and the sheet adds the comparison BETWEEN
    rows without changing what any one of them shows. The width is measured off a standalone
    panel rather than assumed, exactly as the paired figures do it."""
    probe, (probe_ax,), _ = _vpp_panel(False)
    fit(probe, pad=0.5)
    target_w = plot_width(probe, probe_ax)
    plt.close(probe)

    def build(width):
        notes = []
        fig, axes, key_ax = _vpp_panel(True, rows=len(baselines), cols=2, width=width)
        # The rows are laid out at VBL_COMBINED_H_SCALE of the standalone panel's height;
        # the strip keeps its first guess and _vpp_fit_strip() trues it to the key.
        ratios = [VPP_FIG_H * VBL_COMBINED_H_SCALE] * len(baselines) + [VPP_KEY_H]
        axes[0].get_gridspec().set_height_ratios(ratios)
        fig.set_size_inches(width, sum(ratios))
        for r, baseline in enumerate(baselines):
            targets = sorted(set(data[baseline]) & set(data[VPP_FOCUS]))
            for c, stat in enumerate(VBL_COMBINED_COLS):
                ax = axes[r * len(VBL_COMBINED_COLS) + c]
                order = sorted(targets, key=lambda t: data[VPP_FOCUS][t][stat])
                clouds = [(lab, [data[lab][t][stat] for t in order])
                          for lab in (baseline, VPP_FOCUS)]
                # The key goes in the LEFT panel of each row: it names the row's method, and
                # the right panel is the same two arms at the other statistic. Twelve copies
                # of a two-line key would be the most repeated thing on the sheet.
                # No y name on the panel: it is the column's heading instead, below.
                _vpp_draw(ax, order, clouds, span, legend_on=False, ylabel=None,
                          y_step=VBL_Y_STEP, y_anchor=VBL_Y_ANCHOR, paint=VBL_PAINT,
                          marker_scale=VBL_MARKER_SCALE)
                if VBL_COMBINED_BANDS:
                    _vbl_bands(ax, order, clouds, baseline)
                notes.append((ax, 100.0 * sum(a < b for a, b in zip(clouds[1][1],
                                                                   clouds[0][1]))
                              / len(order)))
                if r == 0:
                    ax.set_title(f"{ylabel} {stat}", fontsize=VBL_COMBINED_HEAD_FS,
                                 color=INK, pad=VBL_COMBINED_HEAD_PAD)
                # sharey drops a follower column's numbers; the two columns are far enough
                # apart that each has to carry its own.
                ax.tick_params(labelleft=True, labelsize=14)
                if r != len(baselines) + VBL_COMBINED_XLABEL_ROW:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)
        # One handle per row -- that row's baseline -- and ours once at the end, which is
        # where the drawing order puts it in every other key in the family.
        handles = [axes[r * len(VBL_COMBINED_COLS)].get_legend_handles_labels()[0][0]
                   for r in range(len(baselines))]
        handles += [axes[0].get_legend_handles_labels()[0][1]]
        refit = lambda: fit(fig, pad=0.5, w_pad=1.54)
        refit()
        _vpp_fit_strip(fig, axes, key_ax,
                       _vpp_key_below(fig, key_ax, axes[0], handles=handles), refit,
                       pad=VBL_COMBINED_KEY_PAD)
        for ax, pct in notes:
            _vbl_note(fig, ax, pct)
        return fig, axes

    fig, axes = build(_vpp_pair_width(target_w, build) * VBL_COMBINED_W_SCALE)
    got = plot_width(fig, axes[0])
    stem = f"{VBL_METRIC}_per_pocket_combined"
    save(fig, out, stem)
    print(f"  {filename(stem)}: {len(baselines)} rows x {len(VBL_COMBINED_COLS)} "
          f"({' · '.join(VBL_COMBINED_COLS)}, named once over each column), "
          + " -> ".join(baselines)
          + f"\n    panel {got:.3f} in vs standalone {target_w:.3f} in "
            f"(x{got / target_w:.2f} wide, x{VBL_COMBINED_H_SCALE} tall, both on purpose); "
            "x label and tick numbers on the bottom row only, one key under the sheet")
    return filename(stem)


@figure("fig-baseline-dock-per-pocket", needs=(VINA_PER_MOLECULE,))
def draw_baseline_dock_per_pocket(out):
    """Vina Dock per pocket, CoDE against each baseline on its own panel — mean, median,
    and one stats.txt counting where ours is lower."""
    use_style()
    field, ref_field, ylabel = VPP_METRICS[VBL_METRIC]
    print(f"  {VPP_PROTOCOL}")
    available = vina_bundle(density79=True)
    if not available:
        raise SystemExit("no method has a per-molecule Vina CSV in the results bundle")
    data = {lab: _vpp_per_target(rows, field) for lab, _, rows in available}
    if VPP_FOCUS not in data:
        raise SystemExit(f"the focus arm {VPP_FOCUS} has no {field}; the comparison every "
                         "figure in this folder makes cannot be drawn")
    baselines = [lab for lab, _, _ in available if lab != VPP_FOCUS]
    if not baselines:
        raise SystemExit(f"{VPP_FOCUS} is the only arm with per-molecule Vina; there is "
                         "nothing to compare it against")

    # The crystal ligand is in the CSV beside every rank and on no panel, exactly as in
    # fig-vina — see _vpp_draw(). Read from a run that records it per pocket, since the
    # bundle's per-molecule CSVs are generated molecules only.
    ref, ref_root = {}, _vpp_reference_root()
    if ref_root:
        with open(os.path.join(ref_root, VPP_BASENAME), encoding="utf-8") as handle:
            ref = {t["target"]: t.get(ref_field)
                   for t in json.load(handle)["per_target"]}
        print(f"    reference: {os.path.relpath(ref_root, REPO)}/{VPP_BASENAME}"
              "  (CSV column only — no panel draws it)")
    else:
        print("    reference: none available — the CSV's reference column is empty")

    span = _vbl_span(data, [VPP_FOCUS] + baselines)
    print(f"    shared y {span[0]:.2f} .. {span[1]:.2f} on every panel, ticked every "
          f"{VBL_Y_STEP} on the {VPP_Y_ANCHOR} grid")
    sizes = {lab: _vpp_marker(lab, 0)[1] * VBL_MARKER_SCALE
             for lab in [VPP_FOCUS] + baselines}
    print(f"    markers x{VBL_MARKER_SCALE} in the Highlight tone: "
          + ", ".join(f"{lab} {d:.2f}" for lab, d in sizes.items())
          + f" pt, largest {max(sizes.values()):.2f} pt")

    rows = []
    for baseline in baselines:
        rows += _vbl_pair(out, data, ref, baseline, ylabel, span)
    sheet = _vbl_combined(out, data,
                          [b for b in baselines if b not in VBL_COMBINED_SKIP],
                          ylabel, span)
    _vbl_stats(out, rows, ylabel, span, sheet)


# ════════════════════════════════════════════════════════════════════════════════
# fig-vina-per-atom — Vina Dock against ligand size, in the fig-vina-3line style
# ════════════════════════════════════════════════════════════════════════════════
# FOUR MAIN FILES: two statistics x two arm sets, plus the grid, a median-all opacity
# test, and one CSV behind them. It was TWELVE until 2026-09-22 -- the same four built over three reference variants
# -- and is now the one variant that draws no crystal reference at all; the version token
# went off the names with the other two, so a file is `dock-per-atom-median-core` and not
# `dock-per-atom-v3-median-core`. Which statistic a figure shows is carried by its filename and by its
# y-axis name ("Vina Dock / mean" / "Vina Dock / median", set over two lines so the pair of
# panels keeps one narrow left margin), exactly as in the 3-line figures -- there are no
# panel titles, so the y name is the only thing telling two otherwise identical figures
# apart. One CSV serves all six: the per-arm columns do not depend on which arms are
# drawn, and `plotted_in`, `reference_source` and `reference_firm` are the columns that
# carry what does.
#
# ── WHICH LIGANDS THE GREY CURVE IS ──────────────────────────────────────────────
# There are two possible reference populations. Which one is drawn is stated on EVERY run,
# in the progress line and in the CSV's `reference_source` and `n_reference_total` columns.
# The KEY names it only when it is the CrossDocked set -- "the 79" quietly passed off as
# "CrossDocked" is the mistake this guards against, and that mistake only runs one way.
#
# WHY THE KEY IS ASYMMETRIC, since a matching "(79 pockets)" looks more careful and is not.
# Every other figure in this section draws "Reference ligand" and means these same 79
# crystal ligands. Annotating it here alone would put one key at odds with its neighbours
# while describing an identical population, and a reader moving between them would
# reasonably infer that THIS reference is some other set. It is not. The ambiguity an
# annotation prevents does not exist until a second population does, and at that point the
# CrossDocked label carries the distinction exactly where it is needed. The bare label also
# kept v1 and v2 byte-identical to the figures this port was verified against, which was
# the check the whole family rested on: between an annotation that is redundant in every
# figure already meaning the 79, and losing that check, the annotation goes. Provenance is
# not lost either way -- the log and the CSV carry it unconditionally, which is where a
# reader who needs it will look.
#
#   p79          The 79 crystal ligands of the test set, one per pocket: the docked value
#                out of each run's eval_docking_results_full79.json, the ligand's own
#                heavy-atom count out of that pocket's metrics.json `reference` block.
#                Drawn as plain "Reference ligand", the same key every other figure in the
#                section uses for these same molecules. This is the fallback.
#   crossdocked  The whole CrossDocked set, docked per ligand on another box and dropped
#                into the results bundle at
#                results/task2-drugdesign/Reference/crossdocked_reference_dock.json,
#                which arrives with that bundle's Reference folder. Drawn as
#                "Reference ligand (CrossDocked, n=...)". Schema, so the upload matches:
#
#                    {"protocol": {...}, "ligands": [{"n_atoms": 24, "vina_dock": -8.1},
#                                                    ...]}
#
#                `ligands` must be a non-empty list and every entry must carry both
#                `n_atoms` and `vina_dock`; anything else in the file is ignored, and
#                `protocol` is echoed into the run log so the figure states what it was
#                docked under. A file that does not parse fails LOUDLY with its path and
#                this schema rather than falling back to the 79 behind the reader's back.
#
# The file decides: crossdocked when it is there, p79 when it is not.
# VOXBIND_VINA_REFERENCE=p79|crossdocked forces one, and errors if the forced one is not
# available. The choice is printed either way.
#
# THIS IS WHY THE THINNESS LOGIC STAYS even though CrossDocked will mostly retire it. With
# tens of thousands of ligands behind it the rolling window is never thin and the fade
# never fires, which is the whole point of the upload; with the 79 it is thin at both ends
# and the fade is the honesty cue. The code does not need to know which -- it fades what is
# thin.
#
# ── TWO ARM SETS ─────────────────────────────────────────────────────────────────
# The same split every other figure in this section makes:
#
#   core  VoxBind, CoE and CoDE for mean; VoxBind and CoDE for median.
#   all   every arm that ACTUALLY HAS per-molecule Vina on this box, which today is those
#         two plus TargetDiff.
#
# WHAT AN ARM NEEDS TO BE IN `all`: `<its root>/eval_docking_results_full79.json`, carrying
# `per_target[].per_mol[]` entries with both `n_atoms` and `vina_dock`. An arm whose file is
# absent is SKIPPED WITH A PRINTED NOTE naming the path, not a crash and not a silent drop,
# so the day a file lands `all` simply grows -- adding a method is data, not surgery.
#
# The five published baselines are listed in VPA_ARMS and are skipped today. AR,
# Pocket2Mol, DiffSBDD, DecompDiff and FuncBind are staged here as sample directories
# (exps/baselines_pose/<key>/target_XX/) and scored for PoseBusters and PoseCheck, but
# their metrics.json carries `docking: "none"`: the Vina run behind
# results/task2-drugdesign/_shared/baselines_eval/summary_density79.json happened on the evaluation server
# and only its AGGREGATES came back, so there is no (heavy atoms, vina_dock) pair per
# molecule to bin. Docking them here under this protocol is ~89 CPU-hours per method
# (79 pockets x ~100 molecules, exhaustiveness 32, whole receptor). So `all` DOES NOT MEAN
# "all eight published methods" today, and the run log says which arms it did mean.
#
# The two sets do not share an x span, and that is the point of drawing both: x runs only
# where EVERY drawn arm clears MIN_N, so dropping TargetDiff buys a count back at the right
# end (core reaches 45, all stops at 44) and removes the two interior counts where
# TargetDiff alone is thin.
#
# ── NO CRYSTAL REFERENCE ON EITHER PANEL ─────────────────────────────────────────
# There were three variants until 2026-09-22, differing only in what the crystal reference
# did: v1 GATED its Vina curve where the rolling window thinned (on the 79 the grey stopped
# at 36 atoms), v2 ran it the full span and FADED the thin stretch, v3 drew it nowhere. The
# same four figures were built three times over. On request the family is now v3 alone, and
# VPA_REF keeps its three knobs, so what the other two chose between is recorded here and
# the grey is one flag from coming back exactly as it was.
#
#   WHAT NO REFERENCE BUYS is the y axis and the strip. Nothing on the panel reaches past
#   the models, so the -3..-11 band they are actually compared in fills the height instead
#   of being stretched to -15.5 by TNKS1 (target_72, -15.99) and AKT1 (target_80, -14.66),
#   the two crystal ligands at 42-44 atoms that v1 and v2 both paid for; and with no grey
#   step or fill below, the model areas are read without a fourth series over them.
#   WHAT IT GIVES UP is the benchmark itself: "is this arm better than the crystal ligand
#   at this size" and "does it generate ligands the size of the real one" are not questions
#   these four panels answer. The values are still in the CSV beside every count, and THE
#   GRID -- one cell per method, below -- still draws the grey in every cell, which is where
#   that comparison now lives.
#   WHAT v1 SPENT INSTEAD: nothing on y, but its grey ended eight atoms short of the panel
#   below it, which reads as missing data rather than as thin data. Neither was free, which
#   is why all three were built and why the choice is written down rather than merely made.
#
#   ON THE 79 THE OPEN GATE CHANGED NOTHING while the curve was drawn. Over the plotted
#   span the reference's window never falls below three ligands -- its minimum, at 43-45
#   atoms -- which v2's floor already admits, so v3's grey and v2's were the same line. The
#   gate only ever mattered where x reaches further right: the 79 crystal ligands span 6 to
#   57 heavy atoms but only five sit past 40, and from 46 up the window is down to one or
#   two. Against the CrossDocked reference it is moot in the other direction -- nothing is
#   thin.
#
# ── WHAT THEY SHOW ───────────────────────────────────────────────────────────────
# Two stacked panels over one x axis, the ligand's heavy-atom count.
#
#   top     Vina Dock mean or median at each exact heavy-atom count, for every arm in the
#           set. No crystal reference, per the section above.
#   bottom  How many molecules each set puts at each size, as a share of its own molecules
#           -- a share and not a count, because the reference set and ~7,900 generated
#           molecules do not share a count axis. This is what makes the top panel's tails
#           trustworthy or not, and is itself the finding the figure exists to guard
#           against: the arms differ in the size distribution they generate at least as
#           much as in per-atom binding quality (see the size-confound note; raw pooled
#           Vina Dock is ~80% a size statistic). The reference is off this panel too, so
#           "does the model generate ligands the size of the real one" is a question for the
#           grid below and for the CSV, not for this figure.
#
# EVERY ARM IS GATED THE SAME WAY, per arm rather than per figure. A heavy-atom count
# enters x only if EVERY drawn arm has MIN_N molecules there, which sets the two ends;
# inside that span an arm that dips below MIN_N (TargetDiff does, at 38 and 41 atoms, which
# is why only `all` is broken there) has its curve BROKEN rather than interpolated, because
# a bridge over a count where one method generated 15 molecules is a drawn claim about data
# that is not there. The distribution panel underneath keeps drawing at those counts --
# being thin is exactly what it is for.
#
# ONE STATISTIC PER FIGURE. Both were drawn in one panel at first, dash against solid, and
# it said nothing extra for the ink: at every size the two run within ~0.2 kcal/mol of each
# other, so six curves were three curves drawn twice and the pair merely thickened and
# blurred each series. They are separate figures instead. That near-agreement IS a result
# -- the Ours/VoxBind gap is the whole distribution shifting, not a tail dragging the mean
# -- and it is what makes the two figures nearly interchangeable; the mean is the slightly
# kinder one to CoDE (below VoxBind at 35/40 sizes against the median's 31/40, mean
# per-size gap -0.441 against -0.388 on the `all` span), because CoDE's size distribution
# reaches further right.
#
# INK IS THE SECOND CHANNEL, and it says what a series is FOR: the two arms being compared
# are the thickest, solid, and the only ones with an area under their distribution;
# everything else is thinner and DASHED, because it is what the pair is read against rather
# than a further competitor, and four filled areas on the lower panel would be a stack no
# one can read through. That is also the rule any newly docked baseline arrives under: it
# joins as context ink, not as a third solid curve. CoDE is periwinkle #8291E8 here, the
# lighter tint of our blue this Vina family carries, as the palette records.
#
# PROTOCOL. The published-baseline protocol, exactly as in the 3-line figures: whole
# `*_rec.pdb` receptor, exhaustiveness 32, all 79 pockets, from
# `eval_docking_results_full79.json`. The same runs, so this figure and the 3-line figures
# sit on one axis -- TargetDiff included, which 74_dock_targetdiff_full79.sh re-docked here
# for exactly that reason. The CrossDocked reference is docked elsewhere, so its own
# `protocol` block is echoed into the run log rather than assumed to match.
#
# WHY NO CONFIDENCE BANDS. The thing a band would guard against here is already drawn: x is
# clipped to the counts where EVERY arm has at least MIN_N molecules, so no curve has a
# tail the others cannot answer, and the bottom panel shows how thin each end actually is.
# The per-count sample sizes are in the CSV.
#
# THE REFERENCE IS ROLLED. On the 79 there is exactly ONE crystal ligand per pocket, 1-6 at
# any exact heavy-atom count, so anything per-count off it is noise: raw, its distribution
# is a picket fence of 1.3%-tall steps reaching 7.6% where six pockets happen to share a
# size, and that spike, not the models, would set the lower panel's y scale. Both its
# curves are therefore a centred rolling window of +-REF_WIN atoms, and the window stays
# for the CrossDocked set so the two sources are read the same way. The models are NOT
# rolled, in either panel: they have hundreds of molecules per count.
VPA_BASENAME = "eval_docking_results_full79.json"
VPA_PROTOCOL = "full receptor, exhaustiveness 32, 79 pockets (baseline protocol)"

# The CrossDocked reference drop, REPO-relative because this tree gets copied between
# boxes. The folder is part of the results bundle, so it arrives with that bundle's
# Reference folder and needs no sync path of its own.
VPA_REF_DOCK = ("results", "task2-drugdesign", "Reference",
                "crossdocked_reference_dock.json")
VPA_REF_SOURCES = ("p79", "crossdocked")
VPA_REF_ENV = "VOXBIND_VINA_REFERENCE"
VPA_REF_SCHEMA = ('{"protocol": {...}, "ligands": [{"n_atoms": 24, "vina_dock": -8.1}, '
                  '...]}  -- extra keys ignored')

# The whole difference between the three figures, and all of it about the reference.
# `ref_vina` puts it on the top panel, `ref_dist` on the strip below; it earns its key
# entry by being on either. `ref_gate` names the floor its rolling window has to clear
# before a point is drawn at all -- VPA_REF_GATE maps the three settings to that number --
# and everything except "hard" fades the stretch below VPA_REF_FIRM rather than cutting it.
# `ref_dist_smooth` rolls the reference's distribution; `ref_dist_fill` gives it an area
# under it like the two compared arms have.
# The three knobs the variants differed over, at what was v3's setting: the reference is on
# neither panel, so it is not in the key either -- a series earns that entry by being drawn
# somewhere. ref_gate, ref_dist_smooth and ref_dist_fill are dead while both switches are
# off, and keep their values so the grey comes back exactly as it was if one is flipped.
VPA_REF = dict(ref_vina=False, ref_dist=False, ref_gate="full",
               ref_dist_smooth=True, ref_dist_fill=False)

VPA_FIELD, VPA_REF_FIELD = "vina_dock", "ref_vina_dock"
VPA_STATS = ("mean", "median")       # one figure each, per variant and per arm set

# THE ONE OF THE FOUR THAT IS READ, and the only one with geometry of its own: mean /
# core, cut to VPA_SHORT_SCALE of the Vina panel's height on request (2026-09-16). The
# ticks come with the height. A step of 2 is five labels down a panel that is now four
# fifths as tall, which reads as a denser axis than the one it was before it shrank, and
# the locator's own answer at that height (-2.5 / -5.0 / ...) puts a decimal on an axis
# every other figure here labels in whole kcal/mol; listing them keeps the whole numbers
# and the end points -- the span is spanned, in four labels instead of five.
VPA_SHORT = ("mean", "core")
VPA_SHORT_SCALE = 0.8
VPA_SHORT_TICKS = (-2, -5, -8, -11)

# ... and x on a REVERSED LOG SCALE, asked for 2026-09-16: the figure is read for what the
# arms do at the HEAVY end, so that is the end the axis gives its width to. A log axis is
# the wrong way round for that -- it stretches the small counts -- and the mirror of it is
# a power axis, x**VPA_SHORT_XPOW.
#
# The exponent is 2 and that is not a taste: it is the exact opposite of log at this span.
# Local scale on a log axis is 1/x, so the first plotted count is stretched 39/6 = 6.5x
# harder than the last; on x**k it is k*x**(k-1), so the LAST is stretched (39/6)**(k-1)
# harder than the first, and k = 2 makes that the same 6.5x, pointing the other way. It is
# the reversal of the log axis, not merely a bend that leans the right way.
#
# What it buys: 25..39 atoms, an eighth of the counts and where CoDE pulls under VoxBind and
# the reference, takes 60% of the panel instead of the linear 42%; 6..10, where all three
# arms and the reference are one line, gives up its share. The other eleven keep the linear
# axis, so a reader comparing this one to its median or `all` twin sees the same numbers
# under the ticks and has to notice only the spacing.
#
# Both panels take it: they are sharex, and a strip on a different x from the curves above
# it would be a different figure, not a strip.
VPA_SHORT_XPOW = 2.0
# The linear figure's own multiples of five, and NOTHING ELSE: the tick set is the one thing
# about this figure that should not move when the axis does, so a reader holding it against
# its median or `all` twin finds the same labels and has only the spacing to notice. The
# first plotted count, 6, was tried as an extra low tick and taken out again -- this axis
# puts 10 within 4% of the spine and the two labels touch -- which is also why the stretch
# below 10 needs no label of its own. (On a LOG axis the same set would be wrong: 10 lands a
# third of the way in there, and the low end does need closing.)
VPA_SHORT_XTICKS = (10, 15, 20, 25, 30, 35)
# Padding past the two end counts, as a fraction of the plotted span measured IN THE SCALE,
# so both ends get the same visible gap -- on a non-linear axis a constant number of atoms
# does not. The linear figures pad +-0.6 atoms, which over this span is the 1.8% kept here.
VPA_SHORT_XPAD = 0.018


def _vpa_pow(v):
    """The short figure's x transform. Clipped at 0 because matplotlib hands the scale
    values from off the ends of the view while it lays the axis out, and a negative one
    raised to a fractional power comes back nan and takes the whole axis with it."""
    return np.power(np.clip(np.asarray(v, dtype=float), 0.0, None), VPA_SHORT_XPOW)


def _vpa_root(v):
    """Its inverse, on the same terms."""
    return np.power(np.clip(np.asarray(v, dtype=float), 0.0, None), 1.0 / VPA_SHORT_XPOW)


# THE SECOND FIGURE WITH GEOMETRY OF ITS OWN: median / all, drawn a SECOND time as
# small multiples, one method per cell, asked for 2026-09-16. It does not replace the
# single-panel version -- that one is still the twelve's -- it is written beside it under
# the same stem plus `_grid`.
#
# Why it exists: `all` puts nine arms on one pair of axes, and past about four curves a line
# chart stops being read arm by arm and starts being read as a band. One cell per method
# gives every arm its own uncrowded axes; sharing the scales across the cells is what keeps
# them comparable, so the y limits and the x span are computed ONCE over everything drawn
# and handed to all of them.
#
# EVERY ARM GETS A CELL, ours included, and each cell holds that arm against THE REFERENCE
# and nothing else -- in BOTH of the cell's panels. Nine arms is exactly 3 x 3, and a cell
# reads as one clean question -- where does this method sit against the crystal ligands at
# each size, and does it put its molecules at the same sizes -- rather than as a three-way
# crossing. The reference repeats in all nine and is the only thing the key has to name; the
# method is named by the cell's title, which binds the name to the curve without a nine-row
# legend. The title is set in plain INK, not the method's colour (2026-09-16): colour stays
# on the data -- the curve, the step and the band -- and the type reads as type.
#
# The strip carries the reference EVEN THOUGH the single-panel figure takes it off the
# strip. That setting exists because nine arms plus a tenth grey line is an unreadable
# strip; per cell there are two, so the reason is gone and the comparison is worth having --
# a curve sitting under the reference at a size nobody generates says less than the same
# curve where both populations are dense. It is rolled and left unfilled, as v2 drew it:
# the filled band is the method's, and an area under a rolling average would claim a
# precision the averaging just spent.
VPA_GRID = ("median", "all")
# Columns are fixed and rows follow from how many arms there are, rather than both being
# fixed: a tenth docked baseline should push this to 4 x 3 with two cells left empty, not
# silently fall off the bottom of a hard-coded nine.
VPA_GRID_COLS = 3
# Inches per cell, the Vina panel and its strip together. About half again the single
# figure's panel width, which is what "big" means here: at three across, the shared 14pt
# ticks and 15.5pt names come out LARGER relative to a cell than they are in the one-panel
# figure, so nothing has to be restyled to stay legible. Reshaped 2026-09-16 from
# 4.15 x 4.35 to 1.2x the width and 0.9x the height: wider and flatter cells, so a slope
# reads as a slope rather than as a drop. (0.8x was tried first and walked back to 0.9x
# when the spacer row between cell rows came out, below.)
VPA_GRID_CELL_W, VPA_GRID_CELL_H = 4.15 * 1.2, 4.35 * 0.9
# NO SPACER BETWEEN CELL ROWS, asked for 2026-09-16. The grid used to carry an empty
# gridspec row (ratio 0.62) under every strip but the last; without it a cell row's strip
# is followed directly by the next row's Vina panel. tight_layout still solves ONE hspace
# for the whole gridspec, and the tallest thing it has to clear at any row boundary is the
# next cell's title, so the title keeps exactly its own height of room -- the same gap
# that already sits between a Vina panel and its strip -- and nothing more.
# Column gap, handed to tight_layout as w_pad rather than to the gridspec as `wspace`, and
# THAT IS NOT A STYLE CHOICE. A gridspec built with an explicit hspace/wspace reports
# locally_modified_subplot_params(), and tight_layout SILENTLY REFUSES a figure holding one
# -- get_subplotspec_list hands back None for every axes on it, the engine warns once and
# does nothing, and the figure keeps matplotlib's stock 0.125/0.9/0.11/0.88 margins. That
# is what this figure had: 1.3 inches of dead paper across the top and 1.2 down the right,
# on a figure whose whole point is to be large. Spacing asked for through the pads leaves
# the gridspec unmodified, so tight_layout runs and the margins come back.
VPA_GRID_WPAD = 1.1
VPA_GRID_TITLE_FS = 17
# Headroom over the topmost curve IN ANY CELL. The one-panel figure asks margins() for 7.5%
# per axes; here the axes are shared, so the same figure is +-this fraction of the span
# applied once to the limits every cell then gets.
VPA_GRID_YPAD = 0.075


def _vpa_grid_cells(arms):
    """Every arm, in READING ORDER for the grid, and the shape that holds them.

    The three model arms go LAST, asked for 2026-09-16 -- the six published baselines fill
    the grid from the top and ours lands in the bottom row -- and among themselves they take
    VPA_KEY_ORDER, which is already VoxBind then CoE then ours. So the bottom row of the
    grid and the tail of the key read the same way round, and one tuple still decides it.
    Everything else keeps its VPA_ARMS order."""
    tail = [a for lab in VPA_KEY_ORDER for a in arms if a[0] == lab]
    cells = [a for a in arms if a[0] not in VPA_KEY_ORDER] + tail
    rows = -(-len(cells) // VPA_GRID_COLS)
    return cells, rows, VPA_GRID_COLS

# The y names are set over TWO LINES. One line each ("Vina Dock median", "% of ligands")
# put the widest label's length into the left margin of a 7.6-inch figure; broken after
# the metric they are the same words in half the width, and align_ylabels below keeps the
# two panels' labels on one left edge. The statistic still lives in the y name and
# nowhere else, exactly as in the 3-line figures.
VPA_Y_LABEL = "Vina Dock\n{stat}"
VPA_D_LABEL = "Generated\nfraction"

# These are DATA KEYS as well as labels (they head this folder's CSV columns), so they
# stay plain; display() dresses them for a legend -- VoxBind picks up its sigma=0.9
# subscript there. CoDE is \textsc{CoDE} in the .tex files; matplotlib has no small caps
# without a TeX backend, so the figures carry the plain string.
VPA_VOX_LABEL, VPA_OUR_LABEL = "VoxBind", "CoDE"
VPA_OUR_COLOR = pale(VPA_OUR_LABEL)

# Line weights in points. VPA_MODEL_BOOST widens ONLY the two arms the figure exists to
# compare -- both their Vina curves and their distribution steps -- and is applied as a
# factor on the shared base weights rather than baked into them, so the gap it opens
# against the context series' thinner ink stays visible as a decision.
VPA_MODEL_BOOST = 1.2
VPA_MODEL_LW, VPA_BASE_LW = MODEL_LW * VPA_MODEL_BOOST, 1.9
VPA_DIST_LW = DIST_LW * VPA_MODEL_BOOST

# The reference dash is the shared DASH: legible at hairline weights, and it does not
# shimmer where it runs close to a model curve. VPA_BASE_DASH is the context arms', longer
# so the two kinds of dashed series are told apart by rhythm as well as by colour.
VPA_BASE_DASH = (0, (6, 2))

# label, run root, line width, dash -- DRAWN IN THIS ORDER, so ours lands on top and the
# context series sit under it. Roots are REPO-relative (E) or resolved off the sibling
# checkout (BASEDRUG); nothing here is an absolute path. The five baselines are listed and
# will be SKIPPED until their eval_docking_results_full79.json exists, which is how a newly
# docked method joins `all` without a code change.
VPA_ARMS = [
    ("AR",          f"{E}/baselines_pose/ar",         VPA_BASE_LW, VPA_BASE_DASH),
    ("Pocket2Mol",  f"{E}/baselines_pose/pocket2mol", VPA_BASE_LW, VPA_BASE_DASH),
    ("DiffSBDD",    f"{E}/baselines_pose/diffsbdd",   VPA_BASE_LW, VPA_BASE_DASH),
    ("DecompDiff",  f"{E}/baselines_pose/decompdiff", VPA_BASE_LW, VPA_BASE_DASH),
    ("FuncBind",    f"{E}/baselines_pose/funcbind",   VPA_BASE_LW, VPA_BASE_DASH),
    ("TargetDiff",  str(BASEDRUG / "eval" / "targetdiff"), VPA_BASE_LW, VPA_BASE_DASH),
    # CoE keeps its own run tree like the other two model arms: it carries
    # eval_docking_results_full79.json (same 79 pockets, full-receptor scope, exhaustiveness
    # 32), so it never needs the bundle fallback the staged baselines take.
    ("CoE",         f"{E}/260908_fusion_default_cv2_scratch_8gpu/samples/samples_ep350_test79_n100",
     VPA_MODEL_LW, "-"),
    (VPA_VOX_LABEL, f"{E}/_vanilla_ep923/samples/full_eval_ep923", VPA_MODEL_LW, "-"),
    (VPA_OUR_LABEL, f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
     VPA_MODEL_LW, "-"),
]

VPA_ARMS = ordered_methods(VPA_ARMS, key=lambda row: row[0])
# The published all and mean-core variants retain CoE as a focus arm. Median-core
# uses only VoxBind and CoDE; the opacity test also keeps only those two prominent.
VPA_FOCUS = (VPA_VOX_LABEL, VPA_OUR_LABEL, "CoE")
VPA_CORE = (VPA_VOX_LABEL, VPA_OUR_LABEL)
VPA_TEST_CONTEXT_ALPHA = 0.20
# Every arm takes its colour from the shared palette, so a method is the same colour here
# as in the PoseBusters and PoseCheck figures. CoDE is the one override: this Vina family
# carries the lighter tint of our blue.
VPA_COLORS = {lab: (VPA_OUR_COLOR if lab == VPA_OUR_LABEL else pale(lab))
              for lab, *_ in VPA_ARMS}
VPA_COLORS[REF_LABEL] = REF_COLOR
# The p79 crystal ligand is the same molecule in every run that carries one, but it is
# resolved BY LABEL, never by position in VPA_ARMS -- the staged baselines have no
# reference block at all, and a positional pick would silently become one of them.
VPA_REF_ROOT = next(root for lab, root, _, _ in VPA_ARMS if lab == VPA_VOX_LABEL)

# The reference window is +-REF_WIN atoms, and these are the three floors it can be held
# to. VPA_REF_FIRM is where it holds enough ligands to be read as a curve (v1 gated there,
# v2 and v3 faded below it); VPA_REF_FLOOR was v2's floor for drawing a point at all;
# VPA_REF_ANY is the surviving setting's, and refuses only a window holding nothing. The curve is computed
# once at VPA_REF_ANY and masked down, so one array serves all three gates and the CSV
# reports every value any of them can draw.
VPA_REF_FIRM, VPA_REF_FLOOR, VPA_REF_ANY = 12, 3, 1
VPA_REF_GATE = {"hard": VPA_REF_FIRM, "fade": VPA_REF_FLOOR, "full": VPA_REF_ANY}
VPA_REF_THIN_ALPHA = DATA_ALPHA


def _vpa_slug(label):
    """The CSV's column prefix for an arm: lower case, no spaces."""
    return label.lower().replace(" ", "_").replace("+", "plus")


# Bundle folder for an arm that has no run tree here. The five published baselines were
# docked on the evaluation server and only their aggregates came back until 2026-09-13, when the bundle
# gained <Method>/eval/vina_docking/per_molecule.csv -- target, n_atoms and the three Vina
# columns per molecule, which is exactly what this figure bins. They are read from THERE.
# Our own arms keep their run tree: the bundle's `VoxBind-vanilla` is res_test_100, a
# DIFFERENT run from the _vanilla_ep923 this figure has always drawn, and swapping the
# source underneath an arm would silently redraw it.
VPA_BUNDLE = {"AR": "AR", "Pocket2Mol": "Pocket2Mol", "DiffSBDD": "DiffSBDD",
              "DecompDiff": "DecompDiff", "FuncBind": "FuncBind"}


def _vpa_load_bundle(label):
    """{heavy atoms: [vina dock, ...]} over the 79 density pockets, from the bundle CSV."""
    folder = VPA_BUNDLE.get(label)
    rows = vina_rows(folder, density79=True) if folder else None
    if not rows:
        return None, 0
    sizes = collections.defaultdict(list)
    for r in rows:
        if r.get(VPA_FIELD) is not None and r.get("n"):
            sizes[r["n"]].append(r[VPA_FIELD])
    return (sizes, len({r["target"] for r in rows})) if sizes else (None, 0)


def _vpa_load(root, label=None):
    """{heavy atoms: [vina dock, ...]} pooled over the 79 pockets of one run, or None if
    this arm has not been docked here -- an absent file is a method waiting for its Vina
    run, not an error, and the caller says so in the log instead of drawing an empty
    curve. An arm with no run tree falls to its bundle CSV (see VPA_BUNDLE)."""
    path = os.path.join(root, VPA_BASENAME)
    if not os.path.exists(path):
        return _vpa_load_bundle(label) if label else (None, 0)
    with open(path, encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    sizes = collections.defaultdict(list)
    for t in per_target:
        for m in t.get("per_mol") or []:
            if isinstance(m, dict) and m.get(VPA_FIELD) is not None and m.get("n_atoms"):
                sizes[m["n_atoms"]].append(m[VPA_FIELD])
    return sizes, len(per_target)


def _vpa_load_reference(root):
    """The p79 reference: {heavy atoms: [crystal-ligand vina dock, ...]}, one entry per
    pocket.

    The docked value is in the results file, but the ligand's own atom count is not -- it
    lives in that pocket's metrics.json, under `reference`. Pockets missing either are
    dropped and named by the caller."""
    with open(os.path.join(root, VPA_BASENAME), encoding="utf-8") as handle:
        per_target = json.load(handle)["per_target"]
    sizes, dropped = collections.defaultdict(list), []
    for t in per_target:
        path = os.path.join(root, t["target"], "metrics.json")
        n_atoms = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                n_atoms = (json.load(handle).get("reference") or {}).get("n_atoms")
        value = t.get(VPA_REF_FIELD)
        if n_atoms and value is not None:
            sizes[n_atoms].append(value)
        else:
            dropped.append(t["target"])
    return sizes, dropped


def _vpa_load_crossdocked(path):
    """The CrossDocked reference drop: {heavy atoms: [vina dock, ...]} over every ligand in
    it, plus whatever `protocol` block it carries.

    Every failure raises with the path AND the schema. This file is written on another box
    and pulled in, so the realistic failure is a shape mismatch, and the one thing that
    must never happen is falling back to the 79 while the key still says CrossDocked."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    ligands = payload.get("ligands") if isinstance(payload, dict) else None
    if not isinstance(ligands, list) or not ligands:
        raise ValueError(f"{path}: needs a non-empty 'ligands' list\n"
                         f"  schema: {VPA_REF_SCHEMA}")
    sizes = collections.defaultdict(list)
    for i, m in enumerate(ligands):
        if not isinstance(m, dict) or m.get("n_atoms") is None \
                or m.get(VPA_FIELD) is None:
            raise ValueError(
                f"{path}: ligand {i} has no 'n_atoms' and/or '{VPA_FIELD}'\n"
                f"  schema: {VPA_REF_SCHEMA}")
        sizes[int(m["n_atoms"])].append(float(m[VPA_FIELD]))
    return sizes, payload.get("protocol")


def _vpa_reference_set():
    """(source, {heavy atoms: [vina dock]}, the key's label, the log line, ligand count).

    The file decides -- CrossDocked when the drop is there, the 79 when it is not -- and
    VPA_REF_ENV forces one either way, erroring rather than silently substituting the
    other. Whichever wins, its name goes in the key, in the log and in the CSV."""
    path = REPO.joinpath(*VPA_REF_DOCK)
    forced = os.environ.get(VPA_REF_ENV) or None
    if forced is not None and forced not in VPA_REF_SOURCES:
        raise ValueError(f"{VPA_REF_ENV}={forced!r}: expected one of "
                         f"{' or '.join(VPA_REF_SOURCES)}")
    if forced == "crossdocked" and not path.exists():
        raise FileNotFoundError(
            f"{VPA_REF_ENV}=crossdocked, but {path} is not there\n"
            f"  fetch the results bundle's Reference folder\n"
            f"  schema: {VPA_REF_SCHEMA}")
    if forced == "p79" or (forced is None and not path.exists()):
        sizes, dropped = _vpa_load_reference(VPA_REF_ROOT)
        n = sum(len(v) for v in sizes.values())
        note = (f"p79 · {n} crystal ligands, one per pocket, from {VPA_BASENAME}"
                + (f" (dropped {', '.join(dropped)})" if dropped else ""))
        return "p79", sizes, REF_LABEL, note, n
    sizes, protocol = _vpa_load_crossdocked(path)
    n = sum(len(v) for v in sizes.values())
    if isinstance(protocol, dict):
        protocol = ", ".join(f"{k}={v}" for k, v in protocol.items())
    note = (f"crossdocked · {n:,} ligands from {path.relative_to(REPO)}"
            + (f" · docked under {protocol}" if protocol else ""))
    # One line. It is roughly three times the width of a method label, which is exactly
    # why it cannot sit INSIDE the method grid: a legend column is as wide as its widest
    # entry, so dropping this into one would stretch that column and break the 2 x 4. It
    # gets its own centred key above the grid instead.
    #
    # The count alone carries the distinction the annotation exists for: the p79 set is 79
    # and this one is six figures, so "(n=100,090)" cannot be mistaken for the test
    # pockets' ligands, and the bare "Reference ligand" keeps meaning the 79 exactly as it
    # does in every other figure of the section. Which population it is, and under what
    # protocol, is printed in full on every run and carried in the CSV.
    return "crossdocked", sizes, f"{REF_LABEL} (n={n:,})", note, n


def _vpa_arm_sets(arms):
    """(name, arms) for the two sets, core first. `arms` is what actually has data."""
    return (("core", [a for a in arms if a[0] in VPA_FOCUS]), ("all", list(arms)))


def _vpa_model_curve(sizes, xs, stat):
    """mean/median at each exact heavy-atom count, None below MIN_N.

    None and not the value: a count an arm put fewer than MIN_N molecules at is a number
    the other arms cannot be asked to answer, and plotting it would also let matplotlib
    draw a straight segment over it as though the arm had been measured there."""
    f = st.mean if stat == "mean" else st.median
    return [f(sizes[a]) if len(sizes.get(a, ())) >= MIN_N else None for a in xs]


def _vpa_window_counts(sizes, xs):
    """How many reference ligands each plotted point's rolling window pools."""
    return [sum(len(v) for n, v in sizes.items() if abs(n - a) <= REF_WIN) for a in xs]


def _vpa_reference_curve(sizes, xs, stat, floor):
    """The reference over a centred +-REF_WIN window; None where the window holds fewer
    than `floor` ligands, which leaves a gap in the line rather than an invented value."""
    f = st.mean if stat == "mean" else st.median
    out = []
    for a in xs:
        pool = [v for n, vals in sizes.items() if abs(n - a) <= REF_WIN for v in vals]
        out.append(f(pool) if len(pool) >= floor else None)
    return out


def _vpa_gated(values, counts, gate):
    """The reference curve held to one variant's floor: the value where its window clears
    VPA_REF_GATE[gate], None below. The curve itself is computed once at VPA_REF_ANY, so
    the three gates are three masks over one array rather than three recomputations."""
    floor = VPA_REF_GATE[gate]
    return [v if c >= floor else None for v, c in zip(values, counts)]


def _vpa_split_firm(values, counts):
    """The reference curve cut into the part its window can carry and the part it cannot.

    Returns (firm, thin): both full-length, each holding None wherever the other holds the
    value. A point on the boundary belongs to BOTH, so the solid and the faded stretch meet
    rather than leaving a gap -- a gap would say "no data here", which is the opposite of
    what the fade is for."""
    firm_at = [v is not None and c >= VPA_REF_FIRM for v, c in zip(values, counts)]
    n = len(firm_at)
    edge = [firm_at[i] and any(not firm_at[j] for j in (i - 1, i + 1) if 0 <= j < n)
            for i in range(n)]
    firm = [v if f else None for v, f in zip(values, firm_at)]
    thin = [v if (not f) or e else None for v, f, e in zip(values, firm_at, edge)]
    return firm, thin


def _vpa_furniture(ax, *, ylabel, xlabel, xs):
    """The 3-line figure's axes, over the plotted heavy-atom span. The y names are two
    lines each, so they are centred on one another rather than left-ragged."""
    furniture(ax, ylabel=ylabel, xlabel=xlabel, xlim=(xs[0] - 0.6, xs[-1] + 0.6),
              xloc=XTICK_STEP)   # ticks and vertical rules both, unlike the 3-line figure
                                 # where the rules fall between the labelled ranks
    ax.yaxis.label.set_multialignment("center")


# THE KEY'S OWN ORDER for the three model arms, asked for 2026-09-16: ours last, the two
# it is read against ahead of it. It is NOT the order they are drawn in -- CoDE is drawn
# last so it lands on top where the curves cross, and VoxBind and CoE would swap if the key
# followed VPA_ARMS -- so the reorder is done on the HANDLES only, after the lines exist.
# Any arm not named here keeps its drawing order; the three are placed, in this order, in
# the slots those three already occupy, so in `all` the surrounding baselines do not move.
VPA_KEY_ORDER = (VPA_VOX_LABEL, "CoE", VPA_OUR_LABEL)


def _vpa_key_sorted(series, handles):
    """`handles` with the VPA_KEY_ORDER arms rearranged into that order, in place."""
    slots = [i for i, a in enumerate(series) if a[0] in VPA_KEY_ORDER]
    named = sorted((series[i][0] for i in slots), key=VPA_KEY_ORDER.index)
    by_label = {series[i][0]: handles[i] for i in slots}
    out = list(handles)
    for i, lab in zip(slots, named):
        out[i] = by_label[lab]
    return out


# Above this many entries the key stops fitting in the panel's empty corner and starts
# covering the curves it explains. `core` has up to three and keeps the corner; `all` has nine
# and takes a strip under the figure instead.
VPA_KEY_INSIDE_MAX = 4
# Inside that strip the reference is NOT one of the methods -- it is the benchmark they are
# read against -- so it takes its own centred line above a 2 rows x 4 columns grid of the
# eight methods, which is the shape that fits this figure's width without reaching the axes.
VPA_KEY_NCOL = 4


def _vpa_legend(ax, series, ref_label, *, fig=None, alpha_for=None):
    """The key, drawn in plotting order, its handles carrying each series' dash so an
    identity is readable from the key alone.

    WHERE IT GOES DEPENDS ON HOW MANY THERE ARE. Three series fit in the empty upper right
    and belong there -- next to the data, costing no height. Nine do not: at this panel
    size a nine-row box reaches more than halfway down and across, and in the `all` variant
    it sat squarely over the x axis. Past VPA_KEY_INSIDE_MAX the key moves beneath the
    figure as a reference row over a 2x4 grid of methods.

    `ref_label` is bare "Reference ligand" for the 79 -- the section's own name for them --
    and names the size only for the CrossDocked set, which is the one that would otherwise
    be read as the 79 (see the banner)."""
    handle = lambda lab, colour, lw, style: Line2D(
        [], [], color=colour, lw=lw, ls=style,
        alpha=alpha_for(lab) if alpha_for else 1.0,
        label=ref_label if lab == REF_LABEL else display(lab))
    handles = _vpa_key_sorted(series, [handle(*a[:4]) for a in series])
    if len(handles) <= VPA_KEY_INSIDE_MAX or fig is None:
        legend(ax, handles, loc="upper right")
        return []

    ref = [h for a, h in zip(series, handles) if a[0] == REF_LABEL]
    methods = [h for a, h in zip(series, handles) if a[0] != REF_LABEL]
    ncol = VPA_KEY_NCOL

    # ONE box, built as TWO legends. A legend column is as wide as its widest entry, so
    # the reference -- one line, and about three times the width of a method name -- cannot
    # be a cell of the method grid without stretching whichever column it landed in, and
    # the 2 x 4 would stop being a grid. matplotlib cannot span a cell either. So the two
    # lines get their own layouts, stacked and each centred, and then the frames come off
    # and a single rectangle is drawn round the pair: the reader sees one key, while
    # matplotlib still lays out the grid and the reference independently, which is what
    # keeps the grid even and the reference genuinely centred over it.
    legs = [legend(fig, row_major(methods, ncol), loc="lower center", ncol=ncol,
                   fontsize=11.5, bbox_to_anchor=(0.5, 0.008), columnspacing=1.5)]
    if not ref:
        return legs
    fig.canvas.draw()
    h = legs[0].get_window_extent(fig.canvas.get_renderer()).height / \
        fig.get_size_inches()[1] / fig.dpi
    legs.append(legend(fig, ref, loc="lower center", ncol=1, fontsize=11.5,
                       bbox_to_anchor=(0.5, 0.008 + h + VPA_KEY_ROW_GAP)))
    return legs + [one_frame(fig, legs)]


def _vpa_clear_keys(fig, legs, axes):
    """Move the axes up until the LOWEST THING THEY DRAW clears the figure-level keys.

    tight_layout lays the axes out over the whole figure and cannot see a legend that
    belongs to the FIGURE, so it puts the x label underneath one. Raising `bottom` is not
    enough on its own either: the x label hangs BELOW the axes box and travels up with it,
    so the thing that has to clear the key is the axes' tight bounding box, not its frame.
    Measure both and close the gap -- the same after-the-fact correction fit() makes for an
    overrunning y label, and it keeps holding when the key gains a row or the type moves."""
    if not legs:
        return
    for _ in range(3):                      # a shift changes the layout; re-measure
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        h = fig.get_size_inches()[1] * fig.dpi
        key_top = max(l.get_window_extent(r).y1 for l in legs) / h
        drawn_bottom = min(a.get_tightbbox(r).y0 for a in axes) / h
        gap = key_top + 0.02 - drawn_bottom
        if gap <= 0.001:
            break
        sp = fig.subplotpars
        fig.subplots_adjust(bottom=min(sp.bottom + gap, sp.top - 0.1))


def _vpa_resolve(arms, data, ref):
    """Everything that depends on WHICH arms are drawn, for one arm set.

    x is the only thing the two sets really argue about, and it cascades: the per-count
    curves, the window counts and the shares are all evaluated over it, and the reference's
    rolled share is truncated by its two ends. Per-arm values themselves do not depend on
    the set -- an arm's mean at 30 atoms is its mean at 30 atoms -- which is what lets one
    CSV stand behind both."""
    firm = [a for a in sorted(set.intersection(*(set(data[lab]) for lab, *_ in arms)))
            if all(len(data[lab].get(a, ())) >= MIN_N for lab, *_ in arms)]
    xs = list(range(firm[0], firm[-1] + 1))
    curves = {}
    for stat in VPA_STATS:
        curves[(REF_LABEL, stat)] = _vpa_reference_curve(ref, xs, stat, VPA_REF_ANY)
        for label, *_ in arms:
            curves[(label, stat)] = _vpa_model_curve(data[label], xs, stat)
    exact, totals = {}, {}
    for label, sizes in [(lab, data[lab]) for lab, *_ in arms] + [(REF_LABEL, ref)]:
        exact[label], totals[label] = share(sizes, xs)
    return {"arms": arms, "xs": xs, "curves": curves, "exact": exact, "totals": totals,
            "counts": _vpa_window_counts(ref, xs),
            "thin": {lab: [a for a in xs if len(data[lab].get(a, ())) < MIN_N]
                     for lab, *_ in arms}}


def _vpa_build(out, cfg, stat, aset, S, ref_label, *, test=False):
    """One statistic on one arm set: the figure, and the line about it for the run log."""
    stem = f"vina_dock_per_atom_{stat}_{aset}" + ("_test" if test else "")
    xs, curves, counts = S["xs"], S["curves"], S["counts"]
    short = (stat, aset) == VPA_SHORT
    focus = VPA_CORE if test else VPA_FOCUS
    def opacity(label):
        return (VPA_TEST_CONTEXT_ALPHA if test and label not in focus else 1.0)

    # Reference first and ours last, so the series being read sits on top -- the same
    # ordering the 3-line figures use. The key is the union of the two panels: a series
    # drawn on either one belongs in it, and a series drawn on neither -- the reference --
    # is not in the key at all, which is what makes turning both switches off a complete
    # removal rather than a line with a label left behind.
    series = [(lab, VPA_COLORS[lab], lw, style, z)
              for z, (lab, _, lw, style) in enumerate(S["arms"], start=3)]
    if cfg["ref_vina"] or cfg["ref_dist"]:
        series.insert(0, (REF_LABEL, REF_COLOR, REF_LW, DASH, 2))
    dist_series = [s for s in series if s[0] != REF_LABEL or cfg["ref_dist"]]

    shares = dict(S["exact"])
    if cfg["ref_dist_smooth"]:
        shares[REF_LABEL] = rolled(S["exact"][REF_LABEL], xs)
    ref_values = _vpa_gated(curves[(REF_LABEL, stat)], counts, cfg["ref_gate"])

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(FIG_W, STACK_H), dpi=220, sharex=True,
        gridspec_kw=dict(height_ratios=list(HEIGHT_RATIOS)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bx.set_facecolor("white")

    for label, colour, lw, style, z in series:
        if label != REF_LABEL:
            ax.plot(xs, curves[(label, stat)], color=colour, lw=lw, ls=style, zorder=z,
                    alpha=opacity(label), solid_capstyle="round", dash_capstyle="round")
        elif not cfg["ref_vina"]:
            continue
        elif cfg["ref_gate"] == "hard":
            ax.plot(xs, ref_values, color=colour, lw=lw, ls=style, zorder=z,
                    dash_capstyle="round")
        else:
            firm, thin = _vpa_split_firm(ref_values, counts)
            for ys, alpha in ((thin, VPA_REF_THIN_ALPHA), (firm, 1.0)):
                ax.plot(xs, ys, color=colour, lw=lw, ls=style, zorder=z, alpha=alpha,
                        dash_capstyle="round")

    # The same identities below: colour for the series, dash for the ones it is read
    # against. The arms are per-count steps; the reference is a curve when it is smoothed,
    # because drawing a rolling average stepped would claim a precision it lost. Only the
    # Focus arms get an area; the test variant limits this to VoxBind and CoDE.
    for label, colour, lw, style, z in dist_series:
        stepped = label != REF_LABEL or not cfg["ref_dist_smooth"]
        filled = label in focus or (label == REF_LABEL and cfg["ref_dist_fill"])
        if filled:
            bx.fill_between(xs, shares[label], step="mid" if stepped else None,
                            color=colour, alpha=DIST_FILL * opacity(label),
                            linewidth=0, zorder=z)
        draw = bx.step if stepped else bx.plot
        draw(xs, shares[label], color=colour, ls=style, zorder=z + 5,
             lw=VPA_DIST_LW if label in focus else lw, alpha=opacity(label),
             solid_capstyle="round", dash_capstyle="round",
             **(dict(where="mid") if stepped else {}))

    _vpa_furniture(ax, ylabel=VPA_Y_LABEL.format(stat=stat), xlabel=None, xs=xs)
    _vpa_furniture(bx, ylabel=VPA_D_LABEL, xlabel=X_LABEL, xs=xs)
    if short:
        # AFTER the furniture, never before: set_xscale throws away the axis's locators and
        # formatters, so the ticks have to be put back on the other side of it. The minor
        # locator is emptied because a scale may install one of its own and its marks would
        # appear on an axis whose labels are a listed set.
        #
        # The limits are worked out IN THE TRANSFORM and mapped back, which is the only way
        # a fraction of the span means the same thing at both ends of a bent axis.
        lo_x, hi_x = _vpa_pow(xs[0]), _vpa_pow(xs[-1])
        pad = (hi_x - lo_x) * VPA_SHORT_XPAD
        for a in (ax, bx):
            a.set_xscale("function", functions=(_vpa_pow, _vpa_root))
            a.xaxis.set_major_locator(
                matplotlib.ticker.FixedLocator(list(VPA_SHORT_XTICKS)))
            a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            # Plain counts: the axis is a number of atoms whichever way it is spaced, and a
            # scale's own formatter has no reason to know that.
            a.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            a.set_xlim(float(_vpa_root(lo_x - pad)), float(_vpa_root(hi_x + pad)))
    ax.margins(y=0.075)              # the default 5% puts ours on the bottom spine
    # Every 2 kcal/mol. The locator, left to itself over a ~9 kcal/mol span, picks 3 and
    # labels -3/-6/-9, which reads as a coarser axis than the 3-line figures' step of 2.
    ax.yaxis.set_major_locator(MultipleLocator(2))
    if short:
        ax.yaxis.set_major_locator(
            matplotlib.ticker.FixedLocator(list(VPA_SHORT_TICKS)))
    # Headroom over what is ON THIS PANEL, which is why it reads dist_series: the grey is
    # on the panel above only, and letting its rolled share set this scale would leave the
    # strip short by the height of a series that is not in it.
    bx.set_ylim(0, max(v for lab, *_ in dist_series for v in shares[lab]) * 1.13)
    bx.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    # The strip is ticked 0/4/8 % on request (2026-09-13) rather than the locator's 0/3/6.
    # It was v3's tick while the other two variants kept the locator; they are gone, so it
    # is simply the tick.
    bx.yaxis.set_major_locator(MultipleLocator(4))
    # "Generated fraction" is the name; the per-cent sign on the ticks is where the unit
    # is stated, so the label does not have to carry a parenthesis to stay honest.
    bx.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    # Each y label is otherwise placed against ITS OWN tick labels, so "Vina Dock median"
    # (widest tick "-10") and "Generated fraction" (widest tick "4%") sit at two different
    # x and the pair reads as a ragged left edge. align_ylabels puts both at the outer of
    # the two, which is the only way they line up without hard-coding a coordinate.
    fig.align_ylabels((ax, bx))
    legs = _vpa_legend(ax, series, ref_label, fig=fig, alpha_for=opacity)
    fit(fig, pad=0.5, h_pad=H_PAD)
    if short:
        # Take the height out of the VINA PANEL ONLY: the strip, the gap, the type and
        # every margin stay the size they are in the other eleven, so the two figures can
        # still be read side by side. That is why the figure is laid out at full height
        # first and shrunk after -- a smaller STACK_H up front is divided by the same
        # ratios and takes the strip down with it. fit() has by now put the margins in, so
        # the panel's height is known in inches; dropping the figure by the part of it
        # being given up, and the top ratio by the same factor, leaves the second fit()
        # with exactly the old strip and a panel VPA_SHORT_SCALE as tall.
        drop = (1 - VPA_SHORT_SCALE) * ax.get_position().height * STACK_H
        ax.get_gridspec().set_height_ratios(
            [HEIGHT_RATIOS[0] * VPA_SHORT_SCALE, HEIGHT_RATIOS[1]])
        fig.set_size_inches(FIG_W, STACK_H - drop)
        fit(fig, pad=0.5, h_pad=H_PAD)
    _vpa_clear_keys(fig, legs, (ax, bx))
    lo, hi = ax.get_ylim()
    save(fig, out, stem)

    drawn = [a for a, v in zip(xs, ref_values) if v is not None]
    where = (f"reference {drawn[0]}..{drawn[-1]} atoms" if cfg["ref_vina"]
             else "no reference")
    if not cfg["ref_dist"]:
        where += ", off the strip"
    span = f"x {xs[0]}..{xs[-1]}" + (f" ^{VPA_SHORT_XPOW:g}" if short else "")
    return f"    {stem:40s} {span:12s}   {where}   y {lo:.1f} .. {hi:.1f}"


def _vpa_grid_centre_xlabel(fig, axes):
    """Slide the one x name to the middle of the GRID rather than of its own cell.

    Run last, and only horizontally: the label's x is in ITS axes' coordinates, so a value
    past 1 walks it to the right across the other cells. Doing this before the layout is
    solved would have tight_layout see a label hanging off the side of its axes and reserve
    a right margin for it, paying for the centring in plot width."""
    labelled = [a for a in axes if a.get_xlabel()]
    if not labelled:
        return
    boxes = [a.get_position() for a in axes]
    centre = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    for a in labelled:
        box = a.get_position()
        a.xaxis.label.set_x((centre - box.x0) / box.width)


def _vpa_build_grid(out, cfg, stat, aset, S):
    """median / all again, as one cell per method. Returns its run-log line.

    Everything here is the single-panel figure's content re-laid-out; nothing is recomputed,
    so a value in a cell is the same value the one-panel figure and the CSV already carry."""
    stem = f"vina_dock_per_atom_{stat}_{aset}_grid"
    xs, curves, counts = S["xs"], S["curves"], S["counts"]
    cells, rows, cols = _vpa_grid_cells(S["arms"])
    ref_values = _vpa_gated(curves[(REF_LABEL, stat)], counts, cfg["ref_gate"])
    ref_firm, ref_thin = _vpa_split_firm(ref_values, counts)

    # ONE SCALE FOR ALL THE CELLS, worked out over everything that will be drawn in any of
    # them. Cells that autoscaled themselves would each be right and the figure would be
    # wrong: the whole point of the layout is that a curve's height means the same thing
    # from cell to cell.
    drawn = [v for lab, *_ in cells for v in curves[(lab, stat)] if v is not None]
    drawn += [v for v in ref_values if v is not None]
    pad = (max(drawn) - min(drawn)) * VPA_GRID_YPAD
    ylim = (min(drawn) - pad, max(drawn) + pad)
    ref_share = rolled(S["exact"][REF_LABEL], xs)
    top = max([v for lab, *_ in cells for v in S["exact"][lab]] + ref_share)

    # Two gridspec rows per cell row -- the Vina panel and its strip -- and nothing between
    # one cell row and the next.
    heights = list(HEIGHT_RATIOS) * rows
    fig = plt.figure(figsize=(VPA_GRID_CELL_W * cols, VPA_GRID_CELL_H * rows), dpi=220)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(len(heights), cols, height_ratios=heights)

    axes = []
    for i, (label, _, lw, style) in enumerate(cells):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r * 2, c])
        bx = fig.add_subplot(gs[r * 2 + 1, c], sharex=ax)
        for a in (ax, bx):
            a.set_facecolor("white")
        axes += [ax, bx]

        # Reference under, the cell's method over it -- the one-panel figure's order, so
        # where the two cross the same one is in front as it is there.
        for ys, alpha in ((ref_thin, VPA_REF_THIN_ALPHA), (ref_firm, 1.0)):
            ax.plot(xs, ys, color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=2, alpha=alpha,
                    dash_capstyle="round")
        ax.plot(xs, curves[(label, stat)], color=VPA_COLORS[label], lw=VPA_MODEL_LW,
                ls=style, zorder=3, solid_capstyle="round", dash_capstyle="round")

        # The strip is this method's own size distribution, filled. The one-panel figure
        # withholds the area from its context arms so that nine bands do not read as a
        # stack; one band per cell cannot stack with anything, so every cell gets its area.
        bx.fill_between(xs, S["exact"][label], step="mid", color=VPA_COLORS[label],
                        alpha=DIST_FILL, linewidth=0, zorder=3)
        # Rolled, so it is a curve and not a step: drawing a rolling average stepped would
        # claim the per-count resolution the window averaged away. Under the method's band
        # and over its fill, which is the order the panel above uses too.
        bx.plot(xs, ref_share, color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4,
                dash_capstyle="round")
        bx.step(xs, S["exact"][label], where="mid", color=VPA_COLORS[label], ls=style,
                lw=VPA_DIST_LW, zorder=8, solid_capstyle="round", dash_capstyle="round")

        # Names only on the outer edge: eight copies of "Vina Dock median" down the middle
        # of the figure would be furniture reading as data.
        _vpa_furniture(ax, ylabel=VPA_Y_LABEL.format(stat=stat) if c == 0 else None,
                       xlabel=None, xs=xs)
        # ONE x name for the grid, not one per column: at this cell width the name is
        # wider than a cell, so four of them ran into each other. It is set on the bottom
        # left cell -- an AXES label, so it stays inside that axes' tightbbox and the key
        # below still knows to clear it -- and slid to the middle of the grid after the
        # layout is solved, below, where moving it horizontally cannot cost plot area.
        _vpa_furniture(bx, ylabel=VPA_D_LABEL if c == 0 else None,
                       xlabel=X_LABEL if (r, c) == (rows - 1, 0) else None, xs=xs)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        bx.set_ylim(0, top * 1.13)
        bx.yaxis.set_major_locator(MultipleLocator(4))
        bx.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        # The Vina panel never labels x -- its own strip is directly under it and carries
        # the axis -- and only the left column labels y.
        ax.tick_params(labelbottom=False)
        if r < rows - 1:
            bx.tick_params(labelbottom=False)
        if c > 0:
            ax.tick_params(labelleft=False)
            bx.tick_params(labelleft=False)
        # The method's name: the cell's key, in one line of type, in the house INK.
        ax.set_title(display(label), fontsize=VPA_GRID_TITLE_FS, color=INK, pad=7)

    # NO KEY, asked for 2026-09-16. The only thing one could name is the reference, and a
    # one-row legend under a nine-cell grid spends a strip of the figure restating the
    # identity the whole section already carries -- grey, dashed, under every method. The
    # run log and the CSV say which reference population it is; the other twelve figures
    # name it in their own keys.
    fit(fig, pad=0.6, h_pad=H_PAD, w_pad=VPA_GRID_WPAD)
    fig.align_ylabels([a for a in axes if a.get_ylabel()])
    _vpa_grid_centre_xlabel(fig, axes)
    save(fig, out, stem)
    return (f"    {stem:40s} {rows} x {cols} cells, reading order   "
            + " · ".join(lab for lab, *_ in cells)
            + f"   reference in both panels of every cell   "
            f"y {ylim[0]:.1f} .. {ylim[1]:.1f}")


def _vpa_write_csv(out, sets, arms, data, ref, source, n_ref):
    """One table behind the main figures, opacity test and method grid. The grid re-lays-out
    numbers from this same table and adds none of its own.

    It is one table because almost nothing in it is per figure. An arm's count, mean,
    median and share at a heavy-atom count are properties of that arm, not of the picture
    it is drawn in, so the arm columns are evaluated once over the UNION of the two x spans
    -- which means a row may carry a value at a count only `core` plots, and `plotted_in`
    is the column that says so.

    Four columns carry what does differ. `plotted_in` names the arm sets whose x span
    contains the row. `reference_source` and `n_reference_total` name the POPULATION behind
    every reference column -- p79 or crossdocked, and how many ligands -- which is constant
    down the table and repeated anyway, because a flat CSV read apart from its figure has
    nowhere else to put it and a reference column whose population is unstated is the one
    thing this figure will not ship. `reference_firm` is yes wherever the rolling window
    holds at least VPA_REF_FIRM ligands, which was the whole of the variant difference on
    the Vina panel: v1 drew the reference exactly there, v2 and v3 drew beyond it and
    faded. And the reference's ROLLED share is the one number that genuinely depends on the
    span, because a moving average is truncated at its ends -- hence one column per arm
    set, empty where that set does not reach."""
    union = sorted(set().union(*(set(S["xs"]) for S in sets.values())))
    curves = {}
    for stat in VPA_STATS:
        curves[(REF_LABEL, stat)] = _vpa_reference_curve(ref, union, stat, VPA_REF_ANY)
        for label, *_ in arms:
            curves[(label, stat)] = _vpa_model_curve(data[label], union, stat)
    counts = _vpa_window_counts(ref, union)
    exact = {label: share(sizes, union)[0] for label, sizes
             in [(lab, data[lab]) for lab, *_ in arms] + [(REF_LABEL, ref)]}
    rolled_ref = {name: dict(zip(S["xs"], rolled(S["exact"][REF_LABEL], S["xs"])))
                  for name, S in sets.items()}
    spans = {name: set(S["xs"]) for name, S in sets.items()}

    header = ["heavy_atoms", "plotted_in"]
    for label, *_ in arms:
        k = _vpa_slug(label)
        header += [f"n_{k}", f"{k}_mean", f"{k}_median", f"{k}_pct"]
    header += ["reference_source", "n_reference_total", "n_reference_exact",
               "n_reference_window", "reference_firm", "reference_mean",
               "reference_median", "reference_pct_exact"]
    header += [f"reference_pct_rolled_{name}" for name in sets]
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    rows = []
    for i, a in enumerate(union):
        row = [a, "+".join(n for n, span in spans.items() if a in span)]
        for label, *_ in arms:
            row += [len(data[label].get(a, ())),
                    fmt(curves[(label, "mean")][i]),
                    fmt(curves[(label, "median")][i]),
                    f"{exact[label][i]:.3f}"]
        row += [source, n_ref, len(ref.get(a, ())), counts[i],
                "yes" if counts[i] >= VPA_REF_FIRM else "no",
                fmt(curves[(REF_LABEL, "mean")][i]),
                fmt(curves[(REF_LABEL, "median")][i]),
                f"{exact[REF_LABEL][i]:.3f}"]
        row += [fmt(rolled_ref[name].get(a)) for name in sets]
        rows.append(row)
    return write_csv(out, "vina_dock_per_atom", header, rows)


@figure("fig-vina-per-atom", needs=(VPA_BASENAME, "crossdocked_reference_dock.json"),
        folder="fig-vina/per-atomnum")
def draw_vina_per_atom(out):
    """Vina Dock against heavy-atom count, over a size-distribution strip — mean and median
    x core and all, no crystal reference, plus the per-method grid."""
    data, pockets, arms, missing = {}, {}, [], []
    for arm in VPA_ARMS:
        label, root = arm[0], arm[1]
        sizes, n_pockets = _vpa_load(root, label)
        if sizes is None:
            missing.append((label, os.path.join(root, VPA_BASENAME)))
            continue
        data[label], pockets[label] = sizes, n_pockets
        arms.append(arm)
    absent = [lab for lab in VPA_FOCUS if lab not in data]
    if absent:
        raise FileNotFoundError(
            f"the compared arms are the figure: {', '.join(absent)} has no {VPA_BASENAME}")
    source, ref, ref_label, ref_note, n_ref = _vpa_reference_set()

    use_style()
    print(f"  {VPA_PROTOCOL}")
    print("  arms: " + " · ".join(f"{lab} {pockets[lab]} pockets" for lab, *_ in arms))
    for label, path in missing:
        print(f"    {label} skipped: not docked here, no "
              f"{os.path.relpath(path, REPO)}")
    print(f"  reference: {ref_note}")
    print(f"    drawn as {ref_label!r}")
    for label, by in [(REF_LABEL, ref)] + [(lab, data[lab]) for lab, *_ in arms]:
        vals = [n for n, v in by.items() for _ in v]
        print(f"    size of {label:16s} mean {st.mean(vals):5.2f}  "
              f"median {st.median(vals):5.1f} heavy atoms")

    # The two ends of x are where EVERY arm IN THE SET clears MIN_N, so the sets do not
    # share a span. Inside them x stays contiguous and an arm that dips below is broken by
    # _vpa_model_curve() instead, so one thin patch in one arm no longer truncates the
    # figure for all of them.
    median_core = _vpa_resolve([a for a in arms if a[0] in VPA_CORE], data, ref)
    sets = {}
    for aset, subset in _vpa_arm_sets(arms):
        S = sets[aset] = _vpa_resolve(subset, data, ref)
        xs, curves = S["xs"], S["curves"]
        print(f"  {aset}: " + " + ".join(lab for lab, *_ in subset)
              + f" · x = {xs[0]}..{xs[-1]} heavy atoms, all drawn arms >= {MIN_N} "
              f"molecules at both ends; broken inside where an arm is not: "
              + (", ".join(f"{lab} at {v}" for lab, v in S["thin"].items() if v)
                 or "nowhere"))
        for label in [REF_LABEL] + [lab for lab, *_ in subset]:
            print(f"    {label:16s} n={S['totals'][label]:6d}  "
                  f"{sum(S['exact'][label]):.1f}% of its molecules inside the plotted "
                  f"x range")
        for stat in VPA_STATS:
            active = median_core if (stat, aset) == ("median", "core") else S
            active_curves = active["curves"]
            both = [i for i in range(len(active["xs"]))
                    if active_curves[(VPA_VOX_LABEL, stat)][i] is not None
                    and active_curves[(VPA_OUR_LABEL, stat)][i] is not None]
            wins = [i for i in both
                    if active_curves[(VPA_OUR_LABEL, stat)][i]
                    <= active_curves[(VPA_VOX_LABEL, stat)][i]]
            gap = st.mean([active_curves[(VPA_OUR_LABEL, stat)][i]
                           - active_curves[(VPA_VOX_LABEL, stat)][i] for i in both])
            print(f"    {stat:6s}: CoDE lower than VoxBind at {len(wins)}/{len(both)} "
                  f"sizes, mean per-size gap {gap:+.3f} kcal/mol")

    print(f"  median core: {' + '.join(VPA_CORE)} · "
          f"x = {median_core['xs'][0]}..{median_core['xs'][-1]} heavy atoms")

    cfg = VPA_REF
    # The banner says what the figure DOES with the reference, so with both switches off it
    # says that and stops: the gate and the distribution settings are still in the dict, but
    # printing them would read as a description of something on the page. The other branch
    # stays because one flag brings the grey back and it has to say so when it does.
    if not (cfg["ref_vina"] or cfg["ref_dist"]):
        print("  no reference on either panel, and none in the key")
    else:
        panels = ("Vina panel and strip" if cfg["ref_dist"] else "Vina panel only")
        tail = ("gated off below it" if cfg["ref_gate"] == "hard" else
                f"faded to alpha {VPA_REF_THIN_ALPHA} below it, down to a window of "
                f"{VPA_REF_GATE[cfg['ref_gate']]}")
        print(f"  reference on the {panels}, firm at >= {VPA_REF_FIRM} ligands "
              f"in its ±{REF_WIN}-atom window and {tail}; distribution "
              + ("rolled" if cfg["ref_dist_smooth"] else "per-count")
              + (", filled" if cfg["ref_dist_fill"] else ", unfilled"))
    for stat in VPA_STATS:
        for aset in sets:
            active = median_core if (stat, aset) == ("median", "core") else sets[aset]
            print(_vpa_build(out, cfg, stat, aset, active, ref_label))
            if (stat, aset) == ("median", "all"):
                print(_vpa_build(out, cfg, stat, aset, sets[aset], ref_label, test=True))
            if (stat, aset) == VPA_GRID:
                print(_vpa_build_grid(out, cfg, stat, aset, sets[aset]))
    _vpa_write_csv(out, {**sets, "median_core": median_core}, arms, data, ref,
                   source, n_ref)
    print("  vina_dock_per_atom.csv -- one table behind the main figures and opacity test")

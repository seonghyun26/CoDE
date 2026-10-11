"""fig-molweight: the 2 figures drawn into figures/fig-molweight/.

    python figures/draw.py molweight

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-molweight-distribution                   -> fig-molweight/
    fig-molweight-ecdf                           -> fig-molweight/
"""
from __future__ import annotations

import os
import time
import matplotlib.pyplot as plt
import numpy as np

from draw import (
    ARMS, CORE, DASH, DIST_FILL, DIST_LW, FIG_W, MODEL_LW, P79, PANEL_H, REF_COLOR,
    REF_LABEL, REF_LW, REF_ROOT, arm_handles, figure, fit, furniture, legend, pale,
    pose_data, save, use_style, write_csv,
)

# ════════════════════════════════════════════════════════════════════════════════
# fig-molweight-{distribution,ecdf} — the size prior every other figure is read against
# ════════════════════════════════════════════════════════════════════════════════
# NOT a quality metric -- nothing here says an arm is better -- it is the size prior every
# quality metric in 260910 is read against. Vina score grows with molecule size, strain
# grows with size, rigid-fragment RMSD grows with fragment size. So "which arm wins" is
# only meaningful once you know whether the arms are drawing from the same weight
# distribution, and they are not.
#
# MOLECULAR WEIGHT IS READ STRAIGHT FROM `target_*/samples.sdf`, through RDKit's
# `Descriptors.MolWt` (average mass, implicit hydrogens included).
#
# It used to come from the SMILES in each target's `metrics.json`, and that had to change
# to cover the five published baselines: their sample dirs are staged and complete, but
# their `metrics.json` files are written by the PoseBusters scoring run and appear as it
# finishes each pocket. Keying a size prior to the progress of a scoring run is the wrong
# dependency -- the molecules exist either way -- so this reads the molecules. The two
# agree: over the three arms that have both, SDF and SMILES give the same weight for every
# molecule (the load prints the check), because the recorded SMILES was derived from the
# same SDF entry.
#
# THE ARMS AND THE POCKET SET ARE THE SHARED ONES -- the 79 electron-density pockets. Arms
# hold different numbers of molecules (7,287 for TargetDiff against 7,888 for VoxBind)
# because that is what each sampled; nothing is filtered here. An arm is drawn only if it
# has a samples.sdf for all 79 pockets, which is the same rule arms_for() applies to
# metrics -- an arm mid-stage must not become a curve over part of the data.
MW_X_LABEL = "Molecular weight (Da)"
MW_BIN = 20                  # Da; ~52 bins over the occupied range
MW_XMAX = 800                # the plotted range; the share above it is named in the log
MW_XTICK = 100
# The crystal reference is 79 molecules over ~35 occupied bins -- about two per bin, so its
# raw histogram is a picket fence of 2.5%-tall spikes that says nothing. It is rolled over
# +-MW_REF_ROLL bins and AVERAGED, which keeps it on the same per-bin scale as the models'
# shares, exactly as size_distribution() rolls the reference over ligand size.
MW_REF_ROLL = 2
MW_EDGES = np.arange(0, MW_XMAX + MW_BIN, MW_BIN)
MW_CENTRES = MW_EDGES[:-1] + MW_BIN / 2


def _mw_sdf_weights(paths):
    """(molecular weights, unsanitisable entries, disconnected entries dropped).

    DISCONNECTED MOLECULES ARE DROPPED, because the task-2 evaluator drops them
    before scoring anything else -- so keeping them would put this figure on a different
    molecule set than the strain, clash and PoseBusters figures for the same run. It
    matters for exactly one arm: TargetDiff emits 511 of 7,798 (6.6%) over the 79 pockets
    and every other arm emits none. Their weight is also not a ligand's weight -- it is
    the sum of two or more pieces that never bonded.

    RDKit is imported HERE, not at the top of draw.py: it is the only figure-specific
    dependency in the file, and a top-level import would make every other figure -- and
    --list -- refuse to run on a box that has no rdkit."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    out, bad, disc = [], 0, 0
    for path in paths:
        for m in Chem.SDMolSupplier(path):
            if m is None:
                bad += 1
                continue
            if len(Chem.GetMolFrags(m)) > 1:
                disc += 1
                continue
            out.append(Descriptors.MolWt(m))
    return np.asarray(out), bad, disc


def _mw_samples(root, targets=None):
    """The samples.sdf under `root`, over `targets` or over every target it holds."""
    names = sorted(d for d in os.listdir(root) if d.startswith("target_")) \
        if targets is None else targets
    paths = [os.path.join(root, t, "samples.sdf") for t in names]
    return [p for p in paths if os.path.exists(p)]


def _mw_reference_paths():
    """The deposited ligand in each of the 79 pockets: the one SDF in the target dir that
    is not samples.sdf. Every arm that has one carries the same molecule, so this reads
    ours -- the staged baseline dirs hold no reference at all."""
    out = []
    for t in P79:
        d = os.path.join(REF_ROOT, t)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.endswith(".sdf") and f != "samples.sdf"]
    return out


def _mw_share(v):
    """Percent of a set's molecules in each bin. Molecules beyond MW_XMAX are NOT folded
    into the last bin -- that would put a spike where the data has a tail -- so the shares
    sum to slightly under 100 and the remainder is reported."""
    counts, _ = np.histogram(v, bins=MW_EDGES)
    return 100 * counts / len(v), counts


def _mw_rolled(pct):
    """The reference's shares under a centred +-MW_REF_ROLL-bin window, averaged."""
    k = 2 * MW_REF_ROLL + 1
    pad = np.pad(pct, MW_REF_ROLL, mode="constant")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _mw_block(v, bad=0, disc=0):
    """n, centre, spread and tails for one set -- what the load summary prints."""
    return {
        "n_molecules": int(len(v)), "n_sdf_unsanitisable": int(bad),
        "n_disconnected_dropped": int(disc),
        "mw_mean": round(float(v.mean()), 1),
        "mw_median": round(float(np.median(v)), 1),
        "mw_q25": round(float(np.percentile(v, 25)), 1),
        "mw_q75": round(float(np.percentile(v, 75)), 1),
        "mw_p1": round(float(np.percentile(v, 1)), 1),
        "mw_p99": round(float(np.percentile(v, 99)), 1),
        "mw_min": round(float(v.min()), 1), "mw_max": round(float(v.max()), 1),
        "pct_over_500": round(100 * float((v > 500).mean()), 2),
        f"pct_over_{MW_XMAX}": round(100 * float((v > MW_XMAX).mean()), 2),
    }


def _mw_smiles_crosscheck(arms, mw):
    """Molecules whose recorded SMILES gives a different weight than their SDF entry, over
    the arms that have metrics.json. Reported, never silently trusted: this builder changed
    source and the check is what says the change was free."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    _, p79_rows, _ = pose_data()
    out = {}
    for lab, key, root in arms:
        smi = [r["smi"] for r in p79_rows[key] if r.get("smi")]
        if not smi:
            continue
        sdf = mw[key]
        if len(smi) != len(sdf):
            out[lab] = (f"{len(smi)} SMILES vs {len(sdf)} SDF — not comparable; "
                        f"metrics.json is still being written for this arm")
            continue
        mism = sum(1 for a, b in zip(smi, sdf)
                   if abs(Descriptors.MolWt(Chem.MolFromSmiles(a)) - b) > 0.05)
        out[lab] = f"{mism} of {len(sdf)} differ"
    return out


_MW_CACHE = {}


def _mw_data():
    """(arms that have samples over all 79 pockets, {key: weights}, reference weights).

    Cached for the life of the process: the distribution and the ECDF are the same ~60k
    molecules read out of ~630 SDF files, and drawing both used to mean reading them twice.
    The per-arm summary and the SMILES cross-check print from HERE rather than from a draw
    function, so they appear once per run however many of the two figures are asked for.

    The original also built a weights-over-EVERY-pocket set; it fed only mw_summary.json,
    which the recompute script still owns, so it is not built here."""
    if "d" not in _MW_CACHE:
        t0 = time.time()
        # An arm is in only if it holds a samples.sdf for every one of the 79 pockets.
        arms = [a for a in ARMS if len(_mw_samples(a[2], P79)) == len(P79)]
        missing = [a[0] for a in ARMS if a not in arms]
        mw, bad, disc = {}, {}, {}
        for _, key, root in arms:
            mw[key], bad[key], disc[key] = _mw_sdf_weights(_mw_samples(root, P79))
        mw_ref, bad_ref, _ = _mw_sdf_weights(_mw_reference_paths())
        _MW_CACHE["d"] = (arms, mw, mw_ref)

        n = sum(len(v) for v in mw.values())
        print(f"  [read {n:,} molecules from {len(arms)} arms' samples.sdf "
              f"in {time.time()-t0:.1f}s]")
        print(f"  {len(P79)} pockets · molecular weight from target_*/samples.sdf, "
              f"single-component molecules only · {MW_BIN} Da bins to {MW_XMAX} Da\n")
        print(f"  {'arm':16s} {'mols':>6s} {'median':>8s} {'mean':>7s} {'IQR':>15s} "
              f"{'>500 Da':>8s} {f'>{MW_XMAX} Da':>9s} {'disconn.':>9s}")
        blocks = {lab: _mw_block(mw[key], bad[key], disc[key]) for lab, key, _ in arms}
        blocks[REF_LABEL] = _mw_block(mw_ref, bad_ref)
        for lab in [l for l, _, _ in arms] + [REF_LABEL]:
            b = blocks[lab]
            print(f"  {lab:16s} {b['n_molecules']:6d} {b['mw_median']:8.1f} "
                  f"{b['mw_mean']:7.1f} {b['mw_q25']:7.1f}–{b['mw_q75']:<7.1f} "
                  f"{b['pct_over_500']:7.2f}% {b[f'pct_over_{MW_XMAX}']:8.2f}% "
                  f"{b['n_disconnected_dropped']:9d}")
        if missing:
            print(f"\n  not drawn — no samples.sdf over all {len(P79)} pockets: "
                  + ", ".join(missing))
        print("\n  SDF vs recorded SMILES, molecules whose weight differs by >0.05 Da:")
        for lab, msg in _mw_smiles_crosscheck(arms, mw).items():
            print(f"    {lab:16s} {msg}")
    return _MW_CACHE["d"]


def _mw_variants(arms):
    """(name, arms) for each figure variant, core first -- variants() over the arms that
    have samples, which is a different list from the arms that have a given metric."""
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


@figure("fig-molweight-distribution", needs=("target_*/samples.sdf",))
def draw_molweight_distribution(out):
    """Share of an arm's molecules per 20 Da bin, core and all arms."""
    all_arms, mw, mw_ref = _mw_data()
    use_style()
    for variant, arms in _mw_variants(all_arms):
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Filled steps read well for the two or three arms of `core`; at nine they stack
        # into one opaque mass and the arm you are looking for is the one you cannot see.
        # So the fill is dropped once the figure carries more than the core arms, and the
        # step lines alone do the work -- the same reason the posecheck family draws its
        # eight-method figures as unfilled curves.
        fill = len(arms) <= len(CORE)
        for lab, key, _ in arms:
            pct, _ = _mw_share(mw[key])
            col = pale(lab)
            ax.step(MW_CENTRES, pct, where="mid", color=col, lw=DIST_LW + 0.35, zorder=3)
            if fill:
                ax.fill_between(MW_CENTRES, pct, step="mid", color=col, alpha=DIST_FILL,
                                lw=0, zorder=2)
        pct, _ = _mw_share(mw_ref)
        ax.plot(MW_CENTRES, _mw_rolled(pct), color=REF_COLOR, lw=REF_LW, ls=DASH, zorder=4,
                dash_capstyle="round")

        # "% of molecules" is per ARM -- each curve is normalised by its own total, which
        # is the only way a 79-molecule reference and a 7,888-molecule arm share an axis
        furniture(ax, ylabel=f"% of molecules\nper {MW_BIN} Da", xlabel=MW_X_LABEL,
                  xlim=(0, MW_XMAX), xloc=MW_XTICK)
        ax.set_ylim(bottom=0)
        legend(ax, arm_handles(arms), loc="upper right",
               ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)
        fit(fig, pad=0.5)
        save(fig, out, f"mw_distribution_{variant}")

    rows = []
    for lab, key, _ in all_arms:
        pct, counts = _mw_share(mw[key])
        rows += [[lab, lo, lo + MW_BIN, int(n), round(float(q), 3)]
                 for lo, n, q in zip(MW_EDGES[:-1], counts, pct)]
    pct, counts = _mw_share(mw_ref)
    rows += [[REF_LABEL, lo, lo + MW_BIN, int(n), round(float(q), 3)]
             for lo, n, q in zip(MW_EDGES[:-1], counts, pct)]
    write_csv(out, "mw_histogram", ["arm", "mw_bin_lo", "mw_bin_hi", "n", "pct"], rows)


@figure("fig-molweight-ecdf", needs=("target_*/samples.sdf",))
def draw_molweight_ecdf(out):
    """The distribution figure answers "where does this arm put its molecules"; this one
    answers "how much of an arm sits below any given weight", which is the form to read a
    shift between two arms off. No binning and no rolling -- the reference's 79 molecules
    are 79 honest steps."""
    all_arms, mw, mw_ref = _mw_data()
    use_style()
    for variant, arms in _mw_variants(all_arms):
        fig, ax = plt.subplots(figsize=(FIG_W, PANEL_H), dpi=220)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        def ecdf(v, col, lw, ls):
            x = np.sort(v)
            ax.plot(x, 100 * np.arange(1, len(x) + 1) / len(x), color=col, lw=lw, ls=ls,
                    zorder=4 if ls != "-" else 5, solid_capstyle="round")

        for lab, key, _ in arms:
            ecdf(mw[key], pale(lab), MODEL_LW, "-")
        ecdf(mw_ref, REF_COLOR, REF_LW, DASH)

        furniture(ax, ylabel="Cumulative share (%)", xlabel=MW_X_LABEL, xlim=(0, MW_XMAX),
                  xloc=MW_XTICK)
        ax.set_ylim(0, 100)
        legend(ax, arm_handles(arms), loc="lower right",
               ncol=2 if len(arms) > 4 else 1, fontsize=11.5 if len(arms) <= 4 else 10.0)
        fit(fig, pad=0.5)
        save(fig, out, f"mw_ecdf_{variant}")

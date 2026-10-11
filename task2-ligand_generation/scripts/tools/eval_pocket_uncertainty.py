#!/usr/bin/env python3
"""eval_pocket_uncertainty.py — per-pocket CRYSTALLOGRAPHIC UNCERTAINTY beside per-pocket Vina.

    conda run -n voxbind python task2-ligand_generation/scripts/tools/eval_pocket_uncertainty.py \
        --out results/task2-drugdesign/_shared/260916_pocket_uncertainty/pocket_uncertainty.json

THE QUESTION. CoDE and CoE condition on an X-ray electron-density crop of the pocket. If that
conditioning is what buys the gain, the gain should be LARGER where the map is sharp and the
pocket is well ordered, and smaller where it is smeared. B-factor is the deposited per-atom
measure of exactly that smearing, and occupancy is the deposited measure of how much of the
atom is actually there. So this writes, for each of the 79 density pockets, the B-factor and
occupancy of the pocket the model saw, next to every arm's Vina on that pocket — and the
drawing lives in figures/_parts/uncertainty.py, which does not recompute any of it.

WHERE EACH NUMBER COMES FROM, because three different files describe the same pocket:

  B-factor   the CrossDocked receptor `<PDBId>_<C>_rec.pdb` under targetdiff/data/test_set/.
             It preserves the deposited B-factors EXACTLY -- verified atom-for-atom against
             the deposited mmCIF (8,045 of 8,045 atoms of 4yhj, max |delta| 0.0000) -- so the
             receptor the docking used and the crystallography agree by construction.

  occupancy  the deposited mmCIF only. CrossDocked keeps one altloc per atom and rewrites
             every occupancy to 1.00, so the receptor PDB cannot answer this question at all;
             reading occupancy off it would silently return "1.0 everywhere". Atoms are keyed
             (chain, seqid, atom name) and the altlocs of one key are summed, so `occ` is how
             much of that atom the refinement actually placed and `altloc` is whether it was
             split between conformers. The two are different kinds of disorder and are kept
             apart: a 0.5/0.5 side chain sums to 1.0 and is NOT partial occupancy.

  map        xray_crops_aligned/test_transforms.npz -- the mean density at the corrected
             receptor atoms, which is the gate that picked these 79 pockets out of 100. It is
             the direct measurement of the conditioning input, where B-factor and resolution
             are proxies for it, so it belongs in the same table as the control.

TWO SCOPES, because "the pocket" is not one thing. `pocket` is the 10 A crop the sampler was
conditioned on (the `*_pocket10.pdb` beside the samples); `site` is the receptor atoms within
--site-radius (5 A default) of the crystal ligand, i.e. the wall the ligand actually touches.
The crop is ~3x larger and averages in loops that never see a ligand, so the two can disagree;
both are written and the figures say which one they draw.

RAW AND NORMALIZED B, because raw B is not comparable between entries. A 2.6 A structure is
refined to higher B's than a 1.5 A one everywhere, so a raw-B correlation across 79 entries is
partly a resolution correlation. `b_*_z` is the site/pocket mean in units of that receptor's
own B distribution ((b_site - b_receptor_mean) / b_receptor_std), which asks the within-structure
question -- is THIS pocket unusually floppy for THIS crystal -- and cannot be explained by
resolution. Both are written; a claim that holds only on the raw column is a resolution claim.

Arms are read from the results bundle's per-molecule Vina CSVs, which is the one table every
method shares, restricted to `in_density79` so every arm stands on the same pockets. The
baseline for a "gain" is VoxBind (the published sigma=0.9 reproduction); the delta is computed
per pocket, for the pocket MEAN and the pocket MEDIAN, and for ligand efficiency (kcal/mol per
heavy atom) as well -- 80% of the arm ordering on raw Vina is a molecule-size effect, so a
correlation that survives only on raw Dock is a size correlation and the LE column is what
says so.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics as st
import sys

import gemmi
import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BUNDLE = os.path.join(REPO, "results", "task2-drugdesign")
TESTSET = os.path.join(REPO, "targetdiff", "data", "test_set", "test_set")
CIF_DIR = os.path.join(REPO, "dataset", "data", "cif")
TRANSFORMS = os.path.join(REPO, "dataset", "data",
                          "xray_crops_aligned", "test_transforms.npz")

# (label, bundle folder). The labels are figures/_parts/00_core.py's, so a label changed
# there must change here. VoxBind is the baseline every delta is taken against; the two
# density-conditioned arms are the question; the rest are the CONTROL -- if B-factor predicts
# a gain for arms that never see a density map, the effect is pocket difficulty, not
# conditioning, and only these rows can tell the difference.
ARMS = [
    ("VoxBind",    "VoxBind-vanilla"),
    ("CoDE",       "VoxBind-Ours"),
    ("CoE",        "VoxBind+CoE"),
    ("TargetDiff", "TargetDiff"),
    ("AR",         "AR"),
    ("Pocket2Mol", "Pocket2Mol"),
    ("DiffSBDD",   "DiffSBDD"),
    ("DecompDiff", "DecompDiff"),
    ("FuncBind",   "FuncBind"),
]
BASELINE = "VoxBind"
PER_MOLECULE = os.path.join("eval", "vina_docking", "per_molecule.csv")
METRICS = ("vina_score", "vina_min", "vina_dock")

# The run that carries the crystal ligand's own docked value beside each target. The bundle
# CSVs are generated molecules only, so pocket difficulty -- which is the covariate every
# correlation here has to be read against -- still comes from a run tree.
REF_RUNS = [
    "task2-ligand_generation/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
    "task2-ligand_generation/exps/_vanilla_ep923/samples/full_eval_ep923",
]
REF_BASENAME = "eval_docking_results_full79.json"

# Contact weighting (see pocket_descriptors). CONTACT_CUT is the usual 4.5 A heavy-atom
# contact distance; CONTACT_SIGMA makes the Gaussian shell about one vdW contact thick;
# CONTACT_FLOOR stops 1/d^2 exploding where a crystal contact is unusually short.
CONTACT_CUT, CONTACT_SIGMA, CONTACT_FLOOR = 4.5, 1.5, 2.0


# ──────────────────────────────────────────────────────────────────────────────
# the pockets, and the three files that describe each one
# ──────────────────────────────────────────────────────────────────────────────
def pocket_index(rows_source):
    """{target: {"pocket_index", "subdir", "ligand", "receptor", "pdb_id", "chain"}}.

    Keyed off a bundle CSV rather than a split file because `in_density79` is the bundle's
    own definition of the 79, and every arm is filtered by the same flag."""
    out = {}
    with open(rows_source, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("in_density79") != "1" or r["target"] in out:
                continue
            subdir, base = r["ligand_filename"].split("/", 1)
            receptor = base[:10]                       # PDBId_C_rec, as VinaDockingTask reads it
            pdb_id, chain, _ = receptor.split("_")
            out[r["target"]] = {
                "pocket_index": int(r["pocket_index"]),
                "subdir": subdir,
                "ligand": base,
                "receptor": receptor,
                "pdb_id": pdb_id.lower(),
                "chain": chain,
                # CrossDocked cross-docks: the crystal ligand can come from a DIFFERENT entry
                # than the receptor, in which case nothing about the ligand's own deposited
                # B or occupancy describes this complex. The receptor pocket still does.
                "cross_docked": base[11:].split("_")[0].lower() != pdb_id.lower(),
            }
    return out


def ligand_coords(path):
    """Heavy-atom coordinates of an SDF, parsed from the V2000 block.

    Parsed by hand rather than through RDKit because this needs positions and nothing else,
    and a sanitization failure on a crystal ligand must not be able to drop a pocket."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    n_atoms = int(lines[3][0:3])
    xyz = []
    for line in lines[4:4 + n_atoms]:
        if line[31:34].strip() == "H":
            continue
        xyz.append((float(line[0:10]), float(line[10:20]), float(line[20:30])))
    return np.asarray(xyz, dtype=np.float64)


def receptor_atoms(path):
    """[(chain, seqid, atom name, b_iso, x, y, z)] for one receptor PDB, hydrogens dropped."""
    out = []
    structure = gemmi.read_structure(path)
    structure.remove_hydrogens()
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                out.append((chain.name, residue.seqid.num, atom.name, atom.b_iso,
                            atom.pos.x, atom.pos.y, atom.pos.z))
    return out


def crop_keys(path):
    """{(chain, seqid, atom name)} of a `*_pocket10.pdb` -- the crop the sampler conditioned on.

    A SET of keys rather than the crop's own B column: the crop and the receptor carry the
    same values (it is a literal subset, matched at 0.0000 A), and keying it means the crop,
    the receptor and the deposited entry are all indexed the same way."""
    keys = set()
    structure = gemmi.read_structure(path)
    structure.remove_hydrogens()
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                keys.add((chain.name, residue.seqid.num, atom.name))
    return keys


def deposited_occupancy(cif_path):
    """{(chain, seqid, atom name): (occ_sum, n_altloc)} from the deposited mmCIF.

    Altlocs of one atom are SUMMED, so `occ_sum` is how much of the atom the refinement
    placed and `n_altloc` is how many conformers it was split between. Conflating the two
    would call every 0.5/0.5 side chain "half occupied" when it is fully occupied and
    disordered -- a different statement, and the one B-factor already partly makes."""
    structure = gemmi.read_structure(cif_path)
    structure.remove_hydrogens()
    per_key = collections.defaultdict(lambda: [0.0, 0])
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                slot = per_key[(chain.name, residue.seqid.num, atom.name)]
                slot[0] += atom.occ
                slot[1] += 1
    return {k: (v[0], v[1]) for k, v in per_key.items()}


def deposited_refinement(cif_path):
    """(resolution, r_work, r_free) from the deposited mmCIF; any of them may be None."""
    structure = gemmi.read_structure(cif_path)
    resolution = structure.resolution or None
    r_work = r_free = None
    for ref in structure.meta.refinement:
        if r_work is None and ref.r_work == ref.r_work and ref.r_work > 0:
            r_work = ref.r_work
        if r_free is None and ref.r_free == ref.r_free and ref.r_free > 0:
            r_free = ref.r_free
    return resolution, r_work, r_free


# ──────────────────────────────────────────────────────────────────────────────
# the descriptors
# ──────────────────────────────────────────────────────────────────────────────
def _stats(values):
    if not values:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": st.mean(values), "median": st.median(values), "n": len(values)}


def pocket_descriptors(meta, site_radius):
    """Every crystallographic descriptor of one pocket, or None if a file is missing.

    Two scopes: `pocket` = the 10 A crop the sampler saw, `site` = receptor atoms within
    `site_radius` of the crystal ligand. `receptor` is the whole chain set, and exists only
    to normalize the other two -- raw B is not comparable between entries."""
    rec_path = os.path.join(TESTSET, meta["subdir"], meta["receptor"] + ".pdb")
    lig_path = os.path.join(TESTSET, meta["subdir"], meta["ligand"])
    cif_path = os.path.join(CIF_DIR, meta["pdb_id"] + ".cif")
    for path in (rec_path, lig_path):
        if not os.path.exists(path):
            return None, f"missing {os.path.relpath(path, REPO)}"
    crops = glob.glob(os.path.join(BUNDLE, "*", "samples", meta["target"], "*_pocket10.pdb"))
    if not crops:
        return None, "no *_pocket10.pdb in any bundle arm"

    atoms = receptor_atoms(rec_path)
    xyz = np.asarray([a[4:7] for a in atoms], dtype=np.float64)
    b_all = np.asarray([a[3] for a in atoms], dtype=np.float64)
    keys = [a[:3] for a in atoms]

    lig = ligand_coords(lig_path)
    dist = np.sqrt(((xyz[:, None, :] - lig[None, :, :]) ** 2).sum(-1))   # receptor x ligand
    dmin = dist.min(1)
    in_site = dmin <= site_radius

    pocket_keys = crop_keys(sorted(crops)[0])
    in_pocket = np.asarray([k in pocket_keys for k in keys])

    row = {
        "pdb_id": meta["pdb_id"],
        "chain": meta["chain"],
        "receptor": meta["receptor"],
        "cross_docked": meta["cross_docked"],
        "n_receptor_atoms": int(len(atoms)),
        "n_pocket_atoms": int(in_pocket.sum()),
        "n_site_atoms": int(in_site.sum()),
        "n_ligand_heavy_atoms": int(len(lig)),
        "site_radius": site_radius,
    }

    # ── B-factor, raw and normalized. The z columns are the within-structure question and
    # are the ones a resolution confound cannot explain.
    mu, sigma = float(b_all.mean()), float(b_all.std(ddof=1))
    row["b_receptor_mean"] = mu
    row["b_receptor_median"] = float(np.median(b_all))
    row["b_receptor_std"] = sigma
    for scope, mask in (("pocket", in_pocket), ("site", in_site)):
        sel = b_all[mask]
        row[f"b_{scope}_mean"] = float(sel.mean()) if sel.size else None
        row[f"b_{scope}_median"] = float(np.median(sel)) if sel.size else None
        row[f"b_{scope}_z"] = float((sel.mean() - mu) / sigma) if sel.size and sigma > 0 else None

    # ── CONTACT-WEIGHTED B, which is the quantity the question is really about.
    # `b_site_mean` gives an atom at 4.9 A the same vote as one that is actually touching the
    # ligand at 3.2 A, so it is as much a measure of "what happened to sit inside a sphere"
    # as of the interface. These weight each receptor atom by how much ligand it faces:
    #
    #   invd2   w = sum over ligand atoms of 1/d^2, floored at CONTACT_FLOOR so a near-zero
    #           distance cannot take over the average. Long tail, no cutoff to choose.
    #   gauss   w = sum of exp(-d^2 / 2*sigma^2) with sigma = CONTACT_SIGMA. A soft shell one
    #           van der Waals contact thick; the cutoff is smooth, so no atom flips in or out
    #           of the average because a pose moved 0.1 A.
    #   count   w = number of ligand atoms within CONTACT_CUT. The crudest and the most
    #           legible -- it is the contact count every medicinal chemist already reads.
    #
    # All three are written because they disagree where the site is shallow, and a claim that
    # survives only one weighting is a claim about that weighting.
    inv = 1.0 / np.maximum(dist, CONTACT_FLOOR) ** 2
    weights = {
        "invd2": inv.sum(1),
        "gauss": np.exp(-(dist ** 2) / (2.0 * CONTACT_SIGMA ** 2)).sum(1),
        "count": (dist <= CONTACT_CUT).sum(1).astype(np.float64),
    }
    for name, w in weights.items():
        total = w.sum()
        if total <= 0:
            row[f"b_contact_{name}"] = row[f"b_contact_{name}_z"] = None
            continue
        value = float((w * b_all).sum() / total)
        row[f"b_contact_{name}"] = value
        row[f"b_contact_{name}_z"] = float((value - mu) / sigma) if sigma > 0 else None
    row["n_contact_atoms"] = int((dmin <= CONTACT_CUT).sum())
    # The unweighted site mean over the SAME atoms the contact weights see, so the effect of
    # the weighting can be separated from the effect of the tighter shell.
    tight = dmin <= CONTACT_CUT
    row["b_contact_shell_mean"] = float(b_all[tight].mean()) if tight.any() else None

    # ── occupancy, deposited only. `matched` is printed per pocket: a low match rate means
    # the deposited chain/numbering does not line up with CrossDocked's and the occupancy
    # columns for that pocket describe the wrong atoms, so it is recorded, not assumed.
    if os.path.exists(cif_path):
        occ = deposited_occupancy(cif_path)
        resolution, r_work, r_free = deposited_refinement(cif_path)
        row["resolution"] = resolution
        row["r_work"] = r_work
        row["r_free"] = r_free
        matched = sum(1 for k in keys if k in occ)
        row["occ_match_frac"] = matched / len(keys) if keys else 0.0
        for scope, mask in (("pocket", in_pocket), ("site", in_site)):
            vals = [occ[k] for k, m in zip(keys, mask) if m and k in occ]
            sums = [v[0] for v in vals]
            row[f"occ_{scope}_matched"] = len(vals)
            row[f"occ_{scope}_mean"] = float(np.mean(sums)) if sums else None
            row[f"occ_{scope}_min"] = float(min(sums)) if sums else None
            row[f"occ_{scope}_frac_partial"] = (
                float(np.mean([s < 0.99 for s in sums])) if sums else None)
            row[f"occ_{scope}_frac_altloc"] = (
                float(np.mean([v[1] > 1 for v in vals])) if vals else None)
    else:
        row["occ_match_frac"] = None
        row["resolution"] = row["r_work"] = row["r_free"] = None
        for scope in ("pocket", "site"):
            for suffix in ("matched", "mean", "min", "frac_partial", "frac_altloc"):
                row[f"occ_{scope}_{suffix}"] = None
    return row, None


# ──────────────────────────────────────────────────────────────────────────────
# the arms
# ──────────────────────────────────────────────────────────────────────────────
def arm_per_pocket(folder, targets):
    """{target: {metric_mean, metric_median, le_*, n_mols, atoms_mean}} for one bundle arm.

    Ligand efficiency is per MOLECULE and then aggregated, not the pocket mean divided by
    the pocket mean size: the two differ, and only the first is a ligand efficiency."""
    path = os.path.join(BUNDLE, folder, PER_MOLECULE)
    if not os.path.exists(path):
        return None
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    sizes = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("in_density79") != "1" or r["target"] not in targets:
                continue
            try:
                n_atoms = int(r["n_atoms"])
            except (KeyError, ValueError):
                continue
            sizes[r["target"]].append(n_atoms)
            for m in METRICS:
                v = r.get(m)
                if v in (None, "", "None"):
                    continue
                per[r["target"]][m].append(float(v))
                if n_atoms:
                    per[r["target"]][m + "_le"].append(float(v) / n_atoms)
    out = {}
    for target, metrics in per.items():
        row = {"n_mols": len(sizes[target]),
               "atoms_mean": st.mean(sizes[target]) if sizes[target] else None,
               "atoms_median": st.median(sizes[target]) if sizes[target] else None}
        for key, values in metrics.items():
            row[key + "_mean"] = st.mean(values)
            row[key + "_median"] = st.median(values)
        out[target] = row
    return out


def reference_dock(targets):
    """{target: {"ref_vina_score"/"_min"/"_dock"}} -- the crystal ligand's own docked values.

    This is the pocket-difficulty covariate. Every correlation below has to be read against
    it: if a gain tracks B-factor only because hard pockets are also floppy pockets, this is
    the column that shows it."""
    for rel in REF_RUNS:
        path = os.path.join(REPO, rel, REF_BASENAME)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            per_target = json.load(fh)["per_target"]
        out = {}
        for t in per_target:
            if t["target"] not in targets:
                continue
            out[t["target"]] = {k: t.get(k) for k in
                                ("ref_vina_score", "ref_vina_min", "ref_vina_dock")}
        if out:
            return out, rel
    return {}, None


def map_quality(index_of):
    """{target: {"map_density", "map_rmsd", "map_n_match"}} from the crop builder's own record.

    The density gate that picked these 79 pockets. It is the DIRECT measurement of the
    conditioning input where B-factor and resolution are proxies, so it is the control on
    whether a B-factor correlation is really a map-quality correlation."""
    if not os.path.exists(TRANSFORMS):
        return {}
    npz = np.load(TRANSFORMS)
    return {t: {"map_density": float(npz["density"][i]),
                "map_rmsd": float(npz["rmsd"][i]),
                "map_n_match": int(npz["n_match"][i])}
            for t, i in index_of.items() if i < len(npz["density"])}


# ──────────────────────────────────────────────────────────────────────────────
# the correlations — computed HERE, not in the figure
# ──────────────────────────────────────────────────────────────────────────────
# figures/draw.py draws; it does not re-score, and a correlation coefficient is a result,
# not a drawing decision. Computing it here also keeps scipy out of draw.py, which has to
# run on a box that has numpy and matplotlib and nothing else.
#
# The descriptors a figure may draw, in the order the correlation matrix reads. `label` is
# the axis label, `short` the matrix row -- both live here so the figure never re-words a
# quantity whose definition is in this file.
DESCRIPTORS = [
    ("b_site_mean",   "Site B-factor (Å²)",        "B site"),
    ("b_pocket_mean", "Pocket B-factor (Å²)",      "B pocket"),
    ("b_site_z",      "Site B-factor (z, within structure)",   "B site z"),
    ("b_pocket_z",    "Pocket B-factor (z, within structure)", "B pocket z"),
    ("b_contact_invd2", "Contact-weighted B-factor, 1/d² (Å²)",  "B contact 1/d²"),
    ("b_contact_gauss", "Contact-weighted B-factor, Gaussian (Å²)", "B contact gauss"),
    ("b_contact_count", "Contact-weighted B-factor, contact count (Å²)", "B contact count"),
    ("b_contact_invd2_z", "Contact-weighted B-factor (z, within structure)", "B contact z"),
    ("b_contact_shell_mean", "B-factor of atoms within 4.5 Å (Å²)", "B 4.5 Å shell"),
    ("b_receptor_mean", "Receptor B-factor (Å²)",  "B receptor"),
    ("resolution",    "Resolution (Å)",            "Resolution"),
    ("r_free",        "R$_{free}$",                "R-free"),
    ("occ_site_mean", "Site mean occupancy",       "Occ site"),
    ("occ_site_min",  "Lowest site occupancy",     "Occ site min"),
    ("occ_site_frac_partial",   "Site atoms at occupancy < 1",  "Occ<1 site"),
    ("occ_site_frac_altloc",    "Site atoms with an alternate conformer", "Altloc site"),
    ("occ_pocket_frac_altloc",  "Pocket atoms with an alternate conformer", "Altloc pocket"),
    ("map_density",   "Density at receptor atoms (σ)", "Map σ"),
    ("n_site_atoms",  "Receptor atoms in the site", "Site size"),
    ("ref_vina_dock", "Crystal ligand Vina Dock (kcal/mol)", "Pocket difficulty"),
]
# The outcome columns. Vina is negative-is-better, so a NEGATIVE delta is a gain.
OUTCOMES = [
    ("vina_dock_median",    "Vina Dock (pocket median)"),
    ("vina_dock_mean",      "Vina Dock (pocket mean)"),
    ("vina_score_median",   "Vina Score (pocket median)"),
    ("vina_min_median",     "Vina Min (pocket median)"),
    ("vina_dock_le_median", "Vina Dock per heavy atom (pocket median)"),
]
# What every correlation is read against. Pocket difficulty is the confound that matters:
# if an arm gains most where the crystal ligand docks worst, that is regression toward the
# mean and has nothing to do with the crystal being well ordered.
CONTROL = "ref_vina_dock"


def _rank(values):
    """Average ranks, so ties do not invent an order. Spearman is Pearson on these."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    for start, count in zip(first, counts):
        if count > 1:
            ranks[order[start:start + count]] = ranks[order[start:start + count]].mean()
    return ranks


def _partial(x, y, z):
    """Spearman of x and y with z regressed out of both, on ranks. None if z is constant."""
    xr, yr, zr = _rank(x), _rank(y), _rank(z)
    design = np.column_stack([zr, np.ones_like(zr)])
    if np.ptp(zr) == 0:
        return None
    rx = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    ry = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _slope_ci(x, y, n_boot=2000, seed=0):
    """(slope, lo, hi, span) — OLS slope of y on x with a bootstrap CI over pockets.

    A NULL IS ONLY A RESULT IF IT IS BOUNDED. "rho = 0.02, q = 0.90" says the data does not
    show an effect; it does not say the effect is small, and with 79 pockets a reader is
    right to ask whether a real 1 kcal/mol trend could be hiding in there. The slope CI
    answers that in the unit the question was asked in: multiplied by the observed range of
    the descriptor it is the largest end-to-end effect the data still allows.

    Simple linear regression is cov(x, y) / var(x), so every resample is computed at once
    rather than by re-fitting 2,000 times. Resamples where x collapses to one value are
    dropped -- with occupancy, where 70 of 79 pockets are tied at zero, they are common and
    a least-squares fit on them does not exist."""
    if len(x) < 8 or np.ptp(x) == 0:
        return None, None, None, None
    slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    bx, by = x[idx], y[idx]
    var = bx.var(axis=1)
    ok = var > 0
    if ok.sum() < n_boot // 10:
        return slope, None, None, float(np.ptp(x))
    cov = ((bx - bx.mean(axis=1, keepdims=True)) *
           (by - by.mean(axis=1, keepdims=True))).mean(axis=1)
    slopes = cov[ok] / var[ok]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return slope, float(lo), float(hi), float(np.ptp(x))


def _bh(pvalues):
    """Benjamini-Hochberg q-values.

    There are 15 descriptors x 5 outcomes x 8 arms here. At that many tests one p below
    0.05 is the EXPECTED number under a pure null, so a raw p is not evidence and the
    figures label the q, not the p."""
    idx = [i for i, p in enumerate(pvalues) if p == p]
    ordered = sorted(idx, key=lambda i: pvalues[i])
    n, out, running = len(ordered), [None] * len(pvalues), 1.0
    for rank, i in reversed(list(enumerate(ordered, 1))):
        running = min(running, pvalues[i] * n / rank)
        out[i] = running
    return out


def correlate(pockets, arms, deltas, baseline):
    """{"deltas"/"absolute": {arm: {outcome: [ {descriptor, spearman, p, q, ...} ]}}}.

    Both halves are written because only their DIFFERENCE answers the question. A descriptor
    that predicts every arm's absolute Vina equally is describing the pocket, and it cancels
    in the delta; one that predicts the delta is describing the arm. Reading either alone
    turns a property of CrossDocked into a property of the model."""
    targets = sorted(pockets)
    def column(source, key):
        return np.asarray([(source.get(t, {}).get(key)
                            if source.get(t, {}).get(key) is not None else np.nan)
                           for t in targets], dtype=np.float64)

    control = column(pockets, CONTROL)
    out = {"deltas": {}, "absolute": {}, "descriptors": DESCRIPTORS, "outcomes": OUTCOMES,
           "control": CONTROL, "targets": targets}
    for half, source in (("deltas", deltas), ("absolute", arms)):
        for arm, rows in source.items():
            per_arm = {}
            for outcome, _ in OUTCOMES:
                y = column(rows, outcome)
                entries = []
                for key, _, _ in DESCRIPTORS:
                    x = column(pockets, key)
                    keep = ~(np.isnan(x) | np.isnan(y))
                    n = int(keep.sum())
                    entry = {"descriptor": key, "n": n}
                    if n >= 8 and np.ptp(x[keep]) > 0 and np.ptp(y[keep]) > 0:
                        rho, p = stats.spearmanr(x[keep], y[keep])
                        r, pr = stats.pearsonr(x[keep], y[keep])
                        ctl = control[keep]
                        entry.update(spearman=float(rho), p=float(p),
                                     pearson=float(r), pearson_p=float(pr),
                                     # How many pockets actually carry a value that is not
                                     # the modal one. Nine of 79 pockets have an altloc in
                                     # the site, so an occupancy rho is nine points moving
                                     # against seventy tied ones, and this is the column
                                     # that says so before anyone reads the rho.
                                     n_informative=int(np.sum(
                                         x[keep] != stats.mode(x[keep], keepdims=False).mode)),
                                     partial_spearman=(None if key == CONTROL else
                                                       _partial(x[keep], y[keep], ctl)))
                        slope, lo, hi, span = _slope_ci(x[keep], y[keep])
                        entry.update(slope=slope, slope_lo=lo, slope_hi=hi, span=span,
                                     # The same three numbers in the unit the question was
                                     # asked in: what the descriptor is worth end to end.
                                     effect=None if slope is None else slope * span,
                                     effect_lo=None if lo is None else lo * span,
                                     effect_hi=None if hi is None else hi * span)
                    else:
                        entry.update(spearman=None, p=None, pearson=None, pearson_p=None,
                                     n_informative=None, partial_spearman=None,
                                     slope=None, slope_lo=None, slope_hi=None, span=None,
                                     effect=None, effect_lo=None, effect_hi=None)
                    entries.append(entry)
                qs = _bh([e["p"] if e["p"] is not None else float("nan") for e in entries])
                for entry, q in zip(entries, qs):
                    entry["q"] = q
                per_arm[outcome] = entries
            out[half][arm] = per_arm
    out["absolute"].setdefault(baseline, out["absolute"].get(baseline, {}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(
        BUNDLE, "_shared", "260916_pocket_uncertainty", "pocket_uncertainty.json"))
    ap.add_argument("--site-radius", type=float, default=5.0,
                    help="A around the crystal ligand that counts as the binding site (5.0)")
    ap.add_argument("--baseline", default=BASELINE,
                    help="the arm every delta is taken against (VoxBind)")
    args = ap.parse_args()

    source = os.path.join(BUNDLE, "VoxBind-Ours", PER_MOLECULE)
    if not os.path.exists(source):
        sys.exit(f"no {source} — the bundle has no per-molecule Vina to take the 79 from")
    targets = pocket_index(source)
    print(f"  {len(targets)} density-79 pockets from {os.path.relpath(source, REPO)}")

    pockets, failures = {}, []
    for target, meta in sorted(targets.items()):
        meta = dict(meta, target=target)
        row, why = pocket_descriptors(meta, args.site_radius)
        if row is None:
            failures.append((target, why))
            continue
        row["pocket_index"] = meta["pocket_index"]
        pockets[target] = row
    print(f"  descriptors for {len(pockets)} pockets"
          + (f"; {len(failures)} failed: {failures[:3]}" if failures else ""))

    poor = [(t, r["occ_match_frac"]) for t, r in pockets.items()
            if r["occ_match_frac"] is not None and r["occ_match_frac"] < 0.95]
    print(f"  deposited-entry atom match: median "
          f"{st.median([r['occ_match_frac'] for r in pockets.values() if r['occ_match_frac'] is not None]):.4f}"
          f", {len(poor)} pocket(s) under 0.95"
          + (f" -> {poor[:5]}" if poor else ""))

    for target, extra in map_quality({t: m["pocket_index"] for t, m in targets.items()}).items():
        if target in pockets:
            pockets[target].update(extra)
    refs, ref_run = reference_dock(targets)
    for target, extra in refs.items():
        if target in pockets:
            pockets[target].update(extra)
    print(f"  reference ligand Vina from {ref_run or 'nowhere — no full79 run found'}")

    arms, missing = {}, []
    for label, folder in ARMS:
        rows = arm_per_pocket(folder, targets)
        if rows is None:
            missing.append(label)
            continue
        arms[label] = rows
        print(f"  {label:<11} {len(rows)} pockets, "
              f"{sum(r['n_mols'] for r in rows.values())} molecules")
    if missing:
        print(f"  no per-molecule CSV: {', '.join(missing)}")
    if args.baseline not in arms:
        sys.exit(f"baseline arm {args.baseline!r} has no per-molecule CSV — no delta to take")

    # The deltas. Written out per pocket so the figures never re-derive them, and NEGATIVE
    # is better on every Vina column -- so a "gain" is a NEGATIVE delta, and the figures say
    # so on the axis rather than flipping the sign here.
    base = arms[args.baseline]
    deltas = {}
    for label, rows in arms.items():
        if label == args.baseline:
            continue
        per = {}
        for target, row in rows.items():
            if target not in base:
                continue
            d = {}
            for key, value in row.items():
                if key in ("n_mols", "atoms_mean", "atoms_median"):
                    continue
                other = base[target].get(key)
                if value is not None and other is not None:
                    d[key] = value - other
            d["atoms_mean_delta"] = (row["atoms_mean"] - base[target]["atoms_mean"]
                                     if row["atoms_mean"] and base[target]["atoms_mean"] else None)
            per[target] = d
        deltas[label] = per

    stats_block = correlate(pockets, arms, deltas, args.baseline)
    headline = stats_block["deltas"].get("CoDE", {}).get("vina_dock_median", [])
    for entry in headline:
        if entry["descriptor"] in ("b_site_mean", "b_site_z", "occ_site_frac_altloc"):
            bound = ("" if entry.get("effect_lo") is None else
                     f"  effect over its whole range {entry['effect']:+.2f} "
                     f"[{entry['effect_lo']:+.2f}, {entry['effect_hi']:+.2f}] kcal/mol")
            print(f"  CoDE ΔDock vs {entry['descriptor']:<22} rho={entry['spearman']:+.3f} "
                  f"p={entry['p']:.3f} q={entry['q']:.3f} "
                  f"(informative pockets {entry['n_informative']}/{entry['n']}){bound}")

    out = {
        "protocol": {
            "pockets": "density79 (in_density79 of the bundle's per-molecule Vina CSVs)",
            "b_factor": "deposited, read off the CrossDocked receptor (verified identical)",
            "occupancy": "deposited mmCIF only; altlocs of one atom summed",
            "site_radius": args.site_radius,
            "baseline": args.baseline,
            "vina_sign": "negative is better; a gain is a NEGATIVE delta",
            "reference_run": ref_run,
        },
        # PER-POCKET VALUES and CORRELATIONS are kept in separate blocks that both have a
        # "deltas" key, because they are different shapes of the same word: `deltas` here is
        # {arm: {target: value}} and correlations["deltas"] is {arm: {outcome: [entry]}}.
        # Flattening them into one namespace would give one key two meanings.
        "pockets": pockets,
        "arms": arms,
        "deltas": deltas,
        "correlations": {"deltas": stats_block["deltas"],
                         "absolute": stats_block["absolute"]},
        "descriptors": stats_block["descriptors"],
        "outcomes": stats_block["outcomes"],
        "control": stats_block["control"],
        "targets": stats_block["targets"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"  wrote {os.path.relpath(os.path.abspath(args.out), REPO)}")


main()

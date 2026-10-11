#!/usr/bin/env python3
"""uncertainty_descriptors.py — the crystallographic descriptors, and the statistics on them.

Shared by the two questions that ask the same thing of different data:

  eval_pocket_uncertainty.py    does the GENERATION gain (CoDE/CoE vs VoxBind, Vina) track how
                               well ordered the pocket is?  79 CrossDocked pockets.
  eval_affinity_uncertainty.py  does the REGRESSION gain (C+D+G vs C, pK error) track it?
                               the lp_edrscc_v2 PDBbind test complexes.

The two live in different trees with different file layouts, but "B-factor of the atoms the
ligand touches", "occupancy the refinement actually placed", "Spearman with a BH q and a
bootstrapped slope" are one definition each, and a second copy of any of them is a second
thing to keep in step. So the math is here and the two builders only locate files.

THE ONE RULE THAT MATTERS, and it is a data-provenance rule, not a math one:

    B-FACTORS survive the processed files. CrossDocked's `*_rec.pdb` and PDBbind's
    `*_pocket.pdb` both carry the deposited temperature factors unchanged (verified
    atom-for-atom against the deposited mmCIF: 8,045 of 8,045 atoms of 4yhj, max |Δ| 0.0000).

    OCCUPANCY DOES NOT. Both pipelines keep one altloc per atom and rewrite every occupancy
    to 1.00. Reading occupancy off a processed file returns "1.0 everywhere" — a constant
    that looks like data. It must come from the deposited mmCIF, and that is why
    `deposited_occupancy` takes a CIF path and nothing else does.
"""
from __future__ import annotations

import collections

import gemmi
import numpy as np
from scipy import stats

# Contact weighting. CONTACT_CUT is the usual 4.5 Å heavy-atom contact distance;
# CONTACT_SIGMA makes the Gaussian shell about one van der Waals contact thick;
# CONTACT_FLOOR stops 1/d² exploding where a crystal contact is unusually short.
CONTACT_CUT, CONTACT_SIGMA, CONTACT_FLOOR = 4.5, 1.5, 2.0


# ──────────────────────────────────────────────────────────────────────────────
# reading structures
# ──────────────────────────────────────────────────────────────────────────────
def ligand_coords(path):
    """Heavy-atom coordinates of an SDF, parsed from the V2000 block.

    Parsed by hand rather than through RDKit because this needs positions and nothing else,
    and a sanitization failure on a crystal ligand must not be able to drop a complex."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    n_atoms = int(lines[3][0:3])
    xyz = []
    for line in lines[4:4 + n_atoms]:
        if line[31:34].strip() == "H":
            continue
        xyz.append((float(line[0:10]), float(line[10:20]), float(line[20:30])))
    return np.asarray(xyz, dtype=np.float64)


def protein_atoms(path):
    """[(chain, seqid, atom name, b_iso, x, y, z)] for one PDB/mmCIF, hydrogens dropped."""
    out = []
    structure = gemmi.read_structure(str(path))
    structure.remove_hydrogens()
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                out.append((chain.name, residue.seqid.num, atom.name, atom.b_iso,
                            atom.pos.x, atom.pos.y, atom.pos.z))
    return out


def atom_keys(path):
    """{(chain, seqid, atom name)} of one structure — used to mark which atoms are in a crop."""
    return {a[:3] for a in protein_atoms(path)}


def deposited_occupancy(cif_path):
    """{(chain, seqid, atom name): (occ_sum, n_altloc)} from the deposited mmCIF.

    Altlocs of one atom are SUMMED, so `occ_sum` is how much of the atom the refinement
    placed and `n_altloc` is how many conformers it was split between. Conflating the two
    would call every 0.5/0.5 side chain "half occupied" when it is fully occupied and
    disordered — a different statement, and one B-factor already partly makes."""
    structure = gemmi.read_structure(str(cif_path))
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
    structure = gemmi.read_structure(str(cif_path))
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
def contact_weighted_b(dist, b_all, mu, sigma):
    """Three contact-weighted B-factors plus the plain 4.5 Å shell mean, as a dict.

    WHY WEIGHT AT ALL. A plain mean over "receptor atoms within 5 Å" gives an atom at 4.9 Å
    the same vote as one actually touching the ligand at 3.2 Å, so it measures what happened
    to sit inside a sphere as much as it measures the interface. Each weight below is how
    much ligand a receptor atom faces:

      invd2  Σ 1/d² over ligand atoms, floored at CONTACT_FLOOR so a near-zero distance
             cannot take over the average. Long tail, and no cutoff to justify.
      gauss  Σ exp(−d²/2σ²). A soft shell one contact thick; nothing flips in or out of the
             average because a pose moved 0.1 Å.
      count  the number of ligand atoms within CONTACT_CUT — the crudest and the most
             legible, since it is the contact count a reader already has an intuition for.

    All three are returned because they disagree where the site is shallow, and a claim that
    holds under only one weighting is a claim about that weighting. `shell` is the UNWEIGHTED
    mean over the same 4.5 Å atoms, so the effect of weighting can be told apart from the
    effect of simply using a tighter shell.

    `dist` is (n_receptor, n_ligand); `mu`/`sigma` are that structure's whole-protein B mean
    and sd, for the z columns."""
    dmin = dist.min(1)
    inv = 1.0 / np.maximum(dist, CONTACT_FLOOR) ** 2
    weights = {
        "invd2": inv.sum(1),
        "gauss": np.exp(-(dist ** 2) / (2.0 * CONTACT_SIGMA ** 2)).sum(1),
        "count": (dist <= CONTACT_CUT).sum(1).astype(np.float64),
    }
    out = {}
    for name, w in weights.items():
        total = w.sum()
        if total <= 0:
            out[f"b_contact_{name}"] = out[f"b_contact_{name}_z"] = None
            continue
        value = float((w * b_all).sum() / total)
        out[f"b_contact_{name}"] = value
        out[f"b_contact_{name}_z"] = float((value - mu) / sigma) if sigma > 0 else None
    tight = dmin <= CONTACT_CUT
    out["b_contact_shell_mean"] = float(b_all[tight].mean()) if tight.any() else None
    out["n_contact_atoms"] = int(tight.sum())
    return out


def occupancy_stats(keys, mask, occ, scope):
    """Occupancy summary over the masked atoms, keyed into a deposited-mmCIF map.

    `occ<1` and `altloc` are kept as separate columns on purpose: they are different kinds of
    disorder and the second is far commoner than the first."""
    vals = [occ[k] for k, m in zip(keys, mask) if m and k in occ]
    sums = [v[0] for v in vals]
    return {
        f"occ_{scope}_matched": len(vals),
        f"occ_{scope}_mean": float(np.mean(sums)) if sums else None,
        f"occ_{scope}_min": float(min(sums)) if sums else None,
        f"occ_{scope}_frac_partial": float(np.mean([s < 0.99 for s in sums])) if sums else None,
        f"occ_{scope}_frac_altloc": float(np.mean([v[1] > 1 for v in vals])) if vals else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# the statistics
# ──────────────────────────────────────────────────────────────────────────────
def rank(values):
    """Average ranks, so ties do not invent an order. Spearman is Pearson on these."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    for start, count in zip(first, counts):
        if count > 1:
            ranks[order[start:start + count]] = ranks[order[start:start + count]].mean()
    return ranks


def partial_spearman(x, y, z):
    """Spearman of x and y with z regressed out of both, on ranks. None if z is constant."""
    xr, yr, zr = rank(x), rank(y), rank(z)
    if np.ptp(zr) == 0:
        return None
    design = np.column_stack([zr, np.ones_like(zr)])
    rx = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    ry = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def slope_ci(x, y, n_boot=2000, seed=0):
    """(slope, lo, hi, span) — OLS slope of y on x with a bootstrap CI over samples.

    A NULL IS ONLY A RESULT IF IT IS BOUNDED. "ρ = 0.02, q = 0.90" says the data does not show
    an effect; it does not say the effect is small, and a reader is right to ask whether a real
    trend could be hiding. Multiplied by the observed range of the descriptor, this slope CI is
    the largest end-to-end effect the data still allows, in the unit the question was asked in.

    Simple regression is cov(x, y) / var(x), so the resamples are computed at once rather than
    re-fitted one at a time. Resamples where x collapses to a single value are dropped — with a
    descriptor like occupancy, where most samples are tied, they are common and the fit on them
    does not exist."""
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


def bh(pvalues):
    """Benjamini-Hochberg q-values.

    These builders run every descriptor against every outcome against every arm — over a
    thousand tests. At that many, one p below 0.05 is the number a pure null predicts, so a
    raw p is not evidence and the figures label the q."""
    idx = [i for i, p in enumerate(pvalues) if p == p]
    ordered = sorted(idx, key=lambda i: pvalues[i])
    n, out, running = len(ordered), [None] * len(pvalues), 1.0
    for position, i in reversed(list(enumerate(ordered, 1))):
        running = min(running, pvalues[i] * n / position)
        out[i] = running
    return out


def correlate_one(x, y, control=None, *, min_n=8):
    """Every statistic this project reports for one (descriptor, outcome) pair, as a dict."""
    keep = ~(np.isnan(x) | np.isnan(y))
    n = int(keep.sum())
    entry = {"n": n}
    if n < min_n or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        entry.update(spearman=None, p=None, pearson=None, pearson_p=None, n_informative=None,
                     partial_spearman=None, slope=None, slope_lo=None, slope_hi=None,
                     span=None, effect=None, effect_lo=None, effect_hi=None)
        return entry
    rho, p = stats.spearmanr(x[keep], y[keep])
    r, pr = stats.pearsonr(x[keep], y[keep])
    slope, lo, hi, span = slope_ci(x[keep], y[keep])
    entry.update(
        spearman=float(rho), p=float(p), pearson=float(r), pearson_p=float(pr),
        # How many samples carry a value that is not the modal one. Nine of 79 pockets have an
        # altloc in the site, so an occupancy ρ there is nine points moving against seventy
        # tied ones — this column says so before anyone reads the ρ.
        n_informative=int(np.sum(x[keep] != stats.mode(x[keep], keepdims=False).mode)),
        partial_spearman=(None if control is None else
                          partial_spearman(x[keep], y[keep], control[keep])),
        slope=slope, slope_lo=lo, slope_hi=hi, span=span,
        effect=None if slope is None else slope * span,
        effect_lo=None if lo is None else lo * span,
        effect_hi=None if hi is None else hi * span,
    )
    return entry


def tercile_split(x, y, n_boot=8000, seed=0):
    """Δ in each tercile of x, and the TOP-vs-REST contrast with a permutation p.

    WHY A TERCILE AND NOT JUST THE ρ. Spearman asks whether the effect is MONOTONE across the
    whole descriptor; an effect that is flat over two thirds of the range and then turns on is
    real but nearly invisible to it (ρ = −0.06 for a contrast that a permutation test rejects
    at p = 0.008). So both are reported and they answer different questions: ρ is "does it
    track", this is "is there a regime".

    The contrast is TOP-vs-REST, fixed before looking, because three pairwise tercile
    comparisons chosen after the fact is three chances to find one. The permutation p is on
    the difference of means under label shuffling; the CI is a two-sample bootstrap."""
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 24 or np.ptp(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    edges = np.quantile(x, [0.0, 1 / 3, 2 / 3, 1.0])
    bins = []
    for i in range(3):
        sel = (x >= edges[i]) & ((x <= edges[i + 1]) if i == 2 else (x < edges[i + 1]))
        vals = y[sel]
        if not vals.size:
            bins.append(None)
            continue
        boot = vals[rng.integers(0, len(vals), (n_boot, len(vals)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        bins.append({"lo_edge": float(edges[i]), "hi_edge": float(edges[i + 1]),
                     "n": int(sel.sum()), "mean": float(vals.mean()),
                     "ci_lo": float(lo), "ci_hi": float(hi)})
    cut = edges[2]
    top, rest = y[x >= cut], y[x < cut]
    if not top.size or not rest.size:
        return {"bins": bins, "contrast": None}
    observed = float(top.mean() - rest.mean())
    bt = (top[rng.integers(0, len(top), (n_boot, len(top)))].mean(axis=1)
          - rest[rng.integers(0, len(rest), (n_boot, len(rest)))].mean(axis=1))
    lo, hi = np.percentile(bt, [2.5, 97.5])
    pool = np.concatenate([top, rest])
    n_top = len(top)
    shuffled = np.array([rng.permutation(pool) for _ in range(n_boot)])
    null = shuffled[:, :n_top].mean(axis=1) - shuffled[:, n_top:].mean(axis=1)
    return {"bins": bins,
            "contrast": {"top_minus_rest": observed, "ci_lo": float(lo), "ci_hi": float(hi),
                         "p_permutation": float((np.abs(null) >= abs(observed)).mean()),
                         "n_top": n_top, "n_rest": len(rest)}}

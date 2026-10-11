#!/usr/bin/env python3
"""eval_affinity_uncertainty.py — per-complex AFFINITY error beside per-complex crystallography.

    conda run -n voxbind python figures/_parts/fig_uncertainty/eval_affinity_uncertainty.py

THE QUESTION, on the regression task this time. `eval_pocket_uncertainty.py` asks whether the
GENERATION gain tracks how well ordered the pocket is, and answers on 79 pockets. This asks the
same thing where the answer has two orders of magnitude more data: the frozen-encoder binding
affinity probe predicts one pK per complex, so every complex carries its own ERROR, and the
difference between two encoders is defined per complex rather than per pocket.

    C       coords-only encoder  — the regression-task twin of the CoE arm
    C+D+G   + X-ray density + density gradient — the twin of CoDE

Both are probed in ONE invocation of probe.py so `--intersect` puts them on the
identical complexes with the identical probe head and train/val split; the per-complex test
predictions land in the canonical scatter sink, and this reads them. The error difference
    Δ = |pred_CDG − pK| − |pred_C − pK|
is NEGATIVE where density helped that complex.

WHAT IT IS CORRELATED AGAINST, and the trap in each:

  RSCC        the wwPDB validation report's real-space correlation coefficient — how well the
              deposited ligand (and the pocket residues) actually fit their electron density.
              The most direct measure of the conditioning input there is.
              **TRAP: the split is SELECTED ON IT.** lp_edrscc_v2 keeps only complexes with
              ligand RSCC ≥ 0.8, so the range here is truncated at exactly 0.800 and the
              badly-resolved tail — the half of the axis where a density model should struggle
              — does not exist in this data. Any RSCC correlation measured here is attenuated
              by that restriction and is a lower bound, not an estimate.

  B-factor    from `structures/pbpp-2020/<id>/<id>_{pocket,protein}.pdb`, which preserve the
              deposited temperature factors. Raw, contact-weighted (three weightings) and
              normalized within each structure — raw B is not comparable between entries, since
              a 2.6 Å structure refines to higher B everywhere than a 1.5 Å one.

  occupancy   deposited mmCIF ONLY. PDBbind's pocket.pdb is X-TOOL output with every occupancy
              rewritten to 1.00, exactly as CrossDocked's receptors are, so reading it there
              returns a constant that looks like data.

  controls    resolution, R_free, the ligand's heavy-atom count, and the complex's own pK.
              The last is the one that matters: a regression error is largest at the ends of
              the label range for any model, so if a "density helps here" pattern is really a
              "this complex has an extreme pK" pattern, this column shows it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uncertainty_descriptors as ud                                      # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PDBBIND = os.path.join(REPO, "dataset", "data", "pdbbind")
STRUCTURES = os.path.join(PDBBIND, "structures", "pbpp-2020")
SCATTER_DIR = os.path.join(PDBBIND, "results", "scatter")
CIF_DIR = os.path.join(REPO, "dataset", "data", "cif")
BFACTORS = os.path.join(PDBBIND, "bfactors.json")
RSCC_CSV = os.path.join(PDBBIND, "ligand_rscc.csv")

# (label, probe condition). The labels are the figures', and the mapping to the generation
# arms is the point of the whole file: C is coords-only as CoE is, C+D+G carries the density
# channels as CoDE does. BASELINE is what a "gain" is measured against.
ARMS = [("C", "atomblob"), ("C+D+G", "atomblob_density_gradmag")]
BASELINE = "C"

DESCRIPTORS = [
    ("lig_rscc",             "Ligand RSCC",                              "RSCC ligand"),
    ("poc_rscc_median",      "Pocket residue RSCC (median)",             "RSCC pocket"),
    ("b_contact_invd2",      "Contact-weighted B-factor, 1/d² (Å²)",     "B contact 1/d²"),
    ("b_contact_gauss",      "Contact-weighted B-factor, Gaussian (Å²)", "B contact gauss"),
    ("b_contact_count",      "Contact-weighted B-factor, contacts (Å²)", "B contact count"),
    ("b_contact_shell_mean", "B-factor of atoms within 4.5 Å (Å²)",      "B 4.5 Å shell"),
    ("b_contact_invd2_z",    "Contact-weighted B-factor (z)",            "B contact z"),
    ("mean_pocket_b",        "Pocket B-factor (Å²)",                     "B pocket"),
    ("mean_protein_b",       "Protein B-factor (Å²)",                    "B protein"),
    ("rel_pocket_b",         "Pocket B / protein B",                     "B pocket ratio"),
    ("occ_site_min",         "Lowest site occupancy",                    "Occ site min"),
    ("occ_site_frac_partial", "Site atoms at occupancy < 1",             "Occ<1 site"),
    ("occ_site_frac_altloc", "Site atoms with an alternate conformer",   "Altloc site"),
    ("resolution",           "Resolution (Å)",                           "Resolution"),
    ("r_free",               "R$_{free}$",                               "R-free"),
    ("n_ligand_heavy_atoms", "Ligand heavy atoms",                       "Ligand size"),
    ("y_true",               "Measured pK",                              "Measured pK"),
]
OUTCOMES = [
    ("abs_error",    "|pK error|"),
    ("sq_error",     "squared pK error"),
    ("signed_error", "signed pK error (pred − measured)"),
]
# Every correlation is read against this. A regression error is biggest at the ends of the
# label range for any model, so a pattern that is really "this complex has an extreme pK"
# has to be separable from one about the crystallography.
CONTROL = "y_true"


def scatter_rows(scatter_dir, split_tag):
    """{condition: {pid: {"y_true", "y_pred" (mean over seeds), "n_seeds"}}}.

    SEEDS ARE AVERAGED, not taken one at a time. One probe head's prediction on one complex
    carries real seed noise (the three heads disagree by ~0.1 pK on a typical complex), and
    the quantity here is a per-complex error difference of order 0.1 — so a single seed would
    put noise of the same size as the signal on every point."""
    per = {}
    for path in sorted(glob.glob(os.path.join(scatter_dir, f"scatter_*{split_tag}*.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                slot = per.setdefault(r["condition"], {}).setdefault(
                    r["pid"], {"y_true": float(r["y_true"]), "preds": []})
                slot["preds"].append(float(r["y_pred"]))
    return {cond: {pid: {"y_true": v["y_true"], "y_pred": float(np.mean(v["preds"])),
                         "n_seeds": len(v["preds"])}
                   for pid, v in rows.items()}
            for cond, rows in per.items()}


def complex_descriptors(pid, bfac, rscc):
    """Every crystallographic descriptor of one PDBbind complex, or None with a reason."""
    pocket = os.path.join(STRUCTURES, pid, f"{pid}_pocket.pdb")
    protein = os.path.join(STRUCTURES, pid, f"{pid}_protein.pdb")
    ligand = os.path.join(STRUCTURES, pid, f"{pid}_ligand.sdf")
    for path in (pocket, ligand):
        if not os.path.exists(path):
            return None, f"missing {os.path.basename(path)}"

    atoms = ud.protein_atoms(pocket)
    if not atoms:
        return None, "empty pocket.pdb"
    xyz = np.asarray([a[4:7] for a in atoms], dtype=np.float64)
    b_pocket = np.asarray([a[3] for a in atoms], dtype=np.float64)
    keys = [a[:3] for a in atoms]
    try:
        lig = ud.ligand_coords(ligand)
    except (ValueError, IndexError) as exc:
        return None, f"unparsable ligand.sdf ({exc})"
    if not len(lig):
        return None, "ligand.sdf has no heavy atoms"

    # The z columns normalize against the WHOLE protein, not the pocket: the point of them is
    # to remove the per-structure B scale, and the pocket is too small a sample to define it.
    entry = bfac.get(pid) or {}
    mu = entry.get("mean_protein_b")
    if mu is None or mu != mu:
        mu = float(b_pocket.mean())
    protein_b = (np.asarray([a[3] for a in ud.protein_atoms(protein)], dtype=np.float64)
                 if os.path.exists(protein) else b_pocket)
    sigma = float(protein_b.std(ddof=1)) if protein_b.size > 1 else 0.0

    dist = np.sqrt(((xyz[:, None, :] - lig[None, :, :]) ** 2).sum(-1))
    row = {"pid": pid, "n_pocket_atoms": len(atoms), "n_ligand_heavy_atoms": int(len(lig))}
    row.update(ud.contact_weighted_b(dist, b_pocket, float(mu), sigma))
    for key in ("mean_pocket_b", "mean_protein_b", "rel_pocket_b"):
        value = entry.get(key)
        row[key] = None if value is None or value != value else float(value)

    hit = rscc.get(pid) or {}
    for key in ("lig_rscc", "poc_rscc_median"):
        value = hit.get(key)
        row[key] = float(value) if value not in (None, "", "None") else None

    cif = os.path.join(CIF_DIR, f"{pid}.cif")
    if os.path.exists(cif):
        occ = ud.deposited_occupancy(cif)
        row["resolution"], row["r_work"], row["r_free"] = ud.deposited_refinement(cif)
        row["occ_match_frac"] = (sum(1 for k in keys if k in occ) / len(keys)) if keys else 0.0
        row.update(ud.occupancy_stats(keys, dist.min(1) <= ud.CONTACT_CUT, occ, "site"))
    else:
        row["resolution"] = row["r_work"] = row["r_free"] = row["occ_match_frac"] = None
        row.update(ud.occupancy_stats([], [], {}, "site"))
    return row, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scatter-dir", default=SCATTER_DIR)
    ap.add_argument("--split", default="lp_edrscc_v2")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", "task1-affinity", "_shared", "260916_affinity_uncertainty",
        "affinity_uncertainty.json"))
    args = ap.parse_args()

    preds = scatter_rows(args.scatter_dir, args.split)
    by_label = {label: preds[cond] for label, cond in ARMS if cond in preds}
    missing = [label for label, cond in ARMS if cond not in preds]
    if args.baseline not in by_label or len(by_label) < 2:
        sys.exit(f"need both arms in {args.scatter_dir}; found {sorted(preds)}\n"
                 f"    run: python task1-affinity_regression/probe.py probe "
                 f"--conditions {' '.join(c for _, c in ARMS)} --epoch 99 "
                 f"--split {args.split} --seeds 3 --allow_stale_features")
    if missing:
        print(f"  no predictions for: {', '.join(missing)}")
    pids = sorted(set.intersection(*(set(v) for v in by_label.values())))
    seeds = {label: sorted({v["n_seeds"] for v in rows.values()})
             for label, rows in by_label.items()}
    print(f"  {len(pids)} complexes with both arms ({args.split} test); seeds {seeds}")

    bfac = json.load(open(BFACTORS, encoding="utf-8")) if os.path.exists(BFACTORS) else {}
    rscc = {}
    if os.path.exists(RSCC_CSV):
        with open(RSCC_CSV, newline="", encoding="utf-8") as fh:
            rscc = {r["pid"].lower(): r for r in csv.DictReader(fh)}
    print(f"  B-factor cache {len(bfac):,} complexes; RSCC table {len(rscc):,}")

    complexes, failures = {}, []
    for pid in pids:
        row, why = complex_descriptors(pid, bfac, rscc)
        if row is None:
            failures.append((pid, why))
            continue
        complexes[pid] = row
    print(f"  descriptors for {len(complexes)} complexes"
          + (f"; {len(failures)} failed, e.g. {failures[:3]}" if failures else ""))
    for key in ("lig_rscc", "resolution", "mean_pocket_b"):
        have = sum(1 for r in complexes.values() if r.get(key) is not None)
        print(f"    {key:16s} present for {have}/{len(complexes)}")

    # Per-complex errors and the difference. NEGATIVE Δ = density helped this complex.
    arms, deltas = {}, {}
    for label, rows in by_label.items():
        arms[label] = {pid: {"y_true": rows[pid]["y_true"], "y_pred": rows[pid]["y_pred"],
                             "signed_error": rows[pid]["y_pred"] - rows[pid]["y_true"],
                             "abs_error": abs(rows[pid]["y_pred"] - rows[pid]["y_true"]),
                             "sq_error": (rows[pid]["y_pred"] - rows[pid]["y_true"]) ** 2}
                       for pid in complexes}
    base = arms[args.baseline]
    for label, rows in arms.items():
        if label == args.baseline:
            continue
        deltas[label] = {pid: {key: rows[pid][key] - base[pid][key]
                               for key in ("signed_error", "abs_error", "sq_error")}
                         for pid in complexes}
    for label, rows in arms.items():
        errs = np.array([rows[pid]["abs_error"] for pid in complexes])
        sq = np.array([rows[pid]["sq_error"] for pid in complexes])
        print(f"  {label:6s} MAE {errs.mean():.4f}  RMSE {np.sqrt(sq.mean()):.4f}")

    # y_true is a descriptor (the control) and lives in the complex table, not just the arms.
    for pid in complexes:
        complexes[pid]["y_true"] = arms[args.baseline][pid]["y_true"]

    order = sorted(complexes)
    def column(source, key):
        return np.asarray([(source.get(p, {}).get(key)
                            if source.get(p, {}).get(key) is not None else np.nan)
                           for p in order], dtype=np.float64)
    control = column(complexes, CONTROL)

    correlations = {"deltas": {}, "absolute": {}}
    for half, source in (("deltas", deltas), ("absolute", arms)):
        for label, rows in source.items():
            per_arm = {}
            for outcome, _ in OUTCOMES:
                entries = []
                y = column(rows, outcome)
                for key, _, _ in DESCRIPTORS:
                    entry = ud.correlate_one(column(complexes, key), y,
                                             None if key == CONTROL else control)
                    entry["descriptor"] = key
                    entries.append(entry)
                qs = ud.bh([e["p"] if e["p"] is not None else float("nan") for e in entries])
                for entry, q in zip(entries, qs):
                    entry["q"] = q
                per_arm[outcome] = entries
            correlations[half][label] = per_arm

    # The regime test, beside the monotone one. Computed for the delta half only: a tercile
    # structure in an arm's ABSOLUTE error is just the label range, and says nothing.
    terciles = {}
    for label, rows in deltas.items():
        per_arm = {}
        for outcome, _ in OUTCOMES:
            y = column(rows, outcome)
            per_arm[outcome] = {key: ud.tercile_split(column(complexes, key), y)
                                for key, _, _ in DESCRIPTORS}
        terciles[label] = per_arm

    headline = correlations["deltas"].get("C+D+G", {}).get("abs_error", [])
    for key in ("b_contact_invd2", "lig_rscc"):
        c = (terciles.get("C+D+G", {}).get("abs_error", {}).get(key) or {}).get("contrast")
        if c:
            print(f"  Δ|error| top tercile of {key:<18} minus the rest: "
                  f"{c['top_minus_rest']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] pK, "
                  f"permutation p={c['p_permutation']:.4f}")
    for entry in headline:
        if entry["descriptor"] in ("lig_rscc", "poc_rscc_median", "b_contact_invd2"):
            bound = ("" if entry.get("effect_lo") is None else
                     f"  effect over its range {entry['effect']:+.3f} "
                     f"[{entry['effect_lo']:+.3f}, {entry['effect_hi']:+.3f}] pK")
            print(f"  Δ|error| vs {entry['descriptor']:<18} rho={entry['spearman']:+.3f} "
                  f"p={entry['p']:.3f} q={entry['q']:.3f} n={entry['n']}{bound}")

    out = {
        "protocol": {
            "split": args.split,
            "arms": {label: cond for label, cond in ARMS},
            "baseline": args.baseline,
            "predictions": "probe.py scatter CSVs, averaged over seeds",
            "error_sign": "Δ is arm minus baseline; NEGATIVE means density helped",
            "rscc_caveat": "the split is selected on ligand RSCC >= 0.8 — the axis is "
                           "truncated and any RSCC correlation here is a lower bound",
            "occupancy": "deposited mmCIF only; PDBbind pocket.pdb rewrites occupancy to 1.00",
        },
        "complexes": complexes,
        "arms": arms,
        "deltas": deltas,
        "correlations": correlations,
        "terciles": terciles,
        "descriptors": DESCRIPTORS,
        "outcomes": OUTCOMES,
        "control": CONTROL,
        "targets": order,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"  wrote {os.path.relpath(os.path.abspath(args.out), REPO)}")


main()

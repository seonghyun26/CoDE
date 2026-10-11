"""build_zeroshot_json.py — turn the cached DSMBind zero-shot ENERGY dumps into the
per-cohort result JSONs (one per split and cohort).

方式1 (zero-shot, deterministic — NO seeds, NO training). The model outputs a binding
energy (arbitrary units, sign anti-correlates with pK by construction), so:
  * r / rho are reported as their MAGNITUDE (|r|, |ρ|) — every table marks DSMBind with a
    dagger; we store |value| in the "mean" field so bar charts / un-absmetric'd tables show
    positive bars, and keep the raw signed value in "signed" for the record.
  * RMSE is N/A — energy is not on the pK scale — stored as null so the loaders render "n/a".

Energy dumps consumed (score = -energy):
  base/dsmbind/results/dsmbind_edrscc_preds.csv            (lp_edrscc_v2 TEST, 1320)
  base/dsmbind/results/dsmbind_casf2016_zeroshot_preds.csv (CASF-2016 core, 214)

Cohorts written (all zero-shot, RMSE=null):
  Table 1a   lp_edrscc_v2 / _cl1 / _cl12 / _cl123   -> base/dsmbind/results/dsmbind_zeroshot_{split}.json
  Table 1b   lp_test_novel60/30                     -> base/_casf/novel_subsets/DSMBind__{split}.json
             cl123_test_novel60/30 (+cl123_test)    -> patch base/_casf/cl123_seqfilter_5seed_260818/results.json (rmse->null)
  CASF       leaky / nontrain / clean               -> base/_casf/DSMBind_zeroshot.json
             id60 / id30                            -> base/_casf/casf_similarity/DSMBind__casf_clean_cl3_novel{60,30}.json
"""
import csv
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DSM_ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(DSM_ROOT))
SPLITS = os.path.join(REPO, "voxbind", "splits")
CASF = os.path.join(REPO, "base", "_casf")
RESULTS = os.path.join(DSM_ROOT, "results")

LP_PREDS = os.path.join(RESULTS, "dsmbind_edrscc_preds.csv")
CASF_PREDS = os.path.join(RESULTS, "dsmbind_casf2016_zeroshot_preds.csv")

NOTE = ("方式1 zero-shot DSMBind binding energy (NeurIPS 2023), deterministic single pass, "
        "no seeds/training. score = -energy; sign is arbitrary so |r|/|ρ| are reported and "
        "RMSE is N/A (energy not on the pK scale).")


# ── data loaders ─────────────────────────────────────────────────────────────
def read_preds(path):
    """pid -> (pK, score) from a zero-shot energy dump (pid,pK,energy,score)."""
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["pid"].lower()] = (float(r["pK"]), float(r["score"]))
    return out


def split_test_ids(name):
    return {r["pid"].lower() for r in csv.DictReader(open(os.path.join(SPLITS, f"{name}.csv")))
            if r["split"] == "test"}


def txt_ids(path):
    return {l.strip().lower() for l in open(path) if l.strip()}


def csv_col_ids(path, col="pid"):
    return {r[col].lower() for r in csv.DictReader(open(path))}


# ── metrics + schema ─────────────────────────────────────────────────────────
def signed_metrics(preds, keep):
    pids = [p for p in preds if keep is None or p in keep]
    y = np.array([preds[p][0] for p in pids])
    s = np.array([preds[p][1] for p in pids])
    return float(pearsonr(s, y)[0]), float(spearmanr(s, y).statistic), len(pids)


def block(preds, keep):
    """schema-A metric block with |r|,|ρ| in mean, signed kept, rmse=null."""
    r, rho, n = signed_metrics(preds, keep)
    return {
        "n": n,
        "pearson": {"mean": abs(r), "std": 0.0},
        "spearman": {"mean": abs(rho), "std": 0.0},
        "rmse": None,
        "signed": {"pearson": r, "spearman": rho},
    }


def write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(payload, open(path, "w"), indent=2)
    print(f"  -> {os.path.relpath(path, REPO)}")


def main():
    lp = read_preds(LP_PREDS)
    print(f"lp zero-shot preds: {len(lp)}")

    # ── Table 1a: lp_edrscc_v2 + CL tiers ────────────────────────────────────
    print("\n[Table 1a] lp_edrscc_v2 + CL tiers")
    for split in ("lp_edrscc_v2", "lp_edrscc_v2_cl1", "lp_edrscc_v2_cl12", "lp_edrscc_v2_cl123"):
        if not os.path.exists(os.path.join(SPLITS, f"{split}.csv")):
            print(f"[skip] {split}: no split CSV")
            continue
        keep = split_test_ids(split)
        b = block(lp, keep)
        b.update(model="DSMBind (zero-shot binding energy)", split=split, subset="test",
                 protocol="cached zero-shot energy; deterministic single pass; no seeds", note=NOTE)
        print(f"  {split:22} n={b['n']:4d}  |r|={b['pearson']['mean']:.3f} |rho|={b['spearman']['mean']:.3f}"
              f"  (signed rho={b['signed']['spearman']:+.3f})")
        write(os.path.join(RESULTS, f"dsmbind_zeroshot_{split}.json"), b)

    # ── Table 1b: LP protein-novelty subsets ─────────────────────────────────
    print("\n[Table 1b] lp_test_novel60/30")
    for split, fname in (("lp_test_novel60", "lp_test_novel60.txt"),
                         ("lp_test_novel30", "lp_test_novel30.txt")):
        keep = txt_ids(os.path.join(CASF, fname))
        b = block(lp, keep)
        b.update(method="DSMBind", split=split, note=NOTE)
        print(f"  {split:18} n={b['n']:4d}  |r|={b['pearson']['mean']:.3f} |rho|={b['spearman']['mean']:.3f}")
        write(os.path.join(CASF, "novel_subsets", f"DSMBind__{split}.json"), b)

    # ── Table 1b: patch cl123_seqfilter results.json (rmse -> null) ───────────
    print("\n[Table 1b] cl123_seqfilter results.json  (null the energy-scale rmse)")
    rj_path = os.path.join(CASF, "cl123_seqfilter_5seed_260818", "results.json")
    rj = json.load(open(rj_path))
    dsm = rj.get("methods", {}).get("DSMBind", {})
    for cohort, blk in dsm.items():
        if isinstance(blk, dict) and "rmse" in blk:
            blk["rmse"] = {"mean": None, "std": None}
            for ps in blk.get("per_seed", []):
                ps["rmse"] = None
            print(f"  nulled rmse for cl123 cohort {cohort}")
    json.dump(rj, open(rj_path, "w"), indent=1)

    # ── CASF cohorts (needs the CASF energy dump) ────────────────────────────
    if not os.path.exists(CASF_PREDS):
        print(f"\n[CASF] {CASF_PREDS} not found yet — skipping CASF cohorts.")
        return
    cf = read_preds(CASF_PREDS)
    print(f"\n[CASF] zero-shot preds: {len(cf)}")
    nontrain = {r["pid"].lower() for r in csv.DictReader(open(os.path.join(SPLITS, "casf2016_eval.csv")))
                if str(r.get("in_v2train", "")).strip() == "0"}
    clean = csv_col_ids(os.path.join(SPLITS, "casf2016_clean.csv"))
    out = {"model": "DSMBind (zero-shot binding energy)", "note": NOTE}
    for name, keep in (("leaky", None), ("nontrain", nontrain), ("clean", clean)):
        b = block(cf, keep)
        out[name] = b
        print(f"  {name:9} n={b['n']:4d}  |r|={b['pearson']['mean']:.3f} |rho|={b['spearman']['mean']:.3f}"
              f"  (signed rho={b['signed']['spearman']:+.3f})")
    write(os.path.join(CASF, "DSMBind_zeroshot.json"), out)

    # ── CASF sequence-novelty (id60/id30) ────────────────────────────────────
    print("\n[CASF] casf_similarity id60/id30")
    for cohort, fname in (("casf_clean_cl3_novel60", "casf_clean_cl3_novel60.txt"),
                          ("casf_clean_cl3_novel30", "casf_clean_cl3_novel30.txt")):
        keep = txt_ids(os.path.join(CASF, "casf_similarity", fname))
        b = block(cf, keep)
        b.update(method="DSMBind", split=cohort, note=NOTE)
        print(f"  {cohort:26} n={b['n']:3d}  |r|={b['pearson']['mean']:.3f} |rho|={b['spearman']['mean']:.3f}")
        write(os.path.join(CASF, "casf_similarity", f"DSMBind__{cohort}.json"), b)


if __name__ == "__main__":
    main()

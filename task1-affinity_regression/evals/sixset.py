#!/usr/bin/env python3
"""The six affinity test sets for one frozen-feature cache, measured as the baselines are.

    python task1-affinity_regression/evals/sixset.py CACHE.pt [--head regularized] [--seeds 10] [--csv OUT]

Per seed, a probe head (probe.train_one) is trained on each of two splits, early-stopped on that
split's validation Spearman, and scored on the split's test sets:

  head trained on       test set        complexes   pids (voxbind/splits/)
  lp_edrscc_v2          FULL            1,320       the split's test set
                        CASF-nontrain     124       casf2016_nontrain_cohort.csv
                        CASF-clean         92       casf2016_clean_cohort.csv
  lp_edrscc_v2_cl123    CL3               733       the split's test set
                        CL3-ID60          454       cl123_test_novel60_cohort.csv
                        CL3-ID30          262       cl123_test_novel30_cohort.csv

These are base/evaluate.py's test sets and scoring, so a row here and a baseline's row in
base/results/<method>.csv measure the same thing. CACHE is a probe.py features cache.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))
import probe as P  # noqa: E402

SPLITS = P.REPO / "voxbind" / "splits"
TEST_SETS = {
    "lp_edrscc_v2": [("FULL", None), ("CASF-nontrain", "casf2016_nontrain_cohort.csv"),
                     ("CASF-clean", "casf2016_clean_cohort.csv")],
    "lp_edrscc_v2_cl123": [("CL3", None), ("CL3-ID60", "cl123_test_novel60_cohort.csv"),
                           ("CL3-ID30", "cl123_test_novel30_cohort.csv")],
}
ORDER = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]

# Probe heads, as probe.train_one arguments. default = the Table 1 probe (two layers,
# dropout 0.1); regularized = z-scored features and pK, three layers, dropout 0.5, the head the
# 2026-10 probe search found for groupstats readouts (task1-affinity_regression/README.md).
_COMMON = dict(hidden=128, weight_decay=1e-4, max_epochs=200, patience=30, probe_loss="mse", aux_weight=1.0)
HEADS = {
    "default": dict(_COMMON, n_layers=2, dropout=0.1, lr=1e-3, batch_size=64, standardize=False),
    "regularized": dict(_COMMON, n_layers=3, dropout=0.5, lr=3e-4, batch_size=256, standardize=True),
}


def head_args(head) -> dict:
    """A HEADS name, or a dict that overrides the default head's arguments."""
    if isinstance(head, str):
        return dict(HEADS[head])
    unknown = set(head) - set(HEADS["default"])
    if unknown:
        raise ValueError(f"unknown probe-head arguments {sorted(unknown)}; known: {sorted(HEADS['default'])}")
    return dict(HEADS["default"], **head)


def pid_set(name: str | None, split: str) -> set:
    with open(SPLITS / (name or f"{split}.csv")) as fh:
        return {r["pid"].lower() for r in csv.DictReader(fh) if name or r["split"] == "test"}


def score(pred: dict, labels: dict, keep: set) -> dict:
    """base/evaluate.py's score: Pearson r, Spearman rho and RMSE over keep ∩ predictions."""
    pids = sorted(p for p in keep if p in pred and p in labels)
    if len(pids) < 3:
        return {}
    y = np.array([labels[p] for p in pids])
    x = np.array([pred[p] for p in pids])
    return {"n": len(pids), "pearson": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]),
            "rmse": float(np.sqrt(((x - y) ** 2).mean()))}


def load_cache(path) -> tuple[dict, dict]:
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.pop("features"), blob


def datasets(features: dict, lp, splits=tuple(TEST_SETS)) -> dict:
    """split -> probe.build_dataset arrays; a split's test arrays also hold its cohort pids."""
    labels = {p: float(v) for p, v in zip(lp["pdb_id"], lp["pK"]) if v == v}
    out = {}
    for split in splits:
        split_map, _ = P.load_frozen_split_map(split)
        d = P.build_dataset(features, lp, drop_covalent=True, cl1_only=False, target_map=None,
                            split_map=split_map)
        cohorts = {p for _, f in TEST_SETS[split] if f for p in pid_set(f, split)}
        extra = sorted((cohorts - set(d["test"]["pid"])) & set(features) & set(labels))
        if extra:
            d["test"]["X"] = np.concatenate([d["test"]["X"],
                                             np.stack([features[p].numpy() for p in extra]).astype(np.float32)])
            d["test"]["y"] = np.concatenate([d["test"]["y"], np.array([labels[p] for p in extra], np.float32)])
            d["test"]["pid"] = list(d["test"]["pid"]) + extra
        out[split] = d
    return out


def evaluate(features: dict, head, seeds: int, device: str, splits=tuple(TEST_SETS), log=print) -> list:
    """One row per (split, test set, seed): n, pearson, spearman, rmse, val_spearman."""
    lp = P.load_lp_index(P.LP_CSV)
    labels = {p: float(v) for p, v in zip(lp["pdb_id"], lp["pK"]) if v == v}
    args = head_args(head)
    rows = []
    for split, d in datasets(features, lp, splits).items():
        log(f"  {split}: train {len(d['train']['pid'])} val {len(d['val']['pid'])} "
            f"test and cohorts {len(d['test']['pid'])}")
        for seed in range(seeds):
            m = P.train_one(d, seed=seed, device=device, head="scalar", **args)
            pred = dict(zip(d["test"]["pid"], m["_pred_te"]))
            for test_set, pid_file in TEST_SETS[split]:
                s = score(pred, labels, pid_set(pid_file, split))
                if s:
                    rows.append({"split": split, "test_set": test_set, "seed": seed, **s,
                                 "val_spearman": float(m["best_val_spearman"])})
    return rows


def validation_rho(features: dict, head, seeds: int, device: str, split="lp_edrscc_v2_cl123") -> float:
    """Mean best validation Spearman of the head on one split; picks a readout layer."""
    lp = P.load_lp_index(P.LP_CSV)
    d = datasets(features, lp, (split,))[split]
    args = head_args(head)
    return float(np.mean([P.train_one(d, seed=s, device=device, head="scalar", **args)["best_val_spearman"]
                          for s in range(seeds)]))


def summary(rows: list) -> dict:
    """test set -> n and (mean, population std) over seeds of r, rho, RMSE, and the head's val rho."""
    out = {}
    for test_set in ORDER:
        sub = [r for r in rows if r["test_set"] == test_set]
        if sub:
            out[test_set] = {"n": sub[0]["n"], "seeds": len(sub), **{
                k: [float(np.mean([r[k] for r in sub])), float(np.std([r[k] for r in sub]))]
                for k in ("pearson", "spearman", "rmse", "val_spearman")}}
    return out


def table(summ: dict) -> str:
    f = lambda v: f"{v[0]:.3f}±{v[1]:.3f}"
    lines = [f"  {'test set':<14}{'n':>5}  {'r':>12}  {'ρ':>12}  {'RMSE':>12}  {'val ρ':>12}"]
    lines += [f"  {t:<14}{s['n']:>5}  {f(s['pearson']):>12}  {f(s['spearman']):>12}  {f(s['rmse']):>12}"
              f"  {f(s['val_spearman']):>12}" for t, s in summ.items()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cache", help="probe.py features cache (.pt), absolute or under dataset/data/pdbbind/features/")
    ap.add_argument("--head", default="default", choices=sorted(HEADS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csv", default=None, help="write the per-seed rows here")
    args = ap.parse_args()
    path = Path(args.cache) if Path(args.cache).is_absolute() else P.FEAT_DIR / args.cache
    features, meta = load_cache(path)
    print(f"{path.name}: {len(features)} complexes, dim {meta.get('feature_dim')}, "
          f"epoch {meta.get('epoch')}, run {meta.get('exp_dir')}, head {args.head}")
    rows = evaluate(features, args.head, args.seeds, args.device)
    print(table(summary(rows)))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

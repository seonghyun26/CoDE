"""fill_cl123_novelty.py — fill Table-1b CL3 protein-novelty cohorts for the baselines that
have per-complex cl123-test predictions (GET/EGNN/EGNN+TD/CheapNet/DeepDTA/MolTrans/AEV/HBGSA).

Re-scores each baseline's saved per-seed cl123-test predictions on cl123_test /
cl123_test_novel60 / cl123_test_novel30 (same masks the C/C+D+G/ProFSA campaign used) and
merges into cl123_seqfilter_5seed_260818/results.json['methods'][method][cohort], matching the
{n, seeds, per_seed, pearson/spearman/rmse:{mean,std}} schema the results tables read.
(r/rho/rmse are symmetric in pred/true, so column orientation doesn't matter.)
"""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818"


def ids(f):
    return set(l.strip().lower() for l in open(f) if l.strip())


COHORTS = {
    "cl123_test": ids(f"{D}/cl123_test.txt"),
    "cl123_test_novel60": ids(f"{D}/cl123_test_novel60.txt"),
    "cl123_test_novel30": ids(f"{D}/cl123_test_novel30.txt"),
}


def jsonl_seeds(pat, id_k, pred_k, y_k):
    out = []
    for f in sorted(glob.glob(pat)):
        d = {}
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[str(r[id_k]).lower()] = (float(r[pred_k]), float(r[y_k]))
        if d:
            out.append(d)
    return out


def csv_seeds(pat, pred_c, y_c, id_c="pid"):
    out = []
    for f in sorted(glob.glob(pat)):
        d = {r[id_c].lower(): (float(r[pred_c]), float(r[y_c])) for r in csv.DictReader(open(f))}
        if d:
            out.append(d)
    return out


def aev_loader():
    f = f"{REPO}/base/aevplig/results/aevplig_lp_edrscc_v2_cl123_preds.csv"
    rows = list(csv.DictReader(open(f)))
    out = []
    for s in range(5):
        col = f"pred_seed{s}"
        if col not in rows[0]:
            break
        out.append({r["pid"].lower(): (float(r[col]), float(r["truth"])) for r in rows})
    return out


def hbgsa_loader():
    out = []
    for f in sorted(glob.glob(f"{REPO}/base/hbgsa/results/preds_edrscc_v2_cl123_3p06m_seed*.json")):
        d = json.load(open(f))
        idk = next(k for k in d if k in ("pdb_id", "pid", "id"))
        pk = next(k for k in d if k in ("pred", "y_pred", "prediction", "pk_pred", "pred_pk"))
        yk = next(k for k in d if k in ("y", "label", "y_true", "pk", "truth", "true", "pk_true"))
        out.append({str(p).lower(): (float(pr), float(yy))
                    for p, pr, yy in zip(d[idk], d[pk], d[yk])})
    return out


def zeroshot_csv(path, pred_c, y_c):
    """Zero-shot models don't retrain per tier → their v2 per-complex preds ARE the cl123-test
    preds (masked to the cohort). One 'seed' (single deterministic pass)."""
    d = {r["pid"].lower(): (float(r[pred_c]), float(r[y_c])) for r in csv.DictReader(open(path))
         if r.get(pred_c) not in (None, "", "nan")}
    return [d]


# methods whose correlation sign is arbitrary (energy score) → report |r|/|rho|
ABS_METHODS = {"DSMBind"}

METHODS = {
    "GET":               lambda: jsonl_seeds(f"{REPO}/base/get/_edrscc/preds_GET_cl123_seed*.jsonl", "id", "label", "gt"),
    "EGNN":              lambda: jsonl_seeds(f"{REPO}/base/get/_edrscc/preds_EGNN_cl123_seed*.jsonl", "id", "label", "gt"),
    "EGNN + TargetDiff": lambda: jsonl_seeds(f"{REPO}/base/get/_edrscc/preds_EGNN_TD_cl123_seed*.jsonl", "id", "label", "gt"),
    "CheapNet":          lambda: jsonl_seeds(f"{REPO}/base/cheapnet/_edrscc/preds_lp_edrscc_v2_cl123_seed*.jsonl", "id", "pred", "label"),
    "DeepDTA":           lambda: csv_seeds(f"{REPO}/base/dta/result/preds_DeepDTA_lp_edrscc_v2_cl123_seed*.csv", "pred", "y"),
    "MolTrans":          lambda: csv_seeds(f"{REPO}/base/dta/result/preds_MolTrans_lp_edrscc_v2_cl123_seed*.csv", "pred", "y"),
    "AEV-PLIG":          aev_loader,
    "HBGSA":             hbgsa_loader,
    "DSMBind":           lambda: zeroshot_csv(f"{REPO}/base/dsmbind/results/dsmbind_edrscc_preds.csv", "score", "pK"),
    "Nesso-1":           lambda: zeroshot_csv(f"{REPO}/base/nesso/_edrscc/predictions_v2.csv", "pred_pIC50", "pK"),
    "C":                 lambda: csv_seeds(f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/preds_ours/C_cl123_seed*.csv", "pred", "y"),
    "C+D+G":             lambda: csv_seeds(f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/preds_ours/C_D_G_cl123_seed*.csv", "pred", "y"),
    "C+D+G +corr":       lambda: csv_seeds(f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/preds_ours/C_D_G_corr_cl123_seed*.csv", "pred", "y"),
    "CDG v2":            lambda: csv_seeds(f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/preds_ours/CDG_v2_cl123_seed*.csv", "pred", "y"),
    "CDG v3":            lambda: csv_seeds(f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/preds_ours/CDG_v3_cl123_seed*.csv", "pred", "y"),
    "HonestAffinity":    lambda: csv_seeds(f"{REPO}/base/honestaffinity/cache/cl123_preds/HonestAffinity_v2_seed*.csv", "pred", "y"),
    "BindNet":           lambda: csv_seeds(f"{REPO}/base/_casf/novel_preds/BindNet_cl123_seed*.csv", "pred", "y"),
    "IPNet (frozen)":    lambda: csv_seeds(f"{REPO}/base/ipdiff/_edrscc/cl123_preds/IPNet_frozen_cl123_seed*.csv", "pred", "y"),
    "IPNet (scratch)":   lambda: csv_seeds(f"{REPO}/base/ipdiff/_edrscc/cl123_preds/IPNet_scratch_cl123_seed*.csv", "pred", "y"),
}


def metrics(pred, keep):
    pids = [p for p in pred if p in keep]
    if len(pids) < 3:
        return None
    y = np.array([pred[p][1] for p in pids])
    yh = np.array([pred[p][0] for p in pids])
    return dict(n=len(pids), pearson=float(pearsonr(yh, y)[0]),
                spearman=float(spearmanr(yh, y).statistic),
                rmse=float(np.sqrt(((yh - y) ** 2).mean())))


def main():
    res = json.load(open(f"{D}/results.json"))
    for m, loader in METHODS.items():
        seeds = loader()
        if not seeds:
            print(f"  {m}: NO preds found"); continue
        for cohort, keep in COHORTS.items():
            per = [x for x in (metrics(s, keep) for s in seeds) if x]
            if not per:
                continue
            agg = {"n": per[0]["n"], "seeds": len(per), "per_seed": per}
            for k in ("pearson", "spearman", "rmse"):
                v = [abs(x[k]) if (m in ABS_METHODS and k != "rmse") else x[k] for x in per]
                agg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
            res["methods"].setdefault(m, {})[cohort] = agg
        c60 = res["methods"][m]["cl123_test_novel60"]
        print(f"  {m}: {len(seeds)} seeds | novel60 rho={c60['spearman']['mean']:.3f} n={c60['n']} "
              f"| novel30 rho={res['methods'][m]['cl123_test_novel30']['spearman']['mean']:.3f}")
    json.dump(res, open(f"{D}/results.json", "w"), indent=1)
    print(f"\nwrote {D}/results.json ({len(res['methods'])} methods)")


if __name__ == "__main__":
    main()

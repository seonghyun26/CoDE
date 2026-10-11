"""rescore_casf_cohorts.py — recompute all 5 CASF-2016 cohorts (leaky/nontrain/clean/id60/id30)
from 5-seed per-complex CASF predictions and write them where Table 1c reads:
  - leaky/nontrain/clean -> base/_casf/{key}.json          (load_casf)
  - id60/id30            -> base/_casf/casf_similarity/{safe}__casf_clean_cl3_novel{60,30}.json
                                                            (load_casf_similarity)
Currently wired for the GET-family 5-seed CASF inference (preds_{M}_casf5_seed*.jsonl).
"""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASF = f"{REPO}/base/_casf"
SIM = f"{CASF}/casf_similarity"


def ids(path):
    return set(l.strip().lower() for l in open(path) if l.strip())


EVAL = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
NONTRAIN = {r["pid"].lower() for r in EVAL if str(r.get("in_v2train", "")).strip() == "0"}
CLEAN = {r["pid"].lower() for r in csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_clean.csv"))}
ID60 = ids(f"{SIM}/casf_clean_cl3_novel60.txt") if os.path.exists(f"{SIM}/casf_clean_cl3_novel60.txt") \
    else ids(f"{CASF}/cl123_seqfilter_5seed_260818/../casf_similarity/casf_clean_cl3_novel60.txt")
ID30 = ids(f"{SIM}/casf_clean_cl3_novel30.txt")

# GET-family: display method -> (per-method json key, safe casf_similarity name, preds glob)
FAMILY = {
    "GET":               ("GET",     "GET",              f"{REPO}/base/get/_casf_get/preds/preds_GET_casf5_seed*.jsonl"),
    "EGNN":              ("EGNN",    "EGNN",             f"{REPO}/base/get/_casf_get/preds/preds_EGNN_casf5_seed*.jsonl"),
    "EGNN + TargetDiff": ("EGNN_TD", "EGNN___TargetDiff", f"{REPO}/base/get/_casf_get/preds/preds_EGNN_TD_casf5_seed*.jsonl"),
}


def load_seeds(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        d = {}
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[str(r["id"]).lower()] = (float(r["pred"]), float(r["gt"]))
        if d:
            out.append(d)
    return out


def cohort_metrics(seeds, keep):
    per = []
    for s in seeds:
        pids = [p for p in s if (keep is None or p in keep)]
        if len(pids) < 3:
            continue
        yh = np.array([s[p][0] for p in pids]); y = np.array([s[p][1] for p in pids])
        per.append(dict(n=len(pids), pearson=float(pearsonr(yh, y)[0]),
                        spearman=float(spearmanr(yh, y).statistic),
                        rmse=float(np.sqrt(((yh - y) ** 2).mean()))))
    if not per:
        return None
    agg = {"n": per[0]["n"], "seeds": len(per), "per_seed": per}
    for k in ("pearson", "spearman", "rmse"):
        v = [x[k] for x in per]
        agg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    return agg


def main():
    for method, (jkey, safe, pat) in FAMILY.items():
        seeds = load_seeds(pat)
        if len(seeds) < 5:
            print(f"  {method}: only {len(seeds)} seed preds — skip (need 5)"); continue
        # per-method json: leaky/nontrain/clean
        jpath = f"{CASF}/{jkey}.json"
        d = json.load(open(jpath)) if os.path.exists(jpath) else {"model": method}
        for name, keep in (("leaky", None), ("nontrain", NONTRAIN), ("clean", CLEAN)):
            m = cohort_metrics(seeds, keep)
            if m:
                d[name] = m
        json.dump(d, open(jpath, "w"), indent=2)
        # casf_similarity: id60/id30
        for cohort, keep in (("casf_clean_cl3_novel60", ID60), ("casf_clean_cl3_novel30", ID30)):
            m = cohort_metrics(seeds, keep)
            if not m:
                continue
            m2 = {"method": method, "split": cohort, "n_test": m["n"], "seeds": m["seeds"],
                  "per_seed": m["per_seed"], "pearson": m["pearson"], "spearman": m["spearman"],
                  "rmse": m["rmse"]}
            json.dump(m2, open(f"{SIM}/{safe}__{cohort}.json", "w"), indent=1)
        print(f"  {method}: 5-seed CASF -> leaky ρ={d['leaky']['spearman']['mean']:.3f} "
              f"nontrain {d['nontrain']['spearman']['mean']:.3f} clean {d['clean']['spearman']['mean']:.3f}")


if __name__ == "__main__":
    main()

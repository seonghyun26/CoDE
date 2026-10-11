"""aggregate_bindnet_casf_5seed.py — rescore 5-seed BindNet CASF from per-complex preds.

Reads base/_casf/BindNet_casf2016_preds_seed{0-4}.csv (pid,pred,y), masks into the 5 Table-1c
cohorts, aggregates mean/std over seeds, and writes where Table 1c reads:
  leaky/nontrain/clean -> base/_casf/BindNet.json                 (load_casf)
  id60/id30            -> base/_casf/casf_similarity/BindNet__casf_clean_cl3_novel{60,30}.json
                                                                   (load_casf_similarity)
"""
import csv, glob, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr

CASF = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(CASF))
SIM = f"{CASF}/casf_similarity"


def ids(path):
    return {l.strip().lower() for l in open(path) if l.strip()}


EVAL = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
LEAKY = {r["pid"].lower() for r in EVAL}
NONTRAIN = {r["pid"].lower() for r in EVAL if str(r.get("in_v2train", "")).strip() == "0"}
CLEAN = {r["pid"].lower() for r in csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_clean.csv"))}
ID60 = ids(f"{SIM}/casf_clean_cl3_novel60.txt")
ID30 = ids(f"{SIM}/casf_clean_cl3_novel30.txt")
COHORTS = {"leaky": LEAKY, "nontrain": NONTRAIN, "clean": CLEAN, "id60": ID60, "id30": ID30}


def load_seed_preds():
    out = []
    for f in sorted(glob.glob(f"{CASF}/BindNet_casf2016_preds_seed*.csv")):
        d = {r["pid"].lower(): (float(r["pred"]), float(r["y"])) for r in csv.DictReader(open(f))}
        if d:
            out.append(d)
    return out


def cohort_metrics(seeds, keep):
    per = []
    for s in seeds:
        pids = [p for p in s if p in keep]
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
    seeds = load_seed_preds()
    print(f"loaded {len(seeds)} seed pred files")
    # per-method json: leaky/nontrain/clean
    jp = f"{CASF}/BindNet.json"
    d = json.load(open(jp)) if os.path.exists(jp) else {"model": "BindNet"}
    d["seeds"] = list(range(len(seeds)))
    for name in ("leaky", "nontrain", "clean"):
        m = cohort_metrics(seeds, COHORTS[name])
        if m:
            d[name] = m
    json.dump(d, open(jp, "w"), indent=2)
    # casf_similarity: id60/id30
    for cohort, fname in (("id60", "casf_clean_cl3_novel60"), ("id30", "casf_clean_cl3_novel30")):
        m = cohort_metrics(seeds, COHORTS[cohort])
        if not m:
            continue
        out = {"method": "BindNet", "split": fname, "n_test": m["n"], "seeds": m["seeds"],
               "per_seed": m["per_seed"], "pearson": m["pearson"], "spearman": m["spearman"],
               "rmse": m["rmse"]}
        json.dump(out, open(f"{SIM}/BindNet__{fname}.json", "w"), indent=1)
    print("BindNet CASF 5-seed rho:", " ".join(
        f"{c}={d[c]['spearman']['mean']:.3f}(n{d[c]['n']},s{d[c]['seeds']})"
        for c in ("leaky", "nontrain", "clean") if c in d))
    for cohort, fname in (("id60", "casf_clean_cl3_novel60"), ("id30", "casf_clean_cl3_novel30")):
        p = f"{SIM}/BindNet__{fname}.json"
        if os.path.exists(p):
            j = json.load(open(p))
            print(f"  {cohort}: rho={j['spearman']['mean']:.3f} n={j['n_test']} seeds={j['seeds']}")


if __name__ == "__main__":
    main()

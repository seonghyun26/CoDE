"""Re-score GeoSSL's 5-seed CL3 test predictions on the protein-novelty cohorts and
inject a `methods.GeoSSL` block into the shared CL3-novelty results.json that
the results tables read for Table 1b (cl123_test / cl123_test_novel60 / cl123_test_novel30).

Idempotent: overwrites only the GeoSSL entry. Mirrors base/_casf/score_cl123_novel_subsets.py
(sorted pids, pearson/spearman/rmse per seed → mean±std).
"""
import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CO = os.path.join(REPO, "base", "_casf", "cl123_seqfilter_5seed_260818")
RESULTS_JSON = os.path.join(CO, "results.json")
PREDS = os.path.join(REPO, "base", "geossl", "results", "preds_GEOSSL_cl123_seed*.jsonl")

COHORTS = {
    "cl123_test":         "cl123_test.txt",
    "cl123_test_novel60": "cl123_test_novel60.txt",
    "cl123_test_novel30": "cl123_test_novel30.txt",
}


def load_ids(fn):
    return set(l.strip().lower() for l in open(os.path.join(CO, fn)) if l.strip())


def load_seeds():
    seeds = []
    for f in sorted(glob.glob(PREDS)):
        p = {}
        for line in open(f):
            d = json.loads(line)
            p[str(d["id"]).lower()] = (float(d["label"]), float(d["gt"]))
        seeds.append(p)
    return seeds


def per_seed_metrics(seed, pids):
    pairs = [seed[p] for p in sorted(pids) if p in seed]
    pred = np.array([a for a, _ in pairs])
    y = np.array([b for _, b in pairs])
    return {"n": len(pairs), "pearson": float(pearsonr(pred, y)[0]),
            "spearman": float(spearmanr(pred, y).statistic),
            "rmse": float(np.sqrt(((pred - y) ** 2).mean()))}


def agg(per, key):
    v = np.array([x[key] for x in per])
    return {"mean": float(v.mean()), "std": float(v.std())}


def main():
    seeds = load_seeds()
    assert seeds, "no GeoSSL cl123 preds found"
    block = {}
    for cohort, fn in COHORTS.items():
        pids = load_ids(fn)
        per = [per_seed_metrics(s, pids) for s in seeds]
        block[cohort] = {"n": per[0]["n"], "seeds": len(per), "per_seed": per,
                         "pearson": agg(per, "pearson"), "spearman": agg(per, "spearman"),
                         "rmse": agg(per, "rmse")}
        print(f"  {cohort:20s} n={per[0]['n']:4d} "
              f"r={block[cohort]['pearson']['mean']:.3f} "
              f"rho={block[cohort]['spearman']['mean']:.3f} "
              f"rmse={block[cohort]['rmse']['mean']:.3f}")

    doc = json.load(open(RESULTS_JSON))
    doc.setdefault("methods", {})["GeoSSL"] = block
    json.dump(doc, open(RESULTS_JSON, "w"), indent=2)
    print(f"injected methods.GeoSSL -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()

"""fill_casf_clean.py — add the CASF-2016 clean-92 cohort to the per-method CASF jsons that
only have leaky/nontrain (BindNet, DSMBind, HonestAffinity), computed from their per-complex
casf2016 predictions masked to the clean-92 pid list (casf2016_clean.csv). Matches the
{pearson/spearman/rmse:{mean,std}, n} structure the generator reads via load_casf(m,'clean').
"""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN = set(r["pid"].lower() for r in csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_clean.csv")))

# method json -> (per-seed casf preds glob, abs?)
TARGETS = {
    "BindNet.json":        (f"{REPO}/base/_casf/BindNet_casf2016_preds_seed*.csv", False),
    "DSMBind.json":        (f"{REPO}/base/_casf/DSMBind_casf2016_preds_seed*.csv", True),
    "HonestAffinity.json": (f"{REPO}/base/_casf/HonestAffinity_casf2016_preds_seed*.csv", False),
    # IPNet (IPDiff BAPNet) — frozen json name is IPNet_frozen, scratch json name is IPNet_retrain
    # but its preds CSVs are tagged "scratch". Both probe heads predict pK → positive corr (abs=False).
    "IPNet_frozen.json":   (f"{REPO}/base/_casf/IPNet_frozen_casf2016_preds_seed*.csv", False),
    "IPNet_retrain.json":  (f"{REPO}/base/_casf/IPNet_scratch_casf2016_preds_seed*.csv", False),
}


def metrics(path, do_abs):
    rows = list(csv.DictReader(open(path)))
    pc = next(c for c in rows[0] if c.lower() in ("pred", "score", "y_pred", "prediction"))
    yc = next(c for c in rows[0] if c.lower() in ("y", "pk", "label", "y_true", "truth"))
    pr = [(r["pid"].lower(), float(r[pc]), float(r[yc])) for r in rows
          if r["pid"].lower() in CLEAN and r[pc] not in ("", "nan", None)]
    if len(pr) < 3:
        return None
    yh = np.array([x[1] for x in pr]); y = np.array([x[2] for x in pr])
    r = float(pearsonr(yh, y)[0]); rho = float(spearmanr(yh, y).statistic)
    if do_abs:
        r, rho = abs(r), abs(rho)
    return dict(pearson=r, spearman=rho, rmse=float(np.sqrt(((yh - y) ** 2).mean())), n=len(pr))


def main():
    for jname, (pat, do_abs) in TARGETS.items():
        jpath = f"{REPO}/base/_casf/{jname}"
        d = json.load(open(jpath))
        per = [m for m in (metrics(f, do_abs) for f in sorted(glob.glob(pat))) if m]
        if not per:
            print(f"  {jname}: NO casf preds for clean"); continue
        clean = {"n": per[0]["n"]}
        for k in ("pearson", "spearman", "rmse"):
            v = [x[k] for x in per]
            clean[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        d["clean"] = clean
        json.dump(d, open(jpath, "w"), indent=2)
        print(f"  {jname}: clean-92 rho={clean['spearman']['mean']:.3f} n={clean['n']} "
              f"({len(per)} seeds)")


if __name__ == "__main__":
    main()

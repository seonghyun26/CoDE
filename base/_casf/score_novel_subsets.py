"""score_novel_subsets.py — re-score every baseline that has cached per-complex v2-test
predictions on the LP-PDBBind protein-novelty subsets, WITHOUT any retraining.

Instead of ATOM3D LBA30/60 (which need their own train/val/test → re-run each model),
we keep the SAME lp_edrscc_v2 model/features and only mask the TEST set by each test
protein's max sequence identity to ANY train protein (mmseqs test↔train, /tmp/seqsim):

  lp_test_novel60 = test proteins with max train seq-id < 0.60   (n=813, LBA60-like)
  lp_test_novel30 = test proteins with max train seq-id < 0.30   (n=453, LBA30-like)

For each method with per-pid v2 preds we recompute r / rho / rmse (mean±std over seeds)
on {full, novel60, novel30} and emit schema-A JSON at
  base/_casf/novel_subsets/{safe_method}__{split}.json
which the results tables read for the Table-1b columns. Idempotent.
"""
import csv, glob, json, os, re
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASF = f"{REPO}/base/_casf"
OUT  = f"{CASF}/novel_subsets"; os.makedirs(OUT, exist_ok=True)

def safe(m):
    return "".join(ch if ch.isalnum() else "_" for ch in m)

def load_ids(fn):
    return set(l.strip().lower() for l in open(fn) if l.strip())

NOVEL = {
    "lp_test_novel60": load_ids(f"{CASF}/lp_test_novel60.txt"),
    "lp_test_novel30": load_ids(f"{CASF}/lp_test_novel30.txt"),
}

# ── per-method per-pid prediction loaders → (list_of_seed_pred_dicts, gt_dict) ──
def _csv(path, pcol, ycol, idcol="pid"):
    rows = list(csv.DictReader(open(path)))
    pred = {r[idcol].lower(): float(r[pcol]) for r in rows if r.get(pcol) not in (None, "", "nan")}
    gt   = {r[idcol].lower(): float(r[ycol]) for r in rows if r.get(ycol) not in (None, "", "nan")}
    return pred, gt

def _jsonl(path, pkey, ykey):
    pred, gt = {}, {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line); pid = str(d["id"]).lower()
        pred[pid] = float(d[pkey]); gt[pid] = float(d[ykey])
    return pred, gt

def load_csv_seeds(glob_pat, pcol="pred", ycol="y"):
    seeds, gt = [], {}
    for f in sorted(glob.glob(glob_pat)):
        p, g = _csv(f, pcol, ycol); seeds.append(p); gt.update(g)
    return seeds, gt

def load_jsonl_seeds(glob_pat, pkey, ykey):
    seeds, gt = [], {}
    for f in sorted(glob.glob(glob_pat)):
        p, g = _jsonl(f, pkey, ykey); seeds.append(p); gt.update(g)
    return seeds, gt

def load_aev():
    rows = list(csv.DictReader(open(f"{REPO}/base/aevplig/results/aevplig_edrscc_preds.csv")))
    gt = {r["pid"].lower(): float(r["truth"]) for r in rows}
    seeds = [{r["pid"].lower(): float(r[f"pred_seed{s}"]) for r in rows} for s in range(3)]
    return seeds, gt

def load_hbgsa():
    seeds, gt = [], {}
    for f in sorted(glob.glob(f"{REPO}/base/hbgsa/results/preds_edrscc_v2_3p06m_seed*.json")):
        d = json.load(open(f))
        seeds.append({str(p).lower(): float(v) for p, v in zip(d["pdb_id"], d["pred"])})
        gt.update({str(p).lower(): float(v) for p, v in zip(d["pdb_id"], d["y"])})
    return seeds, gt

# method -> (loader, rank_only)   rank_only: report |rho| only (uncalibrated zero-shot score)
LOADERS = {
    "GET":               (lambda: load_jsonl_seeds(f"{REPO}/base/get/_edrscc/preds_GET_v2_seed*.jsonl", "label", "gt"), False),
    "EGNN":              (lambda: load_jsonl_seeds(f"{REPO}/base/get/_edrscc/preds_EGNN_v2_seed*.jsonl", "label", "gt"), False),
    "EGNN + TargetDiff": (lambda: load_jsonl_seeds(f"{REPO}/base/get/_edrscc/preds_EGNN_TD_v2_seed*.jsonl", "label", "gt"), False),
    "CheapNet":          (lambda: load_jsonl_seeds(f"{REPO}/base/cheapnet/_edrscc/preds_lp_edrscc_v2_seed*.jsonl", "pred", "label"), False),
    "AEV-PLIG":          (load_aev, False),
    "DSMBind":           (lambda: load_csv_seeds(f"{CASF}/novel_preds/DSMBind_v2_seed*.csv", "pred", "y"), False),
    "Nesso-1":           (lambda: load_csv_seeds(f"{REPO}/base/nesso/_edrscc/predictions_v2.csv", "pred_pIC50", "pK"), False),
    "DeepDTA":           (lambda: load_csv_seeds(f"{REPO}/base/dta/result/preds_DeepDTA_v2_seed*.csv", "pred", "y"), False),
    "MolTrans":          (lambda: load_csv_seeds(f"{REPO}/base/dta/result/preds_MolTrans_v2_seed*.csv", "pred", "y"), False),
    "C":                 (lambda: load_csv_seeds(f"{CASF}/novel_preds/C_v2_seed*.csv", "pred", "y"), False),
    "C+D+G":             (lambda: load_csv_seeds(f"{CASF}/novel_preds/C_D_G_v2_seed*.csv", "pred", "y"), False),
    "C+D+G +corr":       (lambda: load_csv_seeds(f"{CASF}/novel_preds/C_D_G__corr_v2_seed*.csv", "pred", "y"), False),
    "IPNet (frozen)":    (lambda: load_csv_seeds(f"{CASF}/novel_preds/IPNet_frozen_v2_seed*.csv", "pred", "y"), False),
    "IPNet (scratch)":   (lambda: load_csv_seeds(f"{CASF}/novel_preds/IPNet_scratch_v2_seed*.csv", "pred", "y"), False),
    "ProFSA":            (lambda: load_csv_seeds(f"{CASF}/novel_preds/ProFSA_v2_seed*.csv", "pred", "y"), False),
    "HBGSA":             (load_hbgsa, False),
    "HonestAffinity":    (lambda: load_csv_seeds(f"{CASF}/novel_preds/HonestAffinity_v2_seed*.csv", "pred", "y"), False),
    "BindNet":           (lambda: load_csv_seeds(f"{CASF}/novel_preds/BindNet_v2_seed*.csv", "pred", "y"), False),
}

def metrics_over(pred, gt, pids, rank_only):
    xs = [(pred[p], gt[p]) for p in pids if p in pred and p in gt]
    if len(xs) < 5:
        return None
    p = np.array([a for a, _ in xs]); y = np.array([b for _, b in xs])
    if np.std(p) == 0:
        return None
    rho = spearmanr(p, y).statistic
    if rank_only:
        return dict(n=len(xs), rho=abs(rho))
    return dict(n=len(xs), r=pearsonr(p, y)[0], rho=rho, rmse=float(np.sqrt(((p - y) ** 2).mean())))

def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (float(np.mean(vals)), float(np.std(vals)))

def emit(method, split, seed_preds, gt, subset_pids, rank_only):
    per = [metrics_over(sp, gt, subset_pids, rank_only) for sp in seed_preds]
    per = [x for x in per if x]
    if not per:
        return None
    n = per[0]["n"]
    out = {"method": method, "split": split, "n_test": n, "seeds": len(per)}
    if agg([x.get("r") for x in per]) is not None:
        m, s = agg([x["r"] for x in per]);   out["pearson"]  = {"mean": m, "std": s}
    m, s = agg([x["rho"] for x in per]);      out["spearman"] = {"mean": m, "std": s}
    if agg([x.get("rmse") for x in per]) is not None:
        m, s = agg([x["rmse"] for x in per]); out["rmse"]     = {"mean": m, "std": s}
    json.dump(out, open(f"{OUT}/{safe(method)}__{split}.json", "w"), indent=2)
    return out

def main():
    print(f"{'method':20} {'full':>22} {'novel60':>22} {'novel30':>22}")
    for method, (loader, rank_only) in LOADERS.items():
        try:
            seed_preds, gt = loader()
        except FileNotFoundError:
            continue
        if not seed_preds or not any(seed_preds):
            continue
        full_pids = set().union(*[set(sp) for sp in seed_preds]) & set(gt)
        cells = {}
        for split, subset in (("lp_edrscc_v2", full_pids),
                              ("lp_test_novel60", NOVEL["lp_test_novel60"]),
                              ("lp_test_novel30", NOVEL["lp_test_novel30"])):
            o = emit(method, split, seed_preds, gt, subset, rank_only)
            cells[split] = o
        def fmt(o):
            if not o:
                return f"{'—':>22}"
            rho = o["spearman"]["mean"]; return f"rho {rho:.3f} (n={o['n_test']})".rjust(22)
        print(f"{method:20} {fmt(cells['lp_edrscc_v2'])} "
              f"{fmt(cells['lp_test_novel60'])} {fmt(cells['lp_test_novel30'])}")

if __name__ == "__main__":
    main()

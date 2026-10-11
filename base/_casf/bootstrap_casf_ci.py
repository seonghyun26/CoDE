"""bootstrap_casf_ci.py — CASF-2016-style 90% CIs, multi-seed (Option B).

Two orthogonal uncertainties are reported for every method:
  - seed (training) variance  -> mean +/- std over the method's seeds        (as before)
  - test-set sampling variance -> 90% bootstrap CI, centered on the seed mean (this file)

CI recipe (faithful extension of CASF-2016 Su et al. JCIM 2019 sec 2.4 to a multi-seed model):
resample the test complexes with replacement (the paper's step); for that ONE shared resample,
compute the metric per seed and AVERAGE the per-seed metric; repeat 10,000x; form the 90% BCa
interval (bias-correction + leave-one-complex-out jackknife acceleration) — the SAME estimator
CASF-2016 uses (scoring_power.py: boot.ci type='bca'). So the CI is on the mean-over-seeds statistic
(matches the reported mean) and captures sampling only. We average the METRIC, not the predictions
(averaging predictions = an ensemble, which inflates r and de-syncs from mean+/-std). This is a
fixed-seed variant of the Multi-Bootstrap (Sellam et al., MultiBERTs, ICLR 2022).

  point = mean_s metric_s(full)   |   std = std_s metric_s(full)   |   ci90 = pct(mean_s metric_s(resample))

Ground truth y = canonical pK from voxbind/splits/casf2016_eval.csv (same for all methods).
Cohorts: leaky = all 214 | nontrain = in_v2train==0 (124) | clean = casf2016_clean.csv (92).
Output: base/_casf/casf_bootstrap_ci.json  keyed by CASF_KEY value ->
  {cohort: {n, seeds, pearson:{mean,std,ci90:[lo,hi]}, spearman:{...}, rmse:{...}}}
"""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import norm, rankdata

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASF = f"{REPO}/base/_casf"
N_BOOT = 10_000
CI = 90
RNG_SEED = 42

# ── canonical ground truth + cohort membership ────────────────────────────────
_EVAL = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
Y = {r["pid"].lower(): float(r["pK"]) for r in _EVAL}
LEAKY = set(Y)
NONTRAIN = {r["pid"].lower() for r in _EVAL if str(r.get("in_v2train", "")).strip() == "0"}
CLEAN = {r["pid"].lower() for r in csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_clean.csv"))}
COHORTS = {"leaky": LEAKY, "nontrain": NONTRAIN, "clean": CLEAN}


# ── per-format seed-file parsers → {pid: pred} ────────────────────────────────
def _csv_p(pred_col="pred", pid_col="pid"):
    return lambda p: {r[pid_col].lower(): float(r[pred_col]) for r in csv.DictReader(open(p))}


def _jsonl_p(pred_key="pred", id_key="id"):
    def f(path):
        out = {}
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[str(r[id_key]).lower()] = float(r[pred_key])
        return out
    return f


def _hbgsa_p():
    def f(path):
        d = json.load(open(path))
        return {p.lower(): float(v) for p, v in zip(d["pdb_id"], d["pred"])}
    return f


G = f"{REPO}/base/get/_casf_get/preds"
# key (the results table's CASF key) -> (glob over ALL seed files, parser)
REG = {
    "CDG_100m_mask075":       (f"{CASF}/CDG_100m_mask075_casf2016_preds_seed*.csv", _csv_p()),
    "C_100m_mask075_coords":  (f"{CASF}/C_100m_mask075_coords_casf2016_preds_seed*.csv", _csv_p()),
    "CDG_100m_mask075_corr5": (f"{CASF}/CDG_100m_mask075_corr5_casf2016_preds_seed*.csv", _csv_p()),
    "CDG_v2": (f"{CASF}/CDG_v2_casf2016_preds_seed*.csv", _csv_p()),
    "CDG_v3": (f"{CASF}/CDG_v3_casf2016_preds_seed*.csv", _csv_p()),
    "ProFSA":   (f"{REPO}/base/profsa/_casf/profsa_casf5_seed*.csv", _csv_p()),
    "CheapNet": (f"{REPO}/base/cheapnet/_edrscc/preds_casf_seed*.jsonl", _jsonl_p()),
    "GET":      (f"{G}/preds_GET_casf5_seed*.jsonl", _jsonl_p()),
    "EGNN":     (f"{G}/preds_EGNN_casf5_seed*.jsonl", _jsonl_p()),
    "EGNN_TD":  (f"{G}/preds_EGNN_TD_casf5_seed*.jsonl", _jsonl_p()),
    "HBGSA":    (f"{REPO}/base/hbgsa/results/preds_casf2016_hbgsa_3p06m_seed*.json", _hbgsa_p()),
    "GeoSSL":   (f"{REPO}/base/geossl/results/preds_GEOSSL_casf_core_seed*.jsonl", _jsonl_p(pred_key="label")),
    "DSMBind_zeroshot": (f"{REPO}/base/dsmbind/results/dsmbind_casf2016_zeroshot_preds.csv", _csv_p(pred_col="score")),
    "BindNet":  (f"{CASF}/BindNet_casf2016_preds_seed*.csv", _csv_p()),
    "DeepDTA":  (f"{CASF}/DeepDTA_casf2016_preds_seed*.csv", _csv_p()),
    "HonestAffinity": (f"{CASF}/HonestAffinity_casf2016_preds_seed*.csv", _csv_p()),
    "MolTrans": (f"{CASF}/MolTrans_casf2016_preds_seed*.csv", _csv_p()),
    "IPNet_frozen":  (f"{CASF}/IPNet_frozen_casf2016_preds_seed*.csv", _csv_p()),
    "IPNet_retrain": (f"{CASF}/IPNet_scratch_casf2016_preds_seed*.csv", _csv_p()),
    "AEV": (f"{CASF}/AEV_casf2016_preds_seed*.csv", _csv_p()),
    "Nesso": (f"{CASF}/Nesso_casf2016_preds_seed0.csv", _csv_p()),                    # single deterministic
}
# zero-shot energies are sign-arbitrary anti-correlations → report |r|,|rho|
ABS_CORR = {"DSMBind_zeroshot"}


def load_seeds(pattern, parser):
    return [parser(p) for p in sorted(glob.glob(pattern))]


def _metric_rows(P, y):
    """P: (S,n) per-seed preds; y: (n,). Returns (pear, spear, rmse) each (S,) — vectorized over seeds."""
    Pc = P - P.mean(1, keepdims=True); yc = y - y.mean()
    pear = (Pc * yc).sum(1) / (np.sqrt((Pc ** 2).sum(1)) * np.sqrt((yc ** 2).sum()) + 1e-12)
    rmse = np.sqrt(((P - y) ** 2).mean(1))
    ry = rankdata(y); RP = np.vstack([rankdata(P[s]) for s in range(P.shape[0])])
    Rc = RP - RP.mean(1, keepdims=True); ryc = ry - ry.mean()
    spear = (Rc * ryc).sum(1) / (np.sqrt((Rc ** 2).sum(1)) * np.sqrt((ryc ** 2).sum()) + 1e-12)
    return pear, spear, rmse


def _bca_bounds(theta, boot, jack, ci=CI):
    """BCa interval (Efron) — matches CASF-2016 scoring_power.py boot.ci(type='bca'). theta=observed
    statistic, boot=bootstrap replicates, jack=leave-one-complex-out jackknife values."""
    boot = np.asarray(boot); jack = np.asarray(jack)
    z0 = norm.ppf(np.clip(np.mean(boot < theta), 1e-6, 1 - 1e-6))          # bias correction
    d = jack.mean() - jack
    den = 6.0 * (d ** 2).sum() ** 1.5
    a = (d ** 3).sum() / den if den > 0 else 0.0                          # acceleration (over complexes)
    out = []
    for q in ((100 - ci) / 2 / 100, 1 - (100 - ci) / 2 / 100):
        za = norm.ppf(q)
        p = norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
        out.append(float(np.percentile(boot, 100 * np.clip(p, 1e-6, 1 - 1e-6))))
    return out


def cohort_ci(seeds, keep, take_abs):
    """seeds: list of {pid:pred}. Returns per-metric {mean,std,ci90} or None. mean/std over seeds;
    CI = BCa bootstrap of the mean-over-seeds metric (resample complexes, average per-seed metric)."""
    common = set(Y) & keep
    for sp in seeds:
        common &= set(sp)
    pids = sorted(common)
    if len(pids) < 3:
        return None
    y = np.array([Y[p] for p in pids])
    P = np.array([[sp[p] for p in pids] for sp in seeds])      # (S, n)
    S, n = P.shape

    def mean_metrics(Pm, yv):
        pe, sp, rm = _metric_rows(Pm, yv)
        if take_abs:
            pe, sp = np.abs(pe), np.abs(sp)
        return pe, sp, rm

    pear, spear, rmse = mean_metrics(P, y)                     # per-seed full-data metrics (S,)
    point = {"pearson": pear, "spearman": spear, "rmse": rmse}

    rng = np.random.default_rng(RNG_SEED)
    bp = np.empty(N_BOOT); bs = np.empty(N_BOOT); br = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        pe, sp, rm = mean_metrics(P[:, idx], y[idx])
        bp[b], bs[b], br[b] = pe.mean(), sp.mean(), rm.mean()    # AVERAGE the per-seed metric

    jp = np.empty(n); js = np.empty(n); jr = np.empty(n)         # leave-one-complex-out jackknife
    for i in range(n):
        m = np.arange(n) != i
        pe, sp, rm = mean_metrics(P[:, m], y[m])
        jp[i], js[i], jr[i] = pe.mean(), sp.mean(), rm.mean()

    def block(pt, dist, jack):
        theta = float(pt.mean())
        lo, hi = _bca_bounds(theta, dist, jack)
        return {"mean": round(theta, 4), "std": round(float(pt.std()), 4),
                "ci90": [round(lo, 4), round(hi, 4)]}
    return {"n": n, "seeds": S,
            "pearson": block(point["pearson"], bp, jp),
            "spearman": block(point["spearman"], bs, js),
            "rmse": block(point["rmse"], br, jr)}


def main():
    out = {"_meta": {"method": "multi-seed BCa: resample complexes, average per-seed metric, BCa CI "
                              "(bias-corr + complex-jackknife accel; matches CASF-2016 scoring_power.py)",
                     "n_boot": N_BOOT, "ci": CI, "rng": RNG_SEED,
                     "y_source": "voxbind/splits/casf2016_eval.csv pK"}}
    print(f"{'method':24s} {'cohort':9s} {'S':>2s} {'n':>4s}  {'r(mean±std)':>14s}   ci90")
    for key, (pattern, parser) in REG.items():
        seeds = load_seeds(pattern, parser)
        if not seeds:
            print(f"{key:24s}  -- no seed files ({os.path.basename(pattern)}), skip"); continue
        out[key] = {}
        for cname, keep in COHORTS.items():
            res = cohort_ci(seeds, keep, key in ABS_CORR)
            if res:
                out[key][cname] = res
                rr = res["pearson"]
                print(f"{key:24s} {cname:9s} {res['seeds']:>2d} {res['n']:>4d}  "
                      f"{rr['mean']:.3f}±{rr['std']:.3f}   [{rr['ci90'][0]:.3f}, {rr['ci90'][1]:.3f}]")
    dst = f"{CASF}/casf_bootstrap_ci.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n[write] {dst}  ({len(out) - 1} methods)")


if __name__ == "__main__":
    main()

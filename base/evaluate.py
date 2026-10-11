#!/usr/bin/env python3
"""Score the baselines' per-complex predictions on the affinity test sets.

    python base/evaluate.py                       # every baseline with predictions on disk
    python base/evaluate.py --methods get,egnn    # some
    python base/evaluate.py --list                # where each baseline's predictions are read

A baseline's train_eval stage leaves its predictions in that baseline's own format and folder;
sources() below says where, for each model and seed. They are copied to
base/results/preds/<method>/<split>_<test|casf>_seed<N>.csv (pid, pred), joined to our labels by
PDB id (LP-PDBBind pK, which CASF-2016 shares) and scored on the test sets the encoders are scored on:

  model trained on     test set        complexes   pids (voxbind/splits/)
  lp_edrscc_v2         FULL            1,320       the split's test set
                       CASF-nontrain     124       casf2016_nontrain_cohort.csv
                       CASF-clean         92       casf2016_clean_cohort.csv
  lp_edrscc_v2_cl123   CL3               733       the split's test set
                       CL3-ID60          454       cl123_test_novel60_cohort.csv
                       CL3-ID30          262       cl123_test_novel30_cohort.csv

base/results/<method>.csv holds one row per (split, test set, seed): n, pearson, spearman, rmse,
and the prediction file it came from. Another split (--splits) is scored on its own test set.
DSMBind predicts an energy whose sign is arbitrary, so its r and rho are reported as magnitudes
and its RMSE is left empty. Needs only numpy and scipy, so any baseline env runs it.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
SPLITS = REPO / "voxbind" / "splits"
LABELS = REPO / "dataset" / "data" / "pdbbind" / "raw" / "LP_PDBBind.csv"
RESULTS = BASE / "results"
CASF = BASE / "_casf"
SEEDS = range(10)

# Test sets per training split: name -> pid file (None = the split's own test set). Any other
# split is scored on its own test set.
TEST_SETS = {
    "lp_edrscc_v2": {"test": [("FULL", None)],
                     "casf": [("CASF-nontrain", "casf2016_nontrain_cohort.csv"),
                              ("CASF-clean", "casf2016_clean_cohort.csv")]},
    "lp_edrscc_v2_cl123": {"test": [("CL3", None),
                                    ("CL3-ID60", "cl123_test_novel60_cohort.csv"),
                                    ("CL3-ID30", "cl123_test_novel30_cohort.csv")]},
}


def test_sets(split: str) -> dict:
    return TEST_SETS.get(split, {"test": [("test", None)], "casf": []})


# ── readers: path (with {seed}) -> {pid: prediction} ──────────────────────────────────────
def per_seed_csv(pred: str = "pred", pid: str = "pid"):
    def read(path: Path) -> dict:
        with open(path) as fh:
            return {r[pid].lower(): float(r[pred]) for r in csv.DictReader(fh) if r[pred] not in ("", "nan")}
    return read


def per_seed_jsonl(pred: str, pid: str = "id"):
    def read(path: Path) -> dict:
        out = {}
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    out[str(d[pid]).lower()] = float(d[pred])
        return out
    return read


def hbgsa_json(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {str(p).lower(): float(v) for p, v in zip(d["pdb_id"], d["pred"])}


class Wide:
    """One file holding every seed as a column (pred_seed0, pred_seed1, ...)."""

    def __init__(self, path: Path, column: str = "pred_seed{seed}"):
        self.path, self.column = path, column

    def seeds(self) -> list:
        if not self.path.exists():
            return []
        with open(self.path) as fh:
            header = next(csv.reader(fh))
        pat = re.compile("^" + re.escape(self.column).replace(r"\{seed\}", r"(\d+)") + "$")
        return sorted(int(m.group(1)) for h in header if (m := pat.match(h)))

    def read(self, seed: int) -> dict:
        col = self.column.format(seed=seed)
        with open(self.path) as fh:
            return {r["pid"].lower(): float(r[col]) for r in csv.DictReader(fh) if r[col] not in ("", "nan")}


# ── where each baseline's predictions are ─────────────────────────────────────────────────
B, V2, CL3 = BASE, "lp_edrscc_v2", "lp_edrscc_v2_cl123"
METHODS = ["aevplig", "bindnet", "cheapnet", "dsmbind", "deepdta", "moltrans", "get", "egnn", "egnn_td",
           "td_egnn", "td_egnn_enc", "geossl", "hbgsa", "honestaffinity", "ipnet_frozen", "ipnet_scratch",
           "nesso", "profsa"]
GET_NAME = {"get": "GET", "egnn": "EGNN", "egnn_td": "EGNN_TD"}
TD_NAME = {"td_egnn": "egnn", "td_egnn_enc": "egnn_enc"}          # base/scripts/targetdiff.sh MODELS
DTA_NAME = {"deepdta": "DeepDTA", "moltrans": "MolTrans"}
SIGN_FREE = {"dsmbind"}           # report |r| and |rho|, no RMSE


def short(split: str) -> str:
    """lp_edrscc_v2 -> v2, lp_edrscc_v2_cl123 -> cl123, anything else unchanged."""
    return "v2" if split == V2 else split.removeprefix(V2 + "_")


def sources(m: str, split: str, kind: str) -> list:
    """Candidate (path with {seed} | Wide, reader) for a method's predictions on a split's test
    set ("test") or, for a model trained on lp_edrscc_v2, on CASF-2016 ("casf"). The first one
    with files on disk is read: where train_eval writes now, then where earlier runs left them."""
    sh = short(split)
    csv_, jsonl = per_seed_csv, per_seed_jsonl
    if kind == "casf":
        if split != V2:
            return []
        if m in GET_NAME:
            return [(B / f"get/_casf_get/preds/preds_{GET_NAME[m]}_casf5_seed{{seed}}.jsonl", jsonl("pred"))]
        if m in TD_NAME:
            return [(B / f"targetdiff/preds/casf2016/{TD_NAME[m]}_seed{{seed}}.csv", csv_())]
        if m in DTA_NAME:
            return [(CASF / f"{DTA_NAME[m]}_casf2016_preds_seed{{seed}}.csv", csv_())]
        return {
            "aevplig": [(CASF / "AEV_casf2016_preds_seed{seed}.csv", csv_())],
            "bindnet": [(CASF / "BindNet_casf2016_preds_seed{seed}.csv", csv_())],
            "cheapnet": [(B / "cheapnet/_edrscc/preds_casf_seed{seed}.jsonl", jsonl("pred"))],
            "dsmbind": [(B / "dsmbind/results/dsmbind_casf2016_zeroshot_preds.csv", csv_("score"))],
            "geossl": [(B / "geossl/results/preds_GEOSSL_casf_core_seed{seed}.jsonl", jsonl("label"))],
            "hbgsa": [(B / "hbgsa/results/preds_casf2016_hbgsa_3p06m_seed{seed}.json", hbgsa_json)],
            "honestaffinity": [(CASF / "HonestAffinity_casf2016_preds_seed{seed}.csv", csv_())],
            "ipnet_frozen": [(CASF / "IPNet_frozen_casf2016_preds_seed{seed}.csv", csv_())],
            "ipnet_scratch": [(CASF / "IPNet_scratch_casf2016_preds_seed{seed}.csv", csv_())],
            "nesso": [(CASF / "Nesso_casf2016_preds_seed{seed}.csv", csv_())],
            "profsa": [(B / "profsa/_casf/profsa_casf5_seed{seed}.csv", csv_())],
        }[m]
    if m in GET_NAME:
        return [(B / f"get/_edrscc/preds_{GET_NAME[m]}_{sh}_seed{{seed}}.jsonl", jsonl("label"))]
    if m in TD_NAME:
        return [(B / f"targetdiff/preds/{split}/{TD_NAME[m]}_seed{{seed}}.csv", csv_())]
    if m in DTA_NAME:
        return [(B / f"dta/result/preds_{DTA_NAME[m]}_{'v2' if split == V2 else split}_seed{{seed}}.csv", csv_())]
    if m in ("ipnet_frozen", "ipnet_scratch"):
        v = m.split("_")[1]
        legacy = {V2: CASF / f"novel_preds/IPNet_{v}_v2_seed{{seed}}.csv",
                  CL3: B / f"ipdiff/_edrscc/cl123_preds/IPNet_{v}_cl123_seed{{seed}}.csv"}
        return [(B / f"ipdiff/preds/{v}/{split}/IPNet_{v}_seed{{seed}}.csv", csv_())] + \
               ([(legacy[split], csv_())] if split in legacy else [])
    if m == "bindnet":
        return [(B / f"bindnet/preds/{split}/BindNet_seed{{seed}}.csv", csv_()),
                (CASF / f"novel_preds/BindNet_{sh}_seed{{seed}}.csv", csv_())]
    if m == "honestaffinity":   # train.py --dump_dir names every split's file "v2"
        return [(B / f"honestaffinity/preds/{split}/HonestAffinity_v2_seed{{seed}}.csv", csv_())] + \
               ([(CASF / "novel_preds/HonestAffinity_v2_seed{seed}.csv", csv_())] if split == V2 else [])
    if m == "profsa":           # lba.py names every split's dump "ProFSA_v2"
        legacy = {V2: CASF / "novel_preds/ProFSA_v2_seed{seed}.csv",
                  CL3: CASF / "cl123_seqfilter_5seed_260818/preds/profsa_fresh/ProFSA_v2_seed{seed}.csv"}
        return [(B / f"profsa/preds/{split}/ProFSA_v2_seed{{seed}}.csv", csv_())] + \
               ([(legacy[split], csv_())] if split in legacy else [])
    return {
        "aevplig": [(Wide(B / f"aevplig/results/aevplig_{'edrscc' if split == V2 else split}_preds.csv"), None)],
        "cheapnet": [(B / f"cheapnet/_edrscc/preds_{split}_seed{{seed}}.jsonl", jsonl("pred"))],
        # zero-shot: one pass over the lp_edrscc_v2 test set serves every split inside it
        "dsmbind": [(B / "dsmbind/results/dsmbind_edrscc_preds.csv", csv_("score"))],
        "geossl": [(B / f"geossl/results/preds_GEOSSL_{sh}_seed{{seed}}.jsonl", jsonl("label"))],
        "hbgsa": [(B / f"hbgsa/results/preds_{'edrscc_v2' if split == V2 else 'edrscc_v2_' + sh}_3p06m_seed{{seed}}.json",
                   hbgsa_json)],
        "nesso": [(B / "nesso/_edrscc/predictions_v2.csv", csv_("pred_pIC50"))],
    }[m]


def read_predictions(source, reader) -> dict:
    """{seed: {pid: pred}} for one (method, split, kind). A path without {seed} is seed 0."""
    if isinstance(source, Wide):
        return {s: source.read(s) for s in source.seeds()}
    if "{seed}" not in str(source):
        return {0: reader(source)} if source.exists() else {}
    out = {}
    for s in SEEDS:
        path = Path(str(source).format(seed=s))
        if path.exists():
            out[s] = reader(path)
    return out


# ── labels and test sets ──────────────────────────────────────────────────────────────────
def load_labels() -> dict:
    with open(LABELS) as fh:
        rows = csv.reader(fh)
        header = next(rows)
        col = header.index("value")
        return {r[0].lower(): float(r[col]) for r in rows if r[col] not in ("", "nan")}


def pid_set(name: str, split: str) -> set:
    path = SPLITS / (name if name else f"{split}.csv")
    with open(path) as fh:
        return {r["pid"].lower() for r in csv.DictReader(fh) if name or r["split"] == "test"}


def score(pred: dict, labels: dict, keep: set, sign_free: bool) -> dict:
    pids = sorted(p for p in keep if p in pred and p in labels)
    if len(pids) < 3:
        return {}
    y = np.array([labels[p] for p in pids])
    x = np.array([pred[p] for p in pids])
    r, rho = pearsonr(x, y)[0], spearmanr(x, y)[0]
    if sign_free:
        return {"n": len(pids), "pearson": abs(r), "spearman": abs(rho), "rmse": ""}
    return {"n": len(pids), "pearson": r, "spearman": rho, "rmse": float(np.sqrt(((x - y) ** 2).mean()))}


def evaluate(method: str, labels: dict, splits: list) -> list:
    rows = []
    for split in splits:
        for kind in ("test", "casf"):
            for source, reader in sources(method, split, kind):
                preds = read_predictions(source, reader)
                if preds:
                    break
            else:
                continue
            src = str(Path(source.path if isinstance(source, Wide) else source).relative_to(REPO))
            out_dir = RESULTS / "preds" / method
            out_dir.mkdir(parents=True, exist_ok=True)
            for seed, pred in sorted(preds.items()):
                with open(out_dir / f"{split}_{kind}_seed{seed}.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["pid", "pred"])
                    w.writerows(sorted(pred.items()))
                for test_set, pid_file in test_sets(split)[kind]:
                    m = score(pred, labels, pid_set(pid_file, split), method in SIGN_FREE)
                    if m:
                        rows.append({"method": method, "split": split, "test_set": test_set,
                                     "seed": seed, **m, "source": src})
    return rows


def write_csv(method: str, rows: list) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{method}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "split", "test_set", "seed", "n", "pearson", "spearman",
                                           "rmse", "source"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def summary(rows: list) -> str:
    order = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
    lines = []
    for t in order:
        sub = [r for r in rows if r["test_set"] == t]
        if not sub:
            continue
        fmt = lambda k: (f"{np.mean([r[k] for r in sub]):.3f}±{np.std([r[k] for r in sub]):.3f}"
                         if sub[0][k] != "" else "  n/a  ")
        lines.append(f"    {t:<14}n={sub[0]['n']:<5} seeds={len(sub)}  r {fmt('pearson')}  "
                     f"rho {fmt('spearman')}  RMSE {fmt('rmse')}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="", help="comma-separated (default: all)")
    ap.add_argument("--splits", default=f"{V2},{CL3}", help="training splits to score")
    ap.add_argument("--list", action="store_true", help="print where each method's predictions are read")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or METHODS
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        ap.error(f"unknown method(s) {unknown}; choose from {METHODS}")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.list:
        for m in methods:
            print(m)
            for split in splits:
                for kind in ("test", "casf"):
                    for i, (source, _) in enumerate(sources(m, split, kind)):
                        src = Path(source.path if isinstance(source, Wide) else source).relative_to(REPO)
                        print(f"  {split:<20} {kind:<5} {'' if i == 0 else '(earlier runs) '}{src}")
        return 0

    labels = load_labels()
    for m in methods:
        rows = evaluate(m, labels, splits)
        if not rows:
            print(f"{m}: no predictions found")
            continue
        path = write_csv(m, rows)
        print(f"{m} -> {path.relative_to(REPO)}\n{summary(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

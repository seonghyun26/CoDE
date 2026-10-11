"""build_dta_splits.py — build (pid,seq,smiles,pK) train/val/test CSVs for DTA baselines
on any frozen split, by re-bucketing LP_PDBBind per the split CSV.

    python build_dta_splits.py --splits lp_edrscc_v2_cl123 [...]

Writes base/data/dta/{split}_{train,val,test}.csv. v2 is built by make_data.py (as v2_*.csv).
"""
import argparse, os, csv, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # base/
from datasource import LP_CSV as LP, SPLITS_DIR as SPL, data_dir                  # noqa: E402
DATA = data_dir("dta")

lp = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pid"})
lp["pid"] = lp["pid"].astype(str).str.lower()
info = {r.pid: (r.seq, r.smiles, r.value) for r in lp.itertuples()
        if isinstance(getattr(r, "seq", None), str) and isinstance(getattr(r, "smiles", None), str)
        and pd.notna(r.value)}

# split -> its (train, val, test) source CSVs; a split not listed uses its own CSV for all three
SOURCES = {
    "clean_ed_v1_indep": ("clean_ed_v1", "clean_ed_v1", "clean_ed_v1_indep"),   # train/val from full CleanSplit, test = indep-109
}
ap = argparse.ArgumentParser()
ap.add_argument("--splits", nargs="+", default=["lp_edrscc_v2_cl123"])
SPLITS = {s: SOURCES.get(s, (s,) * 3) for s in ap.parse_args().splits}
BMAP = {"train": "train", "val": "val", "valid": "val", "test": "test"}

def pids_of(split_csv, bucket):
    df = pd.read_csv(f"{SPL}/{split_csv}.csv")
    return [str(p).lower() for p, s in zip(df["pid"], df["split"]) if BMAP.get(s) == bucket]

for name, (tr_csv, va_csv, te_csv) in SPLITS.items():
    for bucket, src in (("train", tr_csv), ("val", va_csv), ("test", te_csv)):
        rows = [(p, *info[p]) for p in pids_of(src, bucket) if p in info]
        out = f"{DATA}/{name}_{bucket}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pid", "seq", "smiles", "pK"]); w.writerows(rows)
    n = {b: sum(1 for _ in open(f"{DATA}/{name}_{b}.csv")) - 1 for b in ("train", "val", "test")}
    print(f"{name}: {n}")
print("done")

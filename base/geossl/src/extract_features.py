"""Extract frozen GeoSSL-DDM SchNet features for every complex in the target splits.

Features depend only on the complex (not on the split), so we encode each unique pid
ONCE across the union of all requested split CSVs and cache pid -> 128-d vector. Later,
each split's probe just looks up these cached vectors.

Usage:
    python src/extract_features.py --splits atom3d_lba60_edrscc_v2 atom3d_lba30_edrscc_v2 ...
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from complex_graph import build_complex_graph
from schnet import load_pretrained_schnet

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                                    # base/geossl
sys.path.insert(0, os.path.dirname(BASE))                       # base/, for datasource
from datasource import SPLITS_DIR, STRUCT_BASES as STRUCT_ROOTS, data_dir  # noqa: E402
WEIGHTS = os.path.join(BASE, "weights", "geossl_ddm_schnet.pth")
FEAT_PT = os.path.join(data_dir("geossl"), "schnet_feats.pt")   # base/data/geossl


def resolve(pid):
    for r in STRUCT_ROOTS:
        d = os.path.join(r, pid)
        prot = os.path.join(d, f"{pid}_protein.pdb")
        if os.path.exists(prot):
            return prot, os.path.join(d, f"{pid}_ligand.mol2"), os.path.join(d, f"{pid}_ligand.sdf")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # union of pids across requested splits
    pids = []
    for s in args.splits:
        csv = os.path.join(SPLITS_DIR, f"{s}.csv")
        pids += pd.read_csv(csv)["pid"].astype(str).tolist()
    pids = sorted(set(pids))
    print(f"[extract] {len(args.splits)} splits -> {len(pids)} unique pids")

    cache = {}
    if os.path.exists(FEAT_PT):
        cache = torch.load(FEAT_PT, weights_only=False)
        print(f"[extract] resuming from cache with {len(cache)} feats")

    model = load_pretrained_schnet(WEIGHTS, device=device)

    todo = [p for p in pids if p not in cache]
    print(f"[extract] {len(todo)} to encode")
    fails = []
    t0 = time.time()
    for i, pid in enumerate(todo):
        r = resolve(pid)
        if r is None:
            fails.append((pid, "no_struct")); continue
        g = build_complex_graph(*r)
        if g is None:
            fails.append((pid, "build_fail")); continue
        z, pos = g
        with torch.no_grad():
            feat = model(torch.from_numpy(z).to(device), torch.from_numpy(pos).to(device))
        cache[pid] = feat.squeeze(0).float().cpu().numpy()
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)}  ({(i+1)/el:.1f}/s)  fails={len(fails)}")
            torch.save(cache, FEAT_PT)

    torch.save(cache, FEAT_PT)
    print(f"[extract] done: {len(cache)} feats, {len(fails)} fails -> {FEAT_PT}")
    if fails:
        for pid, why in fails[:20]:
            print("   FAIL", pid, why)
        pd.DataFrame(fails, columns=["pid", "why"]).to_csv(
            os.path.join(data_dir("geossl"), "extract_fails.csv"), index=False)


if __name__ == "__main__":
    main()

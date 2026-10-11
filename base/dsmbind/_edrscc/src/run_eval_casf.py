"""run_eval_casf.py — zero-shot DSMBind (方式1) binding-energy dump on the CASF-2016 core.

Mirror of run_eval.py, but iterates voxbind/splits/casf2016_eval.csv instead of the
lp_edrscc_v2 TEST split. Emits a per-complex energy CSV (pid,pK,energy,score=-energy).
NO training, NO seeds — one deterministic forward of the released drug all-atom model.

Cohort scoring (leaky / nontrain / clean-92 and |ρ| with RMSE=n/a) is done separately,
on CPU, by build_zeroshot_json.py so the GPU pass only has to run once.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
DSM_ROOT = os.path.dirname(os.path.dirname(HERE))           # base/dsmbind/
sys.path.insert(0, os.path.dirname(DSM_ROOT))              # base/, for datasource
from datasource import CASF_CSV as _CASF_CSV, LP_CSV, STRUCT_BASES, data_dir, split_csv  # noqa: E402
DSM_DATA = data_dir("dsmbind")                             # base/data/dsmbind: features/<split>.pt
REPO = os.path.dirname(os.path.dirname(DSM_ROOT))           # VoxBind/
sys.path.insert(0, HERE)
sys.path.insert(0, DSM_ROOT)

from rdkit import RDLogger                                  # noqa: E402
RDLogger.DisableLog("rdApp.*")
from prody import confProDy                                 # noqa: E402
confProDy(verbosity="none")
# extract_features shares run_eval.py's data path (pocket.pdb target, ESM-2-3B, DrugDataset).
from extract_features import (CKPT, DrugAllAtomEnergyModel, DrugDataset,   # noqa: E402
                              build_entry, load_esm_embedding)

CASF_CSV = _CASF_CSV
OUT_CSV = os.path.join(DSM_ROOT, "results", "dsmbind_casf2016_zeroshot_preds.csv")
OUT_FAILS = os.path.join(DSM_ROOT, "results", "dsmbind_casf2016_zeroshot_fails.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    frame = pd.read_csv(CASF_CSV)
    frame["pid"] = frame["pid"].astype(str).str.lower()
    pk = dict(zip(frame["pid"], frame["pK"]))
    pids = frame["pid"].tolist()
    if args.limit:
        pids = pids[:args.limit]

    print(f"loading DSMBind drug model from {CKPT}", flush=True)
    model_ckpt, _, model_args = torch.load(CKPT, map_location="cpu")
    model = DrugAllAtomEnergyModel(model_args).cuda()
    model.load_state_dict(model_ckpt)
    model.eval()

    entries, fails = [], {}
    for pid in tqdm(pids, desc="build entries"):
        if pid not in pk or not np.isfinite(pk[pid]):
            fails[pid] = "no_label"; continue
        e, status = build_entry(pid, pk[pid])
        if e is None:
            fails[pid] = status
        else:
            entries.append(e)
    print(f"built {len(entries)} entries | failures {len(fails)}: "
          f"{dict(pd.Series(list(fails.values())).value_counts()) if fails else {}}", flush=True)

    print("computing ESM-2-3B target embeddings (pocket sequences)...", flush=True)
    embedding = load_esm_embedding(entries, ["target_seq"])
    proc = DrugDataset.process(entries, model_args.patch_size)

    rows = []
    with torch.no_grad():
        for entry in tqdm(proc, desc="predict energy"):
            try:
                binder, target = DrugDataset.make_bind_batch([entry], embedding, model_args)
                energy = model.predict(binder, target).item()
            except Exception as e:
                fails[entry["pdb"]] = f"predict:{type(e).__name__}"; continue
            rows.append((entry["pdb"], entry["affinity"], energy, -energy))

    df = pd.DataFrame(rows, columns=["pid", "pK", "energy", "score"])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_FAILS, "w") as f:
        json.dump(fails, f, indent=2)
    print(f"\n===== DSMBind zero-shot CASF-2016 energy dump =====", flush=True)
    print(f"scored {len(df)} / {len(pids)}  | failed {len(fails)}  -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()

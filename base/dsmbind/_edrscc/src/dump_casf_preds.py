"""Dump full-CASF predictions for the LP-trained DSMBind frozen-feature probe.

The LP feature cache already contains most CASF complexes.  The few CASF structures
outside ``lp_edrscc_v2`` are encoded with the same released DSMBind checkpoint, then
the standard three seeded MLP heads are fit on LP train/validation and evaluated on
the full evaluable CASF set.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DSM_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(DSM_ROOT))              # base/, for datasource
from datasource import CASF_CSV as _CASF_CSV, LP_CSV, STRUCT_BASES, data_dir, split_csv  # noqa: E402
DSM_DATA = data_dir("dsmbind")                             # base/data/dsmbind: features/<split>.pt
REPO = os.path.dirname(os.path.dirname(DSM_ROOT))
sys.path.insert(0, HERE)
sys.path.insert(0, DSM_ROOT)

from extract_features import (CKPT, DrugAllAtomEnergyModel, DrugDataset,  # noqa: E402
                              build_entry, load_esm_embedding, pool_features)
from probe_mlp import train_one  # noqa: E402

LP_FEATURES = os.path.join(DSM_DATA, "features", "lp_edrscc_v2.pt")
CASF_CSV = _CASF_CSV
OUT_DIR = os.path.join(REPO, "base", "_casf")


def encode_missing(features, pids, labels):
    missing = [pid for pid in pids if pid not in features]
    if not missing:
        return features
    print(f"encoding {len(missing)} CASF-only complexes: {missing}", flush=True)
    model_state, _, model_args = torch.load(CKPT, map_location="cpu")
    model = DrugAllAtomEnergyModel(model_args).cuda()
    model.load_state_dict(model_state)
    model.eval()
    entries = []
    for pid in missing:
        entry, status = build_entry(pid, labels[pid])
        if entry is None:
            print(f"  skip {pid}: {status}", flush=True)
        else:
            entries.append(entry)
    embedding = load_esm_embedding(entries, ["target_seq"])
    for entry in DrugDataset.process(entries, model_args.patch_size):
        pid = entry["pdb"]
        try:
            binder, target = DrugDataset.make_bind_batch([entry], embedding, model_args)
            z_bind, z_tgt = pool_features(model, binder, target)
            energy = model.predict(binder, target).detach()
            features[pid] = torch.cat([z_bind[0], z_tgt[0], energy], dim=-1).float().cpu().numpy()
            print(f"  encoded {pid}", flush=True)
        except Exception as exc:
            print(f"  skip {pid}: extract:{type(exc).__name__}", flush=True)
    return features


def main():
    bundle = torch.load(LP_FEATURES, map_location="cpu")
    features = dict(bundle["feat"])
    lp_labels = bundle["pK"]
    split = bundle["split"]
    frame = pd.read_csv(CASF_CSV)
    frame["pid"] = frame["pid"].astype(str).str.lower()
    pids = frame["pid"].tolist()
    casf_labels = dict(zip(frame["pid"], frame["pK"]))
    features = encode_missing(features, pids, casf_labels)

    def arrays(ids, labels):
        keep = [pid for pid in ids if pid in features and pid in labels]
        x = np.stack([features[pid] for pid in keep]).astype(np.float32)
        y = np.asarray([labels[pid] for pid in keep], dtype=np.float32)
        return x, y, keep

    train = arrays([pid for pid, part in split.items() if part == "train"], lp_labels)
    valid = arrays([pid for pid, part in split.items() if part == "val"], lp_labels)
    test = arrays(pids, casf_labels)
    data = {"train": train, "val": valid, "test": test}
    print(f"LP train={len(train[2])} val={len(valid[2])} | CASF evaluable={len(test[2])}/{len(pids)}",
          flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in range(3):
        metric, prediction = train_one(data, seed=seed, dim=bundle["dim"], device="cuda")
        path = os.path.join(OUT_DIR, f"DSMBind_casf2016_preds_seed{seed}.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pid", "pred", "y"])
            writer.writerows((pid, prediction[pid], casf_labels[pid]) for pid in test[2])
        print(f"seed{seed}: rho={metric['spearman']:.3f} n={len(prediction)} -> {path}", flush=True)


if __name__ == "__main__":
    main()

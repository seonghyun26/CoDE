#!/usr/bin/env python3
"""One train_prop.py config: TargetDiff's PDBbind config for a model, pointed at our pool, split and seed.

    python base/targetdiff/_edrscc/make_config.py --model egnn|egnn_enc --split S --seed N --out CFG

egnn is configs/prop/pdbbind_general_egnn.yml and egnn_enc pdbbind_general_egnn_enc_final_h.yml, unchanged
but for the dataset paths (base/data/targetdiff/pool) and train.seed.
"""
import argparse
import os
import sys

import yaml

TD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(TD))
from datasource import data_dir  # noqa: E402

TEMPLATE = {"egnn": "pdbbind_general_egnn.yml", "egnn_enc": "pdbbind_general_egnn_enc_final_h.yml"}

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--model", choices=sorted(TEMPLATE), required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
with open(os.path.join(TD, "configs", "prop", TEMPLATE[args.model])) as fh:
    cfg = yaml.safe_load(fh)
pool = os.path.join(data_dir("targetdiff"), "pool")
cfg["dataset"]["path"] = pool
cfg["dataset"]["split"] = os.path.join(pool, f"split_{args.split}.pt")
if "emb_path" in cfg["dataset"]:
    cfg["dataset"]["emb_path"] = os.path.join(pool, "emb_final_h.pt")
cfg["train"]["seed"] = args.seed
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)

#!/usr/bin/env python3
"""A trained TargetDiff property model's predictions on one part of a split, as pid,pred,y.

    python base/targetdiff/_edrscc/predict.py --run RUN_DIR --keys split_<s>.pt --part test --out O.csv

RUN_DIR is a train_prop.py log dir; its highest-numbered checkpoint is the best validation one, since
train_prop.py saves a checkpoint only when the validation loss improves. The prediction is the output
head of the complex's measurement type with no coordinate noise, as train_prop.py evaluates.
"""
import argparse
import csv
import glob
import os
import sys

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

TD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # base/targetdiff
sys.path.insert(0, TD)
import utils.transforms_prop as utils_trans  # noqa: E402
from datasets.pdbbind import PDBBindDataset  # noqa: E402
from utils.misc_prop import get_model  # noqa: E402


def best_checkpoint(run_dir):
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {run_dir}")
    return max(ckpts, key=lambda p: int(os.path.basename(p)[:-3]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="train_prop.py log dir")
    ap.add_argument("--keys", required=True, help="split_<split>.pt or split_casf2016.pt")
    ap.add_argument("--part", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = best_checkpoint(args.run)
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]

    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    dataset = PDBBindDataset(config.dataset.path, transform=Compose([pf, lf]),
                             emb_path=config.dataset.get("emb_path"), heavy_only=config.dataset.heavy_only)
    keys = torch.load(args.keys)[args.part]
    loader = DataLoader(Subset(dataset, keys), config.train.batch_size, shuffle=False,
                        follow_batch=["protein_element", "ligand_element"], exclude_keys=["ligand_nbh_list"])
    model = get_model(config, pf.feature_dim, lf.feature_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(os.path.join(config.dataset.path, "keys.csv")) as fh:
        pid_of = {int(r["key"]): r["pid"] for r in csv.DictReader(fh)}
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, pred = model.get_loss(batch, pos_noise_std=0., return_pred=True)
            for i, p, y in zip(batch.id.tolist(), pred.view(-1).tolist(), batch.y.view(-1).tolist()):
                rows.append((pid_of[i], p, y))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "pred", "y"])
        w.writerows(rows)
    print(f"{os.path.basename(path)} (epoch {ckpt['epoch']}): {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()

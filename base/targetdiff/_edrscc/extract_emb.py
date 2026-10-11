#!/usr/bin/env python3
"""TargetDiff's diffusion features for every complex in the pool, from the official pretrained model.

    CUDA_VISIBLE_DEVICES=N python base/targetdiff/_edrscc/extract_emb.py [--limit 5]

Reads base/data/targetdiff/pool/pool_processed.lmdb (build_data.py) and
base/targetdiff/pretrained_models/pretrained_diffusion.pt, and writes pool/emb_final_h.pt: a list in key
order, as PDBBindDataset(emb_path=...) expects, of {final_h, pred_ligand_v, kl_pos, kl_v}. The property
model with enc_feature_type final_h reads only final_h; kl_pos and kl_v are zeros, since PDBBindDataset
builds its nll features from them regardless. final_h comes from one ScorePosNet3D.fetch_embedding pass
on the pocket and the crystal ligand, rows in the order the property model composes them (protein,
then ligand). Ligand aromaticity is read again from the ligand file, in the stored atom order.
"""
import argparse
import os
import pickle
import sys

import lmdb
import torch
from rdkit import Chem
from torch_geometric.data import Batch

TD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # base/targetdiff
sys.path.insert(0, TD)
sys.path.insert(0, os.path.dirname(TD))                                      # base/, for datasource
from datasource import data_dir  # noqa: E402
import utils.transforms as trans  # noqa: E402
from datasets.pl_data import FOLLOW_BATCH  # noqa: E402
from models.molopt_score_model import ScorePosNet3D  # noqa: E402

CKPT = os.path.join(TD, "pretrained_models", "pretrained_diffusion.pt")


def aromatic_in_order(path, n_expected):
    mol = Chem.MolFromMolFile(path, sanitize=False) if path.endswith(".sdf") else Chem.MolFromMol2File(path, sanitize=False)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    arom = [a.GetIsAromatic() for a in mol.GetAtoms()]
    assert len(arom) == n_expected, f"{len(arom)} atoms in {path}, {n_expected} stored"
    return arom


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="only the first N complexes, nothing saved")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device)
    cfg = ckpt["config"]
    pf = trans.FeaturizeProteinAtom()
    lf = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    model = ScorePosNet3D(cfg.model, protein_atom_feature_dim=pf.feature_dim,
                          ligand_atom_feature_dim=lf.feature_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    amap = trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX
    print(f"{CKPT}: ligand_atom_mode {cfg.data.transform.ligand_atom_mode}, device {device}")

    pool = os.path.join(data_dir("targetdiff"), "pool")
    db = lmdb.open(os.path.join(pool, "pool_processed.lmdb"), readonly=True, lock=False, subdir=False)
    n = db.stat()["entries"]
    total = args.limit or n
    emb = [None] * n
    with torch.no_grad(), db.begin() as txn:
        for idx in range(total):
            d = pickle.loads(txn.get(f"{idx:05d}".encode()))
            if hasattr(d, "ligand_nbh_list"):
                del d["ligand_nbh_list"]
            d = pf(d)
            elem = d.ligand_element.tolist()
            arom = aromatic_in_order(str(d.ligand_filename), len(elem))
            d.ligand_atom_feature_full = torch.tensor(
                [amap.get((int(e), bool(a)), amap[(1, False)]) for e, a in zip(elem, arom)], dtype=torch.long)
            batch = Batch.from_data_list([d], follow_batch=FOLLOW_BATCH).to(device)
            preds = model.fetch_embedding(
                protein_pos=batch.protein_pos, protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch, ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full, batch_ligand=batch.ligand_element_batch)
            fh = preds["final_h"].cpu()
            assert fh.size(0) == batch.protein_pos.size(0) + batch.ligand_pos.size(0)
            emb[idx] = {"final_h": fh, "pred_ligand_v": preds["pred_ligand_v"].cpu(),
                        "kl_pos": torch.zeros(11), "kl_v": torch.zeros(11)}
            if idx < 2 or (idx + 1) % 1000 == 0:
                print(f"  {idx + 1}/{total}: final_h {tuple(fh.shape)}", flush=True)
    if args.limit:
        print("--limit: nothing saved")
        return
    out = os.path.join(pool, "emb_final_h.pt")
    torch.save(emb, out)
    print(f"saved {out} ({sum(e is not None for e in emb)} complexes, final_h dim {fh.size(1)})")


if __name__ == "__main__":
    main()

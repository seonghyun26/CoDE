#!/usr/bin/env python3
"""TargetDiff's PDBbind dataset for our splits: one LMDB over every complex they and CASF-2016 use.

    python base/targetdiff/_edrscc/build_data.py [--splits lp_edrscc_v2,lp_edrscc_v2_cl123]

Writes base/data/targetdiff/pool/:
  pool_processed.lmdb    PDBBindDataset's format, keys 00000.. in order: the 10 Å pocket, the
                         heavy-atom ligand, y = pK and kind = 1/2/3 (Ki/Kd/IC50, datasets.protein_ligand.KMAP)
  keys.csv               key, pid
  split_<split>.pt       {train, val, test: [key]} for each split
  split_casf2016.pt      {test: [key]} for the CASF-2016 complexes the six test sets score
Structures come from base/datasource.py (STRUCT_BASES: <pid>_pocket.pdb, <pid>_ligand.sdf, falling back
to the .mol2 when RDKit cannot parse the SDF); labels and the measurement type from LP_PDBBind.csv.
Run it in the decompdiff env (TargetDiff's dependencies).
"""
import argparse
import csv
import os
import pickle
import re
import sys

import lmdb
import pandas as pd
import torch
from tqdm import tqdm

TD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # base/targetdiff
sys.path.insert(0, TD)
sys.path.insert(0, os.path.dirname(TD))                                      # base/, for datasource
from datasource import CASF_CSV, LP_CSV, data_dir, split_csv, structure_dir  # noqa: E402
from utils.data import PDBProtein  # noqa: E402
from datasets.protein_ligand import KMAP, parse_sdf_file_mol  # noqa: E402
from datasets.pl_data import ProteinLigandData, torchify_dict  # noqa: E402

KIND = {k.upper(): v for k, v in KMAP.items()}                               # KI 1, KD 2, IC50 3


def kind_of(text) -> int:
    m = re.match(r"\s*(KI|KD|IC50)", str(text).upper())
    return KIND[m.group(1)] if m else KIND["KI"]


def read_split(split):
    with open(split_csv(split)) as fh:
        return {r["pid"].strip().lower(): r["split"].strip().lower() for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", default="lp_edrscc_v2,lp_edrscc_v2_cl123")
    args = ap.parse_args()
    splits = {s: read_split(s) for s in args.splits.split(",") if s}
    with open(CASF_CSV) as fh:
        casf = [r["pid"].strip().lower() for r in csv.DictReader(fh)]
    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    lp = lp.drop_duplicates("pid").set_index("pid")
    pids = sorted(set().union(*[set(s) for s in splits.values()]) | set(casf))

    out = os.path.join(data_dir("targetdiff"), "pool")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "pool_processed.lmdb")
    for p in (path, path + "-lock"):
        if os.path.exists(p):
            os.remove(p)
    db = lmdb.open(path, map_size=20 * 1024 ** 3, create=True, subdir=False, readonly=False)
    pid2key, fails, mol2 = {}, {}, 0
    with db.begin(write=True, buffers=True) as txn:
        for pid in tqdm(pids):
            d = structure_dir(pid)
            if pid not in lp.index or d is None:
                fails[pid] = "no label" if d is not None else "no structure"
                continue
            pocket = os.path.join(d, f"{pid}_pocket.pdb")
            ligands = [os.path.join(d, f"{pid}_ligand.{ext}") for ext in ("sdf", "mol2")]
            try:
                pocket_dict = PDBProtein(pocket).to_dict_atom()
                for i, lig in enumerate(ligands):
                    try:
                        ligand_dict = parse_sdf_file_mol(lig, heavy_only=True)
                        mol2 += i
                        break
                    except Exception:
                        if i == len(ligands) - 1 or not os.path.exists(ligands[i + 1]):
                            raise
                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pocket_dict), ligand_dict=torchify_dict(ligand_dict))
                data.protein_filename, data.ligand_filename = pocket, lig
                data.y = torch.tensor(float(lp.loc[pid, "value"]))
                data.kind = torch.tensor(kind_of(lp.loc[pid, "kd/ki"]))
            except Exception as e:  # noqa: BLE001 - recorded per complex
                fails[pid] = f"{type(e).__name__}: {e}"[:120]
                continue
            key = len(pid2key)
            txn.put(key=f"{key:05d}".encode(), value=pickle.dumps(data))
            pid2key[pid] = key
    db.close()

    with open(os.path.join(out, "keys.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "pid"])
        w.writerows((k, p) for p, k in sorted(pid2key.items(), key=lambda x: x[1]))
    for name, sp in splits.items():
        parts = {part: [pid2key[p] for p, s in sp.items() if s == part and p in pid2key]
                 for part in ("train", "val", "test")}
        torch.save(parts, os.path.join(out, f"split_{name}.pt"))
        missing = {part: sum(1 for p, s in sp.items() if s == part and p not in pid2key) for part in parts}
        print(f"{name}: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()) + f" | missing {missing}")
    torch.save({"test": [pid2key[p] for p in casf if p in pid2key]}, os.path.join(out, "split_casf2016.pt"))
    print(f"casf2016: test {sum(p in pid2key for p in casf)} of {len(casf)}")
    print(f"{len(pid2key)} complexes in {path} ({mol2} from the .mol2); {len(fails)} failed")
    for pid, why in sorted(fails.items())[:20]:
        print(f"  {pid}: {why}")


if __name__ == "__main__":
    main()

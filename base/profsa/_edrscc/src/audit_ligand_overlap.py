#!/usr/bin/env python
"""Audit CL3 ligand overlap with ProFSA pseudo-ligands and downstream train ligands."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import time
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr


REPO = Path(__file__).resolve().parents[4]
DEFAULT_PRETRAIN = [
    Path("/tmp/profsa_leak_audit.o2NFiV/dataset/train/train.lmdb"),
    REPO / "base/profsa/data/dataset/train/valid.lmdb",
]
DEFAULT_CL3 = REPO / "base/data/profsa/edrscc_cl123"
DEFAULT_LP_CSV = REPO / "dataset/data/pdbbind/raw/LP_PDBBind.csv"
DEFAULT_MANIFEST = REPO / "voxbind/splits/lp_edrscc_v2_cl123.csv"
DEFAULT_OUT = REPO / "base/profsa/results/profsa_cl123_ligand_audit"
PREDICTIONS = {
    "C": REPO / "base/_casf/novel_preds/C_cl123_seed*.csv",
    "C+D+G": REPO / "base/_casf/novel_preds/C_D_G_cl123_seed*.csv",
    "ProFSA": REPO / "base/profsa/results/profsa_cl123_pretrain_audit/preds/ProFSA_v2_seed*.csv",
}


def environment(path: Path):
    return lmdb.open(
        str(path), subdir=False, readonly=True, lock=False, readahead=False,
        meminit=False, max_readers=1,
    )


def collect_smiles(paths: list[Path], *, pretrain: bool) -> tuple[dict[str, str], int]:
    """Return raw SMILES -> example PID/pseudo-pocket and number of LMDB entries."""
    unique = {}
    total = 0
    for path in paths:
        env = environment(path)
        with env.begin(buffers=True) as txn:
            for _, value in txn.cursor():
                sample = pickle.loads(value)
                smiles = str(sample.get("smi", ""))
                name = str(sample.get("pocket", "")).lower()
                if smiles:
                    unique.setdefault(smiles, name)
                total += 1
        env.close()
        print(f"scanned {path}: entries={total:,}, unique_raw={len(unique):,}", flush=True)
    return unique, total


def canonical_library(raw: dict[str, str]) -> dict:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    records = {}
    invalid = 0
    for index, (smiles, name) in enumerate(raw.items(), 1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if canonical in records:
            continue
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True) if scaffold_mol.GetNumAtoms() else ""
        records[canonical] = {
            "smiles": smiles,
            "name": name,
            "fingerprint": generator.GetFingerprint(mol),
            "scaffold": scaffold,
        }
        if index % 100_000 == 0:
            print(f"canonicalized {index:,}/{len(raw):,}: unique={len(records):,}", flush=True)
    return {"records": records, "invalid": invalid, "generator": generator}


def load_lp_smiles(lp_csv: Path, manifest: Path) -> tuple[dict[str, str], dict[str, str], int]:
    with manifest.open() as handle:
        roles = {row["pid"].lower(): row["split"].lower() for row in csv.DictReader(handle)}
    with lp_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    pid_key = "Unnamed: 0" if "Unnamed: 0" in rows[0] else ("" if "" in rows[0] else "pid")
    smiles = {row[pid_key].lower(): row["smiles"] for row in rows}
    train = {smiles[pid]: pid for pid, role in roles.items() if role == "train"}
    test = {pid: smiles[pid] for pid, role in roles.items() if role == "test"}
    n_train = sum(role == "train" for role in roles.values())
    return train, test, n_train


def nearest(queries: dict[str, str], library: dict, prefix: str) -> list[dict]:
    records = library["records"]
    canonicals = list(records)
    fps = [records[key]["fingerprint"] for key in canonicals]
    scaffolds = {records[key]["scaffold"] for key in canonicals if records[key]["scaffold"]}
    generator = library["generator"]
    rows = []
    for index, (pid, smiles) in enumerate(sorted(queries.items()), 1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid CL3 ligand: {pid} {smiles}")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True) if scaffold_mol.GetNumAtoms() else ""
        similarities = DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol), fps)
        best_index = int(np.argmax(similarities))
        best_key = canonicals[best_index]
        best = records[best_key]
        rows.append({
            "pid": pid,
            f"{prefix}_max_tanimoto": float(similarities[best_index]),
            f"{prefix}_exact_molecule": canonical in records,
            f"{prefix}_exact_scaffold": bool(scaffold and scaffold in scaffolds),
            f"{prefix}_best_name": best["name"],
            f"{prefix}_best_smiles": best["smiles"],
        })
        if index % 100 == 0:
            print(f"{prefix}: queried {index}/{len(queries)}", flush=True)
    return rows


def load_seed_predictions(pattern: Path) -> list[dict[str, tuple[float, float]]]:
    seeds = []
    for path in sorted(glob.glob(str(pattern))):
        with open(path) as handle:
            seeds.append({
                row["pid"].lower(): (float(row["pred"]), float(row["y"]))
                for row in csv.DictReader(handle)
            })
    if not seeds:
        raise FileNotFoundError(pattern)
    return seeds


def metric(seed: dict[str, tuple[float, float]], pids: set[str]) -> dict[str, float]:
    values = np.asarray([seed[pid] for pid in sorted(pids)])
    pred, target = values[:, 0], values[:, 1]
    return {
        "pearson": float(pearsonr(pred, target).statistic),
        "spearman": float(spearmanr(pred, target).statistic),
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
    }


def performance(rows: list[dict]) -> dict:
    row_map = {row["pid"]: row for row in rows}
    cohorts = {
        "all_test": set(row_map),
        "pseudo_tanimoto_lt0p3": {p for p, r in row_map.items() if r["pseudo_max_tanimoto"] < 0.3},
        "pseudo_tanimoto_lt0p5": {p for p, r in row_map.items() if r["pseudo_max_tanimoto"] < 0.5},
        "pseudo_no_exact_scaffold": {p for p, r in row_map.items() if not r["pseudo_exact_scaffold"]},
        "downstream_tanimoto_lt0p3": {p for p, r in row_map.items() if r["downstream_max_tanimoto"] < 0.3},
        "dual_tanimoto_lt0p3": {
            p for p, r in row_map.items()
            if r["pseudo_max_tanimoto"] < 0.3 and r["downstream_max_tanimoto"] < 0.3
        },
    }
    out = {}
    for method, pattern in PREDICTIONS.items():
        seeds = load_seed_predictions(pattern)
        out[method] = {}
        for cohort, pids in cohorts.items():
            per_seed = [metric(seed, pids) for seed in seeds]
            block = {"n": len(pids), "seeds": len(seeds), "per_seed": per_seed}
            for key in ("pearson", "spearman", "rmse"):
                values = [item[key] for item in per_seed]
                block[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
            out[method][cohort] = block
    return out


def summarize(rows: list[dict], prefix: str) -> dict:
    similarity = np.asarray([row[f"{prefix}_max_tanimoto"] for row in rows])
    return {
        "n_test": len(rows),
        "max_tanimoto": {
            "mean": float(similarity.mean()),
            "median": float(np.median(similarity)),
            "q90": float(np.quantile(similarity, 0.9)),
            "maximum": float(similarity.max()),
        },
        "threshold_counts": {
            str(value): int((similarity >= value).sum())
            for value in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0)
        },
        "exact_molecule": int(sum(bool(row[f"{prefix}_exact_molecule"]) for row in rows)),
        "exact_nonempty_bemis_murcko_scaffold": int(
            sum(bool(row[f"{prefix}_exact_scaffold"]) for row in rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-lmdb", type=Path, action="append")
    parser.add_argument("--lp-csv", type=Path, default=DEFAULT_LP_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    pretrain_paths = args.pretrain_lmdb or DEFAULT_PRETRAIN
    started = time.time()
    RDLogger.DisableLog("rdApp.*")

    pseudo_raw, n_pseudo = collect_smiles(pretrain_paths, pretrain=True)
    downstream_raw, queries, n_downstream = load_lp_smiles(args.lp_csv, args.manifest)
    pseudo = canonical_library(pseudo_raw)
    downstream = canonical_library(downstream_raw)
    pseudo_rows = nearest(queries, pseudo, "pseudo")
    downstream_rows = nearest(queries, downstream, "downstream")
    downstream_map = {row["pid"]: row for row in downstream_rows}
    rows = [{**row, **downstream_map[row["pid"]]} for row in pseudo_rows]

    args.output.mkdir(parents=True, exist_ok=True)
    per_test = args.output / "ligand_overlap_per_test.csv"
    with per_test.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "fingerprint": "Morgan/ECFP4 radius=2, 2048 bits",
        "scaffold": "exact canonical nonempty Bemis-Murcko scaffold",
        "proFSA_pretraining": {
            "lmdbs": [str(path) for path in pretrain_paths],
            "entries": n_pseudo,
            "unique_raw_smiles": len(pseudo_raw),
            "unique_canonical_molecules": len(pseudo["records"]),
            "invalid_smiles": pseudo["invalid"],
            **summarize(rows, "pseudo"),
        },
        "cl123_downstream_train": {
            "manifest": str(args.manifest),
            "entries": n_downstream,
            "unique_raw_smiles": len(downstream_raw),
            "unique_canonical_molecules": len(downstream["records"]),
            "invalid_smiles": downstream["invalid"],
            **summarize(rows, "downstream"),
        },
        "performance": performance(rows),
        "seconds": time.time() - started,
    }
    output = args.output / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "performance"}, indent=2))
    print(f"wrote {output} and {per_test}")


if __name__ == "__main__":
    main()

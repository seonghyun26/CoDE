#!/usr/bin/env python
"""Score C, C+D+G, and ProFSA on CL3 test protein-novelty subsets.

Novelty is the maximum MMseqs sequence identity from each
``lp_edrscc_v2_cl123`` test protein to any protein in that split's train set.
The supplied alignment uses columns::

    query target fident alnlen qcov tcov

and the same search constraints as the existing v2 novel30/60 analysis
(minimum identity 30%, both-sequence coverage at least 80%).  A query with no
reported hit therefore has maximum identity below 30%.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ALIGNMENT = Path("/tmp/seqsim/test_v_train.m8")
DEFAULT_PROBE_DIR = REPO / "base/_casf/novel_preds"
DEFAULT_PROFSA_DIR = REPO / "base/profsa/results/profsa_cl123_pretrain_audit/preds"
DEFAULT_PROFSA_AUDIT = REPO / "base/profsa/results/profsa_cl123_pretrain_audit"
DEFAULT_LIGAND_AUDIT = (
    REPO / "base/profsa/results/profsa_cl123_ligand_audit/ligand_overlap_per_test.csv"
)
DEFAULT_OUT = REPO / "base/_casf/cl123_novel_subsets"


def split_ids() -> tuple[set[str], set[str]]:
    import sys

    sys.path.insert(0, str(REPO))
    from voxbind.splits import load_split

    split = load_split("lp_edrscc_v2_cl123")
    return set(split["train"]), set(split["test"])


def max_train_identity(path: Path, train: set[str], test: set[str]) -> dict[str, float]:
    maximum = {pid: 0.0 for pid in test}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) < 6:
                continue
            query, target = fields[0].lower(), fields[1].lower()
            if query not in test or target not in train:
                continue
            identity = float(fields[2])
            qcov, tcov = float(fields[4]), float(fields[5])
            if qcov >= 0.8 and tcov >= 0.8:
                maximum[query] = max(maximum[query], identity)
    return maximum


def load_predictions(pattern: str) -> list[dict[str, tuple[float, float]]]:
    seeds = []
    for path in sorted(glob.glob(pattern)):
        rows = {}
        with open(path) as handle:
            for row in csv.DictReader(handle):
                rows[row["pid"].lower()] = (float(row["pred"]), float(row["y"]))
        seeds.append(rows)
    if not seeds:
        raise FileNotFoundError(pattern)
    return seeds


def metrics(seed: dict[str, tuple[float, float]], pids: set[str]) -> dict[str, float]:
    pairs = [seed[pid] for pid in sorted(pids) if pid in seed]
    pred = np.asarray([row[0] for row in pairs])
    target = np.asarray([row[1] for row in pairs])
    return {
        "n": len(pairs),
        "pearson": float(pearsonr(pred, target).statistic),
        "spearman": float(spearmanr(pred, target).statistic),
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
    }


def aggregate(seeds: list[dict[str, tuple[float, float]]], pids: set[str]) -> dict:
    rows = [metrics(seed, pids) for seed in seeds]
    if len({row["n"] for row in rows}) != 1:
        raise ValueError(f"seed PID coverage differs: {[row['n'] for row in rows]}")
    out = {"n": rows[0]["n"], "seeds": len(rows), "per_seed": rows}
    for key in ("pearson", "spearman", "rmse"):
        values = [row[key] for row in rows]
        out[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return out


def profsa_pretrain_novel(audit: Path, threshold: float) -> set[str]:
    exact = set(json.loads((audit / "exact_overlap_audit.json").read_text())["exact_overlaps"])
    novel = set()
    with (audit / "sequence_overlap_per_test.csv").open() as handle:
        for row in csv.DictReader(handle):
            pid = row["pid"].lower()
            identity = row.get("all_pident", "")
            if pid not in exact and (not identity or float(identity) < threshold):
                novel.add(pid)
    return novel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--profsa-dir", type=Path, default=DEFAULT_PROFSA_DIR)
    parser.add_argument("--profsa-audit", type=Path, default=DEFAULT_PROFSA_AUDIT)
    parser.add_argument("--ligand-audit", type=Path, default=DEFAULT_LIGAND_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=None,
        help="Fail instead of silently aggregating an incomplete seed campaign.",
    )
    args = parser.parse_args()

    train, test = split_ids()
    maximum = max_train_identity(args.alignment, train, test)
    cohorts = {
        "cl123_test": test,
        "cl123_test_novel60": {pid for pid, value in maximum.items() if value < 0.60},
        "cl123_test_novel30": {pid for pid, value in maximum.items() if value < 0.30},
    }
    # Additional leakage stress tests: novel to both the downstream CL3 train set and
    # every identifiable ProFSA pretraining source protein, excluding exact source-PDB hits.
    cohorts["cl123_and_profsa_pretrain_novel60"] = (
        cohorts["cl123_test_novel60"] & profsa_pretrain_novel(args.profsa_audit, 60.0)
    )
    cohorts["cl123_and_profsa_pretrain_novel30"] = (
        cohorts["cl123_test_novel30"] & profsa_pretrain_novel(args.profsa_audit, 30.0)
    )
    if args.ligand_audit.exists():
        with args.ligand_audit.open() as handle:
            ligand = {row["pid"].lower(): row for row in csv.DictReader(handle)}
        protein_novel30 = cohorts["cl123_and_profsa_pretrain_novel30"]
        cohorts["protein30_and_profsa_pseudoligand30"] = {
            pid for pid in protein_novel30
            if float(ligand[pid]["pseudo_max_tanimoto"]) < 0.3
        }
        cohorts["protein30_and_both_ligand30"] = {
            pid for pid in protein_novel30
            if float(ligand[pid]["pseudo_max_tanimoto"]) < 0.3
            and float(ligand[pid]["downstream_max_tanimoto"]) < 0.3
        }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, pids in cohorts.items():
        (args.output / f"{name}.txt").write_text("".join(f"{pid}\n" for pid in sorted(pids)))
    with (args.output / "cl123_test_max_train_identity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pid", "max_cl123_train_identity"])
        writer.writerows((pid, maximum[pid]) for pid in sorted(maximum))

    methods = {
        "C": load_predictions(str(args.probe_dir / "C_cl123_seed*.csv")),
        "C+D+G": load_predictions(str(args.probe_dir / "C_D_G_cl123_seed*.csv")),
        "ProFSA": load_predictions(str(args.profsa_dir / "ProFSA_v2_seed*.csv")),
    }
    if args.expected_seeds is not None:
        counts = {method: len(seeds) for method, seeds in methods.items()}
        incomplete = {
            method: count
            for method, count in counts.items()
            if count != args.expected_seeds
        }
        if incomplete:
            raise ValueError(
                f"expected exactly {args.expected_seeds} seeds per method, got {counts}"
            )
    result = {
        "split": "lp_edrscc_v2_cl123",
        "seed_policy": (
            "fresh downstream probe training; earlier cached CL3 predictions excluded"
        ),
        "expected_seeds": args.expected_seeds,
        "probe_predictions": str(args.probe_dir),
        "profsa_predictions": str(args.profsa_dir),
        "similarity_reference": "lp_edrscc_v2_cl123 train",
        "similarity_rule": "MMseqs fident; qcov>=0.8 and tcov>=0.8; strict < threshold",
        "dual_filter_rule": (
            "strictly below threshold to both CL3 train and all identifiable ProFSA "
            "pretraining source proteins; exact ProFSA source-PDB overlaps excluded"
        ),
        "alignment": str(args.alignment),
        "methods": {
            method: {cohort: aggregate(seeds, pids) for cohort, pids in cohorts.items()}
            for method, seeds in methods.items()
        },
    }
    output = args.output / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

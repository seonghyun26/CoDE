#!/usr/bin/env python
"""Audit ProFSA pretraining overlap against an affinity test split.

The released ProFSA pretraining LMDB names every pseudo-complex as
``<PDB-ID>_<fragment-index>``.  This script reports:

1. exact source-PDB overlap with the requested test split;
2. for exact overlaps, whether the pseudo-fragment and its surroundings occupy
   the experimental ligand pocket in the original coordinate frame;
3. FASTA files needed for a subsequent MMseqs sequence-similarity audit.

Example::

    python -u audit_pretrain_overlap.py \
      --pretrain-lmdb /tmp/profsa/dataset/train/train.lmdb \
      --pretrain-lmdb ../../../data/dataset/train/valid.lmdb \
      --rcsb-seqres /tmp/pdb_seqres.txt.gz \
      --out-dir ../../../results/profsa_cl123_pretrain_audit
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPO / "voxbind/splits/lp_edrscc_v2_cl123.csv"
DEFAULT_LP_CSV = REPO / "dataset/data/pdbbind/raw/LP_PDBBind.csv"
DEFAULT_TEST_LMDB = REPO / "base/data/profsa/edrscc_cl123/test.lmdb"
DEFAULT_OUT = REPO / "base/profsa/results/profsa_cl123_pretrain_audit"
PDB_RE = re.compile(r"^([0-9][A-Za-z0-9]{3})(?:_|$)")


def lmdb_env(path: Path):
    return lmdb.open(
        str(path), subdir=False, readonly=True, lock=False, readahead=False,
        meminit=False, max_readers=1,
    )


def test_ids(manifest: Path) -> set[str]:
    with manifest.open() as handle:
        return {
            row["pid"].lower()
            for row in csv.DictReader(handle)
            if row["split"].lower() == "test"
        }


def load_test_lmdb(path: Path) -> dict[str, dict]:
    out = {}
    env = lmdb_env(path)
    with env.begin(buffers=True) as txn:
        for _, value in txn.cursor():
            sample = pickle.loads(value)
            out[str(sample["pocket"]).lower()] = sample
    env.close()
    return out


def spatial_metrics(pretrain: dict, test: dict) -> dict[str, float]:
    test_ligand = np.asarray(test["coordinates"]).reshape(-1, 3)
    test_pocket = np.asarray(test["pocket_coordinates"]).reshape(-1, 3)
    fragment = np.asarray(pretrain["lig_coord_real"]).reshape(-1, 3)
    surroundings = np.asarray(pretrain["pocket_coordinates"]).reshape(-1, 3)
    pseudo_complex = np.concatenate([fragment, surroundings])

    return {
        "fragment_to_test_ligand_min_A": float(
            cKDTree(test_ligand).query(fragment, k=1)[0].min()
        ),
        "fragment_to_test_pocket_min_A": float(
            cKDTree(test_pocket).query(fragment, k=1)[0].min()
        ),
        "fragment_test_ligand_centroid_A": float(
            np.linalg.norm(fragment.mean(0) - test_ligand.mean(0))
        ),
        # Same-PDB atoms retain their deposited coordinate frame.  A 0.15 A
        # tolerance accommodates serialization/format rounding while remaining
        # far below a chemical bond length.
        "test_pocket_atom_coverage_0p15A": float(
            (cKDTree(pseudo_complex).query(test_pocket, k=1)[0] <= 0.15).mean()
        ),
    }


def scan_pretraining(
    paths: list[Path],
    wanted: set[str],
    test_samples: dict[str, dict],
) -> tuple[Counter, dict[str, list[dict]], list[dict]]:
    source_counts: Counter[str] = Counter()
    exact: dict[str, list[dict]] = {}
    db_stats = []

    for path in paths:
        env = lmdb_env(path)
        started = time.time()
        bad_names = 0
        with env.begin(buffers=True) as txn:
            n_entries = txn.stat()["entries"]
            for index, (key, value) in enumerate(txn.cursor(), 1):
                sample = pickle.loads(value)
                pocket_name = str(sample.get("pocket", ""))
                match = PDB_RE.match(pocket_name)
                if match is None:
                    bad_names += 1
                    continue
                pid = match.group(1).lower()
                source_counts[pid] += 1
                if pid in wanted:
                    record = {
                        "source": path.name,
                        "lmdb_key": bytes(key).decode(),
                        "pseudo_pocket": pocket_name,
                        **spatial_metrics(sample, test_samples[pid]),
                    }
                    exact.setdefault(pid, []).append(record)
                if index % 500_000 == 0:
                    elapsed = time.time() - started
                    print(f"{path.name}: {index:,}/{n_entries:,} ({elapsed:.1f}s)", flush=True)
        env.close()
        db_stats.append({
            "path": str(path),
            "entries": n_entries,
            "unparsed_pocket_names": bad_names,
            "seconds": time.time() - started,
        })
    return source_counts, exact, db_stats


def summarize_exact(exact: dict[str, list[dict]]) -> dict[str, dict]:
    out = {}
    for pid, records in sorted(exact.items()):
        nearest = min(records, key=lambda row: row["fragment_to_test_ligand_min_A"])
        out[pid] = {
            "n_pseudo_complexes": len(records),
            "fragment_to_test_ligand_min_A": min(
                row["fragment_to_test_ligand_min_A"] for row in records
            ),
            "fragment_to_test_pocket_min_A": min(
                row["fragment_to_test_pocket_min_A"] for row in records
            ),
            "fragment_test_ligand_centroid_min_A": min(
                row["fragment_test_ligand_centroid_A"] for row in records
            ),
            "test_pocket_atom_coverage_0p15A_max": max(
                row["test_pocket_atom_coverage_0p15A"] for row in records
            ),
            "nearest_pseudo_pocket": nearest["pseudo_pocket"],
        }
    return out


def write_query_fasta(lp_csv: Path, wanted: set[str], path: Path) -> int:
    frame = pd.read_csv(lp_csv)
    pid_col = "Unnamed: 0" if "Unnamed: 0" in frame.columns else "pid"
    seq = dict(zip(frame[pid_col].astype(str).str.lower(), frame["seq"].astype(str)))
    with path.open("w") as handle:
        for pid in sorted(wanted):
            value = seq.get(pid, "").strip().upper()
            if value and value != "NAN":
                handle.write(f">{pid}\n{value}\n")
    return sum(bool(seq.get(pid, "").strip()) and seq.get(pid, "").strip().upper() != "NAN"
               for pid in wanted)


def write_pretrain_fasta(
    seqres: Path,
    source_ids: set[str],
    path: Path,
    unambiguous_path: Path,
) -> dict:
    opener = gzip.open if seqres.suffix == ".gz" else open
    found_pdbs = set()
    n_sequences = 0
    records: dict[str, list[tuple[str, str]]] = {}
    header = None
    pid = None
    sequence: list[str] = []

    def save_record() -> None:
        if header is not None and pid is not None:
            records.setdefault(pid, []).append((header, "".join(sequence)))

    with opener(seqres, "rt") as source:
        for line in source:
            if line.startswith(">"):
                save_record()
                token = line[1:].split(maxsplit=1)[0]
                candidate_pid = token[:4].lower()
                if candidate_pid in source_ids and "mol:protein" in line:
                    header, pid, sequence = token.lower(), candidate_pid, []
                else:
                    header, pid, sequence = None, None, []
            elif header is not None:
                sequence.append(line.strip().upper())
        save_record()

    with path.open("w") as target:
        for pid, pdb_records in sorted(records.items()):
            found_pdbs.add(pid)
            for token, seq in pdb_records:
                target.write(f">{token}\n{seq}\n")
                n_sequences += 1

    # The released LMDB does not identify the source chain.  PDB entries whose
    # protein chains all have the same sequence are therefore the unambiguous
    # subset: whichever chain supplied the pseudo-fragment, its sequence is known.
    n_unambiguous_pdbs = 0
    with unambiguous_path.open("w") as target:
        for pid, pdb_records in sorted(records.items()):
            unique = {}
            for token, seq in pdb_records:
                unique.setdefault(seq, token)
            if len(unique) == 1:
                seq, token = next(iter(unique.items()))
                target.write(f">{token}\n{seq}\n")
                n_unambiguous_pdbs += 1
    return {
        "source_pdbs_with_protein_sequence": len(found_pdbs),
        "source_pdbs_missing_from_current_rcsb_seqres": len(source_ids - found_pdbs),
        "protein_chain_sequences": n_sequences,
        "source_pdbs_with_one_unique_protein_sequence": n_unambiguous_pdbs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lp-csv", type=Path, default=DEFAULT_LP_CSV)
    parser.add_argument("--test-lmdb", type=Path, default=DEFAULT_TEST_LMDB)
    parser.add_argument("--pretrain-lmdb", type=Path, action="append", required=True)
    parser.add_argument("--rcsb-seqres", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = test_ids(args.manifest)
    test_samples = load_test_lmdb(args.test_lmdb)
    if set(test_samples) != wanted:
        raise ValueError(
            f"manifest/test LMDB mismatch: manifest={len(wanted)}, lmdb={len(test_samples)}, "
            f"symmetric_difference={len(wanted ^ set(test_samples))}"
        )

    source_counts, exact_records, db_stats = scan_pretraining(
        args.pretrain_lmdb, wanted, test_samples
    )
    exact = summarize_exact(exact_records)

    source_path = args.out_dir / "pretrain_source_pdb_ids.txt"
    source_path.write_text("".join(f"{pid}\n" for pid in sorted(source_counts)))
    query_fasta = args.out_dir / "cl123_test.fasta"
    n_queries = write_query_fasta(args.lp_csv, wanted, query_fasta)

    fasta_stats = None
    if args.rcsb_seqres:
        fasta_stats = write_pretrain_fasta(
            args.rcsb_seqres, set(source_counts),
            args.out_dir / "pretrain_source_proteins.fasta",
            args.out_dir / "pretrain_source_unambiguous.fasta",
        )

    summary = {
        "test_split": args.manifest.stem,
        "n_test": len(wanted),
        "pretrain_lmdbs": db_stats,
        "n_pretrain_pseudo_complexes": sum(source_counts.values()),
        "n_unique_pretrain_source_pdbs": len(source_counts),
        "n_exact_source_pdb_overlap": len(exact),
        "fraction_test_exact_source_pdb_overlap": len(exact) / len(wanted),
        "n_pseudo_complexes_from_exact_test_pdbs": sum(
            row["n_pseudo_complexes"] for row in exact.values()
        ),
        "exact_spatial_threshold_counts": {
            "fragment_within_4A_of_test_ligand": sum(
                row["fragment_to_test_ligand_min_A"] <= 4 for row in exact.values()
            ),
            "fragment_within_8A_of_test_ligand": sum(
                row["fragment_to_test_ligand_min_A"] <= 8 for row in exact.values()
            ),
            "test_pocket_coverage_at_least_25pct": sum(
                row["test_pocket_atom_coverage_0p15A_max"] >= 0.25
                for row in exact.values()
            ),
            "denominator_exact_pdbs": len(exact),
        },
        "n_test_sequences_written": n_queries,
        "rcsb_fasta": fasta_stats,
        "exact_overlaps": exact,
    }
    (args.out_dir / "exact_overlap_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "exact_overlaps"}, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()

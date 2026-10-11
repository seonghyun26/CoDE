#!/usr/bin/env python3
"""Validate a decontaminated PLINDER corpus against its own recorded contract.

Version-free: there is no per-version table here. Everything the check needs is discovered
from the corpus directory itself, so this validates any corpus `02_make_clean_corpus.py`
has ever written, including ones built after this file was last touched.

    python dataset/plinder/validate_clean_corpus.py                      # CoDE_pretrain_v1
    python dataset/plinder/validate_clean_corpus.py --corpus ab7v2p1_clean
    python dataset/plinder/validate_clean_corpus.py --all
    python dataset/plinder/validate_clean_corpus.py --loader-smoke       # also builds datasets

What it asserts, all of it recomputed from the manifest and compared to what the corpus
claims about itself in `loader_contract.json`:

  * exactly one `*_keep` mask is present, and the manifest's `ok` is identical to it,
  * the recomputed position counts match the contract's recorded counts (the drift gate:
    a mismatch means the corpus on this machine is not the one the contract describes),
  * no retained position belongs to a PDB in the corpus's removed list,
  * the density box is a hard link to the source corpus's box rather than a copy,
  * train + validation together consume every retained position.

`--loader-smoke` additionally builds the real train and val datasets, checks their lengths
against the contract, and pulls the last sample to confirm density is attached. It needs the
density box on disk and is slow, so it is off by default.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VOX = ROOT / "voxbind"
DATA = ROOT / "dataset/data"
PRETRAIN = DATA / "pretrain"
SPLITS = VOX / "splits/plinder"

# The source corpus, its box and its tuples are read from each corpus's loader_contract.json.
DEFAULT_CORPUS = "CoDE_pretrain_v1"


def corpus_paths(tag: str) -> tuple[Path, Path]:
    """Split dir and resample dir for a corpus tag, or for an explicit resample path."""
    candidate = Path(tag)
    if candidate.is_dir():
        resample = candidate.resolve()
        return SPLITS / resample.name.replace("xray_resample_", ""), resample
    return SPLITS / tag, PRETRAIN / f"xray_resample_{tag}"


def discover_corpora() -> list[str]:
    """Every decontaminated corpus on disk: a loader contract plus a `*_keep` mask."""
    found = []
    for d in sorted(PRETRAIN.glob("xray_resample_*")):
        # A symlink is an old name kept for runs that recorded it, not a second corpus.
        if d.is_symlink():
            continue
        manifest = d / "train_manifest.npz"
        if not (d / "loader_contract.json").exists() or not manifest.exists():
            continue
        try:
            with np.load(manifest, allow_pickle=False) as m:
                if not any(k.endswith("_keep") for k in m.files):
                    continue
        except Exception:
            continue
        found.append(d.name.replace("xray_resample_", ""))
    return found


def _keep_key(manifest_files) -> str:
    keys = [k for k in manifest_files if k.endswith("_keep")]
    if len(keys) != 1:
        raise RuntimeError(
            f"expected exactly one '*_keep' mask in the manifest, found {keys or 'none'}"
        )
    return keys[0]


def _removed_path(split_dir: Path, keep_key: str) -> Path:
    """The removal list that belongs to this keep mask.

    Derived from the key rather than looked up: `paper7_id100_keep` ->
    `paper7_id100_removed_pdbs.txt`.
    """
    path = split_dir / f"{keep_key[:-len('_keep')]}_removed_pdbs.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def validate_metadata(tag: str) -> dict:
    split_dir, resample = corpus_paths(tag)
    contract = json.loads((resample / "loader_contract.json").read_text())

    with np.load(resample / "train_manifest.npz", allow_pickle=False) as manifest:
        keep_key = _keep_key(manifest.files)
        pdb_ids = pd.Series(manifest["pdb_id"].astype(str)).str.lower()
        available = manifest["ok"].astype(bool)
        keep = manifest[keep_key].astype(bool)
    if not np.array_equal(available, keep):
        raise RuntimeError(
            f"{tag}: manifest 'ok' differs from '{keep_key}' in "
            f"{int((available != keep).sum())} positions"
        )

    removed = {ln.strip().lower()
               for ln in _removed_path(split_dir, keep_key).read_text().splitlines()
               if ln.strip()}
    target_box = Path(contract["box_path"])
    source_box = Path(contract["source_resample_dir"]) / target_box.name

    # Recomputed now, from the manifest and the removal list on this machine.
    checks = {
        "source_manifest_positions": len(pdb_ids),
        "retained_manifest_positions": int(available.sum()),
        "removed_unique_pdb": len(removed),
        "box_is_hardlink_to_source": os.path.samefile(source_box, target_box),
    }
    # Compared to what the corpus says about itself. Only keys the contract records are
    # checked, so a corpus written by a future version of the builder still validates.
    drift = {k: {"recomputed": v, "contract": contract[k]}
             for k, v in checks.items() if k in contract and contract[k] != v}
    if drift:
        raise RuntimeError(f"{tag} contract drift: {json.dumps(drift, indent=2)}")

    overlap = len(set(pdb_ids[available]) & removed)
    if overlap:
        raise RuntimeError(f"{tag}: {overlap} retained positions belong to removed PDBs")

    if contract["subset_n"] + contract["subset_val_n"] != checks["retained_manifest_positions"]:
        raise RuntimeError(
            f"{tag}: train/validation contract does not consume every retained position "
            f"({contract['subset_n']} + {contract['subset_val_n']} != "
            f"{checks['retained_manifest_positions']})"
        )

    selection = split_dir / "plinder_selected.csv"
    n_selected = None
    if selection.exists():
        frame = pd.read_csv(selection)
        frame["entry_pdb_id"] = frame.entry_pdb_id.astype(str).str.lower()
        n_selected = {"selection_rows": len(frame),
                      "selection_unique_pdb": int(frame.entry_pdb_id.nunique())}

    return {
        "keep_key": keep_key,
        "removed_list": _removed_path(split_dir, keep_key).name,
        "excluded_positions": int((~available).sum()),
        "available_removed_overlap": overlap,
        "subset_n": contract["subset_n"],
        "subset_val_n": contract["subset_val_n"],
        **checks,
        **(n_selected or {}),
    }


def make_dataset(tag: str, split: str):
    from dataset.crossdocked_density_box import DatasetCrossDockedDensityBox

    _, resample = corpus_paths(tag)
    contract = json.loads((resample / "loader_contract.json").read_text())
    return DatasetCrossDockedDensityBox(
        box_path=contract["box_path"],
        resample_dir=str(resample),
        data_dir=str(DATA),
        data_file=contract["source_data_file"],
        split=split,
        aug=False,
        ligand_radius=-1,
        pocket_radius=-1,
        n_lig_ch=8,
        n_poc_ch=4,
        max_len=30,
        subset_n=int(contract["subset_n"]),
        subset_xray_only=True,
        subset_val_n=int(contract["subset_val_n"]),
        return_gradmag=True,
        cache_size=1,
    )


def loader_smoke(tag: str) -> dict:
    _, resample = corpus_paths(tag)
    contract = json.loads((resample / "loader_contract.json").read_text())
    output = {}
    for split, expected_n in (("train", int(contract["subset_n"])),
                              ("val", int(contract["subset_val_n"]))):
        dataset = make_dataset(tag, split)
        if len(dataset) != expected_n:
            raise RuntimeError(f"{tag} {split} length {len(dataset)} != {expected_n}")
        sample = dataset[len(dataset) - 1]
        if not bool(sample["xray_available"]):
            raise RuntimeError(f"{tag} {split} smoke sample unexpectedly lacks density")
        output[split] = {
            "n": len(dataset),
            "density_shape": list(sample["xray_density"].shape),
            "gradmag_shape": list(sample["xray_gradmag"].shape),
            "ligand_id": str(sample["ligand"]["id"]),
        }
        del sample, dataset
        gc.collect()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--corpus", default=DEFAULT_CORPUS,
                       help=f"corpus tag or resample dir (default: {DEFAULT_CORPUS})")
    group.add_argument("--all", action="store_true",
                       help="validate every corpus on disk that carries a contract")
    parser.add_argument("--loader-smoke", action="store_true",
                        help="also build the real datasets and pull a sample (slow)")
    args = parser.parse_args()

    tags = discover_corpora() if args.all else [args.corpus]
    result = {}
    for tag in tags:
        entry = {"metadata": validate_metadata(tag)}
        if args.loader_smoke:
            entry["loader"] = loader_smoke(tag)
        result[tag] = entry
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

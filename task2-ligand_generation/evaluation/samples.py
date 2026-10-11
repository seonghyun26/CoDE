"""Generated ligands, reference ligands and pockets, laid out the way the evaluation reads them.

A run directory holds one folder per CrossDocked test pocket:

    <run_dir>/target_NN/
        samples.sdf             one record per generated ligand; a record RDKit cannot sanitise,
                                or one with disconnected fragments, counts as invalid
        <ligand>.sdf            the crystal reference ligand
        <ligand>_pocket10.pdb   the 10 Å pocket crop the generator was conditioned on

NN is the pocket's index in dataset/data/data_test.pt, TargetDiff's test split. Baselines released
in TargetDiff's meta format are written into this layout by `stage_meta`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]
# The whole CrossDocked test receptors, <pocket folder>/<PDB id>_<chain>_rec.pdb.
TEST_SET = Path(os.environ.get("CROSSDOCKED_TEST_SET", str(REPO / "dataset/data/crossdocked/test_set")))
# Pocket crops and reference ligands of the 100 test pockets, in this layout (no samples.sdf).
TEST_POCKETS = Path(os.environ.get("CROSSDOCKED_TEST_POCKETS",
                                   str(REPO / "dataset/data/crossdocked/test_pockets")))
TARGET_LISTS = {"p79": Path(__file__).with_name("targets_p79.txt")}


def target_names(targets: Optional[str]) -> Optional[List[str]]:
    """A target list by name ("p79") or file; None or "all" means every target."""
    if targets in (None, "all"):
        return None
    path = TARGET_LISTS.get(targets, Path(targets))
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def target_dirs(run_dir, targets: Optional[str] = None) -> List[Path]:
    """The run's target_NN folders, restricted to a target list when one is given."""
    dirs = sorted(p for p in Path(run_dir).glob("target_*") if p.is_dir())
    keep = target_names(targets)
    return dirs if keep is None else [d for d in dirs if d.name in set(keep)]


def pocket_pdb(target_dir) -> Optional[Path]:
    return next(iter(sorted(Path(target_dir).glob("*_pocket10.pdb"))), None)


def reference_sdf(target_dir) -> Optional[Path]:
    return next((p for p in sorted(Path(target_dir).glob("*.sdf")) if p.name != "samples.sdf"), None)


def load_reference(target_dir) -> Optional[Chem.Mol]:
    """The crystal reference ligand (TargetDiff keeps one per pocket)."""
    path = reference_sdf(target_dir)
    if path is None:
        return None
    mols = [m for m in Chem.SDMolSupplier(str(path), sanitize=True) if m is not None]
    return mols[0] if mols else None


def full_receptor(target_dir) -> Optional[Path]:
    """The whole receptor a target's pocket crop was cut from, or None.

    Crops are named <pocket folder>__<ligand>_pocket10.pdb in our runs and <ligand>_pocket10.pdb in
    TargetDiff's evaluation tree; the receptor is <pocket folder>/<ligand[:10]>.pdb, i.e.
    <PDB id>_<chain>_rec.pdb. None rather than a fallback to the crop, since a run that mixed the
    two receptors could not be read.
    """
    crop = pocket_pdb(target_dir)
    if crop is None:
        return None
    base = crop.name[:-len("_pocket10.pdb")]
    ligand = base.split("__", 1)[1] if "__" in base else base
    name = ligand[:10] + ".pdb"
    if "__" in base:
        path = TEST_SET / base.split("__", 1)[0] / name
        if path.exists():
            return path
    hits = sorted(TEST_SET.glob("*/" + name))
    return hits[0] if len(hits) == 1 else None


def load_samples(target_dir) -> Tuple[int, List[Tuple[int, Chem.Mol]]]:
    """(number of records, [(record index, ligand)] for the valid ones)."""
    n, valid = 0, []
    sdf = Path(target_dir) / "samples.sdf"
    if not sdf.exists():
        return 0, []
    for i, mol in enumerate(Chem.SDMolSupplier(str(sdf), sanitize=True)):
        n += 1
        if mol is None:
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:  # noqa: BLE001
            continue
        if "." not in smiles:
            valid.append((i, mol))
    return n, valid


def parse_pocket(pdb_path) -> Tuple[np.ndarray, List[str]]:
    """Heavy-atom coordinates and elements of a pocket PDB."""
    coords, elements = [], []
    for line in open(pdb_path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        element = line[76:78].strip() if len(line) > 76 else ""
        if not element:
            element = "".join(c for c in line[12:16] if c.isalpha())[:2].capitalize()
        if element not in ("H", "D"):
            coords.append(xyz)
            elements.append(element)
    return np.array(coords, dtype=float), elements


def stage_meta(bundles: Sequence[str], out_dir, template=TEST_POCKETS, targets: Optional[str] = None,
               force: bool = False) -> Tuple[int, int]:
    """Write TargetDiff-format meta bundles into target_NN folders; returns (pockets, molecules).

    A bundle is a .pt list with one entry list per test pocket ({mol, ligand_filename, ...});
    several bundles (main, second half, re-run fill) are concatenated pocket by pocket. The pocket
    crop and reference ligand are copied from `template`. Entry i is checked against target_i's
    reference ligand name, and a mismatch aborts: molecules scored against the wrong pocket give
    numbers that look entirely reasonable.
    """
    import shutil

    import torch

    merged: List[list] = [[] for _ in range(100)]
    for bundle in bundles:
        for i, entries in enumerate(torch.load(bundle, weights_only=False)):
            merged[i].extend(entries)
    keep = target_names(targets)
    pockets = molecules = 0
    for i, entries in enumerate(merged):
        target = "target_{:02d}".format(i)
        if not entries or (keep is not None and target not in keep):
            continue
        src = Path(template) / target
        ref = reference_sdf(src)
        want = entries[0]["ligand_filename"].replace("/", "__")
        if ref is None or ref.name != want:
            raise SystemExit("{}: bundle entry is {}, template holds {}".format(
                target, want, ref.name if ref else "nothing"))
        dst = Path(out_dir) / target
        dst.mkdir(parents=True, exist_ok=True)
        for f in (ref, pocket_pdb(src)):
            if f is not None and not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
        sdf = dst / "samples.sdf"
        if sdf.exists() and not force:
            continue
        writer, n = Chem.SDWriter(str(sdf)), 0
        for entry in entries:
            if entry.get("mol") is not None:
                try:
                    writer.write(entry["mol"])
                    n += 1
                except Exception:  # noqa: BLE001 - an unwritable molecule is data, not a crash
                    pass
        writer.close()
        pockets, molecules = pockets + 1, molecules + n
    return pockets, molecules

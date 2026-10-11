"""Molecular quality: validity, uniqueness, QED, SA, logP, Lipinski, size, diversity and similarity
to the reference ligand.

QED, SA, logP and Lipinski come from TargetDiff's scoring_func.get_chem. Diversity is one minus the
mean pairwise Tanimoto similarity of RDKit fingerprints, the fingerprint TargetDiff's similarity.py
uses. Similarity to the reference ligand is reported for Morgan (radius 2, 2048 bits) and RDKit
fingerprints.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from . import TARGETDIFF  # noqa: F401 - puts the vendored TargetDiff on sys.path
from utils.evaluation.scoring_func import get_chem  # noqa: E402


def chem(mol: Chem.Mol) -> Dict:
    c = get_chem(mol)
    return {"smiles": Chem.MolToSmiles(mol), "qed": float(c["qed"]), "sa": float(c["sa"]),
            "logp": float(c["logp"]), "lipinski": int(c["lipinski"]), "n_atoms": int(mol.GetNumHeavyAtoms())}


def diversity(mols: List[Chem.Mol]) -> float:
    if len(mols) < 2:
        return float("nan")
    fps = [Chem.RDKFingerprint(m) for m in mols]
    return 1.0 - float(np.mean([DataStructs.TanimotoSimilarity(a, b) for a, b in itertools.combinations(fps, 2)]))


def _morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def reference_similarity(mols: List[Chem.Mol], ref: Optional[Chem.Mol]) -> Dict:
    """ref_tanimoto_* (Morgan) and ref_tanimoto_rdk_* (RDKit): mean, max and the share above 0.4 / 0.8."""
    if ref is None or not mols:
        return {}
    out = {}
    for key, fp in (("ref_tanimoto", _morgan), ("ref_tanimoto_rdk", Chem.RDKFingerprint)):
        r = fp(ref)
        sims = np.array([DataStructs.TanimotoSimilarity(fp(m), r) for m in mols])
        out.update({key + "_mean": float(sims.mean()), key + "_max": float(sims.max()),
                    key + "_frac_gt0.4": float((sims > 0.4).mean()),
                    key + "_frac_gt0.8": float((sims > 0.8).mean())})
    return out


def target_quality(n_total: int, mols: List[Chem.Mol], rows: List[Dict], ref: Optional[Chem.Mol]) -> Dict:
    """Per-pocket quality from the valid ligands and their `chem` rows."""
    n_valid = len(mols)
    n_unique = len({r["smiles"] for r in rows})
    out = {"n_total": n_total, "n_valid": n_valid, "n_unique": n_unique,
           "validity": n_valid / n_total if n_total else float("nan"),
           "uniqueness": n_unique / n_valid if n_valid else float("nan"),
           "diversity": diversity(mols)}
    for key in ("qed", "sa", "logp", "lipinski", "n_atoms"):
        out[key] = float(np.mean([r[key] for r in rows])) if rows else float("nan")
    out.update(reference_similarity(mols, ref))
    return out

"""PoseCheck (Harris et al., 2023): steric clashes, strain energy and protein-ligand interactions.

Runs in the evaluation environment through pose_worker.py. Strain follows posecheck 1.3.1's
calculate_strain_energy: the energy after a position-restrained UFF relaxation minus the best of 50
re-embedded, minimised conformers. That definition gives values about ten times smaller than the
pre-2024 one used in published tables, so compare only against baselines scored here.
"""
from __future__ import annotations

import math
import os
import signal
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# A per-ligand wall-clock cap on strain: UFF relaxation can spin for minutes on a bad pose.
STRAIN_TIMEOUT_S = float(os.environ.get("POSE_STRAIN_TIMEOUT_S", "60"))


class _Deadline:
    """SIGALRM wall-clock cap (the worker is single-threaded)."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds and self.seconds > 0:
            signal.signal(signal.SIGALRM, self._fire)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        return False

    def _fire(self, *_):
        raise TimeoutError("exceeded {:g}s".format(self.seconds))


def _interactions(df) -> Dict:
    """Per-type counts of a one-ligand prolif fingerprint (columns: ligand, residue, type)."""
    counts: Dict[str, int] = {}
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        return {"n_interactions": 0, "interactions": {}}
    row = df.iloc[0]
    for col in df.columns:
        kind = col[-1] if isinstance(col, tuple) else str(col)
        try:
            present = bool(row[col])
        except Exception:  # noqa: BLE001
            present = False
        if present:
            counts[kind] = counts.get(kind, 0) + 1
    return {"n_interactions": int(sum(counts.values())), "interactions": counts}


def _seed_strain(seed: int = 0) -> None:
    """Seed posecheck's conformer embedding, which upstream leaves unseeded.

    Without it the same molecule gives a different strain on every run (the reference-ligand median
    moved 33.8 -> 34.6 between two runs). Wraps EmbedMolecule in posecheck's namespace rather than
    editing the installed package.
    """
    from posecheck.utils import strain

    if getattr(strain.AllChem.EmbedMolecule, "_seeded", False):
        return
    original = strain.AllChem.EmbedMolecule

    def seeded(mol, *args, **kwargs):
        kwargs.setdefault("randomSeed", seed)
        return original(mol, *args, **kwargs)

    seeded._seeded = True
    strain.AllChem.EmbedMolecule = seeded


def _strain(pc, mol) -> Tuple[Optional[float], Optional[str]]:
    try:
        with _Deadline(STRAIN_TIMEOUT_S):
            pc.load_ligands_from_mols([mol])
            s = pc.calculate_strain_energy()[0]
        ok = isinstance(s, (int, float)) and math.isfinite(s)
        return (float(s) if ok else None, None)
    except Exception as e:  # noqa: BLE001 - a timeout or any UFF/RDKit failure
        return None, "{}: {}".format(type(e).__name__, e)


def sanitise_receptor(pdb_path: str) -> Optional[str]:
    """A copy prolif can load, or None when nothing needed changing.

    Two features of stock CrossDocked receptors stop the load, costing the whole pocket: HETATM
    metals and cofactors (prolif has no van der Waals radius for e.g. cobalt), and negative residue
    numbers (cast to uint32). HETATM records are dropped and residue numbers shifted to start at 1.
    """
    lines = Path(pdb_path).read_text().splitlines()
    kept = [ln for ln in lines if not ln.startswith("HETATM")]
    records = ("ATOM", "ANISOU", "TER")
    nums = []
    for ln in kept:
        if ln.startswith(records):
            try:
                nums.append(int(ln[22:26]))
            except ValueError:
                pass
    shift = (1 - min(nums)) if nums and min(nums) < 1 else 0
    if len(kept) == len(lines) and shift == 0:
        return None
    if nums and shift and max(nums) + shift > 9999:
        return None
    out = []
    for ln in kept:
        if shift and ln.startswith(records):
            try:
                ln = "{}{:4d}{}".format(ln[:22], int(ln[22:26]) + shift, ln[26:])
            except ValueError:
                pass
        out.append(ln)
    fd, tmp = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return tmp


def score(protein_pdb: str, mols: List) -> List[Dict]:
    """{clashes, strain, n_interactions, interactions} per ligand, aligned with `mols`.

    The protein is protonated and loaded once; clashes and the interaction fingerprint are one batch;
    strain is per ligand under the deadline.
    """
    from posecheck import PoseCheck

    _seed_strain()
    pc = PoseCheck()
    try:
        pc.load_protein_from_pdb(protein_pdb)
    except Exception as e:  # noqa: BLE001
        first = "{}: {}".format(type(e).__name__, e)
        clean = sanitise_receptor(protein_pdb)
        if clean is None:
            return [{"error": "protein load failed: " + first} for _ in mols]
        try:
            pc = PoseCheck()
            pc.load_protein_from_pdb(clean)
        except Exception as e2:  # noqa: BLE001
            return [{"error": "protein load failed: {}; after sanitise: {}: {}".format(
                first, type(e2).__name__, e2)} for _ in mols]
        finally:
            try:
                os.remove(clean)
            except OSError:
                pass
    valid = [(i, m) for i, m in enumerate(mols) if m is not None]
    out: List[Dict] = [{"error": "unparsable mol"} for _ in mols]
    if not valid:
        return out
    pc.load_ligands_from_mols([m for _, m in valid])
    try:
        clashes = pc.calculate_clashes()
    except Exception:  # noqa: BLE001
        clashes = [None] * len(valid)
    try:
        fingerprint = pc.calculate_interactions()
    except Exception:  # noqa: BLE001
        fingerprint = None
    for k, (i, m) in enumerate(valid):
        rec: Dict = {}
        c = clashes[k] if k < len(clashes) else None
        rec["clashes"] = int(c) if isinstance(c, (int, float)) else None
        s, err = _strain(pc, m)
        rec["strain"] = s
        if err is not None:
            rec.setdefault("errors", {})["strain"] = err
        try:
            rec.update(_interactions(fingerprint.iloc[[k]]
                                     if fingerprint is not None and k < fingerprint.shape[0] else None))
        except Exception as e:  # noqa: BLE001
            rec.setdefault("errors", {})["interactions"] = "{}: {}".format(type(e).__name__, e)
        out[i] = rec
    return out

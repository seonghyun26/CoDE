"""AutoDock Vina: score_only, minimize and dock affinities, through TargetDiff's VinaDockingTask.

The paper protocol (Table 3): the whole receptor (<PDB id>_<chain>_rec.pdb), AutoDock Vina 1.2.2,
exhaustiveness 32, a box centred on the ligand with a 5 Å margin (VinaDockingTask's default), and
the 79 pockets with experimental density. The crystal reference ligand is docked the same way.

High affinity is the share of a pocket's valid ligands whose dock score is lower than the reference
ligand's, then averaged (and medianed) over pockets; TargetDiff's own scripts count it differently.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from . import TARGETDIFF  # noqa: F401 - puts the vendored TargetDiff on sys.path

# Vina 1.2.2's python binding still uses np.int, which numpy 1.24 removed (voxeval has 1.26); restore
# the alias, as the paper's docking driver did.
if not hasattr(np, "int"):
    np.int = int

STAGES = ("score_only", "minimize", "dock")
KEYS = {"score_only": "vina_score", "minimize": "vina_min", "dock": "vina_dock"}


def _heal_path() -> None:
    """Put this interpreter's bin first on PATH.

    TargetDiff's docking code shells out to bare `python3`, `pdb2pqr30` and `vina`. When another
    python sits earlier on PATH, prepare_receptor4.py runs without MolKit, writes no .pdbqt and
    reports nothing, and docking then fails on a missing file.
    """
    bin_dir = str(Path(sys.executable).parent)
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p != bin_dir]
    os.environ["PATH"] = os.pathsep.join([bin_dir] + parts)


def available() -> Optional[str]:
    """None when the docking toolchain imports, else the reason it does not."""
    try:
        from utils.evaluation.docking_vina import VinaDockingTask  # noqa: F401
    except Exception as e:  # noqa: BLE001 - ImportError or a native-library load error
        return "{}: {}".format(type(e).__name__, e)
    return None


def prepare_receptor(receptor: Path) -> Optional[str]:
    """Build the receptor .pqr and .pdbqt once, before the docking pool; returns a note or None.

    VinaDockingTask prepares them lazily behind os.path.exists, which races when many workers hit a
    cold cache. Preparing here also surfaces the real cause of a failure (pdb2pqr30, MolKit) instead
    of a later missing-file error. When pdb2pqr30 rejects a pocket spanning non-contiguous chains
    ("gap in biomolecule structure"), the .pdb goes to prepare_receptor4.py directly with Gasteiger
    charges, and the returned note says so.
    """
    import AutoDockTools

    _heal_path()
    base = str(receptor)[:-4]
    pqr, pdbqt = base + ".pqr", base + ".pdbqt"
    if os.path.exists(pdbqt):
        return "gasteiger fallback" if os.path.exists(base + ".fallback") else None
    script = os.path.join(AutoDockTools.__path__[0], "Utilities24/prepare_receptor4.py")
    log: List[str] = []

    def run(cmd: List[str], stage: str):
        r = subprocess.run(cmd, capture_output=True, text=True)
        log.append("=== {} (rc={})\n$ {}\n{}\n{}".format(stage, r.returncode, " ".join(cmd), r.stdout, r.stderr))
        return r

    def fail(stage: str, r) -> None:
        Path(base + ".prep.log").write_text("\n".join(log))
        lines = [ln for ln in r.stderr.strip().splitlines()
                 if ln and not ln.startswith((" ", "\t", "File ", "Traceback"))]
        raise RuntimeError("{} failed for {} (rc={}): {}".format(
            stage, Path(receptor).name, r.returncode, "; ".join(lines[-4:]) or "see " + base + ".prep.log"))

    fallback = False
    if not os.path.exists(pqr):
        r = run(["pdb2pqr30", "--ff=AMBER", str(receptor), pqr], "pdb2pqr30")
        if r.returncode != 0 or not os.path.exists(pqr):
            if "gap in biomolecule" not in r.stderr.lower():
                fail("pdb2pqr30", r)
            fallback = True
    if fallback:
        cmd = [sys.executable, script, "-r", str(receptor), "-o", pdbqt, "-A", "hydrogens",
               "-U", "nphs_lps_waters"]
    else:
        cmd = [sys.executable, script, "-r", pqr, "-o", pdbqt]
    r = run(cmd, "prepare_receptor4.py")
    if r.returncode != 0 or not os.path.exists(pdbqt):
        fail("prepare_receptor4.py", r)
    Path(base + ".prep.log").write_text("\n".join(log))
    if fallback:
        Path(base + ".fallback").write_text("pdb2pqr30 hit a chain gap; Gasteiger charges\n")
        return "gasteiger fallback"
    return None


def _pdbqt_atoms(text: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    return [(ln[76:79].strip() or ln[12:16].strip(), (float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]


def _moved(template: Chem.Mol, ligand_pdbqt: str, pose: str) -> Optional[Chem.Mol]:
    """`template` with the docked pose's coordinates, or None when the atoms cannot be matched.

    Only coordinates cross over: reading Vina's PDBQT back through a bond perceiver changes the
    molecule (aromatic rings fail to kekulise and the SMILES drifts). The prepared .pdbqt keeps the
    input's heavy-atom positions and Vina keeps atom order, so matching by position is exact.
    """
    src, dst = _pdbqt_atoms(ligand_pdbqt), _pdbqt_atoms(pose)
    if len(src) != len(dst):
        return None
    pos = template.GetConformer().GetPositions()
    used, moves = set(), {}
    for i, (_, xyz) in enumerate(src):
        d = np.linalg.norm(pos - np.asarray(xyz), axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-2:              # a hydrogen added during preparation
            continue
        if j in used:
            return None
        used.add(j)
        moves[j] = dst[i][1]
    if len(moves) != template.GetNumHeavyAtoms():
        return None
    out = Chem.Mol(template)
    conf = out.GetConformer()
    for j, xyz in moves.items():
        conf.SetAtomPosition(j, Point3D(*xyz))
    return out


def _pose_block(mol: Chem.Mol, ligand_pdbqt: Optional[str], pose: Optional[str], affinity: float,
                stage: str, row: int) -> Optional[str]:
    """One stage's conformer as an SDF block tagged with its row; None when it cannot be mapped."""
    try:
        base = Chem.RemoveHs(mol)
    except Exception:  # noqa: BLE001
        base = mol
    out = base if pose is None else _moved(base, ligand_pdbqt or "", pose)
    if out is None:
        return None
    out = Chem.Mol(out)
    out.SetProp("row", str(row))
    out.SetProp("vina_stage", stage)
    out.SetProp("vina_" + stage, "{:.3f}".format(affinity))
    buf = io.StringIO()
    writer = Chem.SDWriter(buf)
    writer.write(out)
    writer.close()
    return buf.getvalue()


def dock(mol: Chem.Mol, receptor, row: int = -1, stages=STAGES, exhaustiveness: int = 32,
         cpu: int = 4, tmp_dir="./tmp") -> Dict:
    """Vina affinities of one ligand, {"score_only", "minimize", "dock", "_poses"} or {"error"}.

    Never raises, so a failure stays local to its ligand. `_poses` carries the SDF blocks of the
    conformers each stage produced; callers strip it before storing the row.
    """
    _heal_path()
    try:
        from utils.evaluation.docking_vina import VinaDockingTask
    except Exception as e:  # noqa: BLE001
        return {"error": "toolchain unavailable: {}: {}".format(type(e).__name__, e)}
    try:
        task = VinaDockingTask(str(receptor), deepcopy(mol), tmp_dir=str(tmp_dir))
        result, poses = {}, {}
        for stage in stages:
            r = task.run(mode=stage, exhaustiveness=exhaustiveness, cpu=cpu)[0]
            affinity = float(r["affinity"])
            raw, text, lig = r.get("pose"), None, None
            if raw is not None:      # score_only scores the pose where it sits and returns none
                text = raw[0] if isinstance(raw, (list, tuple)) else raw
                try:
                    with open(task.ligand_path[:-4] + ".pdbqt") as fh:
                        lig = fh.read()
                except OSError:
                    text = None
            block = _pose_block(mol, lig, text, affinity, stage, row)
            if block:
                poses[stage] = block
            result[stage] = affinity
        if poses:
            result["_poses"] = poses
        return result
    except Exception as e:  # noqa: BLE001
        return {"error": "{}: {}".format(type(e).__name__, e)}


def dock_all(mols: List[Tuple[int, Chem.Mol]], receptor: Path, cache_dir: Path, stages=STAGES,
             exhaustiveness: int = 32, cpu: int = 4, workers: int = 8) -> Tuple[Dict[int, Dict], Optional[str]]:
    """Dock [(row, ligand)] against a receptor copied into cache_dir; ({row: result}, prep note).

    The receptor is prepared once up front so the pool's workers only read it. When preparation
    fails, every ligand still runs and records the error.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / Path(receptor).name
    if not cached.exists():
        shutil.copy(receptor, cached)
    note = None
    try:
        note = prepare_receptor(cached)
    except Exception as e:  # noqa: BLE001
        note = "receptor preparation failed: {}: {}".format(type(e).__name__, e)
        workers = 1
    run = partial(dock, receptor=cached, stages=stages, exhaustiveness=exhaustiveness, cpu=cpu,
                  tmp_dir=cache_dir)
    results: Dict[int, Dict] = {}
    if not mols:
        return results, note
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(mols)))) as pool:
        futures = {row: pool.submit(run, mol, row=row) for row, mol in mols}
        for row, fut in futures.items():
            results[row] = fut.result()
    return results, note


def write_poses(results: Dict[int, Dict], out_dir: Path) -> None:
    """Strip `_poses` from the results and write vina_poses/<stage>.sdf, rows in order."""
    blocks = {s: [] for s in STAGES}
    for row in sorted(results):
        for stage, block in (results[row].pop("_poses", None) or {}).items():
            blocks[stage].append(block)
    if any(blocks.values()):
        (out_dir / "vina_poses").mkdir(exist_ok=True)
        for stage, items in blocks.items():
            if items:
                (out_dir / "vina_poses" / (stage + ".sdf")).write_text("".join(items))


def high_affinity(dock_scores: List[float], ref_dock: Optional[float]) -> Optional[float]:
    """Share of a pocket's docked ligands that dock better (lower) than the reference ligand."""
    if ref_dock is None or not dock_scores:
        return None
    return float(np.mean([s < ref_dock for s in dock_scores]))

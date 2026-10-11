"""PoseCheck and PoseBusters in a worker process.

`score_poses` writes a chunk of ligands to an SDF and runs this module on it in a separate process,
so that a UFF relaxation that hangs is cut off by the timeout:

    <python> -m evaluation.pose_worker --protein P.pdb --ligands L.sdf --out O.json --which all

The worker uses the running interpreter when it has the pose tools (the voxeval environment:
posecheck 1.3.1, posebusters 0.6.5, prolif, reduce), else the `moleval` environment next to it, which
held them before voxeval; $MOLEVAL_PY overrides both.

The worker writes one {"posecheck": ..., "posebusters": ...} record per SDF record, in order. Poses
are scored as written, without relaxation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TASK2 = Path(__file__).resolve().parents[1]


def _pose_python() -> str:
    """$MOLEVAL_PY, else this interpreter if it has the pose tools, else the sibling `moleval` env."""
    if os.environ.get("MOLEVAL_PY"):
        return os.environ["MOLEVAL_PY"]
    import importlib.util
    if all(importlib.util.find_spec(m) for m in ("posecheck", "posebusters")):
        return sys.executable
    return str(Path(sys.executable).parents[2] / "moleval" / "bin" / "python")


MOLEVAL_PY = _pose_python()
TIMEOUT_S = float(os.environ.get("POSE_TIMEOUT_S", "600"))   # per chunk
CHUNK = int(os.environ.get("POSE_CHUNK_SIZE", "20"))           # ligands per worker call
TOOLS = ("posecheck", "posebusters")


def score_poses(mols: List, protein_pdb, which: Sequence[str] = TOOLS, tmp_dir=None,
                chunk: int = CHUNK) -> List[Dict]:
    """Pose records aligned with `mols`. A failed chunk records its error and the others are kept."""
    if not os.path.exists(MOLEVAL_PY):
        reason = "no python with posecheck and posebusters: {} (use voxeval or set $MOLEVAL_PY)".format(MOLEVAL_PY)
        return [{t: {"error": reason} for t in which} for _ in mols]
    from rdkit import Chem

    out: List[Dict] = []
    for start in range(0, len(mols), chunk):
        part = mols[start:start + chunk]
        tmp = Path(tempfile.mkdtemp(prefix="pose_", dir=str(tmp_dir) if tmp_dir else None))
        try:
            with Chem.SDWriter(str(tmp / "ligands.sdf")) as writer:
                for m in part:
                    writer.write(m)
            env = dict(os.environ, PYTHONPATH=str(TASK2),
                       PATH=os.pathsep.join([str(Path(MOLEVAL_PY).parent), os.environ.get("PATH", "")]))
            proc = subprocess.run(
                [MOLEVAL_PY, "-m", "evaluation.pose_worker", "--protein", str(protein_pdb),
                 "--ligands", str(tmp / "ligands.sdf"), "--out", str(tmp / "out.json"),
                 "--which", ",".join(which)],
                cwd=str(TASK2), env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
            if proc.returncode != 0 or not (tmp / "out.json").exists():
                raise RuntimeError("worker rc={}: {}".format(
                    proc.returncode, " | ".join((proc.stderr or "").strip().splitlines()[-3:])))
            records = json.loads((tmp / "out.json").read_text())
            if len(records) != len(part):
                raise RuntimeError("{} records for {} ligands".format(len(records), len(part)))
            out.extend(records)
        except Exception as e:  # noqa: BLE001 - includes subprocess.TimeoutExpired
            reason = "mols {}-{}: {}: {}".format(start + 1, start + len(part), type(e).__name__, e)
            out.extend({t: {"error": reason} for t in which} for _ in part)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    from rdkit import Chem, RDLogger

    from . import posebusters, posecheck

    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--ligands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default=",".join(TOOLS))
    args = ap.parse_args(argv)
    which = args.which.split(",")
    mols = list(Chem.SDMolSupplier(args.ligands, sanitize=True))   # None stays in place for a bad record
    scored = {"posecheck": posecheck.score, "posebusters": posebusters.score}
    results = {t: scored[t](args.protein, mols) for t in which}
    records = [{t: results[t][i] for t in which} for i in range(len(mols))]
    Path(args.out).write_text(json.dumps(records))
    print("pose_worker: {} records -> {}".format(len(records), args.out), file=sys.stderr)


if __name__ == "__main__":
    main()

"""Evaluate generated ligands: molecular quality, AutoDock Vina, PoseCheck and PoseBusters.

    cd task2-ligand_generation
    python -m evaluation.eval <run_dir> [--metrics quality,vina,posecheck,posebusters] [--targets p79]
    python -m evaluation.eval <run_dir> --meta <bundle.pt> [<bundle.pt> ...]   # stage a baseline first

<run_dir> holds one target_NN/ folder per test pocket with samples.sdf, the reference ligand and
the pocket crop (samples.py). Each target gets eval.json, reused by --skip-existing; the run gets
eval_results.json with every target's row and the Table 3 summary. The defaults are the paper's
protocol: the whole receptor, exhaustiveness 32 and the 79 pockets with density. Run it in the
`voxeval` environment (AutoDock Vina 1.2.2, PoseCheck, PoseBusters; env/voxeval.environment.yml).

Table 3 aggregation: Vina score/min/dock, QED, SA and atom count over all valid ligands, mean and
median; diversity and high affinity per pocket, then mean and median over pockets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rdkit import RDLogger

from . import quality, samples, vina
from .pose_worker import TOOLS, score_poses

RDLogger.DisableLog("rdApp.*")
METRICS = ("quality", "vina") + TOOLS
POOLED = ("vina_score", "vina_min", "vina_dock", "qed", "sa", "n_atoms")
PER_POCKET = ("diversity", "high_affinity")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _receptor(target_dir: Path, scope: str) -> Optional[Path]:
    return samples.full_receptor(target_dir) if scope == "full" else samples.pocket_pdb(target_dir)


def evaluate_target(target_dir: Path, metrics, args) -> Dict:
    """One pocket: quality, then Vina and pose scores for whatever the cache lacks."""
    t0 = time.time()
    settings = {"dock_scope": args.dock_scope, "pose_scope": args.pose_scope,
                "exhaustiveness": args.exhaustiveness}
    cache_path = target_dir / "eval.json"
    cached = json.loads(cache_path.read_text()) if args.skip_existing and cache_path.exists() else {}
    if cached.get("settings") != settings:
        cached = {}
    old = {(m["idx"], m["smiles"]): m for m in cached.get("per_mol", [])}

    n_total, valid = samples.load_samples(target_dir)
    ref = samples.load_reference(target_dir)
    per_mol = []
    for idx, mol in valid:
        row = quality.chem(mol)
        per_mol.append(dict(old.get((idx, row["smiles"]), {}), idx=idx, **row))
    mols = [m for _, m in valid]
    out = {"target": target_dir.name, "settings": settings}
    out.update(quality.target_quality(n_total, mols, per_mol, ref))
    ref_file = samples.reference_sdf(target_dir)
    out["ref_ligand"] = ref_file.name if ref_file else None
    reference = dict(cached.get("reference", {}))
    if ref is not None:
        reference.update(quality.chem(ref))

    if "vina" in metrics:
        receptor = _receptor(target_dir, args.dock_scope)
        if receptor is None:
            raise FileNotFoundError("no {} receptor for {}".format(args.dock_scope, target_dir))
        out.update(receptor=receptor.name, receptor_scope=args.dock_scope)
        todo = [(m["idx"], mol) for m, (_, mol) in zip(per_mol, valid) if not _finite(m.get("vina_dock"))]
        if ref is not None and not _finite(reference.get("vina_dock")):
            todo.append((-1, ref))
        results, note = vina.dock_all(todo, receptor, target_dir / ".vina_cache",
                                      exhaustiveness=args.exhaustiveness, cpu=args.cpu, workers=args.workers)
        vina.write_poses(results, target_dir)
        by_idx = {m["idx"]: m for m in per_mol}
        for idx, res in results.items():
            dst = reference if idx == -1 else by_idx[idx]
            if "error" in res:
                dst["vina_error"] = res["error"]
                continue
            dst.pop("vina_error", None)
            for stage, key in vina.KEYS.items():
                dst[key] = res[stage]
        if note:
            out["receptor_prep"] = note
        for key in vina.KEYS.values():
            vals = [m[key] for m in per_mol if _finite(m.get(key))]
            out[key] = float(np.mean(vals)) if vals else None
            out["ref_" + key] = reference.get(key)
        out["high_affinity"] = vina.high_affinity(
            [m["vina_dock"] for m in per_mol if _finite(m.get("vina_dock"))], reference.get("vina_dock"))

    tools = [t for t in TOOLS if t in metrics]
    if tools:
        receptor = _receptor(target_dir, args.pose_scope)
        rows = per_mol + ([reference] if ref is not None else [])
        ligands = mols + ([ref] if ref is not None else [])
        todo = [i for i, r in enumerate(rows) if any(not isinstance(r.get(t), dict) or "error" in r[t] for t in tools)]
        if receptor is None:
            for i in todo:
                rows[i].update({t: {"error": "no {} receptor".format(args.pose_scope)} for t in tools})
        elif todo:
            scored = score_poses([ligands[i] for i in todo], receptor, which=tools)
            for i, rec in zip(todo, scored):
                rows[i].update(rec)

    out["per_mol"] = per_mol
    out["reference"] = reference
    out["_secs"] = round(time.time() - t0, 1)
    cache_path.write_text(json.dumps(out, indent=1, default=float))
    return out


def summarize(rows: List[Dict]) -> Dict:
    """Table 3 numbers, plus the mean over pockets of every per-pocket scalar."""
    table: Dict = {"n_targets": len(rows), "n_valid": int(sum(r["n_valid"] for r in rows)),
                   "n_unique": int(sum(r["n_unique"] for r in rows))}
    for key in POOLED:
        vals = [m[key] for r in rows for m in r["per_mol"] if _finite(m.get(key))]
        if vals:
            table[key] = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
    for key in PER_POCKET:
        vals = [r[key] for r in rows if _finite(r.get(key))]
        if vals:
            table[key] = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
    pc = [m["posecheck"] for r in rows for m in r["per_mol"] if isinstance(m.get("posecheck"), dict)]
    for key in ("clashes", "strain"):
        vals = [p[key] for p in pc if _finite(p.get(key))]
        if vals:
            table[key] = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
    pb = [m["posebusters"]["valid"] for r in rows for m in r["per_mol"]
          if isinstance(m.get("posebusters"), dict) and isinstance(m["posebusters"].get("valid"), bool)]
    if pb:
        table["pb_valid"] = float(np.mean(pb))
    keys = sorted({k for r in rows for k, v in r.items() if _finite(v) and not k.startswith("_")})
    mean_over_pockets = {k: float(np.mean([r[k] for r in rows if _finite(r.get(k))])) for k in keys}
    return {"table3": table, "mean_over_pockets": mean_over_pockets}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--metrics", default="quality,vina", help="any of " + ",".join(METRICS))
    ap.add_argument("--targets", default="p79", help="p79, all, or a file of target names")
    ap.add_argument("--meta", nargs="+", help="TargetDiff meta bundles to stage into run_dir first")
    ap.add_argument("--dock-scope", choices=("full", "crop"), default="full")
    ap.add_argument("--pose-scope", choices=("full", "crop"), default="crop")
    ap.add_argument("--exhaustiveness", type=int, default=32)
    ap.add_argument("--cpu", type=int, default=4, help="Vina threads per ligand")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 8),
                    help="ligands docked in parallel")
    ap.add_argument("--skip-existing", action="store_true", help="reuse each target's eval.json")
    ap.add_argument("--out", type=Path, help="default: <run_dir>/eval_results.json")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise SystemExit("unknown metrics {}; choose from {}".format(unknown, METRICS))
    if "vina" in metrics and vina.available():
        raise SystemExit("Vina toolchain unavailable ({}); run it in the voxeval environment".format(vina.available()))
    if args.meta:
        n_pockets, n_mols = samples.stage_meta(args.meta, args.run_dir, targets=args.targets)
        print("staged {} pockets, {} molecules into {}".format(n_pockets, n_mols, args.run_dir), flush=True)

    dirs = samples.target_dirs(args.run_dir, args.targets)
    if not dirs:
        raise SystemExit("no target_* folders in {} for targets={}".format(args.run_dir, args.targets))
    rows = []
    for i, d in enumerate(dirs, 1):
        row = evaluate_target(d, metrics, args)
        rows.append(row)
        print("[{}/{}] {} valid {}/{} {}({}s)".format(
            i, len(dirs), d.name, row["n_valid"], row["n_total"],
            "dock {:.2f} ".format(row["vina_dock"]) if _finite(row.get("vina_dock")) else "", row["_secs"]),
            flush=True)
    result = {"run_dir": str(args.run_dir), "metrics": metrics, "targets": args.targets}
    result.update(summarize(rows))
    result["per_target"] = rows
    out = args.out or args.run_dir / "eval_results.json"
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result["table3"], indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

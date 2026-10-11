#!/usr/bin/env python3
"""web/build.py — a static, server-less browser for the task2 samples.

WHAT IT IS. One pocket, every method side by side: the generated ligands as 2D
depictions and 3D poses, each carrying its own Vina / QED / SA / PoseCheck /
PoseBusters numbers. The task-2 results collector already put the *numbers* in one shape
(`<Method>/eval/<evaluation>/per_molecule.csv`); what was still missing is the
*molecules* those rows describe, in a form a browser can open. This writes both out
as per-target shards next to a single-page viewer:

    web/site/
        index.html                       the viewer (built from web/src/index.html)
        data/manifest.json               methods, targets, metric schema, provenance
        data/targets/target_NN.json      every method's molecules for that pocket,
                                         columnar: {cols: [...], rows: [[...], ...]}
        data/struct/target_NN/
            pocket.pdb                   the shared 10 A crop
            ref.sdf                      the deposited crystal ligand
            <Method>.sdf                 record i  ==  that method's mol_idx i

No server and no build step at view time: `python3 -m http.server` in the output
directory, or just open index.html.

THE CANONICAL MOLECULE ORDER IS THE META, NOT THE _eval DIRECTORY -- the same rule
the task-2 results collector establishes, and for the same reason: the five published
baselines' dumps are split across <M>, <M>_part2 and <M>_gap, and walking them in
directory order pairs the wrong molecules. This reproduces that loop exactly, so SDF
record i is the row whose mol_idx is i.

THE JOIN IS MEASURED, NEVER ASSUMED. For every (method, evaluation) this compares the
table's SMILES at mol_idx against the structure's own SMILES and records the agreement.
At >= JOIN_OK it joins on mol_idx; below that it falls back to (target, smiles) and says
so in the manifest, because some tables really are mis-ordered against the others
(TargetDiff's and VoxBind-Ours' docking passes ran months apart from their pose passes --
33% SMILES agreement on TargetDiff). A value that cannot be matched either way is left
empty, never guessed and never `nan`.

ROSTER IS DISCOVERED, NOT HARDCODED. Any folder under results/task2-drugdesign/ that has
both per-molecule tables and molecules on disk becomes a column. On this box that is the
six published baselines; drop another method's folder into the bundle and re-run.

    python3 web/build.py                      # write web/site/
    python3 web/build.py --limit-targets 3    # smoke
    python3 web/build.py --check              # verify only

Needs rdkit + torch: run it in the voxbind env (`conda run -n voxbind python web/build.py`).
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

# gemmi-before-torch is the project rule for scripts that use both; we use neither
# together here, but torch must come before rdkit's conformer code touches numpy.
import glob
import re
import types

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds, rdMolAlign
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve()
WEB = HERE.parent                      # everything this tool owns lives under web/
ROOT = next(p for p in WEB.parents if (p / "voxbind" / "__init__.py").is_file())
TASK2 = ROOT / "results" / "task2-drugdesign"
TEMPLATE = WEB / "src" / "index.html"  # the viewer source
DEFAULT_OUT = WEB / "site"             # the built, servable site
THIS_BOX = socket.gethostname()

# Folder -> the meta bundle's stem. Everything else is discovered.
META_NAME = {
    "AR": "AR", "Pocket2Mol": "Pocket2Mol", "DiffSBDD": "DiffSBDD",
    "DecompDiff": "DecompDiff_ref_prior", "FuncBind": "FuncBind",
}
META_SUFFIXES = ("", "_part2", "_gap")

# Our own arms were generated on another box; what sits in this bundle under these names
# is a partial copy whose provenance is a different machine. Baselines only by default.
OURS = {"VoxBind", "VoxBind-Ours", "Ours-v2",
        "VoxBind-base-ep350", "VoxBind-base-ep350-n100"}

# A separately sampled DecompDiff run (reference prior, 25 per pocket) whose own
# evaluation kept Vina's poses: the one place on this box with generated, minimised
# AND docked coordinates for the same molecules. Its molecules are not the bundle's
# DecompDiff ones (SMILES overlap 0-1 per pocket), so it is a method of its own.
EVAL_PT_METHODS = {
    "DecompDiff-prior25": ROOT / "base" / "decompdiff" / "eval_results_refprior25",
}
EVAL_PT_NOTE = ("separate DecompDiff sampling run (reference prior, 25 per pocket); "
                "Vina scores and poses from DecompDiff's own evaluation; "
                "no PoseCheck or PoseBusters for it")

# Below this SMILES agreement a table is not joined on mol_idx.
JOIN_OK = 0.99

# The deposited ligand's own Vina Dock, keyed by CrossDocked identity. The donor's
# metrics.json reference block carries QED/SA/PoseCheck but no affinity, and a Reference
# column with an empty Vina cell is the one number a reader most wants there. This came
# out of a different pass (DecompDiff's reference docking), so the manifest says so
# rather than letting it pass for a value from the methods' own run.
REFERENCE_VINA = ROOT / "base" / "decompdiff" / "reference_vina.json"

EVALUATIONS = ("vina_docking", "sample_quality", "posecheck", "posebusters")

# The three poses a generated molecule can be looked at in. Only the generated one is
# derivable from what this bundle holds: the per-molecule tables carry Vina's SCORES
# but never its coordinates, and no .pdbqt or docked/minimised SDF exists anywhere
# under results/task2-drugdesign. Re-docking would produce them, and it needs the full
# receptor, which is not on this box either -- so the other two are declared and left
# unavailable rather than silently shown as the generated pose.
#
# The drop-in contract for when they do get produced:
#     <Method>/poses/<minimized|docked>/target_NN.sdf
# one record per molecule, in the SAME order as the generated SDF -- record i is the
# row whose mol_idx is i. A file that breaks that pairs the wrong pose with the wrong
# numbers, so the builder checks the count and refuses a mismatch.
# Order is the order the viewer shows them in, and the first AVAILABLE one is the
# default -- docked is what the headline Vina number is measured on, so it leads.
POSE_VARIANTS = [
    ("docked",    "Docked",    "_dock", "docked"),
    ("minimized", "Minimized", "_min",  "minimized"),
    ("generated", "Generated", "",      None),
]

# PoseBusters' 20 individual checks as the bundle's CSV names them, in its order. They
# ship as ONE column, pb_fail: a bitmask of the checks a molecule failed (bit i =
# PB_TESTS[i]), so the panel can list them without 20 columns per molecule.
PB_TESTS = [
    "sanitization", "inchi_convertible", "all_atoms_connected", "no_radicals",
    "bond_lengths", "bond_angles", "internal_steric_clash", "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness", "double_bond_flatness", "internal_energy",
    "protein-ligand_maximum_distance", "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors", "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters", "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors", "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
]


def pb_fail_mask(vals: dict):
    """Bitmask of failed checks; None when the molecule has no check results at all."""
    got = [vals.get(t) for t in PB_TESTS]
    if all(v is None for v in got):
        return None
    return sum(1 << i for i, v in enumerate(got) if v == 0)


# The columns each evaluation contributes, in display order. Anything not listed is
# dropped; PoseBusters' individual checks are read only to build pb_fail.
EVAL_COLUMNS = {
    "vina_docking": ["vina_score", "vina_min", "vina_dock"],
    "sample_quality": ["qed", "sa", "logp", "lipinski", "sim_to_ref"],
    "posecheck": ["clashes", "strain", "n_interactions",
                  "HBAcceptor", "HBDonor", "Hydrophobic", "VdWContact"],
    "posebusters": ["pb_valid", "n_pass", "n_checks"] + PB_TESTS,
}

# better: "low" | "high" | None (None = descriptive, never ranked or coloured)
METRIC_SCHEMA = [
    ("vina_dock",       "Vina Dock",    "kcal/mol", "low",  2),
    ("vina_min",        "Vina Min",     "kcal/mol", "low",  2),
    ("vina_score",      "Vina Score",   "kcal/mol", "low",  2),
    ("qed",             "QED",          "",         "high", 2),
    ("sa",              "SA",           "",         "high", 2),
    ("logp",            "logP",         "",         None,   2),
    ("lipinski",        "Lipinski",     "",         "high", 0),
    ("sim_to_ref",      "Sim to ref",   "",         None,   2),
    ("clashes",         "Clashes",      "",         "low",  0),
    ("strain",          "Strain",       "kcal/mol", "low",  0),
    ("n_interactions",  "Interactions", "",         "high", 0),
    ("HBAcceptor",      "HB acceptor",  "",         None,   0),
    ("HBDonor",         "HB donor",     "",         None,   0),
    ("Hydrophobic",     "Hydrophobic",  "",         None,   0),
    ("VdWContact",      "VdW contact",  "",         None,   0),
    ("pb_valid",        "PB valid",     "",         "high", 0),
    ("n_pass",          "PB passed",    "",         "high", 0),
    ("n_checks",        "PB checks",    "",         None,   0),
    ("pb_fail",         "PB failed checks", "",     None,   0),
    ("n_atoms",         "Heavy atoms",  "",         None,   0),
    ("rmsd_dock",       "Redock RMSD",  "Å",        "low",  2),
]
METRIC_KEYS = [k for k, *_ in METRIC_SCHEMA]

# Colour per method: one hue each, readable on both the 2D cards and the 3D overlay.
# Kept in the same family as the de novo Vina chart's colours.
METHOD_COLORS = {
    "AR": "#9b59b6", "Pocket2Mol": "#e67e22", "DiffSBDD": "#e74c3c",
    "TargetDiff": "#3498db", "DecompDiff": "#34495e", "FuncBind": "#16a085",
    "VoxBind": "#1abc9c", "VoxBind-Ours": "#2ecc71", "Ours-v2": "#2ecc71",
    "VoxBind-base-ep350-n100": "#52bb7e",
    "DecompDiff-prior25": "#e84393",
}
FALLBACK_COLORS = ["#7f8c8d", "#c0392b", "#8e44ad", "#2980b9", "#d35400"]


def log(msg: str) -> None:
    print(msg, flush=True)


def canon_smiles(smi: str | None) -> str | None:
    """Table SMILES and structure SMILES come from different writers; compare canonically."""
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi, sanitize=True)
    if m is None:
        return smi
    return Chem.MolToSmiles(m)


# ----------------------------------------------------------------- structure sources
def meta_molecules(folder: Path, meta_stem: str) -> dict[int, list]:
    """{pocket_index: [(mol, smiles, ligand_filename), ...]} in canonical meta order.

    Mirrors the task-2 results collector's baseline order -- same three suffixes, same
    enumerate(pocket) -- so position in this list IS the mol_idx the tables use.
    """
    seq: dict[int, list] = {}
    for suffix in META_SUFFIXES:
        f = folder / "samples" / "meta" / f"{meta_stem}{suffix}.pt"
        if not f.exists():
            continue
        try:
            meta = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as exc:  # a shard that will not load must be loud, not skipped
            log(f"    ! {f.name}: {exc}")
            continue
        for pocket, mols in enumerate(meta or []):
            for e in mols or []:
                if not isinstance(e, dict):
                    continue
                seq.setdefault(pocket, []).append(
                    (e.get("mol"), e.get("smiles"), e.get("ligand_filename")))
    return seq


def native_molecules(folder: Path) -> dict[int, list]:
    """Same shape, for a metrics.py tree: target_NN/{metrics.json, samples.sdf}.

    THE ORDER IS metrics.json's `samples` LIST, NOT THE SDF'S. samples.sdf holds every
    molecule the run generated; the tables' mol_idx counts only the ones that survived
    validity (TargetDiff: 9,878 records against 9,206 rows). Walking the SDF therefore
    puts a different molecule at index i than the tables mean -- measured at 31% SMILES
    agreement before this. So the metrics rows define the sequence and each one claims an
    unused SDF record with the same canonical SMILES; a row whose pose is missing keeps
    its slot with no structure rather than shifting everything after it.
    """
    seq: dict[int, list] = {}
    for tdir in sorted(folder.glob("samples/target_*")):
        try:
            pocket = int(tdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        sdf, mj = tdir / "samples.sdf", tdir / "metrics.json"
        if not sdf.exists():
            continue
        lig = next((p.stem.replace("__", "/") + ".sdf"
                    for p in sorted(tdir.glob("*.sdf")) if p.name != "samples.sdf"), None)
        recs = []
        for m in Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=True):
            smi = None
            if m is not None:
                try:
                    Chem.SanitizeMol(m)
                    smi = Chem.MolToSmiles(m)
                except Exception:
                    smi = None
            recs.append((m, smi))
        rows = []
        if mj.exists():
            try:
                rows = json.loads(mj.read_text()).get("samples", []) or []
            except (json.JSONDecodeError, OSError):
                rows = []
        if not rows:
            seq[pocket] = [(m, s, lig) for m, s in recs]
            continue
        pool: dict[str | None, list] = {}
        for m, s in recs:
            pool.setdefault(s, []).append(m)
        out = []
        for r in rows:
            smi = r.get("smiles")
            bucket = pool.get(canon_smiles(smi)) or pool.get(smi)
            out.append(((bucket.pop(0) if bucket else None), smi, lig))
        seq[pocket] = out
    return seq


def structures_for(folder: Path) -> tuple[dict[int, list], str]:
    stem = META_NAME.get(folder.name)
    if stem and (folder / "samples" / "meta" / f"{stem}.pt").exists():
        return meta_molecules(folder, stem), f"samples/meta/{stem}{{,_part2,_gap}}.pt"
    if any(folder.glob("samples/target_*/samples.sdf")):
        return native_molecules(folder), "samples/target_*/samples.sdf"
    return {}, ""


# --------------------------------------------------------------------- metric tables
def read_table(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    """'' -> None. Never emit nan: one nan string makes every median a reader takes wrong."""
    if v is None or v == "":
        return None
    if v in ("True", "true"):
        return 1
    if v in ("False", "false"):
        return 0
    try:
        f = float(v)
    except ValueError:
        return None
    if f != f:
        return None
    return int(f) if f.is_integer() and abs(f) < 1e15 else round(f, 4)


def index_table(rows: list[dict], cols: list[str]):
    """Two views of one table: by (target, mol_idx) and by (target, canonical smiles)."""
    by_idx, by_smi = {}, {}
    for r in rows:
        t = r.get("target")
        try:
            i = int(r.get("mol_idx", ""))
        except ValueError:
            continue
        vals = {c: num(r.get(c)) for c in cols}
        by_idx[(t, i)] = (r.get("smiles"), vals)
        by_smi.setdefault((t, canon_smiles(r.get("smiles"))), []).append(vals)
    return by_idx, by_smi


# ---------------------------------------------------------------------- pocket donor
def find_pockets() -> dict[str, dict]:
    """{target_NN: {pocket_pdb, ref_sdf, ligand_filename, in_density79, reference}}.

    The 100 CrossDocked pockets are shared by every method, so one donor tree with
    target_NN/*_pocket10.pdb supplies them all.
    """
    donors = sorted(
        d for d in TASK2.iterdir()
        if d.is_dir() and len(list(d.glob("samples/target_*/*_pocket10.pdb"))) >= 50
    )
    if not donors:
        return {}
    donor = donors[0]
    log(f"  pockets from {donor.name}/samples/target_*/")
    out = {}
    for tdir in sorted(donor.glob("samples/target_*")):
        pdb = next(iter(sorted(tdir.glob("*_pocket10.pdb"))), None)
        ref = next((p for p in sorted(tdir.glob("*.sdf")) if p.name != "samples.sdf"), None)
        if pdb is None:
            continue
        entry = {"pocket_pdb": pdb, "ref_sdf": ref,
                 "ligand_filename": (ref.stem.replace("__", "/") + ".sdf") if ref else None,
                 "reference": None}
        mj = tdir / "metrics.json"
        if mj.exists():
            try:
                entry["reference"] = json.loads(mj.read_text()).get("reference")
            except (json.JSONDecodeError, OSError):
                pass
        out[tdir.name] = entry
    return out


# PoseBusters' dock mode runs 22 tests. The bundle's per-molecule tables count 20:
# they leave out mol_pred_loaded and mol_cond_loaded, the two "did the file load"
# checks. Every molecule that has a PoseBusters result loaded by construction, so on
# the 22-test scale the counts are exactly +2 -- verified molecule by molecule on
# TargetDiff, the one method whose full check dict is on this box (9,206/9,206: both
# load checks True, sum of 22 == n_pass + 2, and `valid` == all 22 pass). A row with
# no n_checks never got a result, so it stays empty rather than gaining a false 2.
PB_OMITTED = ("mol_pred_loaded", "mol_cond_loaded")


def pb22(n_pass, n_checks):
    if n_checks == 20 and n_pass is not None:
        return n_pass + 2, 22
    return n_pass, n_checks


def flatten_reference(ref: dict | None) -> dict | None:
    """The donor's reference block, reshaped onto the same metric keys as a molecule."""
    if not isinstance(ref, dict):
        return None
    pc = ref.get("posecheck") or {}
    inter = pc.get("interactions") or {}
    pb = ref.get("posebusters") or {}
    vina = ref.get("vina") if isinstance(ref.get("vina"), dict) else {}
    out = {
        "smiles": ref.get("smiles"), "n_atoms": ref.get("n_atoms"),
        "qed": ref.get("qed"), "sa": ref.get("sa"), "logp": ref.get("logp"),
        "lipinski": ref.get("lipinski"), "sim_to_ref": None,
        "clashes": pc.get("clashes"), "strain": pc.get("strain"),
        "n_interactions": pc.get("n_interactions"),
        "HBAcceptor": inter.get("HBAcceptor"), "HBDonor": inter.get("HBDonor"),
        "Hydrophobic": inter.get("Hydrophobic"), "VdWContact": inter.get("VdWContact"),
        "pb_valid": 1 if pb.get("valid") else (0 if "valid" in pb else None),
        "n_pass": pb.get("n_pass"), "n_checks": pb.get("n_checks"),
        "vina_score": vina.get("score_only"), "vina_min": vina.get("minimize"),
        "vina_dock": vina.get("dock"),
    }
    checks = pb.get("checks") if isinstance(pb.get("checks"), dict) else None
    if checks and len(checks) == 22:
        out["n_pass"] = sum(1 for v in checks.values() if v is True)
        out["n_checks"] = 22
    else:
        out["n_pass"], out["n_checks"] = pb22(out["n_pass"], out["n_checks"])
    out["pb_fail"] = None
    if checks:
        norm = {str(k).strip().lower().replace(" ", "_"): v for k, v in checks.items()}
        out["pb_fail"] = pb_fail_mask(
            {t: (None if norm.get(t) is None else (1 if norm[t] is True else 0))
             for t in PB_TESTS})
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


# ---------------------------------------------------------------------------- output
def write_json(path: Path, payload, force: bool = False) -> bool:
    """Write only when the payload changed, so an unchanged shard keeps its mtime."""
    text = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    if not force and path.exists():
        try:
            if path.read_text() == text:
                return False
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def write_sdf(path: Path, mols: list, force: bool = False) -> int:
    """One record per molecule, in canonical order. A missing mol still takes its slot
    (an empty record) so record i never stops meaning mol_idx i."""
    if path.exists() and not force:
        n = sum(1 for line in path.open(errors="ignore") if line.strip() == "$$$$")
        if n == len(mols):
            return n
    path.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(path))
    written = 0
    for i, m in enumerate(mols):
        if m is None:
            w.write(Chem.MolFromSmiles(""))  # placeholder keeps the index aligned
            continue
        try:
            m = Chem.Mol(m)
            m.SetProp("_Name", f"{path.stem}_{i}")
            w.write(m)
            written += 1
        except Exception:
            w.write(Chem.MolFromSmiles(""))
    w.close()
    return written


def diversity_stats(smiles: list[str | None]) -> dict:
    """Diversity of one method's molecules for one pocket, TargetDiff's definition.

    TargetDiff (and this project's task-2 evaluator) defines it
    on the SET: 1 - the mean pairwise Tanimoto over Chem.RDKFingerprint. To give it a
    mean AND a median like every other quality column, decompose it per molecule:
    d_i = 1 - mean Tanimoto of molecule i to the others. The mean of d_i is EXACTLY
    the set diversity (each pair is counted once from each end), so `mean` is
    TargetDiff's number; `median` is the same quantity, robust to a few outliers.
    RDKFingerprint is topological, so building it from SMILES matches the 3D record.
    """
    from rdkit import DataStructs
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            fps.append(Chem.RDKFingerprint(m))
    n = len(fps)
    if n < 2:
        return {"mean": None, "median": None, "n": n}
    row_sum = [0.0] * n
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for k, sv in enumerate(sims):
            row_sum[i] += sv
            row_sum[i + 1 + k] += sv
    per_mol = [1.0 - rs / (n - 1) for rs in row_sum]
    return {"mean": round(sum(per_mol) / n, 4),
            "median": round(st.median(per_mol), 4), "n": n}


# TargetDiff's high-affinity rule (notebooks/summary.ipynb::compute_high_affinity):
# the share of a pocket's DOCKED molecules whose Vina dock is <= the reference's,
# and a pocket with fewer than 50 docked molecules is left out of their average.
HIGH_AFF_MIN_DOCKED = 50


def high_affinity(docks: list[float], ref_dock: float | None) -> dict:
    if ref_dock is None or not docks:
        return {"share": None, "n_docked": len(docks), "eligible": False}
    good = sum(1 for d in docks if d <= ref_dock)
    return {"share": round(good / len(docks), 4), "n_docked": len(docks),
            "eligible": len(docks) >= HIGH_AFF_MIN_DOCKED}


AD2EL = {"A": "C", "OA": "O", "NA": "N", "SA": "S", "HD": "H", "HS": "H", "NS": "N"}


def pdbqt_heavy(block: str) -> list[tuple]:
    """Heavy atoms of the FIRST model of a Vina PDBQT (its best pose)."""
    out = []
    for line in block.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM", "HETATM")):
            ad = line[77:79].strip() if len(line) > 77 else ""
            el = AD2EL.get(ad, ad if ad else line[12:16].strip()[:1])
            el = el[:1].upper() + el[1:].lower()
            if el != "H":
                out.append((el, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return out


def _skeleton(mol):
    """Same atoms, every bond single, nothing aromatic or charged -- a bare graph."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False); a.SetFormalCharge(0); a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    for bd in rw.GetBonds():
        bd.SetBondType(Chem.BondType.SINGLE); bd.SetIsAromatic(False)
    return rw.GetMol()


def map_pose(template, block):
    """The generated molecule carrying a PDBQT pose's coordinates, or None.

    Vina's PDBQT has the right heavy atoms but REORDERED (docking prep rebuilds the
    atom list along its torsion tree -- 4 of 320 poses kept the order) and no bonds.
    So connectivity is inferred from the docked geometry and the generated molecule's
    heavy-atom graph is found inside it (elements matched, bond orders ignored). The
    match puts each docked coordinate on the right atom of the generated molecule,
    so the result keeps that molecule's atom order and bond orders: record i of every
    pose file is the same molecule, and RMSD needs no further mapping. 98% of poses
    map; the rest stay empty rather than being matched by guesswork.
    """
    heavy = pdbqt_heavy(block or "")
    if len(heavy) != template.GetNumAtoms():
        return None
    xyz = f"{len(heavy)}\n\n" + "\n".join(f"{e} {x:.4f} {y:.4f} {z:.4f}" for e, x, y, z in heavy)
    docked = Chem.MolFromXYZBlock(xyz)
    if docked is None:
        return None
    try:
        rdDetermineBonds.DetermineConnectivity(docked)
    except Exception:
        return None
    match = _skeleton(docked).GetSubstructMatch(_skeleton(template))
    if len(match) != template.GetNumAtoms():
        return None
    out = Chem.Mol(template)
    out.RemoveAllConformers()
    conf = Chem.Conformer(out.GetNumAtoms())
    dp = docked.GetConformer()
    for i, j in enumerate(match):
        q = dp.GetAtomPosition(j)
        conf.SetAtomPosition(i, Point3D(q.x, q.y, q.z))
    out.AddConformer(conf, assignId=True)
    return out


def _heavy(mol):
    try:
        return Chem.RemoveHs(mol)
    except Exception:
        return Chem.RemoveHs(mol, sanitize=False)


def load_eval_pt_method(name: str, folder: Path) -> dict | None:
    """A TargetDiff-style eval dump (eval_NNN_<ligand>.pt, one list per pocket) as a
    method: molecules, their metrics, and the minimised / docked poses."""
    files = sorted(glob.glob(str(folder / "eval_*.pt")))
    if not files:
        return None
    struct, direct, posemols = {}, {}, {}
    n = mapped = 0
    for f in files:
        mt = re.search(r"eval_(\d+)_", Path(f).name)
        if not mt:
            continue
        pocket = int(mt.group(1))
        try:
            recs = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as exc:
            log(f"    ! {Path(f).name}: {exc}")
            continue
        mols, vals, dock, mini = [], [], [], []
        for r in recs:
            gen = _heavy(r["mol"])
            v = r.get("vina") or {}
            aff = lambda mode: ((v.get(mode) or [{}])[0] or {}).get("affinity")
            pose = lambda mode: ((v.get(mode) or [{}])[0] or {}).get("pose")
            d_mol = map_pose(gen, pose("dock"))
            m_mol = map_pose(gen, pose("minimize"))
            n += 2
            mapped += (d_mol is not None) + (m_mol is not None)
            rmsd = None
            if d_mol is not None:
                try:
                    rmsd = round(rdMolAlign.CalcRMS(d_mol, gen), 4)   # symmetry-aware, in place
                except Exception:
                    rmsd = None
            chem = r.get("chem_results") or {}
            num = lambda x: None if x is None else (round(float(x), 4)
                                                     if not isinstance(x, (int,)) else x)
            lip = chem.get("lipinski")
            vals.append({
                "vina_score": num(aff("score_only")), "vina_min": num(aff("minimize")),
                "vina_dock": num(aff("dock")),
                "qed": num(chem.get("qed")), "sa": num(chem.get("sa")),
                "logp": num(chem.get("logp")),
                "lipinski": int(lip) if lip is not None else None,
                "n_atoms": gen.GetNumAtoms(), "rmsd_dock": rmsd,
            })
            mols.append((gen, r.get("smiles"), r.get("ligand_filename")))
            dock.append(d_mol); mini.append(m_mol)
        struct[pocket], direct[pocket] = mols, vals
        posemols[pocket] = {"docked": dock, "minimized": mini}
    log(f"  + {name}: {sum(len(x) for x in struct.values())} molecules over "
        f"{len(struct)} pockets from {folder.relative_to(ROOT)}; "
        f"{mapped}/{n} poses mapped onto their molecule")
    return {"folder": types.SimpleNamespace(name=name), "struct": struct,
            "src": str(folder.relative_to(ROOT)) + "/eval_*.pt", "tables": {},
            "direct": direct, "posemols": posemols, "note": EVAL_PT_NOTE}


def sdf_spans(path: Path) -> list[int]:
    """Byte length of each record in an SDF, in order.

    The viewer fetches ONE molecule out of a file that holds a hundred. With these
    lengths it can turn an index into a byte range and ask for just that slice:
    a pocket costs ~13 KB of structures instead of ~1.3 MB. A prefix sum gives the
    offsets, so only the lengths need shipping.
    """
    data = path.read_bytes()
    needle = b"$$$$\n"
    spans, start = [], 0
    while True:
        k = data.find(needle, start)
        if k < 0:
            break
        end = k + len(needle)
        spans.append(end - start)
        start = end
    return spans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--methods", help="comma-separated folder names (default: discover)")
    ap.add_argument("--include-ours", action="store_true",
                    help="also build the VoxBind/Ours arms (generated on another box)")
    ap.add_argument("--limit-targets", type=int, default=0, help="first N pockets only")
    ap.add_argument("--force", action="store_true", help="rewrite unchanged files")
    ap.add_argument("--check", action="store_true", help="verify and report, write nothing")
    args = ap.parse_args()

    if not TASK2.is_dir():
        log(f"MISSING {TASK2}")
        return 1

    log(f"[browser] bundle {TASK2}")
    pockets = find_pockets()
    if not pockets:
        log("  ! no pocket donor found (need target_NN/*_pocket10.pdb)")
        return 1
    targets = sorted(pockets)
    if args.limit_targets:
        targets = targets[:args.limit_targets]
    log(f"  {len(pockets)} pockets on disk, building {len(targets)}")

    wanted = set(args.methods.split(",")) if args.methods else None
    methods, report = [], {}

    for folder in sorted(TASK2.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        if wanted and folder.name not in wanted:
            continue
        if folder.name in OURS and not (args.include_ours or wanted):
            continue
        tables = {e: folder / "eval" / e / "per_molecule.csv" for e in EVALUATIONS}
        have = {e: p for e, p in tables.items() if p.exists()}
        if not have:
            continue
        struct, struct_src = structures_for(folder)
        if not struct:
            log(f"  - {folder.name}: has tables but no molecules on this box -- skipped")
            report[folder.name] = {"skipped": "no molecules on this box"}
            continue
        log(f"  + {folder.name}: {sum(len(v) for v in struct.values())} molecules "
            f"from {struct_src}, tables {sorted(have)}")
        methods.append({"folder": folder, "struct": struct, "src": struct_src,
                        "tables": have})

    for name, folder in EVAL_PT_METHODS.items():
        if wanted and name not in wanted:
            continue
        if not folder.is_dir():
            log(f"  - {name}: {folder} not on this box -- skipped")
            continue
        em = load_eval_pt_method(name, folder)
        if not em:
            continue
        # the dump is indexed by pocket ordinal; confirm it is the same pocket by the
        # CrossDocked ligand file before trusting that, and drop any that disagree
        ref_lig = {}
        for other in methods:
            for pk, mols_ in other["struct"].items():
                if mols_ and mols_[0][2] and "/" in mols_[0][2]:
                    ref_lig.setdefault(pk, mols_[0][2])
        bad = [pk for pk, mols_ in em["struct"].items()
               if mols_ and pk in ref_lig and mols_[0][2] != ref_lig[pk]]
        for pk in bad:
            for key in ("struct", "direct", "posemols"):
                em[key].pop(pk, None)
        if bad:
            log(f"      ! {name}: {len(bad)} pockets name a different ligand file -- dropped")
        methods.append(em)

    if not methods:
        log("  ! no method had both tables and molecules")
        return 1

    # ---- load every table once, and measure its join against the structures
    for m in methods:
        name = m["folder"].name
        m["idx"], m["smi"], m["join"] = {}, {}, {}
        m["attached"], m["offered"], m["poses"] = {}, {}, {}
        for ev, path in m["tables"].items():
            cols = EVAL_COLUMNS[ev]
            by_idx, by_smi = index_table(read_table(path), cols)
            m["idx"][ev], m["smi"][ev] = by_idx, by_smi
            shared = agree = 0
            for pocket, mols in m["struct"].items():
                t = f"target_{pocket:02d}"
                for i, (_mol, smi, _lig) in enumerate(mols):
                    rec = by_idx.get((t, i))
                    if rec is None:
                        continue
                    shared += 1
                    if canon_smiles(rec[0]) == canon_smiles(smi):
                        agree += 1
            pct = (agree / shared) if shared else 0.0
            mode = "mol_idx" if pct >= JOIN_OK else "smiles"
            m["join"][ev] = {"mode": mode, "agreement": round(pct, 4),
                             "compared": shared, "rows": len(by_idx)}
            flag = "" if mode == "mol_idx" else "   <- falls back to (target, smiles)"
            log(f"      {name:<12} {ev:<14} join={mode:<7} "
                f"SMILES agreement {pct:6.1%} on {shared}{flag}")

    # ---- per-target shards
    ref_vina = {}
    if REFERENCE_VINA.exists():
        try:
            ref_vina = json.loads(REFERENCE_VINA.read_text())
            log(f"  reference Vina from {REFERENCE_VINA.relative_to(ROOT)} "
                f"({len(ref_vina)} pockets)")
        except (json.JSONDecodeError, OSError) as exc:
            log(f"  ! {REFERENCE_VINA}: {exc}")

    out = args.out
    cols = ["i", "smiles"] + METRIC_KEYS
    written, target_rows = 0, []
    for t in targets:
        pocket = int(t.split("_")[1])
        # The meta bundles carry the full "SUBDIR/ligand.sdf" CrossDocked identity; the
        # donor's ref.sdf stem does not. Prefer the full one -- it is what keys the
        # reference table, and what lets a row be traced back to the test set.
        lig_full = next(
            (m["struct"][pocket][0][2] for m in methods
             if m["struct"].get(pocket) and (m["struct"][pocket][0][2] or "").count("/")),
            None)
        reference = flatten_reference(pockets[t]["reference"])
        if reference is not None and lig_full in ref_vina:
            reference["vina_dock"] = ref_vina[lig_full]
        payload = {
            "target": t, "pocket_index": pocket,
            "ligand_filename": lig_full or pockets[t]["ligand_filename"],
            "cols": cols, "methods": {}, "spans": {}, "setstats": {},
            "reference": reference,
        }
        summary = {}
        for m in methods:
            name = m["folder"].name
            mols = m["struct"].get(pocket, [])
            if not mols:
                continue
            used: dict[tuple, int] = {}
            rows, sdf_mols = [], []
            for i, (mol, smi, _lig) in enumerate(mols):
                vals = {k: None for k in METRIC_KEYS}
                for ev in m["tables"]:
                    if m["join"][ev]["mode"] == "mol_idx":
                        rec = m["idx"][ev].get((t, i))
                        got = rec[1] if rec else None
                    else:
                        key = (t, canon_smiles(smi))
                        bucket = m["smi"][ev].get(key) or []
                        n = used.get((ev,) + key, 0)
                        got = bucket[n] if n < len(bucket) else None
                        used[(ev,) + key] = n + 1
                    m["attached"][ev] = m["attached"].get(ev, 0) + (1 if got else 0)
                    m["offered"][ev] = m["offered"].get(ev, 0) + 1
                    if got:
                        vals.update({k: v for k, v in got.items() if v is not None})
                if m.get("direct"):
                    dv = m["direct"].get(pocket, [])
                    if i < len(dv):
                        vals.update({k: x for k, x in dv[i].items() if x is not None})
                if vals.get("n_atoms") is None and mol is not None:
                    try:
                        vals["n_atoms"] = mol.GetNumHeavyAtoms()
                    except Exception:
                        pass
                vals["n_pass"], vals["n_checks"] = pb22(vals["n_pass"], vals["n_checks"])
                vals["pb_fail"] = pb_fail_mask(vals)
                rows.append([i, smi] + [vals[k] for k in METRIC_KEYS])
                sdf_mols.append(mol)
            payload["methods"][name] = rows
            di = cols.index("vina_dock")
            docks = [r[di] for r in rows if r[di] is not None]
            payload["setstats"][name] = {
                "diversity": diversity_stats([r[1] for r in rows]),
                "high_affinity": high_affinity(
                    docks, (reference or {}).get("vina_dock")),
                "n": len(rows),
            }
            docks = [r[cols.index("vina_dock")] for r in rows
                     if r[cols.index("vina_dock")] is not None]
            summary[name] = {"n": len(rows),
                             "vina_dock_median": round(st.median(docks), 3) if docks else None}
            if not args.check:
                sdf_path = out / "data" / "struct" / t / f"{name}.sdf"
                write_sdf(sdf_path, sdf_mols, args.force)
                spans = sdf_spans(sdf_path)
                if len(spans) != len(rows):
                    log(f"      ! {name}/{t}: {len(spans)} records vs {len(rows)} rows "
                        f"-- byte ranges disabled for this one")
                else:
                    payload["spans"][name] = spans
            # Alternative poses, if a re-docking pass has dropped them in. Copied only
            # when the record count matches the generated SDF, so index i keeps meaning
            # the same molecule in every pose.
            for key, _lab, suffix, sub in POSE_VARIANTS:
                if sub is None:
                    m["poses"].setdefault(key, 0)
                    m["poses"][key] += 1
                    continue
                dst = out / "data" / "struct" / t / f"{name}{suffix}.sdf"
                pm = ((m.get("posemols") or {}).get(pocket) or {}).get(key)
                if pm is not None:
                    # mapped onto the generated molecules, so already in their order;
                    # a pose that could not be mapped keeps its slot as an empty record
                    if len(pm) != len(rows):
                        log(f"      ! {name}/{key}/{t}: {len(pm)} poses vs {len(rows)} "
                            f"molecules -- skipped")
                        continue
                    m["poses"][key] = m["poses"].get(key, 0) + 1
                    if not args.check:
                        write_sdf(dst, pm, args.force)
                else:
                    if not isinstance(m["folder"], Path):
                        continue
                    src = m["folder"] / "poses" / sub / f"{t}.sdf"
                    if not src.exists():
                        continue
                    n_rec = src.read_text(errors="ignore").count("$$$$")
                    if n_rec != len(rows):
                        log(f"      ! {name}/{key}/{t}: {n_rec} records vs {len(rows)} "
                            f"molecules -- skipped, the indices would not line up")
                        continue
                    m["poses"][key] = m["poses"].get(key, 0) + 1
                    if not args.check:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(src.read_bytes())
                if not args.check and dst.exists():
                    sp = sdf_spans(dst)
                    if len(sp) == len(rows):
                        payload["spans"][name + suffix] = sp

        if not args.check:
            src_pdb = pockets[t]["pocket_pdb"]
            dst = out / "data" / "struct" / t
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "pocket.pdb").write_bytes(src_pdb.read_bytes())
            if pockets[t]["ref_sdf"]:
                (dst / "ref.sdf").write_bytes(pockets[t]["ref_sdf"].read_bytes())
            if write_json(out / "data" / "targets" / f"{t}.json", payload, args.force):
                written += 1
        target_rows.append({"target": t, "pocket_index": pocket,
                            "ligand_filename": lig_full or pockets[t]["ligand_filename"],
                            "methods": summary})
        if pocket % 20 == 0:
            log(f"    .. {t}")

    # How many molecules actually received a value, now that the shards are built. A join
    # that "works" can still cover half the column -- TargetDiff's docking pass scored a
    # different molecule set from its pose pass, so most of its rows have no Vina at all,
    # and a reader sorting that column has to be able to see that.
    for m in methods:
        for ev, j in m["join"].items():
            off, att = m["offered"].get(ev, 0), m["attached"].get(ev, 0)
            j["attached"], j["offered"] = att, off
            j["coverage"] = round(att / off, 4) if off else 0.0
            if j["mode"] == "smiles":
                j["caveat"] = ("matched on (target, SMILES): the value belongs to that "
                               "molecule but may have been scored on a different pose "
                               "in a separate pass")
            if off and att < off:
                log(f"      {m['folder'].name:<12} {ev:<14} covers {att}/{off} "
                    f"of the molecules built ({att/off:.0%})")

    palette = iter(FALLBACK_COLORS)
    manifest = {
        "generated_on": THIS_BOX,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "results/task2-drugdesign",
        "note": ("Published baselines only -- the VoxBind arms were generated on another "
                 "box and are not in this build."),
        "protocol": {
            "receptor": "10 A pocket crop (the full receptor is not on this box)",
            "pocket_set": "CrossDocked 100 test pockets; in_density79 marks the 79 subset",
            "diversity": ("TargetDiff: 1 - mean pairwise Tanimoto (RDKFingerprint) "
                          "within a pocket; mean = that number exactly, median = the "
                          "median of per-molecule 1 - mean-similarity-to-the-rest"),
            "high_affinity": ("TargetDiff: share of docked molecules with Vina dock "
                              "<= the reference's; pockets with < "
                              f"{HIGH_AFF_MIN_DOCKED} docked are excluded from their "
                              "average and flagged here"),
            "posebusters": ("22-test scale. The bundle's tables count 20 (they omit "
                            + " and ".join(PB_OMITTED) + "); every scored molecule passes "
                            "both, so counts here are the table's + 2"),
            "reference_vina": (str(REFERENCE_VINA.relative_to(ROOT)) + " -- a separate "
                               "reference-docking pass, not the methods' own run")
                              if ref_vina else "not available on this box",
        },
        "metrics": [{"key": k, "label": lab, "unit": u, "better": b, "digits": d}
                    for k, lab, u, b, d in METRIC_SCHEMA],
        # bit i of a molecule's pb_fail is this list's i-th check
        "pb_tests": PB_TESTS,
        # Which poses this build actually contains. The viewer offers all three and
        # disables the ones with no structures, rather than quietly showing the
        # generated pose under a label that promises a docked one.
        "poses": [{
            "key": key, "label": lab, "suffix": suffix,
            "targets": sum(1 for m in methods if m["poses"].get(key)),
            "methods": sorted(m["folder"].name for m in methods if m["poses"].get(key)),
            "available": any(m["poses"].get(key) for m in methods),
            "why": None if sub is None else
                   ("no coordinates in this bundle -- the per-molecule tables carry "
                    "Vina's scores but not its poses. Producing them means re-docking "
                    "against the full receptor, which is not on this box. Drop them at "
                    f"<Method>/poses/{sub}/target_NN.sdf and re-run this builder."),
        } for key, lab, suffix, sub in POSE_VARIANTS],
        "cols": cols,
        "methods": [{
            "name": m["folder"].name,
            "color": METHOD_COLORS.get(m["folder"].name) or next(palette, "#888"),
            "structures": m["src"],
            "evaluations": sorted(m["tables"]),
            "note": m.get("note"),
            "join": m["join"],
            "n_molecules": sum(len(v) for v in m["struct"].values()),
        } for m in methods],
        "targets": target_rows,
    }
    if not args.check:
        write_json(out / "data" / "manifest.json", manifest, force=True)
        tpl = TEMPLATE
        if tpl.exists():
            (out / "index.html").write_text(tpl.read_text())
        else:
            log(f"  ! viewer template missing: {tpl}")
        log(f"[browser] wrote {out}")
        log(f"          {len(target_rows)} targets, {len(methods)} methods, "
            f"{written} shards changed")
    else:
        log("[browser] --check: nothing written")
    unsafe = [(m["folder"].name, ev) for m in methods
              for ev, j in m["join"].items() if j["mode"] != "mol_idx"]
    if unsafe:
        log("  joins that fell back to (target, smiles): "
            + ", ".join(f"{n}/{e}" for n, e in unsafe))
    return 0


if __name__ == "__main__":
    sys.exit(main())

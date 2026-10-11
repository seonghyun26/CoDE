"""registry.py — per-dataset config + script dispatch for the unified build pipeline.

The single place the 5 stage files (s1..s5) import. Holds:
  - the VoxBind atom dictionary + vdW radii (from voxbind/constants.py),
  - a DatasetSpec per dataset (frame/align, density match gate, split source,
    stage-4 voxelize mode, target kind, atom-blob radii),
  - run_stage(): dispatch a stage for a dataset to its ordered build steps.

Each branch DELEGATES to the validated per-dataset script by subprocess, which is the same
code path those scripts always ran, so parity is automatic rather than argued. The scripts
live next to their dataset (`dataset/crossdocked/`, `dataset/pdbbind/`, `dataset/misato/`,
`dataset/plinder/`). Retired builders are in the repo-root `legacy/` folder, which nothing here
calls.

Inlining a branch's logic here instead of shelling out is still possible, but it costs a
numerical-parity run per dataset and buys only process overhead. Delegation stays.
See dataset/build/README.md.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATASET_DIR = Path(__file__).resolve().parent.parent        # .../dataset
REPO_ROOT = DATASET_DIR.parent                              # repo root (VoxBind/)
VOX_ROOT = REPO_ROOT / "voxbind"                            # the voxbind package (constants, splits)

# ── VoxBind atom dictionary (single source of truth) ───────────────────────────
if str(VOX_ROOT) not in sys.path:
    sys.path.insert(0, str(VOX_ROOT))
from constants import ELEMENTS_HASH_CROSSDOCKED, RADIUS_PER_ATOM  # noqa: E402

LIGAND_ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "P"]      # atom-voxel channels 0-6
POCKET_ELEMENTS = ["C", "O", "N", "S"]                      # atom-voxel channels 7-10
VDW_RADII = RADIUS_PER_ATOM["MOL"]                          # element -> vdW radius (Å)

GRID_DIM, RESOLUTION = 64, 0.25

# Atom-blob radius modes (matches crossdocked_xray.py / 01b convention):
#   a positive scalar = ONE uniform blob radius (Å) for every atom of that group;
#   VDW (-1)          = per-element van der Waals radius via VDW_RADII LUT.
VDW = -1.0
UNIFORM = 0.5

DATASETS = ("crossdocked", "pdbbind", "misato", "plinder")

# MISATO's 02b/02c read HDF5 and need an h5py interpreter (NOT the voxbind env): the cellmaes
# env, found through conda by name. Override with MISATO_H5PY_PY=/path/to/python.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from voxbind.utils.conda_env import conda_env_python  # noqa: E402

H5PY_PY = os.environ.get("MISATO_H5PY_PY") or conda_env_python("cellmaes")


# Delegated scripts live in their dataset's folder under dataset/.
_SCRIPT_DIRS = ("crossdocked", "pdbbind", "misato", "plinder")


def _script_path(script: str) -> Path:
    """Resolve a delegated build script to its path under dataset/.

    Searched in a fixed order so resolution is deterministic. A missing script raises here,
    where the message can say what was looked for, rather than surfacing later as a confusing
    subprocess exit.
    """
    for sub in _SCRIPT_DIRS:
        candidate = DATASET_DIR / sub / script
        if candidate.is_file():
            return candidate
    searched = ", ".join(f"dataset/{s}" for s in _SCRIPT_DIRS)
    raise FileNotFoundError(f"build script {script!r} not found (searched: {searched})")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame: str                  # "kabsch" (CrossDocked) | "deposited" (no align)
    match_gate: str             # density↔ligand match criterion (human-readable)
    split_source: str
    voxelize_mode: str          # ATOMS: "on_the_fly" (splat on-GPU at train) | "precompute"
    target: Optional[str]       # "pK" | "qm_md" | None
    ligand_radius: float        # >0 uniform Å, VDW(-1) = per-element vdW LUT
    pocket_radius: float        # >0 uniform Å, VDW(-1) = per-element vdW LUT
    # DENSITY (orthogonal to voxelize_mode, which is about atoms):
    #   "precomputed_crop"  s3 writes frozen 64³ crops; train-time aug shifts them (zero-fill).
    #   "resample_map"      s3 writes only a manifest + recipe and KEEPS the maps; density is
    #                       cropped from the full map at the augmented pose at train time.
    # Only datasets whose s3 delegates to a resample-capable script (00b) can use resample_map.
    density_mode: str = "precomputed_crop"
    # stage -> ordered build steps. Each step is a tuple:
    #   ("run",  script, subcmd|None)   subprocess: python dataset/<script> [subcmd] <passthrough>
    #   ("note", message)               informational, no-op
    #   ("todo", message)               not wired yet -> NotImplementedError (printed under --dry_run)
    stages: dict = field(default_factory=dict)


SPEC = {
    "crossdocked": DatasetSpec(
        name="crossdocked", frame="kabsch",
        match_gate="native_filter tt_min (ligand PDB == receptor PDB) + density@atoms threshold",
        split_source="val_from_train_tail", voxelize_mode="on_the_fly", target=None,
        ligand_radius=UNIFORM, pocket_radius=VDW,
        density_mode="resample_map",   # 00b supports --mode resample (manifest + keep maps)
        stages={
            "s1": [("run", "preprocess_crossdocked.py", None)],
            "s2": [("run", "00a_density_download.py", None)],
            "s3": [("run", "00b_density_preprocess.py", None)],  # align (Kabsch) + crop + arcsinh normalise
                                                                 # resample: s3_crop --resample → 00b --mode resample
            "s4": [("note", "CrossDocked atoms voxelize on-GPU at train time — no precompute.")],
            "s5": [("note", "CrossDocked is self-supervised — no targets.")],
        },
    ),
    "pdbbind": DatasetSpec(
        name="pdbbind", frame="deposited",
        match_gate="inherent holo crystal (PDBe-EDS coverage)",
        # Built outside these stages: voxbind/splits/make_splits.py, from pdbbind/ed_rscc_split.csv.
        split_source="lp_edrscc_v2 (3850/817/1320); CL3 tier lp_edrscc_v2_cl123 (1559/410/733)",
        voxelize_mode="precompute", target="pK",
        ligand_radius=VDW, pocket_radius=VDW,   # canonical probe set = ligvdw (pass --ligand_vdw to s4)
        stages={
            "s1": [("run", "01a_pdbbind_acquire.py", "structures"),
                   ("run", "01a_pdbbind_acquire.py", "index")],
            "s2": [("run", "01a_pdbbind_acquire.py", "density")],
            "s3": [("note", "PDBbind is deposited-frame (no Kabsch); crop is FUSED into the precompute — see s4.")],
            # Reproduce the canonical PROBE voxel sets (element_filter=all throughout; probe is aug-OFF
            # so precompute is correct here). 3 steps; pass only flags common to all (e.g. --max_complexes).
            "s4": [
                # 1. uniform-ligand atoms (0.5 Å) + per-crop density  → voxels/
                ("run", "01b_pdbbind_preprocess.py", "voxelize", {"args": ["--element_filter", "all"]}),
                # 2. vdW-ligand atoms (the canonical probe atoms, read via --atom_source ligvdw)  → voxels_ligvdw/
                ("run", "01b_pdbbind_preprocess.py", "voxelize",
                 {"args": ["--ligand_vdw", "--element_filter", "all", "--no_density",
                           "--out_dir", "dataset/data/pdbbind/voxels_ligvdw"]}),
                # 3. the probe density: arcsinh + z-score on the CrossDocked reference statistics  → voxels_v5/
                ("run", "01b_pdbbind_preprocess.py", "poolnorm",
                 {"args": ["--element_filter", "all", "--stats_source", "reference"]}),
            ],
            "s5": [("note", "pK target lives in pdbbind/index.csv (read directly by the probe). "
                            "Run dataset/pdbbind/extract_bfactors.py for the B-factor target.")],
        },
    ),
    "misato": DatasetSpec(
        name="misato", frame="deposited",
        match_gate="QM/MD coverage + built structures",
        split_source="misato 8:1:1", voxelize_mode="precompute", target="qm_md",
        ligand_radius=VDW, pocket_radius=VDW,
        stages={
            # MISATO spans two envs: 02a/02d run under voxbind; 02b reads HDF5 → needs H5PY_PY.
            "s1": [("run", "02a_misato_acquire.py", "hdf5", {"args": ["--which", "both"]})],  # QM.hdf5 + MD.hdf5
            "s2": [("run", "02a_misato_acquire.py", "pdbe")],   # pass --ids <list> --what both (PDB + EDS maps)
            "s3": [("note", "deposited-frame (no Kabsch); crop fused into s4. PDBbind-missing complexes need "
                            "structures built first: 02c_misato_structures.py build --ids <list>  (run under H5PY_PY).")],
            "s4": [("run", "02d_misato_voxelize.py", None)],    # chunked 01b voxelize+poolnorm + feature extract (voxbind)
            "s5": [("run", "02b_misato_targets.py", "qm", {"py": H5PY_PY}),
                   ("run", "02b_misato_targets.py", "md", {"py": H5PY_PY})],
        },
    ),
    "plinder": DatasetSpec(
        name="plinder", frame="deposited",
        match_gate="RSCC >= 0.8 + 0 unresolved heavy atoms",
        split_source="val_from_train_tail", voxelize_mode="on_the_fly", target=None,
        ligand_radius=VDW, pocket_radius=VDW,
        density_mode="resample_map",   # 04_build_corpus writes an OTF resample manifest (deposited frame, R=I)
        stages={
            # The path that built the corpora training reads. Steps refuse to overwrite their
            # outputs (--force rebuilds); s1 compares with the frozen v2 pin instead of replacing it.
            "s1": [("run", "01_select.py", None, {"args": ["--version", "v2"]})],              # funnel + RSCC gate + dedup + leakage
            "s2": [("run", "03_download.py", None, {"args": ["--what", "both"]})],             # cif + ccp4 into the shared caches
            "s3": [("run", "04_build_corpus.py", None, {"args": ["--version", "v2"]}),          # per-element tuples + OTF manifest
                   ("run", "05_make_boxes.py", None),                                         # box116.dat: the 116³ density box loaders read
                   ("run", "02_make_clean_corpus.py", None)],                                  # CoDE_pretrain_v1: decontaminated mask + dset config
            "s4": [("note", "PLINDER atoms voxelize on-GPU at train time — no precompute.")],
            "s5": [("note", "PLINDER is self-supervised — no targets.")],
        },
    ),
}


def load_script(filename: str):
    """Import a numbered build script by path (names start with a digit → not importable normally).

    Mirrors the importlib trick the plinder builders use. Returns the module so callers can
    invoke its functions (run_structures, run_voxelize, ...) directly if needed.
    """
    path = _script_path(filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_").lstrip("0123456789_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_stage(stage: str, dataset: str, passthrough: list, dry_run: bool = False) -> None:
    """Dispatch one pipeline stage for one dataset to its ordered build steps.

    `passthrough` (unrecognized CLI args) is forwarded to EVERY 'run' step of the stage, so for a
    multi-step stage use only flags common to all steps (e.g. --max_complexes). Step-specific flags
    (e.g. --ligand_vdw) are injected from the spec during inlining, not passed through here.
    """
    if dataset not in SPEC:
        raise SystemExit(f"unknown --dataset {dataset!r}; choose from {DATASETS}")
    spec = SPEC[dataset]
    print(f"[{stage}] dataset={dataset}  frame={spec.frame}  voxelize_mode={spec.voxelize_mode}"
          f"  density_mode={spec.density_mode}  radii(lig={spec.ligand_radius},poc={spec.pocket_radius})")
    steps = spec.stages.get(stage, [])
    if not steps:
        print(f"  (no steps defined for {dataset} at {stage})")
        return
    for kind, *rest in steps:
        if kind == "note":
            print(f"  • {rest[0]}")
        elif kind == "todo":
            msg = f"{stage}/{dataset} not wired yet — {rest[0]}."
            if dry_run:
                print(f"  [todo] {msg}")
            else:
                raise NotImplementedError(
                    msg + "  (Phase 'extend branches'; run the legacy script directly meanwhile.)")
        elif kind == "run":
            script, subcmd = rest[0], rest[1]
            opts = rest[2] if len(rest) > 2 else {}
            fixed = opts.get("args", [])
            py = opts.get("py", sys.executable)
            sub = [subcmd] if subcmd else []
            argv_tail = [*sub, *fixed, *passthrough]
            pyshow = "python" if py == sys.executable else py
            spath = _script_path(script)                       # dataset/<dataset>/<script>
            shown = " ".join([pyshow, str(spath.relative_to(REPO_ROOT)), *argv_tail])
            # gemmi users (00b/01b/03c/02d) need the interpreter's own lib on LD_LIBRARY_PATH
            # (CXXABI); set it per-interpreter so it's correct for H5PY_PY too.
            libdir = str(Path(py).resolve().parent.parent / "lib")
            env = {**os.environ, "LD_LIBRARY_PATH": libdir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}
            if dry_run:
                print(f"  $ {shown}   (dry-run)")
            else:
                print(f"  $ {shown}")
                subprocess.run([py, str(spath), *argv_tail],
                               cwd=str(REPO_ROOT), check=True, env=env)
        else:
            raise ValueError(f"bad step kind {kind!r}")

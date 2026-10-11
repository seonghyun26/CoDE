"""01b_pdbbind_preprocess.py — Preprocess PDBbind v2020 into training voxels.

Consolidated Phase 4 / 4b entry point. Two subcommands:

    python dataset/pdbbind/01b_pdbbind_preprocess.py voxelize   # atoms (+ per-crop density) → voxels/
    python dataset/pdbbind/01b_pdbbind_preprocess.py poolnorm    # the probe density → voxels_v5/

──────────────────────────────────────────────────────────────────────────────
voxelize — voxelize PDBbind v2020 refined complexes (atoms + per-crop density)
──────────────────────────────────────────────────────────────────────────────
For each refined complex with has_struct=True in index.csv:
  1. Parse heavy atoms from {pid}_pocket.pdb (channels 0..3 = C,O,N,S)
     and from {pid}_ligand.sdf  (channels 0..6 = C,O,N,S,F,Cl,P).
  2. Compute the ligand heavy-atom geometric centre ("COM" in the existing
     voxel pipeline terminology).
  3. Recentre both atom sets to the ligand centre. Clamp to ±25 Å
     (matches voxbind.utils.dataset_utils.filter_atoms_by_distance).
  4. Voxelize with voxbind.voxelizer.Voxelizer (pyuul, 64³, 0.25 Å):
       - ligand : 7 channels, uniform 0.5 Å Gaussian radius
       - pocket : 4 channels, element-wise vdW radii
     and concatenate → (11, 64, 64, 64).  This is exactly the tensor
     `atomblob` encoder consumes during training.
  5. If a 2Fo-Fc CCP4 map is available (from `density` acquisition):
       - Load + globally z-score via _load_grid (gemmi).
       - Crop a 64³ × 0.25 Å box at the ligand centre via _crop_density
         with transform=None (PDBbind structures live in the deposited
         crystal frame, so no Kabsch alignment is needed).
       - Locally ±3σ-clip + re-z-score via normalize_crop.
     → (64, 64, 64) density, ready as the trailing channel for `atomblob_density`.

Centring on the ligand centre matches Beyond Atoms §3.1 ("both [grids] centered
on the ligand's center of mass") and the existing VoxBind training pipeline.
As in CrossDocked, this is implemented as an unweighted heavy-atom coordinate
mean, not a mass-weighted chemistry centre of mass.

    python dataset/pdbbind/01b_pdbbind_preprocess.py voxelize
    python dataset/pdbbind/01b_pdbbind_preprocess.py voxelize --max_complexes 20   # smoke test
    python dataset/pdbbind/01b_pdbbind_preprocess.py voxelize --no_density         # atoms only
    python dataset/pdbbind/01b_pdbbind_preprocess.py voxelize --device cpu         # if no GPU
  → voxels/atoms/{pid}.npy  voxels/density/{pid}.npy  voxels/availability.csv

──────────────────────────────────────────────────────────────────────────────
poolnorm — the probe density: pocket-pool arcsinh + z-score
──────────────────────────────────────────────────────────────────────────────
Crops each raw CCP4 map in dataset/data/pdbbind/ccp4/ at the ligand centre, without the per-map
z-score or per-crop clip of `voxelize`, and writes

    x' = (arcsinh(x / s) - mu_a) / sigma_a        s = 0.5

arcsinh is about linear near zero and logarithmic in the tail, so the near-zero bulk is kept and
the bright metal peaks (+30 raw) land near +9 with no clip. mu_a and sigma_a default to the
CrossDocked pocket-pool statistics the encoders were pretrained on (--stats_source reference, the
values built in as REFERENCE_STATS); --stats_source pdbbind fits them on these crops instead.
stats.json records both. Atom voxels are not touched: <out_dir>/atoms links to voxels/atoms.

    python dataset/pdbbind/01b_pdbbind_preprocess.py poolnorm --element_filter all
  → voxels_v5/density/{pid}.npy  voxels_v5/atoms → ../voxels/atoms  voxels_v5/stats.json

The directory keeps its old version name so existing feature caches stay valid. The other pool
normalisations this builder used to write (z-score, max-abs, clip + z-score: voxels_v2..v4) are in
legacy/dataset/pdbbind/01b_pdbbind_preprocess.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import gemmi BEFORE torch so its libstdc++.so.6 is loaded first. If torch
# (or PyUUL's CUDA libs) load an older libstdc++ ahead of gemmi, gemmi's C++
# extension fails with ImportError("... CXXABI_1.3.15 not found ...") inside
# every grid load and density crops silently return None.
# (see feedback_gemmi_torch_import_order memory)
import gemmi  # noqa: F401  (kept first to fix shared-library load order)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voxbind"))

from voxbind.constants import (
    ELEMENTS_HASH_CROSSDOCKED,
    RADIUS_PER_ATOM,
    is_diverse_role_atom,
    vdw_radius,
)
from voxbind.voxelizer import Voxelizer
from dataset.crossdocked_xray import _crop_density, normalize_crop


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR  = Path(__file__).parent.parent / "data" / "pdbbind"
INDEX_CSV    = PDBBIND_DIR / "index.csv"
STRUCT_DIR   = PDBBIND_DIR / "structures" / "pbpp-2020"
CCP4_DIR     = PDBBIND_DIR / "ccp4"
VOX_DIR      = PDBBIND_DIR / "voxels"
ATOMS_DIR    = VOX_DIR / "atoms"
DENSITY_DIR  = PDBBIND_DIR / "voxels_v5"    # the probe voxel set; keeps its old version name

N_LIG_CH = 7   # C,O,N,S,F,Cl,P
N_POC_CH = 4   # C,O,N,S
LIGAND_CHANNEL_ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "P"]
POCKET_CHANNEL_ELEMENTS = ["C", "O", "N", "S"]
LIGAND_SUPPORTED_HEAVY = set(LIGAND_CHANNEL_ELEMENTS)
POCKET_SUPPORTED_HEAVY = set(POCKET_CHANNEL_ELEMENTS)
GRID_DIM     = 64
RESOLUTION   = 0.25
LIGAND_RAD   = 0.5
CUBES_AROUND = 8
VOXELIZE_METADATA_VERSION = 2
CENTER_SOURCE = "ligand_sdf_all_non_h_geometric_centroid"

# poolnorm: x' = (arcsinh(x / s) - mu_a) / sigma_a. arcsinh(x/s) is ~linear for |x|<<s and ~log
# for |x|>>s, so the near-zero bulk is preserved and the +30 metal peaks compress smoothly (s=0.5:
# raw +30 → ≈+9 after the z-score, against +92 for a plain z-score). Every voxel is kept.
ARCSINH_SCALE = 0.5

# The CrossDocked pocket-pool statistics the canonical voxels_v5 were normalised with
# (--stats_source reference), as recorded in that directory's stats.json. They are built in
# because the CrossDocked stats.json they came from is not kept beside every checkout.
REFERENCE_STATS = {
    "path": "dataset/data/pdbbind/voxels_v5/stats.json",
    "arcsinh_scale": 0.5,
    "mu_a": -0.015487552481987226,
    "sigma_a": 0.5017691904023637,
}

# Element → channel index lookup (matches voxbind.constants.ELEMENTS_HASH_CROSSDOCKED)
_E2CH = ELEMENTS_HASH_CROSSDOCKED   # {"C":0, "O":1, "N":2, "S":3, "F":4, "Cl":5, "P":6, "H":7}
# Element-wise vdW radii (Å), shared by pocket atoms and — when --ligand_vdw is
# set — ligand atoms (mirrors crossdocked_xray's pocket / ligand_radius<=0 path).
_VDW_RADIUS_LUT = torch.tensor(
    [RADIUS_PER_ATOM["MOL"][e] for e in _E2CH.keys()],
    dtype=torch.float32,
)

# Hydrogen isotope labels to exclude from the ligand centering point. The voxel
# channels still use only C/O/N/S/F/Cl/P, but centering should not shift when a
# PDBbind ligand contains unsupported heavy atoms such as Br or I.
_HYDROGEN_ELEMS = {"H", "D", "T"}


# ═══════════════════════════════════════════════════════════════════════════════
# voxelize — atoms (11ch) + optional per-crop density
# ═══════════════════════════════════════════════════════════════════════════════

def _load_grid(ccp4_path: Path):
    """Load a CCP4 map → (arr_norm, frac_M_T, nu, nv, nw) or None.

    The voxelize loader: applies a per-map global z-score before cropping.
    (See `load_raw_grid` for the un-normalised variant used by poolnorm.)
    """
    try:
        m = gemmi.read_ccp4_map(str(ccp4_path))
        m.setup(float("nan"))
        grid = m.grid
        cell = grid.unit_cell
        arr = np.array(grid, dtype=np.float32)
        nu, nv, nw = arr.shape
        sigma = arr.std()
        if not np.isfinite(sigma) or sigma < 1e-10:
            return None
        arr_norm = (arr - arr.mean()) / sigma
        orth_mat = np.array(cell.orth.mat.tolist())
        frac_M = np.linalg.inv(orth_mat)
        return arr_norm, frac_M.T, nu, nv, nw
    except Exception:
        return None


def _channel_of(element: str, max_ch: int):
    ch = _E2CH.get(element)
    if ch is None or ch >= max_ch:
        return None
    return ch


def _normalise_element(element: str) -> str:
    """Return a canonical element symbol, e.g. 'CL'/'cl' → 'Cl'."""
    element = element.strip()
    if not element:
        return ""
    return element[0].upper() + element[1:].lower()


def _is_heavy_element(element: str) -> bool:
    """True for non-hydrogen atom symbols, including unsupported channels."""
    element = _normalise_element(element)
    return bool(element) and element not in _HYDROGEN_ELEMS


def expected_voxelize_metadata(args: argparse.Namespace) -> dict:
    """Metadata that defines whether an atom/density voxel cache is reusable."""
    other_channel = getattr(args, "other_channel", False)
    ligand_channels = (
        LIGAND_CHANNEL_ELEMENTS + ["other"] if other_channel else LIGAND_CHANNEL_ELEMENTS
    )
    return {
        "metadata_version": VOXELIZE_METADATA_VERSION,
        "kind": "pdbbind_voxelize",
        "center_source": CENTER_SOURCE,
        "grid_dim": GRID_DIM,
        "resolution": RESOLUTION,
        "cubes_around": CUBES_AROUND,
        "ligand_channels": ligand_channels,
        "pocket_channels": POCKET_CHANNEL_ELEMENTS,
        "element_filter": args.element_filter,
        "ligand_vdw": bool(args.ligand_vdw),
        "ligand_radius": None if args.ligand_vdw else LIGAND_RAD,
        "pocket_radius": (LIGAND_RAD if args.pocket_uniform else "element_vdw"),
        "density_enabled": not bool(args.no_density),
        "density_mode": None if args.no_density else "v1_global_zscore_crop_clip_zscore",
        "index_csv": str(Path(args.index_csv).expanduser().resolve()),
        "struct_dir": str(Path(args.struct_dir).expanduser().resolve()),
        "ccp4_dir": None if args.no_density else str(Path(args.ccp4_dir).expanduser().resolve()),
    }


def validate_voxelize_cache(out_dir: Path, expected: dict, *, overwrite: bool,
                            allow_stale_cache: bool) -> None:
    """Fail fast when existing .npy caches were produced by a different recipe."""
    meta_path = out_dir / "metadata.json"
    has_cache = any((out_dir / name).exists() and any((out_dir / name).glob("*.npy"))
                    for name in ("atoms", "density"))
    if not has_cache:
        return
    if overwrite:
        return
    if not meta_path.exists():
        msg = (
            f"{out_dir} contains cached voxels but no metadata.json. "
            "Re-run with --overwrite to rebuild with the current all-heavy "
            "ligand centre, or pass --allow_stale_cache to reuse legacy files."
        )
        if allow_stale_cache:
            print(f"  [warn] {msg}")
            return
        raise RuntimeError(msg)

    actual = json.loads(meta_path.read_text())
    mismatches = [
        f"{k}: got {actual.get(k)!r}, expected {v!r}"
        for k, v in expected.items()
        if actual.get(k) != v
    ]
    if mismatches and not allow_stale_cache:
        msg = "\n    ".join(mismatches[:12])
        raise RuntimeError(
            f"{out_dir} cache metadata does not match this voxelize command:\n"
            f"    {msg}\n"
            "Re-run with --overwrite, or pass --allow_stale_cache if this is intentional."
        )
    if mismatches:
        print(f"  [warn] stale cache metadata in {meta_path}; reusing because --allow_stale_cache was set")


def _unsupported_heavy(elements: set[str], supported: set[str]) -> list[str]:
    return sorted(e for e in elements if _is_heavy_element(e) and e not in supported)


def parse_pocket_pdb(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Heavy-atom coords + 4-channel atom channels for a pocket PDB file."""
    coords, channels, heavy_elements = [], [], set()
    with path.open() as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            el = _normalise_element(line[76:78])
            if not el:
                # Fall back: first letter of atom name (cols 13–16)
                el = _normalise_element(''.join(c for c in line[12:16] if c.isalpha())[:1])
            if _is_heavy_element(el):
                heavy_elements.add(el)
            ch = _channel_of(el, N_POC_CH)
            if ch is None:
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            channels.append(ch)
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(channels, dtype=torch.long),
        _unsupported_heavy(heavy_elements, POCKET_SUPPORTED_HEAVY),
    )


def parse_ligand_sdf(
    path: Path,
    other_channel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Supported ligand voxels plus all-heavy-atom centroid from SDF V2000.

    Returns
    -------
    coords, channels
        Atoms used for voxelization. By default only the 7-channel CrossDocked
        ligand vocabulary (C/O/N/S/F/Cl/P). When ``other_channel`` is True, every
        OTHER heavy ligand atom (Br/I/B/metals; not H, not a noble-gas/dummy
        artifact) is additionally kept and routed to channel ``N_LIG_CH`` (idx 7),
        mirroring 03c_plinder_preprocess.parse_cif_system(other_channel=True).
    all_heavy_centroid
        Geometric centroid of every non-H ligand atom in the SDF. This is used
        as the PDBbind grid center so unsupported heavy atoms (Br/I/metals) do
        not move the atom and density crops relative to the physical ligand.
    radii
        Per-KEPT-atom vdW radius (Å). Channels 0..6 receive the same element-wise
        vdW radii as ``_VDW_RADIUS_LUT``; channel 7 (the "other" channel) receives
        the true per-element vdW radius. Always returned (a 7-ch caller can ignore
        it); use it via ``voxelize_complex(..., lig_rad_override=radii)`` when
        feeding an 8-channel encoder.
    """
    coords, channels, radii, all_heavy_coords, heavy_elements = [], [], [], [], set()
    with path.open() as f:
        lines = f.readlines()
    n_atoms = int(lines[3][:3])
    for i in range(n_atoms):
        line = lines[4 + i]
        el = _normalise_element(line[31:34])
        if _is_heavy_element(el):
            heavy_elements.add(el)
            all_heavy_coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        ch = _channel_of(el, N_LIG_CH)
        if ch is None:
            # Out-of-vocab heavy atom (Br/I/B/metals). Default: drop (7-ch path).
            # other_channel: route to the 8th channel, gated like 03c on
            # is_diverse_role_atom (drops noble-gas / dummy artifacts).
            if not (other_channel and is_diverse_role_atom(el)):
                continue
            ch = N_LIG_CH                          # "other" channel = idx 7
        coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        channels.append(ch)
        radii.append(vdw_radius(_normalise_element(el)))
    if not all_heavy_coords:
        raise ValueError(f"{path} contains no non-H ligand atoms")
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(channels, dtype=torch.long),
        torch.tensor(all_heavy_coords, dtype=torch.float32).mean(dim=0),
        _unsupported_heavy(heavy_elements, LIGAND_SUPPORTED_HEAVY),
        torch.tensor(radii, dtype=torch.float32),
    )


def voxelize_complex(
    lig_xyz: torch.Tensor, lig_ch: torch.Tensor,
    poc_xyz: torch.Tensor, poc_ch: torch.Tensor,
    voxelizer: Voxelizer,
    device: str,
    ligand_vdw: bool = False,
    pocket_uniform: bool = False,
    ligand_center: "torch.Tensor | None" = None,
    n_lig_ch: int = N_LIG_CH,
    lig_rad_override: "torch.Tensor | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (((n_lig_ch+N_POC_CH),G,G,G) atom voxels, (3,) ligand centroid in deposited frame).

    ligand_vdw=True gives ligand atoms element-wise vdW radii (instead of the
    uniform LIGAND_RAD blob), matching an encoder trained with ligand_radius<=0.

    n_lig_ch selects the number of ligand voxelizer channels (default 7; pass 8 to
    match an 8-channel "other"-channel encoder — the pocket always stays N_POC_CH).
    lig_rad_override, if given, is used as the per-atom ligand radius directly
    (so the channel-7 "other" atoms get their true per-element vdW radius rather
    than a per-channel LUT constant); it overrides the ligand_vdw/uniform path.
    """
    # Heavy-atom centroid in deposited frame (used later for density cropping).
    lig_com = ligand_center if ligand_center is not None else lig_xyz.mean(dim=0)

    # Recentre to the ligand centre, clamp to ±25 Å (mirrors filter_atoms_by_distance).
    lig_xyz_c = (lig_xyz - lig_com).clamp(-25, 25)
    poc_xyz_c = (poc_xyz - lig_com).clamp(-25, 25)

    if lig_rad_override is not None:
        lig_rad = lig_rad_override                       # per-atom radii (incl. ch-7 "other")
    elif ligand_vdw:
        lig_rad = _VDW_RADIUS_LUT[lig_ch]                # element-wise vdW (ligand_radius<=0 path)
    else:
        lig_rad = torch.full_like(lig_ch, LIGAND_RAD, dtype=torch.float32)
    if pocket_uniform:
        poc_rad = torch.full_like(poc_ch, LIGAND_RAD, dtype=torch.float32)
    else:
        poc_rad = _VDW_RADIUS_LUT[poc_ch]

    # Voxelizer expects (B, N, *) shapes; B=1 here.
    lig_dict = {
        "coords":        lig_xyz_c.unsqueeze(0),
        "atoms_channel": lig_ch.float().unsqueeze(0),
        "radius":        lig_rad.unsqueeze(0),
    }
    poc_dict = {
        "coords":        poc_xyz_c.unsqueeze(0),
        "atoms_channel": poc_ch.float().unsqueeze(0),
        "radius":        poc_rad.unsqueeze(0),
    }

    v_lig = voxelizer(lig_dict, num_channels=n_lig_ch)   # (1, n_lig_ch, G, G, G)
    v_poc = voxelizer(poc_dict, num_channels=N_POC_CH)   # (1, 4, G, G, G)
    v_atom = torch.cat([v_lig, v_poc], dim=1)             # (1, n_lig_ch+4, G, G, G)
    return v_atom.squeeze(0).cpu().numpy(), lig_com.numpy()


def crop_density_for(pid: str, lig_com: np.ndarray, ccp4_dir: Path):
    """(G,G,G) locally-normalised density crop, or None if CCP4 unavailable/invalid."""
    p = ccp4_dir / f"{pid}.ccp4"
    if not p.exists():
        return None
    g = _load_grid(p)
    if g is None:
        return None
    arr_norm, frac_T, nu, nv, nw = g
    try:
        crop = _crop_density(arr_norm, frac_T, nu, nv, nw, lig_com,
                              G=GRID_DIM, res=RESOLUTION, transform=None)
    except Exception:
        return None
    return normalize_crop(crop) if crop is not None else None


def should_filter_complex(element_filter: str,
                          lig_unsupported: list[str],
                          poc_unsupported: list[str]) -> str | None:
    """Return a skip reason when the chosen VoxBind element filter excludes it."""
    if element_filter in ("ligand", "all") and lig_unsupported:
        return f"unsupported_ligand={','.join(lig_unsupported)}"
    if element_filter == "all" and poc_unsupported:
        return f"unsupported_pocket={','.join(poc_unsupported)}"
    return None


def remove_cached_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            path.unlink()


def run_voxelize(args: argparse.Namespace) -> None:
    index_csv  = Path(args.index_csv)
    struct_dir = Path(args.struct_dir)
    ccp4_dir   = Path(args.ccp4_dir)
    out_dir    = Path(args.out_dir)
    atom_dir   = out_dir / "atoms"
    dens_dir   = out_dir / "density"
    avail_csv  = out_dir / "availability.csv"
    other_channel = getattr(args, "other_channel", False)
    n_lig_ch   = N_LIG_CH + 1 if other_channel else N_LIG_CH

    atom_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_density:
        dens_dir.mkdir(parents=True, exist_ok=True)

    print("=== PDBbind v2020 — voxelization ===")
    print(f"  index_csv  : {index_csv}")
    print(f"  struct_dir : {struct_dir}")
    print(f"  ccp4_dir   : {ccp4_dir}")
    print(f"  out_dir    : {out_dir}")
    print(f"  device     : {args.device}")
    print(f"  no_density : {args.no_density}")
    print(f"  ligand_vdw : {args.ligand_vdw}")
    print(f"  other_chan : {other_channel} (n_lig_ch={n_lig_ch})")
    print(f"  elem_filter: {args.element_filter}")
    print(f"  overwrite  : {args.overwrite}")
    print(f"  cache_check: {'warn only' if args.allow_stale_cache else 'strict'}")

    expected_meta = expected_voxelize_metadata(args)
    validate_voxelize_cache(
        out_dir,
        expected_meta,
        overwrite=args.overwrite,
        allow_stale_cache=args.allow_stale_cache,
    )

    df = pd.read_csv(index_csv)
    df = df[df["has_struct"].astype(bool)].reset_index(drop=True)
    if args.max_complexes:
        df = df.head(args.max_complexes)
    print(f"  complexes  : {len(df):,}")

    voxelizer = Voxelizer(
        grid_dim=GRID_DIM,
        resolution=RESOLUTION,
        radius=LIGAND_RAD,
        cubes_around=CUBES_AROUND,
        device=args.device,
        backend="pyuul",
    )

    rows: list[dict] = []
    n_skipped = n_filtered = n_err = 0
    err_log: list[tuple[str, str]] = []
    filter_log: list[tuple[str, str]] = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="voxelize", unit="cplx")
    for _, row in pbar:
        pid = row["pdb_id"]
        atom_path = atom_dir / f"{pid}.npy"
        dens_path = dens_dir / f"{pid}.npy"

        wants_atom = args.overwrite or not atom_path.exists()
        wants_dens = (not args.no_density) and (args.overwrite or not dens_path.exists())

        # Already cached — record availability and move on.
        if not (wants_atom or wants_dens):
            rows.append({
                "pdb_id": pid,
                "has_atoms":   atom_path.exists(),
                "has_density": dens_path.exists() if not args.no_density else False,
                "n_lig": -1, "n_poc": -1,
                "filtered": False,
                "filter_reason": "",
                "unsupported_ligand_elements": "",
                "unsupported_pocket_elements": "",
            })
            n_skipped += 1
            continue

        try:
            cdir = struct_dir / pid
            lig_xyz, lig_ch, lig_center, lig_unsupported, lig_rad = parse_ligand_sdf(
                cdir / f"{pid}_ligand.sdf", other_channel=other_channel)
            poc_xyz, poc_ch, poc_unsupported = parse_pocket_pdb(cdir / f"{pid}_pocket.pdb")
            filter_reason = should_filter_complex(
                args.element_filter,
                lig_unsupported,
                poc_unsupported,
            )
            if filter_reason:
                remove_cached_outputs(atom_path, dens_path)
                n_filtered += 1
                filter_log.append((pid, filter_reason))
                rows.append({
                    "pdb_id": pid,
                    "has_atoms": False,
                    "has_density": False,
                    "n_lig": int(lig_xyz.shape[0]),
                    "n_poc": int(poc_xyz.shape[0]),
                    "filtered": True,
                    "filter_reason": filter_reason,
                    "unsupported_ligand_elements": ",".join(lig_unsupported),
                    "unsupported_pocket_elements": ",".join(poc_unsupported),
                })
                pbar.set_postfix(ok=len(rows) - n_filtered - n_err,
                                  filtered=n_filtered, err=n_err, refresh=False)
                continue

            n_lig, n_poc = int(lig_xyz.shape[0]), int(poc_xyz.shape[0])
            if n_lig == 0 or n_poc == 0:
                raise ValueError(f"empty ligand({n_lig}) or pocket({n_poc})")

            v_atom, lig_com = voxelize_complex(
                lig_xyz, lig_ch, poc_xyz, poc_ch, voxelizer, args.device,
                ligand_vdw=args.ligand_vdw,
                pocket_uniform=args.pocket_uniform,
                ligand_center=lig_center,
                n_lig_ch=n_lig_ch,
                lig_rad_override=lig_rad if other_channel else None,
            )
            if wants_atom:
                np.save(str(atom_path), v_atom.astype(np.float16))

            if wants_dens:
                v_dens = crop_density_for(pid, lig_com, ccp4_dir)
                if v_dens is not None:
                    np.save(str(dens_path), v_dens.astype(np.float16))

            rows.append({
                "pdb_id": pid,
                "has_atoms":   atom_path.exists(),
                "has_density": dens_path.exists() if not args.no_density else False,
                "n_lig": n_lig, "n_poc": n_poc,
                "filtered": False,
                "filter_reason": "",
                "unsupported_ligand_elements": ",".join(lig_unsupported),
                "unsupported_pocket_elements": ",".join(poc_unsupported),
            })
            pbar.set_postfix(ok=len(rows) - n_err, err=n_err, refresh=False)

        except Exception as e:
            n_err += 1
            err_log.append((pid, repr(e)[:160]))
            rows.append({
                "pdb_id": pid,
                "has_atoms":   atom_path.exists(),
                "has_density": dens_path.exists() if not args.no_density else False,
                "n_lig": -1, "n_poc": -1,
                "filtered": False,
                "filter_reason": "",
                "unsupported_ligand_elements": "",
                "unsupported_pocket_elements": "",
            })

    # ── Summary + availability CSV ────────────────────────────────────────────
    avail = pd.DataFrame(rows)
    avail.to_csv(avail_csv, index=False)
    (out_dir / "metadata.json").write_text(json.dumps(expected_meta, indent=2) + "\n")

    n_atom = int(avail["has_atoms"].sum())
    n_dens = int(avail["has_density"].sum())

    print()
    print("─" * 64)
    print(f"  Atom voxels written : {n_atom:,} / {len(avail):,}")
    if not args.no_density:
        print(f"  Density crops       : {n_dens:,} / {len(avail):,}")
    print(f"  Already cached      : {n_skipped:,}")
    print(f"  Filtered complexes  : {n_filtered:,}")
    print(f"  Errors              : {n_err:,}")
    if filter_log:
        filter_path = out_dir / "voxelize_filtered.txt"
        filter_path.write_text("\n".join(f"{p}\t{m}" for p, m in filter_log) + "\n")
        print(f"  Filter log          : {filter_path}")
    if err_log:
        err_path = out_dir / "voxelize_errors.txt"
        err_path.write_text("\n".join(f"{p}\t{m}" for p, m in err_log) + "\n")
        print(f"  Error log           : {err_path}")
    print(f"  availability CSV    : {avail_csv}")
    print(f"  metadata JSON       : {out_dir / 'metadata.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# poolnorm — the probe density: pocket-pool arcsinh + z-score
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_grid(ccp4_path: Path):
    """Return (arr_raw, frac_M_T, nu, nv, nw) or None — RAW CCP4 values.

    Unlike `_load_grid`, this does NOT apply the per-map z-score; poolnorm needs
    raw values to compute a single dataset-wide statistic.
    """
    try:
        m = gemmi.read_ccp4_map(str(ccp4_path))
        m.setup(float("nan"))
        grid = m.grid
        cell = grid.unit_cell
        arr = np.array(grid, dtype=np.float32)
        nu, nv, nw = arr.shape
        if not np.isfinite(arr).all():
            return None
        orth_mat = np.array(cell.orth.mat.tolist())
        frac_M = np.linalg.inv(orth_mat)
        return arr, frac_M.T, nu, nv, nw
    except Exception:
        return None


def parse_ligand_com(sdf_path: Path) -> "np.ndarray | None":
    """All-heavy-atom geometric centroid from a ligand SDF V2000 file."""
    try:
        with sdf_path.open() as f:
            lines = f.readlines()
        n_atoms = int(lines[3][:3])
        coords = []
        for i in range(n_atoms):
            line = lines[4 + i]
            el = _normalise_element(line[31:34])
            if not _is_heavy_element(el):
                continue
            coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        if not coords:
            return None
        return np.array(coords, dtype=np.float32).mean(axis=0)
    except Exception:
        return None


def unsupported_ligand_sdf(sdf_path: Path) -> list[str]:
    """Unsupported non-H ligand elements relative to the 7 VoxBind ligand channels."""
    with sdf_path.open() as f:
        lines = f.readlines()
    n_atoms = int(lines[3][:3])
    elems = {
        _normalise_element(lines[4 + i][31:34])
        for i in range(n_atoms)
    }
    return _unsupported_heavy(elems, LIGAND_SUPPORTED_HEAVY)


def unsupported_pocket_pdb(pdb_path: Path) -> list[str]:
    """Unsupported non-H pocket elements relative to the 4 VoxBind pocket channels."""
    elems = set()
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            el = _normalise_element(line[76:78])
            if not el:
                el = _normalise_element(''.join(c for c in line[12:16] if c.isalpha())[:1])
            elems.add(el)
    return _unsupported_heavy(elems, POCKET_SUPPORTED_HEAVY)


def _reference_stats(path: "str | None") -> dict:
    """mu_a / sigma_a to normalise with: REFERENCE_STATS, or those of a stats.json."""
    if not path:
        return dict(REFERENCE_STATS)
    path = Path(path)
    stats = json.loads(path.read_text())
    missing = [k for k in ("mu_a", "sigma_a") if k not in stats]
    if missing:
        raise ValueError(f"{path} is missing required stat key(s): {missing}")
    sigma = float(stats["sigma_a"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"invalid sigma_a in {path}: {sigma}")
    return {
        "path": str(path),
        "arcsinh_scale": float(stats.get("arcsinh_scale", ARCSINH_SCALE)),
        "mu_a": float(stats["mu_a"]),
        "sigma_a": sigma,
    }


def run_poolnorm(args: argparse.Namespace) -> None:
    index_csv  = Path(args.index_csv)
    struct_dir = Path(args.struct_dir)
    ccp4_dir   = Path(args.ccp4_dir)
    atoms_dir  = Path(args.atoms_dir).resolve()
    out_dir    = Path(args.out_dir)
    dens_dir   = out_dir / "density"
    dens_dir.mkdir(parents=True, exist_ok=True)

    # Optional ‖∇ρ‖ channel saved next to the density (drop-in gradmag source for the probe's
    # --noise_voxels_dir): per_sample_zscore(‖∇·‖) of the FINAL normalized density — exactly as the
    # training/probe pipeline derives it.
    gmag_root = gmag_dens = _gmag = None
    if args.save_gradmag:
        from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore
        gmag_root = out_dir.with_name(out_dir.name + "_gradmag")
        gmag_dens = gmag_root / "density"
        gmag_dens.mkdir(parents=True, exist_ok=True)

        def _gmag(c16):
            g = gradient_magnitude3d(torch.from_numpy(c16.astype(np.float32))
                                     .view(1, 1, GRID_DIM, GRID_DIM, GRID_DIM))
            return per_sample_zscore(g).view(GRID_DIM, GRID_DIM, GRID_DIM).numpy().astype(np.float16)

    link = out_dir / "atoms"
    if not link.exists() and not link.is_symlink():
        os.symlink(atoms_dir, link)
        print(f"  [symlink] {link} → {atoms_dir}")

    ref = _reference_stats(args.reference_stats_json) if args.stats_source == "reference" else None
    s = ref["arcsinh_scale"] if ref is not None else args.arcsinh_scale

    print("=== PDBbind probe density: pocket-pool arcsinh + z-score ===")
    print(f"  index_csv : {index_csv}")
    print(f"  ccp4_dir  : {ccp4_dir}")
    print(f"  out_dir   : {out_dir}")
    print(f"  norm      : arcsinh(x / {s}) → z-score  (soft-squash, no clip)")
    print(f"  stats     : {args.stats_source}")
    print(f"  elem_filter: {args.element_filter}")

    df = pd.read_csv(index_csv)
    df = df[df["has_struct"].astype(bool)].reset_index(drop=True)

    # Only complexes that have a CCP4 map on disk
    pids = [pid for pid in df["pdb_id"].tolist()
            if (ccp4_dir / f"{pid}.ccp4").exists()]
    if args.max_complexes:
        pids = pids[:args.max_complexes]
    print(f"  complexes : {len(pids):,}")

    # ── Pass 1: crop each pocket from RAW CCP4 ───────────────────────────────
    crops: dict[str, np.ndarray] = {}    # pid → float16 (G, G, G)
    n_voxels = 0
    min_v = +np.inf
    max_v = -np.inf

    n_skipped = 0
    skip_log: list[tuple[str, str]] = []
    pbar = tqdm(pids, desc="pass 1: crop", unit="cplx")
    for pid in pbar:
        g = load_raw_grid(ccp4_dir / f"{pid}.ccp4")
        if g is None:
            n_skipped += 1; skip_log.append((pid, "load_raw_grid")); continue
        arr, frac_T, nu, nv, nw = g

        lig_path = struct_dir / pid / f"{pid}_ligand.sdf"
        poc_path = struct_dir / pid / f"{pid}_pocket.pdb"
        try:
            lig_unsupported = unsupported_ligand_sdf(lig_path) \
                if args.element_filter in ("ligand", "all") else []
            poc_unsupported = unsupported_pocket_pdb(poc_path) \
                if args.element_filter == "all" else []
        except Exception as e:
            n_skipped += 1; skip_log.append((pid, f"element_filter:{e!r}")); continue
        filter_reason = should_filter_complex(args.element_filter, lig_unsupported, poc_unsupported)
        if filter_reason:
            n_skipped += 1; skip_log.append((pid, filter_reason)); continue

        lig_com = parse_ligand_com(lig_path)
        if lig_com is None:
            n_skipped += 1; skip_log.append((pid, "parse_ligand_com")); continue

        try:
            crop = _crop_density(arr, frac_T, nu, nv, nw, lig_com,
                                  G=GRID_DIM, res=RESOLUTION, transform=None)
        except Exception as e:
            n_skipped += 1; skip_log.append((pid, f"crop:{e!r}")); continue

        n_voxels += crop.size
        min_v = min(min_v, float(crop.min()))
        max_v = max(max_v, float(crop.max()))
        crops[pid] = crop.astype(np.float16)        # save mem for pass 2

    if n_voxels == 0:
        print("[error] no crops produced; aborting.")
        sys.exit(1)

    # Fit of the arcsinh statistics on these crops, in float64; used only with
    # --stats_source pdbbind, recorded either way.
    sum_a = sum_a2 = 0.0
    for _c in crops.values():
        _a = np.arcsinh(_c.astype(np.float64) / s)
        sum_a += _a.sum(); sum_a2 += (_a * _a).sum()
    fit_mu_a    = sum_a / n_voxels
    fit_sigma_a = float(np.sqrt(max(sum_a2 / n_voxels - fit_mu_a * fit_mu_a, 0.0)))
    mu_a, sigma_a = (ref["mu_a"], ref["sigma_a"]) if ref is not None else (fit_mu_a, fit_sigma_a)

    print(f"\n── pocket-pool stats ───────────────────────────────────────────")
    print(f"  n_crops          : {len(crops):,}")
    print(f"  n_voxels (total) : {n_voxels:,}")
    print(f"  raw min / max    : [{min_v:+.4f}, {max_v:+.4f}]")
    print(f"  arcsinh s={s}     : fit μ_a {fit_mu_a:+.6f}  σ_a {fit_sigma_a:.6f}")
    if ref is not None:
        print(f"  reference        : μ_a {mu_a:+.6f}  σ_a {sigma_a:.6f}  ({ref['path']})")
    print(f"  skipped          : {n_skipped:,}")

    stats = {
        "metadata_version": VOXELIZE_METADATA_VERSION,
        "center_source":   CENTER_SOURCE,
        "grid_dim":        GRID_DIM,
        "resolution":      RESOLUTION,
        "element_filter":  args.element_filter,
        "scheme":         "pocket-pool arcsinh soft-squash + z-score (no clip)",
        "formula":        "x' = (arcsinh(x / s) - mu_a) / sigma_a",
        "arcsinh_scale":  s,
        "stats_source":   args.stats_source,
        "reference_stats_json": ref["path"] if ref is not None else None,
        "mu_a":           mu_a,
        "sigma_a":        sigma_a,
        "pdbbind_fit_mu_a":    fit_mu_a,
        "pdbbind_fit_sigma_a": fit_sigma_a,
        "n_crops":        len(crops),
        "n_voxels_total": n_voxels,
        "raw_min":        min_v,
        "raw_max":        max_v,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    if gmag_root is not None:
        (gmag_root / "stats.json").write_text(json.dumps({
            "metadata_version": VOXELIZE_METADATA_VERSION,
            "scheme":          "gradient magnitude ‖∇ρ‖ of PDBbind v5 density",
            "formula":         "x' = per_sample_zscore(‖∇(density_v5)‖)",
            "density_version": "v5", "gradmag": True,
            "grid_dim": GRID_DIM, "resolution": RESOLUTION, "n_crops": len(crops),
        }, indent=2))

    # ── Pass 2: normalise and save ──────────────────────────────────────────
    print(f"\n── pass 2: normalise + save ───────────────────────────────────")
    for pid, crop16 in tqdm(crops.items(), desc="pass 2: normalise + save", unit="cplx"):
        c = ((np.arcsinh(crop16.astype(np.float32) / s) - mu_a) / sigma_a).astype(np.float16)
        np.save(str(dens_dir / f"{pid}.npy"), c)
        if gmag_dens is not None:
            np.save(str(gmag_dens / f"{pid}.npy"), _gmag(c))

    if skip_log:
        skip_path = PDBBIND_DIR / "voxels_poolnorm_skip_log.txt"
        skip_path.write_text("\n".join(f"{p}\t{m}" for p, m in skip_log) + "\n")
        print(f"  skip log         : {skip_path}")

    print(f"\n  crops written    : {len(crops):,} → {dens_dir}")
    print(f"  stats            : {out_dir / 'stats.json'}")
    print("\n  Next:  python task1-affinity_regression/probe.py features --condition <condition>")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess PDBbind v2020 into voxels (voxelize | poolnorm)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser(
        "voxelize",
        help="Voxelize refined set (atoms + optional per-crop density)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pv.add_argument("--index_csv",  default=str(INDEX_CSV))
    pv.add_argument("--struct_dir", default=str(STRUCT_DIR))
    pv.add_argument("--ccp4_dir",   default=str(CCP4_DIR))
    pv.add_argument("--out_dir",    default=str(VOX_DIR))
    pv.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Voxelizer device (cuda recommended)")
    pv.add_argument("--max_complexes", type=int, default=0,
                    help="Limit to first N complexes (0 = all). Smoke testing.")
    pv.add_argument("--no_density", action="store_true",
                    help="Skip CCP4 crop (atom voxels only)")
    pv.add_argument("--pocket_uniform", action="store_true",
                    help="Uniform LIGAND_RAD pocket blobs instead of element-wise vdW "
                         "(match an encoder trained with pocket_radius>0).")
    pv.add_argument("--ligand_vdw", action="store_true",
                    help="Element-wise vdW ligand radii (mirrors crossdocked_xray "
                         "ligand_radius<=0) instead of the uniform 0.5 A blob. Match "
                         "to a gradmag/ligvdw encoder; write to a separate --out_dir.")
    pv.add_argument("--other_channel", action="store_true",
                    help="ligand: route out-of-vocab heavy atoms to an 8th channel "
                         "(matches an 8-ch encoder)")
    pv.add_argument("--element_filter", choices=["none", "ligand", "all"], default="none",
                    help="Drop complexes with unsupported non-H atom types. none = keep "
                         "and ignore unsupported channels; ligand = drop Br/I/etc in "
                         "the ligand only; all = also drop unsupported pocket metals/ions.")
    pv.add_argument("--overwrite",  action="store_true",
                    help="Recompute outputs even if .npy already exists")
    pv.add_argument("--allow_stale_cache", action="store_true",
                    help="Warn instead of failing when existing voxel caches have "
                         "missing/mismatched metadata. Use only for intentional "
                         "legacy-cache reuse.")
    pv.set_defaults(func=run_voxelize)

    pp = sub.add_parser(
        "poolnorm",
        help="The probe density: pocket-pool arcsinh + z-score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pp.add_argument("--index_csv",  default=str(INDEX_CSV))
    pp.add_argument("--struct_dir", default=str(STRUCT_DIR))
    pp.add_argument("--ccp4_dir",   default=str(CCP4_DIR))
    pp.add_argument("--atoms_dir",  default=str(ATOMS_DIR),
                    help="Atom voxels to link as <out_dir>/atoms")
    pp.add_argument("--out_dir",    default=str(DENSITY_DIR))
    pp.add_argument("--stats_source", choices=["reference", "pdbbind"], default="reference",
                    help="reference = normalise with the CrossDocked statistics the encoders "
                         "were pretrained on (REFERENCE_STATS, or --reference_stats_json); "
                         "pdbbind = fit mu_a / sigma_a on these crops")
    pp.add_argument("--reference_stats_json", default=None,
                    help="stats.json with arcsinh_scale, mu_a, sigma_a to use instead of "
                         "REFERENCE_STATS")
    pp.add_argument("--arcsinh_scale", type=float, default=ARCSINH_SCALE,
                    help="s in arcsinh(x/s) with --stats_source pdbbind (a reference brings its own)")
    pp.add_argument("--element_filter", choices=["none", "ligand", "all"], default="none",
                    help="Apply the same unsupported-element filter while building "
                         "density normalisation crops. Use the same value as voxelize.")
    pp.add_argument("--max_complexes", type=int, default=0,
                    help="Limit to first N (0 = all). Smoke testing.")
    pp.add_argument("--save_gradmag", action="store_true",
                    help="Also save ‖∇ρ‖ next to the density: <out_dir>_gradmag/density/{pid}.npy "
                         "= per-sample-z-scored ‖∇(density)‖, + stats.json. Drop-in for the "
                         "probe's --noise_voxels_dir.")
    pp.set_defaults(func=run_poolnorm)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

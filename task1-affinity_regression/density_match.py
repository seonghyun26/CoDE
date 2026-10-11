"""The probe density on the scale the encoder was pretrained on.

Every density crop is stored as x' = (arcsinh(x / s) - mu) / sigma, with the statistics of the set it
was built for. The PDBbind probe voxels use the CrossDocked pocket statistics
(dataset/data/pdbbind/voxels_v5/stats.json), while an encoder pretrained on a PLINDER corpus saw that
corpus's statistics: the normalization in the resample.json of its dset.resample_dir. With the same
arcsinh scale s the two differ by an affine map,

    x'_encoder = a * x'_probe + b,    a = sigma_probe / sigma_encoder,    b = (mu_probe - mu_encoder) / sigma_encoder,

so feature extraction maps the stored voxels at load time instead of rebuilding them. Encoders whose
cfg names no resample.json (pretrained on the CrossDocked crops, or a cfg whose paths have moved) keep
the stored scale. Standard library and OmegaConf only, so run.py can check a cache without torch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
PRETRAIN = REPO / "pretrain"            # the trainer's working directory, for relative cfg paths


def _arcsinh_stats(norm: dict) -> Optional[tuple]:
    if "arcsinh" not in str(norm.get("scheme", norm.get("formula", ""))).lower():
        return None
    try:
        return float(norm["arcsinh_scale"]), float(norm["mu_a"]), float(norm["sigma_a"])
    except (KeyError, TypeError, ValueError):
        return None


def encoder_norm(cfg) -> tuple[Optional[dict], str]:
    """(the encoder's pretraining density normalization, where it was read), or (None, why not)."""
    from omegaconf import OmegaConf
    try:
        rd = OmegaConf.select(cfg, "dset.resample_dir", default=None)
    except Exception as e:  # noqa: BLE001 - an unresolvable interpolation
        return None, f"dset.resample_dir unresolved ({type(e).__name__})"
    if not rd:
        return None, "no dset.resample_dir (pretrained on fixed crops)"
    path = Path(os.path.expanduser(str(rd)))
    # relative to the trainer's working directory, pretrain/ (runs before 2026-10 ran from the repo root)
    tried = [path] if path.is_absolute() else [(PRETRAIN / path).resolve(), (REPO / path).resolve()]
    for base in tried:
        recipe = base / "resample.json"
        if recipe.exists():
            return json.loads(recipe.read_text()).get("normalization"), str(recipe)
    return None, f"{tried[-1] / 'resample.json'} not found"


def plan(cfg, voxel_stats_path, *, overridden: bool = False, disabled: bool = False) -> dict:
    """{applied, scale, shift, reason, encoder_norm, voxel_norm} for one encoder and voxel set.

    overridden: the density comes from another directory (--dens_dir, --noise_voxels_dir), whose values
    are taken as they are.
    """
    out = {"applied": False, "scale": 1.0, "shift": 0.0, "encoder_norm": None, "voxel_norm": None}
    if disabled:
        return {**out, "reason": "--no_density_match"}
    if overridden:
        return {**out, "reason": "density read from an override directory, used as stored"}
    enc, where = encoder_norm(cfg)
    try:
        vox = json.loads(Path(voxel_stats_path).read_text())
    except (OSError, ValueError):
        return {**out, "reason": f"{voxel_stats_path} unreadable"}
    out["voxel_norm"] = {k: vox.get(k) for k in ("arcsinh_scale", "mu_a", "sigma_a")}
    if enc is None:
        return {**out, "reason": where}
    out["encoder_norm"] = {k: enc.get(k) for k in ("scheme", "arcsinh_scale", "mu_a", "sigma_a")}
    e, v = _arcsinh_stats(enc), _arcsinh_stats({"scheme": "arcsinh", **vox})
    if e is None or v is None or abs(e[0] - v[0]) > 1e-12:
        return {**out, "reason": f"not an arcsinh map with the voxels' scale ({where})"}
    scale, shift = v[2] / e[2], (v[1] - e[1]) / e[2]
    applied = abs(scale - 1.0) > 1e-12 or abs(shift) > 1e-12
    return {**out, "applied": applied, "scale": scale, "shift": shift,
            "reason": f"{'mapped to' if applied else 'same statistics as'} {where}"}


def same(a: Optional[dict], b: Optional[dict]) -> bool:
    """Whether two plans transform the density the same way (a missing plan is the stored scale)."""
    def key(p):
        if not p or not p.get("applied"):
            return (False, 1.0, 0.0)
        return (True, round(float(p["scale"]), 12), round(float(p["shift"]), 12))
    return key(a) == key(b)

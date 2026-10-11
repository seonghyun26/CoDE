"""24 Å context from original pocket atoms and experimental CCP4 maps."""

import json
import sys

import gemmi
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from overview_scene import REPO_ROOT


def expanded_volumes(scene):
    ns = scene.ns
    if "expanded_volumes" in ns:
        return ns["expanded_volumes"]
    # Reuse the dataset's exact interpolation and PyUUL occupancy implementation.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import torch
    from voxbind.constants import RADIUS_PER_ATOM
    from dataset.crossdocked_xray import _crop_density
    from voxbind.voxelizer import Voxelizer

    root = REPO_ROOT / "dataset/data/pdbbind"
    sample, resolution = ns["SAMPLE_ID"], ns["RESOLUTION"]
    size = int(round(24 / resolution))
    pad = (size - ns["GRID_DIM"]) // 2
    inner = (slice(pad, pad + ns["GRID_DIM"]),) * 3
    source = root / "structures/pbpp-2020" / sample / f"{sample}_pocket.pdb"
    elements = ["C", "O", "N", "S"]
    coords, channels, radii = [], [], []
    for line in source.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip().upper() or ''.join(c for c in line[12:16] if c.isalpha())[:1].upper()
        if element not in elements:
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        channels.append(elements.index(element))
        radii.append(RADIUS_PER_ATOM["MOL"][element])
    coords = np.asarray(coords, dtype=np.float32) - ns["ligand_center"]
    batch = dict(coords=torch.tensor(coords.clip(-25, 25))[None],
                 atoms_channel=torch.tensor(channels, dtype=torch.float32)[None],
                 radius=torch.tensor(radii, dtype=torch.float32)[None])
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(4, threads))
        voxelizer = Voxelizer(grid_dim=size, resolution=resolution, cubes_around=8,
                              device="cpu", backend="pyuul")
        with torch.no_grad():
            pocket = voxelizer(batch, num_channels=4)[0].max(dim=0).values.numpy()
    finally:
        torch.set_num_threads(threads)
    # Preserve the stored tensor exactly in the original crop (float16 cache).
    pocket[inner] = ns["pocket_voxels"]

    map_path = root / "ccp4" / f"{sample}.ccp4"
    if not map_path.is_file():
        raise ValueError(f"Expanded density map unavailable for {sample.upper()}.")
    ccp4 = gemmi.read_ccp4_map(str(map_path))
    ccp4.setup(float("nan"))
    raw = np.array(ccp4.grid, dtype=np.float32)
    fractional = np.linalg.inv(np.asarray(ccp4.grid.unit_cell.orth.mat.tolist())).T
    density = _crop_density(raw, fractional, *raw.shape, ns["ligand_center"], G=size, res=resolution)
    if not np.isfinite(density).all():
        raise ValueError("Expanded density contains missing map values.")
    stats = json.loads((root / "voxels_v5/stats.json").read_text())
    density = ((np.arcsinh(density / stats["arcsinh_scale"]) - stats["mu_a"]) / stats["sigma_a"]).astype(np.float32)
    density[inner] = ns["density"]
    step = ns["VIEW_STEP"]
    reduce = ns["block_reduce"]
    centers = (np.arange(size // step, dtype=np.float32) * step + (step - 1) / 2 - size / 2) * resolution
    smoothed = gaussian_filter(reduce(density, step, "mean"), ns["DENSITY_DISPLAY_SMOOTH_SIGMA"], mode="nearest")
    points = np.stack(np.meshgrid(centers, centers, centers, indexing="ij"), axis=-1)
    distance = cKDTree(ns["density_reference_coords"]).query(points.reshape(-1, 3))[0].reshape(smoothed.shape)
    weight = np.clip((ns["DENSITY_PROXIMITY_FULL_RADIUS_A"] + ns["DENSITY_PROXIMITY_FEATHER_A"] - distance)
                     / ns["DENSITY_PROXIMITY_FEATHER_A"], 0, 1).astype(np.float32)
    gradient = np.sqrt(sum(g ** 2 for g in np.gradient(density, resolution, edge_order=2)))
    mask = np.pad(ns["spatial_mask_view"], pad // step, constant_values=False)
    result = dict(pocket=reduce(pocket, step, "max"), density=smoothed,
                  proximity_density=smoothed * weight, gradient=reduce(gradient, step, "max"),
                  mask=mask, centers=centers)
    ns["expanded_volumes"] = result
    return result

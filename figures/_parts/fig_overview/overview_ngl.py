"""Aligned molecular structures and voxel geometry for the NGL overview."""

import base64
import json
from types import FunctionType

import numpy as np
from rdkit import Chem
from skimage.measure import marching_cubes

from overview_scene import PALETTE, REPO_ROOT
from overview_expanded import expanded_volumes


def packed(values, dtype="<f4"):
    array = np.asarray(values, dtype=dtype).ravel()
    return base64.b64encode(array.tobytes()).decode()


def lines(x, y, z, color, width, opacity=1):
    starts, ends = [], []
    previous = None
    for point in zip(x, y, z):
        if any(value is None for value in point):
            previous = None
            continue
        if previous is not None:
            starts.append(previous)
            ends.append(point)
        previous = point
    return dict(kind="lines", start=packed(starts), end=packed(ends),
                color=color, width=width, opacity=opacity)


def surface(volume, threshold, origin, spacing, color, opacity=1):
    if not float(volume.min()) < threshold < float(volume.max()):
        return dict(kind="mesh", position="", index="", color=color, opacity=opacity)
    vertices, faces, _, _ = marching_cubes(volume, level=threshold,
                                          spacing=(spacing,) * 3, allow_degenerate=False)
    return dict(kind="mesh", position=packed(vertices + origin), index=packed(faces, "<u4"),
                color=color, opacity=opacity)


def density_wireframe(volume, threshold, origin, spacing, color, width):
    """Isodensity mesh along sampled cell faces, without triangulation diagonals."""
    if not float(volume.min()) < threshold < float(volume.max()):
        return dict(kind="lines", start="", end="", color=color, width=width, opacity=.95, style="mesh")
    vertices, faces, _, _ = marching_cubes(volume, level=threshold, spacing=(spacing,) * 3,
                                          step_size=3, allow_degenerate=False)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    # Keep the lattice-aligned surface network familiar from density-map
    # viewers. Interior triangulation diagonals only add visual clutter.
    lattice = vertices[edges] / (spacing * 3)
    rounded = np.rint(lattice)
    on_plane = np.all(np.isclose(lattice, rounded, atol=1e-5), axis=1)
    same_plane = rounded[:, 0] == rounded[:, 1]
    edges = edges[np.any(on_plane & same_plane, axis=1)]
    vertices += origin
    return dict(kind="lines", start=packed(vertices[edges[:, 0]]), end=packed(vertices[edges[:, 1]]),
                color=color, width=width, opacity=.95, style="mesh")


def centered_structure(path, center, pocket=False):
    mol = (Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False) if pocket else
           Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False))
    if mol is None:
        raise ValueError(f"Could not read {path.name}")
    edit = Chem.RWMol(mol)
    conformer = edit.GetConformer()
    remove = []
    for i, atom in enumerate(edit.GetAtoms()):
        point = np.array(conformer.GetAtomPosition(i)) - center
        conformer.SetAtomPosition(i, point)
        if atom.GetAtomicNum() == 1 or (pocket and np.any(np.abs(point) > 12.15)):
            remove.append(i)
    for i in reversed(remove):
        edit.RemoveAtom(i)
    return Chem.MolToPDBBlock(edit) if pocket else Chem.MolToMolBlock(edit) + "\n$$$$\n"


def centered_protein(scene):
    """Keep complete backbone chains and secondary-structure records for cartoon."""
    ns = scene.ns
    if "cartoon_pdb" not in ns:
        sample = ns["SAMPLE_ID"]
        path = REPO_ROOT / "dataset/data/pdbbind/structures/pbpp-2020" / sample / f"{sample}_protein.pdb"
        records = []
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM  ", "HETATM")):
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) - ns["ligand_center"]
                line = line[:30] + ''.join(f"{value:8.3f}" for value in xyz) + line[54:]
            records.append(line)
        ns["cartoon_pdb"] = '\n'.join(records) + '\n'
    return ns["cartoon_pdb"]


def mask_boxes(ns, mask, gap=0):
    low = -ns["HALF_EXTENT"] + np.argwhere(mask) * ns["MASK_BLOCK_WIDTH_A"] + gap / 2
    high = low + ns["MASK_BLOCK_WIDTH_A"] - gap
    return packed(np.column_stack((low, high)))


def payload(scene, view, iso=None, proximity=True, expanded=False, density_style="mesh"):
    ns = scene.ns
    if iso is None:
        iso = 0.35 if view == "gradient" else 1.0
    if not np.isfinite(iso) or not 0.05 <= iso <= (2 if view == "gradient" else ns["density_ceiling"]):
        raise ValueError("Invalid threshold.")
    if density_style not in ("mesh", "contours"):
        raise ValueError("Invalid density style.")
    layers = []
    context = expanded_volumes(scene) if expanded else None
    centers = context["centers"] if context else ns["centers"]
    pocket = context["pocket"] if context else ns["pocket_view"]
    masked_pocket = np.where(context["mask"], 0.0, pocket) if context else ns["masked_pocket_view"]

    def add(key, label, geometry, visible=True):
        layers.append(dict(key=key, label=label, visible=visible, **geometry))

    def occupancy(volume, color, opacity, origin=None):
        return surface(volume, .45, ns["centers"][0] if origin is None else origin,
                       ns["RESOLUTION"] * ns["VIEW_STEP"], color, opacity)

    if view == "overview":
        add("ligand", "Ligand sticks", dict(kind="ligand", color=PALETTE["ligand"]))
        add("protein", "Protein surface", dict(kind="protein", color=PALETTE["pocket_occupancy"], opacity=.15), False)
        add("protein-sticks", "Protein sticks", dict(kind="protein-sticks", color=PALETTE["pocket"]))
        add("protein-cartoon", "Protein cartoon", dict(kind="protein-cartoon", color=PALETTE["pocket"], opacity=1, gridOffOnly=True))
        add("ligand-occupancy", "Ligand occupancy", occupancy(ns["ligand_view"], PALETTE["ligand_occupancy"], .4), False)
        add("pocket-occupancy", "Pocket occupancy", occupancy(pocket, PALETTE["pocket_occupancy"], .4, centers[0]), False)
    elif view == "gradient":
        add("gradient", "Density gradient |∇ρ|", surface(context["gradient"] if context else ns["density_gradient_view"], iso,
            centers[0], ns["RESOLUTION"] * ns["VIEW_STEP"], PALETTE["gradient"]))
    elif view == "masked":
        add("ligand", "Ligand sticks", dict(kind="ligand", color=PALETTE["ligand"]), False)
        add("protein-sticks", "Protein sticks", dict(kind="protein-sticks", color=PALETTE["pocket"]), False)
        add("ligand-occupancy", "Ligand occupancy", occupancy(ns["masked_ligand_view"], PALETTE["ligand_occupancy"], .85))
        add("pocket-occupancy", "Pocket occupancy", occupancy(masked_pocket, PALETTE["pocket_occupancy"], .85, centers[0]))
    else:
        raise ValueError("Unknown view.")
    # Overview owns the overlay too; toggling it never replaces molecular data.
    if view in ("overview", "masked"):
        boxes = mask_boxes(ns, ns["display_mask"], ns["MASK_GAP_A"])
        add("mask-fill", "Masking blocks", dict(kind="mesh", maskBoxes=boxes,
            color=ns["MASK_FILL_COLOR"], opacity=ns["MASK_FILL_OPACITY"]), view == "masked")
        add("mask-outline", "Mask outline", dict(kind="lines", maskBoxes=boxes,
            outlineOffset=ns["MASK_OUTLINE_OFFSET_A"], color=ns["MASK_LINE_COLOR"],
            width=ns["MASK_LINE_WIDTH"], opacity=1), view == "masked")
    if view != "gradient":
        volume = ns["primary_density_view" if view == "overview" else "masked_density_view"]
        if not proximity:
            volume = ns["density_smoothed_view"]
            if view == "masked":
                volume = np.where(ns["spatial_mask_view"], ns["MASKED_DENSITY_ISO"] - 1.0, volume)
        contour = ns["contour_line_coordinates"]
        if context:
            volume = context["proximity_density" if proximity else "density"]
            if view == "masked":
                volume = np.where(context["mask"], ns["MASKED_DENSITY_ISO"] - 1.0, volume)
            # Reuse notebook contour rules with expanded coordinates, without
            # mutating the shared namespace or another browser's crop.
            contour = FunctionType(contour.__code__, {**ns, "centers": centers}, argdefs=contour.__defaults__)
        if density_style == "mesh":
            geometry = density_wireframe(volume, iso, centers[0], ns["RESOLUTION"] * ns["VIEW_STEP"],
                                         PALETTE["density"], ns["DENSITY_CONTOUR_WIDTH"])
        else:
            x, y, z = contour(volume, iso)
            geometry = dict(**lines(x, y, z, PALETTE["density"], ns["DENSITY_CONTOUR_WIDTH"], .6), style="contours")
        add("density", "Electron density", geometry)
    add("grid", "Voxel grid", dict(kind="rear-grid", extent=ns["HALF_EXTENT"],
        spacing=ns["GRID_SPACING_A"], color="#777C84", fillColor="#F0F1F3",
        borderColor="#62666D", width=ns["GRID_LINE_WIDTH"], borderWidth=ns["GRID_LINE_WIDTH"] * 2, opacity=1))
    result = dict(sample=ns["SAMPLE_ID"], view=view, layers=layers, extent_angstrom=24 if expanded else 16)
    if view in ("overview", "masked"):
        result["pocket_pdb"] = centered_structure(ns["pocket_path"], ns["ligand_center"], True)
        result["ligand_sdf"] = centered_structure(ns["ligand_path"], ns["ligand_center"])
    if view == "masked":
        result["atom_mask_boxes"] = mask_boxes(ns, ns["spatial_mask"])
    if view == "overview":
        result["protein_pdb"] = centered_protein(scene)
    return json.dumps(result).encode()

"""Reuse the overview notebook's rendering code without a Jupyter kernel.

Only the setup and three figure-building sections are evaluated. Widget code
and export cells are excluded; the notebook remains the source of geometry,
colors, voxel alignment, and masking defaults.
"""

import ast
import json
from pathlib import Path

import plotly.io as pio
import numpy as np
from skimage.measure import find_contours

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_SAMPLE = "2p0d"
# Chosen from complete local pairs by axis-aligned ligand visibility, projected
# spread, pocket size, and density coverage; see overview_candidates.json.
SAMPLES = {
    "2p0d": {"label": "2P0D · compact, open pocket", "axis": 0, "sign": -1},
    "4x11": {"label": "4X11 · vertical pocket", "axis": 1, "sign": -1},
    "2dxs": {"label": "2DXS · broad ligand silhouette", "axis": 2, "sign": -1},
    "6ay5": {"label": "6AY5 · original sample", "axis": 2, "sign": 1},
}
PALETTE = {
    "ligand": "#FFB377", "pocket": "#BEC2C8", "density": "#4761F2",
    "ligand_occupancy": "#FFB377", "pocket_occupancy": "#DEE0E3",
    "gradient": "#8092F6", "grid": "#737987",
}
# Qualitative viewer's pastel elements, with orange ligand carbon/occupancy.
LIGAND_ELEMENT_COLORS = {
    "C": "#FFB377", "H": "#F8F8F6", "N": "#8FA9D7", "O": "#E59A9A",
    "F": "#ADCBA6", "P": "#E6B485", "S": "#E3D382", "CL": "#A3C79E",
    "BR": "#CAA28A", "I": "#B8A0D1",
}
DEFAULT_ASPECT = dict(x=1.25, y=1.25, z=1.25)
NGL_CAMERAS = {
    "2p0d": {
        "rotation": [0.0008164604594059425, -0.1458655588337554,
                     -0.6932352125843333, 0.7057956589350693],
        "position": [-1.497001572625262, -1.4907252683723589, 1.0207102414613276],
        "distance": 29.99510411104215,
        "camera_type": "perspective", "fov_degrees": 40,
    },
}


def cube_camera(sample_id):
    if sample_id in NGL_CAMERAS:
        # Inverse NGL rotation applied to +Z (eye) and +Y (up). Also keep
        # the masking face selection consistent with the saved NGL viewpoint.
        x, y, z, w = NGL_CAMERAS[sample_id]["rotation"]
        eye = [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]
        up = [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)]
        return dict(eye=dict(zip("xyz", eye)), up=dict(zip("xyz", up)),
                    center=dict(x=0, y=0, z=0), projection=dict(type="perspective"))
    sample = SAMPLES[sample_id]
    eye = dict(x=0.0, y=0.0, z=0.0)
    eye["xyz"[sample["axis"]]] = sample["sign"] * 2.15
    vertical = 1 if sample["axis"] == 2 else 2
    horizontal = next(axis for axis in range(3) if axis not in (sample["axis"], vertical))
    # About 30° to the side and 22° above each sample's selected face:
    # three cube faces are visible without returning to a steep diagonal.
    eye["xyz"[horizontal]] = -1.25
    eye["xyz"[vertical]] = 1.0
    up = dict(x=0, y=1 if sample["axis"] == 2 else 0,
              z=0 if sample["axis"] == 2 else 1)
    return dict(eye=eye, up=up, center=dict(x=0, y=0, z=0), projection=dict(type="orthographic"))


def display_contours(volume, threshold, centers, stride, min_points):
    """Interpolate slice planes so visual spacing need not be whole voxels."""
    coordinates = [[], [], []]
    for axis in range(3):
        free = [other for other in range(3) if other != axis]
        for fixed in np.arange(0, volume.shape[axis] - 1 + 1e-6, stride):
            low = int(fixed)
            fraction = fixed - low
            plane = np.take(volume, low, axis=axis)
            if fraction:
                plane = plane * (1 - fraction) + np.take(volume, low + 1, axis=axis) * fraction
            for contour in find_contours(plane, level=float(threshold)):
                if len(contour) < min_points:
                    continue
                xyz = np.empty((len(contour), 3), dtype=np.float32)
                xyz[:, axis] = np.interp(fixed, np.arange(len(centers)), centers)
                for column, target in enumerate(free):
                    xyz[:, target] = np.interp(contour[:, column], np.arange(len(centers)), centers)
                for target in range(3):
                    coordinates[target].extend([*xyz[:, target], None])
    return tuple(coordinates)


def load_scene(sample_id=DEFAULT_SAMPLE):
    if sample_id not in SAMPLES:
        raise ValueError("Unknown sample.")
    data_root = REPO_ROOT / "dataset/data/pdbbind"
    required = [data_root / f"voxels_ligvdw/atoms/{sample_id}.npy",
                data_root / f"voxels_v5/density/{sample_id}.npy",
                data_root / f"structures/misato_qm_built/{sample_id}/{sample_id}_ligand.sdf",
                data_root / f"structures/misato_qm_built/{sample_id}/{sample_id}_pocket.pdb"]
    if not all(path.is_file() for path in required):
        raise ValueError(f"Incomplete local sample: {sample_id}")
    camera = cube_camera(sample_id)
    overrides = {
        "SAMPLE_ID": sample_id, "DEFAULT_CAMERA": camera,
        "default_eye_direction": tuple(camera["eye"].values()),
        "LIGAND_COLOR": PALETTE["ligand"], "POCKET_COLOR": PALETTE["pocket"],
        "LIGAND_OCCUPANCY_COLOR": PALETTE["ligand_occupancy"],
        "POCKET_OCCUPANCY_COLOR": PALETTE["pocket_occupancy"],
        "DENSITY_COLOR": PALETTE["density"], "DENSITY_GRADIENT_COLOR": PALETTE["gradient"],
        "GRID_COLOR": PALETTE["grid"],
        "GRID_LINE_WIDTH": 1.44,
        "DENSITY_CONTOUR_WIDTH": 0.6912,  # 0.6 × the previous 1.152
        "DENSITY_CONTOUR_STRIDE": 3.75,  # 6 / 1.6; interpolated voxel planes
        "MASK_FILL_COLOR": "#EBEEEE", "MASK_FILL_OPACITY": 0.90,
        "MASK_LINE_COLOR": "#000000",
        "MASK_LINE_WIDTH": 2.88,  # 0.6 × the previous 4.8
        "PRIMARY_DENSITY_OPACITY": 0.6, "MASKED_DENSITY_OPACITY": 0.6,
    }
    notebook = json.loads((HERE / "fig-overview.ipynb").read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]
               if cell["cell_type"] == "code"]
    sections = [
        ("from pathlib import Path", None),
        ("atom_voxels =", None),
        ("def block_reduce(", None),
        ('LIGAND_COLOR =', "# Keep controls outside"),
        ('DENSITY_GRADIENT_COLOR =', "gradient_visualizer ="),
        ('MASK_RATIO =', "masked_visualizer ="),
    ]
    namespace = {"__name__": "overview_render", "display": lambda *args: None,
                 "_display_contours": display_contours}
    for prefix, stop in sections:
        matches = [source for source in sources if source.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one overview rendering section: {prefix}")
        source = matches[0]
        if stop:
            if stop not in source:
                raise ValueError(f"Missing widget boundary: {stop}")
            source = source.split(stop, 1)[0]
        tree = ast.parse(source)
        # No widget or IPython dependency; use an absolute data root, so the
        # standalone launcher also works from outside the repository.
        clean = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "contour_line_coordinates":
                node.body = ast.parse("return _display_contours(volume, threshold, centers, stride, DENSITY_MIN_CONTOUR_POINTS)").body
            if isinstance(node, ast.Import) and any(a.name == "ipywidgets" for a in node.names):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "IPython.display":
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id in ("display", "print"):
                    continue
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REPO_ROOT" for t in node.targets):
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                      args=[ast.Constant(str(REPO_ROOT))], keywords=[])
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in overrides:
                    node.value = ast.parse(repr(overrides[key]), mode="eval").body
            clean.append(node)
        tree.body = clean
        exec(compile(ast.fix_missing_locations(tree), str(HERE / "fig-overview.ipynb"), "exec"), namespace)
    return OverviewScene(namespace)


class OverviewScene:
    def __init__(self, namespace):
        self.ns = namespace
        self.figures = {"overview": namespace["fig"], "gradient": namespace["gradient_fig"],
                        "masked": namespace["masked_fig"]}
        self._color_ligand_atoms()
        for figure in self.figures.values():
            # Plotly implements orthographic scroll zoom via aspectratio.
            # 'cube' recomputes it on restyle, discarding that zoom.
            figure.update_layout(scene_aspectmode="manual", scene_aspectratio=DEFAULT_ASPECT)

    def _color_ligand_atoms(self):
        """Color atoms and their half-bonds without changing the ligand pose."""
        ns = self.ns
        coords, elements, bonds = (ns[key] for key in ("lig_coords_view", "lig_elements_view", "lig_bonds_view"))
        atom_colors = [LIGAND_ELEMENT_COLORS.get(str(e).upper(), LIGAND_ELEMENT_COLORS["C"])
                       for e in elements]
        vertices, faces, colors = [], [], []
        for point, color in zip(coords, atom_colors):
            ns["append_uv_sphere"](vertices, faces, colors, point, ns["BALL_RADIUS_A"], color)
        spheres = ns["mesh_trace"](vertices, faces, colors, "Ligand atoms", "ligand-model")
        vertices, faces, colors = [], [], []
        for start, end in bonds:
            a, b = coords[start], coords[end]
            ca, cb = atom_colors[start], atom_colors[end]
            # Open-ended halves meet exactly at the midpoint: no overlapping
            # caps, no color interpolation along heteroatom bonds.
            parts = [(a, b, ca)] if ca == cb else [(a, (a + b) / 2, ca), ((a + b) / 2, b, cb)]
            for begin, finish, color in parts:
                ns["append_cylinder"](vertices, faces, colors, begin, finish, ns["STICK_RADIUS_A"], color)
        sticks = ns["mesh_trace"](vertices, faces, colors, "Ligand bonds", "ligand-model")
        self.figures["overview"].data[6].update(sticks.to_plotly_json())
        self.figures["overview"].data[7].update(spheres.to_plotly_json())

    def config(self):
        return {"sample": self.ns["SAMPLE_ID"], "density_max": self.ns["density_ceiling"],
                "camera": self.ns["DEFAULT_CAMERA"], "output_dir": "figures/fig-overview/exports/pre-training",
                "ngl_camera": NGL_CAMERAS.get(self.ns["SAMPLE_ID"]),
                "palette": PALETTE,
                "ligand_elements": {element: LIGAND_ELEMENT_COLORS.get(element, LIGAND_ELEMENT_COLORS["C"])
                                    for element in sorted({str(e).upper() for e in self.ns["lig_elements_view"]})},
                "aspectratio": DEFAULT_ASPECT,
                "samples": [{"id": key, **value} for key, value in SAMPLES.items()]}

    def figure_json(self, view):
        return pio.to_json(self.figures[view], pretty=False).encode()

    def density_json(self, view, iso):
        volume = self.ns["primary_density_view" if view == "overview" else "masked_density_view"]
        trace = self.ns["contour_trace"](volume, iso, self.ns["DENSITY_COLOR"], "Density", "density")
        # Serialize through Plotly to retain its typed-array encoding.
        return pio.to_json({"data": [trace], "layout": {}}, pretty=False).encode()

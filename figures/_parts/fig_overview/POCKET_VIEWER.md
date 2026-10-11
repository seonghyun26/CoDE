# Downstream pocket visualizer

Run from the repository root:

```bash
bash figures/_parts/fig_overview/run_pocket.sh
```

Open <http://localhost:8768>. Use `PORT=8769` to choose another port, or
`VOXBIND_PYTHON=/path/to/python` to select a Python with NumPy, RDKit, Pillow and ReportLab.
The server binds to localhost by default. NGL is served from the repository;
the browser does not need a CDN or external structure service.

The two panels use actual local results:

- Pocket-conditioned ligands: `results/task2-drugdesign/VoxBind-Ours/samples`.
- Macrocyclic peptides: `results/task3-mcp/Ours-receptorED/samples`, with the
  matching PDBbind `pbpp-2020` reference pocket and ligand.

Select a pocket and a crystal reference or generated sample. Samples are in
source SDF order, not ranked. All molecules and pocket atoms are translated by
the same crystal-reference center; generated poses are never fitted to the
reference. Pocket distance selects complete residues within 5, 7 or 10 Å of the
reference ligand, from the available source pocket (it cannot restore atoms
outside that source). Cartoon uses only the backbone available in this crop.

Drag to rotate, scroll to zoom, and right-drag to pan. Toggle surface, sticks,
cartoon or molecule; adjust colors, surface opacity and molecule style.
`Pocket only` and `Pocket + molecule` preserve the camera for paired input/output
figures. `Remember view` stores the camera per target in this browser;
`Reset view` restores an automatically chosen view through an open side.

`Save PNG + PDF` writes a square 1200/1800/2400 px image into
`figures/fig-overview/exports/downstream/` and a matching JSON sidecar containing
the camera, settings and source paths. A matching PDF embeds the exact captured
image at 300 dpi with its alpha channel preserved (raster, not vector geometry).
Page titles and controls are excluded
from the capture. Existing exports are retained with unique filenames.

This viewer illustrates existing generated structures; it does not run a
generation model or assert that every generated MCP sample is cyclic.

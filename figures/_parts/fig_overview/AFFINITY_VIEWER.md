# Binding affinity example selector

```bash
bash figures/_parts/fig_overview/run_affinity.sh
```

Open <http://localhost:8769>. The downstream viewer on port 8768 remains separate.
Set `PORT` or `VOXBIND_PYTHON` to override the port or Python interpreter.

The catalog uses exactly the prediction rows in
`base/_casf/CDG_v2_casf2016_preds_seed0.csv`, the source of the overview scatter:
`y` is Ground Truth pK and `pred` is CDG v2 seed-0 predicted pK. Larger pK means
stronger binding. It is not a binding energy in kcal/mol. Original Kd/Ki labels,
protein class and resolution are shown from `pdbbind/index.csv`; a measurement
label is omitted if the index pK disagrees with the plotted GT.

Structures come from local `pbpp-2020` crystal complexes. Protein and ligand
are translated by the same ligand center; the view is not a predicted pose.
Ligand coordinates and deposited bond types are loaded without chemical
sanitization, so aromatic/metal structures remain available for visualization.
Complete receptor residues within 5/7/10 Å of the crystal ligand are shown.
The full receptor is used when available, otherwise the existing pocket crop.

- Sort by GT, prediction, absolute error or ligand atom count; filter by PDB,
  protein class, GT range or shortlist.
- Click A/B to load a row into the respective comparison panel. Initial panels
  use the highest and lowest available GT; these may be different proteins.
- Star candidates to persist a shortlist in this browser.
- Each panel has an independent palette using `figures/fig-palette/palette-methods.csv`.
  Swatches use the current `color_hex` (Color) column; hatch-only entries are
  omitted because a hatch is not a molecular color. Palette choices track the
  method name when the palette changes; custom colors are retained.
  The chosen color sets ligand carbon atoms (heteroatoms retain element colors),
  and the matching scatter point. Turn off element coloring for a uniform ligand.
  Colors and selected cases are remembered separately for the left/right panel.
- `Update scatter plot` writes `binding_affinity_scatter_selected.png`, `.pdf`,
  `.svg` and a provenance `.json` alongside this viewer. The selected points use
  their exact CSV coordinates and palette colors. The original scatter is kept;
  subsequent updates replace only the `_selected` files. Choose two distinct IDs.
- The bottom-left scatter is a live SVG preview: case and color changes redraw
  immediately without a save request. It always shows all cases, with larger,
  black-outlined selected points. The button explicitly saves the current plot.
- `Save selected pair` writes the selected records, sources, cameras and
  shortlist to JSON in `exports/affinity/`.
- Each panel's `Save PNG + PDF` button saves a transparent PNG, a PDF of the same
  raster capture with alpha preserved at 300 dpi, and a matching
  camera/settings/affinity JSON sidecar. Existing exports are retained.

White 15%-opaque surface, pocket sticks and cartoon are enabled by default.
Ligands use sticks colored by element. View controls are shared with the
downstream pocket viewer.

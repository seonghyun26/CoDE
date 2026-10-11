# Pre-training overview visualizer

Run from the repository root:

```bash
bash figures/_parts/fig_overview/run_overview.sh
```

Open <http://localhost:8767> (`PORT=...`, `HOST=...` to change). No Jupyter server is needed: the
server runs the setup and rendering cells of `fig-overview.ipynb` (widget and export cells
excluded), so the notebook stays the single source of the voxel geometry, and draws it with the
locally served NGL in `assets/`.

- Views: Overview (ligand and pocket sticks, occupancy, density) and Density gradient, sharing the
  camera; Masking is an overlay on Overview. Grid ON crops everything to the 16 Å cube, Grid OFF
  shows the surrounding 24 Å.
- Samples: 2P0D by default, plus 4X11, 2DXS and 6AY5; the selection notes are in
  `overview_candidates.json` (not in git).
- Captures: 1520 × 1520 PNG and/or PDF with transparent background, saved to
  `figures/fig-overview/exports/pre-training/`.
- Test: `test_overview_browser.cjs` drives a running server with Playwright (set `OVERVIEW_URL`,
  `PLAYWRIGHT_CHROMIUM_EXECUTABLE` as needed).

# Qualitative comparison viewer and figures

This directory contains the qualitative viewer, its notebook, the capture
generator and the browser assets.

Run from the repository root:

```bash
bash figures/_parts/fig_qual/run_fig_qual.sh
```

The default address is http://localhost:8766. `HOST`, `PORT`, and
`VOXBIND_PYTHON` can override the launcher defaults.

- `fig_qual_server.py`, `assets/fig_qual_app.js`: server and NGL browser viewer.
- `qual_setup.py`, `qual_utils.py`, `prepare_qual_baselines.py`: data/style setup.
- `qual_capture.py`: comparison-sheet PDF generation.
- `../palette/palette-methods.csv`: shared method colors for PDF title chips.
- `fig-qual.ipynb`: interactive notebook; exports to `exports/`.
- `fig_qual_cameras.json`: saved default cameras.
- `exports/<pocket>/`: method PNG/PDF pairs, comparison PDFs and camera/energy JSON.
- `exports/`: shared reference/preview PDFs and pocket-selection reports.
- `archive/pre-migration-20260924/`: older code that was already in this directory,
  preserved before consolidating the current viewer here.

Original filenames and result bytes were preserved during migration.
The overview figure's notebook (`../fig_overview/fig-overview.ipynb`) supplies
the shared style; model and sample datasets are resolved from the repository root.

Existing ring-nonflatness tools and publication PDF variants remain here.

Capture checks:

```bash
conda run -n voxbind python -m unittest discover \
  -s figures/_parts/fig_qual -p test_qual_capture.py
```

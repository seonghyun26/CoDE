"""fig-qual: the qualitative 3D comparison sheets (one per pocket, nine methods each).

The 3D panels are posed and captured by hand in the viewer -- no script chooses a camera:

    python figures/draw.py --viewer qual        # or: bash figures/_parts/fig_qual/run_fig_qual.sh
    python figures/draw.py --viewer ring        # the ring non-flatness explorer

"Capture all" in the viewer saves every panel (PNG + PDF), the metadata JSON and the
comparison sheet to figures/fig-qual/exports/<POCKET>/. The sheet can be laid out again
from those saved captures without a browser:

    python figures/draw.py fig-qual-comparisons     # also run by: draw_all.sh --with-3d

The code lives here; everything it writes stays in figures/fig-qual/ (exports/, the
camera presets in fig_qual_cameras.json, the drug-qual-example-*.pdf sheets).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from draw import HERE, figure, viewer

CODE = Path(__file__).resolve().parent
OUT = HERE / "fig-qual"


@figure("fig-qual-comparisons", folder="fig-qual", capture=True)
def draw_qual_comparisons(out):
    """Every pocket's comparison sheet (exports/<POCKET>/drug-qual-<POCKET>.pdf), laid out
    again from its saved captures and metadata."""
    from .qual_capture import comparison_pdf
    n = 0
    for meta_path in sorted((OUT / "exports").glob("*/Pocket*.json")):
        meta = json.loads(meta_path.read_text())
        files = {}
        for record in meta["methods"].values():
            name = (record.get("files") or {}).get("png")
            if name and (meta_path.parent / name).is_file():
                files[name] = (meta_path.parent / name).read_bytes()
        (meta_path.parent / f"drug-qual-{meta['pocket_code']}.pdf").write_bytes(comparison_pdf(files, meta))
        n += 1
    print(f"  {n} pocket sheet(s) laid out from saved captures")


def _serve(script, host, port, default_port):
    env = {**os.environ, "HOST": host, "PORT": str(port or default_port)}
    raise SystemExit(subprocess.call(["bash", str(CODE / script)], env=env))


@viewer("qual")
def view_qual(host, port):
    """The qualitative comparison viewer (captures to figures/fig-qual/exports/)."""
    _serve("run_fig_qual.sh", host, port, 8766)


@viewer("ring")
def view_ring(host, port):
    """The ring non-flatness explorer (builds its site once, then serves it)."""
    _serve("run_ring_nonflatness.sh", host, port, 8768)

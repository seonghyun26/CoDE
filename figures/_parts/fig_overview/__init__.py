"""fig-overview: the overview figure's panels.

    python figures/draw.py overview

    fig-overview-binding-affinity            -> fig-overview/binding_affinity_scatter.*
    fig-overview-binding-affinity-selected   -> fig-overview/binding_affinity_scatter_selected.*
    fig-overview-paper                       -> fig-overview/{pretraining_overview, ...}.*

The pocket renders (exports/pre-training/, exports/downstream/) are captured by hand in
the 3D viewers, and the two highlighted complexes of the selected scatter are picked in
the affinity viewer:

    python figures/draw.py --viewer pocket       # or: bash figures/_parts/fig_overview/run_pocket.sh
    python figures/draw.py --viewer affinity

The code lives here; everything it writes stays in figures/fig-overview/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from draw import HERE, figure, viewer

CODE = Path(__file__).resolve().parent
OUT = HERE / "fig-overview"


@figure("fig-overview-binding-affinity", folder="fig-overview", isolate=True)
def draw_overview_binding_affinity(out):
    """CASF-2016 predicted against experimental affinity (CDG v2, seed 0): the small panel."""
    from .binding_affinity import render_plot
    render_plot()


@figure("fig-overview-binding-affinity-selected", folder="fig-overview", isolate=True)
def draw_overview_binding_affinity_selected(out):
    """The same panel with the two complexes picked in the affinity viewer highlighted."""
    from .binding_affinity import render_plot
    picked = json.loads((OUT / "binding_affinity_scatter_selected.json").read_text())["highlights"]
    render_plot([dict(pid=h["pid"], color=h["color"]) for h in picked])


@figure("fig-overview-paper", folder="fig-overview", isolate=True)
def draw_overview_paper(out):
    """The paper's schematic overviews: pre-training, the VoxBind adapter, the single-CDG
    encoder and its downstream tasks."""
    # a script of its own: it sets matplotlib's rcParams at import
    code = subprocess.call([sys.executable, str(CODE / "paper_overviews.py")],
                           env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/voxbind-paper-mpl")})
    if code:
        raise RuntimeError(f"paper_overviews.py exited {code}")


def _serve(script, host, port, default_port):
    env = {**os.environ, "HOST": host, "PORT": str(port or default_port)}
    raise SystemExit(subprocess.call(["bash", str(CODE / script)], env=env))


@viewer("pocket")
def view_pocket(host, port):
    """The pocket render viewer (captures to figures/fig-overview/exports/downstream/)."""
    _serve("run_pocket.sh", host, port, 8768)


@viewer("affinity")
def view_affinity(host, port):
    """The affinity viewer: pick the two highlighted complexes, capture their pockets."""
    _serve("run_affinity.sh", host, port, 8769)

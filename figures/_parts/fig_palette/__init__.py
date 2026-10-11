"""fig-palette: the method palette -- the official swatch sheet and the proposals.

    python figures/draw.py palette

    fig-palette-methods             -> fig-palette/palette-methods.*          (methods.py)
    fig-palette-options             -> fig-palette/proposals/method-palette-options.*  (options.py)
    fig-palette-candidate           -> fig-palette/palette-method-candidate.*  (candidate.py)
    fig-palette-eight-colors        -> fig-palette/palette-method-candidate-eight-colors.*
    fig-palette-green-orange-blue   -> fig-palette/palette-method-candidate-green-orange-blue.*

The proposals are previews only: none of them changes the live palette, which is
fig-palette/method-palette-approved.json, read by draw.py before any figure is drawn.
"""
from __future__ import annotations

import importlib

from draw import figure

from . import methods  # noqa: F401  (registers fig-palette-methods)


def _main(name):
    importlib.import_module(f"{__name__}.{name}").main()


@figure("fig-palette-options", folder="fig-palette/proposals", isolate=True)
def draw_palette_options(out):
    """Lighter, less saturated variants of the method colors, side by side (preview only)."""
    _main("options")


@figure("fig-palette-candidate", folder="fig-palette", isolate=True)
def draw_palette_candidate(out):
    """A candidate Color / Highlight palette (preview only)."""
    _main("candidate")


@figure("fig-palette-eight-colors", folder="fig-palette", isolate=True)
def draw_palette_eight_colors(out):
    """The eight user-specified colors at full opacity (preview only)."""
    _main("eight_colors")


@figure("fig-palette-green-orange-blue", folder="fig-palette", isolate=True)
def draw_palette_green_orange_blue(out):
    """Three color families from light to dark (preview only)."""
    _main("green_orange_blue")

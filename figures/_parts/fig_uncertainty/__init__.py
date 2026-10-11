"""fig-uncertainty: does the density gain track how well ordered the crystal is?

Everything drawn into figures/fig-uncertainty/. Two kinds of code live here:

  docking.py        the seven registry figures (docking/, exploratory/affinity-initial/)
  the analyses      motivation, manuscript, atomic B, three-model, CL3 / density
                    exploratory and the contact-B bin comparisons -- each a module with its
                    own main() that fits, tabulates and plots.

The figures registered below call those modules in their plots-only mode, reading the
saved tables, so `draw.py uncertainty` redraws every uncertainty figure without fitting
anything. With --figures-only (what draw_all.sh passes) the saved tables, notes and
provenance are left untouched as well. To RECOMPUTE, run a module itself, from figures/:

    cd figures && python -m _parts.fig_uncertainty.manuscript
    cd figures && python -m _parts.fig_uncertainty.update_expanded       # the expanded cohort

The modules are imported only when their figure is drawn: several pull in torch.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

import draw
from draw import figure

from . import docking  # noqa: F401  (registers the seven registry figures)


@contextlib.contextmanager
def _plots_only():
    """With --figures-only, a redraw must not rewrite the cached numbers: DataFrame.to_csv
    and Path.write_text become no-ops for tables, notes and JSON under fig-uncertainty/."""
    if not draw.FIGURES_ONLY:
        yield
        return
    import pandas as pd
    base = (draw.HERE / "fig-uncertainty").resolve()
    to_csv, write_text = pd.DataFrame.to_csv, Path.write_text

    def readonly_csv(frame, path_or_buf=None, *a, **kw):
        if path_or_buf is None:
            return to_csv(frame, path_or_buf, *a, **kw)
        return None

    def readonly_text(path, data, *a, **kw):
        if Path(path).resolve().is_relative_to(base) and Path(path).suffix in (".json", ".md", ".csv"):
            return len(data)
        return write_text(path, data, *a, **kw)

    pd.DataFrame.to_csv, Path.write_text = readonly_csv, readonly_text
    try:
        yield
    finally:
        pd.DataFrame.to_csv, Path.write_text = to_csv, write_text


def _module(name):
    return importlib.import_module(f"{__name__}.{name}")


def _main(name, argv=()):
    """Run a module's main() as if from the command line."""
    mod = _module(name)
    saved = sys.argv
    sys.argv = [f"{name}.py", *argv]
    try:
        with _plots_only():
            mod.main()
    finally:
        sys.argv = saved


@figure("fig-uncertainty-motivation", folder="fig-uncertainty/motivation", isolate=True)
def draw_uncertainty_motivation(out):
    """Motivation set: decoding, ranking and supplementary panels from saved statistics."""
    _main("motivation", ["--supplementary"])


@figure("fig-uncertainty-manuscript", folder="fig-uncertainty/affinity/cl3-5seed", isolate=True)
def draw_uncertainty_manuscript(out):
    """Manuscript CDG v2 / CoE five-seed analysis: B-factor bins, concat ablation, decoding."""
    _main("manuscript", ["--plots-only"])


@figure("fig-uncertainty-atomic-b", folder="fig-uncertainty/controls/atomic-b-spatial", isolate=True)
def draw_uncertainty_atomic_b(out):
    """Atomic B-factor spatial control (paired contrasts per complex and seed)."""
    _main("atomic_bfactor", ["--plots-only"])


@figure("fig-uncertainty-three-model", folder="fig-uncertainty/controls/atomic-b-spatial", isolate=True)
def draw_uncertainty_three_model(out):
    """Three-model uncertainty comparison over the atomic-B cohort."""
    _main("three_model")


@figure("fig-uncertainty-cl3-exploratory", folder="fig-uncertainty/exploratory/cl3-filtered", isolate=True)
def draw_uncertainty_cl3_exploratory(out):
    """Exploratory CL3-filtered descriptor scatter and tercile panels, from saved tables."""
    with _plots_only():
        _module("affinity_cl3").redraw_saved()


@figure("fig-uncertainty-density-exploratory",
        folder="fig-uncertainty/exploratory/cl3-filtered/density-uncertainty", isolate=True)
def draw_uncertainty_density_exploratory(out):
    """Exploratory map-uncertainty bins and embedding decoding, from saved tables."""
    with _plots_only():
        _module("affinity_density").redraw_saved_plots()


@figure("fig-uncertainty-bin-comparisons",
        folder="fig-uncertainty/exploratory/contact-b-fixed-four-comparison", isolate=True)
def draw_uncertainty_bin_comparisons(out):
    """Contact-B quartile and fixed-four bin comparisons (and the seed-SD error-bar variant)."""
    quartiles, fixed = _module("contact_quartiles"), _module("contact_fixed_four")
    if draw.FIGURES_ONLY:
        # Read the saved ranking tables instead of re-running the analyses behind them.
        import pandas as pd
        quartiles.analyse_quartiles = lambda source, out, n: pd.read_csv(out / "contact-b-quartiles-ranking.csv")
        fixed.analyse = lambda cutoffs, prefix, source, out, n: pd.read_csv(out / f"{prefix}-ranking.csv")
    _main("contact_quartiles")
    _main("contact_fixed_four")
    _main("contact_fixed_four", ["--error-bars", "seed-sd"])

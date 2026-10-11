"""Small overview panel using real CASF-2016 predictions (CDG v2, seed 0).

Run from any directory: python /path/to/plot_binding_affinity.py
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[2] / "fig-overview"   # figures/fig-overview
SOURCE = OUT.parents[1] / "base/_casf/CDG_v2_casf2016_preds_seed0.csv"


def render_plot(selected=None, output_dir=OUT):
    data = np.genfromtxt(SOURCE, delimiter=",", names=True, dtype=None, encoding="utf-8")
    highlights = []
    for item in selected or []:
        matches = np.flatnonzero(data["pid"] == item["pid"])
        if len(matches) != 1 or not re.fullmatch(r"#[0-9a-fA-F]{6}", item["color"]):
            raise ValueError("Unknown PDB ID or invalid highlight color")
        if any(h["pid"] == item["pid"] for h in highlights):
            raise ValueError("Choose different PDB IDs for the two panels")
        i = matches[0]
        highlights.append(dict(pid=item["pid"], color=item["color"].upper(),
                               predicted=float(data["pred"][i]), experimental=float(data["y"][i])))
    predicted, gt = data["pred"], data["y"]
    if not (np.isfinite(predicted).all() and np.isfinite(gt).all()):
        raise ValueError("Prediction file contains nonfinite values")
    lo = np.floor(min(predicted.min(), gt.min()))
    hi = np.ceil(max(predicted.max(), gt.max()))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 17.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 2.0,
        "axes.edgecolor": "black",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    # Fixed square canvas and identical width/height keep export geometry exact.
    fig = plt.figure(figsize=(3.2, 3.2), dpi=300)
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.78])
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    line_x = np.array([lo, hi])
    ax.plot(line_x, line_x, color="black", alpha=0.25, lw=2.6, zorder=1)
    ordinary = ~np.isin(data["pid"], [item["pid"] for item in highlights])
    ax.scatter(predicted[ordinary], gt[ordinary], s=24 * 1.4, color="#3C5FDF", alpha=0.45,
               edgecolors="none", linewidths=0, zorder=2)
    for item in highlights:
        ax.scatter([item["predicted"]], [item["experimental"]], s=40 * 1.4 * 2,
                   color=item["color"], alpha=1, edgecolors="black",
                   linewidths=ax.spines["bottom"].get_linewidth() * 0.6, zorder=3)
    ax.set(xlabel="Predicted Affinity", ylabel="Experimental Affinity", xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    ax.set_box_aspect(1)
    ticks = np.linspace(lo, hi, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(length=3, width=1.6, labelbottom=False, labelleft=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "binding_affinity_scatter_selected" if selected is not None else "binding_affinity_scatter"
    for ext in ("png", "pdf"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, transparent=True)
        print(path)
    fig.canvas.draw()
    bounds = ax.get_window_extent()
    assert np.isclose(bounds.width, bounds.height), "Axes must have equal physical lengths"
    print(f"Axis lengths at 300 dpi: x={bounds.width:.2f}px, y={bounds.height:.2f}px")
    plt.close(fig)
    return dict(source=str(SOURCE.relative_to(OUT.parents[1])), n=len(gt), highlights=highlights,
                x_label="Predicted Affinity", y_label="Experimental Affinity")


if __name__ == "__main__":
    render_plot()

"""fig-mcp: the 1 figure drawn into figures/fig-mcp/.

    python figures/draw.py fig-mcp-finetune-size

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-mcp-finetune-size                        -> fig-mcp/
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from draw import (
    MISC_PLAIN_RC, SIBLING, _misc_savefig, figure, pale, write_csv,
)

FUNCBIND = Path(os.environ.get("VOXBIND_FUNCBIND", SIBLING / "funcbind"))


# ════════════════════════════════════════════════════════════════════════════════
# fig-mcp-finetune-size — does more MCP fine-tuning buy binding, or just size?
# ════════════════════════════════════════════════════════════════════════════════
# THE ARGUMENT. Vina rewards size, so an arm that draws smaller peptides is penalised and
# an arm that draws larger ones is flattered, regardless of whether it binds better. Binning
# by heavy-atom count removes that: inside a bin every arm draws the same-sized peptides, so
# a gap that survives the bin is a binding difference. This is §4 of the 260827 note applied
# to the MCP family, and it is the reason the pooled numbers in the tables cannot be read on
# their own.
#
# Colours come from the shared palette, where the four arms are registered as an ordinal
# ramp -- they are one model at four amounts of fine-tuning, not four methods.
MCP_FB = str(FUNCBIND / "artifacts" / "reproduction" / "mcpp")
# Four steps of one brown ramp separate cleanly in the bars, but the top panel's lines
# cross, and at a crossing two adjacent steps of a lightness ladder are genuinely hard to
# tell apart. So each arm carries a second identity channel -- line style -- the same way
# this note's published baselines do. The two solid lines are the lightest and the darkest
# step, which is the pair that never needed the help.
MCP_ARMS = [
    ("vanilla",        "FuncBind vanilla",  f"{MCP_FB}/cmp10/_eval/vanilla/eval_docking_results.json",      "-"),
    ("fine-tune 3.17M", "ft_3.17M",         f"{MCP_FB}/cmp10/_eval/finetuned/eval_docking_results.json",    (0, (1, 1.6))),
    ("fine-tune 8.21M", "ft_8.21M",         f"{MCP_FB}/cmp10_r10/_eval/finetuned/eval_docking_results.json", (0, (5, 2.2))),
    ("fine-tune 26.1M", "ft_26.1M",         f"{MCP_FB}/cmp10_r14/_eval/finetuned/eval_docking_results.json", "-"),
]
MCP_BINS = [(20, 29), (30, 39), (40, 49), (50, 59), (60, 89)]
MCP_LBL = ["20–29", "30–39", "40–49", "50–59", "60+"]
MCP_MIN_N = 5      # fewer molecules than this in a bin is a data point, not a trend


@figure("fig-mcp-finetune-size", needs=("mcpp/*/_eval/*/eval_docking_results.json",))
def draw_mcp_finetune_size(out):
    """Median Vina Dock per heavy-atom bin (top) over the molecule count each arm puts in
    that bin (bottom)."""
    d = {lab: json.load(open(p))["per_target"] for lab, _, p, _ in MCP_ARMS}
    # Only targets every arm produced molecules for: comparing an arm against a target where
    # another arm returned nothing is not a comparison.
    common = sorted(set.intersection(*[
        {e["target"] for e in per if e.get("vina_dock") is not None} for per in d.values()]))

    def mols(lab):
        return [(m["n_atoms"], m["vina_dock"])
                for e in d[lab] if e["target"] in common
                for m in (e.get("per_mol") or [])
                if m.get("vina_dock") is not None and m.get("n_atoms")]

    m = {lab: mols(lab) for lab, _, _, _ in MCP_ARMS}
    binned = {lab: [[v for a, v in vals if lo <= a <= hi] for lo, hi in MCP_BINS]
              for lab, vals in m.items()}

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.4, 7.0), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.4, 1], hspace=.13))
        x = np.arange(len(MCP_BINS))

        for lab, key, _, style in MCP_ARMS:
            col = pale(key)
            ys = [np.median(b) if len(b) >= MCP_MIN_N else np.nan for b in binned[lab]]
            ax.plot(x, ys, color=col, lw=2.0, ls=style, marker="o", ms=8.5, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.6, label=lab)
            # An arm that runs out of molecules early gets its label above the point; to the
            # right, the arms that continue would run through it.
            last = max(i for i, y in enumerate(ys) if not np.isnan(y))
            kw = dict(xytext=(10, -2), ha="left") if last == len(MCP_BINS) - 1 \
                else dict(xytext=(0, 23), ha="center")
            ax.annotate(lab, (last, ys[last]), textcoords="offset points",
                        fontsize=10.2, color=col, fontweight="600", va="center", **kw)

        ax.set_ylabel("Vina Dock, median\n(kcal/mol)", fontsize=11.3)
        ax.grid(color="#e6e9ef", lw=.85)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=10.0, ncols=4, loc="lower center",
                  bbox_to_anchor=(.5, 1.01))
        ax.set_xlim(-.45, len(MCP_BINS) - .55)
        ax.margins(y=.20)

        w = .21
        for i, (lab, key, _, _s) in enumerate(MCP_ARMS):
            counts = [len(b) for b in binned[lab]]
            bx.bar(x + (i - 1.5) * w, counts, width=w * .86, color=pale(key), linewidth=0)
            for xi, c in enumerate(counts):
                if c:
                    bx.annotate(str(c), (xi + (i - 1.5) * w, c), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=7.8, color="#7a8699")

        bx.set_ylabel("molecules", fontsize=11.3)
        bx.set_xlabel("heavy atoms", fontsize=11.3)
        bx.set_xticks(x)
        bx.set_xticklabels(MCP_LBL)
        bx.grid(axis="y", color="#e6e9ef", lw=.85)
        bx.set_axisbelow(True)

        for a in (ax, bx):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                a.spines[sp].set_color("#c4cad4")
            a.tick_params(labelsize=10.2, colors="#5b6678")

        _misc_savefig(fig, out, [f"mcp_finetune_size.{ext}" for ext in ("png", "svg")],
                      dpi=170, bbox_inches="tight", facecolor="white")

    write_csv(out, "mcp_finetune_size", ["arm", "bin", "n", "dock_median"],
              [[lab, l, len(b), f"{np.median(b):.3f}" if len(b) >= MCP_MIN_N else ""]
               for lab, _, _, _ in MCP_ARMS for l, b in zip(MCP_LBL, binned[lab])])

    print("  common targets:", len(common), common)
    for lab, _, _, _ in MCP_ARMS:
        ha = [a for a, _ in m[lab]]
        print(f"  {lab:17s} n={len(m[lab]):4d} median heavy={np.median(ha):5.1f} "
              f"bins={[len(b) for b in binned[lab]]} "
              f"medians={[round(float(np.median(b)), 2) if len(b) >= MCP_MIN_N else None for b in binned[lab]]}")

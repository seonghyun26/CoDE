"""fig-ensemble: the 2 figures drawn into figures/fig-ensemble/.

    python figures/draw.py ensemble

Only what this folder's figures use lives here; whatever figures of two or more
folders share is in draw.py and imported below.

    fig-ensemble-6set-3metric                    -> fig-ensemble/
    fig-ensemble-vs-alone                        -> fig-ensemble/
"""
from __future__ import annotations

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from draw import (
    BACKGROUND_ALPHA, DATA_ALPHA, MISC_PLAIN_RC, _misc_savefig, figure, legacy,
)

# ════════════════════════════════════════════════════════════════════════════════
# fig-ensemble-{6set-3metric,vs-alone} — probe-side feature ensembles
# ════════════════════════════════════════════════════════════════════════════════
# Does bolting a frozen baseline encoder onto the champion's features buy anything the
# champion does not already have? `fig1` is six cohorts x three metrics for the champion
# and for champion+partner; `fig2` asks the sharper question -- is the ensemble better than
# the PARTNER alone, or is the partner simply carrying it?
#
# ensemble_results.json is produced by a figure-data builder (not part of this repository),
# which reads feature .pt files that live on the box the probe ran on; it is a recompute
# script, not a drawing script, and it stays there.
#
# These are exploratory panels: bold suptitles, per-panel titles, matplotlib's own tab
# colours. They are deliberately NOT restyled into the house style -- they are not §1/§2
# figures and were never drawn as such.
ENS_COH = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
ENS_COHL = ["FULL", "CL3", "CL3\nID60", "CL3\nID30", "CASF\nnontrain", "CASF\nclean"]
ENS_PARTNERS = ["GET", "EGNN", "CheapNet", "ProFSA"]
ENS_COLORS = {"GET": "#1f77b4", "EGNN": "#2ca02c", "CheapNet": "#ff7f0e", "ProFSA": "#d62728"}
ENS_METRICS = [("r", "Pearson r", False), ("rho", "Spearman ρ", False), ("rmse", "RMSE", True)]


def _ens_results():
    return json.load(open(legacy("fig-ensemble", "ensemble_results.json")))


def _ens_series(d, metric):
    return [d.get(c, {}).get(metric, np.nan) for c in ENS_COH]


@figure("fig-ensemble-6set-3metric", needs=("fig-ensemble/ensemble_results.json",))
def draw_ensemble_6set_3metric(out):
    """Six cohorts x three metrics: champion baseline + champion⊕{GET,EGNN,CheapNet,ProFSA}."""
    r = _ens_results()
    x = np.arange(len(ENS_COH))

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, (mk, ml, lower_better) in zip(axes, ENS_METRICS):
            ax.plot(x, _ens_series(r["champion"], mk), "-o", color="black", lw=2.6, ms=7,
                    label="champion (C+D+G)", zorder=5)
            for p in ENS_PARTNERS:
                ax.plot(x, _ens_series(r["ensembles"][p], mk), "--o", color=ENS_COLORS[p],
                        lw=1.8, ms=5, label=f"⊕ {p}")
            ax.set_xticks(x)
            ax.set_xticklabels(ENS_COHL, fontsize=9)
            ax.set_title(ml + ("  (↓ better)" if lower_better else "  (↑ better)"),
                         fontsize=12, fontweight="bold")
            ax.grid(alpha=BACKGROUND_ALPHA)
            ax.axvspan(3.5, 5.5, color="grey", alpha=BACKGROUND_ALPHA)      # shade CASF holdout region
            if mk == "r":
                ax.legend(fontsize=8.5, loc="lower left", framealpha=DATA_ALPHA)
        fig.suptitle("Probe-side feature ensemble: champion ⊕ baseline encoder (PCA-64), "
                     f"5-seed, n_shared={r['n_shared']}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _misc_savefig(fig, out,
                      [f"fig1_ensemble_6set_3metric.{ext}" for ext in ("png", "pdf")],
                      dpi=150, bbox_inches="tight")
    print("  wrote fig1")

    print("\n  Δρ (ensemble − champion) per cohort:")
    print(f"  {'cohort':<15}" + "".join(f"{p:>10}" for p in ENS_PARTNERS))
    for c in ENS_COH:
        row = f"  {c:<15}"
        for p in ENS_PARTNERS:
            d = r["ensembles"][p].get(c, {}).get("rho", np.nan) \
                - r["champion"].get(c, {}).get("rho", np.nan)
            row += f"{d:>+10.3f}"
        print(row)


@figure("fig-ensemble-vs-alone", needs=("fig-ensemble/ensemble_results.json",))
def draw_ensemble_vs_alone(out):
    """Spearman rho, champion vs partner-ALONE vs ensemble per cohort — does the ensemble
    beat the partner alone?"""
    r = _ens_results()
    x = np.arange(len(ENS_COH))

    with matplotlib.rc_context(rc=MISC_PLAIN_RC):
        fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
        for ax, p in zip(axes2, ENS_PARTNERS):
            w = 0.27
            ax.bar(x - w, _ens_series(r["champion"], "rho"), w, color="black", label="champion")
            ax.bar(x, _ens_series(r["alone"][p], "rho"), w, color=ENS_COLORS[p], alpha=DATA_ALPHA,
                   label=f"{p} alone")
            ax.bar(x + w, _ens_series(r["ensembles"][p], "rho"), w, color=ENS_COLORS[p],
                   label=f"champion ⊕ {p}")
            ax.set_xticks(x)
            ax.set_xticklabels(ENS_COHL, fontsize=8)
            ax.set_title(p, fontsize=12, fontweight="bold")
            ax.grid(alpha=BACKGROUND_ALPHA, axis="y")
            ax.set_ylim(0.40, 0.75)
            ax.legend(fontsize=8, loc="lower left")
        axes2[0].set_ylabel("Spearman ρ", fontsize=11)
        fig2.suptitle("Is the ensemble better than the partner ALONE?  (Spearman ρ per cohort)",
                      fontsize=13, fontweight="bold")
        fig2.tight_layout(rect=[0, 0, 1, 0.94])
        _misc_savefig(fig2, out, [f"fig2_ensemble_vs_alone.{ext}" for ext in ("png", "pdf")],
                      dpi=150, bbox_inches="tight")
    print("  wrote fig2")

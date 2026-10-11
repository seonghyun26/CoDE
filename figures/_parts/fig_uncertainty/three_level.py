#!/usr/bin/env python3
"""B factor decoding at three levels: complex contact B factor, residue and atom.

three-level-decoding   Complex contact B factor (across complexes, MLP on pooled embeddings)
                       + residue/atom B factor (within each complex, token MLP, within-complex z-trained).
atom-level-decoding    Residue/atom x (raw, within-complex z) for the token models only.

Reads only the saved tables in fig-uncertainty/representation/three-level/; the token extraction and
decoders (atom_level_b/) are not in this checkout; their caches are in results/task1-affinity/_shared/260923_atom_level_b/.
Baseline colours are light, desaturated drug-design hues (preview) (not yet in the shared palette).
Run from the repository root: cd figures && python -m _parts.fig_uncertainty.three_level
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr

import draw as house
from .paths import REPRESENTATION

OUT = REPRESENTATION / 'three-level'
KEY = 'b_interface5'                 # 5 Å interface definition (was b_contact_invd2, 1/d² weighted)
LABEL = {'Nesso-1': 'Nesso-1†', 'IPNet-prefusion': 'IPNet'}
COLOR = {'DeepDTA': '#B9CDD3',            # muted Pocket2Mol
         'Nesso-1': '#DAB3C0',            # muted DecompDiff
         'CheapNet': '#C8ADCD',           # muted TargetDiff
         'IPNet': '#AEB5C6', 'IPNet-prefusion': '#AEB5C6',   # muted AR
         'ProFSA': '#BEB3D4',             # muted DiffSBDD
         'ESM-2': '#D3C9A9',              # muted FuncBind
         'CoE': house.pale('CoE'), 'CoDE': house.pale('CoDE')}


def complex_level(order):
    """Complex contact B factor = plain mean B of protein heavy atoms within 5 Å of the ligand.
    Across-complex Spearman of saved out-of-fold MLP predictions (pooled embedding -> MLP, same 5 folds,
    3 seeds; the nested-ridge version is complex_interface5_oof_predictions.csv); 2,000 complex bootstraps."""
    a = pd.read_csv(OUT / 'complex_interface5_mlp_oof_predictions.csv')
    y = a[KEY].to_numpy(); idx = np.random.default_rng(0).integers(0, len(y), (2000, len(y)))
    out = {}
    for m in order:
        p = a[f'{m}__{KEY}'].to_numpy()
        out[m] = (spearmanr(y, p).statistic, *np.percentile([spearmanr(y[i], p[i]).statistic for i in idx], [2.5, 97.5]))
    return out


def local_level(level, target, order):
    """CoE/CoDE: volume-weighted patch pooling (each atom's vdW sphere split over the 2 Å patches it
    overlaps; atomtok_pooling_*). Other models: the atom's/residue's own token (atomtok_all7_*).
    Both runs score the identical 48,591 atoms."""
    s = pd.concat([pd.read_csv(OUT / 'atomtok_all7_scores.csv'), pd.read_csv(OUT / 'atomtok_pooling_scores.csv')])
    s = s[(s.level == level) & (s.target == target)].drop_duplicates('arm', keep='last').set_index('arm')
    key = lambda m: {'IPNet': 'IPNet-prefusion', 'CoE': 'CoE-vol', 'CoDE': 'CoDE-vol'}.get(m, m)
    return {m: tuple(s.loc[key(m), ['within_rho_mean', 'within_lo', 'within_hi']]) for m in order if key(m) in s.index}


def draw_bars(ax, groups, order, ylabel, *, value_fs=11.5, legend_fs=None, legend_where=None, ncol=None):
    """Grouped bars with 95% CIs on an existing axis; '–' marks a model without that granularity."""
    house._pcsz_style(ax, grid_axis='y')
    for side in ['left', 'bottom']: ax.spines[side].set_zorder(6)           # axis over the bars, as in panel A
    step = .84 / len(order)
    for j, m in enumerate(order):
        for g, (_, d) in enumerate(groups):
            x = g + (j - (len(order) - 1) / 2) * step
            if m not in d:
                ax.text(x, .02, '–', ha='center', va='bottom', fontsize=value_fs, color=house.PCSZ_INK); continue
            v, lo, hi = d[m]
            width, edge = step * .93, dict(edgecolor='none')
            if m == 'CoDE':      # ours: 1.1x wider (grown rightward, off its neighbour), outlined at the x-axis weight
                x += width * .05; width *= 1.1
                edge = dict(edgecolor='#3A3A3A',   # black softened slightly toward white
                             linewidth=ax.spines['bottom'].get_linewidth(), gid='ours-outline')
            ax.bar(x, v, width=width, color=COLOR[m], zorder=3, label=LABEL.get(m, m) if g == 0 else None, **edge)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor=house.PCSZ_INK, elinewidth=1.0, capsize=2, capthick=1.0, zorder=4)
            ours = m == 'CoDE'  # our value label: 1 pt larger and bold
            ax.text(x, hi + (.03 if ours else .012), f'{v:.2f}'.replace('0.', '.', 1), ha='center', va='bottom', fontsize=value_fs + (1 if ours else 0),
                    fontweight='bold' if ours else 'normal', color=house.PCSZ_INK, gid='ours-value' if ours else None)
    ax.set_xticks(range(len(groups)), [g for g, _ in groups]); ax.set_xlim(-.55, len(groups) - .45)
    ax.set_ylim(0, 1.05); ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.set_ylabel(ylabel, fontsize=house.PCSZ_LABEL_FS, labelpad=house.PCSZ_YPAD)
    leg = ax.legend(**(legend_where or dict(loc='upper left')), ncol=ncol or len(order), frameon=True, fancybox=False,
                    fontsize=legend_fs or house.PCSZ_KEY_KW['fontsize'] * .9, edgecolor=house.LEGEND_EDGE,
                    facecolor=house.PCSZ_KEY_FACE, framealpha=house.DATA_ALPHA, borderpad=.4, handlelength=1.5,
                    handletextpad=.5, columnspacing=1.0)
    leg.get_frame().set_linewidth(.7)
    halve_outline_in_legend(leg)


def halve_outline_in_legend(legend):
    """The legend swatch of our outlined bar is drawn with half the bar's outline."""
    for text, patch in zip(legend.get_texts(), legend.get_patches()):
        if text.get_text() == LABEL.get('CoDE', 'CoDE'):patch.set_linewidth(patch.get_linewidth() / 2)


ORDER = ['DeepDTA', 'Nesso-1', 'CheapNet', 'IPNet', 'ProFSA', 'ESM-2', 'CoE', 'CoDE']
Z = 'within-complex z B'


def three_level_groups():
    return [('Complex\ncontact B factor', complex_level(ORDER)),
            ('Residue-wise\nB factor', local_level('residue', Z, ORDER)), ('Atom-wise\nB factor', local_level('atom', Z, ORDER))]


def panel(ax, *, value_fs=11.5, legend_fs=None):
    """The three-level panel on an existing axis (used by the combined contact-B figure), legend above it."""
    draw_bars(ax, three_level_groups(), ORDER, 'Decoding\nSpearman ρ', value_fs=value_fs, legend_fs=legend_fs,
              legend_where=dict(loc='lower center', bbox_to_anchor=(.5, 1.02)), ncol=len(ORDER))


def bars(groups, order, ylabel, name, legend_above):
    with plt.rc_context({**house.PCSZ_RC, 'savefig.bbox': None}):
        fig, ax = plt.subplots(figsize=(12.5, 3.6 if legend_above else 3.5)); fig.patch.set_facecolor(house.PCSZ_BG)
        where = dict(loc='upper center', bbox_to_anchor=(.5, 1.13)) if legend_above else None
        draw_bars(ax, groups, order, ylabel, legend_where=where)
        if legend_above: ax.set_ylim(0, 1.08)
        fig.subplots_adjust(left=.09, right=.99, bottom=.2, top=.86 if legend_above else .97)
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'{name}.{ext}', facecolor=house.PCSZ_BG, dpi=220)
        plt.close(fig)


def main():
    groups = three_level_groups()
    pd.DataFrame([dict(group=g.replace('\n', ' '), model=m, rho=v[0], lo=v[1], hi=v[2]) for g, d in groups for m, v in d.items()]
                 ).to_csv(OUT / 'three-level-decoding.csv', index=False)
    bars(groups, ORDER, 'Decoding\nSpearman ρ', 'three-level-decoding', legend_above=True)

    tokens = ['CheapNet', 'IPNet', 'ProFSA', 'ESM-2', 'CoE', 'CoDE']
    groups = [(f'{lvl.title()}\n{lab}', local_level(lvl, t, tokens))
              for lvl in ['atom', 'residue'] for t, lab in [('B', 'B factor'), (Z, 'normalized B factor')]]
    bars(groups, tokens, 'Within-complex\nSpearman ρ', 'atom-level-decoding', legend_above=False)
    print('Wrote three-level-decoding and atom-level-decoding to', OUT)


if __name__ == '__main__':
    main()

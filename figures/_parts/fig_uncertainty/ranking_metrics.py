#!/usr/bin/env python3
"""Extend the motivation B panel to its five decoding descriptors.

Saved manuscript CL3 predictions only. No model fitting or test-based selection.
Run with --plots-only to redraw cached bootstrap tables after style changes.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import draw as house
from .motivation import MAIN_KEYS, LABELS, axes_style, save
from .paths import CL3, ATOMIC_B, MOTIVATION, RANKING_METRICS as OUT

SOURCE = CL3 / 'per-complex.csv'
N_BOOT = 8000
BATCH = 500
RNG_SEED = 20260920
DIRECTION = {key: (-1 if key in ['lig_rscc', 'poc_rscc_median'] else 1) for key in MAIN_KEYS}
SLUGS = dict(zip(MAIN_KEYS, ['contact-b', 'normalized-contact-b', 'ligand-rscc', 'polymer-rscc', 'alternate-location-fraction']))


def corr(pred, truth):
    """Spearman with average ranks, including ties induced by resampling."""
    rp = rankdata(pred, axis=-1)
    ry = rankdata(truth, axis=-1)
    rp -= rp.mean(-1, keepdims=True)
    ry -= ry.mean(-1, keepdims=True)
    return (rp * ry).sum(-1) / np.sqrt((rp * rp).sum(-1) * (ry * ry).sum(-1))


def stratify(x, key):
    if key == 'occ_site_frac_altloc':
        # Keep all equal zero values together: three equal-count bins do not exist.
        cut = float(np.median(x[x > 0]))
        groups = np.where(x == 0, 0, np.where(x < cut, 1, 2))
        return groups, ['Zero', 'Lower positive', 'Higher positive'], [0.0, cut], 'zero plus positive median split'
    cuts = np.quantile(x, [1 / 3, 2 / 3])
    assert cuts[0] < cuts[1], f'Degenerate terciles: {key}'
    return np.searchsorted(cuts, x, side='right'), ['Low', 'Middle', 'High'], cuts.tolist(), 'terciles, ties kept together'


def bootstrap_group(a, b, t, rng):
    values = []
    for _ in range(N_BOOT // BATCH):
        idx = rng.integers(len(t), size=(BATCH, len(t)))
        truth = t[idx][:, None, :]
        av = a[:, idx].transpose(1, 0, 2)
        bv = b[:, idx].transpose(1, 0, 2)
        values.append((corr(bv, truth) - corr(av, truth)).mean(1))
    boot = np.concatenate(values)
    assert len(boot) == N_BOOT and np.isfinite(boot).all()
    return boot


def analyse():
    frame = pd.read_csv(SOURCE)
    assert len(frame) == 733 and frame.pid.is_unique
    original = pd.read_csv(ATOMIC_B / 'per-complex.csv').set_index('pid').loc[frame.pid]
    for arm in ['CoE', 'CoDE']:
        cols = [f'{arm}_seed{s}_prediction' for s in range(5)]
        np.testing.assert_allclose(frame[cols], original[cols], rtol=0, atol=0)
    rows, contrasts, members, seeds = [], [], [], []
    for j, key in enumerate(MAIN_KEYS):
        data = frame.loc[np.isfinite(frame[key])].copy()
        x = data[key].to_numpy()
        groups, labels, cuts, kind = stratify(x, key)
        rng = np.random.default_rng(RNG_SEED + j)
        boot_by_group, points = {}, []
        for g, label in enumerate(labels):
            block = data.loc[groups == g]
            t = block.y_true.to_numpy()
            a, b = [np.stack([block[f'{arm}_seed{s}_prediction'].to_numpy() for s in range(5)]) for arm in ['CoE', 'CoDE']]
            assert len(t) >= 3 and len(np.unique(t)) >= 2
            rho_a, rho_b = corr(a, t[None]), corr(b, t[None])
            for seed in range(5):
                np.testing.assert_allclose(rho_a[seed], spearmanr(a[seed], t).statistic, atol=1e-12)
                np.testing.assert_allclose(rho_b[seed], spearmanr(b[seed], t).statistic, atol=1e-12)
                seeds.append(dict(descriptor=key, group=label, seed=seed, CoE=rho_a[seed], CoDE=rho_b[seed], delta=rho_b[seed] - rho_a[seed]))
            point = float((rho_b - rho_a).mean())
            boot = bootstrap_group(a, b, t, rng)
            lo, hi = np.quantile(boot, [.025, .975])
            boot_by_group[g] = boot
            points.append(point)
            rows.append(dict(descriptor=key, label=LABELS[key].replace('\n', ' '), group=label, group_index=g,
                             n=len(block), n_valid=len(data), n_missing=len(frame) - len(data), binning=kind,
                             threshold_1=cuts[0], threshold_2=cuts[1], x_min=block[key].min(), x_max=block[key].max(),
                             x_median=block[key].median(), y_mean=t.mean(), y_sd=t.std(ddof=1),
                             coe_spearman=rho_a.mean(), code_spearman=rho_b.mean(), estimate=point, ci_lo=lo, ci_hi=hi))
            members.extend(dict(descriptor=key, pid=pid, x=value, group=label, group_index=g) for pid, value in zip(block.pid, block[key]))
        direction = DIRECTION[key]
        contrast = direction * (boot_by_group[2] - boot_by_group[0])
        lo, hi = np.quantile(contrast, [.025, .975])
        contrasts.append(dict(descriptor=key, contrast='higher-proxy minus lower-proxy ranking improvement',
                              higher_group=labels[2 if direction == 1 else 0], lower_group=labels[0 if direction == 1 else 2],
                              estimate=direction * (points[2] - points[0]), ci_lo=lo, ci_hi=hi))
        print(key, 'n=', len(data), 'groups=', [int((groups == g).sum()) for g in range(3)],
              'gaps=', [round(v, 4) for v in points], flush=True)
    table = pd.DataFrame(rows)
    old = pd.read_csv(ATOMIC_B / 'ranking-gap-complex-bootstrap.csv').iloc[:3]
    np.testing.assert_allclose(table.iloc[:3][['estimate', 'ci_lo', 'ci_hi']], old[['estimate', 'ci_lo', 'ci_hi']], rtol=0, atol=1e-12)
    table.to_csv(OUT / 'ranking-by-metric.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / 'higher-vs-lower-proxy.csv', index=False)
    pd.DataFrame(members).to_csv(OUT / 'bin-membership.csv', index=False)
    pd.DataFrame(seeds).to_csv(OUT / 'per-seed-ranking.csv', index=False)
    (OUT / 'provenance.json').write_text(json.dumps(dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        descriptors=MAIN_KEYS, seeds=list(range(5)), bootstrap_replicates=N_BOOT, random_seed_per_descriptor={key:RNG_SEED+j for j,key in enumerate(MAIN_KEYS)},
        contact_b_reproduces_original=True, interval='pointwise percentile 95%, paired complex resampling within fixed bins',
        altloc_rule='zero vs positive below median vs positive at/above median', uncertainty_proxy_direction=DIRECTION), indent=2) + '\n')
    return table


def panel(ax, table, key, ylim, show_ylabel=True, metric="spearman"):
    axes_style(ax)
    block = table.loc[table.descriptor == key].sort_values('group_index')
    x = block.group_index.to_numpy()
    mean, lo, hi = [block[c].to_numpy() for c in ['estimate', 'ci_lo', 'ci_hi']]
    ax.plot(x, mean, 'o-', color=house.pale('CoDE'), lw=2.5 * house.PCSZ_LW_SCALE, markersize=5.5, markerfacecolor='white', markeredgecolor=house.pale('CoDE'), markeredgewidth=1.15, zorder=4)
    ax.vlines(x, lo, hi, color=house.pale('CoDE'), lw=1.15, zorder=3)
    for xx, aa, bb in zip(x, lo, hi):
        ax.hlines([aa, bb], xx - .055, xx + .055, color=house.pale('CoDE'), lw=1.15, zorder=3)
    for xx, value, upper in zip(x, mean, hi):
        ax.annotate(f'{value:+.3f}', (xx, upper), xytext=(0, 6), ha='center', va='bottom',
                    textcoords='offset points', fontsize=10, color=house.PCSZ_INK)
    tick_names = block.range_label.tolist() if 'range_label' in block and block.range_label.notna().all() else (['None', 'Lower +', 'Higher +'] if key == 'occ_site_frac_altloc' else ['Low', 'Middle', 'High'])
    if 'quartile' in block and block.quartile.notna().all():tick_names=block.quartile.tolist()
    ax.set_xticks(x, [f'{name}\nn = {n}' for name, n in zip(tick_names, block.n)])
    ax.axhline(0, color=house.PCSZ_AXIS, lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.set_xlim(float(x.min()) - .3, float(x.max()) + .3)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    if show_ylabel:
        ylabel = 'ΔSpearman\n(CoDE − CoE)' if metric == 'spearman' else 'ΔRMSE (pK)\n(CoDE − CoE)'
        ax.set_ylabel(ylabel, fontsize=house.PCSZ_LABEL_FS, labelpad=house.PCSZ_YPAD)
    ax.set_xlabel('')
    ax.set_title(LABELS[key].replace('\n', ' '), fontsize=13, pad=20)
    direction = 'Lower RSCC → weaker support' if DIRECTION[key] == -1 else ('Higher fraction → more modeled heterogeneity' if key == 'occ_site_frac_altloc' else 'Higher B factor → more disorder')
    ax.text(.5, 1.01, direction, transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='#555555')


def plot(table):
    # Shared scale lets magnitudes and confidence intervals be compared across metrics.
    lo = min(-.06, float(table.ci_lo.min()) - .035)
    hi = max(.125, float(table.ci_hi.max()) + .035)
    ylim = (np.floor(lo / .05) * .05, np.ceil(hi / .05) * .05)
    with plt.rc_context({**house.PCSZ_RC, 'savefig.bbox': None}):
        fig, axes = plt.subplots(1, 5, figsize=(21, 3.7), sharey=True)
        fig.patch.set_facecolor(house.PCSZ_BG)
        for j, (ax, key) in enumerate(zip(axes, MAIN_KEYS)):
            panel(ax, table, key, ylim, show_ylabel=j == 0)
        fig.subplots_adjust(left=.061, right=.991, bottom=.245, top=.82, wspace=.22)
        save(fig, 'uncertainty-motivation-b-affinity-ranking', out=MOTIVATION)


def write_readme(table):
    lines = ['# Affinity ranking gain by the five decoding descriptors', '',
             '[Current B PNG](../../../motivation/uncertainty-motivation-b-affinity-ranking.png) · [PDF](../../../motivation/uncertainty-motivation-b-affinity-ranking.pdf)', '',
             'Each point is mean-over-five-seeds Spearman(CoDE) − Spearman(CoE), recomputed within that descriptor bin. Positive favors CoDE. Error bars are 8000 paired-complex bootstrap percentile 95% intervals, conditional on fitted seeds and fixed thresholds. These are confidence intervals for a performance difference, not per-example predictive uncertainty.', '',
             'The first four descriptors use raw-value terciles, with equal values kept together; bin sizes can differ. Alternate-location fraction has 605 zeros among 732 measured cases, so its groups are zero / lower-positive / higher-positive, split at the median of positive values. One missing alternate-location value is omitted only from that panel. This three-group rule is descriptive, not equal-count terciles.', '',
             'All axes run from smaller to larger RAW descriptor values. Higher B or alternate-location fraction is oriented toward more disorder/heterogeneity; LOWER RSCC is oriented toward weaker model–map support. Polymer RSCC is the report-wide upper median, not a pocket-specific measure. See [metric definitions](../../../motivation/metric-definitions.md).', '',
             'The contact-B values and bootstrap interval endpoints reproduce the original B panel (tolerance 1e-12). All panels share a y scale. There are no panel letters or x-axis titles. ProFSA is absent here because the plotted comparison remains CoDE versus CoE.', '',
             '| Descriptor | Bin | n | Observed x range | CoE ρ | CoDE ρ | Difference [95% CI] |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in table.itertuples():
        lines.append(f'| {row.label} | {row.group} | {row.n} | {row.x_min:.5g}–{row.x_max:.5g} | {row.coe_spearman:.3f} | {row.code_spearman:.3f} | {row.estimate:+.3f} [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}] |')
    lines += ['', '## Scope and reproduction', '',
              'Exploratory stratification of the current CL3 test cohort; different complexes and affinity distributions occur in each bin. Intervals are pointwise, without correction across metrics/bins, protein-family clustering or seed resampling. A subgroup interval above zero is not evidence that its gain differs from another subgroup. `higher-vs-lower-proxy.csv` separately reports that contrast, oriented for each descriptor, also with pointwise intervals. Associations do not establish a causal uncertainty mechanism; manuscript CoE and CoDE pretraining recipes also differ.', '',
              'Run from the repository root:', '', '```bash',
              'cd figures && python -m _parts.fig_uncertainty.ranking_metrics',
              '# Style-only redraw from saved statistics:',
              'cd figures && python -m _parts.fig_uncertainty.ranking_metrics --plots-only', '```', '',
              'Saved files include aggregate results, per-seed rankings, exact complex/bin memberships, source hash, thresholds and observed ranges. Training and feature extraction are not repeated.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if (MOTIVATION / 'active-analysis.json').exists():
        from .update_expanded import redraw_current_figures
        redraw_current_figures(only=['uncertainty-b-affinity-ranking'])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(OUT / 'ranking-by-metric.csv') if args.plots_only else analyse()
    plot(table)
    write_readme(table)


if __name__ == '__main__':
    main()

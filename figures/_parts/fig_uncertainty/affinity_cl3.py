#!/usr/bin/env python3
"""Audit existing affinity-probe predictions on the frozen CL123 test subset.
Run: cd figures && python -m _parts.fig_uncertainty.affinity_cl3
This filters evaluation membership; it does not retrain either probe on CL123.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'figures/_parts/fig_uncertainty'))
from voxbind.splits import load_split, content_hash
import uncertainty_descriptors as ud

SOURCE = ROOT / 'results/task1-affinity/_shared/260916_affinity_uncertainty/affinity_uncertainty.json'
SCATTER = ROOT / 'dataset/data/pdbbind/results/scatter'
import draw as house
from .paths import EXPLORATORY_CL3 as OUT
METRICS = {
    'b_contact_invd2': 'Contact-weighted B: 1/d² (Å²)',
    'b_contact_gauss': 'Contact-weighted B: Gaussian (Å²)',
    'b_contact_count': 'Contact-weighted B: contact count (Å²)',
    'b_contact_invd2_z': 'Contact-weighted B: within-protein z',
    'b_contact_shell_mean': 'Unweighted B: 4.5 Å shell (Å²)',
    'rel_pocket_b': 'Mean pocket B / mean protein B',
}


def bootstrap_mean(y, seed=0):
    rng = np.random.default_rng(seed)
    return np.quantile(y[rng.integers(len(y), size=(8000, len(y)))].mean(1), [.025, .975])


def save(fig, name):
    from draw import prepare, retained_formats
    if not prepare(fig, OUT / name):
        plt.close(fig);return
    for ext in retained_formats(OUT / name):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_results(df, summary, blocks):
    summaries = summary.to_dict("records")
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), sharey=True, layout='constrained')
    for ax, (key, label), s in zip(axes.flat, METRICS.items(), summaries):
        x, y = df[key].to_numpy(), df.delta_abs_error.to_numpy()
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        ax.scatter(x, y, s=13, alpha=house.DATA_ALPHA, color=house.pale('CoDE'), linewidths=0, rasterized=True)
        ax.axhline(0, color='#444444', lw=.8)
        block = blocks[key]
        for i, b in enumerate(block['bins']):
            mask = (x >= b['lo_edge']) & ((x <= b['hi_edge']) if i == 2 else (x < b['hi_edge']))
            if i:
                ax.axvline(b['lo_edge'], color='#a0a0a0', ls=':', lw=1)
            center = np.median(x[mask])
            ax.errorbar(center, b['mean'], yerr=[[b['mean']-b['ci_lo']], [b['ci_hi']-b['mean']]],
                        fmt='o', color='#d36723', capsize=4, ms=6, zorder=5)
        ax.set_xlabel(label)
        ax.set_title(f"n={s['n']}  |  Spearman ρ={s['spearman']:+.3f}, p={s['spearman_p']:.3f}", fontsize=10)
        ax.grid(axis='y', alpha=house.BACKGROUND_ALPHA)
    for ax in axes[:, 0]:
        ax.set_ylabel('Δ absolute pK error (CoDE − CoE)\nNegative: CoDE better')
    fig.suptitle(f'CL1+CL2+CL3 filtered test · {len(df)} paired complexes\n'
                 'One dot per complex; orange: tercile mean ± bootstrap 95% CI; dotted: tercile boundaries', fontsize=13)
    save(fig, 'affinity-cl3-bfactor-scatter')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), sharey=True, layout='constrained')
    for ax, (key, label) in zip(axes.flat, METRICS.items()):
        bs = blocks[key]['bins']
        means = np.array([b['mean'] for b in bs])
        ax.bar(range(3), means, color=house.pale('CoDE'), width=.6)
        ax.errorbar(range(3), means, yerr=[means-[b['ci_lo'] for b in bs], [b['ci_hi'] for b in bs]-means],
                    fmt='none', color='#222222', capsize=5)
        ax.axhline(0, color='#444444', lw=.8)
        ax.set_xticks(range(3), [f"{b['lo_edge']:.3g}–{b['hi_edge']:.3g}\nn={b['n']}" for b in bs])
        ax.set_xlabel(label)
        s = summary.set_index('descriptor').loc[key]
        ax.set_title(f"Top − rest = {s.top_minus_rest:+.3f} pK\np={s.permutation_p:.3f}; BH q={s.contrast_q_bh6:.3f}", fontsize=11)
        ax.grid(axis='y', alpha=house.BACKGROUND_ALPHA)
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean Δ absolute pK error (CoDE − CoE)\nNegative: CoDE better')
    fig.suptitle('CL1+CL2+CL3 filtered test · terciles recomputed within this subset\n'
                 'Bootstrap 95% CI; exploratory top-versus-rest tests, BH correction across six metrics', fontsize=13)
    save(fig, 'affinity-cl3-bfactor-terciles')



def redraw_saved():
    df = pd.read_csv(OUT / "per-complex.csv")
    summary = pd.read_csv(OUT / "descriptor-statistics.csv")
    bins = pd.read_csv(OUT / "terciles.csv")
    blocks = {key: {"bins": bins[bins.descriptor == key].to_dict("records")} for key in METRICS}
    plot_results(df, summary, blocks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SOURCE.read_text())
    assert data['protocol']['split'] == 'lp_edrscc_v2'
    test = set(load_split('lp_edrscc_v2_cl123')['test'])
    original_test = set(load_split('lp_edrscc_v2')['test'])
    assert test <= original_test
    pids = sorted(test & set(data['targets']))
    assert set(pids) == test, 'Missing CL123 test predictions/descriptors'
    seed_preds, sources = {}, []
    for arm, cond in [('CoE', 'atomblob'), ('CoDE', 'atomblob_density_gradmag')]:
        for seed in range(3):
            path = SCATTER / f'scatter_e99_v5_lp_edrscc_v2_{cond}_seed{seed}.csv'
            sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            raw = pd.read_csv(path)
            assert raw.pid.is_unique and set(raw.pid) <= original_test
            assert set(raw.condition) == {cond} and set(raw.seed) == {seed}
            assert test <= set(raw.pid)
            seed_preds[arm, seed] = raw.set_index('pid').loc[pids]
    rows = []
    for pid in pids:
        coe, code = data['arms']['C'][pid], data['arms']['C+D+G'][pid]
        assert np.isclose(coe['y_true'], code['y_true'])
        row = dict(data['complexes'][pid], pid=pid, y_true=coe['y_true'])
        for arm, saved in [('CoE', coe), ('CoDE', code)]:
            preds = np.array([seed_preds[arm, s].loc[pid, 'y_pred'] for s in range(3)])
            assert all(np.isclose(seed_preds[arm, s].loc[pid, 'y_true'], row['y_true']) for s in range(3))
            assert np.isclose(preds.mean(), saved['y_pred'], rtol=0, atol=1e-10)
            row[f'{arm}_prediction'] = preds.mean()
            row[f'{arm}_abs_error'] = abs(preds.mean() - row['y_true'])
            row[f'{arm}_mean_seed_abs_error'] = np.abs(preds - row['y_true']).mean()
            for seed, pred in enumerate(preds):
                row[f'{arm}_seed{seed}_prediction'] = pred
        row['delta_abs_error'] = row['CoDE_abs_error'] - row['CoE_abs_error']
        row['delta_prediction'] = row['CoDE_prediction'] - row['CoE_prediction']
        row['delta_mean_seed_abs_error'] = row['CoDE_mean_seed_abs_error'] - row['CoE_mean_seed_abs_error']
        assert np.isclose(row['delta_abs_error'], data['deltas']['C+D+G'][pid]['abs_error'])
        for seed in range(3):
            row[f'delta_seed{seed}'] = abs(row[f'CoDE_seed{seed}_prediction'] - row['y_true']) - abs(row[f'CoE_seed{seed}_prediction'] - row['y_true'])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'per-complex.csv', index=False)
    blocks, summaries, bins_out = {}, [], []
    for key in METRICS:
        part = df[[key, 'delta_abs_error']].replace([np.inf, -np.inf], np.nan).dropna()
        x, y = part[key].to_numpy(), part.delta_abs_error.to_numpy()
        block = ud.tercile_split(x, y)
        blocks[key] = block
        rho, p = stats.spearmanr(x, y)
        c = block['contrast']
        # Existing helper reports exceedances / 8000; use the finite-permutation correction.
        perm_p = (round(c['p_permutation'] * 8000) + 1) / 8001
        summaries.append(dict(descriptor=key, n=len(y), spearman=rho, spearman_p=p,
                              mean_delta=y.mean(), top_minus_rest=c['top_minus_rest'],
                              contrast_ci_lo=c['ci_lo'], contrast_ci_hi=c['ci_hi'], permutation_p=perm_p))
        edges = [b['lo_edge'] for b in block['bins']] + [block['bins'][-1]['hi_edge']]
        for i, b in enumerate(block['bins']):
            mask = (x >= edges[i]) & ((x <= edges[i + 1]) if i == 2 else (x < edges[i + 1]))
            idx = part.index[mask]
            bins_out.append(dict(descriptor=key, tercile=i + 1, **b,
                                 CoE_mae=df.loc[idx, 'CoE_abs_error'].mean(),
                                 CoDE_mae=df.loc[idx, 'CoDE_abs_error'].mean(),
                                 fraction_CoDE_better=(y[mask] < 0).mean()))
    summary = pd.DataFrame(summaries)
    summary['spearman_q_bh6'] = ud.bh(summary.spearman_p.to_numpy())
    summary['contrast_q_bh6'] = ud.bh(summary.permutation_p.to_numpy())
    summary.to_csv(OUT / 'descriptor-statistics.csv', index=False)
    pd.DataFrame(bins_out).to_csv(OUT / 'terciles.csv', index=False)

    plot_results(df, summary, blocks)

    sensitivity = []
    for outcome in ['delta_abs_error', 'delta_mean_seed_abs_error', 'delta_seed0', 'delta_seed1', 'delta_seed2']:
        y = df[outcome].to_numpy()
        for key in METRICS:
            x = df[key].to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            cut = np.quantile(x[valid], 2/3)
            sensitivity.append(dict(outcome=outcome, descriptor=key, n=int(valid.sum()),
                                    mean_delta=y[valid].mean(),
                                    top_minus_rest=y[valid & (x >= cut)].mean()-y[valid & (x < cut)].mean(),
                                    spearman=stats.spearmanr(x[valid], y[valid]).statistic))
    pd.DataFrame(sensitivity).to_csv(OUT / 'seed-sensitivity.csv', index=False)
    # Partial rank associations are descriptive sensitivity checks, not causal effects.
    partials = []
    for key in METRICS:
        for controls in [['y_true'], ['y_true', 'resolution']]:
            part = df[[key, 'delta_abs_error'] + controls].replace([np.inf, -np.inf], np.nan).dropna()
            r = part.rank().to_numpy()
            design = np.column_stack([np.ones(len(r)), r[:, 2:]])
            residual = r[:, :2] - design @ np.linalg.lstsq(design, r[:, :2], rcond=None)[0]
            partials.append(dict(descriptor=key, controls='+'.join(controls), n=len(r),
                                 partial_spearman=np.corrcoef(residual.T)[0, 1]))
    pd.DataFrame(partials).to_csv(OUT / 'partial-correlations.csv', index=False)
    y = df.delta_abs_error.to_numpy()
    lo, hi = bootstrap_mean(y)
    protocol = dict(evaluation_split='lp_edrscc_v2_cl123/test', prediction_split='lp_edrscc_v2',
                    retrained_on_cl123=False, n_manifest_test=len(test), n_paired=len(df),
                    n_original=len(data['targets']), pid_sha256=content_hash(pids),
                    source_json=str(SOURCE.relative_to(ROOT)), source_json_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    prediction_sources=sources, seeds=[0, 1, 2],
                    aggregation='Average predictions over three seeds, then compute absolute error',
                    difference='abs(pred_CoDE - y_true) - abs(pred_CoE - y_true)',
                    mean_delta=float(y.mean()), mean_delta_ci95=[float(lo), float(hi)],
                    CoE_mae=float(df.CoE_abs_error.mean()), CoDE_mae=float(df.CoDE_abs_error.mean()),
                    fraction_CoDE_better=float((y < 0).mean()), bootstrap_replicates=8000,
                    caveat='Evaluation-only CL123 filtering of original probes; complex bootstrap assumes independent complexes.')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    lines = ['# CL3-filtered affinity error audit', '',
             f"All {len(df)}/{len(test)} frozen CL1+CL2+CL3 test complexes have both predictions and descriptors.",
             'The previous figure used 779 complexes; this analysis uses only the CL123 test manifest.',
             '**Evaluation filtering only:** these are the existing lp_edrscc_v2 probes, not probes retrained on CL123 train/val.', '',
             'CoE = C / atomblob; CoDE = C+D+G / atomblob_density_gradmag, matching the existing affinity figure.',
             'Each point is one complex. Average predictions over seeds 0, 1, 2 first, then take absolute error.',
             'Δ = |prediction(CoDE) − measured pK| − |prediction(CoE) − measured pK|; negative favors CoDE.', '',
             f"MAE: CoE {protocol['CoE_mae']:.4f}; CoDE {protocol['CoDE_mae']:.4f} pK.",
             f"Mean paired Δ = {y.mean():+.4f} pK, bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}].",
             f"CoDE has lower absolute error on {(y < 0).mean():.1%} of complexes.", '',
             '## Interpretation and definitions', '',
             'Scatter plots show the full point cloud without clipping outliers. Orange points are tercile means with percentile bootstrap 95% CIs.',
             'Tercile edges are recomputed on this subset, not reused from the original figure.',
             'Contact weights sum over ligand heavy atoms: inverse-square distances floored at 2 Å; Gaussian sigma=1.5 Å; contact count within 4.5 Å.',
             'Weighted averages use pocket protein heavy atoms. Within-protein z subtracts the whole-protein mean B and divides by its standard deviation.',
             'Raw B-factor may reflect overall structure/refinement differences. Normalized metrics and partial rank correlations with measured pK/resolution are sensitivity checks.',
             'Top-versus-rest permutation p-values use 8000 permutations and a +1 correction; BH q-values adjust the six displayed metrics as an exploratory family.',
             'Complex-level bootstrap assumes independent complexes; related targets may reduce the effective sample size. These are not causal estimates or independent replications.',
             'The RSCC-filtered parent cohort restricts generalization to poorly resolved structures.', '',
             '## Files', '',
             '- affinity-cl3-bfactor-scatter.{png,pdf,svg}: six descriptor scatter panels.',
             '- affinity-cl3-bfactor-terciles.{png,pdf,svg}: companion tercile panels.',
             '- per-complex.csv: paired predictions, errors, differences, seed predictions, descriptors, PDB IDs.',
             '- descriptor-statistics.csv / terciles.csv: full statistics and bin-level MAEs.',
             '- seed-sensitivity.csv: individual seeds and mean seed errors versus errors of mean predictions.',
             '- partial-correlations.csv: descriptive partial rank correlations.',
             '- protocol.json: input hashes, coverage, model mapping, aggregation and summary.', '',
             'Reproduce: `cd figures && python -m _parts.fig_uncertainty.affinity_cl3`', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in protocol.items() if k not in ['prediction_sources']}, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser();parser.add_argument('--plots-only', action='store_true')
    if parser.parse_args().plots_only:redraw_saved()
    else:main()

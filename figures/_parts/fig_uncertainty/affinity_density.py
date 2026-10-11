#!/usr/bin/env python3
"""Binned affinity-error plots and input/embedding uncertainty diagnostics.
Run with the voxbind env's python. No affinity model is retrained.
"""
from pathlib import Path
import json
import sys
import hashlib
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import draw as house
import numpy as np
import pandas as pd
from scipy import stats, ndimage, linalg
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'figures/_parts/fig_uncertainty'))
import uncertainty_descriptors as ud
sys.path.insert(0, str(ROOT / 'figures'))
from .affinity_cl3 import METRICS

from .paths import EXPLORATORY_CL3 as BASE, MAP_DIAGNOSTICS as OUT
DATA = ROOT / 'dataset/data/pdbbind'
UNC = {
    'b_contact_invd2': 'Contact B (1/d²)',
    'b_contact_invd2_z': 'Contact B (protein z)',
    'rel_pocket_b': 'Pocket / protein B',
    'lig_rscc': 'Ligand RSCC',
    'poc_rscc_median': 'Pocket RSCC',
    'occ_site_frac_partial': 'Partial-occupancy fraction',
    'occ_site_frac_altloc': 'Alternate-location fraction',
    'resolution': 'Resolution (Å)',
}
MAP = {
    'map_std': 'Crop density SD',
    'map_p95': 'Crop density 95th percentile',
    'map_sharpness': 'Gradient energy / density variance (Å⁻²)',
    'map_contact_density': 'Contact-weighted density at protein atoms',
    'map_contact_peak_contrast': 'Contact peak contrast at 0.75 Å',
    'map_ligand_density': 'Mean density at ligand atoms',
}
EMBED = {
    'CoE': 'atomblob_e99_v5.pt',
    'CoDE': 'atomblob_density_gradmag_e99_v5.pt',
    'Density-only': 'density_only_learnable_e99_v5.pt',
}
COLORS = {'CoE': house.pale('CoE'), 'CoDE': house.pale('CoDE'), 'Density-only': house._gentle_color('#9a60ad'), 'Map statistics': house._gentle_color('#d47a30')}


def save(fig, name):
    from draw import prepare, retained_formats
    if not prepare(fig, OUT / name):
        plt.close(fig);return
    for ext in retained_formats(OUT / name):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def binned(x, y, bins=10, equal_width=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    edges = (np.linspace(x.min(), x.max(), bins+1) if equal_width else np.unique(np.quantile(x, np.linspace(0, 1, bins+1))))
    groups = np.clip(np.searchsorted(edges, x, side='right')-1, 0, len(edges)-2)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(len(edges)-1):
        sel = groups == i
        a, v = x[sel], y[sel]
        if not len(v):
            continue
        means = v[rng.integers(len(v), size=(4000, len(v)))].mean(1)
        lo, hi = np.quantile(means, [.025, .975])
        rows.append(dict(bin=i+1, n=len(v), x_mean=a.mean(), x_lo=edges[i], x_hi=edges[i+1],
                         mean=v.mean(), ci_lo=lo, ci_hi=hi))
    assert sum(r['n'] for r in rows) == len(x)
    return pd.DataFrame(rows)


def draw_bins(ax, table, color=house.pale('CoDE'), label=None, annotate=False):
    ax.errorbar(table.x_mean.to_numpy(), table['mean'].to_numpy(), yerr=np.array([table['mean']-table.ci_lo, table.ci_hi-table['mean']]),
                fmt='o-', color=color, capsize=3, markersize=5, linewidth=1.2, label=label)
    if annotate:
        for r in table.itertuples():
            ax.annotate(str(r.n), (r.x_mean, r.ci_hi), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=7)
    ax.grid(alpha=house.BACKGROUND_ALPHA)


def map_row(pid):
    arr = np.load(DATA / 'voxels_v5/density' / f'{pid}.npy').astype(np.float64)
    assert arr.shape == (64, 64, 64) and np.isfinite(arr).all()
    ligand = ud.ligand_coords(DATA / 'structures/pbpp-2020' / pid / f'{pid}_ligand.sdf')
    center = ligand.mean(0)
    atoms = ud.protein_atoms(DATA / 'structures/pbpp-2020' / pid / f'{pid}_pocket.pdb')
    xyz = np.asarray([a[4:7] for a in atoms])
    grid = (xyz-center)/.25+32
    valid = ((grid >= 3) & (grid <= 60)).all(1)
    coords = grid[valid]
    assert len(coords)
    rho = ndimage.map_coordinates(arr, coords.T, order=1, mode='nearest')
    around = []
    for axis in range(3):
        for sign in [-1, 1]:
            offsets = np.zeros(3); offsets[axis] = 3*sign
            around.append(ndimage.map_coordinates(arr, (coords+offsets).T, order=1, mode='nearest'))
    dist = np.linalg.norm(xyz[valid, None, :]-ligand[None, :, :], axis=2)
    weights = (1/np.maximum(dist, 2)**2).sum(1)
    lig_grid = (ligand-center)/.25+32
    lig_valid = ((lig_grid >= 0) & (lig_grid <= 63)).all(1)
    ligand_rho = ndimage.map_coordinates(arr, lig_grid[lig_valid].T, order=1, mode='nearest')
    grad_energy = sum(np.mean(g*g) for g in np.gradient(arr, .25))
    return dict(pid=pid, map_std=arr.std(), map_p95=np.quantile(arr, .95),
                map_sharpness=grad_energy/arr.var(),
                map_contact_density=np.average(rho, weights=weights),
                map_contact_peak_contrast=np.average(rho-np.mean(around, axis=0), weights=weights),
                map_ligand_density=ligand_rho.mean(), n_map_protein_atoms=int(valid.sum()),
                fraction_map_protein_atoms=valid.mean(), fraction_map_ligand_atoms=lig_valid.mean())


def ridge_predictions(xtrain, ytrain, xtest, alphas):
    """Fit scaling and ridge on training only; return predictions for all penalties."""
    mean, sd = xtrain.mean(0), xtrain.std(0)
    sd[sd < 1e-10] = 1
    x = (xtrain-mean)/sd
    xt = (xtest-mean)/sd
    ym, ys = ytrain.mean(0), ytrain.std(0)
    ys[ys < 1e-10] = 1
    y = (ytrain-ym)/ys
    u, s, vh = linalg.svd(x, full_matrices=False, check_finite=False, lapack_driver='gesdd')
    uy = u.T@y
    projections = xt@vh.T
    return np.stack([(projections@((s/(s*s+a))[:, None]*uy))*ys+ym for a in alphas], axis=-1)


def folds(n, k, seed):
    ids = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(ids, k)
    return [(np.setdiff1d(np.arange(n), test), test) for test in parts]


def decode(x, y):
    """Five outer folds, three inner folds; select alpha per target by inner MSE."""
    alphas = np.array([.1, 1, 10, 100, 1000, 10000])
    pred, baseline = np.full_like(y, np.nan), np.full_like(y, np.nan)
    foldid = np.zeros(len(y), int)
    selections = []
    for outer, (train, test) in enumerate(folds(len(y), 5, 42)):
        xt, yt = x[train], y[train]
        loss = np.zeros((y.shape[1], len(alphas)))
        for itrain, ival in folds(len(train), 3, 100+outer):
            ipred = ridge_predictions(xt[itrain], yt[itrain], xt[ival], alphas)
            loss += ((ipred-yt[ival, :, None])**2).sum(0)
        chosen = loss.argmin(1)
        predictions = ridge_predictions(xt, yt, x[test], alphas)
        pred[test] = predictions[:, np.arange(y.shape[1]), chosen]
        baseline[test] = yt.mean(0)
        foldid[test] = outer
        selections.append(alphas[chosen].tolist())
    return pred, baseline, foldid, selections


def redraw_saved_embedding_bins():
    decoded=pd.read_csv(OUT/'embedding-oof-predictions.csv')
    scores=pd.read_csv(OUT/'embedding-decoding-statistics.csv')
    bins=pd.read_csv(OUT/'embedding-decoding-bins.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
    selected_unc = ['b_contact_invd2','b_contact_invd2_z','lig_rscc','poc_rscc_median','resolution','rel_pocket_b']
    decbins = []
    for ax, u in zip(axes.flat, selected_unc):
        for name in ['CoE','CoDE','Density-only']:
            table = bins[(bins.uncertainty==u)&(bins.representation==name)]
            score = scores[(scores.representation == name)&(scores.uncertainty == u)].iloc[0]
            draw_bins(ax, table, color=COLORS[name], label=f'{name}: ρ={score.spearman_oof:.2f}')
        lo, hi = decoded[u].min(), decoded[u].max()
        ax.plot([lo,hi],[lo,hi], '--', color='#888888', lw=.8, label='Identity')
        ax.set_xlabel('Observed '+UNC[u]); ax.set_ylabel('Decoded descriptor (out-of-fold)')
        ax.legend(fontsize=8, framealpha=house.DATA_ALPHA)
    fig.suptitle('Uncertainty decoded from learned embeddings · binned observed-versus-predicted\n'
                 'Same folds and complexes; bin-mean CIs condition on fitted OOF predictions', fontsize=13)
    save(fig, 'embedding-uncertainty-binned')


def plot_affinity_bins(df, saved_bins=None):
    binrows = []
    for width in [False, True]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.3), sharey=True, layout='constrained')
        for ax, (key, label) in zip(axes.flat, METRICS.items()):
            table = (binned(df[key], df.delta_abs_error, equal_width=width) if saved_bins is None else saved_bins[(saved_bins.descriptor == key) & (saved_bins.scheme == ('equal_width' if width else 'quantile'))].copy())
            table['descriptor'] = key; table['scheme'] = 'equal_width' if width else 'quantile'
            binrows.append(table)
            draw_bins(ax, table, annotate=True)
            ax.axhline(0, color='#555555', lw=.8)
            ax.set_xlabel(label)
            rho = stats.spearmanr(df[key], df.delta_abs_error).statistic
            ax.set_title(f'Point-level Spearman ρ={rho:+.3f}', fontsize=11)
        for ax in axes[:, 0]:
            ax.set_ylabel('Mean Δ absolute pK error (CoDE − CoE)\nNegative: CoDE better')
        fig.suptitle('CL3 filtered test · 733 complexes · '+('10 equal-width x bins' if width else '10 equal-count x bins')+
                     '\nOne point per bin at mean x; bootstrap 95% CI; numbers show bin counts', fontsize=13)
        save(fig, 'affinity-binned-'+('equal-width' if width else 'quantile'))
    return binrows


def plot_maps(df, corr, saved_bins=None):
    for column, suffix in [('spearman', ''), ('partial_spearman', '-adjusted')]:
        matrix = corr.pivot(index='map_metric', columns='uncertainty', values=column).loc[list(MAP), list(UNC)]
        fig, ax = plt.subplots(figsize=(13, 7.5), layout='constrained')
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        for i in range(len(MAP)):
            for j in range(len(UNC)):
                val = matrix.iloc[i, j]
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center', color='white' if abs(val)>.65 else '#222222')
        ax.set_xticks(range(len(UNC)), list(UNC.values()), rotation=35, ha='right')
        ax.set_yticks(range(len(MAP)), list(MAP.values()))
        ax.set_title('Input density map statistics versus crystallographic descriptors\n'+
                     ('Partial rank correlation: adjust resolution + ligand size (resolution column: size only)' if suffix else
                      'Spearman correlation across CL3 test complexes (n=732–733)'), pad=14)
        fig.colorbar(im, ax=ax, label=column.replace('_', ' '), shrink=.75)
        save(fig, 'map-uncertainty-correlations'+suffix)

    pairs = [('b_contact_invd2','map_sharpness'), ('b_contact_invd2','map_contact_density'),
             ('b_contact_invd2_z','map_contact_peak_contrast'), ('lig_rscc','map_ligand_density'),
             ('resolution','map_sharpness'), ('poc_rscc_median','map_contact_peak_contrast')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.3), layout='constrained')
    mapbins = []
    for ax, (u, m) in zip(axes.flat, pairs):
        table = (binned(df[u], df[m]) if saved_bins is None else saved_bins[(saved_bins.uncertainty == u) & (saved_bins.map_metric == m)].copy()); table['uncertainty'] = u; table['map_metric'] = m
        mapbins.append(table)
        draw_bins(ax, table)
        s = corr[(corr.map_metric == m)&(corr.uncertainty == u)].iloc[0]
        ax.set_xlabel(UNC[u]); ax.set_ylabel(MAP[m])
        ax.set_title(f'Point-level ρ={s.spearman:+.3f}; adjusted ρ={s.partial_spearman:+.3f}', fontsize=10)
    fig.suptitle('Actual density input · binned associations (10 quantile bins)\nBin means and bootstrap 95% CI; correlations computed on individual complexes', fontsize=13)
    save(fig, 'map-uncertainty-binned')
    return mapbins


def plot_decoding_summary(scores, names, n):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), layout='constrained')
    for ax, metric, title in zip(axes, ['spearman_oof','r2_oof'], ['Out-of-fold Spearman ρ', 'Out-of-fold R²']):
        matrix = scores.pivot(index='uncertainty', columns='representation', values=metric).loc[list(UNC), names]
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        for i in range(len(UNC)):
            for j in range(len(names)):
                val = matrix.iloc[i, j]
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center', color='white' if abs(val)>.65 else '#222222')
        ax.set_xticks(range(len(names)), names, rotation=25, ha='right')
        ax.set_yticks(range(len(UNC)), list(UNC.values()))
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=.75)
    fig.suptitle(f'Can uncertainty descriptors be read from the representation? n={n}\n'
                 'Ridge decoding: 5 outer folds × 3 inner folds; test-cohort diagnostic, not an independent benchmark', fontsize=13)
    save(fig, 'embedding-uncertainty-decoding')


def redraw_saved_plots():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    df = pd.read_csv(OUT / 'per-complex-density-uncertainty.csv')
    plot_affinity_bins(df, pd.read_csv(OUT / 'affinity-bins.csv'))
    plot_maps(df, pd.read_csv(OUT / 'map-uncertainty-correlations.csv'), pd.read_csv(OUT / 'map-uncertainty-bins.csv'))
    scores = pd.read_csv(OUT / 'embedding-decoding-statistics.csv')
    names = list(scores.representation.drop_duplicates())
    plot_decoding_summary(scores, names, int(scores.n.iloc[0]))
    redraw_saved_embedding_bins()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    df = pd.read_csv(BASE/'per-complex.csv').sort_values('pid').reset_index(drop=True)
    assert len(df) == 733 and df.pid.is_unique
    binrows = plot_affinity_bins(df)
    pd.concat(binrows).to_csv(OUT/'affinity-bins.csv', index=False)
    print('Saved binned affinity plots', flush=True)

    cached = OUT/'map-statistics.csv'
    if cached.exists():
        maps = pd.read_csv(cached)
        assert set(maps.pid) == set(df.pid)
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = []
            for i, row in enumerate(pool.map(map_row, df.pid)):
                rows.append(row)
                if (i+1)%100 == 0:
                    print(f'Map statistics {i+1}/733', flush=True)
        maps = pd.DataFrame(rows)
        maps.to_csv(cached, index=False)
    df = df.merge(maps, on='pid', validate='one_to_one')
    df.to_csv(OUT/'per-complex-density-uncertainty.csv', index=False)
    correlations = []
    for m in MAP:
        for u in UNC:
            p = df[[m, u, 'resolution', 'n_ligand_heavy_atoms']].loc[:, lambda a: ~a.columns.duplicated()].dropna()
            rho, pv = stats.spearmanr(p[m], p[u])
            controls = [c for c in ['resolution', 'n_ligand_heavy_atoms'] if c != u]
            ranked = p[[m, u]+controls].rank().to_numpy()
            z = np.column_stack([np.ones(len(p)), ranked[:, 2:]])
            resid = ranked[:, :2]-z@np.linalg.lstsq(z, ranked[:, :2], rcond=None)[0]
            partial = np.corrcoef(resid.T)[0, 1]
            correlations.append(dict(map_metric=m, uncertainty=u, n=len(p), spearman=rho, p=pv,
                                     partial_spearman=partial, controls='+'.join(controls)))
    corr = pd.DataFrame(correlations)
    corr['q_bh48'] = ud.bh(corr.p.to_numpy())
    corr.to_csv(OUT/'map-uncertainty-correlations.csv', index=False)
    mapbins = plot_maps(df, corr)
    pd.concat(mapbins).to_csv(OUT/'map-uncertainty-bins.csv', index=False)
    print('Saved map correlation plots', flush=True)

    # Identical complete cases and nested folds for all representations and all targets.
    common = df.dropna(subset=list(UNC)+list(MAP)).reset_index(drop=True)
    y = common[list(UNC)].to_numpy(float)
    features = {'Map statistics': common[list(MAP)].to_numpy(float)}
    provenance = {}
    for name, fname in EMBED.items():
        path = DATA/'features'/fname
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        features[name] = np.stack([bundle['features'][pid].numpy() for pid in common.pid]).astype(float)
        provenance[name] = dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                metadata={k:v for k,v in bundle.items() if k not in ['features', 'cache_signature']})
    decoded = pd.DataFrame({'pid':common.pid})
    summaries, selections = [], {}
    with nullcontext():
        for name, x in features.items():
            assert np.isfinite(x).all()
            pred, baseline, foldid, selected = decode(x, y)
            selections[name] = selected
            decoded['fold'] = foldid
            for j, u in enumerate(UNC):
                decoded[u] = y[:, j]
                decoded[f'{name}__{u}'] = pred[:, j]
                rho = stats.spearmanr(y[:, j], pred[:, j]).statistic
                r2 = 1-np.sum((y[:, j]-pred[:, j])**2)/np.sum((y[:, j]-y[:, j].mean())**2)
                skill = 1-np.sum((y[:, j]-pred[:, j])**2)/np.sum((y[:, j]-baseline[:, j])**2)
                summaries.append(dict(representation=name, uncertainty=u, n=len(y), spearman_oof=rho,
                                      r2_oof=r2, skill_vs_fold_train_mean=skill))
            print(f'Decoded {name}: {len(y)} complexes, {x.shape[1]} features', flush=True)
    decoded.to_csv(OUT/'embedding-oof-predictions.csv', index=False)
    scores = pd.DataFrame(summaries)
    scores.to_csv(OUT/'embedding-decoding-statistics.csv', index=False)
    plot_decoding_summary(scores, list(features), len(common))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
    selected_unc = ['b_contact_invd2','b_contact_invd2_z','lig_rscc','poc_rscc_median','resolution','rel_pocket_b']
    decbins = []
    for ax, u in zip(axes.flat, selected_unc):
        for name in ['CoE','CoDE','Density-only']:
            table = binned(decoded[u], decoded[f'{name}__{u}'])
            table['uncertainty'] = u; table['representation'] = name; decbins.append(table)
            score = scores[(scores.representation == name)&(scores.uncertainty == u)].iloc[0]
            draw_bins(ax, table, color=COLORS[name], label=f'{name}: ρ={score.spearman_oof:.2f}')
        lo, hi = decoded[u].min(), decoded[u].max()
        ax.plot([lo,hi],[lo,hi], '--', color='#888888', lw=.8, label='Identity')
        ax.set_xlabel('Observed '+UNC[u]); ax.set_ylabel('Decoded descriptor (out-of-fold)')
        ax.legend(fontsize=8, framealpha=house.DATA_ALPHA)
    fig.suptitle('Uncertainty decoded from learned embeddings · binned observed-versus-predicted\n'
                 'Same folds and complexes; bin-mean CIs condition on fitted OOF predictions', fontsize=13)
    save(fig, 'embedding-uncertainty-binned')
    pd.concat(decbins).to_csv(OUT/'embedding-decoding-bins.csv', index=False)
    protocol = dict(n_affinity=len(df), n_decoding=len(common), excluded_decoding=sorted(set(df.pid)-set(common.pid)),
                    binning='10 equal-count bins, unique quantile edges; ties kept together; equal-width sensitivity plot also supplied',
                    bootstrap='4000 complex resamples per bin, seed 0; assumes independent complexes',
                    input_density='64³ at 0.25 Å; actual v5 arcsinh + pooled z-score input, not a map synthesized from B-factors',
                    map_stats=json.loads((DATA/'voxels_v5/stats.json').read_text()),
                    atom_sampling='Trilinear; grid index=(xyz-ligand heavy-atom centroid)/0.25+32. Protein sites require 0.75 Å margins.',
                    contact_weights='sum 1/max(distance,2Å)^2 over ligand heavy atoms; only protein atoms inside crop with sampling margin',
                    peak_contrast='Density at protein atom minus mean density at ±0.75 Å along three Cartesian axes; weighted by contact weights',
                    sharpness='Mean squared finite-difference gradient / crop density variance; proxy, not calibrated uncertainty',
                    decoding='Five shuffled outer folds (seed 42), three inner folds (100+outer); scaling fit within each training fold; alpha chosen per target by inner MSE',
                    alphas=[.1,1,10,100,1000,10000], selected_alphas=selections, features=provenance,
                    uncertainty_order=list(UNC),
                    limitations=['Exploratory decoding within CL123 test cohort; no affinity-model retraining.',
                                 'Random complex folds are not protein-family-disjoint and do not test transfer to unseen families.',
                                 'CoDE combines coordinates and density; density-only uses a separate encoder, not an isolated CoDE branch.',
                                 'Correlations and decodability do not prove causal use for affinity or calibrated predictive uncertainty.',
                                 'B-factor, occupancy and resolution are coupled; RSCC is model-map agreement, not uncertainty itself.',
                                 'Original cohort selected on RSCC; restricted range; positive association is not independent crystallographic validation.',
                                 'Raw input map preserves crystallographic processing/model bias and is not raw experimental observations.'])
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    readme = ['# Density and uncertainty diagnostics', '',
              'Reproduce: `cd figures && python -m _parts.fig_uncertainty.affinity_density`', '',
              'All 733 CL123-filtered affinity test complexes have actual v5 density maps and all three cached embeddings.',
              f'Decoding uses {len(common)} complete cases, identical across models and targets.', '',
              '## Outputs', '',
              '- affinity-binned-quantile: requested binned error-difference plot; 10 equal-count bins.',
              '- affinity-binned-equal-width: sensitivity to binning; counts expose sparse tails.',
              '- map-uncertainty-correlations: input map-statistic / descriptor rank correlations.',
              '- map-uncertainty-correlations-adjusted: controls for resolution and ligand size.',
              '- map-uncertainty-binned: six map/descriptor relationships as binned means.',
              '- embedding-uncertainty-decoding: nested out-of-fold decoding correlation and R².',
              '- embedding-uncertainty-binned: observed descriptors vs OOF decoded descriptors, binned.',
              '- CSVs preserve every per-complex value, bin statistic, correlation and OOF prediction.',
              '- protocol.json records map formulas, source hashes, selected ridge penalties and limitations.', '',
              '## Interpretation', '',
              'Map statistics are physically motivated proxies, not calibrated uncertainty estimates. High B broadens atomic density; occupancy, resolution and disorder can change the same signal.',
              'Reference: https://www.ccp4.ac.uk/html/edstats.html and https://www.ccp4.ac.uk/html/sfall.html',
              'RSCC measures agreement of a coordinate model with density; it requires both and is not a density-only confidence score.',
              'Representation decoding probes information retained in embeddings. It does not establish that the affinity head uses that information.',
              'CoE and CoDE are the matching caches from the previous affinity comparison; density-only is a separate encoder control.',
              'All target decoding is exploratory within the existing test cohort using held-out folds. It is not an independently held-out benchmark or a protein-family-disjoint result.',
              'Map correlation q-values adjust all 48 pairs. Binned correlations are computed on individual complexes, not bin means.',
              'Bootstrap bars show uncertainty of means across complexes, not predictive uncertainty of individual complexes.', '',
              '## Main observations', '',
              'Contact B versus input-map sharpness: rho=-0.861; contact-weighted protein-atom density: rho=-0.867.',
              'The B/sharpness partial rank correlation remains -0.789 after adjusting resolution and ligand size.',
              'Ligand RSCC versus density at ligand atoms: rho=0.421 (adjusted 0.435).',
              'Nested OOF B decoding rho: CoE 0.607, CoDE 0.914, density-only 0.924.',
              'Within-protein z-B decoding rho: CoE 0.508, CoDE 0.661, density-only 0.601.',
              'Resolution decoding rho: CoE 0.581, CoDE 0.960, density-only 0.965.',
              'Partial-occupancy decoding is weak and does not improve from CoE to CoDE.',
              'The affinity decile plot is not monotonically decreasing: an improvement around B~30 Å² is more prominent than a smooth trend.', '',
              'See protocol.json for full definitions and caveats.', '']
    (OUT/'README.md').write_text('\n'.join(readme))
    print(scores.to_string(index=False), flush=True)
    print(corr[corr.uncertainty.isin(['b_contact_invd2','b_contact_invd2_z','lig_rscc'])].to_string(index=False), flush=True)


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--plots-only',action='store_true');args=ap.parse_args()
    if args.plots_only:redraw_saved_plots()
    else:main()

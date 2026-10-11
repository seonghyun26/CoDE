#!/usr/bin/env python3
"""Restore retained uncertainty figures to the frozen 733-case CL3 test cohort."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from . import update_expanded as expanded
from .paths import ROOT, CL3, REPRESENTATION, RANKING_METRICS, MOTIVATION
from voxbind.splits import load_split, content_hash

DATA = RANKING_METRICS.parent / 'ranking-by-metric-filtered-cl3'
REPR = REPRESENTATION / 'filtered-cl3'
SOURCE = DATA / 'per-complex.csv'
ORIGINAL = ROOT / 'results/task1-affinity/_shared/260919_cdg_v2_5seed'
SCHEME = 'lp_edrscc_v2_cl123'
CUTS = [20., 30.]
PREFIX = 'fixed-contact-b-20-30'


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    REPR.mkdir(parents=True, exist_ok=True)
    original = CL3 / 'per-complex.csv'
    frame = pd.read_csv(original)
    split = load_split(SCHEME)
    assert len(frame) == 733 and frame.pid.is_unique and set(frame.pid) == set(split['test'])
    reports = pd.read_csv(expanded.SOURCE).set_index('pid').loc[frame.pid]
    np.testing.assert_allclose(frame.poc_rscc_median, reports.protein_rscc_median, rtol=0, atol=1e-12)
    frame['protein_rscc_median'] = reports.protein_rscc_median.to_numpy()
    frame['cohort'] = 'original frozen CL3'
    columns = ['y_true'] + [f'{arm}_seed{s}_prediction' for arm in ['CoE','CoDE'] for s in range(5)]
    frame[columns] = frame[columns].astype(np.float32).astype(float)
    rows, prediction_sources = [], []
    for arm in ['CoE', 'CoDE']:
        for seed in range(5):
            path = ORIGINAL / f'{arm}-{SCHEME}-seed{seed}-predictions.csv'
            pred = pd.read_csv(path).set_index('pid').loc[frame.pid]
            np.testing.assert_array_equal(pred.y_true.to_numpy().astype(np.float32), frame.y_true.to_numpy().astype(np.float32))
            np.testing.assert_array_equal(pred.y_pred.to_numpy().astype(np.float32), frame[f'{arm}_seed{seed}_prediction'].to_numpy().astype(np.float32))
            y = frame.y_true.to_numpy(); p = frame[f'{arm}_seed{seed}_prediction'].to_numpy()
            rows.append(dict(arm=arm, seed=seed, n=733, pearson=pearsonr(y,p).statistic,
                             spearman=spearmanr(y,p).statistic, rmse=np.sqrt(np.mean((p-y)**2)), mae=np.mean(abs(p-y))))
            prediction_sources.append(dict(path=str(path), sha256=expanded.sha(path)))
    frame.to_csv(SOURCE,index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(DATA/'overall-per-seed.csv',index=False)
    summary = metrics.groupby('arm',sort=False)[['pearson','spearman','rmse','mae']].mean().reset_index()
    summary.to_csv(DATA/'overall-summary.csv',index=False)
    print('Frozen 733-case CPU probe metrics:\n'+summary.to_string(index=False),flush=True)

    # Reuse decoders fit on the original complete-case cohort, never subset the 1162-case OOF fits.
    legacy_keys = list(expanded.UNC)
    complete = frame.dropna(subset=legacy_keys).reset_index(drop=True)
    reference_path = REPRESENTATION/'oof-predictions.csv'
    reference = pd.read_csv(reference_path)
    assert len(reference)==732 and reference.pid.equals(complete.pid)
    np.testing.assert_allclose(reference[legacy_keys],complete[legacy_keys],rtol=0,atol=1e-10)
    decoded = reference.copy()
    score_tables=[pd.read_csv(REPRESENTATION/'decoding-statistics.csv')]
    decoding_sources=[dict(path=str(reference_path),sha256=expanded.sha(reference_path))]
    for arm in ['DeepDTA','ProFSA','Nesso-1']:
        folder=REPRESENTATION/'baselines'/arm
        path=folder/'oof-predictions.csv'; baseline=pd.read_csv(path)
        assert baseline.pid.equals(reference.pid) and baseline.fold.equals(reference.fold)
        np.testing.assert_allclose(baseline[legacy_keys],reference[legacy_keys],rtol=0,atol=1e-10)
        for key in legacy_keys:decoded[f'{arm}__{key}']=baseline[f'{arm}__{key}']
        score_tables.append(pd.read_csv(folder/'decoding-statistics.csv'))
        decoding_sources.append(dict(path=str(path),sha256=expanded.sha(path),protocol=str(folder/'decoder.json')))
    scores=pd.concat(score_tables,ignore_index=True)
    assert len(scores)==40 and set(scores.arm)==set(expanded.ARMS) and scores.n.eq(732).all()
    for row in scores.itertuples():
        value=spearmanr(decoded[row.descriptor],decoded[f'{row.arm}__{row.descriptor}']).statistic
        np.testing.assert_allclose(value,row.rho,rtol=0,atol=1e-12)
    scores.descriptor=scores.descriptor.replace({'poc_rscc_median':'protein_rscc_median'})
    decoded=decoded.rename(columns={key:key.replace('poc_rscc_median','protein_rscc_median') for key in decoded.columns})
    scores.to_csv(REPR/'decoding-statistics.csv',index=False)
    decoded.to_csv(REPR/'oof-predictions.csv',index=False)
    (REPR/'provenance.json').write_text(json.dumps(dict(n=732,source_sha256=expanded.sha(SOURCE),
        sources=decoding_sources,protocol='original common 732 complete cases; identical nested 5x3 ridge folds; no expanded-cohort OOF predictions reused',
        protein_rscc='verified identical to the historical polymer target on all 733 cases'),indent=2)+'\n')
    provenance=dict(cohort='frozen filtered CL3',n_affinity=733,n_decoding=732,scheme=SCHEME,
        split_counts={k:len(v) for k,v in split.items()},split_hashes={k:content_hash(v) for k,v in split.items()},
        source=str(original),source_sha256=expanded.sha(original),prepared_source_sha256=expanded.sha(SOURCE),
        prediction_sources=prediction_sources,decoding_sources=decoding_sources,
        affinity_protocol='existing five-seed CPU rerun, raw 640-D frozen features, scalar MSE head, original train/validation split, validation Spearman early stopping',
        historical_manuscript=dict(source='the manuscript table, Table 1a CL123 CDG v2',pearson=.645,spearman=.621,rmse=1.387,
            note='0.645 is Pearson, not Spearman. Current CPU rerun has mean Spearman 0.620937 and RMSE 1.405872; it is not a bitwise reproduction of the historical run. The remaining RMSE discrepancy has not been attributed to a specific cause.'),
        excluded_from_current_figures='all 430 exploratory additional cases; original and expanded tables remain separate')
    (DATA/'cohort-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def redraw_filtered(only=None):
    provenance=json.loads((DATA/'cohort-provenance.json').read_text())
    assert provenance['prepared_source_sha256']==expanded.sha(SOURCE)
    assert provenance['source_sha256']==expanded.sha(CL3/'per-complex.csv')
    assert provenance['split_hashes']['test']==content_hash(load_split(SCHEME)['test'])
    return expanded.redraw_current_figures(only=only,analysis=dict(source=str(SOURCE),data=str(DATA),
        representation=str(REPR),n_affinity=733,cohort='frozen filtered CL3',
        overall_statistics=str(DATA/'overall-summary.csv'),cohort_provenance=str(DATA/'cohort-provenance.json'),
        generator='cd figures && python -m _parts.fig_uncertainty.update_filtered --plots-only'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    manifest_path=MOTIVATION/'figure-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    binning=manifest.get('contact_b_binning','absolute_20_30')
    if not args.plots_only:
        prepare()
        expanded.analyse_affinity(source=SOURCE,out=DATA,expected_n=733)
        if binning=='quartiles':
            from .analyse_quartiles import analyse_quartiles
            analyse_quartiles(SOURCE,DATA,expected_n=733)
        elif binning!='terciles':
            from .analyse_fixed_b import analyse
            choices={'absolute_20_25_30':([20.,25.,30.],'fixed-contact-b-20-25-30'),
                     'absolute_20_30':([20.,30.],'fixed-contact-b-20-30'),
                     'absolute_15_30':([15.,30.],'fixed-contact-b-15-30'),
                     'absolute_20_40':([20.,40.],'fixed-contact-b'),
                     'absolute_10_20_30':([10.,20.,30.],'fixed-contact-b-10-20-30')}
            cuts,prefix=choices[binning]
            analyse(cuts,prefix,source=SOURCE,out=DATA,expected_n=733)
    manifest['cohort']='frozen_filtered_cl3'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    redraw_filtered()


if __name__=='__main__':main()

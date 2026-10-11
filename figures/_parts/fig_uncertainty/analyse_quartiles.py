#!/usr/bin/env python3
"""Contact-B quartiles: per-model Spearman, paired differences and relative changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .analyse_fixed_b import analyse
from .paths import MOTIVATION

PREFIX='contact-b-quartiles'


def analyse_quartiles(source,out,expected_n=733):
    source,out=Path(source),Path(out)
    frame=pd.read_csv(source)
    assert len(frame)==expected_n and frame.pid.is_unique
    cuts=np.quantile(frame.b_contact_invd2.to_numpy(),[.25,.5,.75])
    analyse(cuts,PREFIX,source=source,out=out,expected_n=expected_n)
    for kind in ['ranking','rmse']:
        path=out/f'{PREFIX}-{kind}.csv';table=pd.read_csv(path)
        table['binning']='empirical quartiles; ties kept together; thresholds fixed during bootstrap'
        table['quartile']=['Q1','Q2','Q3','Q4']
        if kind=='ranking':table['relative_spearman_change_pct']=100*table.estimate/table.coe_spearman
        table.to_csv(path,index=False)
    path=out/f'{PREFIX}-provenance.json';meta=json.loads(path.read_text())
    meta.update(selection='User-requested empirical Contact-B quartiles after earlier binning exploration; cohort quantiles, no performance-based cutoff optimization.',
        quantiles=[.25,.5,.75],binning='quartiles; equal values kept together',
        relative_spearman_change_pct='100 * (mean_seed_spearman_CoDE - mean_seed_spearman_CoE) / mean_seed_spearman_CoE',
        interval_scope='pointwise 95% paired-complex percentile intervals for each model mean Spearman and their difference; cutoffs and fitted seeds fixed; no family or multiplicity adjustment',
        reproduce=f'cd figures && python -m _parts.fig_uncertainty.analyse_quartiles --source {source} --out {out} --expected-n {expected_n}')
    path.write_text(json.dumps(meta,indent=2)+'\n')
    table=pd.read_csv(out/f'{PREFIX}-ranking.csv')
    print(table[['quartile','n','x_median','coe_spearman','coe_ci_lo','coe_ci_hi','code_spearman','code_ci_lo','code_ci_hi','relative_spearman_change_pct']].to_string(index=False),flush=True)
    return table


def main():
    active=json.loads((MOTIVATION/'active-analysis.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(active['source']))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--expected-n',type=int,default=active['n_affinity'])
    args=parser.parse_args()
    analyse_quartiles(args.source,args.out or args.source.parent,args.expected_n)


if __name__=='__main__':main()

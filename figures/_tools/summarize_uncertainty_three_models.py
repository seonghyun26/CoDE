#!/usr/bin/env python3
"""Three requested models, evaluated within identical contact-B terciles."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # figures/: draw + _parts
from _parts.fig_uncertainty.paths import ATOMIC_B as FIG, CL3
OLD=ROOT/'results/task1-affinity/_shared/260919_cdg_v2_5seed'
NEW=ROOT/'results/task1-affinity/_shared/260920_atomic_b_spatial'
f=pd.read_csv(FIG/'per-complex.csv').set_index('pid').sort_index()
x=f.b_contact_invd2.to_numpy();y=f.y_true.to_numpy();cuts=np.quantile(x,[1/3,2/3]);g=np.searchsorted(cuts,x,side='right')
assert len(f)==733 and list(np.bincount(g))==[244,244,245]
models={};rows=[];seedrows=[];rng=np.random.default_rng(20260920)
for arm in ['CoE','atomic-b','CoDE']:
 ps=[]
 for seed in range(5):
  path=NEW/f'{arm}-seed{seed}-test-predictions.csv' if arm=='atomic-b' else OLD/f'{arm}-lp_edrscc_v2_cl123-seed{seed}-predictions.csv'
  df=pd.read_csv(path).set_index('pid').sort_index();assert df.index.equals(f.index) and np.allclose(df.y_true,y,atol=1e-6)
  ps.append(df.y_pred.to_numpy())
 models[arm]=np.array(ps)
for label,mask in [('Low',g==0),('Middle',g==1),('High',g==2),('All',np.ones(len(g),bool))]:
 errs={a:abs(p[:,mask]-y[mask]) for a,p in models.items()}
 rhos={a:np.array([spearmanr(y[mask],pred[mask]).statistic for pred in p]) for a,p in models.items()}
 for arm in models:
  maes=errs[arm].mean(1);ds=(errs[arm]-errs['CoE']).mean(0);dr=rhos[arm]-rhos['CoE']
  bs=np.concatenate([ds[rng.integers(len(ds),size=(500,len(ds)))].mean(1) for _ in range(16)])
  lo,hi=np.quantile(bs,[.025,.975])
  rows.append(dict(group=label,n=mask.sum(),b_min=x[mask].min(),b_max=x[mask].max(),arm=arm,
   mae=maes.mean(),mae_seed_sd=maes.std(),delta_mae_vs_CoE=ds.mean(),delta_mae_complex_ci_lo=lo,delta_mae_complex_ci_hi=hi,
   spearman=rhos[arm].mean(),spearman_seed_sd=rhos[arm].std(),delta_spearman_vs_CoE=dr.mean()))
  for seed in range(5):seedrows.append(dict(group=label,arm=arm,seed=seed,n=mask.sum(),mae=maes[seed],spearman=rhos[arm][seed]))
df=pd.DataFrame(rows);df.to_csv(FIG/'three-model-uncertainty-table.csv',index=False);pd.DataFrame(seedrows).to_csv(FIG/'three-model-uncertainty-per-seed.csv',index=False)
print(df.round(5).to_string(index=False))
labels={'CoE':'CoE','atomic-b':'CoE + atomwise pocket B (spatial CNN)','CoDE':'CoDE'}
text=['# CoE, CoE + atomwise B, and CoDE by contact-B uncertainty proxy','',
 'CL123 train/validation, CL3 test: 733 complexes; five probe seeds. Uncertainty proxy: contact-weighted pocket B-factor, higher values indicating greater disorder, not calibrated predictive uncertainty. Tercile cut points: '+', '.join(f'{v:.6f}' for v in cuts)+' Å².',
 '', 'Each MAE is the mean of seed-specific MAEs, not the MAE of ensemble predictions. Spearman is calculated within each subgroup separately for each seed, then averaged. Parentheses show the change relative to CoE in the same subgroup. Negative ΔMAE or positive ΔSpearman favors the augmented model.', '']
for metric,delta,title in [('mae','delta_mae_vs_CoE','MAE (pK), lower is better'),('spearman','delta_spearman_vs_CoE','Spearman, higher is better')]:
 text += ['## '+title,'','| B group | n | CoE | CoE + atomwise B | CoDE |','|---|---:|---:|---:|---:|']
 for group in ['Low','Middle','High','All']:
  block=df[df.group==group].set_index('arm');cells=[]
  for arm in labels:
   r=block.loc[arm];cells.append(f'{r[metric]:.3f}'+(f' ({r[delta]:+.3f})' if arm!='CoE' else ''))
  text.append('| '+group+' | '+str(int(block.iloc[0]['n']))+' | '+' | '.join(cells)+' |')
 text.append('')
text += ['The atomwise-B arm is a compact supervised 1 Å spatial CNN branch added to frozen CoE, using pocket B only. It is not B-aware pretraining or a ligand-B experiment. Occupancy-only and shuffled-B matched controls remain in the accompanying full experiment even though only the three requested arms are tabulated here.',
 '', 'Bootstrap intervals in the CSV resample complexes conditional on the five fitted seeds, without protein-family clustering or multiplicity adjustment. They do not capture all training variation. Comparing Spearman magnitudes across B groups can be affected by their different affinity distributions. The manuscript encoders also differ in masking recipe, so this comparison does not isolate density as the cause of an effect.', '']
(FIG/'three-model-uncertainty-table.md').write_text('\n'.join(text))

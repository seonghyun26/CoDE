"""Exploratory review of all previously computed descriptors; no probe retraining."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # figures/: draw + _parts
from _parts.fig_uncertainty.paths import METRIC_REVIEW as OUT, MAP_DIAGNOSTICS, CL3, LP_CL3
OUT.mkdir(parents=True,exist_ok=True)
DIRECTIONS={
 'b_contact_invd2':1,'b_contact_gauss':1,'b_contact_count':1,
 'b_contact_invd2_z':1,'b_contact_shell_mean':1,'rel_pocket_b':1,
 'lig_rscc':-1,'poc_rscc_median':-1,'occ_site_frac_partial':1,
 'occ_site_frac_altloc':1,'resolution':1,
 'map_std':-1,'map_p95':-1,'map_sharpness':-1,'map_contact_density':-1,
 'map_contact_peak_contrast':-1,'map_ligand_density':-1}

def bh(p):
 p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p))
 q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
 return q

def compare(a,b,seed):
 rng=np.random.default_rng(seed);boots=[];null=[];pool=np.r_[a,b];obs=a.mean()-b.mean()
 for _ in range(16):
  boots.extend((a[rng.integers(len(a),size=(500,len(a)))].mean(1)-b[rng.integers(len(b),size=(500,len(b)))].mean(1)).tolist())
 for _ in range(8000):
  p=rng.permutation(pool);null.append(p[:len(a)].mean()-p[len(a):].mean())
 lo,hi=np.quantile(boots,[.025,.975]);p=(1+np.count_nonzero(np.abs(null)>=abs(obs)))/8001
 return obs,lo,hi,p

rows=[];bins=[];seedrows=[]
maps=pd.read_csv(MAP_DIAGNOSTICS/'map-statistics.csv')
for scheme in ['cl123-trained','lp-trained-cl3-filtered']:
 f=pd.read_csv(({'cl123-trained':CL3,'lp-trained-cl3-filtered':LP_CL3}[scheme])/'per-complex.csv').merge(maps,on='pid',validate='one_to_one')
 assert len(f)==733 and f.pid.is_unique
 for j,(key,direction) in enumerate(DIRECTIONS.items()):
  d=f.dropna(subset=[key,'delta_vs_CoE']).copy();x=d[key].to_numpy();y=d.delta_vs_CoE.to_numpy();oriented=direction*x
  rho,p=stats.spearmanr(x,y)
  q1,q2=np.quantile(oriented,[1/3,2/3])
  if q1==q2:
   group=(oriented>q2).astype(int);high=group==1;kind='nonzero versus zero';labels=['zero','nonzero']
  else:
   group=np.searchsorted([q1,q2],oriented,side='right');high=group==2;kind='upper uncertainty-proxy tercile versus rest';labels=['lower','middle','higher']
  obs,lo,hi,perm=compare(y[high],y[~high],19092026+j)
  controls=[c for c in ['resolution','n_ligand_heavy_atoms'] if c!=key]
  ranks=d[[key,'delta_vs_CoE']+controls].rank().to_numpy();z=np.c_[np.ones(len(d)),ranks[:,2:]]
  residual=ranks[:,:2]-z@np.linalg.lstsq(z,ranks[:,:2],rcond=None)[0]
  partial=np.corrcoef(residual.T)[0,1]
  rows.append(dict(scheme=scheme,descriptor=key,n=len(d),higher_proxy_direction=direction,
    spearman_raw_x_delta=rho,spearman_p=p,partial_spearman_resolution_size=partial,
    contrast_kind=kind,n_higher=high.sum(),higher_minus_rest=obs,ci_lo=lo,ci_hi=hi,contrast_p=perm))
  for g in np.unique(group):
   keep=group==g
   bins.append(dict(scheme=scheme,descriptor=key,uncertainty_proxy_group=labels[g],n=keep.sum(),
      x_min=x[keep].min(),x_max=x[keep].max(),mean_delta=y[keep].mean()))
  for seed in range(5):
   delta=abs(d[f'CoDE_seed{seed}_prediction']-d.y_true)-abs(d[f'CoE_seed{seed}_prediction']-d.y_true)
   seedrows.append(dict(scheme=scheme,descriptor=key,seed=seed,higher_minus_rest=delta[high].mean()-delta[~high].mean()))
summary=pd.DataFrame(rows)
for scheme in summary.scheme.unique():
 sel=summary.scheme==scheme
 summary.loc[sel,'spearman_q_bh17']=bh(summary.loc[sel,'spearman_p'])
 summary.loc[sel,'contrast_q_bh17']=bh(summary.loc[sel,'contrast_p'])
summary['spearman_q_bh34']=bh(summary.spearman_p)
summary['contrast_q_bh34']=bh(summary.contrast_p)
summary.to_csv(OUT/'all-descriptor-review.csv',index=False)
pd.DataFrame(bins).to_csv(OUT/'oriented-bins.csv',index=False)
pd.DataFrame(seedrows).to_csv(OUT/'seed-contrasts.csv',index=False)
print(summary[summary.scheme=='cl123-trained'].round(4).to_string(index=False))
print(pd.DataFrame(bins).query("scheme == 'cl123-trained' and descriptor in ['b_contact_invd2','map_sharpness','map_contact_peak_contrast']").round(4).to_string(index=False))
(OUT/'README.md').write_text('''# Exploratory metric review, 2026-09-20

Uses the saved manuscript 5-seed CoDE and CoE affinity predictions, with all 17 previously available crystallographic or density-map descriptors. No fitting, hyperparameter selection, or model selection is performed. Map statistics come from the original cached 64-cubed v5 density inputs for the same 733 complexes; old model predictions are not used.

Outcome is per-complex mean-over-seeds absolute-error(CoDE) minus absolute-error(CoE), in pK units. Negative favors CoDE. Spearman uses raw descriptor values. For descriptive bin contrasts, higher B, resolution in Angstroms, occupancy/altloc fractions, and lower RSCC/density summaries are oriented toward a hypothesized higher-uncertainty direction. These directions are proxies, not a measured or calibrated uncertainty scale; density gradients can also respond to noise and processing. Zero-inflated occupancy/altloc fractions use nonzero-versus-zero groups, rather than invalid terciles. Tied continuous values remain together.

8000 complex-bootstrap replicates and 8000 random label permutations per contrast, conditional on the existing seeds, without protein clustering. BH correction across all 17 descriptors per protocol and all 34 across both protocols is reported separately for Spearman and bin contrasts. Metrics were inspected on the test cohort: any suggested metric is exploratory, requiring independent confirmation. Partial rank correlations remove linear effects of ranked resolution and ligand size; they are descriptive, not causal adjustments. Probe seed variation is exported separately.

The strongest-looking raw correlation should not be called a validated winning uncertainty metric. In particular, a scalar diagnostic that stratifies examples can itself be concatenated; evidence for retaining its full spatial field requires a matched descriptor-input control.
''')

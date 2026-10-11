#!/usr/bin/env python3
"""Paired complex bootstrap of seed-averaged Spearman gaps in fixed B strata."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # figures/: draw + _parts
from _parts.fig_uncertainty.paths import ATOMIC_B as OUT, CL3
f=pd.read_csv(OUT/'per-complex.csv');y=f.y_true.to_numpy();x=f.b_contact_invd2.to_numpy();groups=np.searchsorted(np.quantile(x,[1/3,2/3]),x,side='right')
p={a:np.stack([f[f'{a}_seed{s}_prediction'].to_numpy() for s in range(5)]) for a in ['CoE','CoDE']}

def corr(pred,truth):
 rp=rankdata(pred,axis=-1);ry=rankdata(truth,axis=-1)
 rp-=rp.mean(-1,keepdims=True);ry-=ry.mean(-1,keepdims=True)
 return (rp*ry).sum(-1)/np.sqrt((rp*rp).sum(-1)*(ry*ry).sum(-1))

rng=np.random.default_rng(20260920);rows=[];boots={}
for g,label in enumerate(['Low','Middle','High']):
 sel=np.where(groups==g)[0];n=len(sel);a=p['CoE'][:,sel];b=p['CoDE'][:,sel];t=y[sel]
 point=float((corr(b,t[None])-corr(a,t[None])).mean());values=[]
 for batch in range(16):
  idx=rng.integers(n,size=(500,n));truth=t[idx][:,None,:]
  av=a[:,idx].transpose(1,0,2);bv=b[:,idx].transpose(1,0,2)
  values.extend((corr(bv,truth)-corr(av,truth)).mean(1).tolist())
 values=np.array(values);boots[label]=values;lo,hi=np.quantile(values,[.025,.975])
 rows.append(dict(comparison='CoDE minus CoE Spearman',group=label,n=n,estimate=point,ci_lo=lo,ci_hi=hi))
point=rows[2]['estimate']-rows[0]['estimate'];contrast=boots['High']-boots['Low'];lo,hi=np.quantile(contrast,[.025,.975])
rows.append(dict(comparison='High minus Low improvement',group='High-Low',n=489,estimate=point,ci_lo=lo,ci_hi=hi))
df=pd.DataFrame(rows);df.to_csv(OUT/'ranking-gap-complex-bootstrap.csv',index=False);print(df.to_string(index=False))
(OUT/'ranking-gap-complex-bootstrap.md').write_text('''# Exploratory ranking-gap uncertainty

Paired bootstrap of complexes within fixed contact-B terciles; 8000 replicates. Each replicate computes Spearman separately for all five fitted seeds and averages the CoDE-minus-CoE differences. Percentile 95% intervals are conditional on fitted seeds, B thresholds, and the present CL3 test cohort. The high-minus-low contrast independently resamples both strata. No protein-family clustering, seed resampling, or multiple-testing adjustment is performed. This is exploratory post-hoc analysis, not confirmation of an uncertainty-specific mechanism. Different affinity distributions across strata and pretraining-recipe differences remain possible explanations.
''')

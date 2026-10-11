#!/usr/bin/env python3
"""Summarize the atomic B voxel experiment, keeping the spatial controls explicit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import draw as house
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'results/task1-affinity/_shared/260920_atomic_b_spatial'
OLD=ROOT/'results/task1-affinity/_shared/260919_cdg_v2_5seed'
from .paths import ATOMIC_B as OUT, CL3
OUT.mkdir(parents=True,exist_ok=True)
LABELS={'CoE':'CoE','CoE-Bscalar':'CoE + scalar B','CoE-Bvector':'CoE + 4 B summaries',
 'occupancy':'CoE + spatial occupancy','atomic-b':'CoE + spatial occupancy + atomic B',
 'shuffled-b':'CoE + spatial occupancy + shuffled B','CoDE':'CoDE'}

def save(fig,name):
 if not house.prepare_publication_figure(fig,OUT/name):
  plt.close(fig);return
 for ext in house.retained_formats(OUT/name):fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)

def bootstrap_mean(x,rng):
 b=np.concatenate([x[rng.integers(len(x),size=(500,len(x)))].mean(1) for _ in range(16)])
 return np.quantile(b,[.025,.975])

def plot_comparison(metrics):
 order=['CoE','CoE-Bvector','occupancy','atomic-b','shuffled-b','CoDE'];colors=[house.pale('CoE'),house._gentle_color('#a7b86b'),'#888888',house._gentle_color('#da9035'),house._gentle_color('#ae7bac'),house.pale('CoDE')]
 labels=['CoE','CoE + 4 B\nsummaries','CoE + spatial\noccupancy','CoE + spatial\natomic B','CoE + spatial\nshuffled B','CoDE']
 fig,axs=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
 for ax,metric,title in zip(axs,['spearman','mae'],['Spearman ρ ↑','MAE (pK) ↓']):
  for i,arm in enumerate(order):
   vals=metrics[metrics.arm==arm][metric].to_numpy();ax.scatter(i+np.linspace(-.10,.10,5),vals,s=23,color=colors[i],alpha=house.DATA_ALPHA)
   ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=0),fmt='D',color=colors[i],capsize=5,ms=6)
  ax.set_xticks(range(len(order)),labels,rotation=25,ha='right');ax.set_ylabel(title);ax.grid(axis='y',alpha=house.BACKGROUND_ALPHA)
 fig.suptitle('Atomwise pocket B-factor at atom-occupied voxel positions\nCL123 train/val → CL3 test (n=733); 5 probe seeds, means ± SD',fontsize=13)
 save(fig,'atomic-b-spatial-comparison')

def plot_saved_terciles():
 table=pd.read_csv(OUT/'terciles.csv')
 fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
 for ax,(a,b) in zip(axs,[('atomic-b','occupancy'),('CoDE','atomic-b')]):
  sub=table[table.comparison==f'{a} minus {b}'].sort_values('tercile')
  for row in sub.itertuples():
   ax.errorbar(row.tercile-1,row.mean,yerr=[[row.mean-row.ci_lo],[row.ci_hi-row.mean]],fmt='o',color=house.pale('CoDE'),capsize=5)
  ax.axhline(0,c='#555555',lw=.8);ax.set_xticks(range(3),['Low B','Middle B','High B']);ax.set_title(LABELS[a]+'\nminus '+LABELS[b],fontsize=10);ax.set_ylabel('Δ mean absolute pK error');ax.grid(axis='y',alpha=house.BACKGROUND_ALPHA)
 fig.suptitle('Contact B terciles; 95% complex-bootstrap CI conditional on five seeds\nNegative favors the first method',fontsize=11);save(fig,'atomic-b-contact-terciles')

def plot_input_example():
 ids=json.loads((SRC/'ids.json').read_text());g=np.load(SRC/'spatial-grids.npy',mmap_mode='r')[ids.index('1bcu')].astype(float);proto=json.loads((SRC/'grid-protocol.json').read_text())
 occupancy=g[7:11].sum((0,3));lig=g[:7].sum((0,3));fields=[]
 for sl in [slice(11,15),slice(15,19)]:
  field=g[sl].sum((0,3));mean=np.divide(field,occupancy,out=np.full_like(field,np.nan),where=occupancy>0.03)
  fields.append(mean*proto['b_std_train_atoms']+proto['b_mean_train_atoms'])
 fig,axs=plt.subplots(1,3,figsize=(12,4.3),layout='constrained');vmin=min(np.nanmin(x) for x in fields);vmax=max(np.nanmax(x) for x in fields)
 im=axs[0].imshow(occupancy.T,origin='lower',extent=[-8,8,-8,8],cmap='Greys');fig.colorbar(im,ax=axs[0],shrink=.7,label='Projected atom mass')
 for ax,field in zip(axs[1:],fields):
  im=ax.imshow(field.T,origin='lower',extent=[-8,8,-8,8],cmap='viridis',vmin=vmin,vmax=vmax);fig.colorbar(im,ax=ax,shrink=.7,label='Projected mean B (Å²)')
 for ax,title in zip(axs,['Pocket atom occupancy','B at the correct atom positions','B shuffled within each element']):
  ax.contour(np.linspace(-7.5,7.5,16),np.linspace(-7.5,7.5,16),lig.T,levels=[.1],colors=['#e66101'],linewidths=.8);ax.set_title(title,fontsize=10);ax.set_xlabel('x (Å)');ax.set_ylabel('y (Å)')
 fig.suptitle('Input example: 1bcu · z projection of 1 Å auxiliary grids\nOrange: ligand occupancy; training receives full 3D fields',fontsize=12);save(fig,'atomic-b-input-example')


def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--plots-only',action='store_true');args=ap.parse_args()
 if args.plots_only:
  plot_comparison(pd.read_csv(OUT/'per-seed-metrics.csv'))
  plot_saved_terciles()
  plot_input_example()
  return
 frame=pd.read_csv(CL3/'per-complex.csv').set_index('pid').sort_index()
 assert len(frame)==733 and frame.index.is_unique
 metrics=[];errors={};preds={};y=frame.y_true.to_numpy();seedrows=[]
 for arm in LABELS:
  e=[];ps=[]
  for seed in range(5):
   if arm in ['occupancy','atomic-b','shuffled-b']:
    path=SRC/f'{arm}-seed{seed}-test-predictions.csv';record=json.loads((SRC/f'{arm}-seed{seed}.json').read_text());reported=record['test']
   else:
    path=OLD/f'{arm}-lp_edrscc_v2_cl123-seed{seed}-predictions.csv';record=json.loads((OLD/f'{arm}-lp_edrscc_v2_cl123-seed{seed}-metrics.json').read_text());reported=record['cl3_test']
   df=pd.read_csv(path).set_index('pid').sort_index()
   assert df.index.equals(frame.index) and np.allclose(df.y_true,y,atol=1e-6)
   p=df.y_pred.to_numpy();assert np.isfinite(p).all();ae=abs(p-y);rho=float(spearmanr(y,p).statistic)
   assert np.isclose(ae.mean(),reported['mae'],atol=1e-6) and np.isclose(rho,reported['spearman'],atol=1e-6)
   metrics.append(dict(arm=arm,seed=seed,spearman=rho,mae=ae.mean(),rmse=np.sqrt(((p-y)**2).mean()),best_val_spearman=record['best_val_spearman']))
   e.append(ae);ps.append(p)
  errors[arm]=np.array(e);preds[arm]=np.array(ps)
  frame[f'{arm}_mean_seed_abs_error']=errors[arm].mean(0)
 metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'per-seed-metrics.csv',index=False)
 summary=metrics.groupby('arm')[['spearman','mae','rmse','best_val_spearman']].agg(['mean',lambda x:x.std(ddof=0)]).rename(columns={'<lambda_0>':'std'})
 summary.to_csv(OUT/'summary.csv');print(summary.to_string(),flush=True)
 contrasts=[];rng=np.random.default_rng(0)
 for a,b in [('atomic-b','occupancy'),('atomic-b','shuffled-b'),('atomic-b','CoE'),('CoDE','atomic-b')]:
  d=(errors[a]-errors[b]).mean(0);lo,hi=bootstrap_mean(d,rng)
  contrasts.append(dict(comparison=f'{a} minus {b}',mean_delta_mae=d.mean(),complex_ci_lo=lo,complex_ci_hi=hi,
    mean_delta_spearman=summary.loc[a,('spearman','mean')]-summary.loc[b,('spearman','mean')]))
  frame[f'delta_{a}_minus_{b}']=d
  for seed in range(5):seedrows.append(dict(comparison=f'{a} minus {b}',seed=seed,delta_mae=(errors[a][seed]-errors[b][seed]).mean(),delta_spearman=spearmanr(y,preds[a][seed]).statistic-spearmanr(y,preds[b][seed]).statistic))
 pd.DataFrame(contrasts).to_csv(OUT/'paired-contrasts.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/'paired-seed-contrasts.csv',index=False);frame.to_csv(OUT/'per-complex.csv')
 plot_comparison(metrics)
 fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained');bins=[];x=frame.b_contact_invd2.to_numpy();groups=np.searchsorted(np.quantile(x,[1/3,2/3]),x,side='right')
 for ax,(a,b) in zip(axs,[('atomic-b','occupancy'),('CoDE','atomic-b')]):
  d=(errors[a]-errors[b]).mean(0)
  for group in range(3):
   v=d[groups==group];lo,hi=bootstrap_mean(v,rng);mean=v.mean();bins.append(dict(comparison=f'{a} minus {b}',tercile=group+1,n=len(v),mean=mean,ci_lo=lo,ci_hi=hi))
   ax.errorbar(group,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=house.pale('CoDE'),capsize=5)
  ax.axhline(0,c='#555555',lw=.8);ax.set_xticks(range(3),['Low B','Middle B','High B']);ax.set_title(LABELS[a]+'\nminus '+LABELS[b],fontsize=10);ax.set_ylabel('Δ mean absolute pK error');ax.grid(axis='y',alpha=house.BACKGROUND_ALPHA)
 fig.suptitle('Contact B terciles; 95% complex-bootstrap CI conditional on five seeds\nNegative favors the first method',fontsize=11);save(fig,'atomic-b-contact-terciles');pd.DataFrame(bins).to_csv(OUT/'terciles.csv',index=False)
 plot_input_example()
 print(pd.DataFrame(contrasts).to_string(index=False),flush=True)

if __name__=='__main__':main()

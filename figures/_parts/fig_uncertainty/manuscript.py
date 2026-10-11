#!/usr/bin/env python3
"""Plots from the manuscript CDG-v2/CoE five-seed probes and B-concat controls."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import draw as house
from scipy import stats
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'figures/_parts/fig_uncertainty'));import uncertainty_descriptors as ud
sys.path.insert(0,str(ROOT/'figures'))
from .affinity_cl3 import METRICS
from .affinity_density import binned,draw_bins,decode,UNC
SRC=ROOT/'results/task1-affinity/_shared/260919_cdg_v2_5seed'
from .paths import AFFINITY as OUT, CL3, LP_CL3, REPRESENTATION, STANDARDIZED_B, EXPLORATORY_CL3
ARMS=['CoE','CoE-Bscalar','CoE-Bvector','CoDE']
COLORS=[house.pale('CoE'),house._gentle_color('#9dc55b'),house._gentle_color('#dd913e'),house.pale('CoDE')]
LABELS={'CoE':'CoE','CoE-Bscalar':'CoE + contact B','CoE-Bvector':'CoE + 4 B descriptors','CoDE':'CoDE (CDG v2)'}

def save(fig,path):
 if not house.prepare_publication_figure(fig,path):
  plt.close(fig);return
 for ext in house.retained_formats(path):fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)

def analyse(scheme,saved_only=False):
 out=CL3 if scheme.endswith('cl123') else LP_CL3;out.mkdir(parents=True,exist_ok=True)
 if saved_only:
  frame=pd.read_csv(out/'per-complex.csv').set_index('pid')
  summary=pd.read_csv(out/'per-seed-metrics.csv')
  corr=pd.read_csv(out/'descriptor-statistics.csv')
  bins=pd.read_csv(out/'terciles.csv')
  agg=summary.groupby('arm')[['pearson','spearman','rmse','mae']].agg(['mean',lambda x: x.std(ddof=0)]).rename(columns={'<lambda_0>':'std'})
 else:
  old=pd.read_csv(EXPLORATORY_CL3/'per-complex.csv').set_index('pid').sort_index()
  frame=old[[*METRICS,*[k for k in UNC if k not in METRICS],'y_true','n_ligand_heavy_atoms']].copy()
  summaries=[];arrays={}
  for arm in ARMS:
   preds=[]
   for seed in range(5):
    p=pd.read_csv(SRC/f'{arm}-{scheme}-seed{seed}-predictions.csv').set_index('pid')
    assert p.index.is_unique and set(frame.index)<=set(p.index)
    p=p.loc[frame.index];assert np.allclose(p.y_true,frame.y_true,atol=1e-6)
    pred=p.y_pred.to_numpy();preds.append(pred);frame[f'{arm}_seed{seed}_prediction']=pred
    e=pred-frame.y_true.to_numpy()
    summaries.append(dict(arm=arm,seed=seed,n=len(e),pearson=stats.pearsonr(pred,frame.y_true).statistic,
       spearman=stats.spearmanr(pred,frame.y_true).statistic,rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e))))
   arrays[arm]=np.asarray(preds)
   frame[f'{arm}_mean_seed_abs_error']=np.abs(arrays[arm]-frame.y_true.to_numpy()).mean(0)
   frame[f'{arm}_ensemble_abs_error']=np.abs(arrays[arm].mean(0)-frame.y_true.to_numpy())
  for baseline in ARMS[:-1]:
   frame[f'delta_vs_{baseline}']=frame.CoDE_mean_seed_abs_error-frame[f'{baseline}_mean_seed_abs_error']
  frame['delta_ensemble_vs_CoE']=frame.CoDE_ensemble_abs_error-frame.CoE_ensemble_abs_error
  frame.to_csv(out/'per-complex.csv')
  summary=pd.DataFrame(summaries);summary.to_csv(out/'per-seed-metrics.csv',index=False)
  agg=summary.groupby('arm')[['pearson','spearman','rmse','mae']].agg(['mean',lambda x: x.std(ddof=0)]).rename(columns={'<lambda_0>':'std'})
  agg.to_csv(out/'metrics-summary.csv')
  tables=[];correlations=[];sensitivity=[]
  for baseline in ARMS[:-1]:
   y=frame[f'delta_vs_{baseline}'].to_numpy()
   for key in METRICS:
    x=frame[key].to_numpy();block=ud.tercile_split(x,y);c=block['contrast'];rho,p=stats.spearmanr(x,y)
    correlations.append(dict(baseline=baseline,descriptor=key,n=len(y),mean_delta=y.mean(),spearman=rho,spearman_p=p,
      top_minus_rest=c['top_minus_rest'],ci_lo=c['ci_lo'],ci_hi=c['ci_hi'],p_permutation=(round(c['p_permutation']*8000)+1)/8001))
    for i,b in enumerate(block['bins']):tables.append(dict(baseline=baseline,descriptor=key,tercile=i+1,**b))
    cut=np.quantile(x,2/3)
    for seed in range(5):
     ds=np.abs(arrays['CoDE'][seed]-frame.y_true.to_numpy())-np.abs(arrays[baseline][seed]-frame.y_true.to_numpy())
     sensitivity.append(dict(baseline=baseline,descriptor=key,seed=seed,mean_delta=ds.mean(),top_minus_rest=ds[x>=cut].mean()-ds[x<cut].mean()))
  corr=pd.DataFrame(correlations)
  for baseline in ARMS[:-1]:
   idx=corr.baseline==baseline;corr.loc[idx,'contrast_q_bh6']=ud.bh(corr.loc[idx,'p_permutation'].to_numpy())
  corr['contrast_q_bh18']=ud.bh(corr.p_permutation.to_numpy())
  corr.to_csv(out/'descriptor-statistics.csv',index=False);pd.DataFrame(sensitivity).to_csv(out/'seed-sensitivity.csv',index=False)
  bins=pd.DataFrame(tables);bins.to_csv(out/'terciles.csv',index=False)
 subtitle='CL123 train/val → CL3 test' if scheme.endswith('cl123') else 'LP train/val → CL3-filtered test'
 for kind in ['terciles','10bins']:
  fig,axs=plt.subplots(2,3,figsize=(14,8.5),sharey=True,layout='constrained')
  for ax,(key,label) in zip(axs.flat,METRICS.items()):
   if kind=='terciles':
    b=bins[(bins.baseline=='CoE')&(bins.descriptor==key)]
    ax.bar(range(3),b['mean'],color=COLORS[-1],width=.6)
    ax.errorbar(range(3),b['mean'],yerr=np.array([b['mean']-b.ci_lo,b.ci_hi-b['mean']]),fmt='none',color='#333333',capsize=4)
    ax.set_xticks(range(3),[f'{r.lo_edge:.3g}–{r.hi_edge:.3g}\nn={r.n}' for r in b.itertuples()])
    s=corr[(corr.baseline=='CoE')&(corr.descriptor==key)].iloc[0]
    ax.set_title(f'Top − rest {s.top_minus_rest:+.3f} pK; BH q={s.contrast_q_bh6:.3f}',fontsize=10)
   else:draw_bins(ax,binned(frame[key],frame.delta_vs_CoE),annotate=True)
   ax.axhline(0,color='#555555',lw=.8);ax.set_xlabel(label);ax.grid(axis='y',alpha=house.BACKGROUND_ALPHA)
  for ax in axs[:,0]:ax.set_ylabel('Mean Δ absolute pK error (CoDE − CoE)\nNegative: CoDE better')
  fig.suptitle(f'Manuscript encoders · 5 probe seeds · {subtitle} (n=733)\nAverage absolute errors over seeds per complex; complex-bootstrap 95% CI',fontsize=12)
  save(fig,out/f'affinity-cl3-bfactor-{kind}')
 fig,axs=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
 for ax,metric in zip(axs[:2],['mae','spearman']):
  for i,arm in enumerate(ARMS):
   vals=summary[summary.arm==arm][metric].to_numpy()
   ax.scatter(np.full(5,i)+np.linspace(-.10,.10,5),vals,s=22,color=COLORS[i],alpha=house.DATA_ALPHA)
   ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=0),fmt='D',color=COLORS[i],capsize=5,ms=6)
  ax.set_xticks(range(4),[LABELS[x] for x in ARMS],rotation=25,ha='right')
  ax.set_ylabel('MAE (pK); lower is better' if metric=='mae' else 'Spearman ρ; higher is better')
  ax.set_title('Seed means ± SD; dots: individual seeds')
 key='b_contact_invd2';bs=bins[bins.descriptor==key]
 for i,base in enumerate(ARMS[:-1]):
  b=bs[bs.baseline==base];xx=np.arange(3)+(i-1)*.15
  axs[2].errorbar(xx,b['mean'],yerr=np.array([b['mean']-b.ci_lo,b.ci_hi-b['mean']]),fmt='o-',color=COLORS[i],capsize=3,label='CoDE − ('+LABELS[base]+')')
 axs[2].axhline(0,color='#555555',lw=.8);axs[2].set_xticks(range(3),['Low B','Middle B','High B'])
 axs[2].set_ylabel('Δ mean absolute pK error');axs[2].legend(fontsize=8, framealpha=house.DATA_ALPHA);axs[2].set_title('B terciles; complex-bootstrap 95% CI')
 fig.suptitle('Does adding B-factor summaries explain the density gain?\n'+subtitle,fontsize=13)
 save(fig,out/'bfactor-concat-ablation')
 print(subtitle,agg.to_string(),flush=True)
 print(corr[corr.descriptor=='b_contact_invd2'].to_string(index=False),flush=True)
 return frame

def representation(frame):
 out=REPRESENTATION;out.mkdir(parents=True,exist_ok=True)
 common=frame.dropna(subset=list(UNC));targets=common[list(UNC)].to_numpy()
 records=[];decoded=pd.DataFrame({'pid':common.index})
 for arm in ['CoE','CoDE']:
  protocol=json.loads((SRC/f'{arm}-protocol.json').read_text());bundle=torch.load(protocol['features'],map_location='cpu',weights_only=False)
  x=np.stack([bundle['features'][pid].numpy() for pid in common.index]).astype(float)
  p,baseline,fold,alphas=decode(x,targets)
  decoded['fold']=fold
  for j,key in enumerate(UNC):
   decoded[key]=targets[:,j];decoded[f'{arm}__{key}']=p[:,j]
   records.append(dict(arm=arm,descriptor=key,n=len(common),rho=stats.spearmanr(targets[:,j],p[:,j]).statistic,
     r2=1-((targets[:,j]-p[:,j])**2).sum()/((targets[:,j]-targets[:,j].mean())**2).sum()))
  (out/f'{arm}-decoder.json').write_text(json.dumps(dict(feature_protocol=protocol,selected_alphas=alphas),indent=2)+'\n')
 scores=pd.DataFrame(records);scores.to_csv(out/'decoding-statistics.csv',index=False);decoded.to_csv(out/'oof-predictions.csv',index=False)
 plot_representation(scores)

def plot_representation(scores):
 fig,axs=plt.subplots(1,2,figsize=(10,6.5),layout='constrained')
 for ax,metric in zip(axs,['rho','r2']):
  matrix=scores.pivot(index='descriptor',columns='arm',values=metric).loc[list(UNC),['CoE','CoDE']]
  im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
  for i in range(len(UNC)):
   for j in range(2):
    v=matrix.iloc[i,j];ax.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(v)>.65 else 'black')
  ax.set_xticks([0,1],['CoE','CoDE']);ax.set_yticks(range(len(UNC)),list(UNC.values()));ax.set_title('OOF Spearman ρ' if metric=='rho' else 'OOF R²')
 fig.colorbar(im,ax=axs,shrink=.7);fig.suptitle(f'Manuscript embeddings: uncertainty decoding (n={int(scores.n.iloc[0])})\nNested 5×3-fold ridge; exploratory within CL3 cohort, not protein-family-disjoint',fontsize=11)
 save(fig,REPRESENTATION/'embedding-uncertainty-decoding');print(scores.to_string(index=False),flush=True)

def standardized_controls(saved_only=False):
 out=STANDARDIZED_B;out.mkdir(parents=True,exist_ok=True)
 arms=['CoE-standardized','CoE-Bvector-standardized','CoDE-standardized']
 labels=['CoE','CoE + 4 B descriptors','CoDE']
 if saved_only:
  df=pd.read_csv(out/'per-seed-metrics.csv')
 else:
  rows=[]
  for arm in arms:
   for seed in range(5):
    p=SRC/f'{arm}-lp_edrscc_v2_cl123-seed{seed}-metrics.json'
    if not p.exists():return
    m=json.loads(p.read_text());rows.append(dict(arm=arm,seed=seed,**m['cl3_test']))
  df=pd.DataFrame(rows);df.to_csv(out/'per-seed-metrics.csv',index=False)
  df.groupby('arm')[['pearson','spearman','rmse','mae']].agg(['mean',lambda x:x.std(ddof=0)]).rename(columns={'<lambda_0>':'std'}).to_csv(out/'summary.csv')
 fig,axs=plt.subplots(1,2,figsize=(9,4.7),layout='constrained')
 for ax,metric in zip(axs,['spearman','mae']):
  for i,arm in enumerate(arms):
   vals=df[df.arm==arm][metric].to_numpy()
   ax.scatter(np.full(5,i)+np.linspace(-.08,.08,5),vals,color=COLORS[[0,2,3][i]],alpha=house.DATA_ALPHA)
   ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=0),fmt='D',capsize=5,color=COLORS[[0,2,3][i]])
  ax.set_xticks(range(3),labels,rotation=20,ha='right');ax.set_ylabel('Spearman ρ' if metric=='spearman' else 'MAE (pK)')
 fig.suptitle('B-factor concat control with equal input scaling · CL123 split\nAll embedding dimensions and B descriptors z-scored on train only; 5 seed means ± SD',fontsize=12)
 save(fig,out/'bfactor-concat-standardized')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--primary-only',action='store_true');ap.add_argument('--plots-only',action='store_true',help='Redraw saved statistics without recomputing analyses or decoders');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
 primary=analyse('lp_edrscc_v2_cl123',saved_only=a.plots_only)
 if not a.primary_only:analyse('lp_edrscc_v2',saved_only=a.plots_only)
 if not a.plots_only:representation(primary)
 standardized_controls(saved_only=a.plots_only)
 if a.plots_only:
  plot_representation(pd.read_csv(REPRESENTATION/'decoding-statistics.csv'))
  return
 (OUT/'README.md').write_text('''# Manuscript CDG v2 / CoE five-seed uncertainty analysis

CDG v2: 260806_cdg_100m_v2_ep100, epoch 25, cached 640-D ChannelViT features.
CoE: the actual manuscript table baseline, 260723_ar_cvit_100m_v2_mask075_coords, epoch 49, 640-D ChannelViT.
This CoE is not the newer atom-biased C_v2 epoch-20 matched-recipe control.

Canonical probe: MSE, raw cached features and raw pK, SiLU hidden 128, dropout 0.1,
Adam lr 1e-3, weight decay 1e-4, batch 64, 200 epochs max, val Spearman early stop/patience 30.
Five seeds 0–4. CPU execution; numerical trajectory can differ from the historical GPU run.
No test selection or test-label fitting for the affinity heads.

- cl3-5seed/: train/val/test = 1559/410/733; matches the manuscript table's CL123 probe source.
- lp-trained-cl3-filtered: train/val = 3850/817, train on LP and evaluate its CL3 test subset of 733; keeps earlier analysis's evaluation-only filtering protocol distinct.
- ../representation/: descriptive nested 5 outer × 3 inner ridge decoding of eight crystallographic descriptors on common complete CL3 cases; this probe uses held-out folds within the test cohort, not a new independent benchmark.

Primary affinity outcome averages absolute errors over five seeds per complex, matching the manuscript's seed-averaged reporting. This differs from taking the error of an ensemble-mean prediction. The latter is also exported as a sensitivity column.
Binned error bars: 95% complex-bootstrap confidence intervals conditional on these seeds; not calibrated per-example uncertainty, seed uncertainty, cluster bootstrap or simultaneous intervals.
Permutation top-versus-rest contrasts have BH correction across six descriptors per baseline and across all 18 comparisons in the CSV.

../controls/standardized-b-concat/ additionally z-scores all embedding dimensions on training data for CoE, CoE+B and CoDE; this is a preprocessing sensitivity check under the same fixed head hyperparameters, not a tuned optimum.

B-concat ablations: add either contact-weighted 1/d² B alone, or contact B, normalized contact B, mean protein B, pocket/protein mean B ratio. Training-only median imputation and z-scoring; identical probe settings. These test complex-level summary concatenation, not atom-level B-factor encoding.
Representation B-factor decodability is not proof of causal use for affinity; original maps depend on measured diffraction and model phases.

Run provenance, checkpoint/feature hashes, all seeds, validation-selected epochs and split ID hashes are under results/task1-affinity/_shared/260919_cdg_v2_5seed/.
''')
if __name__=='__main__':main()

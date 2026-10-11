#!/usr/bin/env python3
"""Analyse and redraw only the retained uncertainty figures on the audited CL3 extension."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'figures/_parts/fig_uncertainty'))
from .paths import MOTIVATION,REPRESENTATION,RANKING_METRICS
from . import motivation
from . import ranking_metrics as ranking
from . import motivation_variants as variants
from .affinity_density import decode,UNC
from .decode_baseline import load_features
import draw as house
RUN=ROOT/'results/task1-affinity/_shared/260921_cl3_extension_cpu'
DATA=RANKING_METRICS.parent/'ranking-by-metric-expanded-cl3'
REPR=REPRESENTATION/'expanded-cl3'
SOURCE=RUN/'extended-descriptors-and-predictions.csv'
KEYS=['b_contact_invd2','b_contact_invd2_z','lig_rscc','protein_rscc_median','occ_site_frac_altloc']
TARGETS=[('protein_rscc_median' if k=='poc_rscc_median' else k) for k in UNC]
ARMS=['CoE','CoDE','DeepDTA','Nesso-1','ProFSA']
LABELS={**motivation.LABELS,'protein_rscc_median':'Protein\nRSCC'}
ACTIVE=MOTIVATION/'active-analysis.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def analyse_affinity(source=SOURCE,out=DATA,expected_n=1163):
 out.mkdir(parents=True,exist_ok=True)
 frame=pd.read_csv(source);assert len(frame)==expected_n and frame.pid.is_unique
 value_cols=['y_true']+[f'{arm}_seed{s}_prediction' for arm in ['CoE','CoDE'] for s in range(5)]
 # Restore the float32 model outputs; CSV parsing must not invent rank differences for ties.
 frame[value_cols]=frame[value_cols].astype(np.float32).astype(float)
 summaries=[];errors=[];members=[];seeds=[];contrasts=[]
 for j,key in enumerate(KEYS):
  block=frame[np.isfinite(frame[key])].copy();groups,labels,cuts,kind=ranking.stratify(block[key].to_numpy(),key)
  rng=np.random.default_rng(ranking.RNG_SEED+j);rng_err=np.random.default_rng(ranking.RNG_SEED+j)
  point_by_group=[];rmse_by_group=[];boot_by_group=[];rmse_boot_by_group=[]
  for g,label in enumerate(labels):
   sub=block[groups==g];truth=sub.y_true.to_numpy();a,b=[np.stack([sub[f'{arm}_seed{s}_prediction'] for s in range(5)]) for arm in ['CoE','CoDE']]
   ra,rb=ranking.corr(a,truth[None]),ranking.corr(b,truth[None]);estimate=float((rb-ra).mean())
   boot=ranking.bootstrap_group(a,b,truth,rng);lo,hi=np.quantile(boot,[.025,.975])
   ea=np.sqrt(((a-truth[None])**2).mean(-1));eb=np.sqrt(((b-truth[None])**2).mean(-1));gap=float((eb-ea).mean())
   error_boot=variants.bootstrap_rmse(a,b,truth,rng_err);el,eh=np.quantile(error_boot,[.025,.975])
   meta=dict(descriptor=key,label=LABELS[key].replace('\n',' '),group=label,group_index=g,n=len(sub),n_valid=len(block),n_missing=len(frame)-len(block),binning=kind,threshold_1=cuts[0],threshold_2=cuts[1],x_min=sub[key].min(),x_max=sub[key].max(),x_median=sub[key].median(),original_n=int(sub.cohort.eq('original frozen CL3').sum()),added_n=int(sub.cohort.eq('additional audited CL3').sum()))
   summaries.append(dict(**meta,coe_spearman=ra.mean(),code_spearman=rb.mean(),estimate=estimate,ci_lo=lo,ci_hi=hi))
   errors.append(dict(**meta,coe_rmse=ea.mean(),code_rmse=eb.mean(),estimate=gap,ci_lo=el,ci_hi=eh))
   for seed in range(5):seeds.append(dict(descriptor=key,group=label,seed=seed,coe_spearman=ra[seed],code_spearman=rb[seed],coe_rmse=ea[seed],code_rmse=eb[seed]))
   members.extend(dict(descriptor=key,pid=p,group=label,group_index=g,value=v) for p,v in zip(sub.pid,sub[key]))
   point_by_group.append(estimate);rmse_by_group.append(gap);boot_by_group.append(boot);rmse_boot_by_group.append(error_boot)
  direction=-1 if key in ['lig_rscc','protein_rscc_median'] else 1
  for metric,points,boots,sign in [('spearman',point_by_group,boot_by_group,1),('rmse',rmse_by_group,rmse_boot_by_group,-1)]:
   contrast=direction*sign*(boots[2]-boots[0]);lo,hi=np.quantile(contrast,[.025,.975])
   contrasts.append(dict(descriptor=key,metric=metric,estimate=direction*sign*(points[2]-points[0]),ci_lo=lo,ci_hi=hi,definition='higher-uncertainty minus lower-uncertainty performance improvement'))
  print('Affinity',key,'n',len(block),'ranking',point_by_group,'rmse',rmse_by_group,flush=True)
 pd.DataFrame(summaries).to_csv(out/'ranking-by-metric.csv',index=False)
 pd.DataFrame(errors).to_csv(out/'rmse-by-metric.csv',index=False)
 pd.DataFrame(members).to_csv(out/'bin-membership.csv',index=False)
 pd.DataFrame(seeds).to_csv(out/'per-seed-metrics.csv',index=False)
 pd.DataFrame(contrasts).to_csv(out/'higher-vs-lower-proxy.csv',index=False)
 (out/'provenance.json').write_text(json.dumps(dict(source=str(source),sha256=sha(source),n=expected_n,keys=KEYS,seeds=list(range(5)),n_boot=8000,interval='paired-complex percentile 95%; fixed bins and fitted seeds; no multiplicity/family/seed resampling',protein_rscc='AA residues only; report-wide upper median, not pocket-specific',ranking_precision='restore original float32 model outputs before rank calculations to preserve ties across CSV roundtrips',cohort='frozen filtered CL3' if expected_n==733 else 'exploratory extension; frozen benchmark unchanged'),indent=2)+'\n')


def nesso_features():
 path=RUN/'Nesso-1-features.npz'
 if path.exists():return path
 original=load_features(ROOT/'results/task1-affinity/Nesso-1/uncertainty-cl3-v1.0.0-seed42-20260921/test-representations.npz')
 extra=(RUN/'additional-quality-pids.txt').read_text().split();missing=[]
 for pid in extra:
  folder=RUN/'nesso/inference/predictions'/pid
  if not (folder/'representation.npy').exists() or not (folder/'affinity.json').exists():missing.append(pid);continue
  value=np.load(folder/'representation.npy');assert value.shape==(384,) and np.isfinite(value).all();original[pid]=value
 if missing:raise RuntimeError(f'Nesso-1 extraction still missing {len(missing)} cases')
 ids=pd.read_csv(SOURCE).pid.tolist();np.savez_compressed(path,pids=np.asarray(ids),features=np.stack([original[p] for p in ids]))
 (RUN/'Nesso-1-features.json').write_text(json.dumps(dict(n=len(ids),added=len(extra),checkpoint='public v1.0.0; original affinity-pretraining leakage caveat retained',seed=42,precision='bf16-mixed',recycling_steps=5,execution_shards=4,feature_layer='mean final z_pp + z_pl + z_ll; 384 dimensions',base='results/task1-affinity/Nesso-1/uncertainty-cl3-v1.0.0-seed42-20260921/run.json',extension='nesso/shard-{0,1,2,3}-status.json'),indent=2)+'\n')
 return path


def analyse_decoding():
 REPR.mkdir(parents=True,exist_ok=True)
 frame=pd.read_csv(SOURCE).dropna(subset=TARGETS).reset_index(drop=True)
 assert 1160<len(frame)<=1163
 truth=frame[TARGETS].to_numpy(float);scores=[];decoded=frame[['pid',*TARGETS]].copy()
 for arm in ARMS:
  target=REPR/f'{arm}-oof.csv';protocol=REPR/f'{arm}-decoder.json'
  if target.exists():
   oof=pd.read_csv(target);assert oof.pid.equals(frame.pid);meta=json.loads(protocol.read_text());assert meta['source_sha256']==sha(SOURCE)
   pred=oof[[f'{arm}__{k}' for k in TARGETS]].to_numpy();fold=oof.fold.to_numpy()
  else:
   if arm in ['CoE','CoDE']:
    base=json.loads((ROOT/f'results/task1-affinity/_shared/260919_cdg_v2_5seed/{arm}-protocol.json').read_text())
    features=load_features(Path(base['features']));extra=load_features(RUN/f'{arm}-extra-features.pt');features.update(extra);sources=[base['features'],str(RUN/f'{arm}-extra-features.pt')]
   else:
    path=nesso_features() if arm=='Nesso-1' else RUN/f'{arm}-features.npz'
    features=load_features(path);sources=[str(path)]
   x=np.stack([features[pid] for pid in frame.pid]);assert np.isfinite(x).all()
   pred,baseline,fold,alphas=decode(x,truth)
   oof=frame[['pid',*TARGETS]].copy();oof['fold']=fold
   for j,k in enumerate(TARGETS):oof[f'{arm}__{k}']=pred[:,j]
   oof.to_csv(target,index=False)
   meta=dict(arm=arm,n=len(frame),dim=x.shape[1],source_sha256=sha(SOURCE),sources=[dict(path=p,sha256=sha(p)) for p in sources],targets=TARGETS,selected_alphas=alphas,protocol='same nested 5x3 ridge; training-fold feature/target scaling; alpha selected by inner MSE; random-complex folds, seed42; not protein-family-disjoint')
   protocol.write_text(json.dumps(meta,indent=2)+'\n')
  if 'fold' in decoded:assert np.array_equal(decoded.fold,fold)
  decoded['fold']=fold
  for j,k in enumerate(TARGETS):
   decoded[f'{arm}__{k}']=pred[:,j]
   scores.append(dict(arm=arm,descriptor=k,n=len(frame),rho=spearmanr(truth[:,j],pred[:,j]).statistic,r2=1-np.square(truth[:,j]-pred[:,j]).sum()/np.square(truth[:,j]-truth[:,j].mean()).sum()))
  print('Decoded',arm,len(frame),flush=True)
 pd.DataFrame(scores).to_csv(REPR/'decoding-statistics.csv',index=False)
 decoded.to_csv(REPR/'oof-predictions.csv',index=False)
 return pd.DataFrame(scores)


def redraw_current_figures(only=None,analysis=None):
 manifest=json.loads((MOTIVATION/'figure-manifest.json').read_text())
 if analysis is None and manifest.get('cohort')=='frozen_filtered_cl3':
  from .update_filtered import redraw_filtered
  return redraw_filtered(only=only)
 analysis=analysis or {}
 source=Path(analysis.get('source',SOURCE));data=Path(analysis.get('data',DATA));representation=Path(analysis.get('representation',REPR))
 n_affinity=analysis.get('n_affinity',1163);cohort=analysis.get('cohort','expanded CL3')
 present={stem for stem in manifest['figures'] if any((MOTIVATION/f'{stem}.{ext}').exists() for ext in ['png','pdf'])}
 if only is not None:present &= set(only)
 motivation.MAIN_KEYS[:]=KEYS;motivation.LABELS.update(LABELS)
 ranking.DIRECTION['protein_rscc_median']=-1
 ranked=pd.read_csv(data/'ranking-by-metric.csv');errors=pd.read_csv(data/'rmse-by-metric.csv')
 binning=manifest.get('contact_b_binning','terciles')
 prefixes={'absolute_20_40':'fixed-contact-b','absolute_10_20_30':'fixed-contact-b-10-20-30','absolute_15_30':'fixed-contact-b-15-30','absolute_20_30':'fixed-contact-b-20-30','quartiles':'contact-b-quartiles','absolute_20_25_30':'fixed-contact-b-20-25-30'}
 assert binning=='terciles' or binning in prefixes, f'Unknown Contact-B bins: {binning}'
 prefix=prefixes.get(binning)
 if prefix:
  provenance=json.loads((data/f'{prefix}-provenance.json').read_text())
  assert provenance['source_sha256']==sha(source), 'Contact-B statistics are stale'
  ranked=pd.concat([pd.read_csv(data/f'{prefix}-ranking.csv'),ranked[ranked.descriptor!='b_contact_invd2']],ignore_index=True)
  errors=pd.concat([pd.read_csv(data/f'{prefix}-rmse.csv'),errors[errors.descriptor!='b_contact_invd2']],ignore_index=True)
 contact=ranked[ranked.descriptor=='b_contact_invd2']
 right_ylim=(min(-.05,float(np.floor(contact.ci_lo.min()/.05)*.05)),max(.15,float(np.ceil(contact.ci_hi.max()/.05)*.05)))
 right_ylim=tuple(manifest.get('combined_right_ylim',right_ylim))
 show_models=manifest.get('combined_affinity_display')=='models'
 if show_models:
  lower=contact[['coe_ci_lo','code_ci_lo']].to_numpy();upper=contact[['coe_ci_hi','code_ci_hi']].to_numpy()
  assert np.isfinite(lower).all() and np.isfinite(upper).all()
  right_ylim=(max(-1.,float(np.floor((lower.min()-.025)/.1)*.1)),min(1.,float(np.ceil((upper.max()+.06)/.1)*.1)))
 needs_decoding=bool(present & {'uncertainty-a-decoding','uncertainty-combined-contactb-combined'})
 scores=pd.read_csv(representation/'decoding-statistics.csv') if (representation/'decoding-statistics.csv').exists() else None
 if needs_decoding and scores is None:raise RuntimeError('Decoding is not complete yet')
 with plt.rc_context({**house.PCSZ_RC,'savefig.bbox':None}):
  if 'uncertainty-bfactor-ecdf' in present:
   from .distribution import plot_distribution
   plot_distribution(source,n_affinity)
  if 'uncertainty-a-decoding' in present:
   fig=plt.figure(figsize=(12.5,3.5),facecolor=house.PCSZ_BG);ax=fig.add_axes([.10,.235,.875,.685]);motivation.decoding(ax,scores,KEYS);motivation.save(fig,'uncertainty-a-decoding')
  if 'uncertainty-b-affinity-ranking' in present:ranking.plot(ranked)
  if 'uncertainty-b-affinity-rmse' in present:variants.rmse_all(errors)
  if 'uncertainty-contact-b-spearman-rmse' in present:variants.contact_comparison(ranked,errors)
  if 'uncertainty-combined-contactb-combined' in present:
   # The combined figure may bin on another complex-level B value (manifest: combined_contact_*); other figures keep b_contact_invd2.
   key=manifest.get('combined_contact_descriptor','b_contact_invd2');combined_ranked=ranked;frame=pd.read_csv(source)
   if key!='b_contact_invd2':
    cprefix=manifest['combined_contact_prefix'];cprov=json.loads((data/f'{cprefix}-provenance.json').read_text())
    values=Path(cprov['values']);assert cprov['source_sha256']==sha(source) and cprov['values_sha256']==sha(values),'Combined contact statistics are stale'
    combined_ranked=pd.concat([pd.read_csv(data/f'{cprefix}-ranking.csv'),ranked],ignore_index=True)
    frame=frame.merge(pd.read_csv(values)[['pid',key]],on='pid',how='left',validate='one_to_one')
   default_key,variants.CONTACT=variants.CONTACT,key
   try:variants.combined_contact(scores,combined_ranked,name='uncertainty-combined-contactb-combined',metric='spearman',right_ylim=right_ylim,distribution_frame=frame if manifest.get('combined_distribution_overlay') else None,show_models=show_models,marker_scale=manifest.get('combined_marker_scale',1.),cap_scale=manifest.get('combined_cap_scale',1.),show_bin_labels=manifest.get('combined_show_bin_labels',True),legend_scale=manifest.get('combined_legend_scale',1.44),left_panel=manifest.get('combined_left_panel','decoding'))
   finally:variants.CONTACT=default_key
 if scores is not None:scores[scores.descriptor.isin(KEYS)].to_csv(MOTIVATION/'decoding-main-panel.csv',index=False)
 ranked.to_csv(MOTIVATION/'ranking-panel.csv',index=False)
 errors.to_csv(MOTIVATION/'rmse-panel.csv',index=False)
 ACTIVE.write_text(json.dumps(dict(cohort=cohort,n_affinity=n_affinity,n_decoding=int(scores.n.iloc[0]) if scores is not None else None,keys=KEYS,source=str(source),decoding_statistics=str(representation/'decoding-statistics.csv'),ranking_statistics=str(data/'ranking-by-metric.csv'),rmse_statistics=str(data/'rmse-by-metric.csv'),combined_metric='spearman',contact_b_binning=binning,contact_b_prefix=prefix,contact_b_ranking_statistics=str(data/f'{prefix}-ranking.csv') if prefix else None,contact_b_rmse_statistics=str(data/f'{prefix}-rmse.csv') if prefix else None,combined_right_ylim=right_ylim,combined_marker_scale=manifest.get('combined_marker_scale',1.),combined_cap_scale=manifest.get('combined_cap_scale',1.),combined_show_bin_labels=manifest.get('combined_show_bin_labels',True),combined_legend_scale=manifest.get('combined_legend_scale',1.44),combined_affinity_display='models' if show_models else 'difference',combined_distribution_overlay=manifest.get('combined_distribution_overlay',False),combined_distribution_xlim=[10,40] if manifest.get('combined_distribution_overlay') else None,combined_point_x='within-bin median B' if manifest.get('combined_distribution_overlay') else 'category index',generator=analysis.get('generator','cd figures && python -m _parts.fig_uncertainty.update_expanded --plots-only'),overall_statistics=analysis.get('overall_statistics'),cohort_provenance=analysis.get('cohort_provenance')),indent=2)+'\n')
 if scores is not None:
  from .notes_current import write_notes
  write_notes()
 print('Redrawn only:',sorted(present),flush=True)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plots-only',action='store_true');ap.add_argument('--affinity-only',action='store_true');ap.add_argument('--decoding-only',action='store_true');a=ap.parse_args()
 if a.plots_only:redraw_current_figures();return
 if not a.decoding_only:analyse_affinity()
 if a.affinity_only:return
 analyse_decoding();redraw_current_figures()
if __name__=='__main__':main()

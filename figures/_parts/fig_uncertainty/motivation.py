#!/usr/bin/env python3
"""Motivation panels styled with the existing PoseCheck strain-ECDF definitions.

Only saved diagnostics are plotted. No new decoder fitting or affinity selection.
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'figures'))
import draw as house
from .paths import (MOTIVATION as OUT, CL3, REPRESENTATION, INPUT_DENSITY,
                               ATOMIC_B, RANKING_METRICS, MOTIVATION_ARCHIVE, MAP_DIAGNOSTICS as MAP_BASE)
LABELS={
 'b_contact_invd2':'Contact\nB factor',
 'b_contact_invd2_z':'Normalized\ncontact B factor',
 'rel_pocket_b':'Pocket / protein\nB factor',
 'lig_rscc':'Ligand\nRSCC',
 'poc_rscc_median':'Polymer\nRSCC',
 'occ_site_frac_partial':'Partial-occupancy\nfraction',
 'occ_site_frac_altloc':'Alternate-location\nfraction',
 'resolution':'Resolution',
}
MAIN_KEYS=['b_contact_invd2','b_contact_invd2_z','lig_rscc','poc_rscc_median','occ_site_frac_altloc']
ALL_KEYS=list(LABELS)
# Use the shared palette, including CoE's pastel green.
MODEL_COLORS={'DeepDTA':house.pale('DeepDTA'),'ProFSA':house.pale('ProFSA'),'Nesso-1':house.pale('Nesso-1'),
              'CoE':house.pale('CoE'),'CoDE':house.pale('CoDE')}
MODEL_ORDER=['DeepDTA','ProFSA','Nesso-1','CoE','CoDE']
MODEL_LABELS={'Nesso-1':'Nesso-1†'}



def load_decoding():
 active=OUT/'active-analysis.json'
 if active.exists():
  import json
  config=json.loads(active.read_text())
  MAIN_KEYS[:]=config['keys'];LABELS['protein_rscc_median']='Protein\nRSCC'
  return pd.read_csv(config['decoding_statistics'])
 table=pd.read_csv(REPRESENTATION/'decoding-statistics.csv')
 reference=pd.read_csv(REPRESENTATION/'oof-predictions.csv')
 for arm in ['DeepDTA','ProFSA','Nesso-1']:
  path=REPRESENTATION/'baselines'/arm
  if not (path/'decoding-statistics.csv').exists():continue
  baseline=pd.read_csv(path/'oof-predictions.csv')
  assert baseline.pid.equals(reference.pid) and baseline.fold.equals(reference.fold)
  assert np.allclose(baseline[ALL_KEYS],reference[ALL_KEYS],equal_nan=True)
  scores=pd.read_csv(path/'decoding-statistics.csv')
  assert set(scores.arm)=={arm} and set(scores.descriptor)==set(ALL_KEYS)
  assert scores.n.eq(len(reference)).all()
  for row in scores.itertuples():
   assert np.isclose(row.rho,spearmanr(baseline[row.descriptor],baseline[f'{arm}__{row.descriptor}']).statistic)
  table=pd.concat([table,scores],ignore_index=True)
 return table


def axes_style(ax):
 house._pcsz_style(ax,grid_axis='y')
 for name in ['left','bottom']:ax.spines[name].set_zorder(6)


def save(fig,name,out=OUT,*,allow_new=False,crop_vertical=False,crop_horizontal=False):
 if out == OUT:
  name=name.replace('uncertainty-motivation-', 'uncertainty-', 1)
  if name=='uncertainty-combined-contact-b-expanded':name='uncertainty-combined-contactb-combined'
  manifest=OUT/'figure-manifest.json'
  if manifest.exists():
   import json
   allowed=json.loads(manifest.read_text())['figures']
   if name not in allowed or (not allow_new and not any((out/f'{name}.{ext}').exists() for ext in ['png','pdf'])):
    plt.close(fig);print('Skipped removed figure:',name,flush=True);return
 out.mkdir(parents=True,exist_ok=True)
 if not house.prepare_publication_figure(fig,out/name):
  plt.close(fig);return
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer()
 for ax in fig.axes:
  xt=[t for p,t in zip(ax.get_xticks(),ax.get_xticklabels()) if ax.get_xlim()[0]<=p<=ax.get_xlim()[1]]
  yt=[t for p,t in zip(ax.get_yticks(),ax.get_yticklabels()) if ax.get_ylim()[0]<=p<=ax.get_ylim()[1]]
  artists=[ax.xaxis.label,ax.yaxis.label,*xt,*yt,*ax.texts]
  for artist in artists:
   if not artist.get_visible() or not artist.get_text():continue
   box=artist.get_window_extent(renderer)
   if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
    raise ValueError(f'Clipped text in {name}: {artist.get_text()} {box.bounds}')
 export_box=None
 if crop_vertical or crop_horizontal:
  from matplotlib.transforms import Bbox
  tight=fig.get_tightbbox(renderer)
  # Crop outer whitespace while preserving physical panel sizes and typography.
  export_box=Bbox.from_extents(
   max(0,tight.x0-.04) if crop_horizontal else 0,
   max(0,tight.y0-.04) if crop_vertical else 0,
   min(fig.get_figwidth(),tight.x1+.04) if crop_horizontal else fig.get_figwidth(),
   min(fig.get_figheight(),tight.y1+.04) if crop_vertical else fig.get_figheight())
 for ext in house.retained_formats(out/name):
  fig.savefig(out/f'{name}.{ext}',facecolor=house.PCSZ_BG,bbox_inches=export_box,pad_inches=0,dpi=220)
  house.sync_saved_figure(out/f'{name}.{ext}')
 plt.close(fig)


def decoding(ax,table,keys,*,legend_scale=1.0,model_colors=None):
 axes_style(ax)
 model_colors=MODEL_COLORS if model_colors is None else model_colors
 arms=[arm for arm in MODEL_ORDER if arm in set(table.arm)]
 step=.32 if len(arms)==2 else .78/len(arms)
 for j,arm in enumerate(arms):
  vals=table[table.arm==arm].set_index('descriptor').loc[keys,'rho'].to_numpy()
  assert len(vals)==len(keys) and np.isfinite(vals).all()
  x=np.arange(len(keys))+(j-(len(arms)-1)/2)*step
  ax.bar(x,vals,width=(.30 if len(arms)==2 else step*.93),color=model_colors[arm],edgecolor='none',label=MODEL_LABELS.get(arm,arm),zorder=3)
  for xx,v in zip(x,vals):ax.text(xx,v+.019,f'{v:.2f}'.replace('-0.', '-.').removeprefix('0'),ha='center',va='bottom',fontsize=9.5,color=house.PCSZ_INK)
 labels=dict(LABELS)
 if len(keys)>5:
  labels.update(occ_site_frac_partial='Partial\noccupancy\nfraction',occ_site_frac_altloc='Alternate\nlocation\nfraction')
 ax.set_xticks(range(len(keys)),[labels[k] for k in keys]);ax.set_ylim(min(0,float(table.rho.min())-.06) if table.rho.min()<0 else 0,1.05);ax.set_xlim(-.55,len(keys)-.45)
 ax.yaxis.set_major_locator(MultipleLocator(.2))
 ax.set_ylabel('Decoding\nSpearman ρ',fontsize=house.PCSZ_LABEL_FS,labelpad=house.PCSZ_YPAD)
 legend_position={'loc':'upper center','bbox_to_anchor':(.48,1)} if len(keys)>5 else {'loc':'upper right'}
 legend=ax.legend(**legend_position,ncol=len(arms),frameon=True,fancybox=False,fontsize=house.PCSZ_KEY_KW['fontsize'] * legend_scale,
  edgecolor=house.LEGEND_EDGE,facecolor=house.PCSZ_KEY_FACE,framealpha=house.DATA_ALPHA,
  borderpad=.4,handlelength=1.5,handletextpad=.5,columnspacing=1.2)
 legend.get_frame().set_linewidth(.7)


def ranking(ax,table):
 axes_style(ax)
 sel=table[table.group.isin(['Low','Middle','High'])].set_index('group').loc[['Low','Middle','High']]
 m=sel.estimate.to_numpy();lo=sel.ci_lo.to_numpy();hi=sel.ci_hi.to_numpy()
 ax.errorbar(range(3),m,yerr=[m-lo,hi-m],fmt='o-',color=MODEL_COLORS['CoDE'],capsize=4,capthick=1.15,
  elinewidth=1.15,lw=2.5*house.PCSZ_LW_SCALE,markersize=5.5,zorder=3)
 for i,v in enumerate(m):
  ax.annotate(f'{v:+.3f}',(i,v),xytext=((-8,-15) if i==2 else (8,-15)),ha=('right' if i==2 else 'left'),
   textcoords='offset points',fontsize=10,color=house.PCSZ_INK)
 ax.axhline(0,color=house.PCSZ_AXIS,lw=1.0,ls=(0,(3,3)),zorder=2)
 ax.set_xticks(range(3),['Low B','Middle B','High B']);ax.set_xlim(-.3,2.3);ax.set_ylim(-.06,.125)
 ax.yaxis.set_major_locator(MultipleLocator(.05))
 ax.set_ylabel('ΔSpearman\n(CoDE − CoE)',fontsize=house.PCSZ_LABEL_FS,labelpad=house.PCSZ_YPAD)
 ax.set_xlabel('')


def map_panel():
 table=pd.read_csv(MAP_BASE/'map-uncertainty-bins.csv')
 table=table[(table.uncertainty=='b_contact_invd2')&(table.map_metric=='map_sharpness')].sort_values('bin')
 corr=pd.read_csv(MAP_BASE/'map-uncertainty-correlations.csv')
 score=corr[(corr.uncertainty=='b_contact_invd2')&(corr.map_metric=='map_sharpness')].iloc[0]
 assert len(table)==10 and table.n.sum()==733
 current=pd.read_csv(CL3/'per-complex.csv')[['pid','b_contact_invd2']]
 maps=pd.read_csv(MAP_BASE/'map-statistics.csv')
 check=current.merge(maps,on='pid',validate='one_to_one');assert len(check)==733
 assert np.isclose(spearmanr(check.b_contact_invd2,check.map_sharpness).statistic,score.spearman)
 fig,ax=plt.subplots(figsize=house.PCSZ_STRAIN_SIZE)
 fig.patch.set_facecolor(house.PCSZ_BG);house._pcsz_style(ax)
 mean=table['mean'].to_numpy();lo=table.ci_lo.to_numpy();hi=table.ci_hi.to_numpy()
 ax.errorbar(table.x_mean,mean,yerr=[mean-lo,hi-mean],fmt='o-',color=house.REF_COLOR,lw=2.5*house.PCSZ_LW_SCALE,
  elinewidth=1.1,capsize=3.5,markersize=4.5,zorder=3)
 ax.set_xlim(5,55);ax.set_ylim(2,4.7)
 ax.set_xlabel('Contact-weighted B-factor (Å²)',fontsize=house.PCSZ_LABEL_FS)
 ax.set_ylabel('Density sharpness\n(Å⁻²)',fontsize=house.PCSZ_LABEL_FS,labelpad=house.PCSZ_YPAD)
 ax.text(.98,.98,f'Spearman ρ = {score.spearman:.3f}\nAdjusted ρ = {score.partial_spearman:.3f}',
  transform=ax.transAxes,ha='right',va='top',fontsize=11,color=house.PCSZ_INK)
 house._pcsz_comparison_layout(fig,ax)
 # Two-line physical-unit label needs slightly more space than the ECDF's single-line x title.
 fig.subplots_adjust(left=.19,right=.98,bottom=.27,top=.94)
 save(fig,'input-density-bfactor',out=INPUT_DENSITY)
 table.to_csv(INPUT_DENSITY/'input-density-bfactor-bins.csv',index=False)
 check[['pid','b_contact_invd2','map_sharpness']].to_csv(INPUT_DENSITY/'input-density-bfactor-points.csv',index=False)


def redraw_current_supplementary():
 a=load_decoding()
 keys=[('protein_rscc_median' if k=='poc_rscc_median' and 'protein_rscc_median' in set(a.descriptor) else k) for k in ALL_KEYS]
 LABELS['protein_rscc_median']='Protein\nRSCC'
 with plt.rc_context({**house.PCSZ_RC,'savefig.bbox':None}):
  fig,ax=plt.subplots(figsize=(17.0 if 'Nesso-1' in set(a.arm) else 14.0,3.6))
  fig.patch.set_facecolor(house.PCSZ_BG)
  decoding(ax,a,keys,model_colors={arm:house.pale(arm) for arm in MODEL_ORDER});fig.subplots_adjust(left=.075,right=.985,bottom=.27,top=.94)
  save(fig,'decoding-all-descriptors',out=REPRESENTATION)
  map_panel()


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--supplementary',action='store_true',help='Also redraw archived combined/contact-B layouts and supplementary descriptors/maps')
 args=ap.parse_args()
 if (OUT/'active-analysis.json').exists():
  from .update_expanded import redraw_current_figures
  redraw_current_figures()
  if args.supplementary:redraw_current_supplementary()
  return
 OUT.mkdir(parents=True,exist_ok=True)
 a=load_decoding()
 with plt.rc_context({**house.PCSZ_RC,'savefig.bbox':None}):
  fig=plt.figure(figsize=(12.5 if 'Nesso-1' in set(a.arm) else 10.5,3.5));fig.patch.set_facecolor(house.PCSZ_BG)
  ax=fig.add_axes([.10,.235,.875,.685]);decoding(ax,a,MAIN_KEYS)
  save(fig,'uncertainty-motivation-a-decoding')
  if args.supplementary:
   old_b=pd.read_csv(ATOMIC_B/'ranking-gap-complex-bootstrap.csv')
   fig=plt.figure(figsize=(18.0 if 'Nesso-1' in set(a.arm) else 16.0,3.5));fig.patch.set_facecolor(house.PCSZ_BG)
   left=fig.add_axes([.065,.235,.515,.685]);right=fig.add_axes([.685,.235,.295,.685])
   decoding(left,a,MAIN_KEYS);ranking(right,old_b)
   save(fig,'uncertainty-motivation',out=MOTIVATION_ARCHIVE)
   fig=plt.figure(figsize=(6.5,3.5));fig.patch.set_facecolor(house.PCSZ_BG)
   ax=fig.add_axes([.205,.235,.770,.685]);ranking(ax,old_b)
   save(fig,'contact-b-only',out=MOTIVATION_ARCHIVE)
   fig,ax=plt.subplots(figsize=(17.0 if 'Nesso-1' in set(a.arm) else 14.0,3.6));fig.patch.set_facecolor(house.PCSZ_BG)
   decoding(ax,a,ALL_KEYS);fig.subplots_adjust(left=.075,right=.985,bottom=.27,top=.94)
   save(fig,'decoding-all-descriptors',out=REPRESENTATION)
   a.to_csv(REPRESENTATION/'decoding-all-descriptors.csv',index=False)
   map_panel()
 from .ranking_metrics import analyse as analyse_ranking, plot as plot_ranking, write_readme
 RANKING_METRICS.mkdir(parents=True,exist_ok=True)
 b=pd.read_csv(RANKING_METRICS/'ranking-by-metric.csv') if (RANKING_METRICS/'ranking-by-metric.csv').exists() else analyse_ranking()
 plot_ranking(b);write_readme(b)
 a[a.descriptor.isin(MAIN_KEYS)].to_csv(OUT/'decoding-main-panel.csv',index=False)
 b.to_csv(OUT/'ranking-panel.csv',index=False)
 (OUT/'caption-and-scope.md').write_text('''# Candidate Figure 2: structural disorder, density support, and affinity ranking

**Style.** Reuses `figures/draw.py` PoseCheck strain-ECDF settings: DejaVu Sans; black left/bottom axes; outward 11 pt ticks; 16.1 pt axis labels; dotted grey grid; the official method Color palette (DeepDTA jade teal, ProFSA purple, Nesso-1 rose, CoE green, CoDE blue), with neutral input-density data in the shared reference grey (#9AA0A6); editable SVG text and embedded TrueType PDF fonts. Large titles and methods footnotes are kept in the caption rather than the artwork. Decoding bar annotations omit the leading zero (e.g. .54) and retain two decimal places.

**Reproduce.** From the repository root: `cd figures && python -m _parts.fig_uncertainty.motivation`. The shared output paths are defined in `figures/_parts/fig_uncertainty/paths.py`. This redraw reads saved decoding and bootstrap tables; it does not refit models.

**Separate panel exports.** The default command produces only `uncertainty-motivation-a-decoding` (12.5 × 3.5 inches with five models) and `uncertainty-motivation-b-affinity-ranking` (five metric panels, 21 × 3.7 inches), each as PNG/PDF/SVG. A/B panel letters and B x-axis titles are omitted. B reports per-bin sample sizes and the descriptor's interpretation direction. Earlier combined/contact-B-only layouts and palette previews are archived under `../exploratory/motivation-layouts-20260921/`; additional descriptor and input-density outputs require `--supplementary`. Bootstrap tables and bin membership are in `../affinity/metric-review/ranking-by-metric/`.

**Main figure caption.** Structural-disorder and density-support information in learned representations, and stratified affinity performance. (a) Out-of-fold Spearman correlations for nested ridge decoding from frozen CoE and CoDE embeddings on 732 complete CL3 cases: contact-weighted B, within-protein-normalized contact B, ligand RSCC, whole-report polymer RSCC (upper median), and the fraction of matched pocket heavy-atom sites within 4.5 Å of the ligand with alternate locations. These capture different crystallographic properties. RSCC measures model–map agreement and mixes accuracy and precision; it is not a calibrated uncertainty value. Alternate-location fraction describes modeled heterogeneity and depends on deposition/refinement choices. (b) CoDE-minus-CoE affinity Spearman, averaged over five fitted probe seeds, stratified separately by each of the five descriptors. The first four use raw-value terciles, keeping ties together, on 733 CL3 cases. Alternate-location fraction has 605 zero values among 732 valid cases; it uses zero / lower-positive / higher-positive groups (605/63/64), with a median split among positive values. Error bars are 8000 paired-complex bootstrap percentile 95% intervals conditional on the fitted seeds and fixed bins. Higher B and alternate-location fraction indicate more disorder/heterogeneity; lower RSCC indicates weaker model–map support. These are distinct proxies, not calibrated predictive uncertainty.

**Descriptor-scope correction (2026-09-21).** The legacy column `poc_rscc_median`, previously labeled "Pocket RSCC", is an upper median over all report entries recognized as amino-acid or nucleotide residues; its producer applies no pocket-distance filter. It is now labeled "Polymer RSCC". The saved decoding targets and scores are unchanged. This descriptor supports a global model–map-agreement interpretation, not a local pocket claim. The ligand RSCC producer selects the largest non-junk ligand residue type (at least six atoms) and its best-RSCC copy, rather than matching the exact PDBbind ligand instance. See [metric definitions](metric-definitions.md) for formulas and limitations.

**Retained supplementary panel (not drawn by default).** [Input density](../input-density/input-density-bfactor.png) (`../input-density/input-density-bfactor`) relates the actual input density's sharpness (mean squared gradient divided by map variance, in Å⁻²) to contact-weighted B. Points are means within 10 equal-count B bins; bars are 95% complex-bootstrap intervals for the mean sharpness. Raw Spearman is -0.861; the partial rank correlation controlling for resolution and ligand heavy-atom count is -0.789. Both correlations are computed from the individual 733 complexes, not the ten bin means. The final bin covers B up to 77.51 Å² even though its mean x is about 50.07 Å². These are input-map descriptors for the same cohort, not predictions from the old exploratory model. Gradient sharpness responds to processing and noise as well as disorder; it is not a posterior uncertainty map.

**Why these additions.** Ligand/polymer RSCC add a model–map-support dimension; alternate locations add modeled heterogeneity beyond B magnitude. All eight previously decoded descriptors, including partial occupancy, relative pocket B, and resolution, are retained in [all-descriptor decoding](../representation/decoding-all-descriptors.png) (`../representation/decoding-all-descriptors`) so the reader can inspect the complete analysis. Partial occupancy does not improve (0.231 CoE vs 0.218 CoDE); resolution decoding is strong (0.628 vs 0.964) but is a global experimental-quality descriptor, not local binding-site uncertainty. No new decoder was fitted to choose the layout.

**Suggested text.** CoDE embeddings retain stronger signals of several crystallographic disorder and density-support descriptors than CoE embeddings. In an exploratory analysis of CL3 test complexes, ranking gains over CoE increase with contact B and normalized contact B, with a weaker directional pattern toward lower ligand RSCC. Polymer RSCC shows little stratification, and alternate-location fraction has a non-monotonic pattern with small positive-value groups. These associations motivate investigating disorder-related information as a contributor to downstream performance, without establishing its causal role.

**Scope.** Decoding uses random complex folds within this cohort, not independent protein-family-disjoint validation. Bootstrap intervals are pointwise, not multiplicity-adjusted, and do not resample training seeds or protein clusters. B-stratified analysis is exploratory and subgroup affinity distributions differ. The high-minus-low ranking-gain contrast is +0.0678, 95% CI [0.0036, 0.1334]. Subgroup MAE intervals cross zero, and all B-concat, spatial, occupancy-only and shuffled-B controls should remain available in the supplement.

**Model-comparison limitation.** The currently used manuscript CoE uses uniform masking, whereas CDG v2 uses atom-biased masking. A recipe-matched CoE comparison is needed to isolate density's contribution. The atomwise-B CNN is a small supervised auxiliary branch, not equivalently pretrained CoE+B.

Physical interpretation of RSCC: Tickle (2012), https://doi.org/10.1107/S0907444911035918. Alternate conformational heterogeneity: van Zundert et al. (2018), https://doi.org/10.1021/acs.jmedchem.8b01292.
''')
 baseline_arms=[arm for arm in ['DeepDTA','ProFSA','Nesso-1'] if arm in set(a.arm)]
 if baseline_arms:
  a.to_csv(REPRESENTATION/'decoding-statistics-with-baselines.csv',index=False)
  note='\n**Additional baseline decoding.** '+', '.join(baseline_arms)+' are evaluated on the identical 732 cases and nested ridge folds. Source features, layer, training protocol and OOF predictions are recorded under `../representation/baselines/<model>/`. DeepDTA and ProFSA were chosen using the prior manuscript CL3 affinity comparison; Nesso-1 was separately requested as an additional reference. No baseline was selected using decoding scores.\n'
  if 'DeepDTA' in baseline_arms:
   note+='DeepDTA uses a new CPU seed-0 CL123 affinity-supervised refit, selected by validation MSE across 100 epochs; its frozen 512-D ligand/protein CNN concatenation is probed. This is a rerun of the baseline recipe, not the original manuscript checkpoint. CoE/CoDE use their pretrained frozen encoders. The DeepDTA training seed is not five independent decoder seeds.\n'
  if 'ProFSA' in baseline_arms:
   note+='ProFSA uses the imported seed-0 1024-D concatenation of frozen pretrained ligand and pocket CLS tokens, before the affinity MLP. The uploaded configuration freezes the encoders, and DrugCLIPReg detaches both vectors before the supervised head. The ligand audit found no exact test-molecule matches among the examined pretraining entries; it does not rule out protein/pocket pretraining overlap.\n'
  if 'Nesso-1' in baseline_arms:
   note+='Nesso-1† uses the public v1.0.0 checkpoint without local affinity fitting. Its 384-D vector concatenates the mean final Pairformer z over protein–protein, protein–ligand and ligand–ligand pairs, fixed before decoding. † The original manuscript flags overlap with affinity pretraining data: this is an additional reference with affinity leakage, not a leakage-free baseline. Protein sequence and ligand SMILES are the inference inputs; no experimental B factors or maps are supplied.\n'
  caption=OUT/'caption-and-scope.md';caption.write_text(caption.read_text()+note)
 print('Updated A decoding and B five-metric ranking PNG/PDF/SVG.' + (' Supplementary exports updated.' if args.supplementary else ''))

if __name__=='__main__':main()

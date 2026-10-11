#!/usr/bin/env python3
"""Three-model B-tercile performance plot from saved seed-specific results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import draw as house
ROOT=Path(__file__).resolve().parents[3]
from .paths import ATOMIC_B as OUT, CL3

def main():
 df=pd.read_csv(OUT/'three-model-uncertainty-per-seed.csv')
 groups=['Low','Middle','High'];arms=['CoE','atomic-b','CoDE']
 labels={'CoE':'CoE','atomic-b':'CoE + per-atom pocket B (spatial CNN)','CoDE':'CoDE'}
 colors={'CoE':house.pale('CoE'),'atomic-b':house._gentle_color('#d58a2e'),'CoDE':house.pale('CoDE')}
 fig,axs=plt.subplots(2,2,figsize=(11.4,8.4))
 rows=[]
 for col,metric in enumerate(['mae','spearman']):
  for row in range(2):
   ax=axs[row,col]
   base=df[df.arm=='CoE'].set_index(['group','seed'])[metric]
   for ai,arm in enumerate(arms):
    means=[];sds=[]
    for group in groups:
     series=df[(df.arm==arm)&(df.group==group)].set_index(['group','seed'])[metric].sort_index()
     assert len(series)==5
     values=series if row==0 else series-base.loc[series.index]
     means.append(values.mean());sds.append(values.std(ddof=0))
     rows.append(dict(panel='absolute' if row==0 else 'change_vs_CoE',metric=metric,group=group,arm=arm,mean=means[-1],seed_sd=sds[-1]))
    ax.errorbar(np.arange(3)+(ai-1)*.055,means,yerr=sds,fmt='o-',color=colors[arm],capsize=4,ms=5,lw=1.7,label=labels[arm])
   ax.set_xticks(range(3),['Low B\nn=244','Middle B\nn=244','High B\nn=245']);ax.set_xlim(-.28,2.28);ax.grid(axis='y',alpha=house.BACKGROUND_ALPHA)
   if row:
    ax.axhline(0,c='#777777',lw=.8,ls='--',zorder=0)
    ax.set_xlabel('Contact-weighted B-factor tercile (higher B → greater disorder proxy)')
   if col==0:
    ax.set_ylabel('MAE (pK)' if row==0 else 'ΔMAE vs CoE (pK)')
    ax.set_title('Absolute error · lower is better' if row==0 else 'MAE change · negative favors the model',fontsize=11)
   else:
    ax.set_ylabel('Spearman ρ' if row==0 else 'ΔSpearman vs CoE')
    ax.set_title('Affinity ranking · higher is better' if row==0 else 'Ranking change · positive favors the model',fontsize=11)
 handles,legend=axs[0,0].get_legend_handles_labels()
 fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.927),ncol=3,frameon=False,fontsize=10)
 fig.suptitle('CoE, CoE + per-atom B, and CoDE across structural-uncertainty proxies\nCL123 train/validation → CL3 test; 5 probe seeds',y=.99,fontsize=13)
 fig.text(.5,.016,'Points: mean over 5 seeds. Bars: ±1 seed SD, not confidence intervals over test complexes.\nPer-atom B arm: frozen CoE + supervised 1 Å spatial CNN; pocket B only.',ha='center',va='bottom',fontsize=9,color='#444444')
 fig.subplots_adjust(top=.86,bottom=.15,left=.09,right=.98,hspace=.40,wspace=.25)
 if house.prepare_publication_figure(fig,OUT/'three-model-uncertainty'):
  for ext in house.retained_formats(OUT/'three-model-uncertainty'):fig.savefig(OUT/f'three-model-uncertainty.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig);pd.DataFrame(rows).to_csv(OUT/'three-model-uncertainty-plot-data.csv',index=False)
 # Check all plot means against the preceding table, independent of the plotting order.
 table=pd.read_csv(OUT/'three-model-uncertainty-table.csv').set_index(['group','arm'])
 for r in rows:
  key=r['metric'] if r['panel']=='absolute' else 'delta_'+r['metric']+'_vs_CoE'
  assert np.isclose(r['mean'],table.loc[(r['group'],r['arm']),key],atol=1e-8)
 print('Saved PNG/PDF/SVG and plot data; all 36 means match the table.')
if __name__=='__main__':main()

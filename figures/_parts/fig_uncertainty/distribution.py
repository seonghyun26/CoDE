#!/usr/bin/env python3
"""Empirical Contact-B distribution for the current uncertainty figure cohort."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .paths import ROOT, MOTIVATION
import draw as house
from .motivation import axes_style, save
sys.path.insert(0, str(ROOT))
from voxbind.splits import load_split, content_hash

STEM = 'uncertainty-bfactor-ecdf'
GUIDES = (20., 30.)


def plot_distribution(source, expected_n=733, create=False):
    source=Path(source)
    frame=pd.read_csv(source)[['pid','b_contact_invd2']]
    assert len(frame)==expected_n and frame.pid.is_unique
    assert np.isfinite(frame.b_contact_invd2).all()
    if expected_n==733:
        assert set(frame.pid)==set(load_split('lp_edrscc_v2_cl123')['test'])
    frame=frame.sort_values(['b_contact_invd2','pid']).reset_index(drop=True)
    x=frame.b_contact_invd2.to_numpy()
    values, counts=np.unique(x, return_counts=True)
    probabilities=np.cumsum(counts)/len(x)
    assert np.all(np.diff(probabilities)>0) and probabilities[-1]==1.
    xmax=max(40.,float(np.ceil(values[-1]/10)*10))
    curve_x=np.r_[0.,values,xmax]
    curve_y=np.r_[0.,probabilities,1.]
    color=house.pale('CoDE')
    guide_rows=[]
    with plt.rc_context({**house.PCSZ_RC,'savefig.bbox':None}):
        fig,ax=plt.subplots(figsize=house.PCSZ_STRAIN_SIZE,facecolor=house.PCSZ_BG)
        axes_style(ax)
        ax.step(curve_x,curve_y,where='post',color=color,lw=2.5*house.PCSZ_LW_SCALE,zorder=3)
        ax.fill_between(curve_x,0,curve_y,step='post',color=color,alpha=house.BACKGROUND_ALPHA,zorder=1)
        for cut in GUIDES:
            cumulative_n=int(np.count_nonzero(x<=cut));prob=cumulative_n/len(x)
            guide_rows.append(dict(b_factor=cut,n_below=int(np.count_nonzero(x<cut)),n_at_or_below=cumulative_n,cumulative_probability=prob))
            ax.vlines(cut,0,prob,color='#9AA0A6',lw=1.1,ls=(0,(4,3)),zorder=2)
            ax.hlines(prob,0,cut,color='#9AA0A6',lw=1.1,ls=(0,(4,3)),zorder=2)
            ax.plot(cut,prob,'o',ms=6.5,mfc='white',mec=color,mew=1.6,zorder=4)
            ax.annotate(f'{100*prob:.1f}%',(cut,prob),xytext=(10,-12),textcoords='offset points',
                        ha='left',va='top',fontsize=11,color=house.PCSZ_INK)
        ax.set_xlim(0,xmax);ax.set_ylim(0,1.02)
        ax.set_xticks(np.arange(0,xmax+1,10))
        ticks=np.arange(0,1.01,.2);ax.set_yticks(ticks,[f'{t:.1f}' for t in ticks])
        ax.set_xlabel('B-factor (Å²)',fontsize=house.PCSZ_LABEL_FS)
        ax.set_ylabel('Cumulative\nprobability',fontsize=house.PCSZ_LABEL_FS,labelpad=house.PCSZ_YPAD)
        ax.text(.97,.10,f'n = {len(x)}',transform=ax.transAxes,ha='right',va='bottom',fontsize=11,color='#555555')
        house._pcsz_comparison_layout(fig,ax)
        save(fig,STEM,allow_new=create)
    out=source.parent
    frame.to_csv(out/f'{STEM}-points.csv',index=False)
    pd.DataFrame(dict(b_factor=values,count=counts,cumulative_probability=probabilities)).to_csv(out/f'{STEM}-curve.csv',index=False)
    pd.DataFrame(guide_rows).to_csv(out/f'{STEM}-guides.csv',index=False)
    metadata=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  descriptor='b_contact_invd2',unit='Å²',n=len(x),pid_hash=content_hash(frame.pid.tolist()),
                  definition='F(b) = number of complexes with contact-weighted B <= b / n; one observation per complex; ties grouped; no smoothing or binning',
                  guides=list(GUIDES),guides_scope='illustrate the requested 20/30 alternative, independently of the affinity panels active bin configuration',
                  min=float(x.min()),median=float(np.median(x)),max=float(x.max()),
                  figure=str(MOTIVATION/f'{STEM}.png'),reproduce='cd figures && python -m _parts.fig_uncertainty.distribution')
    (out/f'{STEM}-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/f'{STEM}.md').write_text(f'''# Contact-weighted B-factor distribution

Empirical cumulative distribution for {len(x)} complexes in the current cohort. Each complex contributes one `b_contact_invd2` value. The y axis is the fraction with B ≤ the x value; ties are grouped and no smoothing or binning is applied. B is the same contact-weighted crystallographic descriptor used in the affinity analysis, not a model-predicted uncertainty score.

The 20 and 30 Å² guides illustrate the recently requested alternative boundaries. They do not change affinity-bin selection. Their cumulative probabilities are {guide_rows[0]['cumulative_probability']:.6f} and {guide_rows[1]['cumulative_probability']:.6f}. Median Contact B is {np.median(x):.6f} Å².

[PNG](../../../motivation/{STEM}.png) · [PDF](../../../motivation/{STEM}.pdf)

Reproduce: `cd figures && python -m _parts.fig_uncertainty.distribution`. Exact points, ECDF steps, guide counts and source hash are saved beside this note.
''')
    print(json.dumps(dict(n=len(x),median=float(np.median(x)),guides=guide_rows,figure=str(MOTIVATION/f'{STEM}.png')),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create',action='store_true',help='Explicitly register and create this new requested figure')
    args=parser.parse_args()
    manifest_path=MOTIVATION/'figure-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if args.create and STEM not in manifest['figures']:
        manifest['figures'].append(STEM)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.create and (STEM not in manifest['figures'] or not any((MOTIVATION/f'{STEM}.{ext}').exists() for ext in ['png','pdf'])):
        print('Skipped removed figure:',STEM);return
    active=json.loads((MOTIVATION/'active-analysis.json').read_text())
    plot_distribution(active['source'],active['n_affinity'],create=args.create)
    from .notes_current import write_notes
    write_notes()


if __name__=='__main__':main()

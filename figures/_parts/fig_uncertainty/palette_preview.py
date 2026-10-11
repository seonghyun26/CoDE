#!/usr/bin/env python3
"""Review-only palette swatches; does not modify any result figure or score."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .paths import MOTIVATION_ARCHIVE
import draw as house
PAIRS=[('1  Coral + olive-gold','#E5A18D','#C9B874'),
       ('2  Current: coral + olive',house.soft('DeepDTA'),house.soft('ProFSA')),
       ('3  Apricot + dusty cyan','#EDC29C','#94C9C5')]
MODELS=['DeepDTA','ProFSA','Nesso-1','CoE','CoDE']
fig,ax=plt.subplots(figsize=(12,4.2));fig.patch.set_facecolor('white');ax.set_xlim(-2.5,5);ax.set_ylim(-.55,3.35);ax.axis('off')
for j,name in enumerate(MODELS):ax.text(j+.4,3.12,name,ha='center',va='center',fontsize=12,color='#222222')
for i,(label,d,p) in enumerate(PAIRS):
 y=2.45-i*1.05;ax.text(-2.35,y+.2,label,ha='left',va='center',fontsize=12,color='#222222')
 for j,c in enumerate([d,p,house.soft('Nesso-1'),house.color('CoE'),house.soft('CoDE')]):
  ax.add_patch(FancyBboxPatch((j,y),.8,.42,boxstyle='round,pad=0,rounding_size=.04',facecolor=c,edgecolor='none'))
  ax.text(j+.4,y-.17,c,ha='center',va='center',fontsize=10,color='#444444')
fig.subplots_adjust(left=.025,right=.99,bottom=.04,top=.96)
out=MOTIVATION_ARCHIVE/'style-previews';out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(out/f'baseline-palette-options.{ext}',dpi=180)

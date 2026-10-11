#!/usr/bin/env python3
"""Candidate-only Color / Highlight palette; never edits the live palette or figures."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import draw as house
from .methods import PAL_ORDER, PAL_EXTRA     # palette-only constants live with the palette
STEM = ROOT / 'fig-palette' / 'palette-method-candidate'
PALETTE_PAIRS = {'Reference ligand': ('#B4B8BE', '#A2A8AD'),
 'AR': ('#435C91', '#2F4472'),
 'Pocket2Mol': ('#9CCFDF', '#6FB3C8'),
 'DiffSBDD': ('#A98BE3', '#8E63CF'),
 'TargetDiff': ('#C878D6', '#A950B7'),
 'DecompDiff': ('#F08EAE', '#D9638D'),
 'FuncBind': ('#E8C86A', '#C9A947'),
 'VoxBind': ('#FFB377', '#FF913A'),
 'CoDE': ('#8092F6', '#4761F2'),
 'CoE': ('#7DCE96', '#4BB572'),
 'DeepDTA': ('#58BDB8', '#329B96'),
 'ProFSA': ('#8D76CF', '#7450BC'),
 'Nesso-1': ('#E78E8D', '#D85B56')}


def main():
    # Confirm that generating the candidate leaves existing figure outputs untouched.
    protected = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in ROOT.rglob('*') if p.suffix in ('.png', '.pdf', '.svg', '.csv')
                 and p.with_suffix('') != STEM}
    rows = []
    for method in PAL_ORDER + PAL_EXTRA:
        if method == 'FuncBind vanilla':
            continue
        pale, soft = house.LEGACY_PALE[house.ALIASES.get(method, method)], house.soft(method)
        unused = method == 'Ours v2' or method.startswith('FuncBind ft ')
        if unused:
            color = highlight = '#FFFFFF'
        else:
            color, highlight = PALETTE_PAIRS[method]
        clamped = False
        rows.append(dict(method=method, label=house.DISPLAY.get(method, method),
                         current_pale=pale.upper(), current_soft=soft.upper(),
                         color=color, highlight=highlight, highlight_saturation_clamped=clamped,
                         hatch='///' if unused else '', hatch_color='#000000' if unused else ''))
    with plt.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'mathtext.default': 'regular'}):
        fig, ax = plt.subplots(figsize=(8.0, .42 * len(rows) + 1.55), dpi=220)
        fig.patch.set_facecolor('white')
        ax.set_position([.035, .045, .93, .84])
        ax.set_xlim(-2.1, 2.65);ax.set_ylim(-len(rows) + .05, 1.1);ax.axis('off')
        fig.text(.5, .974, 'Palette method candidate', ha='center', fontsize=17, weight='bold')
        fig.text(.5, .944, 'Preview only · existing palette and figures unchanged', ha='center', fontsize=10, color='#555555')
        for j, (key, title, subtitle) in enumerate([
            ('color', 'Color', 'Exact colors'),
            ('highlight', 'Highlight', 'Exact highlights')]):
            x = j * 1.35
            ax.text(x + .50, .75, title, ha='center', fontsize=14, weight='bold')
            ax.text(x + .50, .41, subtitle, ha='center', fontsize=8.5, color='#555555')
            for i, row in enumerate(rows):
                y = -i - .2
                if j == 0:
                    ax.text(-.18, y, row['label'], ha='right', va='center', fontsize=12, color='#222222')
                ax.add_patch(Rectangle((x, y - .17), 1.0, .50,
                                       facecolor=row[key], edgecolor=row['hatch_color'] or 'none',
                                       hatch=row['hatch'] or None, linewidth=0))
                ax.text(x + .50, y - .43, 'Black hatch' if row['hatch'] else row[key], ha='center', va='center', fontsize=9, color='black')
        for ext in ['png', 'pdf']:
            fig.savefig(STEM.with_suffix('.' + ext), facecolor='white', bbox_inches='tight', pad_inches=.10)
        plt.close(fig)
    with STEM.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    STEM.with_suffix('.json').write_text(json.dumps(dict(
        status='candidate_only_not_applied', proposed_names={'PALE': 'Color', 'SOFT': 'Highlight'},
        construction='Exact user-supplied Color and Highlight pairs; no brightness, saturation, white-mix, or opacity transform. Inactive variants use black hatching; FuncBind vanilla is omitted.',
        palette_pairs=PALETTE_PAIRS,
        rows=rows), indent=2) + '\n')
    assert all(p.exists() and (p.stat().st_size, p.stat().st_mtime_ns) == value for p, value in protected.items())
    print('Saved candidate:', STEM.with_suffix('.png'))
    print('PASS:', len(protected), 'existing figure/table outputs unchanged')


if __name__ == '__main__':
    main()

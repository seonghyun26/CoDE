#!/usr/bin/env python3
"""Preview three color families from light to dark; never recolor manuscript plots."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
STEM = ROOT / 'fig-palette' / 'palette-method-candidate-green-orange-blue'
PALETTE = {
    'Green': ['#D7EEDB', '#AEDBB8', '#7AC493', '#459D6A', '#246642'],
    'Orange': ['#FFE5C9', '#FBC895', '#F2A45F', '#DB7B32', '#9E4C1E'],
    'Blue': ['#DCEBFA', '#B4D2EE', '#7CAEDC', '#487FB8', '#285184'],
}
LEVELS = ['Light', 'Soft', 'Medium', 'Deep', 'Dark']


def luminance(hex_color):
    rgb = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4
           for v in to_rgb(hex_color)]
    return sum(v * w for v, w in zip(rgb, (.2126, .7152, .0722)))


def main():
    protected = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in ROOT.rglob('*')
                 if p.suffix in ('.png', '.svg', '.pdf', '.csv', '.json')
                 and p.with_suffix('') != STEM}
    rows = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(10.2, 4.7), dpi=200)
        fig.patch.set_facecolor('white')
        ax.set_position([.025, .08, .95, .75])
        ax.set_xlim(-1.1, 5.9)
        ax.set_ylim(-.50, 3.4)
        ax.axis('off')
        fig.text(.5, .945, 'Palette candidate · Green / Orange / Blue',
                 ha='center', fontsize=17, weight='bold')
        fig.text(.5, .875, 'Light → Dark  ·  Preview only',
                 ha='center', fontsize=11, color='#555555')
        for j, level in enumerate(LEVELS):
            ax.text(j * 1.2 + .48, 3.03, level, ha='center', fontsize=11)
        for i, (family, colors) in enumerate(PALETTE.items()):
            values = [luminance(c) for c in colors]
            assert all(a > b for a, b in zip(values, values[1:]))
            y = 2.3 - i * 1.12
            ax.text(-.18, y + .08, family, ha='right', va='center', fontsize=13)
            for j, (level, color) in enumerate(zip(LEVELS, colors)):
                x = j * 1.2
                ax.add_patch(Rectangle((x, y - .17), .96, .54,
                                       facecolor=color, edgecolor='none', linewidth=0))
                ax.text(x + .48, y - .35, color, ha='center', va='top',
                        fontsize=10, color='black')
                rows.append({'family': family, 'level': level, 'color': color})
        for ext in ('png', 'pdf'):
            fig.savefig(STEM.with_suffix('.' + ext), facecolor='white',
                        bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
    with STEM.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['family', 'level', 'color'])
        writer.writeheader()
        writer.writerows(rows)
    STEM.with_suffix('.json').write_text(json.dumps({
        'status': 'candidate_only_not_applied',
        'description': 'Three families, five levels each, with strictly decreasing relative luminance. No method assignments or approved palette changes.',
        'rows': rows}, indent=2) + '\n')
    changed = [str(p) for p, old in protected.items()
               if not p.exists() or (p.stat().st_size, p.stat().st_mtime_ns) != old]
    if changed:
        raise RuntimeError('Other outputs changed during preview generation: ' + ', '.join(changed))
    print('Saved:', STEM.with_suffix('.png'))
    print('PASS: all three ramps get darker; existing outputs unchanged')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Preview the eight user-specified colors at full opacity; no plot changes."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
STEM = ROOT / 'fig-palette' / 'palette-method-candidate-eight-colors'
COLORS = [
    ('Blue', '#7588F5'),
    ('Periwinkle', '#7F86E8'),
    ('Lavender', '#A97AD8'),
    ('Orchid Pink', '#D27ABE'),
    ('Soft Rose', '#EE9FA8'),
    ('Coral', '#F28D7A'),
    ('Peach', '#FFAC6B'),
    ('Misty Cyan', '#79C7D3'),
]
METHOD_COLORS = [
    ('AR', 'Periwinkle'), ('Pocket2Mol', 'Lavender'),
    ('DiffSBDD', 'Orchid Pink'), ('DecompDiff', 'Soft Rose'),
    ('FuncBind', 'Coral'), ('TargetDiff', 'Misty Cyan'),
    ('VoxBind', 'Peach'), ('CoDE', 'Blue'),
]


def main():
    protected = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in ROOT.rglob('*')
                 if p.suffix in ('.png', '.svg', '.pdf', '.csv', '.json')
                 and p.with_suffix('') != STEM}
    color_lookup = dict(COLORS)
    rows = [dict(order=i + 1, method=method, name=name, color=color_lookup[name], alpha=1.0)
            for i, (method, name) in enumerate(METHOD_COLORS)]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(10.0, 4.6), dpi=200)
        fig.patch.set_facecolor('white')
        ax.set_position([.035, .055, .93, .76])
        ax.set_xlim(-.10, 7.75)
        ax.set_ylim(-.20, 3.1)
        ax.axis('off')
        fig.text(.5, .945, 'Method palette candidate · 8 colors', ha='center', fontsize=18, weight='bold')
        fig.text(.5, .875, 'Exact hex colors · 100% opacity · Preview only',
                 ha='center', fontsize=11, color='#555555')
        for i, row in enumerate(rows):
            method, color = row['method'], row['color']
            x, y = (i % 4) * 2.0, 2.2 - (i // 4) * 1.6
            ax.text(x + .82, y + .51, method, ha='center', fontsize=12)
            ax.add_patch(Rectangle((x, y - .13), 1.64, .50,
                                   facecolor=color, edgecolor='none', linewidth=0, alpha=1.0))
            ax.text(x + .82, y - .36, color, ha='center', va='top', fontsize=11, color='black')
        for ext in ('png', 'pdf'):
            fig.savefig(STEM.with_suffix('.' + ext), facecolor='white', bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
    with STEM.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    STEM.with_suffix('.json').write_text(json.dumps({
        'status': 'candidate_only_not_applied',
        'description': 'Eight colors in reading order, fully opaque; Blue updated to #7588F5; Orchid Pink darkened from #D98FC8 to #D27ABE at matching HLS hue/saturation; AR through TargetDiff use Periwinkle, Lavender, Orchid Pink, Soft Rose, Coral, and Misty Cyan in that order. VoxBind uses Peach and CoDE uses Blue. Candidate only; approved palette and manuscript figures are unchanged.',
        'rows': rows}, indent=2) + '\n')
    changed = [str(p) for p, old in protected.items()
               if not p.exists() or (p.stat().st_size, p.stat().st_mtime_ns) != old]
    if changed:
        raise RuntimeError('Other outputs changed during generation: ' + ', '.join(changed))
    print('Saved:', STEM.with_suffix('.png'))
    print('PASS: existing outputs unchanged')


if __name__ == '__main__':
    main()

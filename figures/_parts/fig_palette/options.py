#!/usr/bin/env python3
"""Preview lighter, less saturated method colors without changing the live palette."""
import colorsys
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Rectangle

import draw as house

OUT = Path(__file__).resolve().parents[2] / 'fig-palette' / 'proposals'
METHODS = ['AR', 'Pocket2Mol', 'DiffSBDD', 'TargetDiff', 'DecompDiff', 'FuncBind', 'VoxBind', 'CoE', 'CoDE']
OPTIONS = [('Current', 1., 0.), ('A · Gentle', .85, .12), ('B · Balanced', .70, .20), ('C · Soft', .55, .28)]


def pastel(color, saturation, white_lightness):
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    return to_hex(colorsys.hls_to_rgb(h, l + (1 - l) * white_lightness, s * saturation)).upper()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    current = {name: house.soft(name) for name in METHODS}
    proposals = {label: {name: pastel(color, saturation, white_lightness) for name, color in current.items()}
                 for label, saturation, white_lightness in OPTIONS}
    fig, ax = plt.subplots(figsize=(12.8, 7.9), facecolor='white')
    ax.set_position([.025, .04, .95, .86]);ax.set_axis_off()
    ax.set_xlim(-1.4, 8.0);ax.set_ylim(-.4, len(METHODS) + .8)
    fig.suptitle('Method palette candidates — preview only', x=.51, y=.978, fontsize=18)
    fig.text(.51, .934, 'Same hues; progressively lower saturation and higher lightness. No live figure colors changed.',
             ha='center', fontsize=11, color='#555555')
    for j, (label, sat, light) in enumerate(OPTIONS):
        x = j * 1.95
        ax.text(x + .75, len(METHODS) + .35, label, ha='center', fontsize=14, weight='bold')
        subtitle = 'Existing palette' if j == 0 else f'S × {sat:.2f}; L + {light:.0%} of remaining range'
        ax.text(x + .75, len(METHODS) + .02, subtitle, ha='center', fontsize=8.6, color='#555555')
        for i, name in enumerate(METHODS):
            y = len(METHODS) - 1 - i
            color = proposals[label][name]
            if j == 0:ax.text(-.18, y + .42, name, ha='right', va='center', fontsize=12)
            ax.add_patch(Rectangle((x, y + .2), 1.5, .40, facecolor=color, edgecolor='none'))
            ax.plot([x, x + 1.5], [y + .75] * 2, color=color, lw=3.5, solid_capstyle='round')
            ax.plot(x + .75, y + .75, 'o', ms=6, mfc='white', mec=color, mew=2)
            ax.text(x + .75, y + .04, color, ha='center', va='center', fontsize=8.5, color='#333333')
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'method-palette-options.{ext}', dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    (OUT / 'method-palette-options.json').write_text(json.dumps(dict(status='preview_only_not_applied',
        source='draw.soft(method), the palette used by the ECDF and redocking panels',
        transformation='HLS hue fixed; S multiplied; L moved toward 1 by the specified fraction',
        options=proposals, reference_color_unchanged=house.REF_COLOR), indent=2) + '\n')
    assert current == {name: house.soft(name) for name in METHODS}
    print('Preview only:', OUT / 'method-palette-options.png')
    print(json.dumps(proposals, indent=2))


if __name__ == '__main__':main()

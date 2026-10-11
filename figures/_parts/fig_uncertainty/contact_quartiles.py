#!/usr/bin/env python3
"""Recompute a separate Contact-B quartile preview using the active cohort."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyse_quartiles import analyse_quartiles, PREFIX
from . import motivation
from . import motivation_variants as variants
from .paths import BASE, MOTIVATION


def main():
    active = json.loads((MOTIVATION / 'active-analysis.json').read_text())
    source = Path(active['source'])
    out = BASE / 'exploratory' / 'contact-b-quartiles-comparison'
    table = analyse_quartiles(source, out, active['n_affinity'])
    frame = pd.read_csv(source)
    assert table.n.max() - table.n.min() <= 1
    saved = pd.read_csv(source.parent / f'{PREFIX}-ranking.csv')
    pd.testing.assert_frame_equal(table, saved, check_exact=False, atol=1e-12, rtol=1e-12)
    current = pd.read_csv(active['contact_b_ranking_statistics'])
    comparison = pd.concat([current.assign(scheme='Current fixed ranges'),
                            table.assign(scheme='Quartiles')], ignore_index=True)
    comparison.to_csv(out / 'three-vs-four-slices.csv', index=False)
    decoded = motivation.load_decoding()

    def export(fig, name, **kwargs):
        right = fig.axes[1]
        for artist in list(right.texts):
            artist.remove()
        # The Q1 lower confidence bound is below the current figure's -0.05 limit.
        right.set_ylim(-.08, .15)
        motivation.save(fig, name, out=out, **kwargs)

    original_save = variants.save
    variants.save = export
    try:
        with plt.rc_context({**variants.house.PCSZ_RC, 'savefig.bbox': None}):
            variants.combined_contact(
                decoded, table, name='uncertainty-combined-contactb-quartiles',
                right_ylim=(-.08, .15), distribution_frame=frame, show_models=False)
    finally:
        variants.save = original_save
    lines = [
        '# Contact-B: three fixed ranges versus four quartiles', '',
        f'Active cohort: {active["cohort"]}; {len(frame)} affinity complexes.',
        'The current figure uses B < 20, 20–30, and ≥30 Å², with 301, 242, and 190 complexes.',
        'Quartiles use empirical 25%, 50%, and 75% thresholds; ties stay together.',
        'Each group has 183 or 184 complexes because 733 is not divisible by four.', '',
        '| Quartile | B range (Å², rounded) | n | CoE Spearman | CoDE Spearman | ΔSpearman | 95% CI |',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    for row in table.itertuples():
        lo, hi = row.lower_bound, row.upper_bound
        label = f'<{hi:.2f}' if pd.isna(lo) else (f'≥{lo:.2f}' if pd.isna(hi) else f'{lo:.2f}–{hi:.2f}')
        lines.append(f'| {row.quartile} | {label} | {row.n} | {row.coe_spearman:.3f} | {row.code_spearman:.3f} | {row.estimate:+.3f} | [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}] |')
    contrast = pd.read_csv(out / f'{PREFIX}-contrast.csv').iloc[0]
    lines += ['',
        'The lowest quartile has a small negative point estimate; Q2 becomes positive, and Q3/Q4 have similar positive gains. The effect therefore rises and then plateaus rather than increasing monotonically.',
        'Q1 and Q2 confidence intervals include zero; Q3 and Q4 intervals exclude zero.',
        f'Q4 minus Q1 improvement: {contrast.estimate:+.3f}, 95% CI [{contrast.ci_lo:+.3f}, {contrast.ci_hi:+.3f}].', '',
        'Spearman is computed within each bin per fitted seed and averaged over five seeds. Intervals use 8,000 paired-complex bootstrap draws, holding bins and fitted seeds fixed; pointwise, without multiplicity correction.',
        'All 733 complexes contribute, including B > 40; the plotted x window is 10–40 and points are within-bin medians. The preview extends the y minimum to −0.08 to show the full Q1 interval.',
        'The left panel retains its existing 732-complex decoding analysis.', '',
        'Reproduce: `cd figures && python -m _parts.fig_uncertainty.contact_quartiles`',
        f'Source SHA256: `{hashlib.sha256(source.read_bytes()).hexdigest()}`',
    ]
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'Preview and verified comparison: {out}')


if __name__ == '__main__':
    main()

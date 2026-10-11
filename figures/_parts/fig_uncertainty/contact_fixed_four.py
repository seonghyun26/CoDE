#!/usr/bin/env python3
"""Preview the combined figure with Contact-B cutoffs at 20, 25, and 30 Å²."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyse_fixed_b import analyse
from . import motivation
from . import motivation_variants as variants
from .paths import BASE, MOTIVATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--error-bars', choices=['ci', 'seed-sd'], default='ci')
    args = parser.parse_args()
    active = json.loads((MOTIVATION / 'active-analysis.json').read_text())
    source = Path(active['source'])
    out = BASE / 'exploratory' / 'contact-b-fixed-four-comparison'
    prefix = 'fixed-contact-b-20-25-30'
    if args.error_bars == 'ci':
        table = analyse([20., 25., 30.], prefix, source, out, active['n_affinity'])
    else:
        provenance = json.loads((out / f'{prefix}-provenance.json').read_text())
        assert provenance['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        table = pd.read_csv(out / f'{prefix}-ranking.csv')
    frame = pd.read_csv(source)
    members = pd.read_csv(out / f'{prefix}-bin-membership.csv')
    assert set(members.pid) == set(frame.pid) and members.pid.is_unique
    old = pd.read_csv(active['contact_b_ranking_statistics'])
    # Splitting the middle group must leave the outer groups unchanged.
    if active['contact_b_binning'] == 'absolute_20_30':
        cols = ['n', 'coe_spearman', 'code_spearman', 'estimate']
        np.testing.assert_allclose(table.iloc[[0, 3]][cols], old.iloc[[0, 2]][cols], atol=1e-12)
    pd.concat([old.assign(scheme='Current fixed ranges'),
               table.assign(scheme='Fixed four ranges')]).to_csv(out / 'three-vs-four-slices.csv', index=False)
    plot_table = table.copy()
    if args.error_bars == 'seed-sd':
        seeds = pd.read_csv(out / f'{prefix}-per-seed.csv')
        assert not seeds.duplicated(['group_index', 'seed']).any()
        assert seeds.groupby('group_index').seed.apply(lambda x: set(x) == set(range(5))).all()
        np.testing.assert_allclose(seeds.delta, seeds.code_spearman - seeds.coe_spearman, atol=1e-12)
        summary = seeds.groupby('group_index').delta.agg(['mean', 'std', 'count']).reindex(table.group_index)
        np.testing.assert_allclose(summary['mean'], table.estimate, atol=1e-12)
        summary.insert(0, 'range_label', table.range_label.to_numpy())
        summary['sd_lower'] = summary['mean'] - summary['std']
        summary['sd_upper'] = summary['mean'] + summary['std']
        summary.to_csv(out / f'{prefix}-seed-sd.csv')
        print(summary.to_string(), flush=True)
        # The existing renderer accepts generic interval endpoints under these names.
        plot_table['ci_lo'] = summary.sd_lower.to_numpy()
        plot_table['ci_hi'] = summary.sd_upper.to_numpy()
    decoded = motivation.load_decoding()
    ylim = (min(-.08, float(np.floor(table.ci_lo.min() / .05) * .05)),
            max(.15, float(np.ceil(table.ci_hi.max() / .05) * .05)))

    def export(fig, name, **kwargs):
        right = fig.axes[1]
        for artist in list(right.texts):
            artist.remove()
        error_label = 'Mean ± SD (5 seeds)' if args.error_bars == 'seed-sd' else '95% bootstrap CI'
        right.text(.5, .95, error_label, transform=right.transAxes,
                   ha='center', va='top', fontsize=10, color=variants.house.PCSZ_INK)
        if args.error_bars == 'seed-sd':
            # Small SD bars must remain visible around the mean markers.
            for line in right.lines:
                line.set_markersize(4.5 * 1.4)
                line.set_markeredgewidth(1.4)
                line.set_linewidth(2.4)
            for interval in right.collections:
                interval.set_linewidth(1.4)
                segments = interval.get_segments()
                for segment in segments:
                    if np.isclose(segment[0, 1], segment[1, 1]):
                        center = segment[:, 0].mean()
                        segment[:, 0] = center + (segment[:, 0] - center) * 1.6
                interval.set_segments(segments)
        right.set_ylim(*ylim)
        ticks = np.arange(np.ceil(ylim[0] / .05) * .05, ylim[1] + .001, .05)
        right.set_yticks(ticks, [f'{value:.2f}'.replace('-', '−') for value in ticks])
        motivation.save(fig, name, out=out, **kwargs)

    original_save = variants.save
    variants.save = export
    try:
        with plt.rc_context({**variants.house.PCSZ_RC, 'savefig.bbox': None}):
            variants.combined_contact(decoded, plot_table,
                name='uncertainty-combined-contactb-fixed-20-25-30' + ('-seed-sd' if args.error_bars == 'seed-sd' else ''),
                right_ylim=ylim, distribution_frame=frame, show_models=False)
    finally:
        variants.save = original_save
    if args.error_bars == 'seed-sd':
        lines = ['# Contact-B: mean ± SD across five seeds', '',
            'For each bin and seed s, delta_s = Spearman(CoDE_s) − Spearman(CoE_s).',
            'Points show mean(delta_s); error bars show ± sample SD(delta_s), ddof=1, across five matched seed IDs.',
            'The same evaluation complexes and bins are held fixed. These bars describe seed variability, not evaluation-sample uncertainty or confidence intervals.',
            'Underlying bootstrap CI results remain in the companion ranking table. The y-axis matches the CI preview for direct comparison.', '',
            'Reproduce: `cd figures && python -m _parts.fig_uncertainty.contact_fixed_four --error-bars seed-sd`', '',
            summary.to_csv()]
        (out / 'seed-sd.md').write_text('\n'.join(lines) + '\n')
        print(f'Figure and seed SD: {out}')
        return
    lines = ['# Contact-B: four fixed ranges', '',
        f'Active cohort: {active["cohort"]}, {len(frame)} affinity complexes.',
        'Intervals: B < 20, 20 ≤ B < 25, 25 ≤ B < 30, B ≥ 30 Å².',
        f'Exact boundary-value counts: { {cut: int(frame.b_contact_invd2.eq(cut).sum()) for cut in [20, 25, 30]} }.', '',
        '| B range (Å²) | n | CoE Spearman | CoDE Spearman | ΔSpearman | 95% CI |',
        '|---|---:|---:|---:|---:|---|']
    for row in table.itertuples():
        lines.append(f'| {row.range_label} | {row.n} | {row.coe_spearman:.3f} | {row.code_spearman:.3f} | {row.estimate:+.3f} | [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}] |')
    lines += ['',
        'Spearman is evaluated within each bin and averaged over five fitted seeds. Pointwise 95% intervals use 8,000 paired-complex bootstrap draws with bins and fitted seeds fixed; no multiplicity adjustment.',
        'All 733 affinity complexes contribute; the x window is 10–40 Å² and points are within-bin medians. Left-panel decoding retains its existing 732-complex analysis.',
        'Reproduce: `cd figures && python -m _parts.fig_uncertainty.contact_fixed_four`']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'Figure and statistics: {out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Explicit combined/contact-B/RMSE variants of the current motivation figures.

The usual plot_uncertainty_motivation.py command still exports only A and B.
This command adds the requested variants using saved five-seed predictions.
Use --plots-only to reuse the saved RMSE bootstrap results after style edits.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import draw as house
from .motivation import MAIN_KEYS, load_decoding, decoding, axes_style, save
from .ranking_metrics import DIRECTION, N_BOOT, BATCH, RNG_SEED, panel
from .paths import CL3, MOTIVATION, RANKING_METRICS as DATA

SOURCE = CL3 / 'per-complex.csv'
CONTACT = 'b_contact_invd2'


def rmse(pred, truth):
    return np.sqrt(np.mean((pred - truth) ** 2, axis=-1))


def bootstrap_rmse(a, b, truth, rng):
    values = []
    for _ in range(N_BOOT // BATCH):
        idx = rng.integers(len(truth), size=(BATCH, len(truth)))
        target = truth[idx][:, None, :]
        av = a[:, idx].transpose(1, 0, 2)
        bv = b[:, idx].transpose(1, 0, 2)
        values.append((rmse(bv, target) - rmse(av, target)).mean(1))
    boot = np.concatenate(values)
    assert len(boot) == N_BOOT and np.isfinite(boot).all()
    return boot


def point_and_boot(block, rng):
    target = block.y_true.to_numpy()
    a, b = [np.stack([block[f'{arm}_seed{s}_prediction'].to_numpy() for s in range(5)]) for arm in ['CoE', 'CoDE']]
    ra, rb = rmse(a, target[None]), rmse(b, target[None])
    boot = bootstrap_rmse(a, b, target, rng)
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(coe_rmse=float(ra.mean()), code_rmse=float(rb.mean()), estimate=float((rb - ra).mean()),
                ci_lo=float(lo), ci_hi=float(hi)), boot, ra, rb


def analyse_rmse():
    frame = pd.read_csv(SOURCE).set_index('pid')
    ranking = pd.read_csv(DATA / 'ranking-by-metric.csv')
    membership = pd.read_csv(DATA / 'bin-membership.csv')
    assert len(frame) == 733 and frame.index.is_unique
    rows, seed_rows, contrasts = [], [], []
    metadata = [col for col in ranking.columns if col not in ['coe_spearman', 'code_spearman', 'estimate', 'ci_lo', 'ci_hi']]
    for j, key in enumerate(MAIN_KEYS):
        rng = np.random.default_rng(RNG_SEED + j)
        points, samples = {}, {}
        for old in ranking.loc[ranking.descriptor == key].sort_values('group_index').to_dict('records'):
            member = membership[(membership.descriptor == key) & (membership.group_index == old['group_index'])]
            block = frame.loc[member.pid]
            assert len(block) == old['n']
            np.testing.assert_allclose(block[key], member.x, rtol=0, atol=1e-12)
            stats, boot, ra, rb = point_and_boot(block, rng)
            rows.append({**{col:old[col] for col in metadata}, **stats})
            points[old['group_index']] = stats['estimate']
            samples[old['group_index']] = boot
            for seed in range(5):
                seed_rows.append(dict(descriptor=key, group=old['group'], group_index=old['group_index'], seed=seed,
                                      coe_rmse=ra[seed], code_rmse=rb[seed], delta_rmse=rb[seed] - ra[seed]))
        # A positive value here means the RMSE reduction is larger for the higher proxy group.
        direction = DIRECTION[key]
        contrast = -direction * (samples[2] - samples[0])
        lo, hi = np.quantile(contrast, [.025, .975])
        contrasts.append(dict(descriptor=key, contrast='RMSE improvement: higher-proxy minus lower-proxy',
                              estimate=-direction * (points[2] - points[0]), ci_lo=lo, ci_hi=hi))
        print(key, 'RMSE(CoDE)-RMSE(CoE):', [round(points[g], 5) for g in range(3)], flush=True)
    table = pd.DataFrame(rows)
    # Independent check against the canonical full-cohort per-seed evaluation table.
    full, _, ra, rb = point_and_boot(frame, np.random.default_rng(RNG_SEED + 100))
    source_metrics = pd.read_csv(CL3 / 'per-seed-metrics.csv')
    for arm, actual in [('CoE', ra), ('CoDE', rb)]:
        expected = source_metrics[source_metrics.arm == arm].sort_values('seed').rmse.to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    pd.DataFrame([dict(n=len(frame), **full)]).to_csv(DATA / 'rmse-overall.csv', index=False)
    table.to_csv(DATA / 'rmse-by-metric.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(DATA / 'per-seed-rmse.csv', index=False)
    pd.DataFrame(contrasts).to_csv(DATA / 'rmse-higher-vs-lower-proxy.csv', index=False)
    (DATA / 'rmse-provenance.json').write_text(json.dumps(dict(
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        bin_membership=str(DATA / 'bin-membership.csv'), bin_membership_sha256=hashlib.sha256((DATA / 'bin-membership.csv').read_bytes()).hexdigest(),
        seeds=list(range(5)), bootstrap_replicates=N_BOOT, random_seed_per_descriptor={key:RNG_SEED+j for j,key in enumerate(MAIN_KEYS)},
        estimator='mean over seeds of [sqrt(mean squared error CoDE within bin) - sqrt(mean squared error CoE within bin)]',
        sign='Negative favors CoDE; pK units', confidence_interval='paired-complex percentile 95% within fixed bins; fitted seeds held fixed',
        full_cohort_rmse_matches_canonical_metrics=True), indent=2) + '\n')
    return table


def limits(table, metric='spearman'):
    margin = .035
    low = min(-.04, float(table.ci_lo.min()) - margin)
    high = max(.04, float(table.ci_hi.max()) + margin)
    return np.floor(low / .05) * .05, np.ceil(high / .05) * .05


def contact_panel(ax, table, metric='spearman'):
    block = table[table.descriptor == CONTACT].sort_values('group_index')
    panel(ax, table, CONTACT, limits(block, metric), metric=metric)
    ax.yaxis.set_major_locator(MultipleLocator(.05))
    labels = block.quartile.tolist() if 'quartile' in block and block.quartile.notna().all() else (block.range_label.tolist() if 'range_label' in block and block.range_label.notna().all() else ['Low B', 'Middle B', 'High B'])
    ax.set_xticks(block.group_index.to_numpy(), [f'{label}\nn = {n}' for label, n in zip(labels, block.n)])
    if metric == 'rmse':
        ax.set_title('Contact B · lower ΔRMSE favors CoDE', fontsize=12, pad=20)


def combined_all(decoding_table, ranking):
    fig = plt.figure(figsize=(21, 7.6), facecolor=house.PCSZ_BG)
    grid = fig.add_gridspec(2, 5, left=.061, right=.991, bottom=.115, top=.955, hspace=.70, wspace=.22)
    top = fig.add_subplot(grid[0, :])
    decoding(top, decoding_table, MAIN_KEYS)
    ylim = limits(ranking)
    shared = None
    for j, key in enumerate(MAIN_KEYS):
        ax = fig.add_subplot(grid[1, j], sharey=shared)
        if shared is None:shared = ax
        panel(ax, ranking, key, ylim, show_ylabel=j == 0)
        if j:ax.tick_params(labelleft=False)
    save(fig, 'uncertainty-motivation-combined')


def contact_distribution_overlay(ax, table, frame, label_size, show_models=False):
    """Place binned affinity effects on their numeric-B medians over a full-cohort ECDF."""
    block=table[table.descriptor==CONTACT].sort_values('group_index')
    values=frame[CONTACT].to_numpy(float)
    assert frame.pid.is_unique and np.isfinite(values).all() and len(frame)==int(block.n.sum())
    cuts=block.upper_bound.dropna().to_numpy(float)
    groups=np.searchsorted(cuts,values,side='right')
    for row in block.itertuples():
        sample=values[groups==row.group_index]
        assert len(sample)==row.n
        np.testing.assert_allclose(np.median(sample),row.x_median,rtol=0,atol=1e-10)
    axes_style(ax)
    background=ax.twinx()
    background.set_zorder(1);ax.set_zorder(2);ax.patch.set_visible(False)
    background.patch.set_visible(False)
    x,count=np.unique(values,return_counts=True)
    cumulative=np.cumsum(count)/len(values)
    assert cumulative[-1]==1.
    ecdf_x=np.r_[0.,x,max(40.,x[-1])];ecdf_y=np.r_[0.,cumulative,1.]
    background.step(ecdf_x,ecdf_y,where='post',color='#A7ABB0',lw=1.8,zorder=1)
    background.fill_between(ecdf_x,0,ecdf_y,step='post',color='#B7BBC0',alpha=house.BACKGROUND_ALPHA,zorder=0)
    background.set_ylim(0,1)
    background.set_yticks([.1,.4,.7,1.],['0.1','0.4','0.7','1.0'])
    background.set_ylabel('Cumulative\nprobability',fontsize=label_size,color=house.PCSZ_INK,labelpad=8)
    background.tick_params(axis='y',labelsize=9,direction='out',length=3.5,width=1.1,pad=4,colors=house.PCSZ_AXIS)
    for side in ['top','left','bottom']:background.spines[side].set_visible(False)
    background.spines['right'].set_color(house.PCSZ_AXIS);background.spines['right'].set_linewidth(1.1)
    background.grid(False)
    # All members contribute to the ECDF and to the highest-B performance estimate,
    # including B values above the displayed upper x limit. Do not renormalize the visible window.
    position=block.x_median.to_numpy()
    if show_models:
        for arm,prefix in [('CoE','coe'),('CoDE','code')]:
            mean=block[f'{prefix}_spearman'].to_numpy()
            lo=block[f'{prefix}_ci_lo'].to_numpy();hi=block[f'{prefix}_ci_hi'].to_numpy()
            assert np.isfinite(np.r_[mean,lo,hi]).all()
            color=house.pale(arm)
            ax.vlines(position,lo,hi,color=color,lw=3.68,alpha=house.DATA_ALPHA,zorder=3)
            for xx,lower,upper in zip(position,lo,hi):
                ax.hlines([lower,upper],xx-.35,xx+.35,color=color,lw=2.3,alpha=house.DATA_ALPHA,zorder=3)
            ax.plot(position,mean,'o-',label=arm,color=color,lw=5*house.PCSZ_LW_SCALE,markersize=8.8,
                    markerfacecolor='white',markeredgecolor=color,markeredgewidth=3.68,zorder=4)
        if 'quartile' in block:
            for xx,label in zip(position,block.quartile):
                ax.text(xx,.025,label,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9,color=house.PCSZ_INK)
        legend=ax.legend(loc='upper center',ncol=2,fontsize=9.5,frameon=True,fancybox=False,
                         facecolor='white',edgecolor=house.LEGEND_EDGE,framealpha=house.DATA_ALPHA,
                         handlelength=1.6,handletextpad=.6,columnspacing=1.1,borderpad=.35)
        legend.get_frame().set_linewidth(.7)
    else:
        estimate,lo,hi=[block[key].to_numpy() for key in ['estimate','ci_lo','ci_hi']]
        color=house.pale('CoDE')
        ax.plot(position,estimate,'o-',color=color,lw=5*house.PCSZ_LW_SCALE,markersize=8.8,
                markerfacecolor='white',markeredgecolor=color,markeredgewidth=3.68,zorder=4)
        ax.vlines(position,lo,hi,color=color,lw=3.68,zorder=3)
        for xx,lower,upper,label,value in zip(position,lo,hi,block.range_label,estimate):
            ax.hlines([lower,upper],xx-.5,xx+.5,color=color,lw=2.3,zorder=3)
            # Place high-bin annotations beside the upper caps, inside the axes.
            upper_tail=str(label).startswith('≥')
            low_bin=str(label).startswith('<')
            offset=(0,6) if low_bin else ((-6,-3) if upper_tail else (-8,-3))
            ax.annotate(f'{label}\n{value:+.3f}',(xx,upper),xytext=offset,textcoords='offset points',
                        ha=('center' if low_bin else 'right'),
                        va=('bottom' if low_bin else 'top'),fontsize=12,color=house.PCSZ_INK,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=house.DATA_ALPHA,pad=.6))
    ax.set_xlim(10,40)
    ax.set_xticks([10,20,30,40],['10','20','30','40'],fontsize=label_size)
    ax.set_ylabel('Spearman' if show_models else 'ΔSpearman\n(CoDE − CoE)',fontsize=label_size,labelpad=house.PCSZ_YPAD)
    ax.set_xlabel('');ax.set_title('')
    return background


def combined_contact(decoding_table, ranking, name='uncertainty-combined-contactb-combined', right_ylim=(-.05, .15), metric='spearman', distribution_frame=None, show_models=False, marker_scale=1., cap_scale=1., show_bin_labels=True, legend_scale=1.44, left_panel='decoding'):
    # Reserve additional outer margin for the cumulative-probability axis.
    gap_extra = (11.9 - (1.206 + 9.27)) * .4
    outer_extra = .08
    width = (16.8 if distribution_frame is not None else 16.4) + gap_extra + outer_extra
    height = 4.3
    fig = plt.figure(figsize=(width, height), facecolor=house.PCSZ_BG)
    # Preserve physical panel sizes while accommodating larger tick labels.
    bottom = (.23 * 3.42 + .68) / height
    panel_height = .68 * 3.42 / height
    left = fig.add_axes([(1.206 + outer_extra) / width, bottom, 9.27 / width, panel_height])
    right = fig.add_axes([(11.9 + gap_extra + outer_extra) / width, bottom, 3.132 / width, panel_height])
    if left_panel == 'three-level':
        # Complex contact / residue / atom B factor decoding, token baselines included.
        from . import three_level
        three_level.panel(left, value_fs=12.5, legend_fs=house.PCSZ_KEY_KW['fontsize'] * legend_scale * .8)
    else:
        keys = [key for key in MAIN_KEYS if key != 'occ_site_frac_altloc']
        decoding(left, decoding_table, keys, legend_scale=legend_scale)
    for label in left.texts:label.set_fontsize(13.5 if label.get_gid() == 'ours-value' else 12.5)
    label_size = house.PCSZ_LABEL_FS - 4
    left.set_xticks(left.get_xticks(), [label.get_text() for label in left.get_xticklabels()], fontsize=label_size)
    left.yaxis.label.set_fontsize(label_size)
    left.tick_params(axis='x', pad=left.xaxis.get_major_ticks()[0].get_pad() + 3)
    if distribution_frame is not None:
        assert metric=='spearman'
        contact_distribution_overlay(right,ranking,distribution_frame,label_size,show_models=show_models)
    else:
        contact_panel(right, ranking, metric)
        for line in list(right.lines):
            if len(line.get_xdata()) == 2 and np.allclose(line.get_ydata(), 0):
                line.remove()
            else:
                line.set_linewidth(line.get_linewidth() * 2)
                line.set_markersize(line.get_markersize() * 1.6)
                line.set_markerfacecolor('white')
                line.set_markeredgecolor(line.get_color())
                line.set_markeredgewidth(line.get_markeredgewidth() * 2)
        for interval in right.collections:interval.set_linewidth(np.asarray(interval.get_linewidths()) * 2)
        right.set_title('')
        for text in list(right.texts):
            if 'Higher B' in text.get_text():text.remove()
        block = ranking[ranking.descriptor == CONTACT].sort_values('group_index')
        labels = block.range_label.tolist() if 'range_label' in block and block.range_label.notna().all() else ['Low', 'Middle', 'High']
        right.set_xticks(block.group_index.to_numpy(), labels, fontsize=label_size)
        right.yaxis.label.set_fontsize(label_size)
        right.set_xlim(float(block.group_index.min()) - .35, float(block.group_index.max()) + .12)
    for line in right.lines:
        line.set_markersize(line.get_markersize() * marker_scale)
    for interval in right.collections:
        segments = interval.get_segments()
        for segment in segments:
            if np.isclose(segment[0, 1], segment[1, 1]):
                center = segment[:, 0].mean()
                segment[:, 0] = center + (segment[:, 0] - center) * cap_scale
        interval.set_segments(segments)
    if not show_bin_labels:
        for label in list(right.texts):label.remove()
    right.set_ylim(*right_ylim)
    tick_step=.1 if show_models else .05
    ticks = np.round(np.arange(np.ceil((right_ylim[0] - 1e-9) / tick_step) * tick_step, right_ylim[1] + 1e-9, tick_step), 2)
    tick_labels=[f'{t:.1f}' for t in ticks] if show_models else [f'{t:.2f}'.replace('-', '−') for t in ticks]
    right.set_yticks(ticks,tick_labels)
    for ax in (left, right):
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['bottom'].set_linewidth(1.8)
        ax.tick_params(axis='x', width=1.8)
    right.set_xlabel('Complex contact B factor (Å²)', fontsize=label_size, labelpad=8)
    # Scale titles (including decoding categories) independently of numeric ticks.
    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(label.get_fontsize() * 1.6)
        for spine in ax.spines.values():
            if spine.get_visible():spine.set_linewidth(2.64)
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks():
                for line in (tick.tick1line, tick.tick2line):
                    line.set_markeredgewidth(2.64)
                for label in (tick.label1, tick.label2):
                    label.set_fontsize(label_size * 1.6 - (0 if ax is left and axis is ax.xaxis else 2))
        for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            gridline.set_linewidth(1.2)
            gridline.set_alpha(1.0)
    for bar in left.patches:            # keep our bar's outline at the (rescaled) x-axis weight
        if bar.get_gid() == 'ours-outline':bar.set_linewidth(left.spines['bottom'].get_linewidth())
    for ax in (left, right):ax.yaxis.labelpad += 5
    if left_panel == 'three-level':
        # Ticks match the cumulative-probability axis; one legend centred over both panels.
        left.set_yticks([.1, .4, .7, 1.0]); left.set_ylim(0, 1.05)
        # Grow the canvas upward so the legend is not cropped; panels keep their physical size and position.
        extra = .55; h_old = fig.get_figheight(); fig.set_size_inches(width, h_old + extra)
        for ax in (left, right):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 * h_old / (h_old + extra), pos.width, pos.height * h_old / (h_old + extra)])
        old_legend = left.get_legend(); handles, labels = left.get_legend_handles_labels(); old_legend.remove()
        centre = (left.get_position().x0 + right.get_position().x1) / 2
        top = left.get_position().y1 + .03
        style = dict(ncol=len(labels), frameon=True, fancybox=False, fontsize=old_legend.get_texts()[0].get_fontsize() * 1.3,
                     edgecolor=house.LEGEND_EDGE, facecolor=house.PCSZ_KEY_FACE, framealpha=house.DATA_ALPHA,
                     borderpad=.4, handlelength=1.5, handletextpad=.5, columnspacing=1.2)
        probe = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(centre, top), **style)
        fig.canvas.draw(); span = 1.4 * probe.get_window_extent().width / fig.bbox.width; probe.remove()
        span = min(span, right.get_position().x1 - left.get_position().x0)   # never wider than the two panels
        legend = fig.legend(handles, labels, loc='lower left', mode='expand',      # 1.4x the natural length
                            bbox_to_anchor=(centre - span / 2, top, span, 0), **style)
        legend.get_frame().set_linewidth(.7)
        three_level.halve_outline_in_legend(legend)
    assert np.isclose((right.get_position().x0 - left.get_position().x1) * width, 1.424 * 1.4)
    assert np.isclose(right.get_position().width * width, 3.132)
    if left_panel == 'three-level':
        save_pieces(fig, name, legend, left, [ax for ax in fig.axes if ax is not left])
    save(fig, name, crop_vertical=True, crop_horizontal=True)


def save_pieces(fig, name, legend, left, right_axes, pad=.04):
    """Also export the combined figure as three separately placeable pieces cut from the same canvas
    (identical fonts and sizes): {name}-legend, {name}-representation, {name}-affinity."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    inches = lambda box: Bbox.from_extents(box.x0 / fig.dpi - pad, box.y0 / fig.dpi - pad, box.x1 / fig.dpi + pad, box.y1 / fig.dpi + pad)
    pieces = {'legend': legend.get_window_extent(renderer),
              'representation': left.get_tightbbox(renderer),
              'affinity': Bbox.union([ax.get_tightbbox(renderer) for ax in right_axes])}
    for piece, box in pieces.items():
        for ext in ['png', 'pdf']:
            fig.savefig(MOTIVATION / f'{name}-{piece}.{ext}', facecolor=house.PCSZ_BG, bbox_inches=inches(box), dpi=220)


def contact_comparison(ranking, error):
    # Side by side makes the opposing metric directions explicit.
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 3.8), facecolor=house.PCSZ_BG)
    contact_panel(axes[0], ranking)
    contact_panel(axes[1], error, 'rmse')
    axes[0].set_title('Contact B · higher ΔSpearman favors CoDE', fontsize=12, pad=20)
    fig.subplots_adjust(left=.105, right=.985, bottom=.245, top=.80, wspace=.43)
    save(fig, 'uncertainty-motivation-contact-b-spearman-rmse')


def rmse_all(table):
    fig, axes = plt.subplots(1, 5, figsize=(21, 3.7), sharey=True, facecolor=house.PCSZ_BG)
    for j, (ax, key) in enumerate(zip(axes, MAIN_KEYS)):
        panel(ax, table, key, limits(table, 'rmse'), show_ylabel=j == 0, metric='rmse')
    fig.subplots_adjust(left=.061, right=.991, bottom=.245, top=.82, wspace=.22)
    save(fig, 'uncertainty-motivation-b-affinity-rmse')


def write_note(table):
    overall = pd.read_csv(DATA / 'rmse-overall.csv').iloc[0]
    lines = ['# Combined/contact-B variants and RMSE sensitivity', '',
        '- [Combined A + all five ranking panels](uncertainty-motivation-combined.png) · [PDF](uncertainty-motivation-combined.pdf)',
        '- [Combined A + Contact B only](uncertainty-motivation-combined-contact-b.png) · [PDF](uncertainty-motivation-combined-contact-b.pdf)',
        '- [Contact B: Spearman versus RMSE](uncertainty-motivation-contact-b-spearman-rmse.png) · [PDF](uncertainty-motivation-contact-b-spearman-rmse.pdf)',
        '- [RMSE across all five descriptors](uncertainty-motivation-b-affinity-rmse.png) · [PDF](uncertainty-motivation-b-affinity-rmse.pdf)', '',
        'The Contact-B combined variant contains four decoding descriptors (alternate-location fraction omitted), one-line descriptor labels, a 3.132-inch right panel (1.2 times the compact version; 60% of the original width), single-line Low (244) / Middle (244) / High (245) ticks, no right-panel title or explanatory subtitle, and y limits −0.12 to 0.12 with ticks −0.12, −0.06, 0, 0.06, 0.12. Decoding bar annotations use 10.5 pt (increased from 9.5 pt). The right x-axis limits are −0.35 to 2.12, adding space before the first point. Both x category labels and y-axis titles use 12.1 pt (2 pt smaller than the preceding version); y tick labels are also reduced by 2 pt. The full figure is 15.8 × 3.42 inches, with height reduced to 90% of 3.8 inches. The right-panel curve and confidence-interval line widths, marker edge width is doubled, and markers have white faces with diameter 1.6 times the original (half of the preceding enlarged-marker version), and the special black zero-reference line is removed. Redraw only this file with --contact-only.', '',
        'All figures use the current common palette, omit A/B panel letters, and preserve .54-style decoding bar annotations. The default A/B command remains unchanged; these are explicit requested variants.', '',
        '## RMSE definition', '',
        'Within each descriptor bin, compute RMSE separately for each fitted seed and model, subtract CoE from CoDE, then average the five differences. This is neither RMSE of an ensemble-averaged prediction nor pooled RMSE across seeds. Negative ΔRMSE means CoDE improves absolute prediction accuracy; positive ΔSpearman means CoDE improves ranking. RMSE is in pK units.', '',
        'Use exactly the saved ranking-bin memberships and thresholds. The first four descriptors have 733 valid cases; alternate-location fraction has 732 and groups zero / lower-positive / higher-positive (605/63/64). For each of 8000 bootstrap replicates, resample complexes together for both models and all five seeds, recompute both RMSEs, and average the five differences. The percentile 95% intervals are conditional on the fitted seeds and fixed thresholds. They do not resample seeds or protein families and are not multiplicity-adjusted.', '',
        f'Full CL3 cohort (n = {int(overall.n)}): CoE RMSE {overall.coe_rmse:.4f}; CoDE RMSE {overall.code_rmse:.4f}; ΔRMSE {overall.estimate:+.4f}, 95% CI [{overall.ci_lo:+.4f}, {overall.ci_hi:+.4f}]. These full-cohort seed RMSE values match the canonical affinity evaluation table.', '',
        '| Descriptor | Bin | n | CoE RMSE | CoDE RMSE | ΔRMSE [95% CI] |', '|---|---|---:|---:|---:|---:|']
    for r in table.itertuples():
        lines.append(f'| {r.label} | {r.group} | {r.n} | {r.coe_rmse:.3f} | {r.code_rmse:.3f} | {r.estimate:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] |')
    lines += ['', '## Interpretation', '',
        'Observed Contact-B result: low-B ΔRMSE is +0.0726 [0.0083, 0.1385]; middle and high B are −0.0080 [−0.0613, 0.0461] and −0.0219 [−0.0721, 0.0279]. The higher-versus-lower-B improvement contrast is +0.0945 [0.0135, 0.1761], but neither the middle/high-B subgroup benefit nor the overall RMSE difference excludes zero. All intervals are exploratory and pointwise.', '',
        'Ranking and absolute-error performance are different outcomes. A ranking gain does not imply lower RMSE. The RMSE intervals and higher-versus-lower-proxy contrasts should be inspected before describing a robust absolute-error benefit. A significant difference in gain between strata does not imply that the gain within either stratum is significant. All stratification remains exploratory, and the existing pretraining-recipe and descriptor-definition limitations still apply.', '',
        'Reproduce from the repository root:', '', '```bash',
        'cd figures && python -m _parts.fig_uncertainty.motivation_variants',
        '# Style-only regeneration from saved RMSE bootstrap results:',
        'cd figures && python -m _parts.fig_uncertainty.motivation_variants --plots-only', '```', '',
        'Tables, per-seed values, contrasts and provenance are saved alongside the existing ranking analysis in `../affinity/metric-review/ranking-by-metric/`. No model fitting or feature extraction is repeated.']
    (MOTIVATION / 'combined-and-rmse.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--contact-only', action='store_true', help='Redraw only the edited A + Contact-B combined figure')
    args = parser.parse_args()
    if (MOTIVATION / 'active-analysis.json').exists():
        from .update_expanded import redraw_current_figures
        selected=['uncertainty-combined-contactb-combined'] if args.contact_only else ['uncertainty-combined-contactb-combined','uncertainty-b-affinity-rmse','uncertainty-contact-b-spearman-rmse']
        redraw_current_figures(only=selected)
        return
    DATA.mkdir(parents=True, exist_ok=True)
    if args.contact_only:
        with plt.rc_context({**house.PCSZ_RC, 'savefig.bbox': None}):
            combined_contact(load_decoding(), pd.read_csv(DATA / 'ranking-by-metric.csv'))
        print('Updated only the four-descriptor A + compact Contact-B combined figure.')
        return
    error = pd.read_csv(DATA / 'rmse-by-metric.csv') if args.plots_only else analyse_rmse()
    ranking = pd.read_csv(DATA / 'ranking-by-metric.csv')
    decoded = load_decoding()
    with plt.rc_context({**house.PCSZ_RC, 'savefig.bbox': None}):
        combined_all(decoded, ranking)
        combined_contact(decoded, ranking)
        contact_comparison(ranking, error)
        rmse_all(error)
    write_note(error)
    print('Wrote combined, Contact-B-only combined, Contact-B Spearman/RMSE comparison, and five-descriptor RMSE variants.')


if __name__ == '__main__':
    main()

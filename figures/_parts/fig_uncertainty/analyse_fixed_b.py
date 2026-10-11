#!/usr/bin/env python3
"""Evaluate exploratory absolute Contact-B ranges with saved five-seed predictions."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .update_expanded import SOURCE, DATA, sha
from . import ranking_metrics as ranking
from .motivation_variants import bootstrap_rmse


def bootstrap_models(a,b,truth,rng):
    """Paired complex resampling; average each model's Spearman over fitted seeds."""
    av,bv=[],[]
    for start in range(0,ranking.N_BOOT,ranking.BATCH):
        size=min(ranking.BATCH,ranking.N_BOOT-start)
        idx=rng.integers(len(truth),size=(size,len(truth)))
        target=truth[idx][:,None,:]
        av.append(ranking.corr(a[:,idx].transpose(1,0,2),target).mean(1))
        bv.append(ranking.corr(b[:,idx].transpose(1,0,2),target).mean(1))
    av,bv=np.concatenate(av),np.concatenate(bv)
    assert len(av)==ranking.N_BOOT and np.isfinite(av).all() and np.isfinite(bv).all()
    return av,bv


def analyse(cuts, prefix, source=SOURCE, out=DATA, expected_n=1163, descriptor='b_contact_invd2', values_csv=None, descriptor_label='Contact B'):
    """descriptor/values_csv: bin on another per-complex value (pid-keyed CSV merged onto the source), e.g. the
    5 Å interface B factor; the source file itself is not modified."""
    cuts = np.asarray(cuts, dtype=float)
    assert len(cuts) and np.isfinite(cuts).all() and np.all(np.diff(cuts) > 0)
    assert '/' not in prefix and prefix not in ['', '.', '..']
    labels = ([f'<{cuts[0]:g}'] + [f'{lo:g}–{hi:g}' for lo, hi in zip(cuts[:-1], cuts[1:])]
              + [f'≥{cuts[-1]:g}'])
    frame = pd.read_csv(source)
    assert len(frame) == expected_n and frame.pid.is_unique
    if values_csv is not None:
        extra = pd.read_csv(values_csv)[['pid', descriptor]]
        frame = frame.drop(columns=[descriptor], errors='ignore').merge(extra, on='pid', how='left', validate='one_to_one')
    assert np.isfinite(frame[descriptor]).all()
    values = ['y_true'] + [f'{arm}_seed{s}_prediction' for arm in ['CoE', 'CoDE'] for s in range(5)]
    frame[values] = frame[values].astype(np.float32).astype(float)
    groups = np.searchsorted(cuts, frame[descriptor].to_numpy(), side='right')
    rows, errors, members, seed_rows, boots = [], [], [], [], []
    rng = np.random.default_rng(ranking.RNG_SEED)
    rng_err = np.random.default_rng(ranking.RNG_SEED)
    for g, label in enumerate(labels):
        block = frame[groups == g]
        assert len(block) >= 3, 'Too few complexes for a meaningful Spearman bootstrap'
        truth = block.y_true.to_numpy()
        a, b = [np.stack([block[f'{arm}_seed{s}_prediction'] for s in range(5)]) for arm in ['CoE', 'CoDE']]
        ra, rb = ranking.corr(a, truth[None]), ranking.corr(b, truth[None])
        for seed in range(5):
            np.testing.assert_allclose(ra[seed], spearmanr(a[seed], truth).statistic, atol=1e-12)
            np.testing.assert_allclose(rb[seed], spearmanr(b[seed], truth).statistic, atol=1e-12)
            seed_rows.append(dict(group=label, group_index=g, seed=seed,
                                  coe_spearman=ra[seed], code_spearman=rb[seed], delta=rb[seed]-ra[seed]))
        boot_a, boot_b = bootstrap_models(a, b, truth, rng)
        boot = boot_b - boot_a
        boots.append(boot)
        lo, hi = np.quantile(boot, [.025, .975])
        ea = np.sqrt(((a-truth[None])**2).mean(-1))
        eb = np.sqrt(((b-truth[None])**2).mean(-1))
        err = bootstrap_rmse(a, b, truth, rng_err)
        el, eh = np.quantile(err, [.025, .975])
        meta = dict(descriptor=descriptor, label=descriptor_label, group=label,
                    range_label=label, group_index=g, n=len(block), n_valid=len(frame), n_missing=0,
                    binning='fixed absolute B ranges; open upper tail',
                    lower_bound=float(cuts[g-1]) if g else None,
                    upper_bound=float(cuts[g]) if g < len(cuts) else None,
                    x_min=block[descriptor].min(), x_max=block[descriptor].max(),
                    x_median=block[descriptor].median(),
                    original_n=int(block.cohort.eq('original frozen CL3').sum()),
                    added_n=int(block.cohort.eq('additional audited CL3').sum()))
        rows.append(dict(**meta, coe_spearman=ra.mean(), code_spearman=rb.mean(),
                         estimate=(rb-ra).mean(), ci_lo=lo, ci_hi=hi,
                         coe_ci_lo=np.quantile(boot_a,.025),coe_ci_hi=np.quantile(boot_a,.975),
                         code_ci_lo=np.quantile(boot_b,.025),code_ci_hi=np.quantile(boot_b,.975)))
        errors.append(dict(**meta, coe_rmse=ea.mean(), code_rmse=eb.mean(),
                           estimate=(eb-ea).mean(), ci_lo=el, ci_hi=eh))
        members.extend(dict(pid=pid, group=label, group_index=g, **{descriptor: value})
                       for pid, value in zip(block.pid, block[descriptor]))
        print(f'{label}: n={len(block)}, ΔSpearman={(rb-ra).mean():+.6f} [{lo:+.6f}, {hi:+.6f}]', flush=True)
    table = pd.DataFrame(rows)
    assert table.n.sum() == len(frame) and pd.DataFrame(members).pid.is_unique
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out/f'{prefix}-ranking.csv', index=False)
    pd.DataFrame(errors).to_csv(out/f'{prefix}-rmse.csv', index=False)
    pd.DataFrame(members).to_csv(out/f'{prefix}-bin-membership.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(out/f'{prefix}-per-seed.csv', index=False)
    contrast_lo, contrast_hi = np.quantile(boots[-1]-boots[0], [.025, .975])
    pd.DataFrame([dict(contrast=f'{labels[-1]} minus {labels[0]} ranking improvement',
                       estimate=rows[-1]['estimate']-rows[0]['estimate'],
                       ci_lo=contrast_lo, ci_hi=contrast_hi)]).to_csv(out/f'{prefix}-contrast.csv', index=False)
    provenance = dict(source=str(source), source_sha256=sha(source), descriptor=descriptor,
                      values=str(values_csv) if values_csv else None, values_sha256=sha(Path(values_csv)) if values_csv else None,
                      cuts=cuts.tolist(), unit='Å²',
                      labels=labels, counts=table.n.tolist(),
                      selection='Exploratory user-requested absolute ranges after earlier tercile and absolute-bin analyses; not universal biological disorder cutoffs and no automated cutoff search. The open upper tail pools sparse high-B ranges.',
                      n_boot=ranking.N_BOOT, seed=ranking.RNG_SEED,
                      interval='paired-complex percentile 95%; fixed bins and fitted seeds; pointwise, no multiplicity correction',
                      ranking_precision='restore float32 model outputs before computing ranks',
                      reproduce=f'cd figures && python -m _parts.fig_uncertainty.analyse_fixed_b --cuts {" ".join(f"{x:g}" for x in cuts)} --prefix {prefix} --source {source} --out {out} --expected-n {expected_n}'
                                + (f' --descriptor {descriptor} --values {values_csv} --label "{descriptor_label}"' if values_csv else ''))
    (out/f'{prefix}-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(table[['range_label', 'n', 'coe_spearman', 'code_spearman', 'estimate', 'ci_lo', 'ci_hi']].to_string(index=False), flush=True)

    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuts', nargs='+', type=float, default=[20., 40.])
    parser.add_argument('--prefix', default='fixed-contact-b')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=DATA)
    parser.add_argument('--expected-n', type=int, default=1163)
    parser.add_argument('--descriptor', default='b_contact_invd2')
    parser.add_argument('--values', type=Path, default=None, help='pid-keyed CSV holding --descriptor')
    parser.add_argument('--label', default='Contact B')
    args = parser.parse_args()
    analyse(args.cuts,args.prefix,args.source,args.out,args.expected_n,args.descriptor,args.values,args.label)


if __name__ == '__main__':
    main()

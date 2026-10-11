# Task 1 · binding-affinity regression

Does a frozen density encoder predict binding affinity better than a coordinates-only one?
Every number comes from `probe.py`: mean-pooled features of a frozen encoder, then a small MLP
head trained on the canonical `lp_edrscc_v2` split (`voxbind/splits/`).

Pretrain an encoder and score it on the six test sets, as one YAML config says:

```bash
python task1-affinity_regression/run.py task1-affinity_regression/configs/ab7v2p1_clean.yaml
```

`configs/README.md` describes the config and its stages; the results go to
`results/task1-affinity/<name>/sixset.{csv,json}`. The pieces it runs are below.

```
run.py                        pretraining → features → six test sets, from one config (configs/)
probe.py                      features | probe | finetune (python probe.py --help)
04_probe.sh                   probe one encoder: features once, then a head per target
watch_probe_ckpts.sh          wait for a pretraining checkpoint, then probe it (FULL, CL3, CASF)
eval_cdg_encoder.sh           features plus MSE and MSE+corr heads for one encoder on lp_edrscc_v2
density_match.py              puts the probe density on the encoder's pretraining scale
consolidate_probe_results.py  merges the per-run probe CSVs into one table
configs/                      run.py configs
evals/                        sixset.py (the six test sets), CASF-2016, probe heads, panel5.py
analysis/                     representation similarity and the baseline representation dumps
test/                         external test cohorts: CASP16, BDB2020+, ChEMBL35–PLINDER30, OpenBind (not in git)
log/                          launcher logs (not in git)
```

The affinity-uncertainty evaluation behind fig-uncertainty is in `figures/_parts/fig_uncertainty/`.
Retired one-off analyses (activity cliffs, density retrieval, pocket overlap) are in
`legacy/task1-affinity_regression/`.

Probe an encoder run from `pretrain/exps/` (or any run folder, e.g. `model_zoo/CDG_v2`):

```bash
bash task1-affinity_regression/04_probe.sh --exp <run> \
  --condition atomblob_density_gradmag --split lp_edrscc_v2_cl123 --seeds 5 --gpu 4
```

`--epoch` defaults to `best`, the best-validation checkpoint the trainer records; runs from
before 2026-10 have no record and need `--epoch N`. Features are cached in
`dataset/data/pdbbind/features/` and per-run CSVs written to `dataset/data/pdbbind/results/`.

For a ChannelViT encoder, `--pool groupstats --layer N` replaces the last-layer mean with
[mean, std, per-channel-group std] of the tokens after block N. Train its head on z-scored
features and pK (`--standardize`) with two hidden layers and dropout 0.5:

```bash
bash task1-affinity_regression/04_probe.sh --exp model_zoo/atomblob7_v2p1_cdg_cvit_e99 --epoch 99 \
  --condition atomblob_density_gradmag --split lp_edrscc_v2_cl123 --seeds 10 --pool groupstats --layer 7 \
  -- --standardize --probe_layers 3 --dropout 0.5 --lr 3e-4 --batch_size 256
```

On this encoder it raises CL3 test ρ from 0.588 to 0.631 (10 seeds, MSE loss).

The paper's six test sets are FULL, CL3, CL3-ID60, CL3-ID30, CASF-nontrain and CASF-clean;
their cohort CSVs are in `voxbind/splits/`. The external cohorts beyond them (CASP16, BDB2020+,
ChEMBL35–PLINDER30, OpenBind) are in `test/` (on disk, not in git), each with its own README. External baselines
are in `base/`.

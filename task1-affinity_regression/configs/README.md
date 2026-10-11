# Run configs

One YAML file per experiment drives `run.py`: pretraining, feature extraction and the six test sets.

```bash
python task1-affinity_regression/run.py task1-affinity_regression/configs/ab7v2p1_clean.yaml
python task1-affinity_regression/run.py CONFIG --stages features,evaluate   # skip pretraining
python task1-affinity_regression/run.py CONFIG --dry-run                    # print the commands
```

Each stage is skipped when its output exists: a finished run folder (`checkpoint_final.json`), a
feature cache, `sixset.json`. An unfinished run folder is resumed. `--force` redoes features and
evaluate.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | run folder `pretrain/exps/<name>`; also tags its feature caches and results |
| `pretrain.experiment` | required | preset in `pretrain/configs/experiment/` |
| `pretrain.trainer` | `live` | `legacy` for the presets in `legacy/pretrain/configs/experiment/` |
| `pretrain.gpus` | required | e.g. `4-7` or `0,1` |
| `pretrain.overrides` | `[]` | Hydra overrides, e.g. `[bsz=8, accum_steps=4]` |
| `evaluate.run_dir` | `pretrain/exps/<name>` | the encoder to evaluate, e.g. `model_zoo/CDG_v2` |
| `evaluate.epoch` | `best` | checkpoint; `best` is the trainer's best-validation epoch |
| `evaluate.condition` | required | encoder input, as in `probe.py features --condition` |
| `evaluate.gpu` | first pretraining GPU | the GPU for feature extraction and probe heads |
| `evaluate.seeds` | 5 | probe heads per split, as Table 1 |
| `evaluate.select_seeds` | 5 | heads per candidate layer when a readout lists several |
| `evaluate.num_workers` | 0 | feature-extraction loader workers; with workers the pin-memory thread failed on every run so far, so a failed extraction is retried with 0 |
| `evaluate.features_args` | `[]` | more `probe.py features` arguments |
| `evaluate.readouts` | five last-layer poolings | see below |
| `evaluate.out_dir` | `results/task1-affinity/<name>` | where the results go |

Leave out `pretrain` to evaluate an encoder trained elsewhere. A readout has a `name`, a `pool`, a `layer`
(default 0, the last block) or a list of `layers` (several are compared by CL3 validation ρ and the
best is kept), and a probe `head`: `default` (two layers, dropout 0.1, the Table 1 probe) or `regularized` (z-scored features and pK, three layers, dropout 0.5). Pools: `mean`
(token mean), `max`, `meanstd` (mean and std), `groupmean` (each channel group's mean) and
`groupstats` (mean, std and per-group std). The defaults, as in `ab7v2p1_clean.yaml`, are all five
pools on the last layer with the default head, so only the pooling varies.

The evaluation trains each head on `lp_edrscc_v2` (scored on FULL, CASF-nontrain, CASF-clean) and on
`lp_edrscc_v2_cl123` (CL3, CL3-ID60, CL3-ID30), with `base/evaluate.py`'s test sets and scoring
(`evals/sixset.py`). `<out_dir>/sixset.csv` has one row per readout, test set and seed (the columns
of `base/results/<method>.csv` plus `readout`, `layer`, `val_spearman`); `sixset.json` has mean ±
std, the layer picked and the checkpoint, and `config.yaml` the config the run used.

## The configs here

The clean-corpus campaign: the atomblob7 v2.1 recipe on CoDE_pretrain_v1 and the design-choice ablations
of the CoDE preprint's Table 2.

| Config | Encoder input | Condition |
|---|---|---|
| `ab7v2p1_clean.yaml` | atoms, density, gradmag (CoDE) | `atomblob_density_gradmag` |
| `ab7v2p1_clean_coe.yaml` | atoms (CoE) | `atomblob` |
| `ab7v2p1_clean_coe_drecon.yaml` | atoms; density and gradmag only reconstructed (legacy trainer) | `atomblob` |
| `ab7v2p1_clean_nodens.yaml` | atoms, gradmag | `atomblob` |
| `ab7v2p1_clean_nograd.yaml` | atoms, density | `atomblob_density` |
| `ab7v2p1_clean_occgrad.yaml` | atoms, gradient of the atom occupancies (legacy trainer) | `atomblob` |
| `ab7v2p1_clean_noise.yaml` | atoms, noise drawn from the density histogram, its gradmag; noise at probe time too | `atomblob_density_gradmag` |

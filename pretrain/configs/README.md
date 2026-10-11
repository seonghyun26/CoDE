# Pretrain configs

Hydra configs for density-MAE encoder pretraining (`pretrain/train_density.py`). A run is
composed, not copy-pasted: the base selects one option per axis, and an experiment preset
overrides the axes it cares about and names the run.

```bash
# from the repo root
bash pretrain/scripts/pretrain.sh --experiment cvit_g742_0623 --gpus 0-3

# a one-off variation: override on the command line, don't add a config file
bash pretrain/scripts/pretrain.sh --experiment cvit_g742_0623 --name cvit_interface --gpus 0-3 \
  -- mask=interface mae.mask_ratio=0.85
```

The launcher runs from `pretrain/`, so runs land in `pretrain/exps/<name>/` and logs in
`pretrain/log/<name>.log`. Data paths read `${oc.env:VOXBIND_DATA_ROOT,../dataset/data}`,
relative to that working directory; set the variable or override the `dset` paths after `--` to
point a run elsewhere.

New repeatable recipe → add a file to `experiment/`. Temporary sweep or debug → a command-line
override.

## Defaults

`pretrain.yaml` is the recipe of `260623_ar_cvit_c1_g742` on the clean corpus:

| | |
| --- | --- |
| encoder | `model=v1`: 40M ChannelViT, dim 512 / depth 12 / heads 8, groups `[7,4,2]`, 13 input channels |
| pretext | `mask=uniform` at 0.50, `mae=invfreq` (inverse-frequency channel weighting) |
| corpus | `dset=CoDE_pretrain_v1`, 107,516 positions |
| schedule | 100 epochs, a checkpoint every 5, batch 8 × accum 8 |

The source run used the older 17,430-complex PLINDER corpus, so a bare `pretrain` is that
recipe on clean data, not a replay of the run. The base sets no run name, so a launch without
`--experiment` needs `--name`.

**The final model is the best validation epoch, not the last.** Every epoch writes
`checkpoint.pth.tar` (the latest state, which resume reads); every 5th epoch also writes
`checkpoint_eNNNN.pth.tar`. Whenever validation loss improves the trainer writes
`checkpoint_best.pth.tar` with a `checkpoint_best.json` naming the epoch. At the end it publishes
that file as `checkpoint_final.pth.tar` and as `checkpoint_e<best>.pth.tar` (hard links, no extra
disk). Resume restores the best-so-far record, so a resumed run cannot replace an earlier, better
best. The probe (`task1-affinity_regression/04_probe.sh`, `task1-affinity_regression/probe.py`) defaults to
`--epoch best`, which reads that record; runs from before 2026-10 have none and need `--epoch N`.

Validation loss is reconstruction loss. It can keep falling after the representation stops
improving for affinity, so the periodic checkpoints are there to probe directly when that
matters.

## Axes

**`dset`** — the corpus.

| Option | Corpus |
| --- | --- |
| `CoDE_pretrain_v1` | **Default** (formerly `cdgv2_clean`, still accepted as an alias). `cdgv2` with the proteins of the paper's test sets removed (six affinity sets and the CrossDocked test set): a PDB goes when one of its chains matches a test chain at 100% identity over ≥80% of the shorter sequence. 2,607 PDBs (4.6% of positions) removed. Built by `dataset/plinder/02_make_clean_corpus.py`. |
| `CoDE_pretrain_v1_ID30` | `CoDE_pretrain_v1` without every PDB with a chain similar to a CL3 test chain by ATOM3D's rule (BLASTP identical residues / test-chain length ≥ 30%; `--similarity atom3d`). 94,886 positions (−11.7%); 90,086 in-vocabulary training complexes for the CoDE recipe. |
| `CoDE_pretrain_v1_ID60` | The same at ≥ 60%. 101,628 positions (−5.5%); 96,583 in-vocabulary training complexes. |
| `cdgv2` | PLINDER v2 per-element, 112,733 positions: the CDG_v2 champion's corpus. Not decontaminated. |
| `ab7v2p1` | The atomblob7 v2.1 encoder's corpus through its 96³ box, 106K positions. Built on another server, so not on every machine; its clean view is `02_make_clean_corpus.py --source ab7v2p1`, run there. |

Both cdgv2 options read density from the precomputed 116³ box (`box_path`), which matches
resampling the maps on the fly (r > 0.999), runs about twice as fast, and needs no map cache; the
trainer refuses to start if most sampled complexes come without density.

The six paper test sets are FULL, CL3, CL3-ID60, CL3-ID30, CASF-nontrain and CASF-clean
(`voxbind/splits/`). They nest, so their union is FULL plus CASF-nontrain, 1,357 complexes.

**`model`** — the encoder. Both are ChannelViT-MAE over atomblob7 + density + gradient
magnitude (13 channels, groups `[7,4,2]`).

| Option | Size |
| --- | --- |
| `v1` | 40M: dim 512, depth 12, heads 8. The 0623 recipe. |
| `v2` | 100M: dim 640, depth 18, heads 10. The CDG_v2 champion. |
| `tiny`, `small`, `medium` | 1.2% / 5.2% / 26% of v1's encoder (dim 32/96/256, depth 4/8/10), for the scaling ablation. |

Atomblob7 is the input standard: 7 ligand element channels (C, O, N, S, F, Cl, P) plus 4 pocket
channels (C, N, O, S). The corpus tuples carry an 8th ligand channel for the 0.25% rare tail,
and the loader's `n_lig_ch` default of 7 leaves it out. `patch_embed_mode: channel_group` is
set explicitly in both files, because the code default (`fused`) builds a different model.

**`mae`** — the reconstruction loss: `invfreq` (default; inverse-frequency channel weighting)
or `uniform_weight` (every atom channel equal). Shared loss and training keys are in
`mae/_base.yaml`.

**`mask`** — which blocks are hidden. Each option sets the strategy and its usual ratio;
override the ratio with `mae.mask_ratio=…`.

| Option | Blocks masked | Ratio |
| --- | --- | --- |
| `uniform` | random, all equally likely (default) | 0.50 |
| `atom_biased` | ranked by atom mass plus noise; the CDG_v2 champion | 0.75 |
| `interface` | where ligand and pocket atoms touch, first | 0.75 |
| `interface_atom` | contacts first, then other atom-occupied blocks, empty space last | 0.75 |
| `cluster` | a few large contiguous holes around atom-weighted seeds | 0.50 |
| `ligand` | the ligand's own blocks: predict the ligand from the pocket | 0.25 |
| `per_group` | independent random blocks per channel group | 0.75 |
| `group_complete` | one channel group almost entirely, the others lightly | 0.75 |

`mask/_base.yaml` holds the knobs every option shares: a per-batch random ratio
(`mask_ratio_min`/`max`), an easy-to-hard curriculum (`mask_curriculum*`) and per-group ratios
(`mask_channel_independent`, `mask_group_ratios`).

**`vox`** — voxelisation geometry (grid dim, resolution, radii).

## Presets

| Preset | Recipe |
| --- | --- |
| `cvit_g742_0623` | The default recipe under its own name: v1, uniform 0.50, CoDE_pretrain_v1, 100 epochs. |
| `cdg_v2_champion` | The CDG_v2 affinity champion (`model_zoo/champion`): v2, atom-biased 0.75, cdgv2, 25 epochs, batch 4 × 8. |
| `ab7v2p1_cdg` | The atomblob7 v2.1 encoder (`model_zoo/atomblob7_v2p1_cdg_cvit_e99`): v1 with groups `[7,4,1,1]`, uniform weighting, ab7v2p1. |
| `ab7v2p1_scaling` | The ab7v2p1 recipe for the scaling ablation: choose `model=tiny|small|medium` and `dset.subset_n`; runs on CoDE_pretrain_v1, since ab7v2p1's data is on another server only. |

## Retired options

Options retired in the 2026-10 refactor are in the repo-root `legacy/` folder at the same path:
60 presets, 13 model files (two are the old names of `v1` and `v2`), the `cha`, `champion`,
`invfreq_m050` and `uniform` mae options, and the CrossDocked, `plinder_otf`, v2.7 and v4
corpora. Their presets were pinned so they still resolve exactly as before. The methods they
need are in the frozen trainer, which loads the legacy configs itself: denoise, ELECTRA, ChA-MAE,
MultiMAE, KoLeo, contrastive, data2vec, apo augmentation, modal and ligand dropout, the mFo-DFc
second map, synthetic density, merged/roleblob and two-tower inputs, and the input ablations.

```bash
cd pretrain
torchrun --standalone --nproc_per_node=1 ../legacy/pretrain/train_density.py +experiment=<legacy preset>
```

A preset's data may be in `legacy/dataset/data/` too; move it back to the same path first.
The live trainer refuses a retired option by name instead of ignoring it.

## Layout

| Path | What it is |
| --- | --- |
| `pretrain.yaml` | Base config: one option per axis, run metadata, optim, throughput knobs. |
| `experiment/*.yaml` | Named presets composed on `pretrain.yaml`. |
| `dset/`, `model/`, `mae/`, `mask/`, `vox/` | The axes, one file per option. |
| `archive/` | The monolithic configs the presets were migrated from. Not in git: they exist only on the machine where the migration ran, so the parity tools below work only there. |
| `verify_parity.py`, `migrate_monolith.py`, `code_defaults.py` | Migration tooling. |

`dset/`, `vox/` and `hydra/` also exist in `task2-ligand_generation/configs/`, because each config tree is
self-contained. Keep shared values in step.

## Migration tooling

Run from `pretrain/`.

```bash
python configs/verify_parity.py archive/<monolith> <preset>     # exit 0 = identical trees
python configs/migrate_monolith.py --list                        # plan every monolith
```

`migrate_monolith.py` picks the closest option per axis, writes the residual as overrides, and
deletes its own output unless `verify_parity.py` passes. `code_defaults.py` reads the code's
fallback for each key off the trainer source. Both tools take
`--config-dir ../task2-ligand_generation/configs --base config_train` for the downstream tree.

The generator's `model.fusion` options are documented in `task2-ligand_generation/configs/README.md`.

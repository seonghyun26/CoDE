# Downstream (generator) configs

Hydra configs for the VoxBind generator: `task2-ligand_generation/train_ddp.py` (training) and
`task2-ligand_generation/sample.py` (sampling). Encoder pretraining has its own tree in `pretrain/configs/`.

```bash
# from the repo root: run a named preset
bash task2-ligand_generation/scripts/01_downstream_voxbind.sh --experiment voxbind_fusion_champion_reference --gpus 4-7

# or the flag-built recipe the launcher has always offered
bash task2-ligand_generation/scripts/01_downstream_voxbind.sh --density --encoder vit --encoder-ckpt <ckpt> --gpus 4-7
```

The launcher runs from `task2-ligand_generation/`, so runs land in `task2-ligand_generation/exps/<name>/`.
A preset run is named `<YYMMDD>_<preset>` unless `--name` is given. Relative paths here are relative
to that working directory: data is `../dataset/data`, encoders are `../pretrain/exps/<run>` or
`../model_zoo/<name>`.

Presets name checkpoints relative to `task2-ligand_generation/`, the launchers' working
directory (`../pretrain/exps/...`, `../model_zoo/...`). The generator's training crops
(`xray_crops_aligned_v5`) are not in every checkout; override `dset.crops_dir` and the
checkpoint paths after `--`, the same way as any other key.

## Layout

| Path | What it is |
| --- | --- |
| `config_train.yaml` | Base training config. |
| `config_sample.yaml`, `config_sample_from_file.yaml` | Sampling configs. |
| `experiment/*.yaml` | Named presets composed on `config_train.yaml`. All 5 were migrated from monoliths and verified identical with `verify_parity.py`. |
| `dset/`, `model/`, `vox/`, `wjs/` | The reusable axes. `wjs/` holds the walk-jump sampler settings. |
| `archive/` | The migrated monoliths, out of Hydra's search path. Not in git: they exist only on the machine where the migration ran. |

`dset/crossdocked.yaml`, `dset/crossdocked_xray.yaml`, `vox/` and `hydra/` also exist in
`pretrain/configs/`, because each tree is self-contained. Keep the two copies in step.

Where the archive exists, check a preset against its monolith from `pretrain/`:

```bash
python configs/verify_parity.py --config-dir ../task2-ligand_generation/configs --base config_train \
  archive/config_train_voxbind_fusion_champion_reference voxbind_fusion_champion_reference
```

## Presets

All five are generator trainings on CrossDocked with X-ray density crops, at batch 32 per rank. The
U-REPA and two-tower `adapter` presets, and `voxbind_fusion_champion_oracle` (the reference
preset under the run's earlier name), moved to `legacy/voxbind/configs/experiment/`.

| Preset | What it trains |
| --- | --- |
| `voxbind_fusion_champion_reference` | Frozen 100M champion encoder (`model_zoo/champion_100m_v2_mask075`, epoch 49), `proj_sum` fusion. The encoder also sees the reference ligand's atom channels (`density_encoder_sees_ligand`). Warm-starts from the vanilla σ=0.9 denoiser. |
| `voxbind_frozenenc_channelvit_atomblob7_v2p1` | Frozen atomblob7 v2.1 ChannelViT encoder (groups `[7,4,1,1]`), `proj_sum` fusion, ligand atom channels zeroed in the encoder input. |
| `voxbind_frozenenc_channelvit_atomblob7_v2p1_ligmask` | Same, with density and gradient magnitude also blanked inside the ligand footprint (`density_mask_ligand`), which removes the ligand's own density. |
| `voxbind_frozenenc_channelvit_atomblob7_v2p1_v3` | Same as `_ligmask` with `encoder_replace` fusion. |
| `voxbind_frozenenc_plinder` | Frozen 260616 PLINDER ViT-MAE encoder (fused patch embedding), `proj_sum` fusion. |

## `model.fusion`

How the frozen density encoder reaches the denoiser. The names describe the mechanism; the old
experiment-order names are still accepted and normalised on load, so existing configs and
trained checkpoints keep working.

| Option | Mechanism | Legacy name |
| --- | --- | --- |
| `proj_sum` | Zero-init projection of `[x, density]` added to the ligand+pocket sum. The production recipe: inert at step 0, and the correction can depend on the noisy generation target. | `default` |
| `proj_pocket` | The same projection, fused into the pocket representation alone, so it never sees ligand features. | `protein_first` |
| `token_concat` | Each voxel is concatenated with its own patch's encoder token, i.e. fusion before unpatchify, at full resolution. | `v4` |
| `encoder_replace` | The frozen encoder replaces the pocket encoder entirely and fuses into the ligand features through a normal-init projection. The one mode that is active at step 0; it overfit and was dropped. | `v3` |
| `cross_attn` | The U-Net's attention levels cross-attend to the encoder's tokens. | — |

The authoritative list is `FUSION_MODES` in `voxbind/models/voxbind.py`. An unknown value raises
rather than falling back, so a typo cannot quietly train the wrong experiment.
`task2-ligand_generation/test/test_fusion_modes.py` covers every mode on CPU, including the zero-init inertness
that `proj_sum` depends on.

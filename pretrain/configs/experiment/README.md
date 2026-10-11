# Encoder pretraining presets

Each file is a named Hydra preset composed on `pretrain/configs/pretrain.yaml`. A preset picks
options on the `dset`, `model`, `mae` and `mask` axes and sets a stable `exp_name` and WandB tags.
The table of presets is in `pretrain/configs/README.md`.

Launch a preset from the repo root:

```bash
bash pretrain/scripts/pretrain.sh --experiment cdg_v2_champion --gpus 0-3
```

Override values without creating another file:

```bash
bash pretrain/scripts/pretrain.sh --experiment cdg_v2_champion --name cdg_v2_interface \
  --gpus 0-3 -- mask=interface num_epochs=50 seed=43
```

Queue the same run on the first four free GPUs:

```bash
bash voxbind/scripts/99_chain.sh --count 4 --timeout 24h -- \
  bash pretrain/scripts/pretrain.sh --experiment cdg_v2_champion --name cdg_v2_interface \
    -- mask=interface
```

Create a preset when a configuration is a repeatable recipe; use command-line overrides for
debugging and small sweeps.

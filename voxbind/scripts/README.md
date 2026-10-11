# Shared launcher helpers

What every launcher in the repo shares. The task launchers live with their tasks:
`pretrain/scripts/`, the top level of `task1-affinity_regression/`, `task2-ligand_generation/scripts/`.

- `lib/common.sh` — sourced, not run: `voxbind_repo_root`, `voxbind_exp_dir` (an encoder run
  name → `pretrain/exps/<name>`), `voxbind_resolve_epoch` (`best` → the best-validation epoch),
  GPU-list parsing and logging.
- `lib/gpu.sh` — GPU inspection for `99_chain.sh`.
- `99_chain.sh` — wait until GPUs are idle, reserve them with advisory locks, then run one
  command in the repo root (`--chdir` changes that).

Commands below run from the repo root.

```bash
# queue a pretraining run until four GPUs are free
bash voxbind/scripts/99_chain.sh --count 4 --timeout 24h -- \
  bash pretrain/scripts/pretrain.sh --experiment cdg_v2_champion

# pretrain, then score the best-validation checkpoint on the six affinity test sets
# (leave pretrain.gpus out of the config to use the reserved GPUs)
bash voxbind/scripts/99_chain.sh --count 4 -- \
  python task1-affinity_regression/run.py task1-affinity_regression/configs/ab7v2p1_clean.yaml

# wait for a dataset marker as well
bash voxbind/scripts/99_chain.sh --after-file dataset/data/pretrain/READY --count 4 -- \
  bash pretrain/scripts/pretrain.sh --experiment cdg_v2_champion
```

`99_chain.sh` uses advisory per-GPU locks, several idle samples, and direct argument execution.
It does not use `eval`, kill existing processes, or assume that a process name identifies
ownership. Besides the GPU gate it can wait on `--after-file`, `--after-pid`, or a side-effect-free
`--condition-script`.

Do not add a launcher for a hyperparameter variant. Add or extend a Hydra experiment preset
and run it through the task's launcher.

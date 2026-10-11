# VoxBind — shareable end-to-end pipeline

Self-contained scripts to run the **density-conditioned VoxBind** pipeline on a
fresh machine: environment → data → preprocessing → training → sampling →
evaluation. Written for a coworker who is starting from nothing (no data, no
weights, no envs).

The target model is **VoxBind + electron density**: a walk-jump voxel denoiser
whose pocket branch is conditioned on a *frozen* electron-density encoder
(**CDG_v2**). Evaluation reproduces the paper table — chemical/geometry metrics
+ AutoDock **Vina 1.2.2** docking + **PoseCheck 1.3.1** / **PoseBusters 0.6.5**
(`dock` mode) pose quality.

Each stage is a numbered script and runs independently:

| Step | Script | What it does | Runs on |
|------|--------|--------------|---------|
| 0 | `00_setup_env.sh` | Build the 3 conda envs + clone TargetDiff | CPU |
| 1 | `01_download_data.sh` | Pull the prepared data + weights copy | CPU/net |
| 2 | `02_preprocess_data.sh` | Verify / build `data_{train,test}.pt`, density crops | CPU |
| 3 | `03_train.sh` | Train base denoiser + density-fusion model | **GPU** |
| 4 | `04_sample.sh` | Walk-jump sample ligands (density-conditioned) | **GPU** |
| 5 | `05_evaluate.sh` | Chem + Vina docking + PoseCheck | CPU |

All shared config lives in `_common.sh` (paths, GPUs, env names, the data link).
Every knob is overridable from the environment — no script edits needed.

---

## 0. Prerequisites

- **GPU**: NVIDIA GPU(s), driver ≥ 520 (for CUDA 11.8). Multi-GPU is used
  automatically via `torchrun`; set `GPUS=0,1,2,3` to pick devices.
- **Disk**: ~150–250 GB (CrossDocked + density crops + weights + samples).
- **Conda**: Miniforge / Mamba / Micromamba on `PATH` (native install only).
- **Data link**: where the prepared copy comes from; download links will be announced. Set one
  of these in the environment or in `env.local.sh` at the repository root (git-ignored, read by
  `_common.sh`):
  - `DATA_RCLONE_REMOTE` — an rclone remote you configure, e.g. `myremote:/path/to/VoxBind`. Its
    `model_zoo/` holds the pretrained weights (encoder **and** the base-denoiser warm start); an
    optional `data/` holds the prepared dataset and `targetdiff_test_set/` the full receptors.
  - `DATA_HTTP_URL` — a single `.tar` / `.tar.gz` / `.tar.zst` bundle.

> ℹ️ Without a prepared `data/`, `01` pulls the weights and then points you at the public
> CrossDocked release for the raw data (run `02` to preprocess).

---

## Quickstart

### Option A — Docker (recommended)

The image holds both conda envs, voxbind (training) and voxeval (evaluation). Data/weights stay
outside the image (mounted or pulled at runtime). `script/Dockerfile` (`voxbind:allinone`, with the
older voxdock and moleval envs) remains only as the base of FuncBind's images.

```bash
# build from the repo root
docker build -f env/Dockerfile -t voxbind:env .

# run with GPUs; mount the big dirs so they persist across containers
docker run --rm -it --gpus all --shm-size=16g \
  -v $PWD/dataset/data:/workspace/VoxBind/dataset/data \
  -v $PWD/model_zoo:/workspace/VoxBind/model_zoo \
  -v $PWD/task2-ligand_generation/exps:/workspace/VoxBind/task2-ligand_generation/exps \
  voxbind:env

# inside the container:
export DATA_HTTP_URL=<bundle URL>             # or DATA_RCLONE_REMOTE=<remote>:<path>
bash script/01_download_data.sh
bash script/02_preprocess_data.sh
GPUS=0,1,2,3 bash script/03_train.sh          # MODE=fusion (default)
EXP=voxbind_density_fusion GPUS=0,1,2,3 bash script/04_sample.sh
SAMPLE_DIR=task2-ligand_generation/exps/voxbind_density_fusion/samples/samples_test100 \
  bash script/05_evaluate.sh
```

### Option B — native (bare metal / cluster)

```bash
bash script/00_setup_env.sh                    # builds voxbind (training) and voxeval (evaluation)
export DATA_HTTP_URL=<bundle URL>              # or DATA_RCLONE_REMOTE=<remote>:<path>
bash script/01_download_data.sh
bash script/02_preprocess_data.sh
GPUS=0,1,2,3 bash script/03_train.sh
EXP=voxbind_density_fusion GPUS=0,1,2,3 bash script/04_sample.sh
SAMPLE_DIR=task2-ligand_generation/exps/voxbind_density_fusion/samples/samples_test100 \
  bash script/05_evaluate.sh
```

---

## Step details

### `00_setup_env.sh` — environments
Builds the two conda envs. Idempotent (skips what exists). `FORCE=1 … voxeval` rebuilds one env.

- `voxbind` — training: pretraining, probes, generation (conda-lock + `pip install -e .`)
- `voxeval` — evaluation: Vina **1.2.2**, PoseCheck **1.3.1**, PoseBusters **0.6.5**, QED/SA
  (python 3.10; the pins reproduce the paper's numbers — do not bump). `env/README.md` explains
  the build and its check against the envs it replaced.
- by name only: `voxdock` and `moleval` (the envs voxeval replaced), `voxel-bind`, `targetdiff`

### `01_download_data.sh` — prepared copy
Pulls the weights from the remote (or the bundle) and, if a `data/` folder is there, the
dataset too. Incremental — safe to re-run. Verifies the key artifacts landed. What comes
from where:

```
WEIGHTS (model_zoo/):
  model_zoo/CDG_v2/                       frozen density encoder (+ cfg.yaml)
  model_zoo/voxbind_sig0.9_crossdocked/   base-denoiser warm start (~1.25 GB)
  model_zoo/{C_v2,CD_v2,CG_v2,champion,…} other encoders (optional)

DATASET (data/ on the remote, or fetch public + run 02):
  dataset/data/crossdocked_pocket10/              raw pockets + ligands
  dataset/data/split_by_name.pt                   CrossDocked split
  dataset/data/data_train.pt / data_test.pt       preprocessed tensors (02 builds)
  dataset/data/pretrain/xray_crops_aligned_v5/…   density crops + *_available.npy

EVAL (optional mirror, or from TargetDiff's release):
  <sibling>/targetdiff/data/test_set/             full receptors (docking/pose)
```

Because the base warm start ships in `model_zoo/`, you do **not** need to train
the base denoiser — `03_train.sh` picks it up automatically.

### `02_preprocess_data.sh` — tensors + crops
With the prepared copy this is mostly a **verify** step. If `data_{train,test}.pt`
are missing it rebuilds them from CrossDocked (`preprocess_crossdocked.py`).
`MODE=crops` rebuilds the density corpus from raw 2Fo-Fc maps (heavy; only if
you are *not* using the prepared crops).

### `03_train.sh` — training (GPU)
`MODE` selects the piece:
- `fusion` *(default)* — density-conditioned model: base warm start + **frozen
  CDG_v2** encoder, token fusion. Uses the base warm start and CDG_v2 encoder
  pulled into `model_zoo/` in step 1 — **no base training required**.
- `base` — vanilla VoxBind denoiser from scratch (only if you want to retrain the
  warm-start weights yourself instead of using the shipped one).
- `all` — `base` then `fusion`.
- `encoder` — *(advanced)* re-pretrain the CDG_v2 encoder via MAE; needs the
  PLINDER density corpus. Most users skip this — CDG_v2 ships in step 1.

Encoder geometry is read from `model_zoo/CDG_v2/cfg.yaml` automatically (model_zoo
entries are not interchangeable at fixed dims). Batch is **per-rank**: effective
batch = `BSZ × n_gpus`. Checkpoints land in `task2-ligand_generation/exps/<EXP_NAME>/`.

### `04_sample.sh` — sampling (GPU)
Walk-jump samples ligands over the 100-pocket CrossDocked test split, sharded
across `GPUS`. Auto-detects density conditioning from the checkpoint and feeds
the x-ray crops. Density models sample only map-backed pockets (~79/100 for v5)
— a lower target count is expected. Output: `…/samples/<OUT>/target_XX/`.

### `05_evaluate.sh` — evaluation (CPU)
Runs `task2-ligand_generation/evaluation/eval.py` in voxeval: QED/SA/diversity, Vina docking
against the **whole receptor** (`dataset/data/crossdocked/test_set`), PoseCheck and PoseBusters.
Refuses to run unless voxeval has vina 1.2.2. Writes per-target `eval.json` and a run-level
`eval_results.json` whose `table3` holds the Table 3 numbers.

---

## Environment variables (most useful)

**Global** (`_common.sh`): `REPO_ROOT`, `CODE_ROOT`, `DATA_ROOT`, `GPUS`,
`VOXBIND_ENV` / `VOXEVAL_ENV`, `DATA_RCLONE_REMOTE`, `DATA_HTTP_URL`, `CONDA_LAUNCHER`.

| Script | Vars |
|--------|------|
| `00` | `FORCE` (rebuild an env); positional targets: `voxbind voxeval` (also `voxdock moleval voxel-bind targetdiff`) |
| `01` | `DATA_RCLONE_REMOTE` **or** `DATA_HTTP_URL` |
| `02` | `MODE=verify\|preprocess\|crops`, `FORCE`, `VERSION`, `WORKERS` |
| `03` | `MODE=fusion\|base\|all\|encoder`, `GPUS`, `BSZ`, `NUM_EPOCHS`, `SIGMA`, `EXP_NAME`, `ENCODER`, `WARM_START`, `WANDB` |
| `04` | `EXP` *(required)*, `OUT`, `SAMPLES`, `SPLIT`, `N_POCKETS`, `GPUS` |
| `05` | `SAMPLE_DIR` *(required)*, `METRICS` (quality,vina,posecheck,posebusters), `TARGETS` (p79 \| all \| file), `META`, `EXH`, `CPU`, `WORKERS`, `SKIP_EXISTING` |

---

## Notes & gotchas

- **Vina version is pinned to 1.2.2.** A different build shifts absolute
  affinities; `05` refuses to run otherwise. Rebuild with
  `FORCE=1 bash script/00_setup_env.sh voxeval`.
- **PoseCheck is pinned to 1.3.1.** Upstream silently redefined strain energy;
  1.3.1 is the definition all reported numbers use.
- **PoseBusters is pinned to 0.6.5, and every number is its `dock` config** —
  ligand + protein, no `mol_true`, 22 columns of which 20 are graded (the two
  `*_loaded` columns are informational). `valid` is all-must-pass over whatever
  columns the release returns, so a version that adds or renames a check moves
  every validity number without erroring. `gen` mode is NOT used: its chosen
  binary output is `sucos_within_threshold`, which would enter that same
  all-must-pass and collapse it, so SuCOS is computed separately with gen.yml's
  own function and 0.4 threshold and kept out of `valid`.
- **Full receptors** must be present for meaningful Vina scores and clash counts
  (a 10 Å crop can't clash with atoms it doesn't contain). They come from
  TargetDiff's `test_set` release (folded into the prepared copy).
- **TargetDiff is a sibling** of the VoxBind checkout (`VOXBIND.parent/targetdiff`): its
  test set holds the full receptors (`FULL_RECEPTOR_ROOT`). The Docker image places it at
  `/workspace/targetdiff`.
- **Weights & Biases** logging is off by default (`WANDB=false`). Run
  `wandb login` and pass `WANDB=true` to enable it.
- These scripts wrap the same entrypoints as the internal
  `task2-ligand_generation/scripts/72_` and `73_` launchers (and the retired `70_`, `75_`), but with portable paths
  and no box-specific assumptions.

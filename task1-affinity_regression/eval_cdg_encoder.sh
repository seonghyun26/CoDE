#!/usr/bin/env bash
# Evaluate a CDG (coords+density+gradmag) encoder on binding-affinity regression.
# Extracts frozen mean-pool features on lp_edrscc_v2, then trains the probe head
# with BOTH the plain-MSE recipe and the winning MSE+Pearson-corr aux recipe.
#
#   bash task1-affinity_regression/eval_cdg_encoder.sh <RUN_NAME> <EPOCH|best> <GPU> [AUX_WEIGHT] [CONDITION]
#   e.g. bash task1-affinity_regression/eval_cdg_encoder.sh 260806_cdg_100m_v2_d2vaux05 49 0
#   no-gradmag encoder (12-ch [7,4,1]):  bash task1-affinity_regression/eval_cdg_encoder.sh <RUN> 49 0 5 atomblob_density
#
# Champion reference: mse 0.644 / mse+corr 0.647 (test Spearman).
set -uo pipefail
RUN="${1:?run name (looked up in pretrain/exps/)}"
EP="${2:?epoch}"
GPU="${3:?gpu id for feature extraction}"
AUXW="${4:-5}"
COND="${5:-atomblob_density_gradmag}"      # 13-ch CDG default; atomblob_density = 12-ch (no gradmag)
# Cap CPU: torch defaults to ALL cores → probe head-training + voxelization can hog 15-20
# cores each and drive load >100 (esp. several probes in parallel). Pin to 2 threads + nice.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
# shellcheck source=../voxbind/scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../voxbind/scripts/lib/common.sh"
PY="nice -n 19 $(voxbind_conda_bin voxbind)/python" || exit 1
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1   # task1-affinity_regression/
EXP_DIR="$(voxbind_exp_dir "$RUN")"
EP="$(voxbind_resolve_epoch "$EXP_DIR" "$EP")" || exit 1
TAG="$RUN"
SEEDS="${6:-5}"                            # probe-head seeds (0..SEEDS-1); 5 to match the baselines
COMMON=(--conditions "$COND" --epoch "$EP" --split lp_edrscc_v2 \
        --feature_tag "$TAG" --exp_dir "$EXP_DIR" --allow_stale_features --seeds "$SEEDS")

echo "==== [1/3] extract features: $RUN e$EP on cuda:$GPU ===="
# NB: $PY is intentionally UNQUOTED — it holds a multi-word command ("nice -n 19 <python>")
# that must word-split into argv. Quoting it treats the whole string as one binary path → fails.
CUDA_VISIBLE_DEVICES="$GPU" $PY probe.py features \
  --condition "$COND" --epoch "$EP" --tag "$TAG" \
  --exp_dir "$EXP_DIR" --device cuda:0 --num_workers 0 || { echo "extract FAILED"; exit 1; }

echo "==== [2/3] probe head = MSE ===="
# NB: pin CUDA_VISIBLE_DEVICES here too — without it, --device cuda:0 falls back to physical
# GPU 0, which OOMs the probe when GPU 0 is occupied by another job.
CUDA_VISIBLE_DEVICES="$GPU" $PY probe.py probe "${COMMON[@]}" \
  --probe_loss mse --device cuda:0 --no_wandb --tag "${RUN}_mse"

echo "==== [3/3] probe head = MSE+corr (aux_weight=$AUXW) ===="
CUDA_VISIBLE_DEVICES="$GPU" $PY probe.py probe "${COMMON[@]}" \
  --probe_loss mse+corr --aux_weight "$AUXW" --device cuda:0 --no_wandb --tag "${RUN}_msecorr${AUXW}"

echo "==== DONE: $RUN e$EP ===="

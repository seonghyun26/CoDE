#!/bin/bash
# run_shard_batch.sh SHARD_IDX GPU_ID
# Runs nesso predict in batch mode on a shard directory

SHARD_IDX=$1
GPU_ID=$2

if [ -z "$SHARD_IDX" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: $0 SHARD_IDX GPU_ID"
    exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/../scripts/_conda.sh"
NESSO_BIN=$(envbin nesso)/nesso || exit 1
BASE=$(cd "$(dirname "$0")" && pwd)                       # base/nesso
SHARD_YAML_DIR=$BASE/../data/nesso/shard_${SHARD_IDX}_yamls   # links into base/data/nesso/yamls
OUT_DIR=$BASE/_edrscc/outputs
LOG=$BASE/_edrscc/logs/shard_batch_${SHARD_IDX}.log
CACHE_DIR=$BASE/.cache

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

echo "[shard $SHARD_IDX gpu $GPU_ID] Starting at $(date)" | tee "$LOG"
echo "[shard $SHARD_IDX gpu $GPU_ID] YAMLs: $(ls $SHARD_YAML_DIR/*.yaml 2>/dev/null | wc -l)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU_ID $NESSO_BIN predict "$SHARD_YAML_DIR" \
    --out_dir "$OUT_DIR" \
    --cache "$CACHE_DIR" \
    --devices 1 --accelerator gpu --no_kernels \
    --num_workers 2 \
    --seed 42 \
    2>&1 | tee -a "$LOG"

echo "[shard $SHARD_IDX gpu $GPU_ID] DONE at $(date)" | tee -a "$LOG"

#!/usr/bin/env bash
# timeout-guarded Nesso shard runner — skips large-protein ESM hangs (150s cap)
SHARD=$1; GPU=$2
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/_conda.sh"
NESSO=$(envbin nesso)/nesso || exit 1
BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$BASE/outputs
while IFS= read -r pid; do
  [ -z "$pid" ] && continue
  [ -f "$OUT/predictions/$pid/affinity.json" ] && continue
  [ ! -f "$BASE/yamls/$pid.yaml" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU timeout 300 $NESSO predict "$BASE/yamls/$pid.yaml" --out_dir "$OUT" --cache "$BASE/.cache" >/dev/null 2>&1
done < "$BASE/$SHARD"
echo "[shard $SHARD gpu$GPU] done"

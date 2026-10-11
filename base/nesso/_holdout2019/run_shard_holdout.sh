SHARD_ID=$1; GPU_ID=$2
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/_conda.sh"
NESSO_BIN=$(envbin nesso)/nesso || exit 1
BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
YAMLS=$BASE/yamls; OUT=$BASE/outputs; mkdir -p $OUT
while IFS= read -r pid; do
  [ -f "$OUT/predictions/$pid/affinity.json" ] && continue
  [ ! -f "$YAMLS/$pid.yaml" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU_ID $NESSO_BIN predict "$YAMLS/$pid.yaml" --out_dir "$OUT" --cache "$BASE/.cache" >/dev/null 2>&1
done < $BASE/shard_0$SHARD_ID
echo "[shard $SHARD_ID] done"

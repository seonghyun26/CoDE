#!/usr/bin/env bash
# Nesso-1 (Recursion and Valence Labs, 2026): zero-shot co-folding affinity from sequence and
# SMILES, one deterministic pass (seed 42) over the lp_edrscc_v2 test set and CASF-2016, which
# serves every split inside them. Its training data overlaps our test sets, so the report marks
# it as leaked.
#
#   bash base/scripts/nesso.sh featurize                    input YAMLs -> base/data/nesso/yamls/
#   bash base/scripts/nesso.sh train_eval [--split S ...]   predict, score -> base/results/nesso.csv
#
#   GPUS  GPUs to spread the predictions over, one shard each   default "$GPU"
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy nesso)
GPUS=${GPUS:-$GPU}
go "$BASE/nesso"

case $stage in
  featurize) run "$PY" -u build_inputs.py --split lp_edrscc_v2 --also_casf ;;
  train_eval)
    if training; then
      # One shard of YAML links per GPU, round robin. nesso predict skips complexes that already
      # have a prediction, so a rerun only fills the gaps.
      gpus=($GPUS)
      n=${#gpus[@]}
      for i in "${!gpus[@]}"; do
        run rm -rf "$DATA/nesso/shard_${i}_yamls"
        run mkdir -p "$DATA/nesso/shard_${i}_yamls"
      done
      if [[ -z ${DRY_RUN:-} ]]; then
        k=0
        for y in "$DATA"/nesso/yamls/*.yaml; do
          ln -s "../yamls/${y##*/}" "$DATA/nesso/shard_$((k % n))_yamls/"
          k=$((k + 1))
        done
      fi
      for i in "${!gpus[@]}"; do run bash run_shard_batch.sh "$i" "${gpus[$i]}" & done
      wait
      run "$PY" -u score.py --out_dir _edrscc/outputs
    fi
    evaluate nesso ;;
  *) usage ;;
esac

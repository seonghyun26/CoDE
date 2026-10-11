#!/usr/bin/env bash
# HBGSA (arXiv 2604.23115): hydrogen-bond graph model, our 3.06M-parameter reimplementation of the
# paper, trained from scratch.
#
#   bash base/scripts/hbgsa.sh featurize  [--split S ...]   H-bond cache -> base/data/hbgsa/
#   bash base/scripts/hbgsa.sh train_eval [--split S ...]   SEEDS models per split; lp_edrscc_v2 also
#                                                            CASF-2016 -> base/results/hbgsa.csv
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy hbgsa)
go "$BASE/hbgsa/src"

tag() { local s; s=$(short "$1"); [[ $s == v2 ]] && echo edrscc_v2_3p06m || echo "edrscc_v2_${s}_3p06m"; }

case $stage in
  featurize) for s in "${splits[@]}"; do run "$PY" -u hbonds.py --split_csv "$SPLIT_DIR/$s.csv"; done ;;
  train_eval)
    if training; then
      tags=()
      for s in "${splits[@]}"; do
        run "$PY" -u train.py --split_csv "$SPLIT_DIR/$s.csv" --tag "$(tag "$s")" \
          --seeds "$(comma $SEEDS)" --epochs 150
        tags+=("$(tag "$s")")
      done
      run "$PY" -u ../agg_results.py --tags "${tags[@]}"
      if has_v2; then (go "$BASE/hbgsa"; run "$PY" -u run_casf2016.py --seeds "$(comma $SEEDS)"); fi
    fi
    evaluate hbgsa ;;
  *) usage ;;
esac

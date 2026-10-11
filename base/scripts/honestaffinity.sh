#!/usr/bin/env bash
# HonestAffinity (arXiv 2606.03422): frozen ESM-2 sequence features with a trained head,
# reimplemented from the paper. Runs in the dsmbind env.
#
#   bash base/scripts/honestaffinity.sh featurize  [--split S ...]   records + ESM-2 cache -> base/data/honestaffinity/
#   bash base/scripts/honestaffinity.sh train_eval [--split S ...]   SEEDS heads per split, test predictions in
#                                                                     honestaffinity/preds/<split>/; lp_edrscc_v2
#                                                                     also CASF-2016 -> base/results/honestaffinity.csv
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy dsmbind)
go "$BASE/honestaffinity/src"

case $stage in
  featurize) for s in "${splits[@]}"; do run "$PY" -u data_prep.py --split "$s" --gpu 0; done ;;
  train_eval)
    if training; then
      for s in "${splits[@]}"; do
        run "$PY" -u train.py --split "$s" --seeds $SEEDS --gpu 0 --dump_dir "$BASE/honestaffinity/preds/$s"
      done
      if has_v2; then run "$PY" -u casf_eval.py --seeds $SEEDS --gpu 0; fi
    fi
    evaluate honestaffinity ;;
  *) usage ;;
esac

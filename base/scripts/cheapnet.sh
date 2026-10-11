#!/usr/bin/env bash
# CheapNet (Lim et al., ICLR 2025): hierarchical cross-attention over atom clusters, trained
# from scratch.
#
#   bash base/scripts/cheapnet.sh featurize  [--split S ...]   graphs in base/data/cheapnet/data_<split>/
#   bash base/scripts/cheapnet.sh train_eval [--split S ...]   SEEDS models per split; lp_edrscc_v2 also
#                                                               CASF-2016 -> base/results/cheapnet.csv
#
# lp_edrscc_v2 is featurised first: every other split is a view of its graphs. cheappy wraps the
# env's python with the LD_LIBRARY_PATH CheapNet needs (a direct call fails on a CXXABI mismatch).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
go "$BASE/cheapnet/_edrscc"

case $stage in
  featurize)
    for s in "${splits[@]}"; do
      if [[ $s == lp_edrscc_v2 ]]; then
        run env OMP_NUM_THREADS=1 bash cheappy make_data.py --split "$s" --num_process 12
        run bash cheappy make_casf_data.py
      else
        run bash cheappy make_cl_dataroots.py --splits "$s"
      fi
    done ;;
  train_eval)
    if training; then
      for s in "${splits[@]}"; do
        run bash cheappy train_edrscc.py --data_root "$DATA/cheapnet/data_$s" --tag "$s" --seeds $SEEDS
        if [[ $s == lp_edrscc_v2 ]]; then
          run bash cheappy train_casf_eval.py --seeds $SEEDS
          trained=$LAST_CMD
          for seed in $SEEDS; do
            keep_model cheapnet "$s" "$seed" "$trained" --sub casf "ckpts_casf/cheapnet_casf_seed$seed.pt"
          done
        fi
      done
    fi
    evaluate cheapnet ;;
  *) usage ;;
esac

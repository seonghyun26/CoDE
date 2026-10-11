#!/usr/bin/env bash
# BindNet (Feng et al., ICLR 2024): the released weights are restricted, so the model is
# trained from scratch with the authors' Uni-Core recipe.
#
#   bash base/scripts/bindnet.sh featurize  [--split S ...]   base/data/bindnet/edrscc<_tier>/lba_*.npy,
#                                                              and casf/ with lp_edrscc_v2
#   bash base/scripts/bindnet.sh train_eval [--split S ...]   SEEDS models per split -> test predictions in
#                                                              bindnet/preds/<split>/ -> base/results/bindnet.csv
#
# CASF-2016 is predicted by separate lp_edrscc_v2 models (train_bindnet_casf.sh adds CASF to the
# validation subsets), as the reported CASF numbers were made.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy bindnet)
go "$BASE/bindnet"

case $stage in
  featurize)
    for s in "${splits[@]}"; do
      run "$PY" -u _edrscc/src/make_bindnet_data.py --split "$s" --out_dir "$DATA/bindnet/edrscc$(cltag "$s")"
    done
    if has_v2; then
      run "$PY" -u _edrscc/src/make_casf_data.py
      run ln -sfn ../casf/lba_casf.npy "$DATA/bindnet/edrscc/lba_casf.npy"   # read by train_bindnet_casf.sh
    fi ;;
  train_eval)
    if training; then
      mkdir -p _edrscc/logs
      for s in "${splits[@]}"; do
        t=$(short "$s"); tag=$(cltag "$s")
        for seed in $SEEDS; do
          run_log "_edrscc/logs/train${tag}_seed$seed.log" env GPU="$GPU" SEED="$seed" SPLIT="${tag#_}" \
            LBA_DATA="../data/bindnet/edrscc$tag/lba" bash _edrscc/train_bindnet.sh
          trained=$LAST_CMD
          ckpt=_edrscc/results/bindnet_scratch${tag}_seed$seed/checkpoint_best.pt
          run "$PY" -u _edrscc/src/split_inference.py --split "$s" --seed "$seed" --gpu 0 --checkpoint "$ckpt"
          keep_model bindnet "$s" "$seed" "$trained" "$ckpt" _edrscc/train_bindnet.sh
        done
        run "$PY" -u _edrscc/aggregate_results.py --split "$s" --seeds "$(comma $SEEDS)"
        if [[ $t == v2 ]]; then
          for seed in $SEEDS; do
            run_log "_edrscc/logs/casf_seed$seed.log" env GPU="$GPU" SEED="$seed" SAVE_INT=200 \
              PORT=$((10160 + seed)) bash _edrscc/train_bindnet_casf.sh
            trained=$LAST_CMD
            ckpt=_edrscc/results/bindnet_casf_seed$seed/checkpoint_best.pt
            run env LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((10500 + seed)) \
              "$PY" _edrscc/src/casf_inference.py --gpu 0 --seed "$seed" --checkpoint "$ckpt"
            keep_model bindnet "$s" "$seed" "$trained" --sub casf "$ckpt" _edrscc/train_bindnet_casf.sh
          done
          run "$PY" -u "$BASE/_casf/aggregate_bindnet_casf_5seed.py"
        fi
      done
    fi
    evaluate bindnet ;;
  *) usage ;;
esac

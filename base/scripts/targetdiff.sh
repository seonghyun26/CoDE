#!/usr/bin/env bash
# EGNN and EGNN + TargetDiff (Guan et al., ICLR 2023) with TargetDiff's own property-prediction code: the
# EGNN alone, and the EGNN that also takes the hidden states of TargetDiff's pretrained diffusion model
# (enc_feature_type final_h). Both are trained from scratch; the diffusion model stays frozen.
#
#   bash base/scripts/targetdiff.sh featurize             base/data/targetdiff/pool/: pocket LMDB, split
#                                                          indices, diffusion features (needs a GPU)
#   bash base/scripts/targetdiff.sh train_eval [--split S ...]   SEEDS models per split; lp_edrscc_v2 also
#                                                          CASF-2016 -> base/results/{td_egnn,td_egnn_enc}.csv
#
#   MODELS  which models   default "egnn egnn_enc"
# The diffusion checkpoint is TargetDiff's release. Its Google Drive folder has answered 404 since
# 2026-10, so it comes from the Hugging Face mirror OneScience-Group/TargetDiff, checked against the
# sha256 of a copy we took from the original release on 2026-09-17.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy decompdiff)          # TargetDiff's dependencies (torch 1.13, PyG 2.3), as DecompDiff's
MODELS=${MODELS:-egnn egnn_enc}
T=$BASE/targetdiff
POOL=$DATA/targetdiff/pool
CKPT=pretrained_models/pretrained_diffusion.pt
CKPT_URL=https://huggingface.co/OneScience-Group/TargetDiff/resolve/main/weight/pretrained_diffusion.pt
CKPT_SHA256=8977821339ba31b7103b46835ef4ddbf7f15f3f2efc5cb1f2a405e7905b09db6
go "$T"
slug() { [[ $1 == egnn ]] && echo td_egnn || echo td_egnn_enc; }

case $stage in
  featurize)
    if ! echo "$CKPT_SHA256  $CKPT" | sha256sum -c --quiet - 2>/dev/null; then
      run mkdir -p pretrained_models
      run curl -sSL -o "$CKPT" "$CKPT_URL"
      [[ -n ${DRY_RUN:-} ]] || echo "$CKPT_SHA256  $CKPT" | sha256sum -c -
    fi
    run "$PY" -u _edrscc/build_data.py --splits "$(comma "${splits[@]}")"
    run env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u _edrscc/extract_emb.py ;;
  train_eval)
    if training; then
      for m in $MODELS; do
        for s in "${splits[@]}"; do
          for seed in $SEEDS; do
            dir=$DATA/targetdiff/runs/$s/$m/seed$seed
            run "$PY" _edrscc/make_config.py --model "$m" --split "$s" --seed "$seed" --out "$dir/$m.yml"
            # train_prop.py copies ./models into its log dir, so it runs from base/targetdiff
            run_log "$dir/train.log" env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$T" \
              "$PY" -u scripts/property_prediction/train_prop.py "$dir/$m.yml" --logdir "$dir"
            trained=$LAST_CMD
            [[ -n ${DRY_RUN:-} ]] && rd="<log dir in $dir>" || rd=$(ls -d "$dir"/"$m"_2*/ | tail -1)
            run env CUDA_VISIBLE_DEVICES="$GPU" "$PY" _edrscc/predict.py --run "$rd" \
              --keys "$POOL/split_$s.pt" --out "preds/$s/${m}_seed$seed.csv"
            if [[ $s == lp_edrscc_v2 ]]; then
              run env CUDA_VISIBLE_DEVICES="$GPU" "$PY" _edrscc/predict.py --run "$rd" \
                --keys "$POOL/split_casf2016.pt" --out "preds/casf2016/${m}_seed$seed.csv"
            fi
            # train_prop.py saves only on a validation improvement, so the highest number is the best
            [[ -n ${DRY_RUN:-} ]] && ckpt="<best checkpoint in $rd>" || ckpt=$rd/checkpoints/$(ls "$rd/checkpoints" | sort -n | tail -1)
            keep_model "$(slug "$m")" "$s" "$seed" "$trained" "$ckpt" "$dir/$m.yml"
          done
        done
      done
    fi
    evaluate $(for m in $MODELS; do slug "$m"; done) ;;
  *) usage ;;
esac

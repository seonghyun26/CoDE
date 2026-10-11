#!/usr/bin/env bash
# ProFSA (Gao et al., ICLR 2024): pretrained pocket encoder, frozen, with a regression head (the
# authors' LBA fine-tuning recipe).
#
#   bash base/scripts/profsa.sh featurize  [--split S ...]   LMDBs -> base/data/profsa/edrscc<_tier>/, and
#                                                             casf2016/ with lp_edrscc_v2
#   bash base/scripts/profsa.sh train_eval [--split S ...]   SEEDS heads per split, test predictions in
#                                                             profsa/preds/<split>/; lp_edrscc_v2 also CASF-2016
#                                                             -> base/results/profsa.csv
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy profsa)
P=$BASE/profsa
CKPT=$P/data/log/train/profsa/profsa_release/checkpoints/last.ckpt
go "$P"

# Seed 0 has no suffix, later seeds _seed<N>: lba30_cl123_seed2, lp_edrscc_v2_cl123_seed2.log
suffix() { [[ $1 -gt 0 ]] && echo "_seed$1" || echo ""; }
# The best checkpoint of a run: the one that is not last.ckpt (save_top_k 1), as _casf/infer_casf.py.
best_ckpt() { ls "$1"/checkpoints/*.ckpt | grep -v '/last\.ckpt$' | sort | tail -1; }

case $stage in
  featurize)
    for s in "${splits[@]}"; do
      run "$PY" -u _edrscc/src/make_lmdb.py --split_csv "$SPLIT_DIR/$s.csv" --out_dir "$DATA/profsa/edrscc$(cltag "$s")"
    done
    if has_v2; then run "$PY" -u _casf/make_casf_lmdb.py; fi ;;
  train_eval)
    if training; then
      mkdir -p _edrscc/logs
      for s in "${splits[@]}"; do
        data=$DATA/profsa/edrscc$(cltag "$s")
        for seed in $SEEDS; do
          # PROFSA_DUMP_DIR makes lba.py write the test predictions, as preds/<split>/ProFSA_v2_seed<N>.csv
          run_log "_edrscc/logs/$s$(suffix "$seed").log" \
            env PROFSA_DUMP_DIR="$P/preds/$s" PROFSA_DUMP_SEED="$seed" \
            "$PY" train.py experiment=lba30 logging=csv '~callbacks.rich' \
            dataset.dataset_cfg.train.data_dir="$data" model.cfg.data_dir="$data" \
            model.cfg.pretrained_weights="$CKPT" model.cfg.dropout=0.5 \
            optim.lr=0.0002 scheduler.num_warmup_steps=200 \
            trainer.devices=1 seed="$seed" run_test=true +trainer.enable_progress_bar=false \
            hydra.run.dir="$P/_edrscc/runs/lba30$(cltag "$s")$(suffix "$seed")"
          trained=$LAST_CMD
          rd=$P/_edrscc/runs/lba30$(cltag "$s")$(suffix "$seed")
          [[ -n ${DRY_RUN:-} ]] && ckpt="<best checkpoint in $rd>" || ckpt=$(best_ckpt "$rd")
          keep_model profsa "$s" "$seed" "$trained" "$ckpt" "$rd/.hydra"
        done
        run "$PY" -u _edrscc/src/aggregate_results.py --split "$s" --seeds "$(comma $SEEDS)"
        if [[ $s == lp_edrscc_v2 ]]; then
          mkdir -p _casf
          for seed in $SEEDS; do
            run "$PY" _casf/infer_casf.py --run_dir "$P/_edrscc/runs/lba30$(suffix "$seed")" --ckpt best \
              --out "_casf/profsa_casf5_seed$seed.csv" --casf_data_dir "$DATA/profsa/casf2016"
          done
        fi
      done
    fi
    evaluate profsa ;;
  *) usage ;;
esac

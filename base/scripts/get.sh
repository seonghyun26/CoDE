#!/usr/bin/env bash
# GET (Kong et al., ICML 2024) and the two EGNN baselines shipped in its repo, EGNN and
# EGNN + TargetDiff features, each trained from scratch.
#
#   bash base/scripts/get.sh featurize  [--split S ...]   base/data/get/edrscc<_tier>/*.pkl, and casf2016/
#   bash base/scripts/get.sh train_eval [--split S ...]   SEEDS models per split; lp_edrscc_v2 also CASF-2016
#                                                          -> base/results/{get,egnn,egnn_td}.csv
#
#   METHODS  which models to run   default "GET EGNN EGNN_TD"
# lp_edrscc_v2 is featurised first: every other split is a view of its pickles.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy get)
METHODS=${METHODS:-GET EGNN EGNN_TD}
G=$BASE/get
go "$G"
export PATH=$(dirname "$PY"):$PATH        # scripts/train/train.sh calls python from the env

suffix() { [[ $1 -gt 0 ]] && echo "_seed$1" || echo ""; }
lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }
# Lowest-validation-loss checkpoint named in a training log.
best_ckpt() {
  grep -oE "Validation: [0-9.]+, save path: [^ ]+\.ckpt" "$1" \
    | sed -E 's/Validation: ([0-9.]+), save path: (.*)/\1 \2/' | sort -n | head -1 | awk '{print $2}'
}

case $stage in
  featurize)
    for s in "${splits[@]}"; do
      if [[ $s == lp_edrscc_v2 ]]; then
        run "$PY" -u _edrscc/src/make_data.py --split_csv "$SPLIT_DIR/$s.csv" --out_dir "$DATA/get/edrscc"
        run "$PY" -u _casf_get/make_casf_data.py
      else
        run "$PY" -u _edrscc/make_cl_dataroots.py --splits "$s"
      fi
    done ;;
  train_eval)
    if training; then
      mkdir -p _edrscc/logs
      for m in $METHODS; do
        for s in "${splits[@]}"; do
          t=$(short "$s")
          base_cfg=_edrscc/$(lower "$m")_$t.json
          if [[ ! -f $base_cfg ]]; then   # a new split: the v2 config on its own data and save_dir
            run "$PY" -c "import json; d = json.load(open('_edrscc/$(lower "$m")_v2.json'))
for k in ('train_set', 'valid_set'): d[k] = d[k].replace('/edrscc/', '/edrscc$(cltag "$s")/')
d['save_dir'] = d['save_dir'].replace('_v2', '_$t'); json.dump(d, open('$base_cfg', 'w'), indent=2)"
          fi
          for seed in $SEEDS; do
            cfg=_edrscc/$(lower "$m")_$t$(suffix "$seed").json
            if [[ ! -f $cfg ]]; then      # later seeds: the seed-0 config with its own seed and save_dir
              run "$PY" -c "import json; d = json.load(open('$base_cfg')); d['seed'] = $seed
d['save_dir'] = d['save_dir'].rstrip('/') + '_seed$seed'; json.dump(d, open('$cfg', 'w'), indent=2)"
            fi
            log=_edrscc/logs/train_${m}_${t}_seed$seed.log
            run_log "$log" env GPU="$GPU" PORT=$((9940 + seed)) bash scripts/train/train.sh "$cfg"
            trained=$LAST_CMD
            [[ -n ${DRY_RUN:-} ]] && ckpt="<best checkpoint in $log>" || ckpt=$(best_ckpt "$log")
            run "$PY" inference.py --test_set "$DATA/get/edrscc$(cltag "$s")/test.pkl" --task PDBBind \
              --ckpt "$ckpt" --gpu 0 --save_path "_edrscc/preds_${m}_${t}_seed$seed.jsonl"
            keep_model "$(lower "$m")" "$s" "$seed" "$trained" "$ckpt" "$cfg"
          done
          run "$PY" -u _edrscc/aggregate_results.py --method "$m" --split "$t" --seeds $SEEDS
          if [[ $t == v2 ]]; then
            mkdir -p _casf_get/preds
            for seed in $SEEDS; do
              # the lowest-validation-loss checkpoint, the one the test-set numbers use (the run behind
              # the CASF predictions before 10-09 took the first file find listed)
              topk=$(find "_edrscc/models/${m}_v2$(suffix "$seed")" -name topk_map.txt 2>/dev/null | head -1)
              [[ -n $topk ]] || { echo "missing checkpoints for $m seed $seed" >&2; continue; }
              ckpt=$(sort -n "$topk" | head -1 | awk '{print $2}')
              run "$PY" _casf_get/run_casf_inference.py --ckpt "$G/${ckpt#./}" --gpu 0 \
                --test_pkl "$DATA/get/casf2016/test.pkl" --out "_casf_get/preds/preds_${m}_casf5_seed$seed.jsonl"
            done
          fi
        done
      done
    fi
    evaluate $(lower "$METHODS") ;;
  *) usage ;;
esac

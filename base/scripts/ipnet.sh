#!/usr/bin/env bash
# IPNet, the affinity network inside IPDiff (Huang et al., ICLR 2024). Two rows in Table 1: the
# released network frozen with an MLP head ("frozen"), and the same network trained from scratch
# ("scratch"). Runs in the get env.
#
#   bash base/scripts/ipnet.sh featurize                    structure tensors and frozen IPNet features
#                                                           -> base/data/ipnet/
#   bash base/scripts/ipnet.sh train_eval [--split S ...]   heads / networks per split, test predictions in
#                                                           ipdiff/preds/<variant>/<split>/; lp_edrscc_v2 also
#                                                           CASF-2016 -> base/results/ipnet_{frozen,scratch}.csv
#
#   VARIANTS  which rows to run   default "frozen scratch"
# Training always runs seeds 0-4 (SEEDS in run_frozen.py and run_retrain.py); SEEDS is not used.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy get)
VARIANTS=${VARIANTS:-frozen scratch}
go "$BASE/ipdiff"
export PYTHONPATH=.

case $stage in
  featurize)
    run "$PY" -u _edrscc/precompute_mp.py --pids_file "$DATA/ipnet/all_pids.txt" --out "$DATA/ipnet/tensors_all.pt"
    run "$PY" -u _edrscc/feats_from_tensors.py ;;
  train_eval)
    if training; then
      only=$(comma "${splits[@]}"); has_v2 && only+=",casf"
      for v in $VARIANTS; do
        case $v in
          frozen) run "$PY" -u _edrscc/run_frozen.py --only "$only" ;;
          scratch) run "$PY" -u _edrscc/run_retrain.py --only "$only" ;;
          *) echo "unknown variant $v" >&2; exit 2 ;;
        esac
      done
    fi
    methods=(); for v in $VARIANTS; do methods+=("ipnet_$v"); done
    evaluate "${methods[@]}" ;;
  *) usage ;;
esac

#!/usr/bin/env bash
# DeepDTA (Öztürk et al., Bioinformatics 2018) and MolTrans (Huang et al., Bioinformatics 2021)
# through DeepPurpose: sequence and SMILES only, no structure.
#
#   bash base/scripts/dta.sh featurize  [--split S ...]   sequence/SMILES tables in base/data/dta/
#   bash base/scripts/dta.sh train_eval [--split S ...]   SEEDS models per split (lp_edrscc_v2 also scores
#                                                          CASF-2016) -> base/results/{deepdta,moltrans}.csv
#
#   MODELS  which models to run   default "deepdta moltrans"
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy dta)
MODELS=${MODELS:-deepdta moltrans}
go "$BASE/dta"

case $stage in
  featurize)
    run "$PY" -u make_data.py                       # lp_edrscc_v2 (as v2_*), CASF-2016, 2019 holdout
    others=(); for s in "${splits[@]}"; do [[ $s == lp_edrscc_v2 ]] || others+=("$s"); done
    [[ ${#others[@]} -eq 0 ]] || run "$PY" -u build_dta_splits.py --splits "${others[@]}" ;;
  train_eval)
    if training; then
      for m in $MODELS; do
        for s in "${splits[@]}"; do
          name=$s; [[ $s == lp_edrscc_v2 ]] && name=v2     # base/data/dta names the v2 split "v2"
          for seed in $SEEDS; do run "$PY" -u run_deeppurpose.py --model "$m" --seed "$seed" --split "$name"; done
        done
      done
      run "$PY" -u aggregate_dta.py
    fi
    evaluate $MODELS ;;
  *) usage ;;
esac

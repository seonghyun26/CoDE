#!/usr/bin/env bash
# AEV-PLIG (Valsson et al., Commun. Chem. 2025): GATv2 over radial-AEV ligand graphs,
# trained from scratch.
#
#   bash base/scripts/aevplig.sh featurize  [--split S ...]   AEV graphs of the split's complexes
#   bash base/scripts/aevplig.sh train_eval [--split S ...]   SEEDS models per split; lp_edrscc_v2 also
#                                                              CASF-2016 -> base/results/aevplig.csv
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy aevplig)
GRAPHS=$DATA/aevplig/graphs/aevplig_edrscc_graphs.pickle    # one pickle for every split
go "$BASE/aevplig/src"

case $stage in
  featurize)
    for s in "${splits[@]}"; do
      reuse=(); [[ -e $GRAPHS ]] && reuse=(--reuse "$GRAPHS")   # add only the complexes not built yet
      run "$PY" -u build_graphs.py --split "$s" "${reuse[@]}" --out "$GRAPHS"
    done ;;
  train_eval)
    if training; then
      for s in "${splits[@]}"; do
        # lp_edrscc_v2 keeps its original "edrscc" tag; the copy is the name the report reads
        tag=$s; [[ $s == lp_edrscc_v2 ]] && tag=edrscc
        run "$PY" -u train_edrscc.py --split "$s" --tag "$tag" --seeds $SEEDS --epochs 200
        # The checkpoint name has no split in it, so copy each split's models before the next split
        trained=$LAST_CMD
        for seed in $SEEDS; do
          keep_model aevplig "$s" "$seed" "$trained" --copy "../output/trained_models/aevplig_edrscc_seed$seed.model"
        done
        if [[ $s == lp_edrscc_v2 ]]; then
          run cp ../results/aevplig_edrscc.json ../results/aevplig_lp_edrscc_v2.json
          run "$PY" -u casf_eval.py --seeds $SEEDS --dump_preds
          trained=$LAST_CMD
          for seed in $SEEDS; do
            keep_model aevplig "$s" "$seed" "$trained" --sub casf --copy "../output/casf_ckpts/aevplig_casf_seed$seed.model"
          done
        fi
      done
    fi
    evaluate aevplig ;;
  *) usage ;;
esac

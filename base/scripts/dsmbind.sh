#!/usr/bin/env bash
# DSMBind (Jin et al., bioRxiv 2023): unsupervised binding-energy model, used zero-shot.
# One deterministic pass over the lp_edrscc_v2 test set and CASF-2016 serves every split inside
# them; the energy's sign is arbitrary, so |r| and |rho| are reported and RMSE is not defined.
#
#   bash base/scripts/dsmbind.sh featurize                  nothing to do: it reads the structures
#   bash base/scripts/dsmbind.sh train_eval [--split S ...]  energies -> base/results/dsmbind.csv
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy dsmbind)
go "$BASE/dsmbind"

case $stage in
  featurize) echo "DSMBind is zero-shot and reads the structures directly; nothing to featurise." >&2 ;;
  train_eval)
    if training; then
      run "$PY" -u _edrscc/src/run_eval.py --gpu 0
      run "$PY" -u _edrscc/src/run_eval_casf.py --gpu 0
      run "$PY" -u _edrscc/src/build_zeroshot_json.py    # per-split result JSONs
    fi
    evaluate dsmbind ;;
  *) usage ;;
esac

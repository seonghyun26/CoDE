#!/usr/bin/env bash
# GeoSSL-DDM (Liu et al., arXiv 2206.13602): SchNet pretrained on small molecules by denoising
# distance matching, frozen, with a 2-layer MLP probe. Runs in the aevplig env.
#
#   bash base/scripts/geossl.sh featurize  [--split S ...]   frozen SchNet features (+ CASF-2016)
#                                                             -> base/data/geossl/schnet_feats.pt
#   bash base/scripts/geossl.sh train_eval [--split S ...]   the probe on each split; lp_edrscc_v2 also
#                                                             CASF-2016 -> base/results/geossl.csv
#
# The probe always trains seeds 0-4 (src/probe.py); SEEDS is not used.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PY=$(envpy aevplig)
go "$BASE/geossl"

probe() { run "$PY" -u src/probe.py --device cuda:0 "$@"; }

case $stage in
  featurize) run "$PY" -u src/extract_features.py --splits "${splits[@]}" casf2016_eval casf2016_clean --device cuda:0 ;;
  train_eval)
    if training; then
      for s in "${splits[@]}"; do
        probe --split "$s" --tag "$(short "$s")"
        if [[ $s == lp_edrscc_v2 ]]; then
          probe --split "$s" --test_from casf2016_eval --tag casf_core
          probe --split "$s" --test_from casf2016_eval --nontrain_only --tag casf_nontrain
          probe --split "$s" --test_from casf2016_clean --tag casf_clean92
        fi
      done
    fi
    evaluate geossl ;;
  *) usage ;;
esac

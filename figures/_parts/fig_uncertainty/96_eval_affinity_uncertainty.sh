#!/usr/bin/env bash
# 96_eval_affinity_uncertainty.sh — per-complex affinity error beside per-complex crystallography.
#
#   bash figures/_parts/fig_uncertainty/96_eval_affinity_uncertainty.sh
#   conda run -n voxbind python figures/draw.py fig-uncertainty-affinity-tercile
#
# The regression-task half of the fig-uncertainty question. 95_eval_pocket_uncertainty.sh asks
# whether the GENERATION gain tracks how well ordered the pocket is and answers on 79 pockets;
# this asks it of the binding-affinity probe, which predicts one pK per complex and therefore
# gives an error -- and an error DIFFERENCE between two encoders -- per complex.
#
#   C       coords-only encoder             (the regression twin of the CoE arm)
#   C+D+G   + X-ray density and gradient    (the twin of CoDE)
#
# Both are probed in ONE invocation so `--intersect` puts them on the identical complexes with
# the identical probe head and train/val split; three seeds are dumped and averaged, because a
# single head's per-complex prediction carries seed noise the size of the effect.
#
# NOT THE PUBLISHED NUMBERS, and it says so in the log. The coords-only feature cache on this
# box covers 5,316 complexes, so the probe runs on 779 of the 1,320 test complexes rather than
# all of them. The comparison is still strictly PAIRED -- both arms on the same 779 -- which is
# all the per-complex difference needs; the published C / C+D+G rows are in
# results/task1-affinity/*/metrics.json and come from a box with the full cache.
#
# Inputs it will build if missing: dataset/data/pdbbind/bfactors.json (extract_bfactors.py) and
# ligand_rscc.csv (01d_pdbbind_ligand_rscc.py, one wwPDB validation report per complex).
# CPU only; ~10 min cold, ~30 s once the caches exist.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"
# shellcheck source=../../../voxbind/scripts/lib/common.sh
source "$ROOT/voxbind/scripts/lib/common.sh"
PY="${PY:-$(voxbind_conda_bin voxbind)/python}"
SPLIT="${SPLIT:-lp_edrscc_v2}"
SEEDS="${SEEDS:-3}"
PDBBIND=dataset/data/pdbbind
SCATTER="${SCATTER:-$PDBBIND/results/scatter}"

[ -f "$PDBBIND/bfactors.json" ] || { echo "  building bfactors.json"; "$PY" dataset/pdbbind/extract_bfactors.py; }
[ -f "$PDBBIND/ligand_rscc.csv" ] || {
    echo "  fetching ligand RSCC from the wwPDB validation reports"
    "$PY" -c "import csv,sys;[print(r['pid'].lower()) for r in csv.DictReader(open('voxbind/splits/$SPLIT.csv'))]" > /tmp/_unc_pids.txt
    "$PY" dataset/pdbbind/01d_pdbbind_ligand_rscc.py --workers 16 --pids /tmp/_unc_pids.txt \
        --out dataset/data/pdbbind/ligand_rscc.csv
}

# One dump per seed: the probe writes the scatter CSV for --scatter_seed only, and all three
# are wanted so the per-complex prediction can be a seed mean.
for s in $(seq 0 $((SEEDS - 1))); do
    if ls "$SCATTER"/scatter_*"$SPLIT"*seed"$s".csv >/dev/null 2>&1; then continue; fi
    echo "  probe seed $s"
    "$PY" task1-affinity_regression/probe.py probe \
        --conditions atomblob atomblob_density_gradmag \
        --epoch 99 --split "$SPLIT" --seeds "$SEEDS" \
        --device cpu --no_wandb --no_write_csv --allow_stale_features \
        --scatter_seed "$s" --scatter_dir "$ROOT/$SCATTER" | tail -4
done

"$PY" figures/_parts/fig_uncertainty/eval_affinity_uncertainty.py --split "$SPLIT" --scatter-dir "$SCATTER" "$@"

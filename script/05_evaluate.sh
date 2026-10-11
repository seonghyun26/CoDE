#!/usr/bin/env bash
# 05_evaluate.sh — evaluate a sampling run: molecular quality, Vina docking, PoseCheck, PoseBusters.
#
#   SAMPLE_DIR=task2-ligand_generation/exps/<run>/samples/<sampling> bash script/05_evaluate.sh
#
# Runs task2-ligand_generation/evaluation/eval.py in the voxeval env (AutoDock Vina 1.2.2, PoseCheck,
# PoseBusters, QED/SA in one interpreter). The defaults are the paper's protocol: the whole
# receptor (dataset/data/crossdocked/test_set), exhaustiveness 32 and the 79 pockets with density.
# Writes target_*/eval.json and <SAMPLE_DIR>/eval_results.json, whose "table3" holds the Table 3
# numbers. A baseline in TargetDiff's meta format is staged first with META="a.pt b.pt".
#
# Overrides: SAMPLE_DIR, METRICS (quality,vina,posecheck,posebusters), TARGETS (p79 | all | file),
# META, EXH, CPU, WORKERS, SKIP_EXISTING=1.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

: "${SAMPLE_DIR:?set SAMPLE_DIR (a sampling run dir holding target_*/)}"
case "$SAMPLE_DIR" in /*) ;; *) SAMPLE_DIR="$REPO_ROOT/$SAMPLE_DIR" ;; esac
METRICS="${METRICS:-quality,vina,posecheck,posebusters}"
TARGETS="${TARGETS:-p79}"
EXH="${EXH:-32}"                   # Vina exhaustiveness
CPU="${CPU:-4}"                    # Vina threads per ligand
WORKERS="${WORKERS:-8}"            # ligands docked in parallel
SKIP_EXISTING="${SKIP_EXISTING:-0}"
META="${META:-}"

[ -d "$REPO_ROOT/dataset/data/crossdocked/test_set" ] \
    || die "whole receptors missing: put TargetDiff's test_set at dataset/data/crossdocked/test_set"
require_env "$VOXEVAL_ENV"
vina_ver="$(conda_run "$VOXEVAL_ENV" python -c 'import importlib.metadata as m; print(m.version("vina"))' 2>/dev/null || true)"
[ "$vina_ver" = "1.2.2" ] || die "$VOXEVAL_ENV has vina='${vina_ver:-none}', need 1.2.2 (rebuild: FORCE=1 bash script/00_setup_env.sh voxeval)"

ARGS=( "$SAMPLE_DIR" --metrics "$METRICS" --targets "$TARGETS" --exhaustiveness "$EXH" --cpu "$CPU" --workers "$WORKERS" )
[ "$SKIP_EXISTING" = "1" ] && ARGS+=( --skip-existing )
[ -n "$META" ] && ARGS+=( --meta $META )
mkdir -p "$SAMPLE_DIR"
LOG="$SAMPLE_DIR/evaluate.log"
banner "evaluating $SAMPLE_DIR (metrics=$METRICS targets=$TARGETS exh=$EXH)"
( cd "$REPO_ROOT/task2-ligand_generation" && conda_run "$VOXEVAL_ENV" python -m evaluation.eval "${ARGS[@]}" ) \
    >> "$LOG" 2>&1 || die "evaluation failed (see $LOG)"
log "done -> $SAMPLE_DIR/eval_results.json"

#!/usr/bin/env bash
# 73_evaluate_samples.sh — evaluate a sampling run (quality, Vina, PoseCheck, PoseBusters).
#
#   SAMPLE_DIR=exps/<run>/samples/<sampling> bash task2-ligand_generation/scripts/73_evaluate_samples.sh
#
# Same as script/05_evaluate.sh, which it calls; a relative SAMPLE_DIR is read from
# task2-ligand_generation/. Knobs: SAMPLE_DIR, METRICS, TARGETS, META, EXH, CPU, WORKERS, SKIP_EXISTING.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SAMPLE_DIR:?set SAMPLE_DIR (a sampling run dir holding target_*/)}"
case "$SAMPLE_DIR" in /*) ;; *) SAMPLE_DIR="$HERE/$SAMPLE_DIR" ;; esac
export SAMPLE_DIR
exec bash "$HERE/../script/05_evaluate.sh"

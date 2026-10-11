#!/usr/bin/env bash
# draw_all.sh — redraw every figure in the tree, PNG + PDF, from the saved results.
#
#   bash figures/draw_all.sh             # every figure that is in the tree
#   bash figures/draw_all.sh --with-3d   # also the sheets laid out from saved 3D captures
#   JOBS=4 PYTHON=/path/to/voxbind/python bash figures/draw_all.sh
#
# What it does, in order:
#   1. the official palette sheet (fig-palette-methods);
#   2. every other figure, one process per batch from `draw.py --jobs` -- an evaluation's
#      figures together, each standalone analysis (uncertainty, palette proposals,
#      overviews) on its own -- JOBS at a time;
#      With --with-3d the batches include fig-qual's comparison sheets, laid out again
#      from the saved captures;
#   3. the clash box plot and the strain legend again in their second layout (both kept);
#   4. the manuscript downloads, if figures/manuscript/iclr2027-figures.json exists.
#
# Every draw is --existing-only (a figure removed from the tree stays removed),
# --figures-only (saved tables, notes and provenance are not rewritten) and
# --keep-going (one failure does not stop the rest; the summary lists them).
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
py=${PYTHON:-python}
jobs=${JOBS:-3}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
with3d=()
for arg in "$@"; do
    case "$arg" in
        --with-3d) with3d=(--with-3d) ;;
        *) echo "unknown option: $arg (only --with-3d)" >&2; exit 2 ;;
    esac
done

exec 9>"${TMPDIR:-/tmp}/voxbind-draw-all.lock"
flock -n 9 || { echo "draw_all.sh is already running" >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
logs=$(mktemp -d "${TMPDIR:-/tmp}/voxbind-draw-all.XXXXXX")
printf 'logs: %s\n' "$logs"
common=(--existing-only --keep-going --figures-only)
draw() { "$py" "$here/draw.py" "$@"; }

failed=0
run() {                       # run <log name> <draw.py args...>
    local name=$1; shift
    if draw "$@" "${common[@]}" >"$logs/$name.log" 2>&1; then
        printf 'done:   %s\n' "$name"
    else
        printf 'FAILED: %s\n' "$name" >&2
        tail -n 15 "$logs/$name.log" >&2
        return 1
    fi
}

# 1. the palette sheet first
run palette-methods fig-palette-methods || failed=1

# 2. everything else, JOBS batches at a time
mapfile -t batches < <(draw --jobs "${with3d[@]}")
pids=()
for batch in "${batches[@]}"; do
    read -r -a ids <<<"$batch"
    ids=("${ids[@]/fig-palette-methods}")          # already drawn in step 1
    read -r -a ids <<<"${ids[*]}"
    (( ${#ids[@]} )) || continue
    name=${ids[0]}; (( ${#ids[@]} > 1 )) && name="${ids[0]%-*}+$(( ${#ids[@]} - 1 ))"
    run "$name" "${ids[@]}" &
    pids+=("$!")
    if (( ${#pids[@]} >= jobs )); then
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        pids=()
    fi
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done

# 3. the second layout of the clash box plot and the strain legend
mapfile -t layout_ids < <(draw --ids posecheck | grep -E 'clash-box|legend')
if (( ${#layout_ids[@]} )); then
    run posecheck-second-layout "${layout_ids[@]}" \
        --clash-layout horizontal --legend-layout vertical || failed=1
fi

# 4. the manuscript downloads
if [[ -f "$here/manuscript/iclr2027-figures.json" ]]; then
    draw --sync-downloads >"$logs/downloads.log" 2>&1 && echo "done:   downloads" || failed=1
fi

printf 'finished%s. logs: %s\n' "$([[ $failed == 0 ]] || echo ' WITH FAILURES')" "$logs"
exit "$failed"

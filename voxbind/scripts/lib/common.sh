#!/usr/bin/env bash
# Shared shell helpers for VoxBind launchers. Source this file; do not execute it.

# The repository root (the folder holding voxbind/, pretrain/ and the task folders).
voxbind_repo_root() {
    if [[ -n "${VOXBIND_REPO:-}" ]]; then
        printf '%s\n' "$VOXBIND_REPO"
    else
        cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
    fi
}

# Prefix of conda environment NAME, asked of conda itself (CONDA_EXE, else conda on PATH).
voxbind_conda_prefix() {
    local name="${1:?voxbind_conda_prefix: env name required}" exe d
    # conda's executable: CONDA_EXE, conda on PATH, else a standard install in the home directory.
    exe="${CONDA_EXE:-$(command -v conda 2>/dev/null)}"
    if [[ -z "$exe" ]]; then
        for d in miniforge3 mambaforge miniconda3 anaconda3; do
            [[ -x "$HOME/$d/bin/conda" ]] && { exe="$HOME/$d/bin/conda"; break; }
        done
    fi
    "${exe:-conda}" env list 2>/dev/null | awk -v n="$name" '$1 == n {print $NF; exit}'
}

# bin/ of conda environment NAME; dies when conda does not know it.
voxbind_conda_bin() {
    local name="${1:?voxbind_conda_bin: env name required}" prefix
    prefix="$(voxbind_conda_prefix "$name")"
    [[ -n "$prefix" ]] || voxbind_die "conda env '$name' not found; create it or set the script's python variable"
    printf '%s/bin\n' "$prefix"
}

# Absolute run dir for an encoder run NAME: pretrain/exps/NAME. An existing path is returned
# as is, so a run outside pretrain/exps (e.g. under model_zoo/) can be passed directly.
voxbind_exp_dir() {
    local name="${1:?voxbind_exp_dir: run name required}"
    if [[ -d "$name" ]]; then
        (cd "$name" && pwd)
    else
        printf '%s/pretrain/exps/%s\n' "$(voxbind_repo_root)" "$name"
    fi
}

# Echo the epoch to probe: an integer passes through; `best` reads the trainer's record
# (checkpoint_final.json once a run has finished, checkpoint_best.json while it runs).
voxbind_resolve_epoch() {
    local dir="${1:?voxbind_resolve_epoch: exp dir required}" epoch="${2:?epoch required}" rec
    if [[ "$epoch" != best ]]; then
        printf '%s\n' "$epoch"
        return
    fi
    for rec in "$dir/checkpoint_final.json" "$dir/checkpoint_best.json"; do
        if [[ -f "$rec" ]]; then
            sed -n 's/^ *"epoch": *\([0-9][0-9]*\).*/\1/p' "$rec" | head -1
            return
        fi
    done
    voxbind_die "--epoch best: $dir has no checkpoint_best.json (runs before 2026-10 have none); pass --epoch N"
}

voxbind_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

voxbind_log() {
    printf '[%s] %s\n' "$(voxbind_timestamp)" "$*"
}

voxbind_die() {
    printf '%s: %s\n' "${VOXBIND_CALLER:-voxbind}" "$*" >&2
    exit 1
}

voxbind_expand_gpus() {
    local spec="${1//[[:space:]]/}" item lo hi gpu
    local -a items=() result=()
    local -A seen=()

    [[ -n "$spec" ]] || return 1
    IFS=',' read -r -a items <<< "$spec"
    for item in "${items[@]}"; do
        if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo="${BASH_REMATCH[1]}"
            hi="${BASH_REMATCH[2]}"
            (( lo <= hi )) || return 1
            for ((gpu=lo; gpu<=hi; gpu++)); do
                [[ -z "${seen[$gpu]:-}" ]] || continue
                result+=("$gpu")
                seen[$gpu]=1
            done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            [[ -z "${seen[$item]:-}" ]] || continue
            result+=("$item")
            seen[$item]=1
        else
            return 1
        fi
    done

    local IFS=,
    printf '%s\n' "${result[*]}"
}

voxbind_gpu_count() {
    local expanded
    expanded="$(voxbind_expand_gpus "$1")" || return 1
    awk -F, '{print NF}' <<< "$expanded"
}

voxbind_parse_duration() {
    local value="$1" amount unit multiplier
    [[ "$value" =~ ^([0-9]+)([smhd]?)$ ]] || return 1
    amount="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
    case "$unit" in
        ""|s) multiplier=1;;
        m) multiplier=60;;
        h) multiplier=3600;;
        d) multiplier=86400;;
        *) return 1;;
    esac
    printf '%s\n' "$((amount * multiplier))"
}

voxbind_print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

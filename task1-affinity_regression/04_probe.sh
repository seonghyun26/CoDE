#!/bin/bash
# 04_probe.sh — frozen-encoder probing launcher (task1-affinity_regression/probe.py).
# Extracts the encoder features ONCE for a pretrained encoder, then runs the frozen
# MLP probe for ONE OR MORE downstream tasks (--tasks is a comma list). Features are
# target-agnostic, so every selected task reuses the single extraction. Companion to
# pretrain/scripts/pretrain.sh (same arg-parse / --dry-run / derived-name conventions).
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash task1-affinity_regression/04_probe.sh --exp EXP --condition COND --tasks T1,T2,... [options]
#
# Required:
#   --exp NAME        encoder run name (looked up in pretrain/exps/) or a run dir, e.g. under
#                     model_zoo/; its checkpoint_e<epoch>.pth.tar is probed.
#   --condition COND  encoder input mode — MUST match how EXP was trained:
#                     atomblob | atomblob_ligvdw | atomblob_density | atomblob_weighted |
#                     atomblob_merged_density | atomblob_merged_density_gradmag |
#                     atomblob_density_gradmag | density_gradmag
#
# Tasks (--tasks, comma-separated; aliases & groups expand, order preserved, de-duped):
#   affinity(=pK)  hbonds  bfactor  bfactor_rel(=relb)
#   partial_charges(=pc)   electron_affinity(=ea)   adaptability   rmsf
#   pose_stability(=pose)  interaction_energy(=ie)
#   groups:  bfactors   = bfactor,bfactor_rel
#            misato_qm  = partial_charges,electron_affinity
#            misato_md  = adaptability,rmsf,pose_stability,interaction_energy
#            misato     = misato_qm + misato_md
#            structural = hbonds,bfactor_rel
#            all        = every target
#   default: affinity
#
# Options:
#   --split S        lp_edrscc (default; canonical Kd/Ki-only v2 3850/817/1320) | lp_edrscc_v2_cl123 | lp
#   --epoch N|best   encoder checkpoint epoch; best = the best-validation checkpoint (default best)
#   --atom_source S  auto | default | ligvdw              (default auto)
#   --seeds N        probe seeds (mean±std over N)         (default 3)
#   --gpu N          single GPU id for features + probe    (default 0)
#   --tag TAG        run tag for feature .pt + CSV names   (default: sanitized --exp)
#   --pool P         feature readout: mean | groupstats   (default mean; groupstats needs a
#                    ChannelViT encoder — pair with --layer and, for the probe, -- --standardize)
#   --layer N        read tokens after transformer block N (default 0 = last block)
#   --num_workers N  feature-extraction dataloader workers (default 8)
#   --max_complexes N  cap feature extraction for a smoke test (default 0 = all)
#   --refresh        re-extract features even if cached
#   --wait_gpu       deprecated compatibility gate; prefer voxbind/scripts/99_chain.sh
#   --dry-run        print the resolved commands and exit (no work)
#   -- ARG ...       everything after a bare "--" is passed verbatim to BOTH probe calls
#                    (e.g. -- --head softmax --hidden 256)
#   -h | --help
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   # affinity + structural probes of a Channel-ViT encoder on the canonical split
#   bash task1-affinity_regression/04_probe.sh --exp 260620_plinder_otf_channelvit_vit_mae_40m_pretrain \
#        --condition atomblob_density_gradmag --atom_source ligvdw \
#        --tasks affinity,hbonds,bfactor_rel
#   # group-statistics readout at block 7 with the regularised head (MSE)
#   bash task1-affinity_regression/04_probe.sh --exp model_zoo/atomblob7_v2p1_cdg_cvit_e99 --epoch 99 \
#        --condition atomblob_density_gradmag --split lp_edrscc_v2_cl123 --seeds 10 \
#        --pool groupstats --layer 7 -- --standardize --probe_layers 3 --dropout 0.5 --lr 3e-4 --batch_size 256
set -u

# Portable roots (match pretrain/scripts/pretrain.sh): derive repo from this script, allow env overrides.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../voxbind/scripts/lib/common.sh
source "$SCRIPT_DIR/../voxbind/scripts/lib/common.sh"
VOXBIND_CALLER="04_probe.sh"
ROOT="$(voxbind_repo_root)"
TASK="$ROOT/task1-affinity_regression"
PY="${VOXBIND_PY:-$(voxbind_conda_bin voxbind)}" || exit 1
DATA="$ROOT/dataset/data"
FEAT="$DATA/pdbbind/features"
LOG="$TASK/log"
PROBE="$TASK/probe.py"
ts(){ voxbind_timestamp; }
die(){ voxbind_die "$*"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
EXP=""; COND=""; TASKS_RAW="affinity"; SPLIT="lp_edrscc"; EPOCH=best
VV="v5"    # the voxel set's tag in probe.py cache and result names (VOXELS); not an option
INHERITED_GPUS="${VOXBIND_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
ATOM="auto"; SEEDS=3; GPU="${INHERITED_GPUS%%,*}"; TAG=""; NUM_WORKERS=8; MAXC=0
REFRESH=0; WAIT_GPU=0; DRYRUN=0; POOL="mean"; LAYER=0
PASSTHRU=()

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)            EXP="$2"; shift 2;;
        --condition)      COND="$2"; shift 2;;
        --tasks)          TASKS_RAW="$2"; shift 2;;
        --split)          SPLIT="$2"; shift 2;;
        --epoch)          EPOCH="$2"; shift 2;;
        --atom_source)    ATOM="$2"; shift 2;;
        --atom_dir)       ATOM_DIR="$2"; shift 2;;
        --dens_dir)       DENS_DIR="$2"; shift 2;;
        --seeds)          SEEDS="$2"; shift 2;;
        --gpu)            GPU="$2"; shift 2;;
        --tag)            TAG="$2"; shift 2;;
        --num_workers)    NUM_WORKERS="$2"; shift 2;;
        --pool)           POOL="$2"; shift 2;;
        --layer)          LAYER="$2"; shift 2;;
        --max_complexes)  MAXC="$2"; shift 2;;
        --refresh)        REFRESH=1; shift;;
        --wait_gpu)       WAIT_GPU=1; shift;;
        --dry-run)        DRYRUN=1; shift;;
        --)               shift; PASSTHRU=("$@"); break;;
        -h|--help)        awk 'NR==1{next} /^#/{print;next} {exit}' "$0"; exit 0;;
        *)                die "unknown arg: $1  (see --help)";;
    esac
done

[[ -n "$EXP"  ]] || die "--exp is required (encoder run name or run dir)"
[[ -n "$COND" ]] || die "--condition is required (must match how EXP was trained)"
[[ "$POOL" == mean || "$POOL" == groupstats ]] || die "--pool must be mean or groupstats"
[[ "$LAYER" =~ ^[0-9]+$ ]] || die "--layer must be a block number (0 = last)"

# ── tasks: resolve aliases/groups → ordered, de-duped raw --target list ─────────
expand_task(){
    case "$1" in
        affinity|pk|pK)            echo "pK";;
        hbonds|hbond)              echo "hbonds";;
        bfactor)                   echo "bfactor";;
        bfactor_rel|relb)          echo "bfactor_rel";;
        partial_charges|pc)        echo "partial_charges";;
        electron_affinity|ea)      echo "electron_affinity";;
        hardness)                  echo "hardness";;
        adaptability)              echo "adaptability";;
        rmsf)                      echo "rmsf";;
        pose_stability|pose)       echo "pose_stability";;
        interaction_energy|ie)     echo "interaction_energy";;
        bfactors)                  echo "bfactor bfactor_rel";;
        misato_qm|qm)              echo "partial_charges electron_affinity hardness";;
        misato_md|md)              echo "adaptability rmsf pose_stability interaction_energy";;
        misato)                    echo "partial_charges electron_affinity adaptability rmsf pose_stability interaction_energy";;
        structural|struct)         echo "hbonds bfactor_rel";;
        all)                       echo "pK hbonds bfactor bfactor_rel partial_charges electron_affinity adaptability rmsf pose_stability interaction_energy";;
        *) die "unknown task '$1' (see --help for the task list)";;
    esac
}
TARGETS=()
IFS=',' read -ra _toks <<< "$TASKS_RAW"
for t in "${_toks[@]}"; do
    [[ -z "$t" ]] && continue
    expanded="$(expand_task "$t")" || exit 1   # unknown task → die() printed in subshell; propagate
    for r in $expanded; do
        dup=0; for e in "${TARGETS[@]:-}"; do [[ "$e" == "$r" ]] && dup=1; done
        [[ "$dup" -eq 0 ]] && TARGETS+=("$r")
    done
done
[[ ${#TARGETS[@]} -gt 0 ]] || die "no tasks resolved from --tasks '$TASKS_RAW'"

# ── Resolve encoder dir + checkpoint, derive run tag + feature file ─────────────
EXP_DIR="$(voxbind_exp_dir "$EXP")"
EXP_BASE="$(basename "$EXP_DIR")"
EXP_PATH="$EXP_DIR"
EPOCH="$(voxbind_resolve_epoch "$EXP_PATH" "$EPOCH")" || exit 1
CKPT="$EXP_PATH/checkpoint_e$(printf '%04d' "$EPOCH").pth.tar"
# Mid-run the best epoch may exist only as checkpoint_best.pth.tar.
if [[ ! -f "$CKPT" && "$(voxbind_resolve_epoch "$EXP_PATH" best 2>/dev/null)" == "$EPOCH" ]]; then
    CKPT="$EXP_PATH/checkpoint_best.pth.tar"
fi

# run tag: sanitized exp basename unless overridden — keeps feature .pt + CSVs per-encoder
[[ -z "$TAG" ]] && TAG="$(echo "$EXP_BASE" | tr -c 'A-Za-z0-9_' '_' | sed 's/_\+/_/g; s/^_//; s/_$//')"
# non-default readout → its own cache (probe.py readout_suffix): _groupstats-L7, _mean-L7, …
READOUT=""; READOUT_ARGS=()
if [[ "$POOL" != mean || "$LAYER" != 0 ]]; then
    READOUT="_${POOL}"; [[ "$LAYER" != 0 ]] && READOUT+="-L${LAYER}"
    READOUT_ARGS=( --pool "$POOL" --layer "$LAYER" )
fi
FEATFILE="$FEAT/${COND}_e${EPOCH}_${VV}_${TAG}${READOUT}.pt"

# ── Report ──────────────────────────────────────────────────────────────────
echo "[$(ts)] probe  exp=$EXP_BASE  condition=$COND  split=$SPLIT  epoch=$EPOCH"
echo "          tasks: ${TARGETS[*]}"
echo "          gpu=$GPU  seeds=$SEEDS  atom_source=$ATOM  tag=$TAG  readout=$POOL@${LAYER/#0/last}"
echo "          features: $FEATFILE"

# ── Assemble feature-extraction command ─────────────────────────────────────
FEAT_CMD=( "$PY/python" "$PROBE" features
           --condition "$COND" --epoch "$EPOCH"
           --atom_source "$ATOM" --exp_dir "$EXP_DIR" --tag "$TAG"
           --num_workers "$NUM_WORKERS" )
[[ "$MAXC" -gt 0 ]] && FEAT_CMD+=( --max_complexes "$MAXC" )
[[ ${#READOUT_ARGS[@]} -gt 0 ]] && FEAT_CMD+=( "${READOUT_ARGS[@]}" )
[[ -n "${ATOM_DIR:-}" ]] && FEAT_CMD+=( --atom_dir "$ATOM_DIR" )
[[ -n "${DENS_DIR:-}" ]] && FEAT_CMD+=( --dens_dir "$DENS_DIR" )

# per-task probe command (built in the loop; $T filled per target)
build_probe_cmd(){
    local T="$1"
    PROBE_CMD=( "$PY/python" "$PROBE" probe
                --conditions "$COND" --epoch "$EPOCH"
                --seeds "$SEEDS" --target "$T" --split "$SPLIT"
                --feature_tag "$TAG" --tag "$TAG" --exp_dir "$EXP_DIR"
                --allow_stale_features )
    [[ ${#READOUT_ARGS[@]} -gt 0 ]] && PROBE_CMD+=( "${READOUT_ARGS[@]}" )
    [[ ${#PASSTHRU[@]} -gt 0 ]] && PROBE_CMD+=( "${PASSTHRU[@]}" )
}

# ── Dry-run: print resolved commands and exit ─────────────────────────────────
if [[ "$DRYRUN" -eq 1 ]]; then
    echo
    echo "# (1) extract features once  [skipped at runtime if $FEATFILE exists and no --refresh]"
    echo "CUDA_VISIBLE_DEVICES=$GPU \\"; printf '  %q' "${FEAT_CMD[@]}"; echo
    for T in "${TARGETS[@]}"; do
        build_probe_cmd "$T"
        echo
        echo "# (2) probe  target=$T"
        echo "CUDA_VISIBLE_DEVICES=$GPU \\"; printf '  %q' "${PROBE_CMD[@]}"; echo
    done
    exit 0
fi

# ── Pre-flight: checkpoint must exist ─────────────────────────────────────────
[[ -f "$CKPT" ]] || die "no checkpoint at $CKPT (check --exp / --epoch)"
mkdir -p "$LOG"
cd "$TASK" || die "cd $TASK failed"
RUNLOG="$LOG/probe_${TAG}_$(date +%y%m%d_%H%M%S).log"
echo "          log: $RUNLOG"

# ── Optional GPU idle gate (queue behind a running job) ───────────────────────
if [[ "$WAIT_GPU" -eq 1 ]]; then
    echo "[deprecated] --wait_gpu is retained for compatibility; use voxbind/scripts/99_chain.sh" >&2
    echo "[$(ts)] waiting for GPU $GPU idle (<2GB, 3×60s)..." | tee -a "$RUNLOG"
    free=0
    while [[ "$free" -lt 3 ]]; do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
        if [[ -n "$m" && "$m" -lt 2000 ]]; then free=$((free+1)); else free=0; fi
        sleep 60
    done
fi

# ── (1) extract features once (skip if cached unless --refresh) ───────────────
if [[ -f "$FEATFILE" && "$REFRESH" -eq 0 ]]; then
    echo "[$(ts)] features cached → skip extract ($FEATFILE)" | tee -a "$RUNLOG"
else
    echo "[$(ts)] extract features → $FEATFILE" | tee -a "$RUNLOG"
    CUDA_VISIBLE_DEVICES="$GPU" "${FEAT_CMD[@]}" >> "$RUNLOG" 2>&1 \
        || die "feature extraction failed (see $RUNLOG)"
    [[ -f "$FEATFILE" ]] || die "features not written to $FEATFILE (see $RUNLOG)"
fi

# ── (2) probe each task (sequential; tiny MLP, reuses the one extraction) ──────
fail=0
for T in "${TARGETS[@]}"; do
    build_probe_cmd "$T"
    echo "[$(ts)] probe target=$T (seeds=$SEEDS, split=$SPLIT)" | tee -a "$RUNLOG"
    if CUDA_VISIBLE_DEVICES="$GPU" "${PROBE_CMD[@]}" >> "$RUNLOG" 2>&1; then
        echo "[$(ts)]   target=$T DONE" | tee -a "$RUNLOG"
    else
        echo "[$(ts)]   target=$T FAILED — inspect $RUNLOG" | tee -a "$RUNLOG"; fail=1
    fi
done

echo "[$(ts)] probe run complete  exp=$EXP_BASE  tasks=${TARGETS[*]}  (fail=$fail)" | tee -a "$RUNLOG"
echo "          CSVs under: $DATA/pdbbind/  (probe_results_e${EPOCH}_${VV}_${TAG}*.csv)"
exit "$fail"

# Shared setup for base/scripts/<baseline>.sh. Sourced, never run on its own.
#
# Every baseline script has two stages, run on one or more frozen splits (voxbind/splits/<S>.csv):
#   bash base/scripts/<baseline>.sh featurize  [--split S ...]   our complexes -> base/data/<baseline>/
#   bash base/scripts/<baseline>.sh train_eval [--split S ...]   train, predict the split's test set (and
#                                                                CASF-2016 for lp_edrscc_v2), score the
#                                                                predictions into base/results/<method>.csv
# Environment:
#   SEEDS      seeds to train                 default "0 1 2 3 4", as in Table 1
#   SPLITS     splits when no --split given   default "lp_edrscc_v2 lp_edrscc_v2_cl123"
#   GPU        the one GPU the baseline uses  default 4
#   EVAL_ONLY  set to 1 to skip training and only rescore the predictions on disk
#   DRY_RUN    set to 1 to print the commands without running them
#   CONDA_ENVS folder holding the conda envs  default: ask conda for each env by name
# Every trained model is kept as base/models/<method>/<split>/seed<N>/ (keep_model, base/model_store.py).

SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE=$REPO/base
DATA=$BASE/data                          # featurised inputs, base/data/<baseline>/
SPLIT_DIR=$REPO/voxbind/splits
# Conda envs by name (envbin, envpy).
source "$(dirname "${BASH_SOURCE[0]}")/_conda.sh"
EVAL_PY=${EVAL_PY:-$(envpy voxbind)}
SEEDS=${SEEDS:-0 1 2 3 4}
SPLITS=${SPLITS:-lp_edrscc_v2 lp_edrscc_v2_cl123}
GPU=${GPU:-4}
export CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=${WANDB_MODE:-offline}

# Print a command, then run it unless DRY_RUN is set. LAST_CMD keeps it for keep_model.
LAST_CMD=
run() { LAST_CMD="$*"; echo "+ $*" >&2; [[ -n "${DRY_RUN:-}" ]] || "$@"; }
# Same, with the command's output going to a log file (left untouched by a dry run).
run_log() {
  local log=$1; shift
  LAST_CMD="$*"
  echo "+ $* > $log 2>&1" >&2
  [[ -n "${DRY_RUN:-}" ]] || "$@" > "$log" 2>&1
}
# Keep a trained model's checkpoint and config files in base/models/METHOD/SPLIT/seedSEED/, with
# the command that trained it (base/model_store.py).
#   keep_model METHOD SPLIT SEED "TRAINING COMMAND" [--sub casf] [--copy] FILE...
keep_model() {
  local method=$1 split=$2 seed=$3 cmd=$4
  shift 4
  run "$EVAL_PY" "$BASE/model_store.py" keep --method "$method" --split "$split" --seed "$seed" \
    --command "$cmd" "$@"
}
# Change directory and say so, so a dry run shows where each command runs.
go() { echo "+ cd ${1#"$REPO"/}" >&2; cd "$1"; }

# lp_edrscc_v2 -> v2, lp_edrscc_v2_cl123 -> cl123, anything else unchanged.
short() { case $1 in lp_edrscc_v2) echo v2 ;; lp_edrscc_v2_*) echo "${1#lp_edrscc_v2_}" ;; *) echo "$1" ;; esac; }
# Data-directory suffix: "" for lp_edrscc_v2, _cl123 for lp_edrscc_v2_cl123.
cltag() { local s; s=$(short "$1"); [[ $s == v2 ]] && echo "" || echo "_$s"; }
# "0 1 2" -> 0,1,2
comma() { local s="$*"; echo "${s// /,}"; }
# True unless EVAL_ONLY is set.
training() { [[ -z ${EVAL_ONLY:-} ]]; }
# Is lp_edrscc_v2 (the split CASF-2016 is predicted from) among the requested splits?
has_v2() { local s; for s in "${splits[@]}"; do [[ $s == lp_edrscc_v2 ]] && return 0; done; return 1; }
# Score methods' predictions on the requested splits into base/results/<method>.csv.
evaluate() { run "$EVAL_PY" "$BASE/evaluate.py" --methods "$(comma "$@")" --splits "$(comma "${splits[@]}")"; }

# The script's header comment is its usage text.
usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$SELF" >&2; exit 2; }

stage=${1:-}
[[ -n $stage ]] || usage
shift
splits=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --split) splits+=("$2"); shift 2 ;;
    --split=*) splits+=("${1#--split=}"); shift ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[[ ${#splits[@]} -gt 0 ]] || read -r -a splits <<< "$SPLITS"

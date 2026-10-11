# Conda environments by name for base/ scripts. Sourced, never run; defines functions only.
#
# CONDA_ENVS=<dir> points every lookup at <dir>/<env>; otherwise conda (CONDA_EXE, else conda on
# PATH) is asked where the env lives, so no install location is assumed.

# conda's executable: CONDA_EXE, conda on PATH, else a standard install in the home directory
# (a non-interactive shell has not run conda init).
_conda_exe() {
  if [[ -n ${CONDA_EXE:-} ]]; then echo "$CONDA_EXE"; return; fi
  command -v conda 2>/dev/null && return
  local d
  for d in miniforge3 mambaforge miniconda3 anaconda3; do
    [[ -x $HOME/$d/bin/conda ]] && { echo "$HOME/$d/bin/conda"; return; }
  done
  return 1
}

# bin/ of conda env NAME; returns non-zero (with a message) when it cannot be found.
envbin() {
  if [[ -n ${CONDA_ENVS:-} ]]; then echo "$CONDA_ENVS/$1/bin"; return; fi
  local prefix exe
  exe=$(_conda_exe) || exe=conda
  prefix=$("$exe" env list 2>/dev/null | awk -v n="$1" '$1 == n {print $NF; exit}')
  [[ -n $prefix ]] || { echo "conda env '$1' not found; create it or set CONDA_ENVS" >&2; return 1; }
  echo "$prefix/bin"
}

# Python of conda env NAME.
envpy() { local bin; bin=$(envbin "$1") || return 1; echo "$bin/python"; }

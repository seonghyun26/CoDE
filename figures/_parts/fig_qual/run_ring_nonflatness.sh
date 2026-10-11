#!/bin/bash
# Serve the ring non-flatness explorer (the web version of ring-nonflatness-nglview.ipynb).
#
# Usage:
#   bash figures/_parts/fig_qual/run_ring_nonflatness.sh              # build once if missing, then serve
#   REBUILD=1 bash figures/_parts/fig_qual/run_ring_nonflatness.sh    # rebuild the data first (~2 min/arm)
#   PORT=8877 HOST=0.0.0.0 bash figures/_parts/fig_qual/run_ring_nonflatness.sh
#
# The site is static and every fetch is relative, so it also works through the ncloud
# proxy (.../proxy/<port>/) that strips its prefix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# voxel-bind is the env with RDKit + the MMFF the build needs (and nglview, for the notebook).
# shellcheck source=../../../voxbind/scripts/lib/common.sh
source "${REPO}/voxbind/scripts/lib/common.sh"
PY="${RING_SITE_PYTHON:-$(voxbind_conda_bin voxel-bind)/python}"
SITE="${RING_SITE_DIR:-${REPO}/.cache/ring-nonflatness-site}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8768}"

if [ ! -x "${PY}" ]; then
  echo "Python not found: ${PY} (set RING_SITE_PYTHON)" >&2
  exit 1
fi

if [ "${REBUILD:-0}" = "1" ] || [ ! -f "${SITE}/data/arms.json" ]; then
  "${PY}" "${SCRIPT_DIR}/build_ring_nonflatness_site.py" --out "${SITE}"
else
  # The page itself is cheap to refresh, so a template edit never needs a full rebuild.
  cp "${SCRIPT_DIR}/ring_nonflatness_site.html" "${SITE}/index.html"
fi

# grep, not rg: this box has no ripgrep, and `rg -q` failing reads as "port free".
port_in_use() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
START_PORT="${PORT}"
for _ in $(seq 0 20); do
  port_in_use "${PORT}" || break
  PORT=$((PORT + 1))
done
if port_in_use "${PORT}"; then
  echo "No free port found in ${START_PORT}-${PORT}" >&2
  exit 1
fi
[ "${PORT}" != "${START_PORT}" ] && echo "Port ${START_PORT} busy; using ${PORT} instead" >&2

echo "Ring non-flatness explorer: http://${HOST}:${PORT}/"
exec "${PY}" -m http.server "${PORT}" --bind "${HOST}" --directory "${SITE}"

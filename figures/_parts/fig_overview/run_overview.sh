#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../voxbind/scripts/lib/common.sh
source "${SCRIPT_DIR}/../../../voxbind/scripts/lib/common.sh"
VOXBIND_PYTHON="${VOXBIND_PYTHON:-$(voxbind_conda_bin voxbind)/python}"
exec "${VOXBIND_PYTHON}" "${SCRIPT_DIR}/overview_server.py" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8767}"

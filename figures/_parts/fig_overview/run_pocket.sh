#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${VOXBIND_PYTHON:-python}" "${SCRIPT_DIR}/pocket_server.py" --host "${HOST:-127.0.0.1}" --port "${PORT:-8768}"

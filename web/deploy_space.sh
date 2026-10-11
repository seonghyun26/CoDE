#!/usr/bin/env bash
# Publish web/site/ as a static Hugging Face Space (private by default).
#
#   bash web/deploy_space.sh                  # -> <you>/drug-design-viewer, private
#   OWNER=my-lab bash web/deploy_space.sh     # under an organisation instead
#   PUBLIC=1 bash web/deploy_space.sh         # public (only applies when first created)
#
# Needs a one-time `hf auth login` with a WRITE token. Re-running updates the Space in
# place: data/ is replaced wholesale, so a file the builder no longer writes does not
# linger on the Hub.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../voxbind/scripts/lib/common.sh
source "$HERE/../voxbind/scripts/lib/common.sh"
ENV="${HFSPACE_ENV:-$(voxbind_conda_prefix hfspace)}"
[[ -n "$ENV" ]] || voxbind_die "conda env 'hfspace' not found; set HFSPACE_ENV"
HF="$ENV/bin/hf"
PY="$ENV/bin/python"
NAME="${NAME:-drug-design-viewer}"
SITE="$HERE/site"

[[ -x "$HF" ]] || { echo "no hf CLI at $HF -- see web/README.md (Hugging Face Space)"; exit 1; }
[[ -f "$SITE/index.html" && -f "$SITE/data/manifest.json" ]] \
  || { echo "no built site at $SITE -- run web/build.py first"; exit 1; }

USER_NAME="$("$PY" -c 'from huggingface_hub import whoami; print(whoami()["name"])' 2>/dev/null)" \
  || { echo "not logged in -- run: $HF auth login   (a WRITE token)"; exit 1; }
OWNER="${OWNER:-$USER_NAME}"
REPO="$OWNER/$NAME"
VIS="--private"; [[ "${PUBLIC:-0}" == "1" ]] && VIS="--public"

echo "[space] $REPO  (logged in as $USER_NAME; ${VIS#--} if newly created)"
"$HF" repos create "$REPO" --type space --sdk static "$VIS" --exist-ok

STAMP="$(date '+%Y-%m-%d %H:%M')"
"$HF" upload "$REPO" "$SITE" . --type space --delete "data/**" \
  --commit-message "site: $STAMP"
"$HF" upload "$REPO" "$HERE/space/README.md" README.md --type space \
  --commit-message "space card: $STAMP"

echo "[space] https://huggingface.co/spaces/$REPO"

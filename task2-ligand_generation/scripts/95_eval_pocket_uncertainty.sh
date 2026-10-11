#!/usr/bin/env bash
# 95_eval_pocket_uncertainty.sh — per-pocket crystallographic uncertainty beside per-pocket Vina.
#
#   bash task2-ligand_generation/scripts/95_eval_pocket_uncertainty.sh
#   conda run -n voxbind python figures/draw.py uncertainty
#
# Asks whether the gain CoDE and CoE take over VoxBind is larger where the crystal is well
# ordered -- which is what it should be if the electron-density conditioning is what buys the
# gain. B-factor is the deposited per-atom measure of disorder and occupancy of how much of
# the atom is there; both are written per pocket, for the 10 A crop the sampler saw and for
# the receptor atoms the crystal ligand touches, raw and normalized within each structure.
#
# ~6 s, no GPU. Writes results/task2-drugdesign/_shared/260916_pocket_uncertainty/
# pocket_uncertainty.json, which figures/_parts/uncertainty.py draws and does not recompute.
#
# B-factors come off the CrossDocked receptor (verified atom-for-atom identical to the
# deposited entry); OCCUPANCY CANNOT -- CrossDocked keeps one altloc and rewrites every
# occupancy to 1.00 -- so it is read from the deposited mmCIF in dataset/data/cif/,
# which the density pipeline already caches. A receptor missing from that cache is fetched
# from RCSB below; without network, the occupancy columns for it come out null and the
# B-factor columns are unaffected.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
# shellcheck source=../../voxbind/scripts/lib/common.sh
source "$ROOT/voxbind/scripts/lib/common.sh"
PY="${PY:-$(voxbind_conda_bin voxbind)/python}"
CIF_DIR="${CIF_DIR:-dataset/data/cif}"
OUT="${OUT:-results/task2-drugdesign/_shared/260916_pocket_uncertainty/pocket_uncertainty.json}"

# The receptor entries the 79 pockets need, taken from the bundle's own per-molecule table
# so this can never drift from the pocket set the figures draw.
mkdir -p "$CIF_DIR"
missing=$("$PY" - "$CIF_DIR" <<'PY'
import csv, os, sys
cif_dir = sys.argv[1]
ids = set()
with open("results/task2-drugdesign/VoxBind-Ours/eval/vina_docking/per_molecule.csv",
          newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r.get("in_density79") == "1":
            ids.add(r["ligand_filename"].split("/", 1)[1][:10].split("_")[0].lower())
print(" ".join(sorted(i for i in ids if not os.path.exists(f"{cif_dir}/{i}.cif"))))
PY
)
if [ -n "${missing// /}" ]; then
    echo "  fetching $(printf '%s\n' $missing | wc -l) deposited mmCIF entries from RCSB"
    for id in $missing; do
        curl -sfL --retry 3 -o "$CIF_DIR/$id.cif.part" \
            "https://files.rcsb.org/download/$id.cif" \
            && mv "$CIF_DIR/$id.cif.part" "$CIF_DIR/$id.cif" \
            || echo "  WARNING: $id not fetched — its occupancy columns will be null"
    done
fi

"$PY" task2-ligand_generation/scripts/tools/eval_pocket_uncertainty.py --out "$OUT" "$@"

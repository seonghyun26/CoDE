#!/usr/bin/env python3
"""Compare the figures on disk with the ones committed in git, after a redraw.

    bash figures/draw_all.sh
    python figures/_tools/compare_outputs.py              # against HEAD
    python figures/_tools/compare_outputs.py --ref abc123 --list

A PNG is compared pixel by pixel (decoded, so metadata and compression do not count); a
PDF byte for byte once its timestamps and document id are blanked. The committed side is
read out of git, never off disk -- a redraw writes over the very files it is compared
with. Exit status 1 when any figure differs.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FIG = Path(__file__).resolve().parents[1]
REPO = FIG.parent
VOLATILE = re.compile(rb"/(CreationDate|ModDate)\s*\(D:[^)]*\)|/ID\s*\[<[0-9A-Fa-f]+>\s*<[0-9A-Fa-f]+>\]")


def committed(ref, path):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def same_png(a, b):
    x = np.asarray(Image.open(io.BytesIO(a)).convert("RGBA"))
    y = np.asarray(Image.open(io.BytesIO(b)).convert("RGBA"))
    if x.shape != y.shape:
        return False, f"size {y.shape[1]}x{y.shape[0]} -> {x.shape[1]}x{x.shape[0]}"
    diff = np.any(x != y, axis=-1)
    if not diff.any():
        return True, ""
    return False, f"{diff.mean():.2%} of pixels, max |d| {int(np.abs(x.astype(int) - y.astype(int)).max())}"


def same_pdf(a, b):
    return VOLATILE.sub(b"", a) == VOLATILE.sub(b"", b), "content"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="HEAD", help="git revision to compare with (default HEAD)")
    ap.add_argument("--list", action="store_true", help="print every differing file")
    args = ap.parse_args()

    files = sorted(p for p in FIG.glob("fig-*/**/*") if p.suffix in (".png", ".pdf"))
    counts = {"same": 0, "differ": 0, "new": 0}
    rows = []
    for p in files:
        rel = p.relative_to(REPO).as_posix()
        old = committed(args.ref, rel)
        if old is None:
            counts["new"] += 1
            rows.append(("new", rel, ""))
            continue
        ok, why = (same_png if p.suffix == ".png" else same_pdf)(p.read_bytes(), old)
        counts["same" if ok else "differ"] += 1
        if not ok:
            rows.append(("differ", rel, why))
    # committed figures that are no longer on disk
    listed = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", args.ref, "figures/"],
                            capture_output=True, text=True).stdout.split()
    gone = [f for f in listed if f.startswith("figures/fig-") and Path(f).suffix in (".png", ".pdf")
            and not (REPO / f).exists()]
    print(f"{len(files)} PNG/PDF under figures/fig-*: {counts['same']} identical to {args.ref}, "
          f"{counts['differ']} differ, {counts['new']} not in {args.ref}; {len(gone)} in {args.ref} but gone")
    if args.list or counts["differ"] <= 40:
        for kind, rel, why in rows:
            if kind == "differ" or args.list:
                print(f"  {kind:6s} {rel}  {why}")
        for rel in gone if args.list else []:
            print(f"  gone   {rel}")
    return 1 if counts["differ"] else 0


if __name__ == "__main__":
    sys.exit(main())

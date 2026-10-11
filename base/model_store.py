#!/usr/bin/env python3
"""Trained baseline models, kept as base/models/<method>/<split>/seed<N>/.

Each folder holds the checkpoint under the trainer's own file name, the config the run used and
cmd.txt, one line per training run: time, git commit, working directory and command. A second model
trained for the same split and seed (the one that predicts CASF-2016) goes in a casf/ sub-folder.

Trainers that keep their best weights in memory call save_state(). For the ones that write their
own checkpoints, base/scripts/<baseline>.sh calls this file after training:

    python base/model_store.py keep --method bindnet --split lp_edrscc_v2 --seed 0 [--sub casf] \\
        [--copy] --command "<training command>" FILE_OR_DIR ...

Files of 64 MB or more are hard-linked, so a large checkpoint takes no extra disk; smaller ones and
--copy are copied. A hard link shares the trainer's file, so retraining the same split and seed
rewrites it in place until the new run is kept. BASE_MODELS overrides the base/models root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("BASE_MODELS", BASE / "models"))
LINK_MIN_BYTES = 64 << 20
# Short split tags some baselines pass, mapped to the frozen split names.
SPLIT_NAMES = {"v2": "lp_edrscc_v2", "edrscc": "lp_edrscc_v2", "cl123": "lp_edrscc_v2_cl123"}


# base/models/<method>/<split>/seed<N>[/<sub>]
def model_dir(method: str, split: str, seed: int, sub: str | None = None) -> Path:
    folder = MODELS / method / SPLIT_NAMES.get(split, split) / f"seed{seed}"
    return folder / sub if sub else folder


# The checkout's commit, marked +dirty when base/ has uncommitted changes.
def _commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    dirty = subprocess.run(["git", "-C", str(BASE), "diff", "--quiet", "HEAD", "--", "."],
                           capture_output=True).returncode != 0
    return sha + ("+dirty" if dirty else "")


# Append a run to cmd.txt; the command defaults to this process's own command line.
def record(folder: Path, command: str | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    command = command or " ".join([sys.executable, *sys.argv])
    with open(folder / "cmd.txt", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {_commit()}  cd {os.getcwd()} && {command}\n")


# Save an in-memory state (a state dict, or a dict that also holds scalers or a vocabulary) and
# the config it was trained with; returns the folder.
def save_state(method: str, split: str, seed: int, state, name: str, config: dict | None = None,
               sub: str | None = None) -> Path:
    import torch

    folder = model_dir(method, split, seed, sub)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{name}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, folder / name)
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    record(folder)
    print(f"model -> {folder}", flush=True)
    return folder


# One file as dst: a hard link when large (copy forces a copy), a copy otherwise or across file
# systems. Written under a temporary name, then renamed over any older file.
def _place_file(src: Path, dst: Path, copy: bool = False) -> None:
    src = Path(os.path.realpath(src))
    tmp = dst.with_name(f".{dst.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    linked = False
    if not copy and src.stat().st_size >= LINK_MIN_BYTES:
        try:
            os.link(src, tmp)
            linked = True
        except OSError:
            pass
    if not linked:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


# Keep files or folders a trainer wrote, then record the command that trained them.
def keep(method: str, split: str, seed: int, paths, command: str, sub: str | None = None,
         copy: bool = False) -> Path:
    folder = model_dir(method, split, seed, sub)
    folder.mkdir(parents=True, exist_ok=True)
    for path in map(Path, paths):
        if not path.exists():
            raise FileNotFoundError(path)
        target = folder / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target, copy_function=lambda s, d: _place_file(Path(s), Path(d), copy))
        else:
            _place_file(path, target, copy)
    record(folder, command)
    print(f"model -> {folder}", flush=True)
    return folder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    k = sub.add_parser("keep", help="keep a trainer's checkpoint and config files")
    k.add_argument("--method", required=True)
    k.add_argument("--split", required=True)
    k.add_argument("--seed", type=int, required=True)
    k.add_argument("--sub", help="sub-folder for a second model of the same seed, e.g. casf")
    k.add_argument("--copy", action="store_true", help="copy even large files, e.g. when the trainer "
                                                        "reuses the file name for another split")
    k.add_argument("--command", required=True, help="the command that trained the model")
    k.add_argument("paths", nargs="+")
    args = ap.parse_args()
    keep(args.method, args.split, args.seed, args.paths, args.command, args.sub, args.copy)


if __name__ == "__main__":
    main()

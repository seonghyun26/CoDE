#!/usr/bin/env python3
"""Pretrain an encoder and score it on the six affinity test sets, from one YAML config.

    python task1-affinity_regression/run.py task1-affinity_regression/configs/ab7v2p1_clean.yaml
    python task1-affinity_regression/run.py CONFIG --stages evaluate     # an encoder already trained
    python task1-affinity_regression/run.py CONFIG --dry-run             # print what would run

Stages, each skipped when its output is already there (--force redoes features and evaluate):
  pretrain   pretrain/scripts/pretrain.sh; a run folder without checkpoint_final.json is resumed
  features   probe.py features of the chosen checkpoint, one cache per readout and layer
  evaluate   a layer per readout by CL3 validation ρ, then the six test sets (evals/sixset.py);
             per-seed rows in <out_dir>/sixset.csv, mean ± std in <out_dir>/sixset.json

Run it in the voxbind environment; the config keys are described in configs/README.md.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

TASK = Path(__file__).resolve().parent
REPO = TASK.parent
STAGES = ("pretrain", "features", "evaluate")
EVALUATE = {               # evaluate: keys and their defaults
    "run_dir": None,       # the encoder's run folder; default pretrain/exps/<name>
    "epoch": "best",       # checkpoint epoch, or best = the trainer's best-validation checkpoint
    "condition": None,     # encoder input condition (probe.py features --condition); required
    "gpu": None,           # one physical GPU; default the first pretrain GPU, else 0
    "seeds": 5,            # probe-head seeds per split, as Table 1
    "select_seeds": 5,     # seeds per candidate layer when a readout lists several
    "num_workers": 0,      # extraction loader workers; with workers the pin-memory thread failed on
                           # every 2026-10-10 run, idle GPUs included (a failure is retried with 0)
    "features_args": [],   # more probe.py features arguments; paths relative to the repository
    # Every readout pools the last layer's tokens with the Table 1 head; only the pooling varies.
    "readouts": [
        {"name": "mean", "pool": "mean", "head": "default"},
        {"name": "max", "pool": "max", "head": "default"},
        {"name": "meanstd", "pool": "meanstd", "head": "default"},
        {"name": "groupmean", "pool": "groupmean", "head": "default"},
        {"name": "groupstats", "pool": "groupstats", "head": "default"},
    ],
    "out_dir": None,       # default results/task1-affinity/<name>
}
PRETRAIN = {"experiment": None, "trainer": "live",
            "gpus": None,          # default: the GPUs voxbind/scripts/99_chain.sh reserved (VOXBIND_GPUS)
            "overrides": []}
READOUT = {"name", "pool", "layer", "layers", "head"}


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def die(msg: str) -> None:
    log(f"error: {msg}")
    sys.exit(1)


def shown(path: Path) -> str:
    """A path relative to the repository when it is inside it."""
    path = Path(path)
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    unknown = set(cfg) - {"name", "pretrain", "evaluate"}
    if unknown:
        die(f"{path}: unknown keys {sorted(unknown)}")
    if not cfg.get("name"):
        die(f"{path}: name is required")
    pre = cfg.get("pretrain")
    if pre is not None:
        if set(pre) - set(PRETRAIN):
            die(f"{path}: unknown pretrain keys {sorted(set(pre) - set(PRETRAIN))}")
        cfg["pretrain"] = pre = {**PRETRAIN, **pre}
        if pre["gpus"] is None:
            pre["gpus"] = os.environ.get("VOXBIND_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
        if not pre["experiment"] or not pre["gpus"]:
            die(f"{path}: pretrain needs experiment and gpus (or GPUs from 99_chain.sh)")
    ev = cfg.get("evaluate") or {}
    if set(ev) - set(EVALUATE):
        die(f"{path}: unknown evaluate keys {sorted(set(ev) - set(EVALUATE))}")
    cfg["evaluate"] = ev = {**EVALUATE, **ev}
    if not ev["condition"]:
        die(f"{path}: evaluate.condition is required")
    for ro in ev["readouts"]:
        if set(ro) - READOUT or "name" not in ro:
            die(f"{path}: a readout has keys {sorted(ro)}; allowed {sorted(READOUT)}, name required")
    if ev["gpu"] is None:
        ev["gpu"] = int(str(pre["gpus"]).replace("-", ",").split(",")[0]) if pre else 0
    name = cfg["name"]
    ev["run_dir"] = REPO / (ev["run_dir"] or f"pretrain/exps/{name}")
    ev["out_dir"] = REPO / (ev["out_dir"] or f"results/task1-affinity/{name}")
    return cfg


def run(cmd: list, dry: bool, env: dict | None = None, cwd: Path = REPO, logfile: Path | None = None) -> int:
    shown = " ".join(shlex.quote(str(c)) for c in cmd) + (f" > {logfile}" if logfile else "")
    log(("would run: " if dry else "run: ") + shown)
    if dry:
        return 0
    env = {**os.environ, **(env or {})}
    if logfile is None:
        return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env).returncode
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a") as fh:
        return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def training_now(name: str) -> bool:
    """A process on this machine is training this run (pretrain.sh passes exp_name=<name>)."""
    try:
        return subprocess.run(["pgrep", "-f", f"exp_name={name}( |$)"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


# ── stages ──────────────────────────────────────────────────────────────────────────────────────
def stage_pretrain(cfg: dict, dry: bool) -> None:
    pre, name, run_dir = cfg.get("pretrain"), cfg["name"], cfg["evaluate"]["run_dir"]
    if not pre:
        log(f"pretrain: not in the config; evaluating {run_dir}")
        return
    if (run_dir / "checkpoint_final.json").exists():
        log(f"pretrain: {name} is finished (checkpoint_final.json)")
        return
    if training_now(name):
        msg = f"pretrain: {name} is being trained by another process; run the later stages when it ends"
        if not dry:
            die(msg)
        log(msg)
        return
    cmd = ["bash", REPO / "pretrain/scripts/pretrain.sh", "--experiment", pre["experiment"],
           "--name", name, "--gpus", str(pre["gpus"])]
    if pre["trainer"] != "live":
        cmd += ["--trainer", pre["trainer"]]
    overrides = [str(o) for o in pre["overrides"]]
    if (run_dir / "checkpoint.pth.tar").exists():
        log(f"pretrain: resuming {name} from its last checkpoint")
        overrides.append(f"resume={run_dir}")
    if overrides:
        cmd += ["--", *overrides]
    torchrun = Path(sys.executable).parent / "torchrun"
    env = {"VOXBIND_PY": str(torchrun.parent)} if torchrun.exists() else None
    if run(cmd, dry, env) != 0:
        die(f"pretraining failed; see pretrain/log/{name}.log")
    if not dry and not (run_dir / "checkpoint_final.json").exists():
        die(f"pretraining ended without {run_dir}/checkpoint_final.json; see pretrain/log/{name}.log")


def resolve_epoch(ev: dict, dry: bool = False):
    """evaluate.epoch, with best read from the trainer's record (final, else best so far)."""
    if str(ev["epoch"]) != "best":
        return int(ev["epoch"])
    for record in ("checkpoint_final.json", "checkpoint_best.json"):
        path = ev["run_dir"] / record
        if path.exists():
            return int(json.loads(path.read_text())["epoch"])
    if dry:
        return "best"
    die(f"epoch best: {ev['run_dir']} has no checkpoint_best.json yet; set evaluate.epoch")


def readout_layers(ro: dict) -> list:
    if ro.get("layers"):
        return [int(x) for x in ro["layers"]]
    return [int(ro.get("layer", 0))]


def cache_path(cfg: dict, epoch: int, pool: str, layer: int) -> Path:
    """probe.py features' output name for this run, checkpoint and readout."""
    ev = cfg["evaluate"]
    tag = "_".join(str(cfg["name"]).split())
    readout = "" if pool in ("mean", "cls_concat") and not layer else f"_{pool}" + (f"-L{layer}" if layer else "")
    return REPO / "dataset/data/pdbbind/features" / f"{ev['condition']}_e{epoch}_v5_{tag}{readout}.pt"


def density_plan(ev: dict) -> Optional[dict]:
    """How probe.py will scale the density for this encoder (density_match.py), or None without a cfg."""
    if not (ev["run_dir"] / "cfg.yaml").exists():
        return None
    import density_match
    from omegaconf import OmegaConf
    args = [str(a) for a in ev["features_args"]]
    return density_match.plan(
        OmegaConf.load(ev["run_dir"] / "cfg.yaml"), REPO / "dataset/data/pdbbind/voxels_v5/stats.json",
        overridden=any(a in args for a in ("--dens_dir", "--noise_voxels_dir")),
        disabled="--no_density_match" in args)


def cache_current(path: Path, plan: Optional[dict]) -> bool:
    """A cache that uses density was extracted with the density scale probe.py would use now."""
    if plan is None:
        return True
    import torch
    import density_match
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.get("dens_dir") is None or density_match.same(blob.get("density_match"), plan)


def stage_features(cfg: dict, dry: bool, force: bool) -> None:
    ev, name = cfg["evaluate"], cfg["name"]
    if not dry and not ev["run_dir"].is_dir():
        die(f"features: no run folder {ev['run_dir']}")
    epoch = resolve_epoch(ev, dry)
    plan = density_plan(ev)
    for ro in ev["readouts"]:
        pool = ro.get("pool", "mean")
        for layer in readout_layers(ro):
            out = cache_path(cfg, epoch, pool, layer)
            if out.exists() and not force:
                if cache_current(out, plan):
                    log(f"features: {out.name} exists")
                    continue
                log(f"features: {out.name} has another density scale than this encoder's; extracting again")
            cmd = [sys.executable, TASK / "probe.py", "features", "--condition", ev["condition"],
                   "--epoch", epoch, "--exp_dir", ev["run_dir"], "--tag", name]
            if pool != "mean" or layer:
                cmd += ["--pool", pool] + (["--layer", layer] if layer else [])
            cmd += [str(a) for a in ev["features_args"]]
            logfile = TASK / "log" / name / f"features{out.stem.split(name, 1)[-1] or '_mean'}.log"
            env = {"CUDA_VISIBLE_DEVICES": str(ev["gpu"])}
            rc = run(cmd + ["--num_workers", ev["num_workers"]], dry, env, REPO, logfile)
            if rc != 0 and int(ev["num_workers"]) > 0:
                log(f"features: failed with {ev['num_workers']} workers (see {logfile}); retrying with 0")
                rc = run(cmd + ["--num_workers", 0], dry, env, REPO, logfile)
            if rc != 0:
                die(f"features: extraction failed; see {logfile}")


def stage_evaluate(cfg: dict, dry: bool, force: bool, config_path: Path) -> None:
    ev, name = cfg["evaluate"], cfg["name"]
    out_csv, out_json = ev["out_dir"] / "sixset.csv", ev["out_dir"] / "sixset.json"
    if out_json.exists() and not force:
        log(f"evaluate: {shown(out_json)} exists")
        return
    if dry:
        log(f"would evaluate {[ro['name'] for ro in ev['readouts']]} on the six test sets "
            f"({ev['seeds']} seeds, GPU {ev['gpu']}) into {ev['out_dir']}")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ev["gpu"])
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, str(TASK / "evals"))
    import sixset  # noqa: E402  (imports torch: after CUDA_VISIBLE_DEVICES)

    epoch = resolve_epoch(ev)
    rows, result = [], {"name": name, "config": str(config_path), "run_dir": str(ev["run_dir"]),
                        "epoch": epoch, "condition": ev["condition"], "seeds": ev["seeds"],
                        "git": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                              capture_output=True, text=True).stdout.strip(),
                        "readouts": {}}
    for ro in ev["readouts"]:
        pool, head, layers = ro.get("pool", "mean"), ro.get("head", "default"), readout_layers(ro)
        caches = {}
        for layer in layers:
            path = cache_path(cfg, epoch, pool, layer)
            features, meta = sixset.load_cache(path)
            # the run folder's last two parts: the repository may sit elsewhere, e.g. in a container
            if Path(meta.get("exp_dir", "")).parts[-2:] != ev["run_dir"].resolve().parts[-2:] or int(meta["epoch"]) != epoch:
                die(f"evaluate: {path.name} was built from {meta.get('exp_dir')} epoch {meta.get('epoch')}; "
                    f"rerun with --force")
            caches[layer] = (path, features)
        picked, val = layers[0], {}
        if len(layers) > 1:
            for layer, (path, features) in caches.items():
                val[layer] = sixset.validation_rho(features, head, ev["select_seeds"], "cuda")
                log(f"evaluate: {ro['name']} layer {layer}: CL3 val ρ {val[layer]:.4f} ({ev['select_seeds']} seeds)")
            picked = max(val, key=val.get)
        path, features = caches[picked]
        log(f"evaluate: {ro['name']} = {path.name}, head {head}")
        ro_rows = sixset.evaluate(features, head, ev["seeds"], "cuda", log=log)
        summ = sixset.summary(ro_rows)
        print(sixset.table(summ), flush=True)
        rows += [{"method": name, "readout": ro["name"], "layer": picked, **r,
                  "source": shown(path)} for r in ro_rows]
        result["readouts"][ro["name"]] = {"pool": pool, "layer": picked, "head": head,
                                          "layer_val_rho": val or None,
                                          "cache": shown(path), "test_sets": summ}
    ev["out_dir"].mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out_json.write_text(json.dumps(result, indent=1) + "\n")
    (ev["out_dir"] / "config.yaml").write_text(config_path.read_text())
    log(f"evaluate: wrote {shown(out_csv)} and {shown(out_json)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("config", type=Path, help="YAML config (task1-affinity_regression/configs/)")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list from {','.join(STAGES)}")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--force", action="store_true", help="redo features and evaluate even if their outputs exist")
    args = ap.parse_args()
    stages = [s for s in args.stages.split(",") if s]
    if set(stages) - set(STAGES):
        die(f"unknown stages {sorted(set(stages) - set(STAGES))}")
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    log(f"{cfg['name']}: stages {','.join(stages)} ({shown(config_path)})")
    if "pretrain" in stages:
        stage_pretrain(cfg, args.dry_run)
    if "features" in stages:
        stage_features(cfg, args.dry_run, args.force)
    if "evaluate" in stages:
        stage_evaluate(cfg, args.dry_run, args.force, config_path)


if __name__ == "__main__":
    main()

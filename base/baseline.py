#!/usr/bin/env python3
"""Run the external affinity baselines of Table 1 from one place.

    python base/baseline.py --list                                   # baselines and their scripts
    python base/baseline.py --status                                 # what base/results/ holds
    python base/baseline.py --baselines get,profsa --dry-run         # print the exact commands
    python base/baseline.py --all                                    # every baseline: train_eval, 5 seeds
    python base/baseline.py --all --eval-only                        # rescore the predictions on disk
    python base/baseline.py --baselines cheapnet --stages featurize,train_eval --splits lp_edrscc_v2

Each baseline's commands live in its own script, base/scripts/<script>.sh, which also runs on its
own (`bash base/scripts/aevplig.sh train_eval --split lp_edrscc_v2`). Two stages:
  featurize   our complexes (base/datasource.py) -> base/data/<baseline>/
  train_eval  train on each split, predict its test set (and CASF-2016 for lp_edrscc_v2), and score
              the predictions into base/results/<method>.csv (base/evaluate.py); each trained model
              is kept as base/models/<method>/<split>/seed<N>/ (base/model_store.py)
This file picks the scripts, runs one baseline per GPU and keeps one log per baseline under
log/baselines/<timestamp>/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
SCRIPTS = BASE / "scripts"

STAGES = ("featurize", "train_eval")
DEFAULT_SPLITS = ("lp_edrscc_v2", "lp_edrscc_v2_cl123")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_GPUS = (4, 5, 6, 7)          # 0-3 is the pretraining lane


@dataclass
class Baseline:
    label: str
    script: str                      # base/scripts/<script>.sh
    mode: str                        # trained | frozen_probe | zero_shot
    env: dict = field(default_factory=dict)    # picks one model in a multi-model script


BASELINES: dict[str, Baseline] = {
    "aevplig": Baseline(
        "AEV-PLIG (Commun. Chem. 2025): GATv2 over radial-AEV ligand graphs",
        "aevplig", "trained"),
    "bindnet": Baseline(
        "BindNet (ICLR 2024): trained from scratch, the released weights are restricted",
        "bindnet", "trained"),
    "cheapnet": Baseline(
        "CheapNet (ICLR 2025): hierarchical cross-attention over atom clusters",
        "cheapnet", "trained"),
    "dsmbind": Baseline(
        "DSMBind (bioRxiv 2023): unsupervised binding energy, zero-shot, |r| and |rho|",
        "dsmbind", "zero_shot"),
    "deepdta": Baseline(
        "DeepDTA (Bioinformatics 2018) via DeepPurpose: sequence + SMILES",
        "dta", "trained", {"MODELS": "deepdta"}),
    "moltrans": Baseline(
        "MolTrans (Bioinformatics 2021) via DeepPurpose: sequence + SMILES",
        "dta", "trained", {"MODELS": "moltrans"}),
    "get": Baseline(
        "GET (ICML 2024): E(3)-equivariant generalist interaction transformer",
        "get", "trained", {"METHODS": "GET"}),
    "egnn": Baseline(
        "EGNN, as shipped in the GET repo",
        "get", "trained", {"METHODS": "EGNN"}),
    "egnn_td": Baseline(
        "the GET repo's EGNN with N(0, 1 A) coordinate noise in training and inference (noisy_sigma 1); "
        "reported as EGNN + TargetDiff until 2026-10-10, which it is not (td_egnn_enc is)",
        "get", "trained", {"METHODS": "EGNN_TD"}),
    "td_egnn": Baseline(
        "EGNN, TargetDiff's property-prediction code",
        "targetdiff", "trained", {"MODELS": "egnn"}),
    "td_egnn_enc": Baseline(
        "EGNN + TargetDiff: the same EGNN on the hidden states of TargetDiff's pretrained diffusion model",
        "targetdiff", "trained", {"MODELS": "egnn_enc"}),
    "geossl": Baseline(
        "GeoSSL-DDM (arXiv 2206.13602): small-molecule SchNet, frozen, + MLP probe",
        "geossl", "frozen_probe"),
    "hbgsa": Baseline(
        "HBGSA (arXiv 2604.23115): H-bond graph model, our 3.06M reimplementation",
        "hbgsa", "trained"),
    "honestaffinity": Baseline(
        "HonestAffinity (arXiv 2606.03422): frozen ESM-2 features + trained head",
        "honestaffinity", "trained"),
    "ipnet_frozen": Baseline(
        "IPNet from IPDiff (ICLR 2024): released network, frozen, + MLP head",
        "ipnet", "frozen_probe", {"VARIANTS": "frozen"}),
    "ipnet_scratch": Baseline(
        "IPNet from IPDiff (ICLR 2024): the same network trained from scratch",
        "ipnet", "trained", {"VARIANTS": "scratch"}),
    "nesso": Baseline(
        "Nesso-1 (Recursion, 2026): zero-shot co-folding affinity; trained on our test sets",
        "nesso", "zero_shot"),
    "profsa": Baseline(
        "ProFSA (ICLR 2024): pretrained pocket encoder, frozen, + regression head",
        "profsa", "frozen_probe"),
}

GROUPS = {"all": list(BASELINES)} | {
    mode: [k for k, b in BASELINES.items() if b.mode == mode]
    for mode in ("trained", "frozen_probe", "zero_shot")}


RESULTS = BASE / "results"
MAIN_TEST_SET = {"lp_edrscc_v2": "FULL", "lp_edrscc_v2_cl123": "CL3"}


def shown_path(path: Path) -> str:
    """A path relative to the repo when it is inside it."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def run_baseline(name: str, splits: list, seeds: tuple, gpu: int, stages: tuple,
                 log_dir: Path, dry_run: bool, eval_only: bool) -> tuple:
    """Run the requested stages of one baseline, in order, stopping at the first failure."""
    bl = BASELINES[name]
    env = {**os.environ, **bl.env, "SEEDS": " ".join(map(str, seeds)), "GPU": str(gpu)}
    if dry_run:
        env["DRY_RUN"] = "1"
    if eval_only:
        env["EVAL_ONLY"] = "1"
    log_path = log_dir / f"{name}.log"
    started = time.time()
    with open(os.devnull if dry_run else log_path, "a") as log:
        for stage in (s for s in STAGES if s in stages):
            argv = ["bash", str(SCRIPTS / f"{bl.script}.sh"), stage]
            for split in splits:
                argv += ["--split", split]
            shown = " ".join(f"{k}={v}" for k, v in bl.env.items()) + " " + " ".join(argv)
            print(f"[{name}/{stage}] gpu{gpu} $ {shown.strip()}", flush=True)
            if dry_run:
                out = subprocess.run(argv, env=env, capture_output=True, text=True)
                print("".join(f"    {line}\n" for line in out.stderr.splitlines()), end="")
                continue
            log.write(f"\n=== {datetime.now():%F %T} {stage}\n$ {shown.strip()}\n")
            log.flush()
            rc = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                log.write(f"[FAIL rc={rc}]\n")
                return name, "failed", f"{stage} rc={rc}", log_path
    return name, "ok", f"{(time.time() - started) / 60:.1f} min", log_path


def cmd_list() -> None:
    print(f"{'baseline':<16}{'mode':<14}{'script':<30}what it is")
    print("-" * 122)
    for name, bl in BASELINES.items():
        script = f"scripts/{bl.script}.sh" + (f" {next(iter(bl.env.values()))}" if bl.env else "")
        print(f"{name:<16}{bl.mode:<14}{script:<30}{bl.label}")


def cmd_status(splits: list) -> None:
    """Seeds and mean Spearman rho per split's main test set, from base/results/<method>.csv."""
    print(f"{'baseline':<16}" + "".join(f"{s:<26}" for s in splits))
    print("-" * (16 + 26 * len(splits)))
    for name in BASELINES:
        path = RESULTS / f"{name}.csv"
        rows = list(csv.DictReader(open(path))) if path.exists() else []
        cells = ""
        for split in splits:
            test = MAIN_TEST_SET.get(split, "test")
            rho = [float(r["spearman"]) for r in rows if r["split"] == split and r["test_set"] == test]
            cells += f"{f'{test} rho {sum(rho) / len(rho):.3f} ({len(rho)} seeds)' if rho else '-':<26}"
        print(f"{name:<16}{cells}")
    print("\nRescore what is on disk: python base/baseline.py --all --eval-only")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", default="",
                    help=f"comma-separated names, or a group: {', '.join(GROUPS)}")
    ap.add_argument("--all", action="store_true", help="same as --baselines all")
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--gpus", default=",".join(map(str, DEFAULT_GPUS)),
                    help="GPU pool; one baseline runs per GPU")
    ap.add_argument("--stages", default="train_eval", help=f"subset of {', '.join(STAGES)}")
    ap.add_argument("--eval-only", action="store_true",
                    help="train_eval without training: rescore the predictions on disk")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a baseline whose results CSV already has every split")
    ap.add_argument("--log-dir", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.list:
        cmd_list()
        return 0
    if args.status:
        cmd_status(splits)
        return 0

    picked = args.baselines.strip() or ("all" if args.all else "")
    if not picked:
        ap.error("pass --baselines <names|group> or --all (see --list)")
    names = GROUPS.get(picked) or [n.strip() for n in picked.split(",") if n.strip()]
    unknown = [n for n in names if n not in BASELINES]
    if unknown:
        ap.error(f"unknown baseline(s) {unknown}; see --list")
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"unknown stage(s) {bad}; choose from {STAGES}")

    if args.skip_existing:
        def complete(n):
            path = RESULTS / f"{n}.csv"
            have = {r["split"] for r in csv.DictReader(open(path))} if path.exists() else set()
            return all(s in have for s in splits)
        done = [n for n in names if complete(n)]
        for n in done:
            print(f"[skip] {n}: base/results/{n}.csv has {', '.join(splits)}")
        names = [n for n in names if n not in done]
    if not names:
        print("nothing to run.")
        return 0

    log_dir = Path(args.log_dir) if args.log_dir else \
        REPO / "log" / "baselines" / datetime.now().strftime("%y%m%d_%H%M%S")
    print(f"{len(names)} baseline(s), stages={list(stages)}, splits={splits}, "
          f"seeds={list(seeds)}, gpus={gpus}")
    if args.dry_run:
        for i, name in enumerate(names):
            run_baseline(name, splits, seeds, gpus[i % len(gpus)], stages, log_dir, True, args.eval_only)
        print("\n[dry-run] nothing executed.")
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logs: {shown_path(log_dir)}\n")

    # One worker per GPU; a baseline keeps its GPU for all of its stages.
    free = list(gpus)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        pending = {}
        queue = list(names)
        while queue or pending:
            while queue and free:
                name, gpu = queue.pop(0), free.pop(0)
                fut = pool.submit(run_baseline, name, splits, seeds, gpu, stages, log_dir, False,
                                  args.eval_only)
                pending[fut] = gpu
            finished, _ = concurrent.futures.wait(pending, return_when="FIRST_COMPLETED")
            for fut in finished:
                free.append(pending.pop(fut))
                results.append(fut.result())
                name, state, detail, _ = results[-1]
                print(f"[{state}] {name} {detail}", flush=True)

    failed = [r for r in results if r[1] != "ok"]
    for name, _, detail, log_path in failed:
        print(f"FAILED {name}: {detail} (log: {shown_path(log_path)})")
    print("\nScores: base/results/<method>.csv.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

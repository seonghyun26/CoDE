"""Bring a checkout to the current folder layout. Run it once per checkout, after pulling:

    python dataset/build/migrate_layout.py            # dry run: report what would change
    python dataset/build/migrate_layout.py --apply    # do it

Git moves the code; this moves what git does not track, for two layout changes. Each part does
nothing when there is nothing left to move, so the script is safe to re-run.

2026-10-01, data: voxbind/dataset/data -> dataset/data.
  1. moves the folder (a rename, so it needs one filesystem);
  2. rewrites every symlink in the repo whose target lies in the old location, as a relative
     link to the same file in the new one (absolute links break when a checkout moves);
  3. rewrites the old absolute prefix in the data's text records (.json .csv .tsv .yaml .yml
     .txt). Loaders read some of these, e.g. ccp4_dir in each corpus's resample.json; a stale
     one makes every sample load without density.

2026-10-06, task folders: voxbind/ keeps only the shared library; the generator moved to
task2-ligand_generation/, the affinity probe to task1-affinity_regression/, the model zoo to
model_zoo/.
  1. splits voxbind/exps by run: pretraining runs go to pretrain/exps, generator runs (and
     anything without a pretraining config) to task2-ligand_generation/exps;
  2. moves what git left in the old folders (weights, wandb, caches, local files) into the new
     ones, merging folder by folder; voxbind/log and voxbind/logs go to log/;
  3. rewrites symlinks into any moved location, as in the data move;
  4. in the moved generator runs' configs, points encoder checkpoints at their new place
     (density_pretrained_path under voxbind/exps or model_zoo), so sample.py can rebuild them.
A file that exists in both places is never overwritten; it is reported and left where it is.

Before changing anything, --apply writes a tar.gz of each text file it will rewrite and a TSV of
each symlink it will change to the repo root, so the step can be undone.

--old-root names the repo root the data was built under, when that differs from this checkout's
current path (the data's records then carry the other path).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".json", ".csv", ".tsv", ".yaml", ".yml", ".txt"}
SKIP_DIRS = {".git", "__pycache__"}

T1 = "task1-affinity_regression"
T2 = "task2-ligand_generation"
# Folders git leaves behind on 2026-10-06, and where their contents go. Earlier entries win, so a
# subfolder is listed before its parent.
TASK_DIRS = [
    ("voxbind/model_zoo", "model_zoo"),
    ("voxbind/configs", f"{T2}/configs"),
    ("voxbind/notebook", f"{T2}/notebook"),
    ("voxbind/wandb", f"{T2}/wandb"),
    ("voxbind/.claude", f"{T2}/.claude"),
    ("voxbind/test/results", f"{T2}/test/results"),
    ("voxbind/test", f"{T1}/evals"),
    ("voxbind/analysis", f"{T1}/analysis"),
    ("dataset/analysis", f"{T1}/analysis"),
    ("voxbind/scripts/tools", f"{T1}/scripts/tools"),     # local tools seen so far are affinity evals
    ("voxbind/scripts/workflows", f"{T2}/scripts/workflows"),
    ("voxbind/log", "log"),
    ("voxbind/logs", "log"),
]
# Files git moved on 2026-10-06; only used to repair symlinks that pointed at them.
TASK_FILES = {
    **{f"voxbind/{f}": f"{T2}/{f}" for f in ("train.py", "train_ddp.py", "sample.py", "sample_from_file.py")},
    "dataset/01c_pdbbind_probe.py": f"{T1}/probe.py",
    "dataset/consolidate_probe_results.py": f"{T1}/consolidate_probe_results.py",
    **{f"voxbind/scripts/{f}": f"{T1}/scripts/{f}" for f in (
        "04_probe.sh", "96_eval_affinity_uncertainty.sh", "eval_cdg_encoder.sh", "watch_probe_ckpts.sh",
        "panel5.py", "tools/eval_affinity_uncertainty.py", "tools/uncertainty_descriptors.py",
        "workflows/pretrain_then_probe.sh")},
    **{f"voxbind/scripts/{f}": f"{T2}/scripts/{f}" for f in (
        "00_data_density_process.sh", "01_downstream_voxbind.sh", "02_sample.sh", "72_sample_8gpu.sh",
        "73_dock_baseline_protocol_79.sh", "73_evaluate_samples.sh", "74_dock_targetdiff_full79.sh",
        "85_fill_pose_eval_4runs.sh", "86_pose_eval_baselines.sh", "89_eval_crossdocked_jsd.sh",
        "95_eval_pocket_uncertainty.sh", "rebuild_moleval_env.sh", "rebuild_voxdock_env.sh", "mcp")},
    **{f"voxbind/test/{f}": f"{T2}/test/{f}" for f in (
        "repa_b1_linear.py", "repa_cka_profile.py", "test_fusion_modes.py", "v4_gpu_smoke.py")},
    **{f"voxbind/test/{f}": f"dataset/test/{f}" for f in (
        "smoke_otf_density_aug.py", "smoke_resample_reader.py", "test_resample_loader.py")},
    "voxbind/test/benchmark_vit_impl.py": "pretrain/test/benchmark_vit_impl.py",
}


def _walk_links(root: Path):
    """Every symlink under root, without following links into other trees."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except (PermissionError, FileNotFoundError):
            continue
        for e in entries:
            if e.is_symlink():
                yield Path(e.path)
            elif e.is_dir(follow_symlinks=False) and e.name not in SKIP_DIRS:
                stack.append(Path(e.path))


def _text_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [n for n in dirnames if n not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix in TEXT_SUFFIXES and not p.is_symlink():
                yield p


def make_relocate(moves: list[tuple[str, str]]):
    """Map an absolute path to where it is after `moves` [(old, new)]; unaffected paths pass through."""
    moves = sorted(moves, key=lambda m: len(m[0]), reverse=True)

    def relocate(path: str) -> str:
        for old, new in moves:
            if path == old or path.startswith(old + "/"):
                return new + path[len(old):]
        return path
    return relocate


def plan_links(relocate):
    """[(link after the move, old target, new target)] for links a move breaks.

    Planned before anything moves: each link is resolved from where it is now, so a relative link
    whose target moves with it is left alone, and both the link and its target are relocated.
    Links that change are written relative (absolute links break when a checkout moves).
    """
    changes = []
    for link in _walk_links(REPO):
        target = os.readlink(link)
        here = str(link)
        lexical = os.path.normpath(os.path.join(os.path.dirname(here), target))
        here2, lexical2 = relocate(here), relocate(lexical)
        if lexical2 == lexical and (here2 == here or os.path.isabs(target)):
            continue
        new = os.path.relpath(lexical2, os.path.dirname(here2))
        if new != target:
            changes.append((Path(here2), target, new))
    return changes


def plan_text(data_root: Path, old_prefixes: list[str]):
    """[(file, n_occurrences)] for text records that carry an old absolute prefix."""
    out = []
    if not data_root.is_dir():
        return out
    for p in _text_files(data_root):
        try:
            s = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        n = sum(s.count(pre) for pre in old_prefixes)
        if n:
            out.append((p, n))
    return out


def apply_links(links, stamp: str, tag: str) -> None:
    if not links:
        return
    manifest = REPO / f"migrate_layout_{tag}links_{stamp}.tsv"
    with manifest.open("w") as fh:
        for link, old, new in links:
            fh.write(f"{link.relative_to(REPO)}\t{old}\t{new}\n")
    for link, _old, new in links:
        link.unlink()
        os.symlink(new, link)
    print(f"  rewrote {len(links)} symlink(s); old targets in {manifest.name}")


def rewrite_texts(edits: dict[Path, str], stamp: str, tag: str) -> None:
    """Write new contents, keeping each file's mtime, after backing the originals up."""
    if not edits:
        return
    backup = REPO / f"migrate_layout_{tag}textpaths_{stamp}.tar.gz"
    with tarfile.open(backup, "w:gz") as tar:
        for p in edits:
            tar.add(p, arcname=str(p.relative_to(REPO)))
    for p, s in edits.items():
        st = p.stat()
        p.write_text(s)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    print(f"  rewrote {len(edits)} text record(s); originals in {backup.name}")


# ── 2026-10-01: the data folder ───────────────────────────────────────────────────────────────

def migrate_data(args, roots: list[str], stamp: str) -> int:
    old_data = REPO / "voxbind" / "dataset" / "data"
    new_dataset = REPO / "dataset"
    new_data = new_dataset / "data"
    old_prefixes = sorted({f"{r}/voxbind/dataset/" for r in roots}, key=len, reverse=True)
    print("2026-10-01 data move")

    relocate = make_relocate([(p.rstrip("/"), str(new_dataset)) for p in old_prefixes])
    links = plan_links(relocate)

    # 1. move the data folder
    if old_data.is_dir() and not old_data.is_symlink():
        if new_data.exists() and any(new_data.iterdir()):
            print(f"  STOP: both {old_data} and a non-empty {new_data} exist; merge them by hand.")
            return 1
        print(f"  move {old_data} -> {new_data}")
        if args.apply:
            if new_data.exists():
                new_data.rmdir()
            os.rename(old_data, new_data)
    else:
        print(f"  move: nothing to do ({old_data} is not a folder)")
    leftovers = [p for p in (REPO / "voxbind" / "dataset").glob("*")
                 if p.name not in ("__pycache__",)] if (REPO / "voxbind" / "dataset").is_dir() else []
    if leftovers and (args.apply or not old_data.is_dir()):
        print(f"  note: voxbind/dataset/ still holds {len(leftovers)} untracked item(s), e.g. "
              f"{leftovers[0].name}; review and remove them by hand")

    # After a dry-run "move", plan against where the data will be, i.e. still the old folder.
    data_root = new_data if (args.apply or not old_data.is_dir()) else old_data

    # 2. symlinks (planned against the layout before the move)
    print(f"  symlinks pointing into the old location: {len(links)}")
    for link, old, new in links[:3]:
        print(f"    {link.relative_to(REPO)}: {old} -> {new}")

    # 3. text records
    texts = plan_text(data_root, old_prefixes)
    print(f"  text records with an old absolute prefix: {len(texts)} file(s), "
          f"{sum(n for _, n in texts)} occurrence(s)")
    for p, n in texts[:3]:
        print(f"    {p.relative_to(REPO)} ({n})")

    if not args.apply:
        return 0
    apply_links(links, stamp, "")
    edits = {}
    for p, _ in texts:
        s = p.read_text()
        for pre in old_prefixes:
            s = s.replace(pre, f"{new_dataset}/")
        edits[p] = s
    rewrite_texts(edits, stamp, "")
    broken = [l for l, _, _ in links if not l.exists()]
    print(f"  check: {len(broken)} rewritten link(s) do not resolve")
    for l in broken[:5]:
        print(f"    {l.relative_to(REPO)} -> {os.readlink(l)}")
    return 1 if broken else 0


# ── 2026-10-06: the task folders ──────────────────────────────────────────────────────────────

def run_kind(run: Path) -> str:
    """'pretrain' for an encoder pretraining run, else 'generator'.

    voxbind/exps was the generator's run folder, so a folder without a pretraining config
    (evaluation work dirs, empty or partial runs) goes with the generator.
    """
    text = ""
    for name in ("cfg.yaml", "config.yaml", ".hydra/config.yaml"):
        p = run / name
        if p.is_file():
            text += p.read_text(errors="replace")
    if re.search(r"^(smooth_sigma|wjs):", text, re.M):
        return "generator"
    if re.search(r"^(mae|pretext\w*):", text, re.M):
        return "pretrain"
    return "generator"


def merge_plan(src: Path, dst: Path, moves: list, conflicts: list, caches: list) -> None:
    """Moves that put src at dst; where both are folders, merge them entry by entry."""
    if src.name == "__pycache__":
        caches.append(src)
        return
    if not dst.exists() and not dst.is_symlink():
        moves.append((src, dst))
        return
    if src.is_dir() and not src.is_symlink() and dst.is_dir() and not dst.is_symlink():
        for e in sorted(src.iterdir()):
            merge_plan(e, dst / e.name, moves, conflicts, caches)
        return
    conflicts.append((src, dst))


def _remove_empty(d: Path) -> None:
    """Remove d and its subfolders bottom-up when they hold nothing."""
    if not d.is_dir() or d.is_symlink():
        return
    for sub in d.iterdir():
        if sub.is_dir() and not sub.is_symlink():
            _remove_empty(sub)
    if not any(d.iterdir()):
        d.rmdir()


def generator_cfg_edits(runs: list[Path], pretrain_runs: set[str], generator_runs: set[str]) -> dict[Path, str]:
    """New text for generator-run configs whose run or encoder paths point at an old location."""
    def alt(names):
        return "|".join(re.escape(r) for r in sorted(names, key=len, reverse=True)) or r"(?!)"
    run_alt, gen_alt = alt(pretrain_runs), alt(generator_runs)
    rules = [
        (re.compile(rf"/voxbind/exps/(?=({run_alt})(/|$|\s|[\'\"]))", re.M), "/pretrain/exps/"),
        (re.compile(rf"/voxbind/exps/(?=({gen_alt})(/|$|\s|[\'\"]))", re.M), f"/{T2}/exps/"),
        (re.compile(r"/voxbind/model_zoo/"), "/model_zoo/"),
        # relative paths were written for the old working directory, voxbind/
        (re.compile(rf"(?<=[\s:='\"])exps/(?=({run_alt})/)"), "../pretrain/exps/"),
        (re.compile(r"(?<=[\s:='\"])model_zoo/"), "../model_zoo/"),
    ]
    edits = {}
    for run in runs:
        for name in ("cfg.yaml", "config.yaml", ".hydra/config.yaml", ".hydra/overrides.yaml"):
            p = run / name
            if not p.is_file() or p.is_symlink():
                continue
            s = new = p.read_text(errors="replace")
            for rx, repl in rules:
                new = rx.sub(repl, new)
            if new != s:
                edits[p] = new
    return edits


def migrate_tasks(args, roots: list[str], stamp: str) -> int:
    print("2026-10-06 task folders")
    moves, conflicts, caches = [], [], []

    # 1. split voxbind/exps by run
    old_exps = REPO / "voxbind" / "exps"
    kinds = {}
    if old_exps.is_dir() and not old_exps.is_symlink():
        for run in sorted(old_exps.iterdir()):
            kinds[run.name] = run_kind(run) if run.is_dir() else "generator"
            dst_root = REPO / ("pretrain" if kinds[run.name] == "pretrain" else T2) / "exps"
            merge_plan(run, dst_root / run.name, moves, conflicts, caches)
    if kinds:
        n_pre = sum(k == "pretrain" for k in kinds.values())
        print(f"  voxbind/exps: {n_pre} pretraining run(s) -> pretrain/exps, "
              f"{len(kinds) - n_pre} other -> {T2}/exps")

    # 2. what git left in the other old folders
    nested = {REPO / old for old, _ in TASK_DIRS}
    for old, new in TASK_DIRS:
        src = REPO / old
        if src.is_dir() and not src.is_symlink():
            n0 = len(moves)
            for e in sorted(src.iterdir()):
                if e not in nested:            # a listed subfolder is moved on its own
                    merge_plan(e, REPO / new / e.name, moves, conflicts, caches)
            if len(moves) > n0:
                print(f"  {old}: {len(moves) - n0} item(s) -> {new}")
    # compiled caches of modules that moved away (a cache folder is cleared only of those)
    stale_pyc = [f for d in (REPO / "voxbind" / "__pycache__", REPO / "voxbind" / "scripts" / "__pycache__")
                 if d.is_dir() for f in d.glob("*.pyc")
                 if not (d.parent / f"{f.name.split('.')[0]}.py").exists()]
    for src, dst in conflicts:
        print(f"  CONFLICT: {src.relative_to(REPO)} and {dst.relative_to(REPO)} both exist; left in place")
    keep = {"__init__.py", "constants.py", "metrics.py", "voxelizer.py", "bench_voxelizer.py",
            "run_bench_voxelizer.sh", "models", "utils", "splits", "scripts", "__pycache__", "exps",
            *[Path(o).parts[1] for o, _ in TASK_DIRS if o.startswith("voxbind/")]}
    if (REPO / "voxbind").is_dir():
        odd = sorted(p.name for p in (REPO / "voxbind").iterdir() if p.name not in keep)
        if odd:
            print(f"  note: voxbind/ also holds {', '.join(odd)}; not part of the move, review by hand")
    print(f"  moves planned: {len(moves)}, stale cache folders: {len(caches)}, "
          f"stale compiled modules: {len(stale_pyc)}")

    # 3. symlinks, planned against the layout before the move (git-moved code included)
    pairs = [(str(src), str(dst)) for src, dst in moves]
    for old, new in [*TASK_DIRS, *TASK_FILES.items()]:
        pairs += [(f"{r}/{old}", str(REPO / new)) for r in roots]
    for src, dst in moves:   # other roots' absolute links into moved entries
        pairs += [(f"{r}/{src.relative_to(REPO)}", str(dst)) for r in roots[1:]]
    links = plan_links(make_relocate(pairs))
    print(f"  symlinks pointing into a moved location: {len(links)}")
    for link, old, new in links[:3]:
        print(f"    {link.relative_to(REPO)}: {old} -> {new}")

    # 4. generator-run configs, keyed by where they will be
    pretrain_runs = {k for k, v in kinds.items() if v == "pretrain"}
    if (REPO / "pretrain" / "exps").is_dir():
        pretrain_runs |= {p.name for p in (REPO / "pretrain" / "exps").iterdir()}
    gen_runs = [src for src, dst in moves if dst.parent == REPO / T2 / "exps"]
    if (REPO / T2 / "exps").is_dir():
        gen_runs += [p for p in (REPO / T2 / "exps").iterdir() if p.is_dir()]
    relocate_moved = make_relocate(pairs)
    generator_runs = {k for k, v in kinds.items() if v != "pretrain"} | {p.name for p in gen_runs}
    edits = {Path(relocate_moved(str(p))): s
             for p, s in generator_cfg_edits(gen_runs, pretrain_runs, generator_runs).items()}
    print(f"  generator-run configs with an old encoder path: {len(edits)}")
    for p in list(edits)[:3]:
        print(f"    {p.relative_to(REPO)}")

    if not args.apply:
        return 0
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
    for d in caches:
        for f in d.glob("*.pyc"):
            f.unlink()
        _remove_empty(d)
    for f in stale_pyc:
        f.unlink()
    for old in ["voxbind/exps", *[o for o, _ in TASK_DIRS]]:
        _remove_empty(REPO / old)
    print(f"  moved {len(moves)} item(s), cleared {len(caches)} stale cache folder(s)")
    apply_links(links, stamp, "tasks_")
    rewrite_texts(edits, stamp, "tasks_")
    broken = [l for l, _, _ in links if not l.exists()]
    print(f"  check: {len(broken)} rewritten link(s) do not resolve")
    for l in broken[:5]:
        print(f"    {l.relative_to(REPO)} -> {os.readlink(l)}")
    return 1 if (broken or conflicts) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    ap.add_argument("--old-root", action="append", default=[],
                    help="another repo root the data was built under (repeatable)")
    args = ap.parse_args()

    roots = [str(REPO), *[str(Path(r)) for r in args.old_root]]
    stamp = time.strftime("%y%m%d_%H%M%S")
    print(f"[{'APPLY' if args.apply else 'DRY RUN'}] repo {REPO}")
    rc = migrate_data(args, roots, stamp)
    if rc:
        return rc
    rc = migrate_tasks(args, roots, stamp)
    if not args.apply:
        print("dry run: nothing changed. Re-run with --apply.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

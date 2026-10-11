#!/usr/bin/env python3
"""migrate_monolith.py — turn a legacy `config_train_*.yaml` into an experiment preset.

The monoliths each restate every axis in full. A preset instead selects one option per axis
and lists only what it changes, which is what makes the axes reusable. This finds the closest
option per axis, writes the residual as explicit overrides, and then proves the result by
calling `verify_parity.py`: a preset is only kept if the resolved config tree is identical.

    python configs/migrate_monolith.py --list                     # what is migratable, and why
    python configs/migrate_monolith.py --all --dry-run            # plan every monolith
    python configs/migrate_monolith.py --config config_train_... --name my_preset
    python configs/migrate_monolith.py --all --write              # write the ones that verify

Run from voxbind/. Writing is opt-in and never overwrites an existing preset without --force.

HOW THE MATCH WORKS. For each axis (dset, model, mae) every option is composed on top of
`pretrain.yaml` and its block compared leaf by leaf against the monolith's block. The option
with the fewest differing leaves wins, and those leaves become `axis.key: value` lines in the
preset. Top-level keys that differ from the base are emitted the same way. That is a search,
not an inference, so the parity check afterwards is what actually makes it trustworthy: if the
chosen option plus residual does not resolve to the monolith, the preset is reported as failing
and not written.

WHAT IT CANNOT DO. A monolith whose axis has no near option still produces a preset, just one
carrying a long residual; that verifies but buys little, so `--list` reports the residual size
and leaves the judgement to you. Run metadata (`exp_name`, `wandb_tags`) is carried over
verbatim because it names the run, and the throughput and inert-flag exemptions are whatever
`verify_parity.py` already applies -- this script does not relax the gate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
import code_defaults  # noqa: E402

# Which config tree is being migrated. Defaults are the pretrain tree; main() switches them
# with --config-dir / --base / --axes for the downstream (generator) tree.
CONFIG_DIR = TOOLS_DIR
BASE = "pretrain"
AXES = ["dset", "model", "mae"]


class SkipConfig(Exception):
    """Not migratable against the pretrain base; the message says why."""


# Top-level keys a preset never carries. Every other differing top-level key is carried over
# (an allow-list missed keys like `gradmag_on_gpu`).
SKIP_TOP = {"dset", "model", "mae", "vox", "wjs", "hydra", "exp_name", "wandb_tags", "exp_dir",
            "output_dir", "resume", "resume_epoch", "amp", "compile", "optimizer", "ema",
            "channels_last", "num_workers", "prefetch_factor", "wandb_entity"}
# Pre-P1b monoliths put these at the top level; the trainer and verify_parity both fold them
# under `model`, so they are matched as part of the model axis, not emitted top-level.
LEGACY_MODEL_KEYS = ("input_mode", "with_gradmag")


def options(axis: str) -> list:
    """Selectable options for an axis: files in the group dir, minus the _-prefixed includes."""
    return sorted(p.stem for p in (CONFIG_DIR / axis).glob("*.yaml")
                  if not p.stem.startswith("_"))


@contextlib.contextmanager
def _composable(config_name):
    """Yield a config name Hydra can compose with defaults resolved from the root.

    An archived monolith (`archive/config_train_X`) lists its defaults as `- vox: default`,
    which Hydra resolves relative to the config's own directory -- `archive/vox/default`, which
    does not exist. Rather than rewrite 70 archived records, compose a temporary copy placed at
    the config root, where those defaults resolve as they did before archiving.
    """
    if "/" not in config_name:
        yield config_name
        return
    src = os.path.join(str(CONFIG_DIR), config_name + ".yaml")
    tmp_name = "_tmp_verify_" + config_name.replace("/", "__")
    tmp = os.path.join(str(CONFIG_DIR), tmp_name + ".yaml")
    shutil.copyfile(src, tmp)
    try:
        yield tmp_name
    finally:
        os.remove(tmp)


def resolve(config_name: str, overrides=None) -> dict:
    with _composable(config_name) as name, \
            initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=name, overrides=overrides or [])
    d = OmegaConf.to_container(cfg, resolve=False)
    d.pop("hydra", None)
    return d


def flat(block, prefix="") -> dict:
    out = {}
    if isinstance(block, dict):
        for k, v in block.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = block
    return out


def _canonical_flat(group: str, items: dict) -> dict:
    """Aliased values under their canonical names (model.fusion `default` -> `proj_sum`)."""
    return {k: code_defaults.canonical(group, k, v) for k, v in items.items()}


def _restore_implicit(group: str, have: dict, want: dict) -> dict:
    """Keys the option sets but the monolith left out.

    Leaving a key out does not mean "unset": the run used the code's fallback. So wherever the
    option's value differs from that fallback, the preset must set the fallback back, or it
    silently trains a different recipe -- inverse-frequency weighting where the monolith ran
    uniform, say. A key with no known fallback cannot be restored faithfully; it is left out,
    and the parity check then refuses the preset, which is the correct outcome.
    """
    out = {}
    for k, v in have.items():
        if k in want or "." in k:
            continue
        known, default = code_defaults.default_for(group, k)
        if known and v != default:
            out[k] = default
    return out


def best_option(axis: str, target: dict, base_overrides: list) -> tuple:
    """(option, residual) — the option whose block differs from `target` in the fewest leaves.

    The residual has two parts: values the monolith sets differently, and keys the monolith
    left implicit whose code fallback differs from the option's value.
    """
    want = _canonical_flat(axis, flat(target or {}))
    best, best_resid = None, None
    for opt in options(axis):
        try:
            have = _canonical_flat(
                axis, flat(resolve(BASE, [*base_overrides, f"{axis}={opt}"]).get(axis, {})))
        except Exception:
            continue
        resid = {k: v for k, v in want.items() if k not in have or have[k] != v}
        resid.update(_restore_implicit(axis, have, want))
        if best_resid is None or len(resid) < len(best_resid):
            best, best_resid = opt, resid
    return best, (best_resid or {})


def is_downstream(cfg: dict) -> bool:
    """True for a generator/downstream config, which composes on `config_train.yaml`.

    Those carry a U-Net denoiser (`ch_mults`) or a density `fusion` mode, neither of which the
    pretrain base has. Matching them against pretrain's axes produces a preset that may pass
    parity while being meaningless, so they are skipped: migrating them needs a downstream
    preset group that does not exist yet.
    """
    model = cfg.get("model", {}) or {}
    return "ch_mults" in model or "fusion" in model


def plan(config_name: str) -> dict:
    """Choose an option per axis and collect the residual, without writing anything."""
    old = resolve(config_name)
    if BASE == "pretrain" and is_downstream(old):
        raise SkipConfig("downstream/generator config — composes on config_train.yaml, "
                         "not pretrain.yaml; needs a downstream preset group")
    model_block = old.setdefault("model", {}) or {}
    for k in LEGACY_MODEL_KEYS:
        if k in old:
            model_block.setdefault(k, old.pop(k))
    old["model"] = model_block
    base_overrides = ["exp_name=_plan"]
    chosen, residual = {}, {}
    for axis in AXES:
        opt, resid = best_option(axis, old.get(axis, {}), base_overrides)
        chosen[axis] = opt
        residual[axis] = resid

    base = resolve(BASE, base_overrides)
    for k in LEGACY_MODEL_KEYS:                      # base carries these under model now
        base.pop(k, None)
    top = {k: v for k, v in old.items()
           if k not in SKIP_TOP and k not in LEGACY_MODEL_KEYS
           and (k not in base or base[k] != v)}
    # A top-level key the monolith omits ran at the code fallback; the base may now carry a
    # different value (the base moved to accum_steps 8, for one), so set the fallback back.
    for k, v in base.items():
        if k in SKIP_TOP or k in LEGACY_MODEL_KEYS or k in old:
            continue
        known, default = code_defaults.default_for("", k)
        if known and v != default:
            top[k] = default
    return {"config": config_name, "chosen": chosen, "residual": residual, "top": top,
            "exp_name": old.get("exp_name"), "wandb_tags": old.get("wandb_tags"),
            "n_residual": sum(len(v) for v in residual.values()) + len(top)}


def _unflatten(flat_items: dict) -> dict:
    """{"density_vit.depth": 12} -> {"density_vit": {"depth": 12}}."""
    out: dict = {}
    for dotted, value in flat_items.items():
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return out


def render(p: dict, name: str) -> str:
    """The preset file. Overrides are emitted as real nested YAML -- a residual key such as
    `model.density_vit.depth` becomes `model: {density_vit: {depth: ...}}`, which Hydra merges
    into the selected option. Correctness is not assumed: the caller runs verify_parity.py on
    the written file and deletes it if the resolved tree differs from the monolith."""
    import yaml

    header = [f"# @package _global_",
              f"# Migrated from {p['config'].removeprefix('archive/')}.yaml; verify_parity.py: resolves identically.",
              "defaults:"]
    header += [f"  - override /{axis}: {p['chosen'][axis]}" for axis in AXES]
    body: dict = {}
    if p["exp_name"]:
        body["exp_name"] = p["exp_name"]
    if p["wandb_tags"]:
        body["wandb_tags"] = p["wandb_tags"]
    body.update(p["top"])
    for axis in AXES:
        if p["residual"][axis]:
            body[axis] = _unflatten(p["residual"][axis])
    text = yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=100)
    return "\n".join(header) + "\n\n" + text


def verify(config_name: str, preset: str) -> tuple:
    r = subprocess.run([sys.executable, str(TOOLS_DIR / "verify_parity.py"),
                        "--config-dir", str(CONFIG_DIR), "--base", BASE,
                        config_name, preset],
                       capture_output=True, text=True, cwd=CONFIG_DIR.parent)
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()


def monoliths() -> list:
    """Monoliths still at the top level, plus the archived ones (as `archive/<name>`).

    After the 2026-10-01 migration every pretrain monolith sits in archive/; Hydra resolves a
    config name with a subdirectory, so they can still be re-verified from there.
    """
    top = [p.stem for p in CONFIG_DIR.glob("config_train_*.yaml")]
    arch = [f"archive/{p.stem}" for p in (CONFIG_DIR / "archive").glob("config_train_*.yaml")]
    return sorted(top + arch)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="one config_train_* name (without .yaml)")
    g.add_argument("--all", action="store_true")
    g.add_argument("--list", action="store_true", help="plan every monolith, write nothing")
    ap.add_argument("--name", help="preset name (default: derived from the config name)")
    ap.add_argument("--write", action="store_true", help="write the presets that verify")
    ap.add_argument("--dry-run", action="store_true", help="print the preset, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing preset")
    ap.add_argument("--config-dir", type=Path, default=None,
                    help="config tree to migrate (default: the pretrain tree beside this file)")
    ap.add_argument("--base", default=None, help="base config name (default: pretrain)")
    ap.add_argument("--axes", default=None,
                    help="comma-separated axis groups (default: dset,model,mae)")
    args = ap.parse_args()

    global CONFIG_DIR, BASE, AXES
    if args.config_dir:
        CONFIG_DIR = args.config_dir.resolve()
    if args.base:
        BASE = args.base
    if args.axes:
        AXES[:] = [a.strip() for a in args.axes.split(",") if a.strip()]
    (CONFIG_DIR / "experiment").mkdir(exist_ok=True)

    names = monoliths() if (args.all or args.list) else [args.config]
    rows = []
    for config_name in names:
        try:
            p = plan(config_name)
        except SkipConfig as e:
            rows.append((config_name, None, f"skipped: {e}"))
            continue
        except Exception as e:                                  # noqa: BLE001
            rows.append((config_name, None, f"plan failed: {type(e).__name__}: {e}"))
            continue
        preset = args.name if (args.config and args.name) else \
            config_name.split("/")[-1].replace("config_train_", "")
        text = render(p, preset)

        if args.list:
            rows.append((config_name, p, f"residual={p['n_residual']} " + " ".join(
                f"{a}={p['chosen'][a]}" for a in AXES)))
            continue

        target = CONFIG_DIR / "experiment" / f"{preset}.yaml"
        if args.dry_run:
            print(f"\n===== {config_name} -> experiment/{preset}.yaml "
                  f"(residual {p['n_residual']}) =====\n{text}")
            rows.append((config_name, p, "dry-run"))
            continue
        if target.exists() and not args.force:
            rows.append((config_name, p, f"exists, skipped: experiment/{preset}.yaml"))
            continue
        target.write_text(text)
        ok, out = verify(config_name, preset)
        if ok:
            rows.append((config_name, p, f"PASS -> experiment/{preset}.yaml "
                                         f"(residual {p['n_residual']})"))
        else:
            target.unlink()
            head = next((l for l in out if l.startswith("FAIL")), out[0] if out else "no output")
            rows.append((config_name, p, f"FAIL, not written: {head[:90]}"))

    width = max(len(r[0]) for r in rows) if rows else 10
    for config_name, _p, note in rows:
        print(f"  {config_name:<{width}}  {note}")
    if not args.list and not args.dry_run:
        passed = sum(1 for r in rows if r[2].startswith("PASS"))
        print(f"\n{passed}/{len(rows)} verified and written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

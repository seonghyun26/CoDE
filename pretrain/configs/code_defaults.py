"""code_defaults.py — the value a config key takes when a config does not set it.

A terse monolithic config that omits a key does not run "without" it: it runs with whatever
fallback the code reads, `cfg.mae.get("channel_weighting", "uniform")` and so on. To compare a
monolith against a preset fairly, or to migrate one faithfully, you need that fallback. Reading
it off the source is the only reliable way, so this parses the trainer and the dataset factory
with `ast` and collects every `cfg.<group>.get("key", <literal>)` it finds.

Two rules keep the table honest:
  * Only LITERAL defaults are collected (numbers, strings, bools, None, and lists/tuples of
    those). A computed default cannot be known statically, so that key is left out and gets
    compared strictly.
  * A key read with DIFFERENT defaults in different places is CONFLICTING and left out too.
    Its "default" depends on which code path ran, so treating absence as one value would be a
    guess.

Keys read as attributes (`cfg.mae.mask_ratio`) have no fallback at all -- a config that omits
them crashes -- so they never appear here and are always compared strictly.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

PRETRAIN = Path(__file__).resolve().parents[1]      # VoxBind/pretrain
REPO = PRETRAIN.parent                               # VoxBind

# Files whose `.get` fallbacks define behaviour for pretraining configs.
SOURCES = (
    PRETRAIN / "train_density.py",
    PRETRAIN / "train_common.py",
    REPO / "dataset" / "__init__.py",
    # downstream (generator) code, for configs composed on task2-ligand_generation/configs/config_train.yaml
    REPO / "voxbind" / "train_ddp.py",
    REPO / "voxbind" / "train.py",
    REPO / "voxbind" / "models" / "__init__.py",
)
GROUPS = ("mae", "dset", "model", "vox", "wjs")

# Values the code renames on load (legacy -> canonical), read from the dict literal in the source,
# so `fusion: default` and `fusion: proj_sum` compare equal.
_ALIAS_SOURCES = {
    ("model", "fusion"): (REPO / "voxbind" / "models" / "voxbind.py", "FUSION_ALIASES"),
}
ALIASED = tuple(_ALIAS_SOURCES)


@lru_cache(maxsize=None)
def aliases(group: str, key: str) -> dict:
    """{legacy name: canonical name} for an aliased key; {} for any other key."""
    if (group, key) not in _ALIAS_SOURCES:
        return {}
    path, name = _ALIAS_SOURCES[(group, key)]
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            return dict(ast.literal_eval(node.value))
    raise LookupError(f"{name} is not a module-level literal in {path}")


def canonical(group: str, key: str, value):
    """`value` under its canonical name, e.g. model.fusion `default` -> `proj_sum`."""
    if isinstance(value, str):
        return aliases(group, key).get(value, value)
    return value


def _literal(node):
    try:
        return True, ast.literal_eval(node)
    except Exception:                                   # noqa: BLE001
        return False, None


def _group_of(func_value) -> str | None:
    """`cfg.mae.get` -> "mae"; `cfg.get` -> "" (top level); anything else -> None."""
    if isinstance(func_value, ast.Attribute) and isinstance(func_value.value, ast.Name):
        if func_value.value.id == "cfg" and func_value.attr in GROUPS:
            return func_value.attr
    if isinstance(func_value, ast.Name) and func_value.id == "cfg":
        return ""
    return None


@lru_cache(maxsize=1)
def table() -> dict:
    """{(group, key): default} for every unambiguous literal fallback in SOURCES.

    group is "" for top-level keys. Conflicting keys are excluded.
    """
    seen: dict = {}
    conflicted: set = set()
    for path in SOURCES:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and len(node.args) == 2):
                continue
            group = _group_of(node.func.value)
            if group is None:
                continue
            ok_key, key = _literal(node.args[0])
            ok_val, val = _literal(node.args[1])
            if not (ok_key and isinstance(key, str) and ok_val):
                continue
            val = canonical(group, key, val)       # a legacy-named fallback is the same default
            slot = (group, key)
            if slot in seen and seen[slot] != val:
                conflicted.add(slot)
            seen.setdefault(slot, val)
    return {k: v for k, v in seen.items() if k not in conflicted}


def default_for(group: str, key: str):
    """(known, value). `known` is False when there is no single literal fallback."""
    t = table()
    return ((group, key) in t), t.get((group, key))


if __name__ == "__main__":
    t = table()
    print(f"{len(t)} unambiguous literal defaults")
    for (group, key), val in sorted(t.items()):
        print(f"  {group or '<top>'}.{key} = {val!r}")

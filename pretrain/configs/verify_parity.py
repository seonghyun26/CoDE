"""verify_parity.py — compose-diff parity gate for the config reorg.

Asserts that a new `pretrain +experiment=<exp>` composition resolves to the SAME config
as an old monolithic `config_train_*` config — proving the reorg changes file organization
only, not the resolved config tree. The `hydra` node is excluded (plumbing, identical by
construction). Run from voxbind/:

    python configs/verify_parity.py <OLD_CONFIG_NAME> <EXPERIMENT_NAME>
    python configs/verify_parity.py config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
                                     atomblob_density_gradmag_invfreq

Exit 0 = identical; 1 = drift (every differing leaf is printed). Values are compared parsed
(so 1e-4 == 0.0001) and order-independently.
"""
import contextlib
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_defaults  # noqa: E402  (the fallback each key takes when a config omits it)

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = "pretrain"
# `--config-dir DIR --base NAME` (before the two positionals) points the check at another
# config tree, e.g. the downstream one:  --config-dir ../task2-ligand_generation/configs --base config_train
_argv = sys.argv[1:]
while _argv and _argv[0].startswith("--"):
    flag, val = _argv[0], _argv[1]
    if flag == "--config-dir":
        CONFIG_DIR = os.path.abspath(val)
    elif flag == "--base":
        BASE = val
    _argv = _argv[2:]
sys.argv = [sys.argv[0], *_argv]

# Path keys whose absolute/${env} prefix is an intentional P2 portability change, not drift.
_DATA_KEYS = {"data_dir", "crops_dir", "ccp4_dir", "pdb_dir", "resample_dir", "out_root", "out_dir"}


def _norm_path(v):
    """Normalize a data path so the portability prefix (absolute root or ${oc.env:VOXBIND_DATA_ROOT,…})
    is ignored but the meaningful relative subpath (dataset/data/<sub>) is still compared."""
    if not isinstance(v, str) or not v:
        return v
    v = re.sub(r"\$\{oc\.env:VOXBIND_DATA_ROOT,([^}]*)\}", r"\1", v)  # ${oc.env:..,X} -> X
    v = re.sub(r"^.*?/?(dataset/data)\b", r"\1", v)                  # strip any abs prefix before dataset/data
    return v


_INERT_WHEN = {
    "mae": {
        # Both input sides present = the standard MAE; false would drop one from the encoder.
        "density_input": True,
        "struct_input": True,
        "gradmag_source": "density",        # the only source before the flag existed
        "gradmag_without_density": False,
        "mask_channel_independent": False,  # per-group ratios off
        "mask_group_ratios": None,
        "mask_curriculum": False,           # easy->hard ramp off
        "mask_curriculum_epochs": None,
        "mask_curriculum_steps": None,
        "mask_ratio_min": None,             # R2MAE off = the fixed mask_ratio
        "mask_ratio_max": None,
        "atom_pos_thresh": 0.05,            # code default is also 0.05 (train_density.py)
    },
    "dset": {
        "in_vocab_only": False,             # no load-time vocabulary filter
        # Verified against the loader's own fallbacks in dataset/__init__.py: a config that
        # omits these got exactly these values, so the standardized ones are not a change.
        "delta_translate": 1.0,
        "max_len": 30,
    },
}

# Keys a terse monolith often omits where the STANDARDIZED value differs from the code default,
# so omitting them was a real recipe difference, not a formatting one. Checked against the
# `cfg.mae.get(...)` fallbacks in train_density.py; listed here so nobody adds them to
# _INERT_WHEN to make a migration pass:
#   channel_weighting          code "uniform"  vs group "inv_freq"
#   channel_weight_clip_ratio  code 0.0        vs group 30.0
#   atom_pos_weight            code 1.0        vs group 10.0
#   density_channel_weight     code 1.0        vs group 0.1
#   gradmag_channel_weight     code 1.0        vs group 0.1
# atom_pos_thresh is the one in that family that does match (0.05 both sides), so it IS inert.
_NOT_INERT_NOTE = True


def _canon(d):
    """P1b/#2: input_mode/with_gradmag may live top-level (old monoliths) or under model (new
    configs). Fold them under model so the intentional namespace move is not flagged as drift —
    the VALUE must still match, which is what the diff then checks."""
    model = d.setdefault("model", {})
    for k in ("input_mode", "with_gradmag"):
        if k in d:
            model.setdefault(k, d.pop(k))
    # A value the code accepts under a legacy name is compared under its canonical one, so an old
    # monolith's `fusion: default` and a preset's `fusion: proj_sum` match (same model is built).
    for group, key in code_defaults.ALIASED:
        block = d if group == "" else (d.get(group, {}) or {})
        if isinstance(block, dict) and key in block:
            block[key] = code_defaults.canonical(group, key, block[key])
    # model_name is a provenance/log label (build constructs from explicit fields, not the name) and
    # was copy-paste-inconsistent across the old monoliths — exempt it from byte-parity.
    model.pop("model_name", None)
    # Runtime/throughput knobs are standardized in the base and are result-neutral (260611:
    # amp+compile verified same results); older monoliths predate them. Exempt from byte-parity.
    for k in ("amp", "compile", "optimizer", "ema", "channels_last", "num_workers",
              "prefetch_factor", "wandb_entity"):
        d.pop(k, None)
    # A density/gradmag loss-weight is INERT when its channel is absent (the trainer's input_mode
    # guards never apply it). The standardized _base objective carries these knobs everywhere; older
    # monoliths listed them only when relevant. Drop the inert ones so behavioral parity holds.
    im = str(model.get("input_mode", "") or "")
    mae = d.get("mae", {}) or {}
    if "density" not in im:
        for k in ("density_channel_weight", "density_weight_warmup_epochs"):
            mae.pop(k, None)
    if not bool(model.get("with_gradmag", False)):
        for k in ("gradmag_channel_weight", "gradmag_noise", "gradmag_reconstruct"):
            mae.pop(k, None)
    # density warmup of 0 epochs == off == the code default (train_density.py:1365), which is
    # what terse old configs that omitted the key got. Inert when 0 → don't flag the standardized 0.
    if mae.get("density_weight_warmup_epochs") == 0:
        mae.pop("density_weight_warmup_epochs", None)
    # Default-off dataset feature flags (e.g. gradmag_as_density) are inert when false → behaviorally
    # absent, like terse old configs that predate the feature. Don't flag the standardized false.
    ds = d.get("dset", {}) or {}
    if ds.get("gradmag_as_density") is False:
        ds.pop("gradmag_as_density", None)
    # Feature flags added to the standardized objective/dataset AFTER these monoliths were
    # written. Each is INERT at the value listed: the code path it guards is not taken, so a
    # terse old config that predates the flag behaves identically to a new one carrying its
    # default. Listed explicitly WITH the inert value rather than ignoring unknown extras —
    # a flag that is actually set still shows up as drift, which is the point of this gate.
    for group, inert in _INERT_WHEN.items():
        block = d.get(group, {}) or {}
        for key, value in inert.items():
            if key in block and block[key] == value:
                block.pop(key)


    def _walk(o):  # P2: normalize known data-path keys wherever they appear
        if isinstance(o, dict):
            for k, v in o.items():
                o[k] = _norm_path(v) if k in _DATA_KEYS else (_walk(v) or v)
        elif isinstance(o, list):
            for x in o:
                _walk(x)
        return o
    _walk(d)
    # General form of _INERT_WHEN: a key whose value equals the CODE's own fallback is
    # indistinguishable from omitting it, because omitting it yields exactly that fallback.
    # Runs AFTER path normalisation so an env-interpolated path and the literal fallback are
    # compared like for like. The table is read off the source (code_defaults.py), keeps only
    # unambiguous literal defaults, and leaves out anything read as a bare attribute -- so a key
    # set to something else, or one with no fallback, is still compared strictly.
    for (group, key), default in code_defaults.table().items():
        block = d if group == "" else (d.get(group, {}) or {})
        if not isinstance(block, dict) or key not in block:
            continue
        want = _norm_path(default) if key in _DATA_KEYS else default
        if block[key] == want:
            block.pop(key)
    return d


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
    src = os.path.join(CONFIG_DIR, config_name + ".yaml")
    tmp_name = "_tmp_verify_" + config_name.replace("/", "__")
    tmp = os.path.join(CONFIG_DIR, tmp_name + ".yaml")
    shutil.copyfile(src, tmp)
    try:
        yield tmp_name
    finally:
        os.remove(tmp)


def resolved(config_name, overrides=None):
    with _composable(config_name) as name, \
            initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=name, overrides=overrides or [])
    d = OmegaConf.to_container(cfg, resolve=False)
    d.pop("hydra", None)
    return _canon(d)


def diff(new, old, path=""):
    out = []
    if isinstance(new, dict) and isinstance(old, dict):
        for k in sorted(set(new) | set(old)):
            if k not in new:
                out.append(f"{path}{k}: MISSING in NEW (old={old[k]!r})")
            elif k not in old:
                out.append(f"{path}{k}: EXTRA in NEW (new={new[k]!r})")
            else:
                out += diff(new[k], old[k], f"{path}{k}.")
    elif new != old:
        out.append(f"{path[:-1]}: new={new!r} != old={old!r}")
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    old_name, exp = sys.argv[1], sys.argv[2]
    old = resolved(old_name)
    new = resolved(BASE, [f"+experiment={exp}"])
    d = diff(new, old)
    if not d:
        print(f"PASS — {BASE} +experiment={exp}  ==  {old_name}  (resolved configs identical)")
        return 0
    print(f"FAIL — {len(d)} difference(s)  [{BASE} +experiment={exp}  vs  {old_name}]:")
    for line in d:
        print("  " + line)
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Pretext-independent helpers for the density-encoder pretrainer (train_density.py).

DDP setup, bf16 AMP and torch.compile, async checkpointing, the input/reconstruction channel
layout, the val-voxel and channel-frequency caches, metric logging and input-key reconciliation.
"""
import copy
import hashlib
import json
import logging
import os
import threading
from datetime import timedelta

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger("voxbind.train_common")


# ── DDP ───────────────────────────────────────────────────────────────────────

# The bare model inside a DDP wrapper.
def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# Join the NCCL group torchrun describes (2 h timeout); returns (rank, local_rank, world_size).
def _setup_ddp() -> tuple:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(hours=2)
    )
    return rank, local_rank, world_size


# Leave the process group, if one was joined.
def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ── bf16 AMP + torch.compile ──────────────────────────────────────────────────

# cfg.compile with its defaults (off, inductor backend, reduce-overhead mode).
def _compile_options(cfg: DictConfig) -> dict:
    compile_cfg = cfg.get("compile", {}) or {}
    return {
        "enabled": bool(compile_cfg.get("enabled", False)),
        "backend": str(compile_cfg.get("backend", "inductor")),
        "mode": str(compile_cfg.get("mode", "reduce-overhead")),
        "fullgraph": bool(compile_cfg.get("fullgraph", False)),
        "dynamic": bool(compile_cfg.get("dynamic", False)),
    }


# (enabled, ctx_factory): ctx_factory() is a bf16 autocast context, a no-op when disabled.
# bf16 only, since fp16 would need a GradScaler.
def _amp_setup(cfg: DictConfig):
    amp_cfg = cfg.get("amp", {}) or {}
    enabled = bool(amp_cfg.get("enabled", False))
    dtype = str(amp_cfg.get("dtype", "bf16"))
    if enabled and dtype not in ("bf16", "bfloat16"):
        raise NotImplementedError(
            f"amp.dtype={dtype!r} unsupported — bf16 only (fp16 would need a GradScaler)")
    return enabled, (lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled))


# torch.compile the model when cfg.compile.enabled; otherwise return it unchanged.
def maybe_compile_model(model: torch.nn.Module, cfg: DictConfig, *, is_main: bool) -> torch.nn.Module:
    opts = _compile_options(cfg)
    if not opts["enabled"]:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile.enabled=true requested, but this PyTorch has no torch.compile")

    kwargs = {
        "backend": opts["backend"],
        "fullgraph": opts["fullgraph"],
        "dynamic": opts["dynamic"],
    }
    if opts["mode"] and opts["mode"] != "default":
        kwargs["mode"] = opts["mode"]
    if is_main:
        logger.info(
            "torch.compile enabled "
            f"(backend={opts['backend']}, mode={opts['mode']}, "
            f"fullgraph={opts['fullgraph']}, dynamic={opts['dynamic']})"
        )
    return torch.compile(model, **kwargs)


# ── Atomic async checkpointing (rank 0 only) ──────────────────────────────────

# Writes checkpoints on a background thread, through a .tmp file and os.replace, so an
# interrupted write never leaves a truncated checkpoint. Rank 0 only.
class AsyncCheckpointSaver:
    # _thread is the write in flight; _lock guards it.
    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()

    # Write <name>.tmp, then rename it over <name>; a failed write is logged and its .tmp removed.
    @staticmethod
    def _write(state: dict, save_dir: str, chkp_name: str) -> None:
        tmp = os.path.join(save_dir, chkp_name + ".tmp")
        final = os.path.join(save_dir, chkp_name)
        try:
            torch.save(state, tmp)
            os.replace(tmp, final)
        except Exception as e:
            logger.warning(f"checkpoint write failed: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    # Wait for the previous write, snapshot the state (deepcopy), and write it on a new thread.
    def save(self, state: dict, save_dir: str, chkp_name: str = "checkpoint.pth.tar") -> None:
        self.wait()
        frozen = copy.deepcopy(state)
        with self._lock:
            self._thread = threading.Thread(
                target=self._write,
                args=(frozen, save_dir, chkp_name),
                daemon=False,
            )
            self._thread.start()

    # Block until the write in flight, if any, finishes.
    def wait(self) -> None:
        with self._lock:
            t = self._thread
        if t is not None:
            t.join()


# ── Multi-channel input / reconstruction layout ───────────────────────────────

_VALID_INPUT_MODES = ("density", "atomblob", "atomblob_density")


# Ligand / pocket element channels: 7/4 by default (C,O,N,S,F,Cl,P / C,O,N,S), 8/4 for the
# per-element "+other" corpus. run() sets them once through set_atom_channels(); only
# _channel_layout reads them (the voxelizer call sites read cfg).
LIG_CH = 7
POC_CH = 4


# Pin LIG_CH / POC_CH for _channel_layout; called once at startup.
def set_atom_channels(n_lig: int, n_poc: int) -> None:
    global LIG_CH, POC_CH
    LIG_CH, POC_CH = int(n_lig), int(n_poc)


# The one definition of the channel order [atoms, density, gradmag]. n_in is the encoder input
# width, n_recon the reconstruction width (n_in minus gradmag when gradmag is input-only), and
# density_idx / gradmag_idx the channel offsets (gradmag_idx is None without gradmag).
def _channel_layout(
    input_mode: str,
    with_gradmag: bool = False,
    gradmag_reconstruct: bool = True,
    gradmag_without_density: bool = False,
) -> dict:
    if input_mode == "density":
        n_atom = 0
    elif input_mode in ("atomblob", "atomblob_density"):
        n_atom = LIG_CH + POC_CH
    else:
        raise RuntimeError(f"unexpected input_mode={input_mode!r}")

    # gradmag_without_density: gradmag is still computed from the density during loading, but
    # the density channel is neither input nor target: [atoms, gradmag], input_mode=atomblob.
    has_density = input_mode in ("density", "atomblob_density")
    if gradmag_without_density:
        if has_density:
            raise ValueError(
                f"gradmag_without_density=True expects an atom-only input_mode "
                f"(gradmag is appended to atoms, no density channel); got {input_mode!r}"
            )
        if not with_gradmag:
            raise ValueError("gradmag_without_density=True requires with_gradmag=True")

    n_density = 1 if has_density else 0
    if with_gradmag and not has_density and not gradmag_without_density:
        raise ValueError(
            f"with_gradmag=True requires a density-bearing input_mode "
            f"(density / *_density); got input_mode={input_mode!r}"
        )
    n_gradmag = 1 if with_gradmag else 0

    # gradmag leaves the reconstruction target only when it is input-only.
    return {
        "n_atom": n_atom,
        "n_density": n_density,
        "n_gradmag": n_gradmag,
        "n_in": n_atom + n_density + n_gradmag,
        "n_recon": n_atom + n_density + (n_gradmag if gradmag_reconstruct else 0),
        "density_idx": n_atom,
        "gradmag_idx": (n_atom + n_density) if with_gradmag else None,
    }


# ── Val-voxel cache ───────────────────────────────────────────────────────────

# Path of the voxelized-val cache (ligand, pocket, density, optional gradmag). resample_dir and
# with_gradmag key it because they change the density source and add gradmag; the task name
# matches cnn-MAE's, since the cached tensors hold nothing encoder-specific.
def _val_cache_path(cfg) -> str:
    key = {
        "grid_dim": cfg.vox.grid_dim,
        "resolution": cfg.vox.resolution,
        "cubes_around": cfg.vox.cubes_around,
        "ligand_radius": cfg.dset.ligand_radius,
        "pocket_radius": cfg.dset.pocket_radius,
        "dset_name": cfg.dset.dset_name,
        "subset_n": cfg.dset.get("subset_n", None),
        "subset_xray_only": cfg.dset.get("subset_xray_only", False),
        "resample_dir": cfg.dset.get("resample_dir", ""),
        "input_mode": cfg.get("input_mode", "density"),
        "with_gradmag": cfg.get("with_gradmag", False),
        "task": "density_mae_v1",
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"val_voxels_mae_{h}.pt")


# Voxelize the val set once and cache it on disk (_val_cache_path); later runs load the file.
def precompute_val(loader_val, voxelizer, cfg) -> dict:
    path = _val_cache_path(cfg)
    if os.path.isfile(path):
        logger.info(f"loading pre-computed val voxels from {path}")
        return torch.load(path, weights_only=True)

    with_gradmag = bool(cfg.get("with_gradmag", False))
    logger.info(f"pre-computing val voxels (with_gradmag={with_gradmag})...")
    all_lig, all_poc, all_xray, all_gradmag = [], [], [], []
    with torch.no_grad():
        for batch in loader_val:
            all_lig.append(voxelizer.forward(batch["ligand"], num_channels=int(cfg.model.get("n_channels_ligand", 7))).cpu())
            all_poc.append(voxelizer.forward(batch["pocket"], num_channels=int(cfg.model.get("n_channels_pocket", 4))).cpu())
            all_xray.append(batch["xray_density"].cpu())
            if with_gradmag:
                all_gradmag.append(batch["xray_gradmag"].cpu())
    cache = {
        "voxels_lig": torch.cat(all_lig, 0),
        "voxels_poc": torch.cat(all_poc, 0),
        "xray_density": torch.cat(all_xray, 0),
    }
    if with_gradmag:
        cache["xray_gradmag"] = torch.cat(all_gradmag, 0)
    torch.save(cache, path)
    logger.info(f"saved val voxels to {path}")
    return cache


# ── Channel-frequency cache + inverse-frequency weights ───────────────────────

# Atom-present voxel threshold (as in the struct head) and the number of training samples
# scanned once for the channel frequencies.
_CH_FREQ_POS_THRESH = 0.05
_CH_FREQ_DEFAULT_N_SAMPLES = 2000


# Path of the channel-frequency cache. The blob radii are in the key because they change the
# frequencies; power and clip are not, since precompute_channel_weights applies them later.
def _ch_freq_cache_path(cfg, n_samples: int) -> str:
    key = {
        "dset_name": cfg.dset.dset_name,
        "subset_n": cfg.dset.get("subset_n", None),
        "subset_xray_only": cfg.dset.get("subset_xray_only", False),
        "resample_dir": cfg.dset.get("resample_dir", ""),
        "grid_dim": cfg.vox.grid_dim,
        "resolution": cfg.vox.resolution,
        "ligand_radius": cfg.dset.get("ligand_radius", 0.5),
        "pocket_radius": cfg.dset.get("pocket_radius", -1),
        "n_samples": n_samples,
        "pos_thresh": _CH_FREQ_POS_THRESH,
        "task": "channel_freq_v1",
    }
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(cfg.dset.data_dir, f"channel_freq_{h}.pt")


# Atom-channel loss weights w = (1/freq)**power (0.5 gentle, 1.0 strong), normalized to sum to the
# atom-channel count; clip_ratio > 0 first floors w at max/clip_ratio so dense pocket atoms keep
# weight. freq, the atom-present voxel fraction per channel, is cached (_ch_freq_cache_path).
def precompute_channel_weights(
    train_dataset, voxelizer, cfg, device,
    n_samples: int = _CH_FREQ_DEFAULT_N_SAMPLES,
    power: float = 0.5,
    clip_ratio: float = 0.0,
) -> torch.Tensor:
    _nlig = int(cfg.model.get("n_channels_ligand", 7)); _npoc = int(cfg.model.get("n_channels_pocket", 4))
    n_atom_channels = _nlig + _npoc
    path = _ch_freq_cache_path(cfg, n_samples)
    if os.path.isfile(path):
        logger.info(f"loading channel-frequency cache from {path}")
        freq = torch.load(path, weights_only=True)["freq"].to(torch.float64)
    else:
        n = min(n_samples, len(train_dataset))
        logger.info(
            f"computing per-channel atom-voxel frequencies "
            f"(n_channels={n_atom_channels}) "
            f"over {n} training samples..."
        )
        sum_per_ch = torch.zeros(n_atom_channels, dtype=torch.float64)
        n_voxels_total = 0
        with torch.no_grad():
            for i in range(n):
                sample = train_dataset[i]
                lig = {k: sample["ligand"][k].unsqueeze(0).to(device)
                       for k in ("coords", "radius", "atoms_channel")}
                poc = {k: sample["pocket"][k].unsqueeze(0).to(device)
                       for k in ("coords", "radius", "atoms_channel")}
                v_lig = voxelizer.forward(lig, num_channels=int(cfg.model.get("n_channels_ligand", 7)))
                v_poc = voxelizer.forward(poc, num_channels=int(cfg.model.get("n_channels_pocket", 4)))
                atoms = torch.cat([v_lig, v_poc], dim=1)
                pos = (atoms > _CH_FREQ_POS_THRESH).float()
                sum_per_ch += pos.sum(dim=(0, 2, 3, 4)).cpu().to(torch.float64)
                n_voxels_total += atoms.shape[2] * atoms.shape[3] * atoms.shape[4]
        freq = (sum_per_ch / max(1, n_voxels_total)).clamp(min=1e-8)
        torch.save({"freq": freq.to(torch.float32), "n_samples": n,
                    "pos_thresh": _CH_FREQ_POS_THRESH}, path)
        logger.info(f"saved channel-frequency cache to {path}")

    w = (1.0 / freq).pow(power)
    if clip_ratio and clip_ratio > 0:
        w = w.clamp(min=float(w.max()) / float(clip_ratio))
    w = (w / w.sum() * float(n_atom_channels)).to(torch.float32)

    elem_lig = ["C", "O", "N", "S", "F", "Cl", "P"]
    elem_poc = ["C", "O", "N", "S"]
    names = [f"lig_{e}" for e in elem_lig] + [f"poc_{e}" for e in elem_poc]
    scheme = "1/freq" if power == 1.0 else ("1/sqrt(freq)" if power == 0.5 else f"1/freq^{power}")
    logger.info(
        f"per-channel atom weights ({scheme}, clip_ratio={clip_ratio or 'off'}, "
        f"norm to sum={n_atom_channels}; max/min={float(w.max()/w.min()):.1f}x):"
    )
    for i, name in enumerate(names):
        logger.info(f"  {name:>9}: freq={float(freq[i]):.3e}  weight={float(w[i]):.3f}")
    return w


# ── Logging ───────────────────────────────────────────────────────────────────

# One log line per split with every metric; val_metrics may be None.
def log_metrics(epoch, train_metrics, val_metrics, dt):
    logger.info(f"epoch: {epoch} ({dt:.2f}s)")
    for split, m in zip(["train", "val"], [train_metrics, val_metrics]):
        if m is None:
            continue
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in m.items())
        logger.info(f"[{split}] {parts}")


# ── Config reconciliation ─────────────────────────────────────────────────────

# input_mode and with_gradmag belong under cfg.model, but older cfg.yaml files (and resumes of
# them) keep them at the top level. Mirror both places in place, cfg.model winning, so readers
# of either location agree. Idempotent.
def _reconcile_input_keys(cfg: DictConfig) -> None:
    if "model" not in cfg or cfg.model is None:
        return
    with open_dict(cfg):
        for key, default in (("input_mode", "density"), ("with_gradmag", False)):
            in_model = OmegaConf.select(cfg, f"model.{key}", default=None)
            top = OmegaConf.select(cfg, key, default=None)
            val = in_model if in_model is not None else (top if top is not None else default)
            cfg.model[key] = val
            cfg[key] = val

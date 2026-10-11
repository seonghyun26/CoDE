"""DDP masked-autoencoder pretraining of the density ViT encoder.

The encoder reads voxelized ligand and pocket atoms with the X-ray density and its gradient
magnitude; blocks of the input are masked (mask/ config group) and reconstructed.

Checkpoint layout:
  state_dict_ema          full DensityViTMAE EMA
  encoder_state_dict_ema  encoder.* slice, loaded by VoxBind and the affinity probe

Launch: bash pretrain/scripts/pretrain.sh --experiment <preset> --gpus 0-3

Methods and flags removed in the 2026-10 refactor run from legacy/pretrain/train_density.py:
denoise, ELECTRA, ChA-MAE, MultiMAE, KoLeo, contrastive, data2vec, apo augmentation, modal and
ligand dropout, the mFo-DFc second map, synthetic density, merged/roleblob and two-tower inputs,
and the input ablations.
"""

import contextlib
import gc
import json
import logging
import math
import os
import random
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataset import create_dataloaders
from voxbind.models.adamw import AdamW
from voxbind.models.mae_ops import (
    make_atom_biased_block_mask, make_block_mask, make_cluster_mask, make_interface_mask,
    make_interface_atom_mask,
)
from voxbind.models.density_vit import DensityViTMAE
from pretrain.train_common import (
    AsyncCheckpointSaver, _amp_setup, _channel_layout, _cleanup_ddp, _reconcile_input_keys,
    _setup_ddp, _unwrap, _VALID_INPUT_MODES, log_metrics, maybe_compile_model,
    precompute_channel_weights, precompute_val, set_atom_channels,
)
from voxbind.models.ema import ModelEma
from voxbind.utils.base_utils import create_exp_dir, seed_everything
from voxbind.voxelizer import Voxelizer

logger = logging.getLogger("train-density-vit-mae")


# ── Prefetcher ────────────────────────────────────────────────────────────────

# Prepares each training batch one step ahead on a side CUDA stream: voxelize, assemble the
# input_mode channels [atoms, density, gradmag], add density noise, mask. Yields (x_in masked
# input, x_clean target, mask True = masked, target_str atom voxels for the struct head).
class MAEPrefetcher:
    # mask_ratio_min/max (both set, max > min): each batch draws its ratio from U[min, max] (R2MAE).
    # channel_groups: the ChannelViT groups, for per-group masking; mask_channel_independent
    # re-draws the spatial mask per group at mask_group_ratios (e.g. density harder than atoms).
    def __init__(
        self,
        loader,
        voxelizer: Voxelizer,
        sigma_noise: float,
        block_size: int,
        mask_ratio: float,
        generator: torch.Generator = None,
        input_mode: str = "density",
        mask_strategy: str = "uniform",
        mask_atom_tau: float = 1.0,
        mask_n_seeds: int = 4,
        mask_ratio_min: float = None,
        mask_ratio_max: float = None,
        with_gradmag: bool = False,
        gradmag_reconstruct: bool = True,
        gradmag_noise: bool = False,
        gradmag_without_density: bool = False,
        n_lig_ch: int = 7,
        n_poc_ch: int = 4,
        channel_groups: tuple = None,
        mask_channel_independent: bool = False,
        mask_group_ratios: tuple = None,
        group_complete_ratio: float = 0.9,
        group_complete_light_ratio: float = 0.15,
        group_complete_family_weights=None,
    ):
        assert input_mode in _VALID_INPUT_MODES, (
            f"input_mode={input_mode!r}; expected one of {_VALID_INPUT_MODES}"
        )
        assert mask_strategy in ("uniform", "atom_biased", "cluster", "ligand", "interface", "interface_atom", "per_group", "group_complete"), (
            f"mask_strategy={mask_strategy!r}; expected 'uniform', 'atom_biased', 'cluster', 'ligand', 'interface', 'interface_atom', 'per_group' or 'group_complete'"
        )

        self.loader = loader
        self.voxelizer = voxelizer
        self.sigma_noise = sigma_noise
        self.block_size = block_size
        self.mask_ratio = mask_ratio

        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self._var_ratio = (
            mask_ratio_min is not None and mask_ratio_max is not None
            and float(mask_ratio_max) > float(mask_ratio_min)
        )
        if self._var_ratio:
            print(f">> [MAEPrefetcher] VARIABLE mask_ratio ~ U[{mask_ratio_min}, {mask_ratio_max}] "
                  f"(per-batch, train-only); fixed fallback = {mask_ratio}", flush=True)

        self.generator = generator
        self.input_mode = input_mode
        self.mask_strategy = mask_strategy
        self.mask_atom_tau = mask_atom_tau
        self.mask_n_seeds = mask_n_seeds
        self.with_gradmag = with_gradmag
        self.gradmag_reconstruct = gradmag_reconstruct
        self.gradmag_noise = gradmag_noise
        self.n_lig_ch = int(n_lig_ch)
        self.n_poc_ch = int(n_poc_ch)

        self.channel_groups = tuple(int(c) for c in channel_groups) if channel_groups else None
        self.mask_channel_independent = bool(mask_channel_independent)
        self.mask_group_ratios = ([float(r) for r in mask_group_ratios]
                                  if mask_group_ratios is not None else None)
        self.group_complete_ratio = float(group_complete_ratio)
        self.group_complete_light_ratio = float(group_complete_light_ratio)
        self.group_complete_family_weights = group_complete_family_weights
        self.gradmag_without_density = bool(gradmag_without_density)

        self.layout = _channel_layout(
            input_mode, with_gradmag, gradmag_reconstruct,
            gradmag_without_density=self.gradmag_without_density,
        )
        self.stream = torch.cuda.Stream()

    # One batch: voxelize the atoms, assemble x_clean, add density noise, draw the mask, zero the
    # masked voxels. Runs on self.stream, so it overlaps the previous batch's compute.
    def _step(self, batch):
        device = self.voxelizer.device
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                v_lig = self.voxelizer.forward(batch["ligand"], num_channels=self.n_lig_ch)
                v_poc = self.voxelizer.forward(batch["pocket"], num_channels=self.n_poc_ch)

                # 2Fo-Fc crop, ±3σ-clipped and z-scored by the dataset; atom-only modes skip it.
                d_clean = None
                if self.input_mode in ("density", "atomblob_density"):
                    d_clean = batch["xray_density"].to(device, non_blocking=True).unsqueeze(1)

                # Atom-weighted masks (atom_biased, cluster) need the summed atom occupancy.
                atoms_sum = None
                if self.mask_strategy in ("atom_biased", "cluster"):
                    atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)

                # Gradient magnitude ‖∇ρ‖, precomputed by the dataset from the aligned crop.
                g_clean = None
                if self.with_gradmag:
                    g_clean = batch["xray_gradmag"].to(device, non_blocking=True).unsqueeze(1)

                # x_clean = [atoms, density] per input_mode, then gradmag as the last channel.
                if self.input_mode == "density":
                    x_clean = d_clean
                elif self.input_mode == "atomblob":
                    x_clean = torch.cat([v_lig, v_poc], dim=1)
                elif self.input_mode == "atomblob_density":
                    x_clean = torch.cat([v_lig, v_poc, d_clean], dim=1)
                else:
                    raise RuntimeError(f"unexpected input_mode={self.input_mode!r}")

                if self.with_gradmag:
                    x_clean = torch.cat([x_clean, g_clean], dim=1)

                # Noise on the density channel, located by index (gradmag may follow it). gradmag is
                # noised only with gradmag_noise; atom-only inputs get no noise.
                if self.sigma_noise > 0 and self.input_mode != "atomblob":
                    # One (B,1,G³) noise channel, from self.generator when one is set.
                    def _randn_channel():
                        shape = (x_clean.shape[0], 1, *x_clean.shape[2:])
                        if self.generator is not None:
                            return torch.randn(shape, device=device, generator=self.generator)
                        return torch.randn(shape, device=device)
                    x_noisy = x_clean.clone()
                    d_idx = self.layout["density_idx"]
                    x_noisy[:, d_idx:d_idx + 1] = (
                        x_noisy[:, d_idx:d_idx + 1] + _randn_channel() * self.sigma_noise
                    )
                    if self.with_gradmag and self.gradmag_noise:
                        g_idx = self.layout["gradmag_idx"]
                        x_noisy[:, g_idx:g_idx + 1] = (
                            x_noisy[:, g_idx:g_idx + 1] + _randn_channel() * self.sigma_noise
                        )
                else:
                    x_noisy = x_clean

                B = x_clean.shape[0]
                G = x_clean.shape[-1]

                # Per-batch ratio from U[min, max] when the R2MAE range is set (val keeps the fixed
                # ratio, so val loss stays comparable across epochs); else the fixed mask_ratio.
                if self._var_ratio:
                    _gdev = self.generator.device if self.generator is not None else device
                    _u = torch.rand((), device=_gdev, generator=self.generator).item()
                    ratio = self.mask_ratio_min + (self.mask_ratio_max - self.mask_ratio_min) * _u
                else:
                    ratio = self.mask_ratio

                # Mask by mask_strategy, (B,1,G³) for all channels or (B,C,G³) per channel group:
                # ligand hides ligand-heavy blocks, interface the ligand-pocket contacts
                # (interface_atom then atoms, solvent); group_complete hides one group per sample.
                if self.mask_channel_independent and self.channel_groups is not None:
                    _rr = self.mask_group_ratios or [ratio] * len(self.channel_groups)

                    # Spatial mask at ratio r: interface, atom_biased, else uniform blocks.
                    def _spatial(r):
                        if self.mask_strategy == "interface":
                            return make_interface_mask(
                                v_lig.sum(dim=1, keepdim=True), v_poc.sum(dim=1, keepdim=True),
                                self.block_size, r, tau=self.mask_atom_tau, generator=self.generator)
                        if self.mask_strategy == "atom_biased":
                            return make_atom_biased_block_mask(
                                v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True),
                                self.block_size, r, tau=self.mask_atom_tau, generator=self.generator)
                        return make_block_mask(B, G, self.block_size, r, device)

                    parts = [_spatial(float(_rr[gi]) if gi < len(_rr) else ratio)
                             .expand(-1, g, -1, -1, -1)
                             for gi, g in enumerate(self.channel_groups)]
                    mask = torch.cat(parts, dim=1)
                elif self.mask_strategy == "atom_biased":
                    mask = make_atom_biased_block_mask(
                        atoms_sum, self.block_size, ratio,
                        tau=self.mask_atom_tau, generator=self.generator,
                    )
                elif self.mask_strategy == "cluster":
                    mask = make_cluster_mask(
                        atoms_sum, self.block_size, ratio,
                        n_seeds=self.mask_n_seeds, generator=self.generator,
                    )
                elif self.mask_strategy == "ligand":
                    lig_sum = v_lig.sum(dim=1, keepdim=True)
                    mask = make_atom_biased_block_mask(
                        lig_sum, self.block_size, ratio,
                        tau=self.mask_atom_tau, generator=self.generator,
                    )
                elif self.mask_strategy == "interface":
                    mask = make_interface_mask(
                        v_lig.sum(dim=1, keepdim=True), v_poc.sum(dim=1, keepdim=True),
                        self.block_size, ratio,
                        tau=self.mask_atom_tau, generator=self.generator,
                    )
                elif self.mask_strategy == "interface_atom":
                    mask = make_interface_atom_mask(
                        v_lig.sum(dim=1, keepdim=True), v_poc.sum(dim=1, keepdim=True),
                        self.block_size, ratio,
                        tau=self.mask_atom_tau, generator=self.generator,
                    )
                elif self.mask_strategy == "per_group":
                    assert self.channel_groups is not None, \
                        "per_group masking needs model.channel_groups"
                    parts = [
                        make_block_mask(B, G, self.block_size, ratio, device).expand(-1, g, -1, -1, -1)
                        for g in self.channel_groups
                    ]
                    mask = torch.cat(parts, dim=1)
                elif self.mask_strategy == "group_complete":
                    assert self.channel_groups is not None, \
                        "group_complete masking needs model.channel_groups"
                    nG = len(self.channel_groups)
                    hard = float(self.group_complete_ratio)
                    light = float(self.group_complete_light_ratio)

                    # Group each sample hides: uniform, or from group_complete_family_weights.
                    if self.group_complete_family_weights is not None:
                        w = torch.tensor(list(self.group_complete_family_weights),
                                         device=device, dtype=torch.float)
                        assert w.numel() == nG, "group_complete_family_weights len must == nG"
                        pick = torch.multinomial(w, B, replacement=True, generator=self.generator)
                    else:
                        pick = torch.randint(nG, (B,), device=device, generator=self.generator)

                    parts = []
                    for gi, g in enumerate(self.channel_groups):
                        mh = make_block_mask(B, G, self.block_size, hard, device)
                        ml = make_block_mask(B, G, self.block_size, light, device)
                        sel = (pick == gi).view(B, 1, 1, 1, 1)
                        mg = torch.where(sel, mh, ml)
                        parts.append(mg.expand(-1, g, -1, -1, -1))
                    mask = torch.cat(parts, dim=1)
                else:
                    mask = make_block_mask(B, G, self.block_size, ratio, device)

                x_in = x_noisy * (~mask).to(x_noisy.dtype)
                target_str = torch.cat([v_lig, v_poc], dim=1)
        return x_in, x_clean, mask, target_str

    # Yields batch i while batch i+1 is prepared on the side stream.
    def __iter__(self):
        loader_it = iter(self.loader)
        try:
            next_batch = next(loader_it)
        except StopIteration:
            return
        next_out = self._step(next_batch)
        for batch in loader_it:
            torch.cuda.current_stream().wait_stream(self.stream)
            cur = next_out
            next_out = self._step(batch)
            yield cur
        torch.cuda.current_stream().wait_stream(self.stream)
        yield next_out

    # Batches per epoch, as in the loader.
    def __len__(self):
        return len(self.loader)


# ── Losses ────────────────────────────────────────────────────────────────────

# Masked-voxel reconstruction losses. L_dens: MSE over the masked entries of the reconstructed
# channels, weighted by ch_weight per channel and atom_pos_weight at atom-present voxels, with
# per-modality parts L_dens_atom/density/gradmag. L_str: structure-head MSE, or 0 without it.
def compute_losses(
    out_pretext, a_hat, x_clean, target_str, mask,
    struct_pos_weight: float = 1.0,
    struct_pos_thresh: float = 0.05,
    input_mode: str = "density",
    ch_weight: torch.Tensor = None,
    atom_pos_weight: float = 1.0,
    atom_pos_thresh: float = 0.05,
    with_gradmag: bool = False,
    gradmag_reconstruct: bool = True,
    gradmag_without_density: bool = False,
):
    losses = {}
    m = mask.to(out_pretext.dtype)
    n_channels = out_pretext.shape[1]

    # Normalize by the scored (channel, voxel) entries: a (B,1,G³) mask scores every channel at
    # its voxels, while a per-group (B,C,G³) mask already counts each channel's own entries.
    per_channel_mask = (m.shape[1] == n_channels and n_channels > 1)
    n_masked_spatial = m.sum().clamp(min=1.0)
    masked_entries = n_masked_spatial if per_channel_mask else (n_masked_spatial * n_channels)

    # The target is the leading n_recon channels of x_clean (input-only gradmag trails it).
    target = x_clean if n_channels == x_clean.shape[1] else x_clean[:, :n_channels]
    diff_sq = (out_pretext - target) ** 2 * m

    # atom_pos_weight multiplies the error at atom-present voxels of the atom channels.
    lay = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct,
                          gradmag_without_density=gradmag_without_density)
    n_atom = lay["n_atom"]
    if atom_pos_weight > 1.0 and n_atom > 0:
        pos_atom = (x_clean[:, :n_atom] > atom_pos_thresh).to(diff_sq.dtype)
        atom_mult = 1.0 + (atom_pos_weight - 1.0) * pos_atom
        if n_atom < n_channels:
            pad = torch.ones(
                (atom_mult.shape[0], n_channels - n_atom, *atom_mult.shape[2:]),
                device=atom_mult.device, dtype=atom_mult.dtype,
            )
            full_mult = torch.cat([atom_mult, pad], dim=1)
        else:
            full_mult = atom_mult
        diff_sq = diff_sq * full_mult

    # ch_weight sums to n_channels, so weighting keeps the overall loss scale.
    if ch_weight is not None:
        assert ch_weight.shape[0] == n_channels, (
            f"ch_weight has {ch_weight.shape[0]} entries but out_pretext has "
            f"{n_channels} channels"
        )
        w_full = ch_weight.to(diff_sq.device, diff_sq.dtype).view(1, n_channels, 1, 1, 1)
        diff_sq_weighted = diff_sq * w_full
        L_pretext = diff_sq_weighted.sum() / masked_entries
    else:
        L_pretext = diff_sq.sum() / masked_entries

    # Per-modality parts over the reconstructed channels [atoms, density, gradmag], when the
    # target mixes two or more modalities.
    n_density = lay["n_density"]
    recon_gradmag = with_gradmag and gradmag_reconstruct
    n_modalities = (n_atom > 0) + (n_density > 0) + (1 if recon_gradmag else 0)
    if n_modalities >= 2:
        # Weighted MSE over channels [c0, c1), normalized like L_dens.
        def _masked_mean(c0, c1):
            if ch_weight is not None:
                num = (diff_sq[:, c0:c1] * w_full[:, c0:c1]).sum()
            else:
                num = diff_sq[:, c0:c1].sum()
            denom = (m[:, c0:c1].sum().clamp(min=1.0) if per_channel_mask
                     else n_masked_spatial * (c1 - c0))
            return num / denom

        if n_atom > 0:
            losses["L_dens_atom"] = _masked_mean(0, n_atom)
        if n_density > 0:
            losses["L_dens_density"] = _masked_mean(n_atom, n_atom + n_density)
        if recon_gradmag:
            g0 = n_atom + n_density
            losses["L_dens_gradmag"] = _masked_mean(g0, g0 + 1)
    losses["L_dens"] = L_pretext

    # Without the structure head L_str is 0, kept for callers that always read it.
    if a_hat is not None:
        if struct_pos_weight > 1.0:
            pos = (target_str > struct_pos_thresh).to(a_hat.dtype)
            w = 1.0 + (struct_pos_weight - 1.0) * pos
            L_str = ((a_hat - target_str) ** 2 * w).mean()
        else:
            L_str = F.mse_loss(a_hat, target_str)
        losses["L_str"] = L_str
    else:
        losses["L_str"] = torch.zeros((), device=out_pretext.device)
    return losses


# One deterministic validation pass: noise and masks come from a generator seeded per batch, so
# every epoch scores the same inputs. Only atom_biased and cluster keep their masks here; every
# other strategy is scored with uniform block masks at mask_ratio.
def val_epoch(cfg, model, val_cache, device, ch_weight=None) -> dict:
    model.eval()
    amp_enabled, amp_ctx_fn = _amp_setup(cfg)
    channels_last = bool(cfg.get("channels_last", False))
    block = cfg.mae.block_size
    ratio = cfg.mae.mask_ratio
    sigma_noise = cfg.mae.sigma_noise
    input_mode = str(cfg.get("input_mode", "density"))
    mask_strategy = str(cfg.mae.get("mask_strategy", "uniform"))
    mask_atom_tau = float(cfg.mae.get("mask_atom_tau", 1.0))
    mask_n_seeds = int(cfg.mae.get("mask_n_seeds", 4))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    gradmag_noise = bool(cfg.mae.get("gradmag_noise", False))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct,
                             gradmag_without_density=bool(cfg.mae.get("gradmag_without_density", False)))
    need_atoms_sum = mask_strategy in ("atom_biased", "cluster")
    pretext_key = "L_dens"
    lambda_pretext = float(cfg.mae.get("lambda_dens", 1.0))
    lambda_str = float(cfg.mae.get("lambda_str", 1.0))

    voxels_lig = val_cache["voxels_lig"]
    voxels_poc = val_cache["voxels_poc"]
    cached_xray = val_cache.get("xray_density", None)
    cached_gradmag = val_cache.get("xray_gradmag", None)
    n = voxels_lig.shape[0]

    L_pretext_sum, L_str_sum, n_batches = 0.0, 0.0, 0
    L_dens_atom_sum, L_dens_density_sum, L_dens_gradmag_sum = 0.0, 0.0, 0.0
    gen = torch.Generator(device=device)
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, n, cfg.bsz)):
            gen.manual_seed(batch_idx + 1)
            v_lig = voxels_lig[start:start + cfg.bsz].to(device, non_blocking=True)
            v_poc = voxels_poc[start:start + cfg.bsz].to(device, non_blocking=True)
            d_clean = None
            atoms_sum = None
            if need_atoms_sum:
                atoms_sum = v_lig.sum(dim=1, keepdim=True) + v_poc.sum(dim=1, keepdim=True)
            if input_mode in ("density", "atomblob_density"):
                d_clean = cached_xray[start:start + cfg.bsz].to(device, non_blocking=True).unsqueeze(1)
            g_clean = None
            if with_gradmag:
                g_clean = cached_gradmag[start:start + cfg.bsz].to(device, non_blocking=True).unsqueeze(1)

            if input_mode == "density":
                x_clean = d_clean
            elif input_mode == "atomblob":
                x_clean = torch.cat([v_lig, v_poc], dim=1)
            elif input_mode == "atomblob_density":
                x_clean = torch.cat([v_lig, v_poc, d_clean], dim=1)
            else:
                raise RuntimeError(f"unexpected input_mode={input_mode!r}")

            if with_gradmag:
                x_clean = torch.cat([x_clean, g_clean], dim=1)

            if sigma_noise > 0 and input_mode != "atomblob":
                x_noisy = x_clean.clone()
                d_idx = layout["density_idx"]
                x_noisy[:, d_idx:d_idx + 1] = (
                    x_noisy[:, d_idx:d_idx + 1]
                    + torch.randn((x_clean.shape[0], 1, *x_clean.shape[2:]),
                                  device=device, generator=gen) * sigma_noise
                )
                if with_gradmag and gradmag_noise:
                    g_idx = layout["gradmag_idx"]
                    x_noisy[:, g_idx:g_idx + 1] = (
                        x_noisy[:, g_idx:g_idx + 1]
                        + torch.randn((x_clean.shape[0], 1, *x_clean.shape[2:]),
                                      device=device, generator=gen) * sigma_noise
                    )
            else:
                x_noisy = x_clean

            B, G = x_clean.shape[0], x_clean.shape[-1]
            if mask_strategy == "atom_biased":
                mask = make_atom_biased_block_mask(
                    atoms_sum, block, ratio, tau=mask_atom_tau, generator=gen,
                )
            elif mask_strategy == "cluster":
                mask = make_cluster_mask(
                    atoms_sum, block, ratio, n_seeds=mask_n_seeds, generator=gen,
                )
            else:
                blocks = torch.rand(B, 1, G // block, G // block, G // block,
                                    device=device, generator=gen) < ratio
                mask = blocks.repeat_interleave(block, 2)\
                             .repeat_interleave(block, 3)\
                             .repeat_interleave(block, 4)

            x_in = x_noisy * (~mask).to(x_noisy.dtype)

            target_str = torch.cat([v_lig, v_poc], dim=1)
            if channels_last:
                x_in = x_in.to(memory_format=torch.channels_last_3d)
            with amp_ctx_fn():
                out_pretext, a_hat = model(x_in)
            out_pretext = out_pretext.float()
            if a_hat is not None:
                a_hat = a_hat.float()
            losses = compute_losses(
                out_pretext, a_hat, x_clean, target_str, mask,
                struct_pos_weight=float(cfg.mae.get("struct_pos_weight", 1.0)),
                struct_pos_thresh=float(cfg.mae.get("struct_pos_thresh", 0.05)),
                input_mode=input_mode,
                ch_weight=ch_weight,
                atom_pos_weight=float(cfg.mae.get("atom_pos_weight", 1.0)),
                atom_pos_thresh=float(cfg.mae.get("atom_pos_thresh", 0.05)),
                with_gradmag=with_gradmag,
                gradmag_reconstruct=gradmag_reconstruct,
                gradmag_without_density=bool(cfg.mae.get("gradmag_without_density", False)),
            )
            L_pretext_sum += losses[pretext_key].item()
            L_str_sum += losses["L_str"].item()
            if "L_dens_atom" in losses:
                L_dens_atom_sum += losses["L_dens_atom"].item()
            if "L_dens_density" in losses:
                L_dens_density_sum += losses["L_dens_density"].item()
            if "L_dens_gradmag" in losses:
                L_dens_gradmag_sum += losses["L_dens_gradmag"].item()
            n_batches += 1
            if cfg.debug and n_batches >= 5:
                break

    n_b = max(1, n_batches)
    metrics = {
        pretext_key: L_pretext_sum / n_b,
        "L_str": L_str_sum / n_b,
        "loss": (lambda_pretext * L_pretext_sum + lambda_str * L_str_sum) / n_b,
    }
    recon_gradmag = with_gradmag and gradmag_reconstruct
    n_modalities = (layout["n_atom"] > 0) + (layout["n_density"] > 0) + (1 if recon_gradmag else 0)
    if n_modalities >= 2:
        if layout["n_atom"] > 0:
            metrics["L_dens_atom"] = L_dens_atom_sum / n_b
        if layout["n_density"] > 0:
            metrics["L_dens_density"] = L_dens_density_sum / n_b
        if recon_gradmag:
            metrics["L_dens_gradmag"] = L_dens_gradmag_sum / n_b
    return metrics


# ── Train loop ────────────────────────────────────────────────────────────────

# One training epoch: gradient accumulation (DDP syncs only on the stepping micro-batch),
# optional grad clipping, an EMA update per optimizer step. Returns (metrics, global_step).
def train_epoch(cfg, prefetcher, model, optimizer, model_ema, device, global_step,
                ch_weight=None, ema_source_model=None) -> tuple:
    model.train()
    input_mode = str(cfg.get("input_mode", "density"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct,
                             gradmag_without_density=bool(cfg.mae.get("gradmag_without_density", False)))
    pretext_key = "L_dens"
    lambda_pretext = float(cfg.mae.get("lambda_dens", 1.0))
    lambda_str = float(cfg.mae.get("lambda_str", 1.0))

    L_pretext_sum, L_str_sum, grad_norm_sum, n_batches = 0.0, 0.0, 0.0, 0
    L_dens_atom_sum, L_dens_density_sum, L_dens_gradmag_sum = 0.0, 0.0, 0.0
    accum_steps = max(1, int(cfg.get("accum_steps", 1)))
    grad_clip = float(cfg.mae.get("grad_clip", 0.0))
    optimizer.zero_grad(set_to_none=True)
    n_iter = len(prefetcher)

    amp_enabled, amp_ctx_fn = _amp_setup(cfg)
    channels_last = bool(cfg.get("channels_last", False))
    for i, (x_in, x_clean, mask, target_str) in enumerate(prefetcher):
        is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_iter)
        sync_ctx = contextlib.nullcontext()
        if not is_step and hasattr(model, "no_sync"):
            sync_ctx = model.no_sync()
        if channels_last:
            x_in = x_in.to(memory_format=torch.channels_last_3d)
        with sync_ctx:
            with amp_ctx_fn():
                out_pretext, a_hat = model(x_in)

            # Loss math in fp32; the forward ran in bf16 under autocast.
            out_pretext = out_pretext.float()
            if a_hat is not None:
                a_hat = a_hat.float()
            losses = compute_losses(
                out_pretext, a_hat, x_clean, target_str, mask,
                struct_pos_weight=float(cfg.mae.get("struct_pos_weight", 1.0)),
                struct_pos_thresh=float(cfg.mae.get("struct_pos_thresh", 0.05)),
                input_mode=input_mode,
                ch_weight=ch_weight,
                atom_pos_weight=float(cfg.mae.get("atom_pos_weight", 1.0)),
                atom_pos_thresh=float(cfg.mae.get("atom_pos_thresh", 0.05)),
                with_gradmag=with_gradmag,
                gradmag_reconstruct=gradmag_reconstruct,
                gradmag_without_density=bool(cfg.mae.get("gradmag_without_density", False)),
            )
            L_pretext = losses[pretext_key]
            L_str = losses["L_str"]
            loss = lambda_pretext * L_pretext + lambda_str * L_str
            loss.backward()
        if is_step:
            if grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    _unwrap(model).parameters(), max_norm=grad_clip
                )
                grad_norm_sum += float(gn)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model_ema.update(ema_source_model if ema_source_model is not None else _unwrap(model))
            global_step += 1
        L_pretext_sum += L_pretext.item()
        L_str_sum += L_str.item()
        if "L_dens_atom" in losses:
            L_dens_atom_sum += losses["L_dens_atom"].item()
        if "L_dens_density" in losses:
            L_dens_density_sum += losses["L_dens_density"].item()
        if "L_dens_gradmag" in losses:
            L_dens_gradmag_sum += losses["L_dens_gradmag"].item()
        n_batches += 1
        if cfg.debug and i == 10:
            break

    n_b = max(1, n_batches)
    metrics = {
        pretext_key: L_pretext_sum / n_b,
        "L_str": L_str_sum / n_b,
        "loss": (lambda_pretext * L_pretext_sum + lambda_str * L_str_sum) / n_b,
    }
    recon_gradmag = with_gradmag and gradmag_reconstruct
    n_modalities = (layout["n_atom"] > 0) + (layout["n_density"] > 0) + (1 if recon_gradmag else 0)
    if n_modalities >= 2:
        if layout["n_atom"] > 0:
            metrics["L_dens_atom"] = L_dens_atom_sum / n_b
        if layout["n_density"] > 0:
            metrics["L_dens_density"] = L_dens_density_sum / n_b
        if recon_gradmag:
            metrics["L_dens_gradmag"] = L_dens_gradmag_sum / n_b
    if grad_clip > 0:
        n_steps = max(1, n_batches // accum_steps)
        metrics["grad_norm"] = grad_norm_sum / n_steps
    return metrics, global_step


# ── Main ──────────────────────────────────────────────────────────────────────

# Fail when most sampled training complexes load without density: the loader returns zeros for a
# map missing from the cache, so training would silently lack density. Restores the RNG state.
def _check_density_coverage(dataset, n: int = 64, min_frac: float = 0.9) -> None:
    states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    idx = random.Random(0).sample(range(len(dataset)), min(n, len(dataset)))
    have = sum(bool(dataset[i]["xray_available"]) for i in idx)
    random.setstate(states[0]); np.random.set_state(states[1]); torch.random.set_rng_state(states[2])
    if have < min_frac * len(idx):
        raise RuntimeError(
            f"only {have}/{len(idx)} sampled training complexes have density; check the map cache "
            f"(dset.ccp4_dir) or use the precomputed box (dset.box_path)")
    logger.info(f"density coverage: {have}/{len(idx)} sampled training complexes")


# The DensityViTMAE for this run, sized by cfg.model and the channel layout.
def _build_vit(cfg: DictConfig, device, layout: dict, is_main: bool):
    input_mode = str(cfg.get("input_mode", "density"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    dual_head = bool(cfg.model.get("dual_head", False))
    if with_gradmag and dual_head:
        raise ValueError(
            "dual_head is not yet supported with with_gradmag (the dual head "
            "assumes a lone trailing density channel); use the single MAE head."
        )

    head_hidden_dim = int(cfg.model.get("head_hidden_dim", 0))
    head_depth = int(cfg.model.get("head_depth", 2))
    head_style = str(cfg.model.get("head_style", "conv"))
    splat_k = int(cfg.model.get("splat_k", 4))
    splat_aniso = bool(cfg.model.get("splat_aniso", False))
    pos_encoding = str(cfg.model.get("pos_encoding", "learnable"))
    n_in_cfg = int(cfg.model.get("n_in_channels", 1))
    if n_in_cfg != layout["n_in"]:
        raise RuntimeError(
            f"input_mode={input_mode!r} with_gradmag={with_gradmag} expects "
            f"model.n_in_channels={layout['n_in']}, got {n_in_cfg}"
        )

    # Optional learned radius-conditioned atom embedding (model.radius_embed_k > 0) in place of
    # the one-hot atom channels; only the patch embed changes, the target stays the raw channels.
    radius_embed = None
    rek = int(cfg.model.get("radius_embed_k", 0))
    if rek > 0:
        from voxbind.constants import vdw_radius
        if layout["n_atom"] <= 0:
            raise RuntimeError("radius_embed_k>0 requires per-element atom channels "
                               f"(atomblob* input_mode); got input_mode={input_mode!r}")
        LIG_EL = ["C", "O", "N", "S", "F", "Cl", "P"]
        POC_EL = ["C", "O", "N", "S"]
        n_lig = int(cfg.model.get("n_channels_ligand", 7))
        n_poc = int(cfg.model.get("n_channels_pocket", 4))

        # vdW radius per element channel (carbon past the element list).
        def _radii(elems, n):
            return [vdw_radius(elems[i]) if i < len(elems) else vdw_radius("C")
                    for i in range(n)]

        # n_pt: the density (+ gradmag) channels that pass through unembedded.
        n_pt = layout["n_in"] - layout["n_atom"]
        scales_cfg = cfg.model.get("radius_embed_scales", None)
        scales = [float(s) for s in scales_cfg] if scales_cfg is not None else [0.0]
        radius_embed = {
            "lig_radii": _radii(LIG_EL, n_lig),
            "poc_radii": _radii(POC_EL, n_poc),
            "k": rek,
            "hidden": int(cfg.model.get("radius_embed_hidden", 16)),
            "n_passthrough": n_pt,
            "scales": scales,
        }
        if is_main:
            logger.info(f"radius-embed front-end: k={rek} scales={scales} "
                        f"lig_radii={radius_embed['lig_radii']} poc_radii={radius_embed['poc_radii']} "
                        f"n_pt={n_pt} → encoder patch_embed={2*rek+n_pt}ch groups=({rek},{rek},{n_pt}), "
                        f"recon={layout['n_recon']}ch")

    model = DensityViTMAE(
        grid_dim=int(cfg.vox.grid_dim),
        patch_size=int(cfg.model.patch_size),
        n_in_channels=n_in_cfg,
        n_recon_channels=layout["n_recon"],
        n_channels=int(cfg.model.n_channels),
        dim=int(cfg.model.dim),
        depth=int(cfg.model.depth),
        n_heads=int(cfg.model.heads),
        mlp_ratio=int(cfg.model.mlp_ratio),
        dropout=float(cfg.model.dropout),
        n_struct_channels=int(cfg.model.n_struct_channels),
        dual_head=dual_head,
        head_hidden_dim=head_hidden_dim,
        head_depth=head_depth,
        head_style=head_style,
        splat_k=splat_k,
        splat_aniso=splat_aniso,
        pos_encoding=pos_encoding,
        rope_fast=bool(cfg.model.get("rope_fast", False)),
        patch_embed_mode=str(cfg.model.get("patch_embed_mode", "fused")),
        channel_groups=(tuple(int(c) for c in cfg.model.channel_groups)
                        if cfg.model.get("channel_groups", None) else None),
        channel_group_dropout=float(cfg.model.get("channel_group_dropout", 0.0)),
        radius_embed=radius_embed,
        drop_path=float(cfg.model.get("drop_path", 0.0)),
        n_memory_tokens=int(cfg.model.get("n_memory_tokens", 0)),
    ).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"DensityViTMAE has {(n_params/1e6):.02f}M parameters "
                    f"(input_mode={input_mode}, "
                    f"with_gradmag={with_gradmag} (recon={gradmag_reconstruct}), "
                    f"n_in={layout['n_in']} n_recon={layout['n_recon']}, "
                    f"dual_head={dual_head}, head_style={head_style}, "
                    f"head_hidden_dim={head_hidden_dim or '(c_half)'}, head_depth={head_depth})")
    return model


# The pretraining run: DDP, resume, data, model and optimizer, EMA, val cache and channel
# weights, then the epoch loop with checkpointing; the best-validation checkpoint becomes final.
def run(cfg: DictConfig) -> None:
    assert torch.cuda.is_available(), "GPU required."
    rank, local_rank, world_size = _setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    start_epoch = 0

    # Normalize the input keys before create_exp_dir writes cfg.yaml; pin the atom-channel
    # counts before any _channel_layout call or model build.
    _reconcile_input_keys(cfg)
    set_atom_channels(int(cfg.model.get("n_channels_ligand", 7)), int(cfg.model.get("n_channels_pocket", 4)))
    create_exp_dir(cfg, write=is_main)
    if world_size > 1:
        dist.barrier()

    logger.info(f"DDP: world_size={world_size} rank={rank} local_rank={local_rank}")
    if is_main:
        logger.info(f"saving experiments in: {cfg.output_dir}")
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.seed + rank)

    # Resume: reload the run's cfg.yaml, keeping wandb, resume_epoch and num_epochs from the
    # command line.
    if cfg.resume is not None and os.path.isdir(cfg.resume):
        if is_main:
            logger.info(f"resuming from: {cfg.resume}")
        resume = cfg.resume
        wandb_override = cfg.wandb
        resume_epoch_override = cfg.resume_epoch
        num_epochs_override = cfg.num_epochs
        cfg = OmegaConf.load(os.path.join(cfg.resume, "cfg.yaml"))
        cfg.output_dir, cfg.resume = resume, resume
        cfg.wandb = wandb_override
        cfg.resume_epoch = resume_epoch_override
        cfg.num_epochs = num_epochs_override

    # Again, for a cfg.yaml reloaded on resume.
    _reconcile_input_keys(cfg)

    if is_main:
        logger.info("cfg:\n" + OmegaConf.to_yaml(cfg))

    # W&B tags must be strings (OmegaConf parses numeric list items as ints); the run name drops
    # exp_name's YYMMDD_ prefix, which the run directory keeps.
    if is_main and cfg.wandb:
        try:
            _tags = cfg.get("wandb_tags", None)
            _tags = [str(tag) for tag in _tags] if _tags else None
            import re as _re
            _wandb_name = _re.sub(r"^\d{6}_", "", cfg.exp_name)
            wandb.init(
                project=cfg.get("wandb_project", "binding-affinity"),
                entity=cfg.get("wandb_entity", None),
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=_wandb_name,
                tags=_tags,
                dir=cfg.output_dir,
                resume="allow",
                settings=wandb.Settings(code_dir=".", init_timeout=300),
            )
        except Exception as e:
            logger.warning(f"wandb.init failed (continuing without): {e}")

    loader_train, loader_val, _ = create_dataloaders(
        cfg, distributed=True, rank=rank, world_size=world_size,
    )
    n_train, n_val = len(loader_train.dataset), len(loader_val.dataset)
    if is_main and str(cfg.get("input_mode", "density")) != "atomblob":
        _check_density_coverage(loader_train.dataset)
    if is_main:
        accum = max(1, int(cfg.get("accum_steps", 1)))
        logger.info(f"train/val set size: {n_train}/{n_val}")
        logger.info(f"train batches per rank: {len(loader_train)} "
                    f"(effective batch = {cfg.bsz * world_size * accum})")

    # Model: ViT encoder + MAE head (+ optional structure head), widths from the channel layout.
    input_mode = str(cfg.get("input_mode", "density"))
    with_gradmag = bool(cfg.get("with_gradmag", False))
    gradmag_reconstruct = bool(cfg.mae.get("gradmag_reconstruct", True))
    gradmag_noise = bool(cfg.mae.get("gradmag_noise", False))
    gradmag_channel_weight = float(cfg.mae.get("gradmag_channel_weight", 1.0))
    layout = _channel_layout(input_mode, with_gradmag, gradmag_reconstruct,
                             gradmag_without_density=bool(cfg.mae.get("gradmag_without_density", False)))
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    model = _build_vit(cfg, device, layout, is_main)

    # channels_last_3d (faster cuDNN Conv3d) before optimizer, compile, DDP and EMA, so every
    # copy inherits it.
    if bool(cfg.get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last_3d)
        if is_main:
            logger.info("channels_last_3d memory format enabled")

    # Optimizer: Muon on the 2D transformer-block weights with AdamW for the rest (DDP-safe:
    # equal all-reduced grads give equal updates; constant LR); fused AdamW with the vendored
    # betas and eps=1e-8 (the fused kernel returns NaN at eps=0); else the vendored AdamW.
    _opt_type = str(cfg.get("optimizer", {}).get("type", "adamw")).lower()
    using_muon = _opt_type == "muon"
    if using_muon:
        from voxbind.models.muon import SingleDeviceMuonWithAuxAdam
        muon_lr  = float(cfg.get("optimizer", {}).get("muon_lr", 0.02))
        muon_mom = float(cfg.get("optimizer", {}).get("muon_momentum", 0.95))
        muon_p, adam_p = [], []
        for _n, _p in model.named_parameters():
            if not _p.requires_grad:
                continue
            (muon_p if (_p.ndim == 2 and "blocks." in _n) else adam_p).append(_p)
        assert len(muon_p) > 0, ("Muon selected 0 matrices — no 2D param name contains 'blocks.'; "
                                 "check the model's parameter names before running.")
        optimizer = SingleDeviceMuonWithAuxAdam([
            dict(params=muon_p, use_muon=True,  lr=muon_lr, momentum=muon_mom, weight_decay=cfg.wd),
            dict(params=adam_p, use_muon=False, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.wd),
        ])
        if is_main:
            _nm = sum(p.numel() for p in muon_p) / 1e6
            _na = sum(p.numel() for p in adam_p) / 1e6
            logger.info(f"Muon+AuxAdam: {len(muon_p)} block matrices ({_nm:.1f}M) @ muon_lr={muon_lr} "
                        f"mom={muon_mom}; {len(adam_p)} aux tensors ({_na:.1f}M) @ AdamW lr={cfg.lr}; "
                        f"wd={cfg.wd}. Constant LR (epoch scheduler skipped for muon).")
    elif bool(cfg.get("optimizer", {}).get("fused", False)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd,
                                      betas=(0.99, 0.999), eps=1e-8, fused=True)
        if is_main:
            logger.info("fused AdamW enabled (betas=(0.99,0.999), eps=1e-8 ~ vendored AdamW)")
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optimizer.zero_grad()

    voxelizer = Voxelizer(
        grid_dim=cfg.vox.grid_dim,
        resolution=cfg.vox.resolution,
        cubes_around=cfg.vox.cubes_around,
        device=device,
    )

    # Resume: model weights from the checkpoint's EMA state, optimizer state, epoch and step,
    # loaded before the DDP wrap.
    global_step = 0
    if cfg.resume is not None:
        ckpt_path = os.path.join(cfg.output_dir, "checkpoint.pth.tar")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("state_dict_ema", ckpt.get("state_dict"))
            sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            if is_main:
                logger.info(f"resumed from epoch {start_epoch - 1} (global_step={global_step})")
            if cfg.resume_epoch is not None:
                start_epoch = cfg.resume_epoch

    # Compile, then DDP. find_unused_parameters stays False: ChannelViT runs every group's patch
    # embed before dropping groups, so every parameter gets a gradient.
    model_train = maybe_compile_model(model, cfg, is_main=is_main)
    if world_size > 1:
        model_train = torch.nn.parallel.DistributedDataParallel(
            model_train,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    model_ema = ModelEma(model, decay=cfg.mae.ema_decay,
                         foreach=bool(cfg.get("ema", {}).get("foreach", False)))

    # Rank 0 builds the val cache first; the other ranks then load it.
    if is_main:
        val_cache = precompute_val(loader_val, voxelizer, cfg)
    if world_size > 1:
        dist.barrier()
    if not is_main:
        val_cache = precompute_val(loader_val, voxelizer, cfg)

    # Collect now: the train loader's workers fork later, and a CUDA tensor left in a garbage
    # cycle (e.g. from building the val cache) would be freed in a child and abort it.
    gc.collect()

    # Per-channel loss weights [atoms, density, gradmag], rescaled to sum to n_recon so the loss
    # scale is unchanged. build_ch_weight stays None (unweighted path) when all options are default.
    build_ch_weight = None
    dens_warmup_epochs = 0
    ch_weighting = str(cfg.mae.get("channel_weighting", "uniform"))
    if ch_weighting not in ("uniform", "inv_sqrt_freq", "inv_freq"):
        raise ValueError(
            f"unknown channel_weighting={ch_weighting!r}; expected "
            f"'uniform', 'inv_sqrt_freq', or 'inv_freq'"
        )

    # inv_sqrt_freq / inv_freq: atom weights (1/freq)**0.5 or 1/freq; channel_weight_clip_ratio
    # caps the rare/common ratio; density and gradmag take their configured weights.
    freq_power = 1.0 if ch_weighting == "inv_freq" else 0.5
    channel_weight_clip = float(cfg.mae.get("channel_weight_clip_ratio", 0.0))
    density_channel_weight = float(cfg.mae.get("density_channel_weight", 1.0))
    needs_atom_weights = ch_weighting in ("inv_sqrt_freq", "inv_freq")
    needs_dens_downweight = (
        input_mode == "atomblob_density" and density_channel_weight != 1.0
    )
    needs_gradmag_weight = (
        with_gradmag and gradmag_reconstruct and gradmag_channel_weight != 1.0
    )

    # Optional warm-up: the density/gradmag weights ramp 0 -> target (cosine) over the first
    # density_weight_warmup_epochs, from absolute epoch 0 so a resume continues the ramp.
    dens_warmup_epochs = int(cfg.mae.get("density_weight_warmup_epochs", 0))
    has_aux_recon = (
        input_mode == "atomblob_density" or (with_gradmag and gradmag_reconstruct)
    )
    needs_warmup = dens_warmup_epochs > 0 and has_aux_recon

    if needs_atom_weights or needs_dens_downweight or needs_gradmag_weight or needs_warmup:
        if needs_atom_weights and input_mode not in ("atomblob", "atomblob_density"):
            raise ValueError(
                f"channel_weighting={ch_weighting!r} requires an atomblob* input_mode; "
                f"got input_mode={input_mode!r}"
            )

        n_atom = layout["n_atom"]
        n_density = layout["n_density"]

        # Atom weights: rank 0 scans and caches the frequencies, then the other ranks load them.
        if needs_atom_weights:
            if is_main:
                atom_w = precompute_channel_weights(
                    loader_train.dataset, voxelizer, cfg, device=device,
                    power=freq_power, clip_ratio=channel_weight_clip,
                )
            if world_size > 1:
                dist.barrier()
            if not is_main:
                atom_w = precompute_channel_weights(
                    loader_train.dataset, voxelizer, cfg, device=device,
                    power=freq_power, clip_ratio=channel_weight_clip,
                )
        else:
            atom_w = torch.ones(n_atom)

        # Warm-up factor for the density/gradmag weights: cosine 0 -> 1, then 1.
        def _aux_scale(epoch: int) -> float:
            if dens_warmup_epochs <= 0:
                return 1.0
            p = min(1.0, max(0.0, float(epoch) / float(dens_warmup_epochs)))
            return 0.5 * (1.0 - math.cos(math.pi * p))

        # (weight vector for this epoch, warm-up factor): [atoms, density?, gradmag?] rescaled to
        # sum to n_recon. Rebuilt each epoch; only the warm-up factor changes.
        def build_ch_weight(epoch: int):
            scale = _aux_scale(epoch)
            parts = [atom_w]
            if n_density > 0:
                parts.append(torch.tensor([density_channel_weight * scale]))
            if with_gradmag and gradmag_reconstruct:
                parts.append(torch.tensor([gradmag_channel_weight * scale]))
            raw = torch.cat(parts, dim=0)
            assert raw.shape[0] == layout["n_recon"], (
                f"weight vector length {raw.shape[0]} != n_recon {layout['n_recon']}"
            )

            # All-zero weights (e.g. density-only at warm-up factor 0) fall back to uniform.
            denom = float(raw.sum())
            if denom <= 0.0:
                denom = float(raw.shape[0])
                raw = torch.ones_like(raw)
            return (raw * (float(layout["n_recon"]) / denom)).to(device), scale

        if is_main:
            logger.info(
                f"using channel weights on {layout['n_recon']} channels  "
                f"channel_weighting={ch_weighting}  channel_weight_clip_ratio={channel_weight_clip or 'off'}  "
                f"density_channel_weight={density_channel_weight}  "
                f"gradmag_channel_weight={gradmag_channel_weight}  "
                f"density_weight_warmup_epochs={dens_warmup_epochs or 'off'}"
            )


    if is_main:
        logger.info("start density-ViT-MAE pre-training...")

    # LR schedule (not applied with Muon): 'constant' keeps cfg.lr; 'cosine' warms up linearly
    # over lr_warmup_epochs, then decays to lr_min by epoch num_epochs. Epochs count from 0, so a
    # resumed run continues the same schedule instead of restarting it.
    _lr_sched = str(cfg.get("lr_schedule", "constant"))
    _lr_warmup = int(cfg.get("lr_warmup_epochs", 0))
    _lr_min = float(cfg.get("lr_min", 0.0))
    _lr_base = float(cfg.lr)
    _lr_total = int(cfg.num_epochs)

    # LR at epoch ep under the schedule above.
    def _lr_at_epoch(ep):
        if _lr_sched == "constant":
            return _lr_base
        if _lr_warmup > 0 and ep < _lr_warmup:
            return _lr_base * float(ep + 1) / float(_lr_warmup)
        prog = float(ep - _lr_warmup) / float(max(1, _lr_total - _lr_warmup))
        prog = min(1.0, max(0.0, prog))
        return _lr_min + 0.5 * (_lr_base - _lr_min) * (1.0 + math.cos(math.pi * prog))

    # num_epochs is the run's total: a resumed run trains only the epochs it has left.
    epochs_iter = tqdm(
        range(start_epoch, cfg.num_epochs), initial=start_epoch, total=cfg.num_epochs,
        desc="Epochs", disable=not is_main,
    )
    ckpt_saver = AsyncCheckpointSaver() if is_main else None

    # Final model = lowest-validation-loss epoch. The record is a sidecar JSON, so checkpoint.pth.tar
    # stays the resume state and a resumed run cannot replace a better earlier best.
    best_json = os.path.join(cfg.output_dir, "checkpoint_best.json")
    best_val, best_epoch = math.inf, -1
    if is_main and os.path.isfile(best_json):
        try:
            _rec = json.loads(open(best_json).read())
            best_val, best_epoch = float(_rec["val_loss"]), int(_rec["epoch"])
            logger.info(f"best-val record restored: epoch {best_epoch}, val loss {best_val:.5f}")
        except Exception as e:                                       # noqa: BLE001
            logger.warning(f"could not read {best_json} ({e}); best-val tracking restarts")
    last_epoch = cfg.num_epochs - 1

    for epoch in epochs_iter:
        t0 = time.time()
        if _lr_sched != "constant" and not using_muon:
            _cur_lr = _lr_at_epoch(epoch)
            for _pg in optimizer.param_groups:
                _pg["lr"] = _cur_lr
            if is_main:
                logger.info(f"[epoch {epoch}] lr={_cur_lr:.3e} "
                            f"(schedule={_lr_sched}, warmup={_lr_warmup}ep, "
                            f"peak={_lr_base:.2e}, min={_lr_min:.1e})")
        if hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        # This epoch's channel weights; they change only during the warm-up.
        if build_ch_weight is not None:
            ch_weight, aux_scale = build_ch_weight(epoch)
            if is_main and dens_warmup_epochs > 0 and epoch <= dens_warmup_epochs:
                n_atom_ = layout["n_atom"]
                msg = f"[epoch {epoch}] density/gradmag warm-up scale={aux_scale:.4f}"
                if layout["n_density"] > 0:
                    msg += f"  eff density={float(ch_weight[n_atom_]):.4f}"
                if with_gradmag and gradmag_reconstruct:
                    msg += f"  eff gradmag={float(ch_weight[n_atom_ + layout['n_density']]):.4f}"
                logger.info(msg)
        else:
            ch_weight, aux_scale = None, 1.0

        # Mask-ratio curriculum (mae.mask_curriculum): mask_ratio_min -> max over
        # mask_curriculum_epochs, linearly or in mask_curriculum_steps equal steps, then held.
        # It replaces the per-batch R2MAE draw.
        if bool(cfg.mae.get("mask_curriculum", False)):
            _cmin = float(cfg.mae.mask_ratio_min); _cmax = float(cfg.mae.mask_ratio_max)
            _cramp = int(cfg.mae.get("mask_curriculum_epochs", None) or cfg.num_epochs)
            _csteps = cfg.mae.get("mask_curriculum_steps", None)
            _e = max(0, epoch)
            if _csteps and int(_csteps) >= 2:
                _ns = int(_csteps)
                _stage = min(_ns - 1, int(_e / (max(1, _cramp) / _ns)))
                _epoch_ratio = _cmin + (_cmax - _cmin) * (_stage / (_ns - 1))
                _shape = f"step {_stage + 1}/{_ns}"
            else:
                _prog = min(1.0, max(0.0, _e / float(max(1, _cramp - 1))))
                _epoch_ratio = _cmin + (_cmax - _cmin) * _prog
                _shape = f"linear prog={_prog:.2f}"
            _pf_min, _pf_max = None, None
            if is_main:
                logger.info(f">> [mask_curriculum] epoch {epoch}: mask_ratio={_epoch_ratio:.4f} "
                            f"({_cmin}->{_cmax} over {_cramp}ep, {_shape})")
        else:
            _epoch_ratio = cfg.mae.mask_ratio
            _pf_min = cfg.mae.get("mask_ratio_min", None)
            _pf_max = cfg.mae.get("mask_ratio_max", None)

        prefetcher = MAEPrefetcher(
            loader_train, voxelizer,
            sigma_noise=cfg.mae.sigma_noise,
            block_size=cfg.mae.block_size,
            mask_ratio=_epoch_ratio,
            input_mode=input_mode,
            mask_strategy=str(cfg.mae.get("mask_strategy", "uniform")),
            mask_atom_tau=float(cfg.mae.get("mask_atom_tau", 1.0)),
            mask_n_seeds=int(cfg.mae.get("mask_n_seeds", 4)),
            mask_ratio_min=_pf_min,
            mask_ratio_max=_pf_max,
            with_gradmag=with_gradmag,
            gradmag_reconstruct=gradmag_reconstruct,
            gradmag_noise=gradmag_noise,
            gradmag_without_density=bool(cfg.mae.get("gradmag_without_density", False)),
            n_lig_ch=int(cfg.model.get("n_channels_ligand", 7)),
            n_poc_ch=int(cfg.model.get("n_channels_pocket", 4)),
            channel_groups=(tuple(int(c) for c in cfg.model.channel_groups)
                            if cfg.model.get("channel_groups", None) else None),
            mask_channel_independent=bool(cfg.mae.get("mask_channel_independent", False)),
            mask_group_ratios=(list(cfg.mae.get("mask_group_ratios", None))
                               if cfg.mae.get("mask_group_ratios", None) else None),
            group_complete_ratio=float(cfg.mae.get("group_complete_ratio", 0.9)),
            group_complete_light_ratio=float(cfg.mae.get("group_complete_light_ratio", 0.15)),
            group_complete_family_weights=cfg.mae.get("group_complete_family_weights", None),
        )
        train_metrics, global_step = train_epoch(
            cfg, prefetcher, model_train, optimizer, model_ema, device, global_step,
            ch_weight=ch_weight, ema_source_model=model,
        )

        val_metrics = None
        if epoch % max(1, int(cfg.mae.val_every)) == 0:
            val_metrics = val_epoch(cfg, model_ema.module, val_cache, device,
                                    ch_weight=ch_weight)

        if is_main:
            log_metrics(epoch, train_metrics, val_metrics, time.time() - t0)
            if cfg.wandb and wandb.run is not None:
                log_payload = {"train": train_metrics, "epoch": epoch}
                if val_metrics is not None:
                    log_payload["val"] = val_metrics
                if dens_warmup_epochs > 0:
                    log_payload["aux_weight_scale"] = aux_scale
                try:
                    wandb.log(log_payload, step=global_step)
                except Exception as e:
                    logger.warning(f"wandb.log failed: {e}")

            # Each epoch: checkpoint.pth.tar, the resume state (EMA, encoder slice, optimizer).
            encoder_sd = {
                k: v for k, v in model_ema.module.state_dict().items()
                if k.startswith("encoder.")
            }
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "metrics": {"train": train_metrics, "val": val_metrics},
                "cfg": cfg,
                "state_dict_ema": model_ema.module.state_dict(),
                "encoder_state_dict_ema": encoder_sd,
                "optimizer": optimizer.state_dict(),
            }
            ckpt_saver.save(state, save_dir=cfg.output_dir)
            ckpt_every = max(1, int(cfg.mae.ckpt_every))

            # Tagged checkpoint every ckpt_every epochs and at the last epoch (not epoch 0, whose
            # weights are never used).
            if epoch > 0 and ((epoch % ckpt_every == 0) or (epoch == last_epoch)):
                ckpt_saver.save(state, save_dir=cfg.output_dir,
                                chkp_name=f"checkpoint_e{epoch:04d}.pth.tar")

            # New best val loss: checkpoint_best.pth.tar, plus checkpoint_best.json with the epoch.
            if val_metrics is not None and "loss" in val_metrics:
                v = float(val_metrics["loss"])
                if math.isfinite(v) and v < best_val:
                    best_val, best_epoch = v, epoch
                    ckpt_saver.save(state, save_dir=cfg.output_dir,
                                    chkp_name="checkpoint_best.pth.tar")
                    with open(best_json, "w") as fh:
                        json.dump({"epoch": epoch, "val_loss": v,
                                   "val_metrics": {k: float(x) for k, x in val_metrics.items()}},
                                  fh, indent=2)
                    logger.info(f"new best val loss {v:.5f} at epoch {epoch}")

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        torch.cuda.empty_cache()

    if is_main and ckpt_saver is not None:
        ckpt_saver.wait()
        _finalize_best_checkpoint(cfg.output_dir, best_epoch, best_val, last_epoch)
    if world_size > 1:
        dist.barrier()
    _cleanup_ddp()


# Publish the best-validation checkpoint as checkpoint_final.pth.tar and checkpoint_eNNNN.pth.tar,
# as hard links (copies where links fail) to avoid a second ~GB file. Without validation the last
# tagged checkpoint is published; checkpoint_final.json records which and why.
def _finalize_best_checkpoint(out_dir: str, best_epoch: int, best_val: float,
                              last_epoch: int) -> None:
    import shutil

    src = os.path.join(out_dir, "checkpoint_best.pth.tar")
    reason = "best validation loss"
    if best_epoch < 0 or not os.path.isfile(src):
        src = os.path.join(out_dir, f"checkpoint_e{last_epoch:04d}.pth.tar")
        reason = "last epoch (validation never ran, so there is no best)"
    if not os.path.isfile(src):
        logger.warning(f"no checkpoint to publish as final under {out_dir}")
        return
    epoch = best_epoch if best_epoch >= 0 else last_epoch

    # Hard-link src to dst, or copy where the filesystem refuses links.
    def _link(dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)

    dst = os.path.join(out_dir, "checkpoint_final.pth.tar")
    if os.path.lexists(dst):
        os.remove(dst)
    _link(dst)

    # Also as checkpoint_eNNNN, so `--epoch best` and `--epoch N` load the same file.
    numbered = os.path.join(out_dir, f"checkpoint_e{epoch:04d}.pth.tar")
    if not os.path.lexists(numbered):
        _link(numbered)
    with open(os.path.join(out_dir, "checkpoint_final.json"), "w") as fh:
        json.dump({"source": os.path.basename(src), "reason": reason, "epoch": epoch,
                   "val_loss": best_val if math.isfinite(best_val) else None}, fh, indent=2)
    logger.info(f"final checkpoint = {os.path.basename(src)} ({reason}"
                + (f", epoch {best_epoch}, val loss {best_val:.5f})" if best_epoch >= 0 else ")"))


# Refuse options the 2026-10 refactor moved to legacy/pretrain/train_density.py.
def _check_live_options(cfg: DictConfig) -> None:
    input_mode = cfg.model.get("input_mode", cfg.get("input_mode", "density"))
    moved = {
        "mae.method": cfg.mae.get("method", None) not in (None, "mae"),
        "mae.pretext_style": cfg.mae.get("pretext_style", "mae") != "mae",
        "model.arch": cfg.model.get("arch", "vit_mae") != "vit_mae",
        "mae.koleo_weight": float(cfg.mae.get("koleo_weight", 0.0)) > 0.0,
        "mae.density_input": not cfg.mae.get("density_input", True),
        "mae.struct_input": not cfg.mae.get("struct_input", True),
        "mae.ligand_input": not cfg.mae.get("ligand_input", True),
        "mae.gradmag_source": cfg.mae.get("gradmag_source", "density") != "density",
        "mae.mask_as_channel": bool(cfg.mae.get("mask_as_channel", False)),
        "mae.density_source": cfg.mae.get("density_source", "xray") != "xray",
        "mae.contrastive_weight": float(cfg.mae.get("contrastive_weight", 0.0)) > 0.0,
        "mae.data2vec_weight": float(cfg.mae.get("data2vec_weight", 0.0)) > 0.0,
        "mae.apo_prob": float(cfg.mae.get("apo_prob", 0.0)) > 0.0,
        "mae.modal_mask_prob": float(cfg.mae.get("modal_mask_prob", 0.0)) > 0.0,
        "mae.ligand_mask_prob": float(cfg.mae.get("ligand_mask_prob", 0.0)) > 0.0,
        "mae.density_visible": bool(cfg.mae.get("density_visible", False)),
        "dset.density_sources": int(cfg.dset.get("density_sources", 1)) > 1,
        "input_mode": input_mode not in _VALID_INPUT_MODES,
    }
    hits = [k for k, v in moved.items() if v]
    if hits:
        raise ValueError(f"{', '.join(hits)}: no longer in this trainer; run "
                         f"legacy/pretrain/train_density.py (see legacy/README.md)")


# Hydra entry point: refuse legacy-only options, then run.
@hydra.main(config_path="configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    _check_live_options(cfg)
    run(cfg)


if __name__ == "__main__":
    main()

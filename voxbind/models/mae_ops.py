"""mae_ops.py — shared voxel/density ops for the density-MAE pretexts.

Pure functions used across pre-training (train_density.py), VoxBind and the dataset
density pipeline: Gaussian blur, 3D block masking, per-sample z-scoring and gradient
magnitude. No model state — just tensor transforms.

(Extracted from the former density_mae.py, whose DensityMAE CNN model — superseded
by the ViT encoder in density_vit.py — was removed along with its deleted trainer.)
"""

import torch
import torch.nn.functional as F


# ── Density helpers ───────────────────────────────────────────────────────────

def gaussian_blur3d(x: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """Separable 3D Gaussian blur. sigma_vox is in voxel units (1 vox = 0.25 Å)."""
    half = max(1, int(3.0 * sigma_vox + 0.5))
    coords = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-0.5 * (coords / sigma_vox) ** 2)
    g = g / g.sum()
    kx = g.view(1, 1, -1, 1, 1)
    ky = g.view(1, 1, 1, -1, 1)
    kz = g.view(1, 1, 1, 1, -1)
    x = F.conv3d(x, kx, padding=(half, 0, 0))
    x = F.conv3d(x, ky, padding=(0, half, 0))
    x = F.conv3d(x, kz, padding=(0, 0, half))
    return x


def gradient_magnitude3d(x: torch.Tensor, spacing: float = 1.0) -> torch.Tensor:
    """Per-voxel gradient magnitude ‖∇x‖ via central finite differences.

    An edge / iso-surface detector for a density field: large where the density
    changes fastest (atom-blob shells, 2Fo-Fc iso-surfaces), complementing the
    raw density magnitude. Replicate padding preserves the G³ shape and avoids
    spurious large gradients at the boundary faces.

    `spacing` (voxel size) only rescales the output globally, so when the result
    is subsequently per-sample z-scored it is irrelevant (default 1.0 = voxel
    units, matching the rest of the density pipeline).

    Args
    ----
    x : (B, 1, G, G, G) tensor (CPU or CUDA).

    Returns
    -------
    (B, 1, G, G, G) non-negative tensor, same dtype/device as `x`.
    """
    xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    gx = (xp[:, :, 2:, 1:-1, 1:-1] - xp[:, :, :-2, 1:-1, 1:-1]) / (2.0 * spacing)
    gy = (xp[:, :, 1:-1, 2:, 1:-1] - xp[:, :, 1:-1, :-2, 1:-1]) / (2.0 * spacing)
    gz = (xp[:, :, 1:-1, 1:-1, 2:] - xp[:, :, 1:-1, 1:-1, :-2]) / (2.0 * spacing)
    return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)


def protein_attenuation_field(
    poc_occ: torch.Tensor,
    sigma_vox: float = 7.0,
    quantile: float = 0.90,
    strength: float = 1.0,
) -> torch.Tensor:
    """alpha(x) in [0,1] over the density grid, built from PROTEIN ATOMS ONLY.

    High alpha near protein atoms (preserve density), low in open space (attenuate) —
    i.e. we remove *the space a ligand could occupy*, never "the ligand". Ligand
    coordinates never enter construction, which is what makes this (a) free of any
    leakage argument (the mask shape cannot encode ligand shape, unlike
    `VoxBind.ligand_occupancy_mask`) and (b) computable at GENERATION time, where no
    ligand exists — the diagnostic condition and the deployment condition are identical.

    A Gaussian-blurred occupancy is a smooth, monotone-in-distance proxy for
    distance-to-nearest-protein-atom, and `sigma_vox` sets the transition width
    (1 vox = 0.25 A, so sigma 6-8 gives the ~1.5-2 A falloff of the map's own blur).
    A boundary sharper than the map's intrinsic resolution is physically impossible and
    the network would read it as a feature.

    Normalisation uses a high quantile of the values *at occupied voxels* rather than the
    max, so dense protein cores don't compress the rest of the field toward 0.

    Args:
        poc_occ:   (B,1,G,G,G) protein/pocket occupancy (sum over pocket atom channels).
        sigma_vox: transition width in voxels.
        quantile:  proximity value mapped to alpha = 1.
        strength:  global dose. alpha_eff = 1 - strength*(1-alpha); 0 = no-op, 1 = full.
                   Sweeping this is the performance-vs-alpha dose-response axis.

    Returns:
        (B,1,G,G,G) float in [0,1].
    """
    prox = gaussian_blur3d(poc_occ, sigma_vox)
    occupied = poc_occ > 0
    if occupied.any():
        ref = torch.quantile(prox[occupied].float(), quantile)
    else:                                        # degenerate crop: no pocket atoms
        ref = prox.amax()
    alpha = (prox / ref.clamp(min=1e-8)).clamp(0.0, 1.0)
    if strength != 1.0:
        alpha = 1.0 - float(strength) * (1.0 - alpha)
    return alpha


def attenuate_density(
    density: torch.Tensor,
    alpha: torch.Tensor,
    noise_sigma_vox: float = 7.0,
) -> torch.Tensor:
    """rho' = alpha*rho + (1-alpha)*noise — the "nothing here but water" fill.

    The noise is BLURRED to `noise_sigma_vox` and renormalised to unit variance, so its
    gradient statistics resemble real density. This is not cosmetic: VoxBind derives a
    gradmag channel from this map, and white N(0,1) noise has enormous high-frequency
    gradient — it would make that channel wildly out-of-distribution and turn "density
    removed" into "gradmag broken". Unit variance also keeps the z-scored scale the
    encoder was pretrained on.

    Prefer this over plain `alpha*rho`, which drives the region to the normalisation mean
    and reads as "unknown"; a perfectly flat region is itself a tell.
    """
    noise = torch.randn_like(density)
    if noise_sigma_vox and noise_sigma_vox > 0:
        noise = gaussian_blur3d(noise, noise_sigma_vox)
        noise = noise / noise.std().clamp(min=1e-6)
    return alpha * density + (1.0 - alpha) * noise


def make_block_mask(
    bsz: int, grid_dim: int, block_size: int, ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample a (B, 1, G, G, G) bool mask where masked voxels = True.

    Mask is generated at (G/block)³ resolution then upsampled by repetition,
    so masked regions are cubes of `block_size`³ voxels.
    """
    assert grid_dim % block_size == 0, f"grid_dim {grid_dim} % block_size {block_size} != 0"
    gb = grid_dim // block_size
    blocks = torch.rand(bsz, 1, gb, gb, gb, device=device) < ratio   # (B,1,gb,gb,gb)
    mask = blocks.repeat_interleave(block_size, dim=2)\
                 .repeat_interleave(block_size, dim=3)\
                 .repeat_interleave(block_size, dim=4)
    return mask  # (B, 1, G, G, G) bool


def make_atom_biased_block_mask(
    atoms_sum: torch.Tensor,
    block_size: int,
    ratio: float,
    tau: float = 1.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample a (B, 1, G, G, G) bool mask biased toward atom-occupied blocks.

    `atoms_sum` is (B, 1, G, G, G) — sum of all atom-voxel channels. For each
    sample we score each block by its atom-mass via avg_pool3d, add Gaussian
    noise scaled by `tau · std(scores)`, and take the top floor(ratio · gb³)
    blocks. Mask cardinality is exact per sample.

      tau → 0   : deterministic top-K (every most-atom-occupied block masked first)
      tau → ∞   : equivalent to uniform random masking (noise dominates)
      tau = 1   : moderate bias — atom-occupied blocks dominate but training
                  still sees some empty-space variety.
    """
    B, _, G, _, _ = atoms_sum.shape
    assert G % block_size == 0, f"grid_dim {G} % block_size {block_size} != 0"
    gb = G // block_size
    n_blocks = gb ** 3
    n_mask = int(round(ratio * n_blocks))
    # Per-block atom mass: avg_pool3d gives mean; × block³ = sum over the block.
    score = F.avg_pool3d(atoms_sum, block_size) * (block_size ** 3)  # (B, 1, gb, gb, gb)
    score = score.flatten(2)                                        # (B, 1, gb³)
    std = score.std(dim=2, keepdim=True).clamp(min=1e-6)
    if generator is not None:
        noise = torch.randn(score.shape, device=score.device, generator=generator)
    else:
        noise = torch.randn_like(score)
    priority = score + noise * std * tau
    _, top_idx = priority.topk(n_mask, dim=2)
    blocks_flat = torch.zeros_like(score, dtype=torch.bool)
    blocks_flat.scatter_(2, top_idx, True)
    blocks = blocks_flat.view(B, 1, gb, gb, gb)
    mask = blocks.repeat_interleave(block_size, dim=2)\
                 .repeat_interleave(block_size, dim=3)\
                 .repeat_interleave(block_size, dim=4)
    return mask  # (B, 1, G, G, G) bool


def make_cluster_mask(
    atoms_sum: torch.Tensor,
    block_size: int,
    ratio: float,
    n_seeds: int = 4,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample a (B,1,G,G,G) bool mask as a few LARGE CONTIGUOUS atom-anchored clusters.

    A dramatic alternative to scattered block masking (make_block_mask / atom_biased,
    which both drop many small independent blocks). For each sample we pick `n_seeds`
    seed blocks (∝ atom mass, like atom_biased), then mask the floor(ratio·gb³) blocks
    that are CLOSEST (min Euclidean block-grid distance) to any seed. That yields
    `n_seeds` large contiguous holes centred on chemistry, so the encoder must inpaint
    whole substructures from long-range context instead of filling many tiny gaps —
    the MAE "large contiguous masks force semantic inference" regime, but atom-anchored.
    Mask cardinality is exact per sample. Reduces to a single big ball when n_seeds=1.
    """
    B, _, G, _, _ = atoms_sum.shape
    assert G % block_size == 0, f"grid_dim {G} % block_size {block_size} != 0"
    gb = G // block_size
    n_blocks = gb ** 3
    n_mask = max(1, int(round(ratio * n_blocks)))
    k = max(1, min(int(n_seeds), n_blocks))
    dev = atoms_sum.device
    # per-block atom mass (≥0) → seed-sampling weights, biased toward atoms
    w = (F.avg_pool3d(atoms_sum, block_size) * (block_size ** 3)).reshape(B, n_blocks)
    w = w.clamp_min(0) + 1e-6
    # integer block-grid coords (n_blocks, 3)
    rng = torch.arange(gb, device=dev)
    coords = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1)\
                  .reshape(-1, 3).float()
    mask_flat = torch.zeros(B, n_blocks, dtype=torch.bool, device=dev)
    for b in range(B):
        seeds = torch.multinomial(w[b], k, replacement=False, generator=generator)
        dist = torch.cdist(coords, coords[seeds]).amin(dim=1)        # (n_blocks,) → nearest seed
        sel = dist.topk(n_mask, largest=False).indices
        mask_flat[b, sel] = True
    mask = mask_flat.view(B, 1, gb, gb, gb)
    return mask.repeat_interleave(block_size, dim=2)\
               .repeat_interleave(block_size, dim=3)\
               .repeat_interleave(block_size, dim=4)  # (B, 1, G, G, G) bool


def make_interface_mask(
    lig_sum: torch.Tensor,
    poc_sum: torch.Tensor,
    block_size: int,
    ratio: float,
    tau: float = 0.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Sample a (B,1,G,G,G) bool mask over the ligand–pocket INTERFACE (contact region).

    The interface is where binding affinity originates: voxels where ligand and pocket
    atoms are adjacent. We score each block by how much it has BOTH ligand and pocket
    mass nearby — block-level product of (dilated) ligand and pocket occupancy — and mask
    the floor(ratio·gb³) highest-scoring blocks. The encoder then reconstructs the contact
    interface from its surroundings → learns binding-interaction structure. `tau`>0 adds
    Gaussian tie-break noise (0 = deterministic top-K interface). Exact mask cardinality.
    """
    B, _, G, _, _ = lig_sum.shape
    assert G % block_size == 0, f"grid_dim {G} % block_size {block_size} != 0"
    gb = G // block_size
    n_blocks = gb ** 3
    n_mask = max(1, int(round(ratio * n_blocks)))
    # block-resolution occupancy, dilated by 1 block so adjacency (not overlap) counts
    lb = F.max_pool3d(F.avg_pool3d(lig_sum, block_size), kernel_size=3, stride=1, padding=1)
    pb = F.max_pool3d(F.avg_pool3d(poc_sum, block_size), kernel_size=3, stride=1, padding=1)
    score = (lb * pb).flatten(2)                                    # (B,1,gb³) high at the interface
    if tau > 0:
        std = score.std(dim=2, keepdim=True).clamp(min=1e-6)
        noise = (torch.randn(score.shape, device=score.device, generator=generator)
                 if generator is not None else torch.randn_like(score))
        score = score + noise * std * tau
    _, top_idx = score.topk(n_mask, dim=2)
    blocks_flat = torch.zeros_like(score, dtype=torch.bool)
    blocks_flat.scatter_(2, top_idx, True)
    blocks = blocks_flat.view(B, 1, gb, gb, gb)
    return blocks.repeat_interleave(block_size, dim=2)\
                 .repeat_interleave(block_size, dim=3)\
                 .repeat_interleave(block_size, dim=4)  # (B, 1, G, G, G) bool


def make_interface_atom_mask(
    lig_sum: torch.Tensor,
    poc_sum: torch.Tensor,
    block_size: int,
    ratio: float,
    tau: float = 0.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """PRIORITY masking: INTERFACE blocks first → ATOM-occupied blocks → EMPTY space.

    make_interface_mask masks the top-K interface blocks, but once the (few) true contact
    blocks are exhausted its tie-break fill lands on ARBITRARY zero-score blocks (mixing
    atom-occupied and empty at random). Here the top-K spills over in a MEANINGFUL order:
    strongest ligand–pocket contacts, then the rest of the atom-occupied region (densest
    first, atom_biased-style), and only then EMPTY solvent. So at any `ratio` the most
    binding-relevant components are always masked first, with empty space included last
    (which the caller asked for — reconstructing solvent context can still help).

    Strict tiering via large integer bases (+20 interface / +10 atom / +0 empty) so the
    within-tier score and the `tau` tie-break jitter never cross a tier boundary.
    Returns a (B,1,G,G,G) bool mask (broadcast across channels), exact cardinality.
    """
    B, _, G, _, _ = lig_sum.shape
    assert G % block_size == 0, f"grid_dim {G} % block_size {block_size} != 0"
    gb = G // block_size
    n_blocks = gb ** 3
    n_mask = max(1, int(round(ratio * n_blocks)))
    # DILATED occupancy (adjacency) for the interface score — ligand & pocket atoms are
    # adjacent, not overlapping, so the contact needs a 1-block dilation to register.
    lb_d = F.max_pool3d(F.avg_pool3d(lig_sum, block_size), kernel_size=3, stride=1, padding=1).flatten(2)
    pb_d = F.max_pool3d(F.avg_pool3d(poc_sum, block_size), kernel_size=3, stride=1, padding=1).flatten(2)
    interface = lb_d * pb_d                                      # (B,1,gb³) high at contact (incl. the gap)
    # RAW (non-dilated) occupancy for the atom tier — tier-1 = blocks that ACTUALLY contain
    # ligand/pocket atoms (not the adjacent ring), so empty solvent stays strictly tier-0.
    atom = (F.avg_pool3d(lig_sum, block_size) + F.avg_pool3d(poc_sum, block_size)).flatten(2)

    def _norm01(x):                                            # within-tier rank in [0,1)
        lo = x.amin(dim=2, keepdim=True)
        rng = (x.amax(dim=2, keepdim=True) - lo).clamp(min=1e-6)
        return (x - lo) / rng

    has_if = (interface > 0).to(interface.dtype)
    has_atom = (atom > 0).to(atom.dtype)
    rnd = (torch.rand(interface.shape, device=interface.device, generator=generator)
           if generator is not None else torch.rand_like(interface))   # empty-tier order
    score = has_if * (20.0 + _norm01(interface)) \
        + (1.0 - has_if) * has_atom * (10.0 + _norm01(atom)) \
        + (1.0 - has_atom) * rnd                                # empty tier ∈ [0,1)
    if tau > 0:                                                 # small in-tier jitter (< tier gap)
        noise = (torch.randn(score.shape, device=score.device, generator=generator)
                 if generator is not None else torch.randn_like(score))
        score = score + noise.clamp(-3, 3) * tau * 0.1
    _, top_idx = score.topk(n_mask, dim=2)
    blocks_flat = torch.zeros_like(score, dtype=torch.bool)
    blocks_flat.scatter_(2, top_idx, True)
    blocks = blocks_flat.view(B, 1, gb, gb, gb)
    return blocks.repeat_interleave(block_size, dim=2)\
                 .repeat_interleave(block_size, dim=3)\
                 .repeat_interleave(block_size, dim=4)  # (B, 1, G, G, G) bool


def per_sample_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score each (C, G, G, G) volume independently along spatial dims."""
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    std = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=eps)
    return (x - mu) / std

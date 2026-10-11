"""CPU regression test for the density fusion modes.

Runs without CUDA and without a pretrained checkpoint (tiny random encoder, G=32), so it
is cheap enough to run before any launch that touches the generator. It guards three
things the 2026-10-01 fusion rename could have broken:

  1. every mode in FUSION_MODES builds and does a forward pass,
  2. the legacy config/checkpoint names still resolve to their canonical mode,
  3. the zero-init discipline: at step 0 the density branch is INERT, so two different
     density maps give a bit-identical output. `encoder_replace` is the documented
     exception -- it fuses through a normal-init context_proj, which is exactly why it
     overfit and was dropped -- so it is asserted to be active, not inert.

    python task2-ligand_generation/test/test_fusion_modes.py        # or: pytest task2-ligand_generation/test/test_fusion_modes.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from voxbind.models.voxbind import (  # noqa: E402
    FUSION_ALIASES,
    FUSION_MODES,
    VoxBind,
    canonical_fusion,
)

GRID = 32
# Modes whose density branch is zero-initialised, i.e. inert at step 0.
ZERO_INIT_MODES = ("proj_sum", "proj_pocket", "token_concat", "cross_attn")

_BASE = dict(
    n_channels_ligand=7, n_channels_pocket=4, n_channels=32,
    ch_mults=[1, 2, 2, 4], is_attn=[False, False, True, True], n_blocks=1,
    n_groups=16, dropout=0.0, smooth_sigma=0.9,
    with_density=True, density_encoder_type="vit",
    density_grid_dim=GRID, density_vit_patch=8,
    density_vit_dim=64, density_vit_depth=2, density_vit_heads=4,
    density_vit_mlp_ratio=2, density_vit_dropout=0.0,
    density_vit_n_in_channels=13,
    density_vit_patch_embed_mode="channel_group",
    density_vit_channel_groups=[7, 4, 2],
)


def _build(mode: str) -> VoxBind:
    torch.manual_seed(0)
    return VoxBind(fusion=mode, **_BASE).eval()


def _inputs():
    torch.manual_seed(0)
    lig = torch.randn(2, 7, GRID, GRID, GRID)
    poc = torch.rand(2, 4, GRID, GRID, GRID)
    # Two deliberately different densities: different mean AND different scale.
    d1 = torch.rand(2, 1, GRID, GRID, GRID)
    d2 = torch.rand(2, 1, GRID, GRID, GRID) * 3.0 + 1.0
    return lig, poc, d1, d2


def test_aliases_resolve():
    for legacy, canonical in FUSION_ALIASES.items():
        assert canonical_fusion(legacy) == canonical, legacy
        assert canonical in FUSION_MODES, canonical
    for mode in FUSION_MODES:
        assert canonical_fusion(mode) == mode, mode


def test_unknown_fusion_rejected():
    # A typo must raise, never fall through to the default branch and quietly train the
    # wrong experiment.
    for bad in ("protien_first", "v5", "", "Default"):
        try:
            canonical_fusion(bad)
        except ValueError:
            continue
        raise AssertionError(f"fusion={bad!r} was accepted")


def test_every_mode_forwards():
    lig, poc, d1, _ = _inputs()
    for mode in FUSION_MODES:
        model = _build(mode)
        assert model.fusion == mode
        with torch.no_grad():
            out = model(lig, poc, density=d1)
        assert out.shape == (2, 7, GRID, GRID, GRID), (mode, out.shape)
        assert torch.isfinite(out).all(), mode


def test_legacy_names_build_the_canonical_model():
    lig, poc, d1, _ = _inputs()
    for legacy, canonical in FUSION_ALIASES.items():
        model = _build(legacy)
        assert model.fusion == canonical, (legacy, model.fusion)
        with torch.no_grad():
            model(lig, poc, density=d1)


def test_zero_init_density_is_inert_at_step_zero():
    lig, poc, d1, d2 = _inputs()
    for mode in ZERO_INIT_MODES:
        model = _build(mode)
        with torch.no_grad():
            delta = (model(lig, poc, density=d1) - model(lig, poc, density=d2)).abs().max().item()
        assert delta == 0.0, f"{mode}: density is not inert at init (max delta {delta:.3e})"


def test_encoder_replace_is_active_at_step_zero():
    # Documented exception: a normal-init context_proj, not zero-init.
    lig, poc, d1, d2 = _inputs()
    model = _build("encoder_replace")
    with torch.no_grad():
        delta = (model(lig, poc, density=d1) - model(lig, poc, density=d2)).abs().max().item()
    assert delta > 0.0, "encoder_replace should fuse density from step 0"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'OK' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)

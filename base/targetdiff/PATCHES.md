# Changes to upstream TargetDiff

This folder is [guanjq/targetdiff](https://github.com/guanjq/targetdiff) at commit
`142f1eb7178480d435fe0b8cb95a99beb48997c7` (MIT, `LICIENCE`), code only: `configs/`, `datasets/`,
`models/`, `scripts/`, `utils/`. Our glue is in `_edrscc/`; the pretrained checkpoints go in
`pretrained_models/` (not in git, see `../scripts/targetdiff.sh`). Two changes to the upstream code:

- `utils/misc_prop.py`: `PropPredNetEnc` is built with `output_dim=3` instead of 1. Its forward
  selects the output by the complex's measurement type (`F.one_hot(kind - 1, output_dim)`, kind 1/2/3
  = Ki/Kd/IC50), which fails for Kd and IC50 with one output. With three, the model with TargetDiff
  features has the same per-type heads as the plain EGNN (`PropPredNet`, `output_dim=3`).
- `datasets/protein_ligand.py`, `utils/data.py`: `dtype=np.long`, `np.int` and `np.bool`, which
  NumPy 1.24 removed, become the builtins `int` and `bool` they aliased.

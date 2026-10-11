# TargetDiff evaluation code (vendored)

`utils/evaluation/`, `utils/reconstruct.py` and `utils/data.py` are copied unchanged from
[TargetDiff](https://github.com/guanjq/targetdiff) (Guan et al., ICLR 2023), commit `43ad802`,
through the fork [seonghyun26/TargetDIff](https://github.com/seonghyun26/TargetDIff) at `d0fc5b9`.
The fork changes two files for current numpy and RDKit only: `utils/data.py` (`np.long`,
`np.bool`, `np.int` → `int`, `bool`) and `utils/evaluation/sascorer.py` (drops `rdkit.six`).

The files keep their own package names (`utils.evaluation.*`, `utils.reconstruct`), so
`evaluation/__init__.py` puts this folder on `sys.path`.

Licenses: TargetDiff is MIT (`LICENSE`). `utils/reconstruct.py` is TargetDiff's copy of
[liGAN](https://github.com/mattragoza/liGAN)'s `fitting.py` and stays GPL-2.0, as its header says.

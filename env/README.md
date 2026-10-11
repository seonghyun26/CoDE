# Environments

Two environments run the pipeline, one for training and one for evaluation:

| Env | Python | Runs | Spec | Lock |
|---|---|---|---|---|
| `voxbind` | 3.10 | pretraining, affinity probes, ligand generation | `voxbind.environment.yml` | `voxbind.conda-lock.yml` |
| `voxeval` | 3.10 | QED/SA/diversity, AutoDock Vina 1.2.2, PoseCheck 1.3.1, PoseBusters 0.6.5 | `voxeval.environment.yml` | `voxeval.conda-linux-64.lock` + `voxeval.pip.lock.txt` |

`bash script/00_setup_env.sh` builds both, and `env/Dockerfile` builds an image that holds both. The
locks are rendered for linux-64.

## voxeval

Every generated-ligand metric runs in this one interpreter. Three constraints shape it:

- PoseBusters 0.6.5 needs RDKit 2024.9 or newer.
- Vina 1.2.2, the docking version of the paper, has no wheel, and conda-forge's Python 3.10 build needs
  Boost 1.74, which no conda RDKit from 2024.9 on can share.
- So RDKit comes from PyPI, whose wheel carries its own Boost, and Vina 1.2.2 is built from its sdist
  against conda-forge's boost-cpp 1.74. The build writes the `vina/version.py` that Vina's release
  workflow writes and uses setuptools < 66, which still accepts the sdist's `python_requires`; the
  source is unchanged. `build_voxeval` in `script/00_setup_env.sh` does this.

On 2026-10-09 it was checked against the two environments it replaced (voxdock for Vina and QED/SA,
moleval for the pose tools) on 18 molecules, AR samples and crystal ligands in three pockets. Vina
score and minimize, QED, PoseBusters and PoseCheck's clashes and interactions were identical. SA
matched voxbind's RDKit 2024.03.6, where the paper's QED and SA come from, on all 18, where voxdock
differed on one. Vina dock differed by at most 0.044, within its run-to-run noise. PoseCheck strain
had the same median; one molecule moved from 23.4 to 35.8 with the RDKit version.

## Rebuild

```bash
bash script/00_setup_env.sh                    # voxbind and voxeval, whichever is missing
FORCE=1 bash script/00_setup_env.sh voxeval    # rebuild one

# voxbind by hand (needs conda-lock)
conda-lock install -n voxbind env/voxbind.conda-lock.yml
conda run -n voxbind pip install -e .          # from the repo root
```

To change voxbind, edit `voxbind.environment.yml` and re-lock:

```bash
conda-lock lock -f env/voxbind.environment.yml -p linux-64 --lockfile env/voxbind.conda-lock.yml
```

To change voxeval, edit `voxeval.environment.yml`, rebuild, check the evaluation against the old one,
and export the pair again (`micromamba env export --explicit --md5` for the conda layer, the PyPI
packages minus vina, AutoDockTools_py3 and posecheck for the pip layer).

## The environments voxeval replaced

`voxdock` (Vina 1.2.2 on Python 3.8, `voxdock.conda-lock.yml`) and `moleval` (the pose tools) produced
the paper's evaluation; build them by name (`bash script/00_setup_env.sh voxdock moleval`) to rerun
an old evaluation exactly. FuncBind's image (`script/Dockerfile`, `voxbind:allinone`) and its MCP
docking still use them.

## `voxel-bind` — one env for VoxBind *and* FuncBind

Optional, not part of the paper pipeline, not built by default. It exists so FuncBind's
density-conditioning branch can `import voxbind.models.density_vit` in its own process
instead of reaching across two interpreters. **`import funcbind.train_fb` succeeds here
with no changes to the FuncBind source.** Build it by name:

```bash
# on a box whose default envs dir survives a restart
bash script/00_setup_env.sh voxel-bind

# here, /opt/conda/envs is a container overlay that a restart wipes — install onto the PVC
VOXELBIND_PREFIX=$HOME/.conda/envs/voxel-bind bash script/00_setup_env.sh voxel-bind
```

Locked as an **explicit pair**, not a `conda-lock.yml`: most of the stack (torch, rdkit,
scipy) is pip in this env, so the byte-exact pair needs no conda-lock tool at all.

```bash
conda create -p $HOME/.conda/envs/voxel-bind --file env/voxel-bind.conda-linux-64.lock
ENV=$HOME/.conda/envs/voxel-bind
PATH="$ENV/bin:$PATH" CC="$ENV/bin/gcc" $ENV/bin/pip install -r env/voxel-bind.pip.lock.txt
$ENV/bin/pip install --no-deps posecheck==1.3.1
$ENV/bin/pip install -e .
```

Roughly 7.8 GB and ~225 pip packages, against 7.1 GB / 117 for the VoxBind side alone.

- **Docking here is 1.2.7 and must not be used.** Python 3.12 cannot have vina 1.2.2 by
  any route — no wheel past cp39, the PyPI sdist fails to build (`Could not find version
  string for AutoDock Vina`), and conda-forge's oldest py312 build is 1.2.7. The 1.2.7 in
  this env exists only so `import vina` works in FuncBind's import chain. Dock in `voxeval`
  (vina 1.2.2), which `script/05_evaluate.sh` uses and checks; FuncBind's
  `scripts/chain_dock_mcp_run.sh` still calls `/opt/conda/envs/voxdock/bin/python`.
- **It is not the paper env.** `voxbind` above is py3.10 / cu118 / numpy 1.26 and is where
  the published numbers come from. `voxel-bind` is the stack the FuncBind smoke test
  (`test/mcp_default_fusion_smoke.py`, 11/11) was validated on: python 3.12, torch 2.5.1 + cu124
  wheels, numpy 2.4.6, rdkit 2025.3.6.
- **`CC` is load-bearing.** `cpdb-protein` is source-only on PyPI and FuncBind's MCP
  sampling calls it at runtime (`save_sdf_pdb` → `extract_sequences_from_pdb`), so it
  cannot be skipped. The compiler comes from the conda layer (`c-compiler`); this box has
  no system `cc`.
- **PyRosetta is not installed here, on purpose.** FuncBind reaches it only through
  `metrics_ab` → `utils_rosetta.interface_energy` — the antibody metrics — and that import
  is lazy since 2026-09-14, so MCP and CrossDocked work never touches it. Leaving it out
  saves 4.2 GB and keeps a licence-gated package out of shared images. For antibody runs:
  `mamba install -p $HOME/.conda/envs/voxel-bind -c https://conda.graylab.jhu.edu pyrosetta`
  — that channel is reachable and carries py312 builds, while the wheel host in FuncBind's
  pyproject (`graylab.jhu.edu`) is not. **Commercial use requires a licence.**
- **numpy is conda-owned here.** A conda numpy arrived once as a dependency and replaced the
  pip numpy 2.4.6 with 2.5.3 *while pip's metadata still read 2.4.6* — a mismatch
  `importlib.metadata` cannot see. numpy is therefore pinned in the conda lock and kept out
  of the pip lock, so exactly one tool owns it. Do not `pip install numpy` into this env.
- **`mamba run -n voxel-bind` does not work here, `conda run -n` does.** mamba 2.x resolves
  a name only against `$MAMBA_ROOT_PREFIX/envs` (= `/opt/conda/envs`), so an env under
  `$HOME/.conda/envs` is nameless to it. Prefer the absolute interpreter,
  `$HOME/.conda/envs/voxel-bind/bin/python`.


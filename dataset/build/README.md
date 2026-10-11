# Unified dataset-build pipeline (`dataset/build/`)

**One file per pipeline *stage***, each with a `--dataset {crossdocked,pdbbind,misato,plinder}`
if-branch — instead of one script per dataset. Replaces the per-dataset sprawl
(`01a/01b`, `02a–d`, `01`–`05`) for PDBbind / MISATO / PLINDER.

> **CrossDocked stages live in `dataset/crossdocked/`** (`preprocess_crossdocked.py`, `00a`,
> `00b`, `00j`); the loaders `crossdocked.py` and `crossdocked_xray.py` stay at `dataset/`. The
> `crossdocked` branch here **delegates** to those scripts. Retired builders (the noise and v7
> ablations, `00c`–`00i`) are in `legacy/dataset/`.

## Decisions (260616, user-confirmed)
- **Scope:** all four datasets unified behind these stages.
- **Voxelize = hybrid:** dense voxels precomputed to disk **only for the downstream probe
  sets** (PDBbind, MISATO). Pretraining (CrossDocked, PLINDER) keeps the compact
  `(pocket,ligand)` point-cloud tuples + **on-GPU voxelization at train time** — that point
  cloud *is* the precompute; dense voxels for 78k CrossDocked would be ~450 GB and would kill
  rotation augmentation.
- **Layout (2026-10-02):** every delegated script lives in its dataset's folder:
  `dataset/crossdocked/`, `dataset/pdbbind/`, `dataset/misato/`, `dataset/plinder/`.
  `registry._script_path()` searches those directories in a fixed order and raises if a script is
  missing. Helpers for external benchmark sets are in `dataset/external/`, loader smoke tests in
  `dataset/test/`, and retired download/processing code and data in the repo-root `legacy/`
  folder, which mirrors the original paths and is not in git. The affinity probes that used to
  sit here moved to `task1-affinity_regression/` on 2026-10-06.

## Stages
| stage | file | does (per-dataset branch) |
|---|---|---|
| 1 | `s1_acquire.py` | download/locate originals; select+dedup+leakage (PLINDER); element→channel map + **mask** unsupported atoms (keep complex); size filter; **split assign**; emit `(pocket,ligand)` tuples + manifest |
| 2 | `s2_density.py` | download CCP4/EDS for **matched** entries; per-dataset **match gate**; resumable + stream-and-discard |
| 3 | `s3_crop.py` | **align** → crop box at ligand centroid → **normalize** (canonical arcsinh+z) → **availability gate** → gradmag; emit crops + `*_available.npy` + stats.json |
| 4 | `s4_voxelize.py` | **hybrid**: precompute dense voxels for probe sets; no-op (point clouds) for pretraining |
| 5 | `s5_targets.py` | downstream labels: PDBbind pK · MISATO QM/MD · B-factors |
| — | `registry.py` | per-dataset config (below) + the VoxBind atom dictionary |

## Per-dataset config (`registry.py`)
| dataset | frame → align | density match gate | split source | stage-4 | targets |
|---|---|---|---|---|---|
| **crossdocked** | docking → crystal (**Kabsch**) | `native_filter tt_min` + density-at-atoms | val-from-train-tail | on-the-fly | none (SSL) · **delegates to `00*`** |
| **pdbbind** | deposited (identity) | inherent holo (PDBe-EDS coverage) | `lp_edrscc_v2` 3850/817/1320 (CL3: 1559/410/733), built by `voxbind/splits/make_splits.py` outside the stages | **precompute** | pK from `index.csv` |
| **misato** | deposited (identity) | built structures + QM/MD coverage | MISATO 8:1:1 | **precompute** | QM/MD from hdf5 |
| **plinder** | deposited (identity) | **RSCC ≥ 0.8** + 0 unresolved heavy | val-from-train-tail | on-the-fly | none (SSL) |

## Atom-blob radii (stage 4 + train-time voxelizer)
Atom channels rasterize as blobs whose radius is set **independently for ligand and pocket**,
each with two modes (already supported in `crossdocked_xray.py.__getitem__` via the
`*_radius` / `*_radius_lut` branches):
- **uniform** — one scalar radius for every atom (`ligand_radius` / `pocket_radius` = `R > 0`, e.g. 0.5 Å)
- **vdW** — per-element van der Waals radius via LUT (`ligand_radius` / `pocket_radius` = `-1`)

→ four combinations from two registry knobs: `ligand_radius ∈ {R, vdW}` × `pocket_radius ∈ {R, vdW}`.
The **same** knobs drive the on-GPU train-time voxelizer (pretraining) **and** the precomputed
probe voxels (stage 4) — they must match or frozen features won't line up.

> **Default = vdW for both** (`ligand_radius = pocket_radius = -1`, "ligvdw"). Ligand and pocket
> are separate channel groups, so the two modes can differ without leaking anything; the merged
> lig+poc input, where a uniform/vdW mix leaked identity by blob size, was retired on 2026-10-06.

## Run
```
# each stage takes --dataset {crossdocked,pdbbind,misato,plinder}; add --dry_run to preview
python dataset/build/s1_acquire.py  --dataset pdbbind        # 01a structures + index
python dataset/build/s2_density.py  --dataset pdbbind        # 01a density (PDBe-EDS)
python dataset/build/s4_voxelize.py --dataset pdbbind        # 01b voxelize+poolnorm (crop fused in; s3 is a no-op)
python dataset/build/s5_targets.py  --dataset pdbbind        # pK is in index.csv; bfactors via extract_bfactors
```

## Wiring notes (current: branches delegate to the NN_ scripts via subprocess)
- **PDBbind** — s1→`01a structures+index`, s2→`01a density`, s4→`01b voxelize+poolnorm` (crop fused), s5: pK lives in `index.csv`.
- **PLINDER** — s1 `01_select --version v2` (compared with the frozen pin, never replacing it), s2 `03_download --what both` (cif + ccp4 into the shared caches), s3 `04_build_corpus` (per-element OTF corpus), `05_make_boxes` (the `box116.dat` density box the loaders read; it refuses to overwrite an existing box without `--force`) and `02_make_clean_corpus` (`CoDE_pretrain_v1`: the proteins of the paper's test sets removed, the six affinity sets and the CrossDocked test pockets). The parser uses gemmi, so the dispatcher sets `LD_LIBRARY_PATH=<interp>/lib` automatically (CXXABI). The v1 path (`03b` download, `03c` crop builder) is retired: `03b` is in `legacy/`, and `03c` stays only because `04_build_corpus` imports its parser.
- **MISATO** — **dual-env**: `02a` (acquire) & `02d` (voxelize+feature-extract, *fused*) run under voxbind; **`02b` targets run under an h5py interpreter** (`MISATO_H5PY_PY`, default cellmaes). PDBbind-missing complexes must first be built via `02c_misato_structures.py build --ids <list>` (h5py) — noted in s3, not auto-run (needs an id list).
- **CrossDocked** — delegated to `crossdocked/`: `preprocess_crossdocked`, `00a`, `00b`.
- Passthrough forwards to **every** step in a stage → for multi-step stages use only common flags; step-specific flags (e.g. `--ligand_vdw`) are registry-driven.

## Inlining a branch (optional, not done)
Every branch currently shells out to the per-dataset script, so it runs the exact code path
that produced the corpora on disk and parity is automatic. Replacing a `("run", ...)` step with
inline Python would remove a subprocess and nothing else, and it costs a numerical-equivalence
run for that dataset first: regenerate a handful of samples through the new code and assert the
crops, tuples and voxels match the current outputs. Delegation is the supported state.

## Moving an existing checkout to the current layout

Two moves changed the folders: on 2026-10-01 the package moved from `voxbind/dataset/` to
`dataset/`, and on 2026-10-06 `voxbind/` was split into the shared library, `pretrain/exps/`,
`task1-affinity_regression/`, `task2-ligand_generation/` and `model_zoo/`. Git carries the code
but not what it ignores (data, runs, weights, logs), so on any machine other than this one those
stay behind. After pulling, run from the repo root:

```bash
python dataset/build/migrate_layout.py            # dry run: what would change
python dataset/build/migrate_layout.py --apply
```

For 2026-10-01 it moves the data folder, rewrites links into the old location as relative links
(including the ones under `base/`), and rewrites the old absolute prefix in the data's text
records, such as `ccp4_dir` in each corpus's `resample.json`. For 2026-10-06 it sends each run in
`voxbind/exps/` to `pretrain/exps/` (encoder pretraining) or `task2-ligand_generation/exps/`
(everything else), merges what git left in the old folders into the new ones, sends
`voxbind/log/` to `log/`, repairs symlinks, and points the generator runs' encoder paths at
their new place. It never overwrites a file; a name taken in both places is reported and left.
With `--apply` it first writes a TSV of the old link targets and a tar.gz of every text record it
rewrites to the repo root.

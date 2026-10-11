# `voxbind/splits/` — frozen, version-controlled data splits

**Problem this solves.** The PDBbind affinity split used to be recomputed on every server from
server-local files (which density crops built, which wwPDB validation reports downloaded), so a
few failed downloads gave a slightly different test set and slightly different metrics. A split
is now a committed `pid,split` manifest, frozen once on a complete machine, so `git pull` gives
every server the identical member set. Local availability is a separate, loud check.

## The six test sets

The paper reports affinity on six test sets. They nest: CL3-ID30 ⊂ CL3-ID60 ⊂ CL3 ⊂ FULL, and
CASF-clean ⊂ CASF-nontrain; their union is 1,357 complexes.

| Set | n | File | Definition |
| --- | --- | --- | --- |
| FULL | 1320 | `lp_edrscc_v2.csv` (test) | LP-PDBBind `new_split` ∩ density available ∩ ligand & pocket RSCC ≥ 0.8 ∩ Kd/Ki, non-covalent |
| CL3 | 733 | `lp_edrscc_v2_cl123.csv` (test) | FULL restricted to LP-PDBBind cleaning levels CL1-3 |
| CL3-ID60 | 454 | `cl123_test_novel60_cohort.csv` | CL3 test complexes whose protein is below 60% sequence identity to every CL3 training protein |
| CL3-ID30 | 262 | `cl123_test_novel30_cohort.csv` | the same below 30% |
| CASF-nontrain | 124 | `casf2016_nontrain_cohort.csv` | CASF-2016 complexes not in the `lp_edrscc_v2` train set |
| CASF-clean | 92 | `casf2016_clean_cohort.csv` | CASF-2016 complexes in neither train nor val |

`task1-affinity_regression/evals/probe_common_6set.py` builds the same sets in code, and
`dataset/plinder/02_make_clean_corpus.py` removes their proteins from the pretraining corpus.
`casf2016_eval.csv` carries the 214 evaluable CASF-2016 complexes with pK and an `in_v2train`
flag; `casf2016_clean.csv` and `casf2016_testonly_split.csv` are the CASF-clean and CASF test
lists in the forms the probe tools and baselines read.

## Layout

```
voxbind/splits/
  __init__.py        load_split() · check_local_availability() · content_hash()
  make_splits.py     the only place a split is computed (reads primary inputs)
  verify.py          network-free hash gate: run before any train/probe
  MANIFEST.json      per-scheme counts + sha256 + provenance (the pin)
  <scheme>.csv       pid,split  (the frozen membership)
  plinder/           frozen PLINDER pretraining selections (plinder/README.md)
```

`MANIFEST.json` pins two schemes, `lp_edrscc_v2` (3850 / 817 / 1320) and `lp_edrscc_v2_cl123`
(1559 / 410 / 733), and their test-anchored variants `<scheme>_ID30` / `_ID60`: the train complexes
whose protein has ATOM3D identity ≥ 30 / 60% to a CL3 test protein are removed
(`atom3d_identity.py`), val and test unchanged. Train: lp_edrscc_v2 3324 / 3655, lp_edrscc_v2_cl123
1279 / 1430. The other schemes (`lp_edrscc_v1`, the CL1 and CL12 tiers, `time_v1`,
`misato_md_v1`, `clean_ed_v1`, ATOM3D LBA30/60, local LBA30/60, holdout 2019) were retired on
2026-10-05; their manifests, the full `MANIFEST.json` and the builder that made them are in
`legacy/voxbind/splits/`. To use one, copy its CSV back and its entry into `MANIFEST.json`.

## Usage

```python
from voxbind.splits import load_split, check_local_availability

split = load_split("lp_edrscc_v2")         # {"train":[pids], "val":[...], "test":[...]}
                                           # raises loudly if the manifest drifted
avail = check_local_availability(split["test"], present=have_features,
                                 label="lp_edrscc_v2/test")
# avail["used"] = pids to score; avail["content_hash"] stamps the effective set
```

In the probe: `python task1-affinity_regression/probe.py probe --split lp_edrscc` (or
`lp_edrscc_v2_cl123`).

## Regenerate / verify

```bash
python voxbind/splits/verify.py          # assert manifests intact (exit 1 on drift)
python voxbind/splits/make_splits.py     # only on a machine with complete inputs, then commit
```

# Frozen PLINDER pretraining selections

Which ligand instances make up each PLINDER pretraining corpus, pinned so every server builds the
same corpus. One folder per corpus:

| Folder | Corpus | Selection |
| --- | --- | --- |
| `v2/` | `cdgv2` | 113,976 ligand instances, 41,833 PDBs |
| `CoDE_pretrain_v1/` | `CoDE_pretrain_v1` | `v2` minus 2,607 PDBs whose proteins are in the paper's test sets (six affinity sets + CrossDocked test) |
| `CoDE_pretrain_v1_ID30/` | `CoDE_pretrain_v1_ID30` | `CoDE_pretrain_v1` minus 5,662 PDBs with ATOM3D identity ≥ 30% to a CL3 test chain |
| `CoDE_pretrain_v1_ID60/` | `CoDE_pretrain_v1_ID60` | the same at ≥ 60%: minus 2,914 PDBs |
| `v2p1/` | `ab7v2p1` | 37,885 ligand instances, 34,313 PDBs, all inside `v2` |

Each folder holds `plinder_selected.csv` (one row per kept ligand instance),
`plinder_inputs.json` (bucket, filters, build parameters and the sha256 of the selection) and,
where there is one, `plinder_funnel.json` (the provenance funnel). The `CoDE_pretrain_v1*` folders also keep the
decontamination record: the cohort FASTA, every removed PDB with its best matching test chain,
and an audit JSON.

`voxbind.splits.frozen_plinder_selection(version=...)` resolves `splits/plinder/<version>/` and
fails loudly if the CSV no longer matches its pinned sha256. The default version is `v2`.

Re-pin a selection deliberately (run where the leakage files live):

```bash
python dataset/plinder/01_select.py --version v2 --dry_run   # funnel only, writes nothing
python dataset/plinder/02_make_clean_corpus.py               # rebuilds CoDE_pretrain_v1 (refuses to overwrite without --force)
python dataset/plinder/02_make_clean_corpus.py --source CoDE_pretrain_v1 --similarity atom3d --identity 30   # CoDE_pretrain_v1_ID30
```

The retired selections (v1, v3, v2.2 LBA, v2.7, v4) and the apo-pocket pairing are in
`legacy/voxbind/splits/plinder/` and `legacy/dataset/plinder/`.

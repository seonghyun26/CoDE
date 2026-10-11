# web/ — Drug Design Viewer

One CrossDocked pocket, one generated ligand per method, side by side: a square 3D
panel each with synchronised cameras, and that molecule's Vina / quality / pose numbers
underneath. Static and server-less — no backend, no build step at view time.

```
web/
├── build.py          reads results/task2-drugdesign/, writes site/
├── src/index.html    the viewer (edit this; build.py copies it into site/)
└── site/             the built site — open it, or serve it
    ├── index.html
    └── data/
        ├── manifest.json          methods, pockets, metric schema, poses, provenance
        ├── targets/target_NN.json every method's molecules for that pocket, columnar
        └── struct/target_NN/
            ├── pocket.pdb         the shared 10 Å crop
            ├── ref.sdf            the deposited crystal ligand
            └── <Method>.sdf       record i == that method's mol_idx i
```

```bash
python3 web/build.py                     # write web/site/   (needs rdkit + torch)
python3 web/build.py --limit-targets 3   # smoke build
python3 web/build.py --check             # verify the joins, write nothing

bash web/serve.sh                        # http://localhost:8741
bash web/serve.sh --https                # self-signed, for direct access
```

`serve.sh` binds **127.0.0.1 only**: `0.0.0.0` would publish an unauthenticated page to every
network the server is on. Reach it with

```bash
ssh -N -L 8741:localhost:8741 <server>   # then http://localhost:8741
```

which is also a secure context, so the browser drops the "Not secure" label and the
clipboard API works. `HOST=<LAN address> bash web/serve.sh` opens it to the LAN instead.

Use the voxbind env's python (`conda run -n voxbind python web/build.py`) — the builder needs rdkit and torch.
Editing only the viewer? Change `src/index.html` and copy it over `site/index.html`;
a full rebuild is only needed when the data changes.

## What is in it

Published baselines only: AR, DecompDiff, DiffSBDD, FuncBind, Pocket2Mol, TargetDiff —
100 pockets, 56,783 molecules, ~108 MB — plus **DecompDiff-prior25**, a separate DecompDiff
sampling run (reference prior, 25 per pocket, 2,035 molecules over 97 pockets) read from
`base/decompdiff/eval_results_refprior25/` (`EVAL_PT_METHODS` in `build.py`). It shares no
SMILES with the bundle's DecompDiff, so it is its own panel, not a merge; it has Vina and
quality numbers but no PoseCheck / PoseBusters. The roster is **discovered**, not hardcoded: any
folder under `results/task2-drugdesign/` with both per-molecule tables and molecules on
disk becomes a panel. The `VoxBind*` / `Ours*` folders are skipped by default
(`--include-ours` overrides) because those arms were generated on another box and what
sits in the bundle is a partial copy.

## The two things the build measures rather than assumes

**Molecule order.** The canonical sequence is the meta bundle's
(`<M>{,_part2,_gap}.pt`, the rule the task-2 results collector follows), or — for a
metrics.py tree — the `metrics.json` `samples` list, *not* the SDF's own order
(`samples.sdf` holds all 9,878 generated where `mol_idx` counts the 9,206 valid).
SDF record *i* is the row whose `mol_idx` is *i*, verified at 56,783/56,783.

**Every join.** Each table's SMILES at `mol_idx` is compared against the structure's
own; 23 of 24 (method, evaluation) pairs join at 100%. Below 99% the build falls back to
`(target, SMILES)` and says so in the manifest and on the panel. TargetDiff's docking
pass scored a *different molecule set* from its pose pass, so only 27% of its molecules
have a Vina number at all — the panel shows `2511/9206` rather than ranking a subset in
silence. Missing values are empty, never `nan`, and sort last.

## One molecule, not the whole file

A panel shows one molecule, but a method's SDF for a pocket holds ~100 of them. Fetching
the file to draw one record cost 1.3 MB per pocket and used 1% of it.

So `build.py` measures **each record's byte length** and ships them in the shard as
`spans: {<Method>: [n0, n1, …]}` (lengths, not offsets -- the viewer prefix-sums them,
which keeps the shard ~3 KB larger instead of ~20 KB). The viewer then asks for one
record with an HTTP `Range` header:

```
per pocket   1,420 KB  ->  132 KB     (6 molecules = 10.6 KB, all 206 Partial Content)
```

A server that ignores `Range` answers 200 with the whole body; the viewer detects that,
caches it and slices locally, so the page is correct either way. Python's
`http.server` is one of those, so `serve.sh` carries a small 206-capable handler;
Hugging Face's CDN supports ranges already.

Structure text is held in a bounded LRU (the last 8 pockets). Before that it was an
unbounded map, which reached ~139 MB of strings after browsing all 100 pockets.

## Poses

The viewer offers **Docked / Minimized / Generated**. For the six bundle baselines only
*generated* exists: the per-molecule tables carry Vina's *scores* but never its
coordinates, and no `.pdbqt` or docked/minimised SDF is anywhere under
`results/task2-drugdesign/`. A pose button is enabled when any method has that pose;
methods without it are greyed over.

DecompDiff-prior25 does have them: its eval dump keeps Vina's PDBQT output, whose atoms
are reordered. `map_pose` infers bonds from the docked coordinates and substructure-matches
the generated molecule's skeleton onto them, so the written record is the generated
molecule with docked coordinates (4,050 / 4,070 mapped; the rest are left empty rather
than guessed). The same match gives its **redock RMSD** (symmetry-aware, no realignment),
the only method with one.
Producing them means re-docking, which needs the full receptor — also not on this box
(everything here is scored against the 10 Å crop). The unavailable buttons are disabled
and say so on hover rather than quietly showing the generated pose under another name.

To add them, drop files at:

```
results/task2-drugdesign/<Method>/poses/{minimized,docked}/target_NN.sdf
```

one record per molecule **in the same order as the generated SDF**, then re-run
`build.py`. It refuses any file whose record count disagrees, because a mismatch would
pair the wrong pose with the wrong numbers.

## Interactions

Each panel's eyes draw its interactions as dumbbells: a bead on every atom involved and
a bar between them. H-bond, hydrophobic and the types with no PoseCheck row (ionic,
π-stacking, cation–π, halogen, metal) come from NGL's contact engine, used hidden only
for its `getContactData()`. Pocket and ligand are concatenated for it, which puts them
in models 0 and 1, and NGL drops every cross-model contact unless
`masterModelIndex` names one of them; with the default not a single ligand–pocket
contact survived. vdW contacts are not an NGL type, so `vdwPairs()` applies ProLIF's
rule (distance ≤ sum of vdW radii, MDAnalysis radii, tolerance 0) and draws the closest
pair per residue, as ProLIF counts per residue. Heavy atoms only, so counts can differ
from PoseCheck's.

## PoseBusters checks

The bundle's per-molecule PoseBusters CSV has its 20 individual checks; the shard
carries them as one column, `pb_fail`, a bitmask over `manifest.pb_tests` (bit i = the
i-th check failed). The panel's PoseBusters row folds open onto the checks that
molecule failed. Verified: the number of set bits equals `n_checks - n_pass` for every
molecule.

## Statistics

The statistics pane shows mean / median per method for the pocket and, under it, four
density charts (Gaussian KDE, unit area, pooled 1st–99th percentile range). Each chart's
metric selector sits where its x-axis title would, with the unit beside it when the
metric has one. The reference is one molecule, so it is drawn as a dashed kernel of the
methods' median bandwidth, its peak matched to the tallest curve (a shape, not a
density). The eye beside a method in the table shows or hides it in the charts, the one
in the corner toggles them all, and the axes are fixed by every method so toggling never
rescales them. Chart colours are a temporary pastel set (`DIST_PALETTE` in
`src/index.html`), to be replaced by the project palette.

## Publishing: Hugging Face Space

`site/` is a plain static directory, and the viewer reads `data/` by relative path, so
it publishes unchanged as a **static Space** (page and data together, private by
default, so only people with access who are logged in can open it):

```bash
# one-time
conda create -n hfspace python=3.11 && conda run -n hfspace pip install -U huggingface_hub
conda run -n hfspace hf auth login      # a WRITE token

# every time the data or the viewer changes
python web/build.py            # (voxbind env) rebuild site/
bash web/deploy_space.sh       # -> https://huggingface.co/spaces/<you>/drug-design-viewer
OWNER=<org> bash web/deploy_space.sh   # under an organisation instead
```

`deploy_space.sh` creates the Space if it is missing (`sdk: static`), uploads `site/`
replacing `data/**` wholesale so nothing stale lingers, and writes the Space card from
`space/README.md`. Edit the viewer in `src/`, never on the Hub.

**Automatic for the page, manual for the data.** `.github/workflows/deploy-space.yml`
uploads `src/index.html` (and the Space card) to the Space on every push to `main` that
changes them, using the repository secret `HF_TOKEN`. The data cannot go that way: it
is built from `results/task2-drugdesign/`, which is not in git, so a data change is
still `build.py` + `deploy_space.sh` on the machine that holds `results/`. A viewer change that needs new data
columns should therefore be pushed together with, or after, a data deploy. `site/` is
not committed (`web/.gitignore`).

A GitHub Pages page with `data/` in a Hugging Face dataset also works (HF's `resolve`
answers cross-origin `Range` requests with 206 and echoes the Origin, verified), but
both must then be public, and the viewer would need a data base-URL option.

Reference Vina comes from `base/decompdiff/reference_vina.json`, a separate
reference-docking pass, named as such in `manifest.protocol`.

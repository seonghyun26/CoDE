# figures/ — the paper's figures, and the one command that draws them

```bash
python figures/draw.py --list                 # every figure, by evaluation
python figures/draw.py fig-vina-per-pocket-dock
python figures/draw.py posecheck              # a whole evaluation
bash   figures/draw_all.sh                    # redraw everything in the tree (PNG + PDF)
```

Run with the `voxbind` environment's python (`PYTHON=` for `draw_all.sh`). Figures are
**PNG and PDF**; SVG is no longer written.

## Layout

```
figures/
├── draw.py          what figures of two or more folders share + the command line
├── draw_all.sh      the full redraw (palette sheet, every batch, both layouts, downloads)
├── README.md
├── _parts/          the code that draws ONE folder's figures, one module (or package) per folder
│   ├── fig_vina.py, fig_posebusters.py, fig_posecheck.py, fig_rigid_consistency.py,
│   │   fig_molweight.py, fig_similarity.py, fig_mcp.py, fig_ensemble.py, fig_jsd.py,
│   │   fig_legend.py
│   ├── fig_uncertainty/   docking.py + the analyses (motivation, manuscript, …) and their paths.py
│   ├── fig_palette/       the official sheet (methods.py) + the palette proposals
│   ├── fig_overview/      binding-affinity scatter, paper overviews, the pocket/affinity viewers
│   └── fig_qual/          the qualitative 3D viewer, its capture layout, notebooks, assets
├── _tools/          analyses nothing imports, jsd-tables.ipynb, compare_outputs.py (redraw check)
├── _download/       the figures the manuscript uses, under their LaTeX names (see below)
├── assets/          images that are not drawn here (palette.png, xray pipeline.png, …)
└── fig-*/           the figures themselves, one folder per evaluation
```

`figures/fig-X/` is drawn by `_parts/fig_X` — nothing else writes there, and the code for
it lives nowhere else. Where an evaluation splits into questions, its folder keeps the
sub-folders (`fig-vina/{per-pocket,per-atomnum,baseline}/`,
`fig-posecheck/{strain-energy,clash,interaction,redock}/`, `fig-uncertainty/…`); a figure
names its sub-folder on its `@figure`.

## Drawing

```bash
python figures/draw.py --list                          # ids, folders, one line each
python figures/draw.py fig-posecheck-clash-box-by-method --clash-layout horizontal
python figures/draw.py --all --keep-going              # every figure (not the 3D sheets)
python figures/draw.py uncertainty --figures-only      # redraw only: keep saved tables
bash   figures/draw_all.sh                             # what a full refresh runs
bash   figures/draw_all.sh --with-3d                   # … plus the fig-qual sheets
JOBS=4 PYTHON=$(conda run -n voxbind which python) bash figures/draw_all.sh
```

* **A name** is a figure id (`fig-posecheck-clash-per-atom`) or a whole evaluation
  (`posecheck`). Ids stay `fig-<eval>-<name>`; only the output folder is the evaluation.
* **`--existing-only`** redraws only files already in the tree, so a figure removed from the
  paper stays removed. **`--figures-only`** redraws pictures without rewriting the CSV,
  JSON, notes and provenance beside them. `draw_all.sh` uses both, plus `--keep-going`.
* **Isolated figures.** The standalone analyses (uncertainty, palette proposals, overviews)
  set matplotlib's global style, so a figure registered with `isolate=True` runs in its
  own process whenever more than one figure is drawn.
* **3D panels** are posed and captured by hand in a browser viewer:
  `python figures/draw.py --viewer qual|ring|pocket|affinity` (or the `run_*.sh` beside
  each viewer in `_parts/`). What can be rebuilt from saved captures is registered with
  `capture=True` (`fig-qual-comparisons`) and drawn only when named or with `--with-3d`.

## Inputs a draw needs

`draw.py` reads two files when it is imported, before any figure is drawn:

* `fig-palette/method-palette-approved.json` — the official palette. **It is not in git**
  (the repository ignores `*.json`), so a fresh checkout cannot draw until it is copied
  in; `VOXBIND_METHOD_PALETTE=<file>` points elsewhere.
* `task2-ligand_generation/exps/frozenenc_probes/p79_targets.json` — the 79 density pockets.

The figures then read the evaluation outputs (`task2-ligand_generation/exps/…/metrics.json`, the docking
JSONs, `results/task2-drugdesign/`, the figure-data artifacts in `figures/_data/`, and the sibling
checkouts `../base_drug` and `../funcbind`, overridable with `VOXBIND_BASEDRUG`,
`VOXBIND_FUNCBIND`, `VOXBIND_SIBLING_ROOT`). A figure whose input is missing fails with
the path rather than drawing an empty panel.

## The figures

For matching landscape steric-clash and redocking-median panels, plus a one-column key:

```bash
python figures/draw.py \
  fig-posecheck-clash-box-by-method fig-redock-median fig-posecheck-strain-legend \
  --clash-layout horizontal --legend-layout vertical
```

`--clash-layout vertical` keeps the original portrait plot (the default); `horizontal`
places methods along x and writes `fig-posecheck/clash/clash-box-by-method-horizontal.*`.
The landscape clash, redocking-median, and strain ECDF panels share `PCSZ_STRAIN_SIZE`
(about 6.95 × 2.75 inches), with identical PNG/PDF page dimensions. The ECDF has no
embedded legend; clash and redocking median omit the method names and ticks on x.
Redocking bars retain the percentage within 2 Å, without the explanatory caption above.
Both use the shared `soft()` palette and the section's `RB_ORDER` method order.
`--legend-layout vertical` writes `vertical.*`
in `fig-legend/`, with the reference first and methods in one column;
`horizontal` writes `horizontal.*` with all entries in one row (the default). Both label the reference
`Reference`. PNG and PDF are produced.

`fig-posecheck-strain-rotbond-grid` keeps outlier points aligned at each box's
rotatable-bond count, without horizontal jitter. Regenerate with:

```bash
python figures/draw.py fig-posecheck-strain-rotbond-grid
```

The strain cumulative-probability (`ecdf-by-size-*`) and all-method median
(`per-atom-all-methods-median`) panels use `PCSZ_STRAIN_HEIGHT_SCALE = 1.1 * 0.9 * 0.9`: their previous canvas height
is reduced by a further 10%, with width unchanged. The landscape clash and redocking-median figures follow
the ECDF canvas size and share fixed axes bounds (left 16%, right 98%, bottom 26%,
top 94%), so their plotting areas align. The mean, portrait clash, and grid layouts
keep their own sizes.

**`fig-vina/` is three folders** (2026-09-22) — `per-pocket/` (affinity ranked pocket by
pocket), `per-atomnum/` (affinity against ligand size) and `baseline/` (ours against one
baseline at a time). One evaluation, three questions that are read separately, and a flat
folder of ~90 files left the reader doing the sorting. Each figure names its own sub-folder
on its `@figure`, so the ids and the commands below are unchanged.

The per-pocket Vina panels (`fig-vina-per-pocket-{dock,score,min}`, both the `core` and
the `all` variant, mean/median/pair) are drawn at `VPP_WIDEN = 1.1` times their previous
width with the height unchanged, and **no panel draws the crystal ligand**: the grey
dashed reference was already off the crowded `all` panels and is now off `core` too. Each
figure's CSV still carries the `reference` column per rank, and the run log still prints
its mean beside the arms. y is ticked every `VPP_Y_STEP = 3` kcal/mol on the grid through
`VPP_Y_ANCHOR = -4` (so Dock reads -13 / -10 / -7 / -4), instead of the automatic locator's
label on every integer; a tick outside a panel's own span is simply not drawn.

`fig-vina/baseline/` (`fig-baseline-dock-per-pocket`, 2026-09-22) is the same `core` panel drawn
**once per baseline**: CoDE against AR, Pocket2Mol, DiffSBDD, DecompDiff, FuncBind,
TargetDiff and VoxBind in turn, mean and median, two clouds each. Same ranking (pockets
sorted by CoDE's own value), same colours and markers each method carries everywhere else,
same CSV beside every figure. Two things differ from `per-pocket/`: y is ticked every
**2** kcal/mol (`VBL_Y_STEP`, still on the `VPP_Y_ANCHOR = -4` grid), and **all fourteen
panels share one y range** — ours is one of the two clouds in every figure, so a span fitted
per pair would give CoDE a different shape in each file and make the comparison between two
of these figures a comparison across two scales. The VoxBind pair *is* the existing `core`
figure, so it keeps that name (`dock-per-pocket-{mean,median}-core`). `baseline/stats.txt`
holds every pair's count in one table: on what share of the 79 pockets CoDE's statistic is
the more negative one, with the mean of (CoDE − baseline) beside it.

`dock-per-pocket-combined` is the same panels on ONE SHEET, 6 rows x 2 columns: a row per
**published** baseline in the drug-design table's order (`METHOD_ORDER`: AR, Pocket2Mol,
DiffSBDD, TargetDiff, DecompDiff, FuncBind), **median left and mean right**. Every cell is
the standalone panel at the standalone width, so a row is the two files it sits beside on
disk. VoxBind is not a row — it is the model ours modifies rather than a method ours is
benchmarked against, and a +0.3 row beside six +1.x ones would read as the same kind of
claim; its pair keeps its own `…-core` files. Only the BOTTOM row carries the x-axis name
and its tick numbers (`VBL_COMBINED_XLABEL_ROW`): twelve panels over one ranking is the same
79 pockets twelve times, and the tick marks and rules are in every panel regardless.

```bash
python figures/draw.py fig-baseline-dock-per-pocket
```

`fig-vina-per-atom` is **one variant, and the version token is off its names**
(2026-09-22): it was v1 (reference gated) / v2 (reference faded) / v3 (no reference) over
the same mean/median x core/all, twelve figures; v1 and v2 were dropped on request and the
survivor is now `dock-per-atom-{mean,median}-{core,all}`, not `dock-per-atom-v3-…`. It
draws no crystal reference at all — not the grey dashed curve on the Vina panel, not the
grey share on the strip, and so not in the key either (`VPA_REF`, `ref_vina=False,
ref_dist=False`). The *grid* still draws the reference in every cell, since a cell is one
method read against it, and every value is still in `dock-per-atom.csv`. What v1 and v2
chose between is kept in the part file's header, and their knobs are still in `VPA_REF`, so
the grey is one flag from coming back. The median core contains VoxBind and CoDE. The
separate `dock-per-atom-median-all-test` keeps those two at full opacity and draws every
other method at 20% opacity for comparison. Regenerate with:

```bash
python figures/draw.py \
  fig-vina-per-pocket-dock fig-vina-per-pocket-score fig-vina-per-pocket-min \
  fig-vina-per-atom
```

`fig-posecheck-rotbond-distribution-all-methods` writes the method distributions in
4 rows × 2 columns, with an unfilled dotted grey reference in every panel. The
second caption line reports the reference share beyond 12 bonds from the same data.

Uncertainty figures are organized by purpose under [fig-uncertainty/](fig-uncertainty/README.md):
current A/B in `motivation/`, detailed affinity/decoding results in `affinity/` and
`representation/`, actual input density in `input-density/`, B-factor controls in
`controls/`, docking in `docking/`, earlier runs in `exploratory/`, and literature in
`references/`. The code is the package `_parts/fig_uncertainty/`; its `paths.py` defines the
shared destinations. The linked index lists every reproduction command.

Figures are grouped by **evaluation** — one folder per question, named by it:

```
figures/fig-vina/per-pocket/   dock/score/min per pocket, every arm
figures/fig-vina/per-atomnum/  Vina Dock against heavy-atom count, over a size strip
figures/fig-vina/baseline/     Vina Dock per pocket, ours against ONE baseline per figure
figures/fig-posebusters/       validity, which checks fail (and which fail together), SuCOS
figures/fig-posecheck/         strain, clashes and interaction fingerprints
figures/fig-rigid-consistency/ rigid-fragment RMSD against fragment size
figures/fig-molweight/         where each arm puts its molecules
figures/fig-similarity/        similarity and novelty against the crystal ligand
figures/fig-mcp/               the macrocyclic-peptide fine-tune ladder
figures/fig-ensemble/          champion + partner encoder cohorts
figures/fig-jsd/               bond/pair/atom-type JSD and rings against CrossDocked ligands
figures/fig-uncertainty/       does the gain track the pocket's B-factor and occupancy?
figures/fig-posecheck/redock/   how far Vina moves a generated pose when it re-docks it
figures/fig-legend/            standalone horizontal and vertical method legends
figures/fig-palette/           the official method palette sheet and the proposals
figures/fig-overview/          the overview panels (affinity scatter, schematics, pocket renders)
figures/fig-qual/              the qualitative 3D comparison sheets and their captures
```

`fig-jsd/` draws `results/task2-drugdesign/_shared/260913_crossdocked_jsd/crossdocked_jsd.json`,
which `bash task2-ligand_generation/scripts/89_eval_crossdocked_jsd.sh` writes (~15 s, no GPU). The reference
drawn is **every CrossDocked train+test ligand** of `split_by_name.pt`, each distinct molecule
weighing one (100,100 files are 8,829 molecules). The 100 **test** ligands — the protocol
behind TargetDiff's Table 2/3 and VoxBind's Table 2 — are too few to estimate a bond-length
distribution (16 C=N bonds), but they are still scored, beside a pose-weighted train+test set,
so the published scale stays available; the script's `--selfcheck` re-scores TargetDiff's own
samples against its published row on every run.
`fig-posebusters/pb-budget-{core,all}.*` (2026-09-23) is **what each arm's validity rate is
made of**: one bar per method, 0–100% of its molecules, segmented by what took a molecule out.
It is not another view of the heat map — those columns OVERLAP (a molecule failing bond angles
and a clash is in both), and overlapping shares cannot be stacked. Here a molecule is counted
**once**, by the groups it fails: none (valid), exactly one (that group's segment), more than
one (`Two or more groups`), or — for the molecules that never reached a check —
`Did not load`, added to the population from `loading_checks.json` (2026-09-23) so the bar
partitions *everything the arm produced*, not only what was scored. The segments therefore sum
to the arm, which the build asserts. That population is why CoDE's valid reads 67.4 here and
67.5 on the heat map: 8 molecules it could not load are in this denominator and not in that
one.

**The x axis starts at 45%**, not 0 (2026-09-23, on request), and the plot area is **half** its
first width (0.8× first, then 0.5×: 9.90 → 4.95 in of axis, canvas 13.46 → 9.50 in — at this
width the key, not the plot, is what the canvas has to be wide enough for). Every segment keeps its true
width — the cut only clips the `valid` block, whose value is written at its own end — and it
buys 1.7× the width for the failure budget, which is the part the figure is for. The crystal
ligands are ruled off from the eight methods in the x axis's own pen, and the key is two rows
of four **above** the plot (2026-09-23, on request — it spent the day below it), **fitted to the
figure's width** rather than set (2026-09-23, on request: "legend 가로 길에 맞게 키워줘"): drawn
at 13.5 pt, measured, rescaled by how far it falls short of 98% of the canvas, drawn again —
which lands at ~15.3 pt and 96% of the width here. Everything in a matplotlib key is in font
units, so its width is very nearly linear in the point size and two passes suffice; the loop
stops when they agree to 0.4%, and `PB_BUD_LEG_FS_MAX` caps it. The key also cuts four names
down: `PoseBusters valid` → `All passed`, `Protein clash & overlap` →
`Protein clash`, `Self-clash & energy` → `Self-clash`, `Two or more groups` → `2+ groups`
(2026-09-23, on request) — the last of those is what the run log had always called it, and the
log now reads the map too, so the figure and the log cannot drift apart. That is a
DISPLAY map in this figure only — `PB_GROUPS` still names the groups in full, so the CSV, the
run log, the group-bar figures and their filenames are untouched, and the cut names never reach
anything that is read without this figure round it. The halves dropped are second checks
within a group, not second kinds of failure. Its entries **read down, not across**: `Valid` over `Bond
geometry`, `Planarity` over `Ring pucker`, the two clashes, then the two segments that are not
single groups. That is matplotlib's own column-major fill, so the handles go in unreordered and
`row_major()`, which exists to undo it, is not called here. **At 13.5 pt those four columns are
what the canvas has to be wide enough for**, not the 0.5× plot: the key needs ~8.5 in of
labels (10.2 before the names were cut and the column spacing tightened) and the plot 8.3, so
the canvas settles at 8.51 in and the key spans it. Its columns sit at **0.8 em** rather than matplotlib's 2.0 (2026-09-23, on
request — the alternative, four rows of two, stacks instead of spreading and was not wanted):
27 pt of paper between entries three times over is what made four columns need 10.2 in, and
tightening it took 0.67 in off the figure. (`mode="expand"` was tried and is not used: it clamps the key to the
width it is given instead of growing the canvas, so `Protein clash & overlap` ran into the next
swatch. Two columns is the layout that fits inside a 0.5× plot.) The valid rate at the end of
each `valid` block is set `center_baseline`, not `center` — a text box is as tall as the font,
so centring the BOX puts digits, which use no descender, half a descent above the bar's middle;
measured off the PNG they now sit within half a pixel of it. **The key sits 0.16 in off the
plot**, a distance in INCHES rather than a figure fraction — the two variants are 6.6 in and
3.5 in tall, so one fraction meant two different gaps, and under the plot it also had to be
measured against the x LABEL at `TITLE_SIZE` (the size the export boundary gives it, not the
size it has while being drawn) rather than against the axes. Above the plot none of that
applies: nothing sits between the key and the top spine, so the gap is measured straight and
corrected once against `fit`'s own top margin. Measured off both PNGs it lands at 0.105 in of
visible white, identically in `core` and `all`. It is the UpSet figure's
per-molecule sets aggregated to the heat map's six groups, so the three figures cannot
disagree.

What it shows in one row each: the voxel arms spend nearly their whole failure budget on **one**
group — ring pucker, 19.0% of VoxBind's molecules and 19.3% of CoDE's — while **DiffSBDD (13.2%)
and FuncBind (14.8%) spend theirs on `Two or more groups`**, molecules wrong in several ways at
once, against 2.9–8.5% elsewhere. Pocket2Mol's is mostly protein clash (12.9%), AR's is spread
over four groups, and `Valence & connectivity` never takes a molecule from anyone, so it is
dropped from the key and named in the run log instead.

Segment fills give **every group that occurs its own lightness**, with the hatch as a second
cue on **the key's bottom row** — bond geometry, ring pucker, self-clash and did-not-load
(2026-09-23, on request: the key is filled down its columns, so those are its odd entries).
**The hatch is grey, not white** (`GRID`, 2026-09-23, on request): matplotlib draws a patch's
hatch in its edge colour, so this is also that segment's outline, and white was invisible on
the two lightest steps — exactly where a hairline segment most needs a texture. Unhatched
segments keep the white edge. It was three lightness steps crossed with a
hatch — six segments in three pairs — and a pair is two segments of the *same* blue told apart
by hatching alone, which fails exactly where this figure spends its width: `Planarity` at 0.3%
of a bar is a hairline, and a hairline has no room to show a hatch. The ladder is also spent on
the groups that actually occur, not on the six `PB_GROUPS` names — `Valence & connectivity`
takes no molecule from anyone, so it no longer eats the lightest rung. Measured off the PNG the
five live groups land at luminance **0.70 / 0.57 / 0.43 / 0.30 / 0.163**, an even 0.134 apart,
under the near-white valid block at 0.91 and above `2+ groups` at 0.108. **The darkest group is
CoDE's own colour** (2026-09-23, on request): the ladder's dark end is `highlight("CoDE")`'s
luminance and the shared ramp now runs *through* that colour, so the rung is `#4761F2` exactly,
with no hex literal in this family. It lands on `Self-clash`, next to `Protein clash`. Protein
clash itself cannot be CoDE and that is arithmetic: as the fourth of five rungs its luminance
is ¼·(lightest) + ¾·(darkest), which at CoDE's 0.163 forces a lightest rung of 0.42 — no light
blue left in the bar — and a ramp ending near black, taking every heat map in the deck with
it. **The last segments
are plain**: `2+ groups` lost its cross hatch on the same request — it is the darkest thing in
the bar and sits at the end of every row, where a heavy texture did nothing the colour was not
already doing. `Did not load` keeps a fine dotted one only so the key does not show two
identical swatches; it is 0.1% of two arms, read in the key rather than on the bar. The steps
are cut from **the deck's shared heat ramp** (`HEAT_CMAP`, the one `heat_colorbar` draws), and
the ladder is inverted from luminance by sampling it, so it follows whatever `00_core` says the
ramp is. The shared ramp itself was rebuilt twice the
same day, both on request: first light grey to a darkened `highlight("CoDE")`, then — final —
**two given colours, `#E8F2F7 → #465CCF`**, blended straight in sRGB with nothing else chosen;
the deep end then took a little more saturation on request (HLS S 0.588 → 0.676 at the same hue
and lightness, `#3C55D9`), and then stopped being a colour of its own at all — it is now the
line from the light end **through `highlight("CoDE")`**, continued to `HEAT_CODE_AT` = 0.83 of
the way, so the deck's CoDE blue is a point *on* the bar (at 83%) and the end, `#2643F1`, is
what lies beyond it. `highlight()` and not `color()`: the lighter CoDE tone `#8092F6` would
bottom the bar out at luminance 0.247, where the PoseBusters map paints every cell from 94 to
100 the same periwinkle. Ending *on* CoDE was rejected too — every heat map in the deck spends
its top on 95–100% cells and would end on a method's identity colour, which is what the palette
rule exists to prevent; passing through it says *this much* where ending on it says *this
method*
(it had run `#FAFDFB → #5E70E3`). That is one edit in `00_core` and it moves **five** figures,
not this one: the per-rotbond strain grid, the two uncertainty correlation matrices, the
PoseBusters group bars and this budget — all five are redrawn together whenever `HEAT_STOPS`
changes. Luminance range 0.98–0.20 → 0.87–0.108, so cells ten points apart separate further
than under any earlier version, and the light end is a pale blue visible against white paper
rather than a near-white that disappears into it. `pb-valid-heatmap` and `pb-check` are NOT on
the shared ramp either, as of the same day's last pass: **all seven heat figures now read
`HEAT_CMAP`**, and the local `PB_HEAT_CMAP` (white → `#8291E8`) and `PB_TAB_CMAP` (white →
CoDE's highlight) are deleted. What stays per-figure is the NORM, which is the thing those two
actually needed: the group map floors at its worst cell, the check table at 45. The method palette is
deliberately not used: its hues mean *methods* everywhere else in this deck.

`pb-valid-heatmap-*` also lost 1.15 in of canvas (2026-09-23): `PB_HEAT_SIDE` was 3.75 in on a
fear of clipping rather than a measurement. Measured, the row names take 1.85 in and the longest
one starts at the layout pad (0.10 in) at every allowance from 3.75 down to 2.10 — constrained
layout places the labels from the left and spends the surplus on the *right*. At 2.60 the left
white goes 0.35 in → 0.10 in and the figure is 14.49 in wide.

`pb-valid-heatmap-*`'s counts column is **`Passed (22)`**, not `Passed (20)` (2026-09-23):
it now counts everything PoseBusters reports, the two loading columns included, rather than
only the 20 that `valid` is graded on. It is counted per molecule, not offset: every scored molecule
carries all 22 checks as booleans, none of them `None` (checked over all 60,536).

**The molecules that could not be loaded are in that mean too**, at 1 of 22 — `mol_cond_loaded`
is true of them, `mol_pred_loaded` is false by definition, and the other twenty were never
evaluated, which the count treats as not passed exactly as `valid`'s all-must-pass rule does.
They are absent from `metrics.json` (that is *why* they could not be loaded), so the mean's
population is the scored molecules plus them: VoxBind 21.62 → **21.61** over 7,892, CoDE
21.58 → **21.56** over 7,881, every other arm unchanged. `posebusters-valid-heatmap.csv`
carries both denominators — `n_molecules` for the percentages, `n_counted_for_mean` for the
counts column — and `n_tests` reads 22. Note the median is **22.0 for all nine arms**, so the
mean is the only statistic here that separates them.

`fig-posebusters/pb-check-{core,all}.*` (2026-09-22) is **all 22 checks PoseBusters
reports**, one row each, as a pass-rate table under a `Category` / `Specific check` pair of
columns (both left-aligned, a long category name broken **before** its ampersand — `Valence`
over `& connectivity` (2026-09-23, on request; it broke after the `&` first) — so the column is
1.45 in rather than 2.10, check names opening with a capital, ruled off category by
category in **true black** — the deck's ink is #514F52 and reads as grey at a rule's width,
so the rules are the one thing in this figure that is #000 — with a frame under the headings,
under the summary row and under the last row) — the per-check detail behind
`pb-valid-heatmap-*`, whose eight columns are groups of them, and which existed only as
`check-pass-rates.csv` until now. The row blocks ARE those groups, in the same order, so the
two read together; the top row is labelled **`Overall valid`** in the merged `Category`/`Specific
check` cell (2026-09-23) and is `valid` itself — the AND over the 20 graded checks, not a sum
of the rows under it, which is why it is not called `Total`
(`n_checks` is 20 in every record — `mol_pred_loaded` and `mol_cond_loaded` are reported and
not graded, and sit in their own block at the bottom). Its numbers are in `pb-check.csv`.
**A row is 0.374 in tall** (2026-09-23, on request — a tenth more than the 0.34 it was). Nothing
in the row grew with it (names at 13 pt, numbers at 14.5), so what the tenth buys is air; every
other vertical measure in this figure is written in *rows* — the block gap, the heading strip,
the foot — so they follow it and the table's proportions are unchanged.
The **range** is this figure's own (2026-09-23) — the shared ramp's light end at 45, its deep
end at 100, keyed by a long bar **down the right-hand side** (it lay under the table until
2026-09-23; the same three numbers in `PB_TAB_CBAR` are inches of width rather than of height
now, and the bar's length is a share of the table's height) — because the group heat map's 50–100 spends
its whole range above the worst check cell, 76.9, and paints twenty rows one shade. The ramp
itself is `HEAT_CMAP`, like every other heat figure in the deck: this table had its own white →
CoDE-highlight ramp for part of 2026-09-23 and gave it up the same day, on request, once it was
clear the floor and not the colours was what it needed. Both variants take the same norm, so a
number is the same colour in `core` as in `all`.

**Six rows are drawn in neutral grey because they cannot fail on this benchmark**: the
cofactor and water checks have nothing to measure — the receptor every arm is scored against
is the pocket10 crop, which carries no HETATM and no HOH. The build asserts those six are
100.0, so a run where one starts failing stops rather than painting a real failure grey.
`Sanitization`, `All atoms connected` and `No radicals` stay on the ramp although nothing
fails them: "nobody broke this" and "this could not break" are different findings.

**The two loading rows are measured, not assumed** (2026-09-23). They were grey too, on the
argument that a molecule PoseBusters could not load would not be in `metrics.json` to be
counted — an assumption, and the wrong one. A figure-data builder (not part of this repository)
(the `moleval` env, ~3 min, no GPU) re-runs **PoseBusters' own `Loading` module** — dock.yml's
first, the one that emits those two columns, with the other nineteen filtered out — over every
record of every `samples.sdf`, and writes `loading_checks.json`. Two findings came out of it:

* **`Mol pred loaded` is not 100% for the voxel arms**: VoxBind 99.95% (4 molecules of 7,892)
  and CoDE 99.90% (8 of 7,881) do not survive `SDMolSupplier(sanitize=True)`. Those are
  exactly the molecules missing from their `metrics.json`, so no other row in the deck sees
  them. `Mol cond loaded` is 100% everywhere — the receptor always loads.
* **60 molecules were never PoseBusters-scored at all**: AR 40 and DiffSBDD 20 carry
  `{"error": "chunk k/5: worker timed out after 1800s"}` instead of a result. They load fine;
  the scoring run gave up on them. **Every PoseBusters figure in this folder silently excludes
  them** — the run log now prints the `scored` / `records` / `unscored` columns per arm so the
  gap is on the record, and `pb-check.csv`'s `n_molecules` is the row's own denominator (the
  loading rows are over every record, the other rows over the molecules that were scored).
  TargetDiff's own gap is larger and is not a timeout: its `samples.sdf` holds 7,798 records
  against `metrics.json`'s 7,287, a filter applied upstream of this deck. That leaves **eleven live rows**, and they
spread the field widest at `bond angles` (TargetDiff 77.4 against Pocket2Mol's 97.9) and
`non-aromatic ring non-flatness` (CoDE and FuncBind at 77.5 against AR's 99.9) — the baselines
are not failing a different *set* of checks from ours, they trip different ones of the same
eleven. A cell reading 100.0 is exactly 100.0: rates below it are floored, so 99.98% reads
99.9 rather than rounding up to a perfect score. **The figure carries no footnote** (it did
until 2026-09-23): what the grey means is caption material, and it is here, in the run log's
`(by construction)` marks and in `all-checks.csv`'s `cannot_fail_here` column.

`fig-posebusters/upset-{core,all}.*` (2026-09-22) is the **co-failure** view of the checks
`check-failures-*` draws as marginals. A column is a molecule's *exact* failed set — `bond
angles + bond lengths` counts the molecules that fail those two **and nothing else** — so the
columns are disjoint, each is a share of that arm's own molecules, and the empty set (the
PoseBusters-valid ones) is left to the run log and `upset-sets.csv`. The 16 columns drawn are
the sets reaching 0.25% in some arm, ranked by their worst arm; the crystal ligands get a bar
but are kept out of that ranking, where one of 79 molecules is 1.3%. They are **drawn in blocks
of set size** (2026-09-23, on request): every single-check set, then every pair, then every
triple, still worst-first inside a block, separated by a blank (there was a rule at each
boundary first, dropped on request once the blocks stopped touching and each grew its own
numbered axis — the blank and the axis already say where a block ends). Which sets are drawn
does not change — only the order — and what the blocks buy is the reading the figure exists
for: how much of an arm's failure is one thing going wrong and how much is several at once
becomes a distance along the axis rather than a count of dots.

**Each block carries its own y scale** (2026-09-23, on request). On one scale the panel is set
by the single-check block, whose tallest column is 19.2%, while the pairs top out at 5.0 and
the triples at 1.9 — two thirds of the columns drawn in the bottom tenth of the panel, where
they cannot be ranked at all. A shared scale is the right default when blocks are comparable
and these are not: inside a block the question is *which set*, and across blocks it is answered
by the blocks themselves. Every block starts at 0 and carries its own numbered axis, so no
height is read across a rule by mistake. Mechanically each row of the figure is one axes per
block in a subgridspec, the blank between them (`PB_UPSET_BLOCK_GAP`, 0.62 in) is what those
axes are written in, and the gaps are **added to the canvas** rather than taken out of the
columns, so a dot keeps its spacing. Both rows are laid out again from the settled span after
`fit()`, in inches, because two subgridspecs' idea of `wspace` does not survive tight_layout,
and the x label is re-centred on the whole matrix by hand since only the first block owns it.

**The rows are the all-checks table's rows** (2026-09-23, on request: the two figures disagreed
about what a row even was). They used to be the checks the drawn sets are made of, ranked by
their marginal — eight rows in an order of their own, and `double bond flatness`, `inchi
convertible` and `protein-ligand maximum distance` were rows of `pb-check-*` and not of this
figure at all. Now a row is a row of that table: `PB_TAB_GROUPS`' order, which is also the heat
map's column order, and the table's own spelling (`Non-aromatic ring non-flatness`, capital and
all, wrapped only because the names sit between the canvas edge and the matrix here). Two things
are still left out, and neither is a departure: the two `Not scored` rows, because `valid` is the
AND over the 20 **graded** checks and a failed set can never contain a loading column; and the
checks nobody fails, which a table can afford to draw at 100.0 and a matrix cannot — a row of
grey dots beside an empty bar says a check exists, which is the table's job, at 0.72 in of a
figure already 12 in tall. A row with no *black* dot is kept: `inchi convertible` fails 0.1% of
DecompDiff and every set it belongs to is rarer than the cut, so its marginal bar is the whole of
what it has to say.

**The marginal panel is on the right and there is no key** (2026-09-23, on request). The
per-check marginal — taken from the breakdown figure's own table so the two cannot disagree, and
counting every molecule failing the check, so it is not the row's columns added up — sat to the
*left* of the matrix, which is where UpSet's own examples put it. Two things paid for that: the
row names had to sit between the two panels, so each was a measurement `tight_layout` could get
wrong, and the figure opened on a bar chart whose rows were 2.6 in to its right. Moved, the left
edge is the row names, the matrix begins where the eye does, and both bar panels are down the
right-hand side, one over the other. Its bars grow away from the matrix (the mirrored `wide → 0`
axis is what a left-hand marginal needs and would hang this one off the canvas). The key went
with it: nine series unlabelled is a real cost, paid once for the deck rather than in every
figure — the colours are the house method palette, the breakdown figure beside it carries them
labelled, and the bars are in the same order, reference first, in every group here.

It answers the question `valid` raises and the marginals cannot: an all-must-pass rate is set
by how many molecules carry **at least one** failure, not by how many failures there are. Failure
here is usually a single check — 82% of VoxBind's failing molecules and 79% of CoDE's, but only
53% of DiffSBDD's and 61% of FuncBind's, which fail two checks or more in a fifth of everything
they draw against 4–10% everywhere else. The tall column is a different check for nearly every
method (ring non-flatness for the voxel arms and FuncBind, bond angles for TargetDiff, minimum
distance to protein for Pocket2Mol, aromatic ring flatness for AR), and the co-failures that do
exist share a cause: bond angles with bond lengths, minimum distance with volume overlap.

`fig-uncertainty/docking/` asks whether the gain CoDE and CoE take over VoxBind is larger where the
crystal is **well ordered** — which is what it should be if the electron-density conditioning
is what buys the gain. `bash task2-ligand_generation/scripts/95_eval_pocket_uncertainty.sh` (~6 s, no GPU)
writes `results/task2-drugdesign/_shared/260916_pocket_uncertainty/pocket_uncertainty.json`:
per pocket, the deposited B-factor and occupancy of the 10 Å crop the sampler saw and of the
receptor atoms within 5 Å of the crystal ligand, plus resolution, R$_{free}$, the map σ that
gated these 79 pockets, and every arm's per-pocket Vina. **B-factors come off the CrossDocked
receptor** — it preserves the deposited values atom-for-atom (verified, max |Δ| 0.0000 over
8,045 atoms) — but **occupancy cannot**: CrossDocked keeps one altloc and rewrites every
occupancy to 1.00, so it is read from the deposited mmCIF, which the script fetches into
`dataset/data/cif/` if the density pipeline has not already cached it.

The answer is **no**, and `bfactor-absolute-dock-all.*` is why: B-factor correlates with
*absolute* per-pocket Vina at ρ ≈ −0.06 … −0.26 for **all nine arms**, including the six that
never see a density map, so it is describing the pocket rather than the method and it cancels
in the subtraction. Hence every figure in this folder is drawn twice — as a gain over VoxBind
and in absolute terms — and `correlation-matrix-{delta,absolute}.*` put all 15 descriptors
against all arms on one sheet. Bold marks a coefficient surviving Benjamini–Hochberg over
the 15 descriptors of its column; the raw p is in `correlations.csv` and is not the number to
read, since the JSON holds 15 × 5 × 8 tests. Occupancy is nearly a constant on this test set —
70 of 79 pockets have no alternate conformer in the site at all — so the left panel of
`occupancy-delta-dock-core.*` counts how many pockets it can separate before the scatters
beside it can be over-read.

`fig-uncertainty/exploratory/affinity-initial/` preserves the **initial regression analysis**, where the same question has
779 paired points instead of 79. `bash figures/_parts/fig_uncertainty/96_eval_affinity_uncertainty.sh` probes
the coords-only encoder (**C**, the regression twin of CoE) and the density one (**C+D+G**, the
twin of CoDE) in ONE call so `--intersect` puts them on identical complexes with an identical
probe head, averages three seeds, and writes
`results/task1-affinity/_shared/260916_affinity_uncertainty/affinity_uncertainty.json`. The
descriptor set gains **ligand RSCC** from the wwPDB validation reports — the most direct measure
of the conditioning input there is, and the one to read first — with the caveat that
`lp_edrscc_v2` is *selected on* RSCC ≥ 0.8, so the axis is truncated and any RSCC correlation
measured on it is a lower bound. B-factors are **contact-weighted** (1/d², Gaussian, contact
count) as well as plain: a mean over "atoms within 5 Å" gives an atom at 4.9 Å the same vote as
one touching at 3.2 Å.

`affinity-tercile-abs-error.*` is the figure that carries the result, and it exists because the
effect is **not monotone**: density's help is flat over two thirds of the B-factor range and
then turns on, which Spearman barely sees (ρ = −0.06) while a permutation test on the
pre-specified top-vs-rest contrast rejects at p = 0.007 (−0.117 pK [−0.201, −0.033]). The
direction is the opposite of the obvious hypothesis — density helps most where the interface is
**floppiest**, not where the crystal is cleanest — and it survives partialling out resolution
and the complex's own pK. RSCC shows no such regime (p = 0.32). Note the caches: the coords-only
features on this box cover 5,316 complexes, so the probe runs 779 of the 1,320 test complexes;
the comparison is still strictly paired, but these are not the published C / C+D+G rows.

`fig-posecheck/redock/` is the one thing the Vina table cannot say. `vina_score` scores the pose **as
generated**; `vina_dock` scores the pose Vina **found** — and an affinity never says how far
the molecule had to move between them. `ecdf-{core,all}.*` (two panels: the ECDF for the
body, the share BEYOND each distance on a log axis for the tail, both over 0–20 Å) and
`median-{core,all}.*` draw that distance: the heavy-atom RMSD between each molecule's generated and redocked pose,
computed **in place** (both poses are in the receptor's frame) with molecular symmetry taken
into account, never superposed — superposing first answers "is this the same conformer" and
reads 0.87 Å against 3.01 Å in place.

**Read it against the crystal ligands, never against 2 Å.** They are redocked by the
identical protocol and are the control, and on these pocket10 crops they come back at
2.69 Å with 42.3% within 2 Å — that is this protocol's floor, not a model's failure. CoDE is
the only arm that gets near it (2.78 Å, 41.2%), against 3.39 Å / 35.8% for the VoxBind it
modifies, and it wins that comparison in all five heavy-atom bins. The baseline ordering is
NOT size-independent, though: AR and Pocket2Mol sample much smaller molecules (median 15
heavy atoms against CoDE's 26) and Pocket2Mol takes the three largest bins, so read the
`all` figures as "how reproducible are the poses each arm actually produces", and the
per-bin table for anything else.

**CoDE's tail is the one place it is worse, and it is VoxBind's.** Past ~5 Å CoDE holds more
molecules than any baseline — 8.8% beyond 8 Å against 2.7–6.5% — which is why the ECDF runs
to 20 Å with a log survival panel instead of stopping at 8 Å, where it first did and where
the tail was invisible. Much of it is size (beyond 8 Å is 0–1% under 16 heavy atoms for
every arm, 9–26% above 30, and half of CoDE's molecules have 26 or more), but not all:
at 26–30 atoms CoDE and VoxBind are 15.5% and 14.9% against TargetDiff's 4.9%, and their
two survival curves lie on top of each other from 5 Å on — CoDE's gain is the 1–5 Å body,
not the tail. Those tail molecules move their centroid only ~2 Å: they come back reversed
in place. `tail.csv` has the thresholds, p90/95/99, the per-size shares and that centroid
shift. The RMSDs themselves are measured by
a figure-data builder, which is not part of this repository (RDKit + the staged gen/docked pose
pairs, `moleval` env, ~5 s), which also writes that per-bin table.

`fig-jsd/jsd-tables.ipynb` shows the same numbers as tables (bond JSD, distribution summary,
ring sizes, ring statistics, re-evaluation vs published) and prints each one as LaTeX to paste.

A file is named by the figure, minus whatever its folder already says — so
`fig-posecheck/` holds `clash-per-atom-mean-all.png`, `strain-ecdf-by-size-le15.png`,
`strain-box-per-rotbond-core.png`, and `fig-rigid-consistency/` holds `rigid-consistent-box-all.png`
rather than `rigid_box_all.png`. The `core` / `all` suffix is the arm set and stays: two
figures that differ in which arms they draw must not be able to share a name. Each folder
also carries the CSVs behind its figures; where two figures export the same table (the pose
per-atom table, the interaction table) it is now written once instead of once per figure.

The rigid consistency distribution is drawn with
`python figures/draw.py fig-consistency-rigid-box` (the former
`fig-consistency-rigid-violin` command is an alias). `rigid-consistent-box-core.*` and `rigid-consistent-box-all.*`
use twelve fragment-size bins in two rows: **3, 4, 5, 6, 7, 8** above
**9, 10, 11, 12, 13, 14+**. The last bin pools all fragments with at least 14 heavy atoms;
2-atom fragments are excluded. Boxes show the IQR and median, whiskers extend to the
most extreme observations within 1.5×IQR, and individual outlier points are hidden.
Both rows use linear RMSD axes: 0–0.6 Å above (0.2 steps) and 0–1.2 Å below
(0.4 steps), with one shared Fragment RMSD label; whiskers beyond
these display limits are clipped. Boxes have no border; median, whisker, and axis lines
are drawn thicker for readability. Axis-label, legend-font, and axis-line proportions
follow `fig-jsd/summary-all.png`.
Bins with fewer than five fragments are omitted and logged; their x ticks remain.
The method legend sits below both panels in one row and labels the crystal ligands
**Reference**. `fragment-by-bin.csv` retains raw-data summaries (including the historical
coarse binning) and exports the box whiskers. Older `violin-*.{png,pdf}` files are
historical outputs and are no longer generated.

For the appendix, `python figures/draw.py fig-consistency-rigid-box-expanded` draws the
same figure with the 14+ group taken apart:
`rigid-consistent-box-expanded-{core,all}.{png,pdf}`, one box per fragment size in
rows of ten, with a single pooled tail at the end. **The cut is computed from the arms the
figure draws, so the two variants differ**: `core` runs **3–41** with **42+** pooled
(40 bins, four rows), `all` runs **3–30** with **31+** pooled (29 bins, three rows).
These two figures pack ten bins a row where the main figure takes six, and draw each row at
four fifths of its height: at six a row and a full row height they ran to 25 in (`core`) and
18 in (`all`) of canvas, and a figure that tall only reaches a page shrunk to a third, where
every box is a hairline. Ten a row and a flatter row halve that — 12.1 in and 9.3 in — at
the same canvas width, so a box keeps its size on the page. Ten fits because these labels
are one number wide where the main figure's are ranges (`15` against `9–10`), and the bins
are drawn at the same width either way; only how many sit on a row changes.
The rule is to keep splitting sizes while at least three quarters of the drawn arms hold
five fragments at that size, and to pool from the first size that fails — VoxBind and CoDE
both reach 41, while FuncBind is under five from 27 up and DiffSBDD and DecompDiff join it
at 31. At three quarters a single thin method may drop out of a bin (in `all`: FuncBind at
27, 29 and 30, DecompDiff at 28, AR at 29); its box is simply absent, its x tick stays, and
the run log names every bin nobody drew a box in. Rigid fragments reach 64 heavy atoms
(VoxBind; 38 for the crystal ligands), but the tail is thin — 0.03–1.0% of a method’s
fragments have 31 or more, 0.00–0.12% have 42 or more. Each row takes its own ceiling, the
smallest round number clearing that row’s quartiles (0.6 Å floor, four ticks); whiskers past
it are clipped and the box statistics are unaffected. Everything else (statistics, colours,
legend, five-fragment cutoff) matches the main figure. The crystal ligands clear the cutoff
only below 16 atoms, so every row past the second carries generated molecules alone.
`fragment-by-bin-expanded.csv` exports these bins, with `expanded_core` / `expanded_all` in
its binning column because the two pool from a different size; `fragment-by-bin.csv` is
unchanged.

**Type size on the wide variants.** The main `rigid-consistent-box-all` uses the
`interaction-pair-all` axis scale: 17.36 pt ticks, 19.36 pt axis names, and 2.64 pt
visible axes, with its existing canvas dimensions preserved. The other wide variants
scale type and rules with their canvas — 26.28 pt ticks and 29.31 pt axis names at
18 in — and mark themselves as already styled so the export boundary leaves them alone.
The `core` variants sit at the reference width and are drawn exactly as the rest of the tree
is; scaling only ever goes up. The method key is the one thing that does not scale: it is a
single row of nine names already as wide as the canvas, and scaling it makes `fit()` reserve
the overflow and squeeze the panels into the middle of the figure.

`python figures/draw.py fig-consistency-rigid-size-distribution` adds
`rigid-fragment-size-distribution-{core,all}.{png,pdf}` beside the boxes,
without a legend. Grouped bars use the same method colors and 3–13 / 14+ bins;
shares sum to 100% within each method over fragments with at least 3 heavy atoms.
All counts are included, including bins below the box plot’s five-fragment cutoff.
`fragment-size-distribution.csv` exports counts, denominators, and percentages.

The same command also writes
`rigid-fragment-size-distribution-expanded-{core,all}.{png,pdf}`: the same shares over
the expanded figure’s own bins, in the same rows of ten at the same width and the same four
fifths row height, so it reads directly under the boxes. Each row carries its own share ceiling — the 3-atom bin holds a third of
every method’s fragments and the 30+ bins hold tenths of a percent, and on one linear axis
the whole tail is a flat line. The axis stays linear rather than going log, because these
bars are shares and a log axis stops their heights being proportional to the numbers they
stand for; every row starts at zero and prints its ceiling. The crystal ligands’ tail bars
are single fragments — 204 of them have three heavy atoms or more, so one fragment is
0.49% and their bars out there are quantisation, not a distribution.
`fragment-size-distribution-expanded.csv` exports them with the same binning column.

The interaction fingerprints live in `fig-posecheck/` under an `interaction-` prefix,
because that is what produces them: PoseCheck runs ProLIF over the same pocket10 crop and
writes the fingerprint into the same `posecheck` block of the same `metrics.json` that
strain and clashes come from (`compute_baseline_interactions.py` imports PoseCheck directly
to make it for the five baselines, which carry no posecheck block of their own).
`draw.py interaction` still selects exactly those six.

A name on the command line may be a **figure id** (`fig-posecheck-clash-per-atom`) or a
whole **evaluation** (`posecheck`) — ids stay `fig-<eval>-<name>` so one figure is still
addressable on its own; only the output folder is the evaluation.

PNG to look at, PDF to place, with live text (`pdf.fonttype: 42` — Type 3 is what
journals reject and Illustrator mangles). `--formats png` skips the PDF when you only want
to look.

The names on disk are not the names the original builders used. `STEMS` in `draw.py` is the
map between the two trees — which is how the port was verified figure by figure. Families
call `save()` with the original name; the rename happens once, at the writing edge.

`fig-jsd-metric-summary` writes `fig-jsd/metric-summary.*`: five horizontal
bar-chart panels (atom type, all heavy-atom pair distances under 12 Å, mean
bond-type JSD, ring size, and ring count per molecule), with shared method order and colours, independent y-axis ranges, and a value above
each bar. Method ticks are hidden; panel names
appear beneath the x axes, and the JSD label appears only on the left. Axis styling
and axis-label sizes match the strain ECDF. No legend or reference line is drawn. Scores compare against the
CrossDocked train+test reference of distinct molecules. As in the existing JSD
protocol, values are Jensen–Shannon distances using natural logarithms. Ring-size
scores are computed from all observed ring-size counts, including sizes above nine.

The strain ECDF, per-atom median, landscape clash and redock median share fixed
axes bounds, 11 pt ticks, 16.1 pt y labels, 1.1 pt spines and uncropped canvases.

The metric summary retains the panel widths of the original twice-per-atom-median
canvas, with extra total width for larger inter-panel gaps; height is unchanged.
It uses 16.1 pt axis titles, 11 pt ticks, 1.1 pt axes and 0.9 pt grid.
Panel gaps are 0.704 inches (1.6 times the original); bar centres stay unchanged, with bar width 0.78 of their pitch; bar values omit the leading zero (e.g. `.094`).
Bar values use 7 pt text at a uniform offset above each bar. The Bond length panel reports
the mean over eight bond types. Each panel includes zero and its upper limit as ticks.

PoseCheck/JSD is exported as five standalone figures: ECDF, median strain,
horizontal clash box, median redocking RMSD, and JSD metric summary. These PDFs
are mirrored under `_download`; local composite generation was removed on 2026-09-23.

Ring-count JSD uses the full `n_rings_counts` histogram (including zero rings),
without pooling the high-count tail; it uses the same natural-log JS distance
and distinct-molecule reference as the other summary metrics.

## Where the code is

**`draw.py` holds only what figures of two or more folders share**: the house style and
the export-time axis style (`apply`/`prepare`, applied in `save()`), the palette
(`color`, `pale`, `highlight`, `dash`, `display`, `ALIASES`, `METHOD_ORDER`), the
per-molecule pose loader, the Vina bundle readers, the registry (`@figure`, `FIGURES`,
`figure_folder`, `save`, `write_csv`, `STEMS`), the manuscript download sync and the
command line. **A `_parts/fig_X` module holds everything only `fig-X/` uses** and imports
the shared names explicitly (`from draw import …`). Which statement went where (2026-10-06)
was decided by use, not by topic: a helper or constant one folder's figures need lives in
that folder's module; one that two folders need is in `draw.py`, together with everything
it in turn depends on. Running `python figures/draw.py` imports every `_parts/fig_*`
module, which is what registers their figures.

Adding a figure is one function in the module of the folder it draws into:

```python
@figure("fig-posecheck-something", folder="fig-posecheck/strain-energy", needs=("metrics.json",))
def draw_something(out):
    """One line, shown by --list."""
    ...
    save(fig, out, "stem_name")       # PNG + PDF, house axis style applied
```

If the new figure needs a helper another folder's module already has, move that helper to
`draw.py` rather than importing one module from another. A standalone script that sets
global matplotlib style registers with `isolate=True`.

The **uncertainty analyses** keep their own `main()` and can recompute (fit decoders,
rewrite tables) when run directly from `figures/`:
`cd figures && python -m _parts.fig_uncertainty.manuscript`. `draw.py uncertainty` calls
them in their plots-only mode. Their shared destinations are in
`_parts/fig_uncertainty/paths.py`; see [fig-uncertainty/README.md](fig-uncertainty/README.md).

`_tools/` holds the analyses no figure imports (`bootstrap_uncertainty_ranking.py`,
`review_uncertainty_metrics.py`, `summarize_uncertainty_three_models.py`; run them as
scripts), `jsd-tables.ipynb` (the JSD numbers as tables and LaTeX) and
`compare_outputs.py`, which compares the figures on disk with a git revision after a redraw.

## What this does NOT do

It draws; it does not re-score. The per-molecule inputs — `metrics.json` under each run
tree, `eval_docking_results_full79.json`, the PoseCheck/PoseBusters exports — come from
`task2-ligand_generation/scripts/73_evaluate_samples.sh`, `85_fill_pose_eval_4runs.sh`,
`86_pose_eval_baselines.sh` and the docking driver. The figure-data builders, which are
not part of this repository, remain the source of truth for **data**; this tree is
the source of truth for **pictures**.

The expensive per-molecule load (~60k molecules over eight run trees) happens **once per
process** no matter how many figures ask for it — which is why `draw_all.sh` draws an
evaluation's figures in one process.

## House style

**Axis style (2026-09-22).** The approved `uncertainty-combined-contactb-combined`
proportions are applied to every axes figure at the export boundary (`apply()` in
`draw.py`, called by `save()` through `prepare()`): visible spines and tick strokes
**2.64 pt**, numeric ticks **17.36 pt**, axis titles and categorical x labels **19.36 pt**,
grid strokes **1.2 pt**. Families keep their grids, data, limits, palettes, legends and
annotations. Cramped layouts and outer margins are enlarged rather than labels dropped.
`VOXBIND_STYLE_AUDIT=/path/log.jsonl` records the exported axis sizes, clipping and tick
overlaps; `VOXBIND_FIGURE_RETAINED=<inventory.json>` restricts saving to listed files
(what `--existing-only` writes).

**Opacity.** Two shared levels: `DATA_ALPHA = 0.75` for translucent data and
`BACKGROUND_ALPHA = 0.15` for background fills and grids; opaque elements stay opaque.
**Method order** is `METHOD_ORDER` / `ordered_methods()`: AR → Pocket2Mol → DiffSBDD →
TargetDiff → DecompDiff → FuncBind → VoxBind → CoDE (CoE between VoxBind and CoDE).

## Palette

[palette-methods.png](fig-palette/palette-methods.png) is the official **Color / Highlight**
sheet (`python figures/draw.py fig-palette-methods`). Its source of truth is
`fig-palette/method-palette-approved.json` — *not in git, see above* — loaded before any
figure-family constant. `color()` and `pale()` return the approved Color for active
methods, `highlight()` the Highlight; `soft()` is kept for historical compatibility.
Inactive methods are hatched on the sheet and keep their legacy colors in archived plots.
The former sheet is `palette-methods-v1.*`. The proposals (`fig-palette-options`,
`-candidate`, `-eight-colors`, `-green-orange-blue`) are previews and never change the
approved palette.

## Manuscript downloads

[_download/](_download/README.md) holds the figures `Density4Drug.zip`'s
`ICLR2027/0_main.tex` uses, under their LaTeX file names, so the folder can be copied over
`ICLR2027/resource/image/`. `save()` copies a mapped figure there the moment it is redrawn;
`draw_all.sh` syncs all of them at the end.

```bash
python figures/draw.py --inventory figures/Density4Drug.zip   # (re)build the map, then sync
python figures/draw.py --sync-downloads                        # copy every mapped figure
```

The map is `figures/manuscript/iclr2027-figures.json` (not in git, like every `*.json`);
without it nothing is mirrored.

## Restructured 2026-10-06

`draw.py` used to be ONE 14,075-line file concatenated from `_parts/*.py` by
`_parts/assemble.sh`, with the uncertainty, palette and overview plots as separate scripts
beside it. Now the shared code is `draw.py` itself (~2,200 lines), each figure folder's code
is its own module, and `draw_all.sh` replaces `_parts/redraw_*.sh` and `_parts/runners/`.
Checked on the code, since this box has no figure inputs:

* every top-level statement of the old `draw.py` is in exactly one new file, verbatim
  (944 of 960); the other 16 are the intended edits — the PNG+PDF default, the
  `palette/`→`fig-palette/` and `legend/`→`fig-legend/` folders, the `capture` / `isolate`
  flags, `--figures-only` in `write_csv`, two CLI options read as `draw.<name>`, the inlined
  `publication_style` and download sync, and the rewritten CLI;
* the registry is identical: all 70 figures keep their ids, docstrings and output folders,
  and 15 more are registered (the uncertainty analyses, palette proposals, overviews and the
  qual sheets, previously separate scripts);
* pyflakes finds no undefined name; every moved module imports; the pocket, affinity and
  qual viewers serve from their new place (the ring explorer needs another box's python).

Not yet checked: the rendered files. On a box with the inputs, run `bash figures/draw_all.sh`
and compare against the committed figures (`git diff --stat figures/`, and
`python figures/_tools/compare_outputs.py` for a pixel-level PNG / PDF report).

## How the 2026-09-13 port was verified

Every registry figure was checked against the one its original figure-data builder drew,
by sha256: **69 byte-identical, 16 stale-committed, 0 differing,
2 not drawn**. The 16 "stale-committed" are cases where the CHECKED-IN PNG was the wrong
reference (inputs re-scored after commit, pre-2026-09-10 legend labels, three treatments
that sit behind `--all` in the builder, and one file committed under matplotlib 3.7.3).
`STEMS` in `draw.py` is the map between the two trees' filenames, which is what made that
comparison possible. A PNG's `tEXt` chunk names the matplotlib that wrote it; read the
committed side out of git (`git cat-file blob HEAD:<path>`), never off disk.

## Known gap

`fig-ensemble-6set-3metric` and `fig-ensemble-vs-alone` cannot be drawn without
`ensemble_results.json` (written by `gen_ensemble_data.py` outside this repository,
not shared). They fail with the missing path; drop the JSON in
`figures/_data/fig-ensemble/` and they draw.

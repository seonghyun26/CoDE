---
title: Drug Design Viewer
emoji: 💊
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: Generated ligands per CrossDocked pocket, side by side
license: cc-by-nc-4.0
---

# Drug Design Viewer

One CrossDocked pocket at a time, one generated ligand per method, side by side —
square 3D panels with synchronised cameras, each molecule's Vina / quality / pose
numbers beside it, and per-pocket statistics with density charts.

A static page: `index.html` reads everything from `data/` next to it.

- `data/manifest.json` — methods, pockets, metric schema, poses, provenance
- `data/targets/target_NN.json` — every method's molecules for a pocket, columnar
- `data/struct/target_NN/` — the 10 Å pocket crop, the crystal ligand, and one SDF
  per method (and per pose, where a method has docked / minimized poses)

Built by `web/build.py` in the VoxBind repo and published with
`web/deploy_space.sh`; edit there, not here.

## License

The molecules shown are released under CC BY-NC 4.0, the most restrictive license among their
sources: DecompDiff, whose molecules were generated with its code and checkpoint, is CC BY-NC 4.0;
AR, Pocket2Mol, DiffSBDD and TargetDiff are MIT; FuncBind is Apache-2.0. CrossDocked2020, the source
of the pockets and crystal ligands, states no license; its structures derive from the Protein Data
Bank (CC0). The viewer's code (`index.html`, `style.css`) is Apache-2.0, like the VoxBind repository.

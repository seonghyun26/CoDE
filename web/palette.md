# Drug Design Viewer — palette

Generated from `web/src/index.html` (`:root` tokens) and `web/build.py` (`METHOD_COLORS`).
The CSS variables are the source of truth; regenerate this file rather than editing it.

## UI and 3D tokens

| token | light | dark | role |
|---|---|---|---|
| `--accent` | `#8092F6` | `#8092F6` | links, focus rings |
| `--accent-strong` | `#4761F2` | `#4761F2` | primary action (Capture all), selected controls, crop box |
| `--accent-tint` | `rgba(128,146,246,.12)` | `rgba(128,146,246,.2)` | selected-row / active-toggle fill |
| `--ink` | `#1D1D1F` | `#F5F5F7` | body text, headings, ligand carbons in 3D |
| `--muted` | `#86868B` | `#98989D` | secondary text, metric names |
| `--faint` | `#AEAEB2` | `#6E6E73` | tertiary text, placeholders, disabled |
| `--bg` | `#F5F5F7` | `#000000` | page background (panels float on it) |
| `--panel` | `#FFFFFF` | `#1C1C1E` | panel / card surface |
| `--fill` | `#F5F5F7` | `#2C2C2E` | control fill |
| `--fill-hi` | `#EBEBED` | `#3A3A3C` | control fill, hover |
| `--sep` | `rgba(0,0,0,.08)` | `rgba(255,255,255,.1)` | rules between table families |
| `--hair` | `rgba(0,0,0,.05)` | `rgba(255,255,255,.07)` | hairlines inside panels |
| `--ok` | `#248A3D` | `#30D158` | PoseBusters pass |
| `--bad` | `#D70015` | `#FF453A` | PoseBusters fail |
| `--cover` | `#EDEFF1` | `#232527` | grey cover over a panel with no structure for the pose |
| `--mol-protein` | `#8092F6` | `#8092F6` | 3D · crop protein cartoon |
| `--mol-context` | `#C9D0FB` | `#C9D0FB` | 3D · deposited entry, faint context |
| `--mol-box` | `#4761F2` | `#4761F2` | 3D · the 10 Å crop box |
| `--mol-ligand-c` | `#1D1D1F` | `#1D1D1F` | 3D · reference ligand carbons |
| `--mol-ghost` | `#A3A8AE` | `#A3A8AE` | 3D · reference ghost behind each method's ligand |
| `--mol-surface` | `#FFFFFF` | `#FFFFFF` | 3D · pocket surface |
| `--int-hbond` | `#2F80ED` | `#2F80ED` | 3D · H-bond dumbbells (incl. backbone) and their eyes |
| `--int-hydrophobic` | `#F2994A` | `#F2994A` | 3D · hydrophobic contacts and their eyes |
| `--int-pistack` | `#27AE60` | `#27AE60` | 3D · π-stacking |
| `--int-cationpi` | `#9B51E0` | `#9B51E0` | 3D · cation–π |
| `--int-ionic` | `#EB5757` | `#EB5757` | 3D · ionic / salt bridge |
| `--int-halogen` | `#2DB7A3` | `#2DB7A3` | 3D · halogen bond |
| `--int-metal` | `#8D6E63` | `#8D6E63` | 3D · metal coordination |

Where light and dark agree the token is not overridden in dark mode. 3D tokens
repeat their light value on purpose: the NGL canvases stay white in both themes.

## Method colours

One hue per method, used for its ligand in 3D. Kept in the family of
the de novo Vina chart's colours so figures and viewer agree.

| method | colour |
|---|---|
| AR | `#9b59b6` |
| Pocket2Mol | `#e67e22` |
| DiffSBDD | `#e74c3c` |
| TargetDiff | `#3498db` |
| DecompDiff | `#34495e` |
| DecompDiff-prior25 | `#e84393` |
| FuncBind | `#16a085` |
| VoxBind | `#1abc9c` |
| VoxBind-Ours | `#2ecc71` |
| Ours-v2 | `#2ecc71` |
| VoxBind-base-ep350-n100 | `#52bb7e` |

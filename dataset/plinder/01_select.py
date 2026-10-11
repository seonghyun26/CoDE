"""01_select.py — PLINDER selection: pick the complexes, freeze the list.

Step 1  COMPLEX filter   — is this a real, representable protein+ligand complex?
Step 2  DENSITY filter   — does the experimental density actually match the ligand?
Step 3  FREEZE CSV       — pin the survivors to splits/plinder/<version>/ with a sha256
                           so every server builds from the same list.

Selection is metadata-only (the PLINDER annotation table). The raw mmCIF and 2Fo-Fc maps
are fetched later by 03_download into the shared caches data/cif + data/ccp4, and cropped
by 04_build_corpus into the per-element corpus.

    python dataset/plinder/01_select.py --version v2 --dry_run   # funnel only
    python dataset/plinder/01_select.py --version v2             # build + freeze
    python dataset/plinder/01_select.py --version v2p1

MERGED 2026-10-01. This absorbs `03a_plinder_select.py`, which was a second selection script
with its own filter set, its own dedup, and a freeze that wrote a FLAT `splits/plinder/`
directory under the hardcoded name `plinder_v1`. Every consumer reads
`splits/plinder/<version>/`, so that freeze could not re-pin anything, and the plinder `s1`
stage of the build pipeline invoked it with bare defaults -- a resolution cap of 2.5 A,
single-ligand-only, train-split-only, dedup by PDB -- which selects a completely different
corpus from the frozen one. That stage now runs this script with an explicit `--version`.

What came across from 03a: the annotation-table download, the `pdb` and `ecod` dedup modes,
`--n`/`--seed` subsampling, and the `single_ligand` / `plinder_splits` / `allow_covalent`
filters, which existed here as recipe keys that nothing read. They are real stages now, and
they are appended only when the recipe asks for them, so every frozen recipe stays a no-op.

COLUMNS ARE VERSION-KEYED. The frozen column list drifted: v1, v2 and the decontaminated
v2.4-v2.7 views carry 14 columns, while v2.1 and v3 carry 15, because pocket RSCC was added
for v2.1 and v2 was never re-frozen. `keep_cols` picks the right list per recipe, so re-running
v2 emits 14 columns rather than silently widening the CSV.

V2 IS NO LONGER BYTE-REPRODUCIBLE, AND THAT IS EXPECTED. Verified 2026-10-01: a v2 re-run
matches the frozen funnel exactly through the first six stages, then diverges at
`exclude downstream-test` (166,218 against the frozen 166,448), ending at 113,796 instances /
41,763 PDB rather than 113,976 / 41,833. The freeze itself records why: `pdbbind/index.csv`
now hashes differently from the sha256 pinned in `plinder_inputs.json`, because the PDBbind
general-set expansion grew the held-out test/val set, which excludes 87 more PLINDER PDBs.
Nothing in this script can recover that, and it should not try -- the frozen CSV is the source
of truth and the recorded hashes exist to make exactly this drift visible.

Two cosmetic differences, in plinder_funnel.json only, never in the CSV: v2's recorded funnel
labels its RSCC stage `RSCC >= 0.8` where this prints `ligand RSCC >= 0.8`, and v2's recorded
`filters` block predates the knobs added since.
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, hashlib, json, re, urllib.request
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
import torch; torch.set_num_threads(2)

REPO_ROOT = Path(__file__).resolve().parents[2]   # repo root (VoxBind/)
VOX = REPO_ROOT / "voxbind"
DATA = REPO_ROOT / "dataset" / "data"
PLINDER = DATA / "plinder"

GCS = "https://storage.googleapis.com/plinder/2024-06/v2"
ANNOT_URL = f"{GCS}/index/annotation_table.parquet"
SPLIT_URL = f"{GCS}/splits/split.parquet"

# Frozen column lists. 14-col predates the pocket-RSCC column; 15-col includes it.
KEEP_15 = ["entry_pdb_id", "system_id", "system_biounit_id", "ligand_asym_id", "ligand_ccd_code",
           "ligand_rdkit_canonical_smiles", "entry_resolution",
           "system_ligand_validation_average_rscc", "system_pocket_validation_average_rscc",
           "system_ligand_validation_average_occupancy",
           "ligand_num_heavy_atoms", "ligand_molecular_weight",
           "system_pocket_ECOD", "system_pocket_UniProt", "split"]
KEEP_14 = [c for c in KEEP_15 if c != "system_pocket_validation_average_rscc"]
KEEP_COLS = {"v2": KEEP_14, "full": KEEP_15}

# Recipes. Every key is read; a zero / False / empty value makes its stage a no-op, which is
# how the frozen recipes stay reproducible as knobs get added.
#   min_rscc          ligand RSCC floor
#   min_pocket_rscc   pocket RSCC floor         (0 = skip)
#   max_res           resolution cap in A       (0 = skip; note res is NaN for some entries,
#                                                and a NaN fails the cap, so this is not the
#                                                same as a very large cap)
#   min/max_heavy     ligand heavy-atom window
#   single_ligand     keep only systems with exactly one proper ligand chain
#   allow_covalent    False appends the !covalent stage
#   plinder_splits    keep only these PLINDER splits ([] = all)
#   instance_dedup    one row per (PDB, ligand_asym_id), best RSCC first  (03a did this
#                     unconditionally; the frozen sets were built without it, so it is off)
#   dedup             group dedup: none | pdb_ccd | pdb | ecod
#   row_order         annotation (as the table orders them) | rscc_desc
#   keep_cols         which frozen column list to emit
RECIPES = {
    "v2":   dict(min_rscc=0.80, min_pocket_rscc=0.0, max_res=0, min_heavy=6, max_heavy=50,
                 single_ligand=False, allow_covalent=False, plinder_splits=[],
                 instance_dedup=False, dedup="none", row_order="annotation", keep_cols="v2"),
    # v2.1: CLEAN rebuild after the 260718 audit found v2's 6.5x is illusory (59%
    # same-ligand-same-PDB duplicates, +23% worse than 2.5 A, pocket density unfiltered).
    "v2p1": dict(min_rscc=0.80, min_pocket_rscc=0.80, max_res=2.5, min_heavy=6, max_heavy=50,
                 single_ligand=False, allow_covalent=False, plinder_splits=[],
                 instance_dedup=False, dedup="pdb_ccd", row_order="annotation", keep_cols="full"),
    # v3: clean and scaled. res <= 3.0 rather than 2.5 for size, and dedup="none" because the
    # real dedup is the DENSITY-AWARE pose dedup applied at build time
    # (04_build_corpus --pose_dedup_tau), which keeps density-distinct poses and drops
    # symmetry copies.
    "v3":   dict(min_rscc=0.80, min_pocket_rscc=0.80, max_res=3.0, min_heavy=6, max_heavy=50,
                 single_ligand=False, allow_covalent=False, plinder_splits=[],
                 instance_dedup=False, dedup="none", row_order="annotation", keep_cols="full"),
}
BUILD = dict(shuffle_seed=1234, val_tail=100, max_len=30,
             norm_version="v6_arcsinh_z", pocket_radius=10.0, frame="deposited (transform=None)")

# Overridable from the CLI; anything else in a recipe is structural.
_NUMERIC_OVERRIDES = ("min_rscc", "min_pocket_rscc", "max_res", "min_heavy", "max_heavy")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Download url -> dest if not already there (atomic via a .tmp sibling)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  downloading {dest.name} from {url} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"  -> {dest} ({dest.stat().st_size/1e6:.0f} MB)")


_CROSSDOCKED_ID_RE = re.compile(
    r"(?:^|/)(?P<receptor>[0-9][A-Za-z0-9]{3})_[^_/]+_rec_"
    r"(?P<ligand_source>[0-9][A-Za-z0-9]{3})_",
    re.IGNORECASE,
)


def crossdocked_test_pdb_ids(data) -> set[str]:
    """Extract real receptor + ligand-source PDB IDs, not target-family prefixes."""
    ids: set[str] = set()
    for index, (pocket, ligand) in enumerate(data):
        matches = [
            _CROSSDOCKED_ID_RE.search(str(record["id"]))
            for record in (pocket, ligand)
        ]
        if any(match is None for match in matches):
            raise ValueError(f"cannot parse CrossDocked test row {index}")
        parsed = [
            (match.group("receptor").lower(), match.group("ligand_source").lower())
            for match in matches
        ]
        if parsed[0] != parsed[1]:
            raise ValueError(f"CrossDocked test row {index} has mismatched IDs: {parsed}")
        ids.update(parsed[0])
    return ids


def leakage_holdout():
    """4-char lowercase PDB ids held out by any downstream eval, + their file hashes."""
    excl, hashes = set(), {}
    idx = DATA / "pdbbind" / "index.csv"
    if idx.exists():
        d = pd.read_csv(idx)
        excl |= set(d.loc[d.new_split.isin(["test", "val"]), "pdb_id"].astype(str).str.lower())
        hashes["pdbbind_index_csv"] = {"path": str(idx.relative_to(REPO_ROOT)), "sha256": sha256(idx)}
    pool = DATA / "misato" / "pool_pids.txt"
    if pool.exists():
        excl |= {l.strip().lower() for l in pool.read_text().splitlines() if l.strip()}
        hashes["misato_pool_pids"] = {"path": str(pool.relative_to(REPO_ROOT)), "sha256": sha256(pool)}
    for cd in [DATA / "data_test.pt", DATA / "pretrain" / "data_test.pt"]:
        if cd.exists():
            data = torch.load(cd, weights_only=False)
            excl |= crossdocked_test_pdb_ids(data)
            hashes["crossdocked_test_pt"] = {"path": str(cd.relative_to(REPO_ROOT)), "sha256": sha256(cd)}
            break
    return excl, hashes


def apply_dedup(sel: pd.DataFrame, recipe: dict) -> pd.DataFrame:
    """Instance dedup, then group dedup. Both deterministic, so every server freezes the
    same rows. The tie-break differs per mode on purpose: `pdb_ccd` sorts lexicographically,
    which is what v2.1 and v3 were frozen with, while `pdb` and `ecod` keep the best-RSCC
    representative, which is what 03a did."""
    if recipe["instance_dedup"]:
        pre = len(sel)
        rscc = pd.to_numeric(sel.system_ligand_validation_average_rscc, errors="coerce")
        sel = (sel.assign(_rscc=rscc).sort_values("_rscc", ascending=False)
                  .drop_duplicates(["entry_pdb_id", "ligand_asym_id"])
                  .drop(columns="_rscc"))
        print(f"\n[instance dedup] {pre:,} -> {len(sel):,} (one row per PDB + ligand chain)")

    mode = recipe["dedup"]
    if mode == "none":
        return sel.reset_index(drop=True)
    pre = len(sel)
    if mode == "pdb_ccd":
        sel = (sel.sort_values(["entry_pdb_id", "ligand_ccd_code", "ligand_asym_id"])
                  .drop_duplicates(["entry_pdb_id", "ligand_ccd_code"], keep="first"))
    elif mode in ("pdb", "ecod"):
        rscc = pd.to_numeric(sel.system_ligand_validation_average_rscc, errors="coerce")
        ranked = sel.assign(_rscc=rscc).sort_values("_rscc", ascending=False)
        if mode == "pdb":
            sel = ranked.drop_duplicates(["entry_pdb_id"]).drop(columns="_rscc")
        else:
            # ECOD nan/empty means unclustered: keep those rows rather than collapsing them.
            has = ranked.system_pocket_ECOD.notna() & ranked.system_pocket_ECOD.ne("")
            sel = pd.concat([ranked[has].drop_duplicates(["system_pocket_ECOD"]),
                             ranked[~has]]).drop(columns="_rscc")
    else:
        raise ValueError(f"unknown dedup mode {mode!r}")
    print(f"\n[dedup {mode}] {pre:,} -> {len(sel):,} ({pre - len(sel):,} redundant copies dropped)")
    return sel.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v2", choices=list(RECIPES),
                    help="selection recipe and freeze directory")
    ap.add_argument("--dry_run", action="store_true", help="print the funnel, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing frozen selection (default: compare and keep it)")
    ap.add_argument("--no_download", action="store_true",
                    help="fail instead of fetching a missing annotation table or split")
    ap.add_argument("--dedup", choices=["none", "pdb_ccd", "pdb", "ecod"],
                    help="override the recipe's group dedup")
    ap.add_argument("--instance_dedup", action="store_true",
                    help="also keep one row per (PDB, ligand chain), best RSCC first")
    ap.add_argument("--n", type=int, default=0, help="subsample to N rows (0 = all)")
    ap.add_argument("--seed", type=int, default=42, help="subsample seed")
    for key in _NUMERIC_OVERRIDES:
        ap.add_argument(f"--{key}", type=float, default=None, help=f"override recipe {key}")
    args = ap.parse_args()

    recipe = dict(RECIPES[args.version])
    for key in _NUMERIC_OVERRIDES:
        value = getattr(args, key)
        if value is not None:
            recipe[key] = int(value) if key in ("min_heavy", "max_heavy") else value
    if args.dedup:
        recipe["dedup"] = args.dedup
    if args.instance_dedup:
        recipe["instance_dedup"] = True
    overridden = recipe != RECIPES[args.version]

    freeze = VOX / "splits" / "plinder" / args.version
    print(f"[recipe] {args.version}: {recipe}")
    if overridden:
        print("[recipe] CLI overrides applied — this will not reproduce the frozen selection")
    print()

    annot = PLINDER / "index" / "annotation_table.parquet"
    split_pq = PLINDER / "splits" / "split.parquet"
    for url, dest in ((ANNOT_URL, annot), (SPLIT_URL, split_pq)):
        if not dest.exists():
            if args.no_download:
                raise SystemExit(f"[fatal] missing {dest} (--no_download)")
            fetch(url, dest)

    excl, leak_hashes = leakage_holdout()
    print(f"[leakage] {len(excl):,} held-out PDB ids\n")

    cols = list(dict.fromkeys(KEEP_15[:-1] + [
        "entry_determination_method", "ligand_is_proper", "ligand_is_ion", "ligand_is_artifact",
        "ligand_is_cofactor", "ligand_is_covalent", "ligand_is_invalid", "ligand_is_rdkit_loadable",
        "ligand_num_unresolved_heavy_atoms", "system_proper_num_ligand_chains"]))
    df = pq.read_table(annot, columns=cols).to_pandas()
    spl = pq.read_table(split_pq).to_pandas()
    scol = "split" if "split" in spl.columns else [c for c in spl.columns if "split" in c.lower()][0]
    key = "system_id" if "system_id" in spl.columns else [c for c in spl.columns if c.lower().endswith("system_id")][0]
    df["split"] = df.system_id.astype(str).map(dict(zip(spl[key].astype(str), spl[scol].astype(str)))).fillna("none")

    res = pd.to_numeric(df.entry_resolution, errors="coerce")
    rscc = pd.to_numeric(df.system_ligand_validation_average_rscc, errors="coerce")
    prscc = pd.to_numeric(df.system_pocket_validation_average_rscc, errors="coerce")
    nh = pd.to_numeric(df.ligand_num_heavy_atoms, errors="coerce")
    un = pd.to_numeric(df.ligand_num_unresolved_heavy_atoms, errors="coerce")
    nlig = pd.to_numeric(df.system_proper_num_ligand_chains, errors="coerce")
    b = lambda c: df[c].eq(True)

    complex_subs = [
        ("all",                              pd.Series(True, index=df.index)),
        ("X-ray diffraction",                df.entry_determination_method.eq("X-RAY DIFFRACTION")),
        ("proper & !ion/artifact/cofactor",  b("ligand_is_proper") & ~b("ligand_is_ion") & ~b("ligand_is_artifact") & ~b("ligand_is_cofactor")),
        ("!invalid & rdkit_loadable",        ~b("ligand_is_invalid") & b("ligand_is_rdkit_loadable")),
        (f"{recipe['min_heavy']} <= heavy <= {recipe['max_heavy']}", nh.ge(recipe['min_heavy']) & nh.le(recipe['max_heavy'])),
    ]
    if not recipe["allow_covalent"]:
        complex_subs.append(("!covalent", ~b("ligand_is_covalent")))
    if recipe["single_ligand"]:
        complex_subs.append(("single proper ligand chain", nlig.eq(1)))
    if recipe["plinder_splits"]:
        complex_subs.append((f"split in {recipe['plinder_splits']}",
                             df.split.isin(recipe["plinder_splits"])))
    complex_subs.append(("exclude downstream-test", ~df.entry_pdb_id.str.lower().isin(excl)))

    density_subs = [
        ("unresolved_heavy == 0",                un.eq(0)),
        (f"ligand RSCC >= {recipe['min_rscc']}", rscc.ge(recipe['min_rscc'])),
    ]
    if recipe["min_pocket_rscc"] > 0:
        density_subs.append((f"pocket RSCC >= {recipe['min_pocket_rscc']}", prscc.ge(recipe["min_pocket_rscc"])))
    if recipe["max_res"] and recipe["max_res"] > 0:
        density_subs.append((f"resolution <= {recipe['max_res']} A", res.le(recipe["max_res"])))

    stages = [("STEP 1 — COMPLEX filter", complex_subs), ("STEP 2 — DENSITY filter", density_subs)]
    print(f"{'stage':<40}{'inst':>12}{'removed':>11}{'uniq_pdb':>11}")
    print("-" * 74)
    mask = pd.Series(True, index=df.index); prev = None; funnel = []
    for group, subs in stages:
        print(group)
        for name, cond in subs:
            mask = mask & cond.reindex(df.index).fillna(False)
            ni = int(mask.sum()); npdb = int(df.loc[mask, "entry_pdb_id"].nunique())
            rem = "" if prev is None else f"{ni - prev:+,}"
            print(f"  {name:<38}{ni:>12,}{rem:>11}{npdb:>11,}")
            funnel.append({"stage": name, "ligand_instances": ni, "unique_pdb": npdb}); prev = ni

    sel = df.loc[mask, KEEP_COLS[recipe["keep_cols"]]].reset_index(drop=True)
    sel = apply_dedup(sel, recipe)

    if recipe["row_order"] == "rscc_desc":
        order = pd.to_numeric(sel.system_ligand_validation_average_rscc, errors="coerce")
        sel = sel.assign(_o=order).sort_values("_o", ascending=False).drop(columns="_o").reset_index(drop=True)
    if args.n and args.n < len(sel):
        sel = sel.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        print(f"[subsample] n={args.n} seed={args.seed}")

    n, npdb, nccd = len(sel), sel.entry_pdb_id.nunique(), sel.ligand_ccd_code.nunique()
    print(f"\n{args.version} selection: {n:,} instances / {npdb:,} PDB / {nccd:,} chemotypes")

    if args.dry_run:
        print("\n[dry_run] nothing written."); return

    freeze.mkdir(parents=True, exist_ok=True)
    csv = freeze / "plinder_selected.csv"
    # The freeze is the source of truth and a re-run can differ (v2 does: pdbbind/index.csv grew),
    # so an existing freeze is compared, never replaced; --force re-pins it.
    if csv.exists() and not args.force:
        if sel.to_csv(index=False) == csv.read_text():
            print(f"\n[freeze] identical to the frozen {csv}; nothing written.")
        else:
            old_n = sum(1 for _ in csv.open()) - 1
            print(f"\n[freeze] DIFFERS from the frozen {csv} ({old_n:,} -> {n:,} rows). Kept the "
                  f"freeze; downstream stages read it. Re-run with --force to re-pin.")
        return
    sel.to_csv(csv, index=False)
    (freeze / "plinder_funnel.json").write_text(json.dumps(
        {"funnel": funnel, "n_selected": int(n), "n_unique_pdb": int(npdb), "n_chemotypes": int(nccd)}, indent=2))
    inputs = {
        "name": f"plinder_{args.version}",
        "description": f"Frozen PLINDER {args.version} ligand-matched density pretraining selection.",
        "plinder_bucket": "gs://plinder/2024-06/v2",
        "selection_csv": "plinder_selected.csv",
        "selection_sha256": sha256(csv),
        "n_selected": int(n),
        "filters": recipe,
        "build": BUILD,
        "leakage_holdout": leak_hashes,
        "note": "Built via dataset/plinder/01_select.py. Downloads go to the shared data/cif + "
                "data/ccp4 caches; 04_build_corpus writes the tuples and manifest. Re-pin "
                "deliberately, with a version bump.",
    }
    (freeze / "plinder_inputs.json").write_text(json.dumps(inputs, indent=2))
    print(f"\n[frozen] {freeze}/  (selected.csv sha256 {inputs['selection_sha256'][:12]}…)")


if __name__ == "__main__":
    main()

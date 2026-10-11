"""04_build_corpus.py — build a PLINDER OTF pretraining corpus from the SHARED caches.

Replaces four builders that differed only in the channel layout they asked the parser for, or
in whether they parsed at all:

    04_build.py                   ->  --layout roleblob
    04_build_perelem.py           ->  --layout perelem      (the default)
    04b_build_atomblob8.py        ->  --layout atomblob8 --jobs 80
    04c_build_v2p1_atomblob7.py   ->  --filter_from ... --max_lig_ch 6   (see FILTER MODE)

All parse through 03c's cif parser and the SHARED data/cif + data/ccp4 caches, so no structure
or map is ever re-downloaded. Output is pure OTF: tuples plus a manifest, no frozen crops. The
laydown (shuffle(1234) -> drop VAL_SZ -> max_len filter) mirrors DatasetCrossDockedDensity so
manifest row i aligns with dataset index i.

    python dataset/plinder/04_build_corpus.py --limit 300          # smoke
    python dataset/plinder/04_build_corpus.py                      # the v2 corpus
    python dataset/plinder/04_build_corpus.py --layout roleblob
    python dataset/plinder/04_build_corpus.py --layout atomblob8 --jobs 80
    python dataset/plinder/04_build_corpus.py --version v2p1

LAYOUTS. `perelem` is the default and produces the corpus the training defaults read, through
the v2.7 decontaminated view of it: 8 ligand channels (C,O,N,S,F,Cl,P plus "other" for the
0.25% rare tail) and 4 pocket channels (C,N,O,S). ATOMBLOB7 is the standard the encoder
actually consumes -- the loader's `n_lig_ch` default of 7 means channel 7 is never fed in, so
the tuples are written with the tail and read without it. `atomblob8` parses identically but
writes only tuples, because it is position-aligned with the unsuffixed corpus and reuses that
manifest and box. `roleblob` keeps every element but discards element identity (all atoms on
channel 0), so it is an ablation.

FILTER MODE builds atomblob7 by removing complexes rather than by parsing. `--filter_from`
loads an existing tuple file, drops every complex with a ligand atom above `--max_lig_ch`, and
then runs the normal laydown so a fresh manifest stays aligned. That is what produced v2.1: v2
minus the ~5% of complexes with an out-of-vocab heavy atom, leaving a corpus that needs only
the 7-element vocabulary. `--resample_from` inherits a source density recipe.

MERGED 2026-10-01 BY CODE READING, NOT BY COMPARISON. The outputs of the two scripts folded in
here do not exist on this machine -- `data_train_plinder_v2_atomblob8.pt`,
`data_train_plinder_v2p1_atomblob7.pt` and `xray_resample_plinder_v2p1/` are all absent -- so
the `atomblob8` and filter paths could not be checked against what they originally produced.
The parse flags, output names, laydown and alignment check were carried over literally. The
`roleblob` and `perelem` paths WERE verified against the corpora on disk: output naming across
five version/layout combinations, and `train_time_note` byte-identical in four directories.

Two deliberate repairs to the folded-in code, both of which only ever made things worse:
  1. The filter path refuses an empty corpus. 04c dropped that guard and would have crashed
     inside the laydown instead of saying what went wrong.
  2. `--resample_from` rewrites `train_time_note` and records `source_resample_dir` rather than
     copying the source recipe verbatim. 04c copied v2's file unchanged, so the recipe sitting
     in a v2.1 directory described the diverse single-channel corpus while the tuples beside it
     were 7-channel. Descriptive only, but false.

NOT MERGED. `build_v4_atomblob7_otf.py` stays separate: it flips `ok` bits on an existing
manifest and pins provenance hashes, a different job from building tuples. `05_make_boxes.py`
is the next stage, not a duplicate of this one.

NORMALISED vs the originals, from the earlier two-way merge. The frozen-selection CSV must
exist (04_build's check; 04_build_perelem only verified the hash when plinder_inputs.json
happened to be there). The launch hint prints `data_file=pretrain/...`, which 04_build printed
without the prefix, not a path the loader accepts. Each layout's `train_time_note` is kept
verbatim, including its hardcoded "v2" wording: every `_perelem` directory on disk carries
that exact string, even the v2p1 and v3 ones, so interpolating the version would make a re-run
differ from the corpora the published runs used.
"""
import gemmi  # noqa: F401 — before torch (shared-lib load order)
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, hashlib, importlib.util, json, random
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm
torch.set_num_threads(2)

REPO_ROOT = Path(__file__).resolve().parents[2]   # repo root (VoxBind/)
VOX_ROOT = REPO_ROOT / "voxbind"
DATA = REPO_ROOT / "dataset" / "data"
CIF_DIR = DATA / "cif"                        # SHARED structures
CCP4_DIR = DATA / "ccp4"                      # SHARED density (OTF reads at train time)
PRETRAIN = DATA / "pretrain"

# The density normalisation every PLINDER corpus carries (v6 stats). Its source,
# xray_crops_aligned_v6/stats.json, is not kept. This used to be read from the v1 recipe
# (plinder/xray_resample_plinder/resample.json), which moved to legacy/ on 2026-10-05; the v2 and
# CoDE_pretrain_v1 recipes hold the same block.
V6_NORMALIZATION = {
    "scheme": "pocket-pool arcsinh soft-squash + z-score (v6 stats, PLINDER)",
    "formula": "x' = (arcsinh(x / s) - mu_a) / sigma_a",
    "arcsinh_scale": 0.5,
    "mu_a": -0.015437547297403289,
    "sigma_a": 0.5084830820835629,
}

# Channel layouts. `parse` goes straight to 03c.parse_cif_system; `suffix` is appended to both
# output names (empty for roleblob, which owns the unsuffixed names).
LAYOUTS = {
    "perelem": dict(
        parse=dict(diverse=False, other_channel=True),
        suffix="_perelem", n_lig_ch=8, write_manifest=True,
        banner="PER-ELEMENT+other build (8 lig / 4 poc)",
        histogram="ligand channel histogram (0-6=C/O/N/S/F/Cl/P, 7=other):",
        train_time_note=(
            "v2 PER-ELEMENT+other (data_train_plinder_v2_perelem.pt): ligand atoms_channel 0..7 "
            "(7=other rare tail), pocket 0..3 (C/N/O/S); per-atom vdW radius stored; reads SHARED data/ccp4."
        ),
    ),
    "roleblob": dict(
        parse=dict(diverse=True),
        suffix="", n_lig_ch=1, write_manifest=True,
        banner="DIVERSE build (role-split, OTF)",
        histogram=None,                        # one channel; a histogram says nothing
        train_time_note=(
            "v2 DIVERSE role-split (data_train_plinder_v2.pt): per-atom vdW radius, "
            "atoms_channel=0 → single role channel admits all elements; reads SHARED data/ccp4."
        ),
    ),
    # Same parse as perelem, but writes only tuples: it is position-aligned with the unsuffixed
    # corpus and reuses that manifest and box, which the alignment check at the end confirms.
    "atomblob8": dict(
        parse=dict(diverse=False, other_channel=True),
        suffix="_atomblob8", n_lig_ch=8, write_manifest=False,
        banner="ATOMBLOB-8ch build (per-element + other_channel), tuples only",
        histogram="ligand channel histogram (0-6=C/O/N/S/F/Cl/P, 7=other):",
        train_time_note=None,
        align_ref_suffix="",                   # compare against the unsuffixed tuples
    ),
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


M = load_module(REPO_ROOT / "dataset" / "plinder" / "03c_plinder_preprocess.py", "p03c")


def version_paths(version: str, suffix: str):
    """Selection freeze, output tuples and output manifest dir for a version + layout.

    All versions reuse the SHARED data/cif + data/ccp4 caches, so nothing is downloaded.
    """
    return (VOX_ROOT / "splits" / "plinder" / version,
            PRETRAIN / f"data_train_plinder_{version}{suffix}.pt",
            PRETRAIN / f"xray_resample_plinder_{version}{suffix}")


def _refuse_overwrite(out_tuples: Path, out_dir, force: bool) -> None:
    """Stop before parsing if this corpus already exists.

    A precomputed density box (05_make_boxes.py) is indexed by manifest position, so
    rebuilding the tuples or manifest under it silently misaligns box-based training.
    """
    existing = [p for p in (out_tuples, out_dir and out_dir / "train_manifest.npz") if p and p.exists()]
    if existing and not force:
        raise SystemExit("[refuse] corpus already built: " + ", ".join(str(p) for p in existing)
                         + "\n         Pass --force to rebuild it (rebuild any box after).")


def resolve_selection(freeze: Path) -> Path:
    csv = freeze / "plinder_selected.csv"
    if not csv.exists():
        raise SystemExit(f"[fatal] no frozen selection at {csv} — run 01_select.py first.")
    inp = freeze / "plinder_inputs.json"
    if inp.exists():
        want = json.loads(inp.read_text()).get("selection_sha256")
        got = hashlib.sha256(csv.read_bytes()).hexdigest()
        if want and got != want:
            raise SystemExit(f"[fatal] selection hash MISMATCH:\n  expected {want}\n  got {got}")
    return csv


def _tuple_from_parsed(parsed, key: str):
    lig_xyz, lig_ch, poc_xyz, poc_ch, _lu, _pu, lig_rad, poc_rad = parsed
    max_len = round(float((lig_xyz.max(0).values - lig_xyz.min(0).values).max()), 2)
    pocket_ = {"id": f"plinder/{key}", "coords": poc_xyz.to(torch.float32),
               "atoms_channel": poc_ch.to(torch.uint8), "radius": poc_rad.to(torch.float32)}
    ligand_ = {"id": f"plinder/{key}", "coords": lig_xyz.to(torch.float32),
               "atoms_channel": lig_ch.to(torch.uint8), "radius": lig_rad.to(torch.float32),
               "max_len": max_len}
    return pocket_, ligand_


# ── parsing: serial, or one task per cif across a worker pool ───────────────────────

def parse_serial(sel, parse_kwargs, pocket_radius):
    tuples, meta, skipped, ch_hist = [], [], {}, Counter()
    for r in tqdm(list(sel.itertuples(index=False)), desc="build", unit="cplx"):
        pid = str(r.entry_pdb_id).lower(); key = f"{pid}_{r.ligand_asym_id}"
        cif = CIF_DIR / f"{pid}.cif"
        if not cif.exists():
            skipped[key] = "missing_cif"; continue
        try:
            parsed = M.parse_cif_system(cif, r.ligand_asym_id, r.ligand_ccd_code,
                                        pocket_radius, **parse_kwargs)
        except Exception as e:
            skipped[key] = f"error:{type(e).__name__}"; continue
        if parsed is None:
            skipped[key] = "parse_failed"; continue
        tup = _tuple_from_parsed(parsed, key)
        for c in tup[1]["atoms_channel"].tolist():
            ch_hist[c] += 1
        tuples.append(tup)
        meta.append((pid, str(r.ligand_ccd_code)))
    return tuples, meta, skipped, ch_hist


_W = None


def _worker_init():
    """Each worker loads the parser once: the module object is not picklable."""
    global _W
    _W = load_module(REPO_ROOT / "dataset" / "plinder" / "03c_plinder_preprocess.py", "p03c_w")


def _parse_pid(task):
    """Parse every row of one pdb_id with the cif loaded once."""
    pid, rows, parse_kwargs, pocket_radius = task
    cif = CIF_DIR / f"{pid}.cif"
    if not cif.exists():
        return [(idx, None, "missing_cif") for idx, _ in rows]
    out = []
    for idx, r in rows:
        try:
            parsed = _W.parse_cif_system(cif, r["ligand_asym_id"], r["ligand_ccd_code"],
                                         pocket_radius, **parse_kwargs)
        except Exception as e:
            out.append((idx, None, f"error:{type(e).__name__}")); continue
        if parsed is None:
            out.append((idx, None, "parse_failed")); continue
        out.append((idx, _tuple_from_parsed(parsed, f"{pid}_{r['ligand_asym_id']}"), None))
    return out


def parse_parallel(sel, parse_kwargs, pocket_radius, jobs):
    """Same tuples in the same order as parse_serial: rows are reassembled in CSV order and
    failures dropped, which is what appending on success produces."""
    groups = defaultdict(list)
    records = sel.to_dict("records")
    for idx, r in enumerate(records):
        groups[str(r["entry_pdb_id"]).lower()].append((idx, r))
    tasks = [(pid, rows, parse_kwargs, pocket_radius) for pid, rows in groups.items()]

    results, reasons = {}, {}
    with mp.Pool(jobs, initializer=_worker_init) as pool:
        for k, chunk in enumerate(pool.imap_unordered(_parse_pid, tasks, chunksize=4)):
            for idx, tup, reason in chunk:
                results[idx] = tup
                if reason:
                    reasons[idx] = reason
            if (k + 1) % 2000 == 0 or (k + 1) == len(tasks):
                done = sum(v is not None for v in results.values())
                print(f"  pids {k+1:,}/{len(tasks):,}  parsed {done:,}", flush=True)

    tuples, meta, ch_hist = [], [], Counter()
    for i in range(len(records)):
        tup = results.get(i)
        if tup is None:
            continue
        for c in tup[1]["atoms_channel"].tolist():
            ch_hist[c] += 1
        tuples.append(tup)
        meta.append((str(records[i]["entry_pdb_id"]).lower(), str(records[i]["ligand_ccd_code"])))
    return tuples, meta, {f"row{i}": why for i, why in reasons.items()}, ch_hist


def pose_dedup(tuples, meta, tau: float):
    """Density-aware pose dedup: keep one copy per density cluster within each (PDB, ligand).

    Grouped PID -> CCD so each CCP4 map is loaded once and freed before the next PID;
    caching every map blows up host RAM at ~40k PDBs.
    """
    cdm = load_module(REPO_ROOT / "dataset" / "crossdocked_density.py", "cdm_pd")
    pdd = load_module(REPO_ROOT / "dataset" / "plinder" / "pose_dedup.py", "pose_dedup")
    pid_groups = defaultdict(lambda: defaultdict(list))
    for i, (pid, ccd) in enumerate(meta):
        pid_groups[pid][ccd].append(i)
    keep = np.zeros(len(tuples), dtype=bool)
    n_nomap = 0
    for pid, ccd_map in tqdm(pid_groups.items(), desc=f"pose-dedup(t={tau})", unit="pdb"):
        g = None
        if any(len(idxs) > 1 for idxs in ccd_map.values()):
            p = CCP4_DIR / f"{pid}.ccp4"
            g = cdm._load_raw_grid(p) if p.exists() else None
        for ccd, idxs in ccd_map.items():
            if len(idxs) == 1:
                keep[idxs[0]] = True; continue
            if g is None:                               # no map → keep all copies (rare)
                keep[idxs] = True; n_nomap += 1; continue
            arr, fMT, nu, nv, nw = g
            rf = lambda center, R: cdm._resample_density(arr, fMT, nu, nv, nw, center,
                                                         R_aug=R, G=40, res=0.35)
            by_n = defaultdict(list)                    # only equal-atom copies are Kabsch-comparable
            for i in idxs:
                by_n[len(tuples[i][1]["coords"])].append(i)
            for _n, sub in by_n.items():
                if len(sub) == 1:
                    keep[sub[0]] = True; continue
                coords = [tuples[i][1]["coords"].numpy().astype(np.float64) for i in sub]
                for k in pdd.pose_dedup_group(coords, rf, tau=tau, cap=16):
                    keep[sub[k]] = True
        g = None                                        # free this PID's map before the next
    kept = [t for t, k in zip(tuples, keep) if k]
    print(f"  pose-dedup(tau={tau}): {len(keep):,} → {len(kept):,} kept "
          f"({int(keep.sum())} distinct-pose crops; {n_nomap} groups had no map)")
    return kept


def write_manifest(tuples, out_dir: Path, train_time_note, resample_from) -> int:
    """Laydown + manifest. Mirrors DatasetCrossDockedDensity so row i aligns with index i."""
    order = list(tuples)
    random.Random(1234).shuffle(order)
    order = order[: len(order) - M.VAL_SZ]
    order = [t for t in order if t[1]["max_len"] <= M.MAX_LEN]
    n = len(order)
    if not n:
        raise SystemExit("  [abort] the laydown kept nothing — check VAL_SZ and the max_len filter.")
    pdb_ids = [str(p["id"]).split("/")[-1].split("_")[0].lower() for p, _l in order]
    centroids = np.stack([l["coords"].to(torch.float32).mean(0).numpy()
                          for _p, l in order]).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train_manifest.npz", pdb_id=np.array(pdb_ids), centroid=centroids,
             R=np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy(),
             t=np.zeros((n, 3), dtype=np.float32), ok=np.ones(n, dtype=bool))
    np.save(str(out_dir / "train_available.npy"), np.ones(n, dtype=bool))

    if resample_from:
        # Inherit the source recipe (same maps, norm and grid) but correct its provenance:
        # copied verbatim, its note described a different corpus.
        recipe = json.loads((resample_from / "resample.json").read_text())
        recipe["source_resample_dir"] = str(resample_from.resolve())
        recipe["train_time_note"] = train_time_note or (
            f"Tuples beside this manifest were built by 04_build_corpus.py; density recipe "
            f"inherited from {resample_from.name}.")
    else:
        recipe = dict(kind="resample_manifest", grid_dim=M.GRID_DIM, resolution=M.RESOLUTION,
                      ccp4_dir=str(CCP4_DIR.resolve()), ccp4_ext=".ccp4",
                      frame="deposited (transform=None): R=I, t=0", norm_version="v6",
                      normalization=dict(V6_NORMALIZATION), train_time_note=train_time_note)
    (out_dir / "resample.json").write_text(json.dumps(recipe, indent=2))
    print(f"  manifest: {n:,} train positions → {out_dir}")
    return n


def verify_alignment(tuples, ref_path: Path) -> None:
    """A tuples-only layout reuses another corpus's manifest and box, so its rows must line up
    position by position with that corpus. A mismatch means do not train."""
    if not ref_path.exists():
        print(f"  [align] no reference at {ref_path.name} — cannot verify")
        return
    ref = torch.load(str(ref_path), weights_only=False)
    same_n = len(ref) == len(tuples)
    mism = [i for i in range(min(len(ref), len(tuples)))
            if ref[i][1]["id"] != tuples[i][1]["id"]]
    print(f"  [align] ref={len(ref):,} new={len(tuples):,} same_n={same_n} "
          f"id_mismatches={len(mism)}")
    if same_n and not mism:
        print(f"  [align] OK — identical id order, reuses {ref_path.stem}'s manifest and box")
    else:
        raise SystemExit(f"  [align] MISALIGNED — do NOT train "
                         f"(first mismatch at index {mism[0] if mism else 'n/a'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="perelem", choices=sorted(LAYOUTS),
                    help="channel layout (default: perelem)")
    ap.add_argument("--version", default="v2", help="selection freeze / output suffix (v2, v2p1, …)")
    ap.add_argument("--limit", type=int, default=0, help="first N rows (smoke)")
    ap.add_argument("--pocket_radius", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=1,
                    help=">1 parses with a worker pool, one cif per task (same tuples and order)")
    ap.add_argument("--pose_dedup_tau", type=float, default=0.0,
                    help=">0 enables DENSITY-AWARE pose dedup: within each (PDB, ligand) group keep one "
                         "copy per density cluster (aligned density-crop corr < tau); drops symmetry copies.")
    ap.add_argument("--align_ref", type=Path, default=None,
                    help="tuples .pt a tuples-only layout must be position-aligned with "
                         "(defaults to the layout's own reference)")
    # Filter mode: no parsing at all.
    ap.add_argument("--filter_from", type=Path, default=None,
                    help="build by filtering an existing tuples .pt instead of parsing cifs")
    ap.add_argument("--max_lig_ch", type=int, default=6,
                    help="filter mode: drop complexes with a ligand atom above this channel "
                         "(6 keeps the 7-element vocabulary)")
    ap.add_argument("--out_tuples", type=Path, default=None, help="filter mode: output tuples .pt")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing corpus (tuples / manifest)")
    ap.add_argument("--out_dir", type=Path, default=None, help="filter mode: output manifest dir")
    ap.add_argument("--resample_from", type=Path, default=None,
                    help="inherit this directory's resample.json instead of writing a fresh one")
    args = ap.parse_args()

    def _abs(p):
        return None if p is None else (Path(p) if Path(p).is_absolute() else DATA / p)

    # ── filter mode ────────────────────────────────────────────────────────────────
    if args.filter_from:
        if not args.out_tuples or not args.out_dir:
            raise SystemExit("[fatal] --filter_from needs --out_tuples and --out_dir")
        src_path, out_tuples, out_dir = _abs(args.filter_from), _abs(args.out_tuples), _abs(args.out_dir)
        _refuse_overwrite(out_tuples, out_dir, args.force)
        src = torch.load(str(src_path), weights_only=False)
        tuples = [(p, l) for (p, l) in src if int(l["atoms_channel"].max()) <= args.max_lig_ch]
        dropped = len(src) - len(tuples)
        print(f"=== filter build (ligand channels <= {args.max_lig_ch}) ===\n"
              f"  {src_path.name}: {len(src):,} → {len(tuples):,} kept ({dropped:,} dropped)")
        if not tuples:
            raise SystemExit("  [abort] the filter removed everything.")
        worst = max(int(l["atoms_channel"].max()) for _p, l in tuples)
        if worst > args.max_lig_ch:
            raise SystemExit(f"  [abort] filter failed: max ligand channel {worst}")
        PRETRAIN.mkdir(parents=True, exist_ok=True)
        torch.save(tuples, str(out_tuples))
        print(f"  {out_tuples.name}: {len(tuples):,} tuples")
        note = (f"Filtered from {src_path.name} to ligand channels 0..{args.max_lig_ch} "
                f"(out-of-vocab complexes dropped); per-atom vdW radius stored; "
                f"reads SHARED data/ccp4.")
        n = write_manifest(tuples, out_dir, note, _abs(args.resample_from))
        print(f"  subset_n = {max(n - M.VAL_SZ, 0):,}  subset_val_n = {M.VAL_SZ}")
        print(f"  NEXT: python dataset/plinder/05_make_boxes.py --resample_dir {out_dir} --jobs 80")
        return

    # ── parse mode ─────────────────────────────────────────────────────────────────
    layout = LAYOUTS[args.layout]
    freeze, out_tuples, out_dir = version_paths(args.version, layout["suffix"])
    _refuse_overwrite(out_tuples, out_dir if layout["write_manifest"] else None, args.force)
    csv = resolve_selection(freeze)
    sel = pd.read_csv(csv)
    if args.limit:
        sel = sel.head(args.limit)
    print(f"=== {args.version} {layout['banner']} ===\n  selection: {csv} ({len(sel):,} rows)"
          f"\n  cifs: {CIF_DIR}")

    parse = parse_parallel if args.jobs > 1 else parse_serial
    extra = (args.jobs,) if args.jobs > 1 else ()
    tuples, meta, skipped, ch_hist = parse(sel, layout["parse"], args.pocket_radius, *extra)

    print(f"  built {len(tuples):,} tuples  (skipped {len(skipped):,})")
    for reason, n in Counter(skipped.values()).most_common():
        print(f"     skip {reason:20s} {n}")
    if layout["histogram"]:
        tot = sum(ch_hist.values())
        print(f"  {layout['histogram']}")
        for c in range(layout["n_lig_ch"]):
            print(f"     ch {c}: {ch_hist.get(c,0):>10,}  ({100*ch_hist.get(c,0)/max(tot,1):5.2f}%)")
    if not tuples:
        raise SystemExit("  [abort] no tuples built.")

    if args.pose_dedup_tau > 0:
        tuples = pose_dedup(tuples, meta, float(args.pose_dedup_tau))
        if not tuples:
            raise SystemExit("  [abort] pose-dedup removed everything.")

    PRETRAIN.mkdir(parents=True, exist_ok=True)
    torch.save(tuples, str(out_tuples))
    print(f"  {out_tuples.name}: {len(tuples):,} tuples")

    if layout["write_manifest"]:
        n = write_manifest(tuples, out_dir, layout["train_time_note"], _abs(args.resample_from))
        launch = (f"  launch: dset.resample_dir={out_dir} "
                  f"dset.data_file=pretrain/{out_tuples.name} "
                  f"dset.subset_n={max(n - M.VAL_SZ, 0)} dset.subset_val_n={M.VAL_SZ}")
        if args.layout == "perelem":
            launch += f" dset.n_lig_ch={layout['n_lig_ch']}"
        else:
            launch += " dset.subset_xray_only=true dset.ligand_radius=-1 dset.pocket_radius=-1"
        print(launch)
    else:
        ref_suffix = layout.get("align_ref_suffix", "")
        ref = _abs(args.align_ref) if args.align_ref else version_paths(args.version, ref_suffix)[1]
        verify_alignment(tuples, ref)
        print(f"  no manifest written: this layout reuses "
              f"{version_paths(args.version, ref_suffix)[2].name}")


if __name__ == "__main__":
    main()

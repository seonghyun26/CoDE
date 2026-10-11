"""make_splits.py — (re)generate the frozen PDBbind split manifests.

This is the ONLY place a split is *computed*. It reads primary inputs (which live
under the git-ignored ``dataset/data/`` tree) and writes committed ``pid,split``
manifests + ``MANIFEST.json`` (counts, sha256, provenance). Daily train/probe code
never recomputes — it reads the frozen manifest via ``voxbind.splits.load_split``.

Run it intentionally, on a machine with COMPLETE inputs (full RSCC table, etc.),
then commit the regenerated manifests:

    cd VoxBind            # the repository root
    python voxbind/splits/make_splits.py                 # both schemes
    git add -f voxbind/splits/*.csv voxbind/splits/MANIFEST.json

Schemes
-------
  lp_edrscc_v2        LP-PDBBind ``new_split`` ∩ ED-available ∩ (lig & poc RSCC ≥ 0.8) ∩
                      (Kd or Ki), non-covalent: 3850 / 817 / 1320. Its test set is FULL.
  lp_edrscc_v2_cl123  lp_edrscc_v2 restricted to LP-PDBBind cleaning levels CL1-3: 1559 / 410 /
                      733. Its test set is CL3.
  <base>_ID30/_ID60   lp_edrscc_v2 or lp_edrscc_v2_cl123 without the train complexes whose protein
                      has ATOM3D identity >= 30 / 60% (voxbind/splits/atom3d_identity.py) to a
                      CL3 test protein. Val and test are the base scheme's.

The other retired schemes (lp_edrscc_v1, the CL1/CL12 tiers, time_v1, misato_md_v1,
clean_ed_v1) and their builders are in legacy/voxbind/splits/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from voxbind.splits import SPLITS_DIR, SPLIT_NAMES, manifest_hash, scheme_info  # noqa: E402

PDBBIND = REPO_ROOT / "dataset" / "data" / "pdbbind"
LP_CSV = PDBBIND / "raw" / "LP_PDBBind.csv"
ED_RSCC = PDBBIND / "ed_rscc_split.csv"

RSCC_THRESHOLD = 0.8

# Test-anchored variants: scheme -> (base scheme, ATOM3D identity cutoff).
ATOM3D_SCHEMES = {f"{base}_ID{cut}": (base, cut / 100)
                  for base in ("lp_edrscc_v2", "lp_edrscc_v2_cl123") for cut in (30, 60)}
# The test split every *_ID30/_ID60 variant is measured against.
ANCHOR_SPLIT = "lp_edrscc_v2_cl123"


# ── primary-input loaders ────────────────────────────────────────────────────
def _load_quality_frame() -> pd.DataFrame:
    """ed_rscc_split rows passing the deterministic quality bar, joined with year.

    Quality bar = non-covalent ∧ lig_rscc ≥ τ ∧ poc_rscc ≥ τ. This is the shared
    universe for both lp_edrscc and time schemes; they differ only in PARTITION.
    """
    if not ED_RSCC.exists():
        raise FileNotFoundError(f"missing primary input: {ED_RSCC}")
    ed = pd.read_csv(ED_RSCC)
    ed["pid"] = ed["pid"].astype(str).str.lower()
    cov = ed["covalent"].astype(str).str.lower().isin(["true", "1"])
    q = ed[(~cov) & (ed["lig_rscc"] >= RSCC_THRESHOLD) & (ed["poc_rscc"] >= RSCC_THRESHOLD)].copy()

    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    lp["year"] = pd.to_datetime(lp["date"], errors="coerce").dt.year
    # affinity measurement type — the 'kd/ki' column actually holds Kd / Ki / IC50
    lp["mtype"] = (lp["kd/ki"].astype(str)
                   .str.extract(r"^\s*(Kd|Ki|IC50)", expand=False, flags=re.IGNORECASE)
                   .str.lower())
    # LP-PDBBind per-complex cleaning-level flags (nested: CL3 ⊆ CL2 ⊆ CL1),
    # stored as the strings "True"/"False" in the CSV → coerce to bool.
    for c in ("CL1", "CL2", "CL3"):
        lp[c] = lp[c].astype(str).str.strip().str.lower().isin(["true", "1"])
    q = q.merge(lp[["pid", "year", "mtype", "CL1", "CL2", "CL3"]], on="pid", how="left")
    return q


# ── scheme builders → list[(pid, split)] ─────────────────────────────────────
def build_lp_edrscc_v2(q: pd.DataFrame) -> tuple[list, dict]:
    """lp_edrscc ∩ affinity measured as Kd or Ki (drop assay-dependent IC50).

    IC50 varies with substrate/enzyme concentration and mechanism, so pIC50 is not
    directly comparable to the thermodynamic pKd/pKi. Restricting to Kd/Ki gives a
    cleaner regression target. Identical ED + (lig & poc RSCC ≥ 0.8) + non-covalent
    quality bar and the same sequence-clustered new_split partition as v1 — just the
    Kd/Ki subset (≈ 3850 / 817 / 1320).
    """
    sub = q[q["new_split"].astype(str).str.lower().isin(SPLIT_NAMES)
            & q["mtype"].isin(["kd", "ki"])]
    pairs = [(p, s.lower()) for p, s in zip(sub["pid"], sub["new_split"].astype(str))]
    prov = {
        "description": "lp_edrscc_v1 ∩ (Kd or Ki) — IC50 dropped; ED-available ∩ (lig & poc RSCC ≥ 0.8), non-covalent",
        "partition": "LP_PDBBind 'new_split' column (sequence-identity clustered)",
        "generalization": "novel-target (sequence-dedup); Kd/Ki-only affinity labels (no assay-dependent IC50)",
        "inputs": {"ed_rscc_split": ED_RSCC.name, "lp_pdbbind": LP_CSV.name,
                   "rscc_threshold": RSCC_THRESHOLD, "measurement_types": ["Kd", "Ki"]},
    }
    return pairs, prov


def build_lp_edrscc_v2_cl(q: pd.DataFrame, n_levels: int) -> tuple[list, dict]:
    """lp_edrscc_v2 further restricted to LP-PDBBind cleaning level(s) CL1..CLn.

    LP-PDBBind ships three nested, per-complex "cleaning levels" (CL3 ⊆ CL2 ⊆ CL1)
    that flag structural / annotation quality with progressively stricter criteria.
    These variants apply the filter to ALL splits (train/val/test), so each is a
    self-contained, progressively-cleaner benchmark that is an EXACT SUBSET of the
    frozen lp_edrscc_v2 partition (identical split assignments — a pid keeps its v2
    bucket). ``n_levels`` ∈ {1,2,3} requires CL1 (and CL2, and CL3) to be True.

    Diverse-split trial: measures how binding-affinity generalization moves as the
    benchmark is cleaned to LP-PDBBind's stricter quality tiers.
    """
    if n_levels not in (1, 2, 3):
        raise ValueError(f"n_levels must be 1, 2 or 3 (got {n_levels})")
    cl_cols = ["CL1", "CL2", "CL3"][:n_levels]
    mask = (q["new_split"].astype(str).str.lower().isin(SPLIT_NAMES)
            & q["mtype"].isin(["kd", "ki"]))
    for c in cl_cols:
        mask = mask & q[c].fillna(False).astype(bool)
    sub = q[mask]
    pairs = [(p, s.lower()) for p, s in zip(sub["pid"], sub["new_split"].astype(str))]

    # Invariant: a CL variant MUST be an exact subset of the frozen v2 manifest
    # (same pids, same split). Guards against any drift in the quality frame vs the
    # committed v2 CSV — if this trips, v2 itself changed and must be re-frozen first.
    v2_csv = SPLITS_DIR / "lp_edrscc_v2.csv"
    if v2_csv.exists():
        v2 = pd.read_csv(v2_csv)
        v2map = dict(zip(v2["pid"].astype(str).str.lower(), v2["split"].astype(str).str.lower()))
        for p, s in pairs:
            if v2map.get(p) != s:
                raise AssertionError(
                    f"CL variant pid {p!r} (split {s!r}) is not in frozen lp_edrscc_v2 "
                    f"with the same split (v2 has {v2map.get(p)!r}) — re-freeze v2 first."
                )

    tag = "+".join(cl_cols)
    prov = {
        "description": f"lp_edrscc_v2 ∩ LP-PDBBind cleaning level(s) {tag} (applied to all splits); "
                       f"ED-available ∩ (lig & poc RSCC ≥ 0.8) ∩ (Kd or Ki), non-covalent",
        "partition": "LP_PDBBind 'new_split' column (sequence-identity clustered)",
        "generalization": f"novel-target (sequence-dedup); Kd/Ki-only; LP cleaning {tag} — a "
                          f"progressively-cleaner subset of lp_edrscc_v2 (all splits filtered)",
        "inputs": {"ed_rscc_split": ED_RSCC.name, "lp_pdbbind": LP_CSV.name,
                   "rscc_threshold": RSCC_THRESHOLD, "measurement_types": ["Kd", "Ki"],
                   "cleaning_levels": cl_cols},
    }
    return pairs, prov


# LP-PDBBind chains of each pid (the colon-separated seq column) as {pid__chain<i>: sequence}.
def _lp_chains(pids) -> dict:
    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pid"})
    seq = dict(zip(lp["pid"].astype(str).str.lower(), lp["seq"].fillna("").astype(str)))
    chains = {}
    for pid in pids:
        for i, raw in enumerate(seq.get(pid, "").split(":")):
            residues = "".join(r for r in raw.upper() if "A" <= r <= "Z")
            if residues:
                chains[f"{pid}__chain{i}"] = residues
    missing = set(pids) - {name.split("__")[0] for name in chains}
    if missing:
        raise RuntimeError(f"{len(missing)} pids lack sequences in {LP_CSV.name}: {sorted(missing)[:5]}")
    return chains


# The frozen base scheme without its train complexes similar (ATOM3D identity >= cutoff) to a CL3
# test complex; val and test unchanged.
def build_atom3d(scheme: str) -> tuple[list, dict]:
    from voxbind.splits import atom3d_identity, load_split

    base, cutoff = ATOM3D_SCHEMES[scheme]
    split = load_split(base)
    anchor = set(load_split(ANCHOR_SPLIT)["test"])

    # As ATOM3D, the test chains are searched against every chain of the dataset; train hits count.
    every = sorted(anchor.union(*split.values()))
    similar = atom3d_identity.similar_pairs(_lp_chains(sorted(anchor)), _lp_chains(every), cutoff)
    removed = set(similar["subject"].str.split("__").str[0]) & set(split["train"])
    pairs = [(p, s) for s in SPLIT_NAMES for p in split[s] if not (s == "train" and p in removed)]
    prov = {
        "description": (f"{base} without the train complexes whose protein has ATOM3D identity >= "
                        f"{cutoff:.0%} to a CL3 test protein; val and test unchanged"),
        "base_scheme": base,
        "base_sha256": scheme_info(base)["sha256"],
        "anchor": f"{ANCHOR_SPLIT} test",
        "similarity": "atom3d",
        "identity_min_percent": round(cutoff * 100),
        "identity_definition": ("BLASTP HSP identical residues / test-chain length "
                                "(atom3d/protein/sequence.py find_similar)"),
        "train_removed": len(removed),
        "search_tool": atom3d_identity.blast_version(),
        "inputs": {"lp_pdbbind": LP_CSV.name},
    }
    return pairs, prov


def _counts(pairs) -> dict:
    c = {s: 0 for s in SPLIT_NAMES}
    for _, s in pairs:
        c[s] += 1
    return c


def _write_manifest_csv(pairs, path: Path) -> None:
    df = pd.DataFrame(sorted(pairs), columns=["pid", "split"])
    # de-dupe defensively (a pid should appear once); keep first split if collision
    df = df.drop_duplicates(subset="pid", keep="first")
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate frozen PDBbind split manifests.")
    ap.add_argument("--scheme", default="all",
                    choices=["all", "lp_edrscc_v2", "lp_edrscc_v2_cl123", *ATOM3D_SCHEMES])
    args = ap.parse_args()

    manifest_path = SPLITS_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("note",
                        "Frozen, version-controlled splits. Source of truth is the CSV; "
                        "MANIFEST.json pins counts+sha256. Regenerate via make_splits.py.")
    manifest.setdefault("schemes", {})

    base_schemes = ["lp_edrscc_v2", "lp_edrscc_v2_cl123"]
    want = base_schemes + list(ATOM3D_SCHEMES) if args.scheme == "all" else [args.scheme]
    q = _load_quality_frame() if set(want) & set(base_schemes) else None

    for scheme in want:
        if scheme == "lp_edrscc_v2":
            pairs, prov = build_lp_edrscc_v2(q)
        elif scheme == "lp_edrscc_v2_cl123":
            pairs, prov = build_lp_edrscc_v2_cl(q, 3)
        else:
            pairs, prov = build_atom3d(scheme)
        fname = f"{scheme}.csv"

        csv_path = SPLITS_DIR / fname
        _write_manifest_csv(pairs, csv_path)
        # re-read what we wrote so counts+hash describe the on-disk artifact exactly
        df = pd.read_csv(csv_path)
        on_disk = list(zip(df["pid"].astype(str), df["split"].astype(str)))
        counts = _counts(on_disk)
        h = manifest_hash(on_disk)
        manifest["schemes"][scheme] = {
            "file": fname,
            "counts": {**counts, "total": sum(counts.values())},
            "sha256": h,
            **prov,
        }
        print(f"[{scheme:14s}] {fname:18s} "
              f"train={counts['train']:5d} val={counts['val']:5d} test={counts['test']:5d}  "
              f"sha256={h[:12]}…")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {manifest_path.relative_to(REPO_ROOT)}")
    print("commit with:  git add -f voxbind/splits/*.csv voxbind/splits/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

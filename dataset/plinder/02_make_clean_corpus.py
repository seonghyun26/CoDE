#!/usr/bin/env python3
"""Remove the paper's test-set proteins from a pretraining corpus.

    python dataset/plinder/02_make_clean_corpus.py                       # cdgv2 -> CoDE_pretrain_v1
    python dataset/plinder/02_make_clean_corpus.py --source ab7v2p1      # where the ab7v2p1 corpus was built
    python dataset/plinder/02_make_clean_corpus.py --dry-run             # resolve, write nothing
    python dataset/plinder/02_make_clean_corpus.py --fetch-crossdocked   # refresh the receptor FASTA
    python dataset/plinder/02_make_clean_corpus.py --source CoDE_pretrain_v1 --similarity atom3d --identity 30
                                                                         # -> CoDE_pretrain_v1_ID30

Cohorts are the test sets the paper reports. Six are for affinity: FULL, CL3, CL3-ID60, CL3-ID30,
CASF-nontrain and CASF-clean; they nest, so their union is FULL plus CASF-nontrain (1,357
complexes). The seventh is the CrossDocked test set of ligand generation (dataset/data/data_test.pt,
100 pockets): its receptor and ligand-source PDB entries, and the receptor chains as sequences.

Policy: a corpus PDB is removed when one of its protein chains is the same protein as a cohort
chain -- DIAMOND identity >= `--identity` (default 100) over an alignment covering >= `--coverage`
(default 80) of the shorter sequence -- or when it is a cohort PDB entry itself. Every ligand
observation of a removed PDB goes. Lower `--identity` for a homology filter (ID30 was v2.7's).
With `--similarity atom3d` the rule is ATOM3D's instead: BLASTP of each cohort chain against the
corpus chains, and a chain is similar when one HSP has identical residues / cohort-chain length
>= `--identity` (voxbind/splits/atom3d_identity.py). Its default cohort is the CL3 test set alone:
CoDE_pretrain_v1 already holds out every test set, and the ID30/ID60 views drop CL3 homologs on top.

The tuples and the density box are never copied: the box is hard-linked and only the manifest
availability mask changes, so a clean corpus is position-identical to its source.

A cohort protein with no sequence is an error, not a warning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VOX = ROOT / "voxbind"
DATA = ROOT / "dataset/data"

# Source corpora. A clean corpus is a masked view of one of these; `dset` names the
# pretrain/configs/dset/ option its loader config is derived from.
SOURCES = {
    "cdgv2": dict(
        dset="cdgv2",
        split=VOX / "splits/plinder/v2",
        resample=DATA / "pretrain/xray_resample_plinder_v2_perelem",
        data_file=DATA / "pretrain/data_train_plinder_v2_perelem.pt",
        box="box116.dat",
    ),
    # The clean default corpus; CoDE_pretrain_v1_ID30/ID60 are masked views of it.
    "CoDE_pretrain_v1": dict(
        dset="CoDE_pretrain_v1",
        split=VOX / "splits/plinder/CoDE_pretrain_v1",
        resample=DATA / "pretrain/xray_resample_CoDE_pretrain_v1",
        data_file=DATA / "pretrain/data_train_plinder_v2_perelem.pt",
        box="box116.dat",
    ),
    # Built on another server; its PDBs are a subset of v2's, so the v2 FASTA covers them.
    "ab7v2p1": dict(
        dset="ab7v2p1",
        split=VOX / "splits/plinder/v2p1",
        resample=DATA / "pretrain/xray_resample_plinder_v2p1",
        data_file=DATA / "pretrain/data_train_plinder_v2p1_atomblob7.pt",
        box="box96.dat",
    ),
}
PLINDER_FASTA = ROOT / "dataset/data/plinder/plinder_v2_pretrain.fasta"   # written by the OpenBind leakage audit
LP_CSV = DATA / "pdbbind/raw/LP_PDBBind.csv"
CROSSDOCKED_TEST = DATA / "data_test.pt"
# Receptor chain sequences of the CrossDocked test pockets, fetched from RCSB once and kept in the
# repo so a rebuild does not need the network (--fetch-crossdocked rewrites it).
CROSSDOCKED_FASTA = VOX / "splits/crossdocked_test_receptors.fasta"

# The DIAMOND binary that the minimol env's gget wheel ships, found through conda by env name.
def default_diamond() -> Path | None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from voxbind.utils.conda_env import conda_env_prefix

    prefix = conda_env_prefix("minimol")
    hits = sorted(prefix.glob("lib/python*/site-packages/gget/bins/Linux/diamond")) if prefix else []
    return hits[0] if hits else None
HIT_COLUMNS = [
    "query", "subject", "pident", "alignment_length", "query_length",
    "subject_length", "query_coverage", "subject_coverage", "evalue", "bitscore",
]

# The six test sets the paper reports (task1-affinity_regression/evals/probe_common_6set.py defines the same sets).
COHORTS = {
    "full": dict(label="FULL", kind="load_split", split="lp_edrscc_v2", partition="test"),
    "cl3": dict(label="CL3", kind="load_split", split="lp_edrscc_v2_cl123", partition="test"),
    "cl3_id60": dict(label="CL3-ID60", kind="csv", path=VOX / "splits/cl123_test_novel60_cohort.csv"),
    "cl3_id30": dict(label="CL3-ID30", kind="csv", path=VOX / "splits/cl123_test_novel30_cohort.csv"),
    "casf_nontrain": dict(label="CASF-nontrain", kind="csv",
                          path=VOX / "splits/casf2016_nontrain_cohort.csv"),
    "casf_clean": dict(label="CASF-clean", kind="csv", path=VOX / "splits/casf2016_clean_cohort.csv"),
    "crossdocked_test": dict(label="CrossDocked-test", kind="crossdocked", path=CROSSDOCKED_TEST),
}
DEFAULT_COHORTS = list(COHORTS)
# The corpus name a source is written under when --tag is not given.
DEFAULT_TAGS = {"cdgv2": "CoDE_pretrain_v1"}
ALL_COHORTS_SLUG = "paper7"


# ── helpers ─────────────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diamond_binary(explicit: str | None) -> Path:
    candidate = Path(explicit) if explicit else Path(shutil.which("diamond") or default_diamond() or "diamond")
    if not candidate.is_file():
        raise FileNotFoundError("DIAMOND not found; pass --diamond")
    return candidate


def hardlink_box(source: Path, destination: Path) -> None:
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(f"refusing to replace non-source box: {destination}")
    os.link(source, destination)


def _load_split(name: str, partition: str) -> set[str]:
    # The repo root, not voxbind/: `voxbind.splits` is a package under ROOT.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from voxbind.splits import load_split

    return {str(pid).strip().lower() for pid in load_split(name)[partition]}


def crossdocked_test_receptors() -> dict[str, set[str]]:
    """{receptor PDB id: its chain ids} of the CrossDocked test pockets, e.g. 2z3h_A_rec -> {A}."""
    import re
    import torch

    pattern = re.compile(r"(?:^|/)([0-9][A-Za-z0-9]{3})_([^_/]+)_rec_", re.IGNORECASE)
    receptors: dict[str, set[str]] = {}
    for pocket, _ in torch.load(CROSSDOCKED_TEST, weights_only=False):
        match = pattern.search(str(pocket["id"]))
        if match is None:
            raise ValueError(f"cannot parse CrossDocked test pocket {pocket['id']!r}")
        receptors.setdefault(match.group(1).lower(), set()).add(match.group(2))
    return receptors


def crossdocked_test_pids() -> set[str]:
    """Receptor and ligand-source PDB ids of the CrossDocked test set (01_select.py's parser)."""
    import importlib.util
    import torch

    spec = importlib.util.spec_from_file_location("plinder_select", Path(__file__).with_name("01_select.py"))
    select = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(select)
    return select.crossdocked_test_pdb_ids(torch.load(CROSSDOCKED_TEST, weights_only=False))


def fetch_crossdocked_fasta() -> None:
    """Write CROSSDOCKED_FASTA from RCSB: one record per test receptor chain (author chain ids)."""
    import re
    import urllib.request

    records, missing = [], []
    for pdb, chains in sorted(crossdocked_test_receptors().items()):
        with urllib.request.urlopen(f"https://www.rcsb.org/fasta/entry/{pdb.upper()}") as response:
            text = response.read().decode()
        found = set()
        for header, sequence in re.findall(r">([^\n]+)\n([A-Z]+)", text):
            # "Chains A, B[auth C]": the bracketed id is the author chain the PDB file uses.
            tokens = header.split("|")[1].removeprefix("Chains ").removeprefix("Chain ").split(",")
            authors = {(re.search(r"\[auth ([^\]]+)\]", t) or re.match(r"\s*(\S+)", t)).group(1)
                       for t in tokens}
            for chain in sorted(chains & authors):
                records.append(f">{pdb}__chain{chain}\n{sequence}\n")
                found.add(chain)
        missing += [f"{pdb}_{c}" for c in sorted(chains - found)]
    if missing:
        raise RuntimeError(f"no RCSB sequence for CrossDocked test receptor chains {missing}")
    CROSSDOCKED_FASTA.write_text("".join(records))
    print(f"wrote {CROSSDOCKED_FASTA.relative_to(ROOT)}: {len(records)} receptor chains")


def cohort_pids(name: str) -> set[str]:
    """Resolve one cohort to its set of held-out PDB ids."""
    spec = COHORTS[name]
    if spec["kind"] == "csv":
        frame = pd.read_csv(spec["path"])
        return set(frame["pid"].astype(str).str.strip().str.lower())
    if spec["kind"] == "crossdocked":
        return crossdocked_test_pids()
    return _load_split(spec["split"], spec["partition"])


def cohort_sources(name: str) -> list[dict]:
    """Provenance record for one cohort: the file and its hash, or the split scheme."""
    spec = COHORTS[name]
    if spec["kind"] == "csv":
        return [{"path": str(spec["path"].relative_to(ROOT)), "sha256": sha256(spec["path"])}]
    if spec["kind"] == "crossdocked":
        return [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in (CROSSDOCKED_TEST, CROSSDOCKED_FASTA)]
    return [{"split": spec["split"], "partition": spec["partition"]}]


def write_cohort_fasta(pids: set[str], path: Path, crossdocked: bool = False) -> tuple[int, int]:
    """One FASTA record per cohort chain: the LP-PDBBind chains of `pids`, plus the CrossDocked
    test receptor chains when `crossdocked`. Returns (n_complexes, n_chains)."""
    labels = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pid"})
    labels["pid"] = labels.pid.astype(str).str.lower()
    seq_of = dict(zip(labels.pid, labels.seq.fillna("").astype(str)))

    n_chains = n_complexes = 0
    missing: list[str] = []
    with path.open("w") as handle:
        for pid in sorted(pids):
            chains = 0
            for chain_index, raw_sequence in enumerate(str(seq_of.get(pid, "")).split(":")):
                sequence = "".join(r for r in raw_sequence.upper() if "A" <= r <= "Z")
                if not sequence:
                    continue
                handle.write(f">{pid}__chain{chain_index}\n{sequence}\n")
                n_chains += 1
                chains += 1
            if chains:
                n_complexes += 1
            else:
                missing.append(pid)
    if missing:
        raise RuntimeError(
            f"{len(missing)} cohort proteins lack sequences in {LP_CSV.name}: {missing[:5]}"
        )
    if crossdocked:
        records = CROSSDOCKED_FASTA.read_text()
        with path.open("a") as handle:
            handle.write(records)
        names = [line[1:] for line in records.splitlines() if line.startswith(">")]
        n_chains += len(names)
        n_complexes += len({name.split("__")[0] for name in names})
    return n_complexes, n_chains


def search(binary: Path, cohort_fasta: Path, output: Path, threads: int,
           identity: float, coverage: float) -> pd.DataFrame:
    database = output.with_suffix("")
    subprocess.run(
        [str(binary), "makedb", "--in", str(cohort_fasta), "--db", str(database), "--quiet"],
        check=True,
    )
    subprocess.run(
        [
            str(binary), "blastp", "--query", str(PLINDER_FASTA),
            "--db", str(database.with_suffix(".dmnd")), "--out", str(output),
            "--max-target-seqs", "1000", "--evalue", "100", "--id", str(identity),
            "--sensitive", "--quiet", "--threads", str(threads), "--outfmt", "6",
            "qseqid", "sseqid", "pident", "length", "qlen", "slen",
            "qcovhsp", "scovhsp", "evalue", "bitscore",
        ],
        check=True,
    )
    hits = pd.read_csv(output, sep="\t", header=None, names=HIT_COLUMNS)
    hits = hits[
        (hits.pident >= identity)
        & (hits[["query_coverage", "subject_coverage"]].max(axis=1) >= coverage)
    ].copy()
    hits["plinder_pdb"] = hits["query"].str.split("_").str[0].str.lower()
    hits["cohort_pdb"] = hits["subject"].str.split("__").str[0].str.lower()
    return hits


# ATOM3D rule: BLASTP of each cohort chain against the corpus chains; pairs at identity >= identity%.
def search_atom3d(cohort_fasta: Path, threads: int, identity: float) -> pd.DataFrame:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from voxbind.splits import atom3d_identity

    pairs = atom3d_identity.similar_pairs(atom3d_identity.read_fasta(cohort_fasta),
                                          atom3d_identity.read_fasta(PLINDER_FASTA),
                                          identity / 100, threads=threads)
    pairs["plinder_pdb"] = pairs.subject.str.split("_").str[0].str.lower()
    pairs["cohort_pdb"] = pairs["query"].str.split("__").str[0].str.lower()
    return pairs


# One row per removed corpus PDB: its most identical cohort chain under the ATOM3D rule.
def best_witnesses_atom3d(pairs: pd.DataFrame, source_pdbs: set[str]) -> pd.DataFrame:
    pairs = pairs[pairs.plinder_pdb.isin(source_pdbs)]
    best = (pairs.sort_values(["plinder_pdb", "identity", "bitscore"], ascending=[True, False, False])
            .drop_duplicates("plinder_pdb")
            .rename(columns={"subject": "plinder_chain", "query": "cohort_chain",
                             "qlen": "cohort_chain_length", "slen": "plinder_chain_length"}))
    best["atom3d_identity"] = (100 * best.identity).round(2)
    columns = ["plinder_pdb", "plinder_chain", "cohort_pdb", "cohort_chain", "atom3d_identity", "nident",
               "cohort_chain_length", "plinder_chain_length", "evalue", "bitscore"]
    return best[columns].sort_values("plinder_pdb").reset_index(drop=True)


def best_witnesses(hits: pd.DataFrame, source_pdbs: set[str]) -> pd.DataFrame:
    """One row per removed PLINDER PDB: its strongest qualifying hit into the cohort."""
    hits = hits[hits.plinder_pdb.isin(source_pdbs)].copy()
    best = (
        hits.sort_values(
            ["plinder_pdb", "pident", "bitscore", "query_coverage", "subject_coverage"],
            ascending=[True, False, False, False, False],
        )
        .drop_duplicates("plinder_pdb")
        .rename(columns={"query": "plinder_chain", "subject": "cohort_chain"})
    )
    columns = ["plinder_pdb", "plinder_chain", "cohort_pdb", "cohort_chain", "pident",
               "query_coverage", "subject_coverage", "alignment_length",
               "query_length", "subject_length", "evalue", "bitscore"]
    return best[columns].sort_values("plinder_pdb").reset_index(drop=True)


# ── outputs ─────────────────────────────────────────────────────────────────────────

# A path under the repo as a repo-relative string, for the records kept in git; others unchanged.
def repo_relative(value):
    if isinstance(value, str) and value.startswith(str(ROOT) + os.sep):
        return os.path.relpath(value, ROOT)
    return value


def write_loader_view(ctx: dict, removed: set[str], audit: dict) -> None:
    src, out = ctx["source"], ctx["resample"]
    out.mkdir(parents=True, exist_ok=True)
    source_manifest = src["resample"] / "train_manifest.npz"
    with np.load(source_manifest, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload = {k: v for k, v in payload.items() if not k.endswith("_keep")}
    pdb_ids = pd.Series(payload["pdb_id"].astype(str)).str.lower()
    source_ok = payload["ok"].astype(bool)
    keep = source_ok & ~pdb_ids.isin(removed).to_numpy()
    payload["ok"] = keep
    payload[f"{ctx['slug']}_keep"] = keep
    np.savez(out / "train_manifest.npz", **payload)
    np.save(out / "train_available.npy", keep)

    policy = {"cohort": ctx["cohort_description"], **ctx["policy"]}
    recipe = json.loads((src["resample"] / "resample.json").read_text())
    recipe.update({
        "name": ctx["name"],
        "source_resample_dir": str(src["resample"].resolve()),
        "availability_filter": {
            **policy,
            "removed_positions": int((source_ok & ~keep).sum()),
            "retained_positions": int(keep.sum()),
        },
        "train_time_note": (f"{ctx['tag']} reuses the {ctx['source_name']} tuples and density "
                            f"rows. Only positions with train_manifest.ok=True are exposed."),
    })
    (out / "resample.json").write_text(json.dumps(recipe, indent=2) + "\n")

    source_box, target_box = src["resample"] / src["box"], out / src["box"]
    hardlink_box(source_box, target_box)
    meta_name = f"{Path(src['box']).stem}_meta.json"
    if (src["resample"] / meta_name).exists():
        box_meta = json.loads((src["resample"] / meta_name).read_text())
        box_meta.update({
            "name": ctx["name"],
            "source_box": str(source_box.resolve()),
            "available_rows": int(keep.sum()),
            "note": (f"Hard-linked, position-identical {ctx['source_name']} box. The "
                     f"availability mask excludes {ctx['cohort_description']} proteins."),
        })
        (out / meta_name).write_text(json.dumps(box_meta, indent=2) + "\n")

    contract = {
        "kind": f"{ctx['name']}_loader_contract",
        "source": ctx["source_name"],
        "source_data_file": str(src["data_file"].resolve()),
        "source_resample_dir": str(src["resample"].resolve()),
        "resample_dir": str(out.resolve()),
        "box_path": str(target_box.resolve()),
        "box_is_hardlink_to_source": os.path.samefile(source_box, target_box),
        "source_manifest_sha256": sha256(source_manifest),
        "filtered_manifest_sha256": sha256(out / "train_manifest.npz"),
        "source_manifest_positions": len(keep),
        "source_available_positions": int(source_ok.sum()),
        "removed_manifest_positions": int((source_ok & ~keep).sum()),
        "retained_manifest_positions": int(keep.sum()),
        "subset_n": int(keep.sum()) - ctx["val_n"],
        "subset_val_n": ctx["val_n"],
        **policy,
        "removed_unique_pdb": len(removed),
    }
    (out / "loader_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    audit.update({key: repo_relative(value) for key, value in contract.items()})


def write_frozen_selection(ctx: dict, removed: set[str], audit: dict) -> None:
    src, out = ctx["source"], ctx["split"]
    out.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(src["split"] / "plinder_selected.csv")
    selected["entry_pdb_id"] = selected.entry_pdb_id.astype(str).str.lower()
    kept = selected[~selected.entry_pdb_id.isin(removed)].copy()
    selection_path = out / "plinder_selected.csv"
    kept.to_csv(selection_path, index=False)

    inputs = json.loads((src["split"] / "plinder_inputs.json").read_text())
    inputs.update({
        "name": ctx["tag"],
        "description": (f"{ctx['source_name']} with the proteins of {ctx['cohort_description']} "
                        f"removed ({ctx['rule']})."),
        "source_version": ctx["source_name"],
        "selection_sha256": sha256(selection_path),
        "n_selected": len(kept),
        "materialized_loader": {
            "source_manifest_positions": audit["source_manifest_positions"],
            "retained_manifest_positions": audit["retained_manifest_positions"],
            "subset_n": audit["subset_n"],
            "subset_val_n": audit["subset_val_n"],
            "source_data_file": str(src["data_file"].relative_to(ROOT)),
            "resample_dir": str(ctx["resample"].relative_to(ROOT)),
        },
    })
    inputs.setdefault("filters", {})[f"{ctx['slug']}_decontamination"] = {
        "cohorts": [COHORTS[c]["label"] for c in ctx["cohorts"]],
        **ctx["policy"],
        "policy": ("drop every ligand observation from a PDB with a qualifying protein-chain "
                   "hit into a selected cohort, or that is a cohort entry itself"),
        "removed_unique_pdb": audit["removed_unique_pdb"],
        "removed_manifest_positions": audit["removed_manifest_positions"],
    }
    holdout = inputs.setdefault("leakage_holdout", {})
    for name in ctx["cohorts"]:
        holdout[name] = {"label": COHORTS[name]["label"], "sources": cohort_sources(name)}
    holdout["plinder_v2_chain_fasta"] = {
        "path": str(PLINDER_FASTA.relative_to(ROOT)), "sha256": sha256(PLINDER_FASTA)
    }
    inputs["note"] = (f"{ctx['tag']} reuses the {ctx['source_name']} tuples and the "
                      f"position-aligned density box; the manifest availability mask is the "
                      f"authoritative load-time filter.")
    (out / "plinder_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")

    # The funnel is carried forward only if the source corpus has one.
    source_funnel = src["split"] / "plinder_funnel.json"
    if source_funnel.exists():
        funnel = json.loads(source_funnel.read_text())
        funnel["funnel"].append({
            "stage": f"remove {ctx['cohort_description']} proteins ({ctx['rule']})",
            "ligand_instances": len(kept),
            "unique_pdb": int(kept.entry_pdb_id.nunique()),
        })
        funnel.update({
            "n_selected": len(kept),
            "n_unique_pdb": int(kept.entry_pdb_id.nunique()),
            "n_chemotypes": int(kept.ligand_ccd_code.nunique()),
        })
        (out / "plinder_funnel.json").write_text(json.dumps(funnel, indent=2) + "\n")


def write_config(ctx: dict, audit: dict) -> Path:
    """The source corpus's dset config with the clean view's paths and counts."""
    source_cfg = ROOT / "pretrain/configs/dset" / f"{ctx['source']['dset']}.yaml"
    path = source_cfg.with_name(f"{ctx['tag']}.yaml")
    root = "${oc.env:VOXBIND_DATA_ROOT,../dataset/data}"
    resample = f"{root}/{ctx['resample'].relative_to(DATA)}"
    values = {
        "resample_dir": resample,
        "box_path": f"{resample}/{ctx['source']['box']}",
        "subset_n": audit["retained_manifest_positions"] - ctx["val_n"],
        "subset_val_n": ctx["val_n"],
    }
    body = []
    for line in source_cfg.read_text().splitlines():
        if line.startswith("#"):
            continue
        key = line.split(":", 1)[0]
        if key in values:
            line = f"{key}: {values[key]}"
        body.append(line)
    header = [
        f"# {ctx['tag']}: {ctx['source_name']} without the proteins of {ctx['cohort_description']}",
        f"# ({ctx['rule']}).",
        "# Built by dataset/plinder/02_make_clean_corpus.py.",
        f"# Retained {audit['retained_manifest_positions']} positions "
        f"({audit['retained_unique_pdb']} PDBs); removed {audit['removed_unique_pdb']} PDBs "
        f"({audit['removed_manifest_positions']} positions, {audit['removed_position_pct']}%).",
    ]
    path.write_text("\n".join(header + body) + "\n")
    return path


# ── driver ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCES), default="cdgv2")
    parser.add_argument("--cohorts",
                        help=f"comma-separated cohorts to exclude, from {list(COHORTS)} "
                             f"(default: all seven; cl3 with --similarity atom3d)")
    parser.add_argument("--tag", help="name to write the corpus under (default: CoDE_pretrain_v1 for cdgv2, else <source>_clean)")
    parser.add_argument("--similarity", choices=("diamond", "atom3d"), default="diamond",
                        help="diamond: identity over the aligned region with --coverage; atom3d: "
                             "BLASTP identical residues / cohort-chain length, as ATOM3D")
    parser.add_argument("--identity", type=float, default=100.0, help="minimum identity, percent")
    parser.add_argument("--coverage", type=float, default=80.0,
                        help="minimum coverage of the shorter sequence, percent")
    parser.add_argument("--val-n", type=int, default=100)
    parser.add_argument("--diamond")
    parser.add_argument("--threads", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--write-config", dest="write_config", action="store_true", default=True)
    parser.add_argument("--no-write-config", dest="write_config", action="store_false")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing corpus of this name")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve the cohorts and report counts; write nothing")
    parser.add_argument("--fetch-crossdocked", action="store_true",
                        help=f"rewrite {CROSSDOCKED_FASTA.relative_to(ROOT)} from RCSB and exit")
    args = parser.parse_args()
    if args.fetch_crossdocked:
        fetch_crossdocked_fasta()
        return

    default_cohorts = ["cl3"] if args.similarity == "atom3d" else DEFAULT_COHORTS
    cohorts = [c.strip() for c in (args.cohorts or ",".join(default_cohorts)).split(",") if c.strip()]
    unknown = [c for c in cohorts if c not in COHORTS]
    if unknown:
        raise SystemExit(f"unknown cohort(s) {unknown}; choose from {list(COHORTS)}")
    if not cohorts:
        raise SystemExit("--cohorts selected nothing")

    src = SOURCES[args.source]
    atom3d = args.similarity == "atom3d"
    tag = args.tag or (f"{args.source}_ID{int(args.identity)}" if atom3d
                       else DEFAULT_TAGS.get(args.source, f"{args.source}_clean"))
    slug_base = (ALL_COHORTS_SLUG if set(cohorts) == set(COHORTS)
                 else "_".join(sorted(cohorts)))
    ctx = {
        "tag": tag,
        "source": src,
        "source_name": args.source,
        "cohorts": cohorts,
        "slug": f"{slug_base}_{'atom3d' if atom3d else 'id'}{int(args.identity)}",
        "identity": args.identity,
        "coverage": args.coverage,
        "rule": (f"ATOM3D identity >= {args.identity:g}%: BLASTP identical residues / test-chain length"
                 if atom3d else
                 f"identity >= {args.identity:g}%, shorter coverage >= {args.coverage:g}%"),
        "policy": ({"similarity": "atom3d", "identity_min_percent": args.identity,
                    "identity_definition": "BLASTP HSP identical residues / test-chain length "
                                           "(atom3d/protein/sequence.py find_similar)"}
                   if atom3d else
                   {"identity_min_percent": args.identity,
                    "shorter_sequence_coverage_min_percent": args.coverage}),
        "val_n": args.val_n,
        "split": VOX / "splits/plinder" / tag,
        "resample": DATA / f"pretrain/xray_resample_{tag}",
        "cohort_description": " + ".join(COHORTS[c]["label"] for c in cohorts),
    }
    ctx["name"] = f"{tag}_{ctx['slug']}"

    required = [src["split"] / "plinder_selected.csv", src["resample"] / "train_manifest.npz",
                src["resample"] / src["box"], src["data_file"], PLINDER_FASTA, LP_CSV]
    required += [COHORTS[c]["path"] for c in cohorts if COHORTS[c]["kind"] in ("csv", "crossdocked")]
    if "crossdocked_test" in cohorts:
        required.append(CROSSDOCKED_FASTA)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    existing_contract = ctx["resample"] / "loader_contract.json"
    if existing_contract.exists() and not args.dry_run and not args.force:
        previous = json.loads(existing_contract.read_text())
        raise SystemExit(
            f"[refusing] {ctx['resample'].name} already exists "
            f"(retained {previous.get('retained_manifest_positions')} positions, removed "
            f"{previous.get('removed_unique_pdb')} PDBs). Pass --force to overwrite it, or "
            f"--tag <other> to write a new corpus."
        )

    with np.load(src["resample"] / "train_manifest.npz", allow_pickle=False) as manifest:
        source_pids = pd.Series(manifest["pdb_id"].astype(str)).str.lower()
        source_ok = manifest["ok"].astype(bool)
    source_pdbs = set(source_pids[source_ok])

    pids: set[str] = set()          # held-out entries (same-entry removal)
    sequence_pids: set[str] = set() # entries whose LP-PDBBind chains are searched
    per_cohort = {}
    for name in cohorts:
        got = cohort_pids(name)
        per_cohort[name] = len(got)
        pids |= got
        if COHORTS[name]["kind"] != "crossdocked":
            sequence_pids |= got
    crossdocked = "crossdocked_test" in cohorts
    print(f"cohorts {cohorts} → {len(pids)} unique held-out pids "
          f"({', '.join(f'{k}={v}' for k, v in per_cohort.items())})", flush=True)

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="plinder_clean_dry_") as tmp:
            n_complex, n_chain = write_cohort_fasta(sequence_pids, Path(tmp) / "cohort.fasta",
                                                    crossdocked)
        print(json.dumps({
            "source": args.source, "tag": tag, "slug": ctx["slug"], "cohorts": per_cohort,
            "cohort_pids": len(pids), "cohort_complexes": n_complex,
            "cohort_chains": n_chain, "cohort_entries_in_source": len(source_pdbs & pids),
            "would_write": {"split": str(ctx["split"]), "resample": str(ctx["resample"])},
            "dry_run": True,
        }, indent=2))
        return

    ctx["split"].mkdir(parents=True, exist_ok=True)
    cohort_fasta = ctx["split"] / f"{ctx['slug']}_cohort.fasta"
    n_complex, n_chain = write_cohort_fasta(sequence_pids, cohort_fasta, crossdocked)
    print(f"  cohort fasta: {n_complex} complexes, {n_chain} chains; running "
          f"{'BLASTP' if atom3d else 'DIAMOND'}...", flush=True)
    if atom3d:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from voxbind.splits.atom3d_identity import blast_version
        witnesses = best_witnesses_atom3d(search_atom3d(cohort_fasta, args.threads, args.identity),
                                          source_pdbs)
        tool_version = blast_version()
    else:
        binary = diamond_binary(args.diamond)
        with tempfile.TemporaryDirectory(prefix="plinder_clean_") as tmp:
            hits = search(binary, cohort_fasta, Path(tmp) / "hits.tsv", args.threads,
                          args.identity, args.coverage)
        witnesses = best_witnesses(hits, source_pdbs)
        tool_version = subprocess.check_output([str(binary), "version"], text=True).strip()
    searched_removed = set(witnesses.plinder_pdb)
    same_entry = source_pdbs & pids
    removed = searched_removed | same_entry
    fasta_pdbs = {line[1:].split("_")[0].lower()
                  for line in PLINDER_FASTA.read_text().splitlines() if line.startswith(">")}

    removed_positions = int((source_ok & source_pids.isin(removed).to_numpy()).sum())
    retained_positions = int(source_ok.sum()) - removed_positions
    audit = {
        "name": ctx["name"],
        "tag": tag,
        "source": args.source,
        "cohorts": per_cohort,
        "cohort_description": ctx["cohort_description"],
        "cohort_pids": len(pids),
        "cohort_complexes": n_complex,
        "cohort_chain_queries": n_chain,
        "cohort_fasta_sha256": sha256(cohort_fasta),
        **ctx["policy"],
        "search_tool": tool_version,
        "searched_removed_pdb": len(searched_removed),
        "same_entry_removed_pdb": len(same_entry),
        "unsearched_source_pdb": len(source_pdbs - fasta_pdbs),
        "source_manifest_positions": int(len(source_pids)),
        "source_available_positions": int(source_ok.sum()),
        "source_unique_pdb": len(source_pdbs),
        "removed_manifest_positions": removed_positions,
        "removed_unique_pdb": len(removed),
        "retained_manifest_positions": retained_positions,
        "retained_unique_pdb": len(source_pdbs - removed),
        "removed_position_pct": round(100 * removed_positions / int(source_ok.sum()), 3),
        "subset_n": retained_positions - args.val_n,
        "subset_val_n": args.val_n,
    }

    witnesses.to_csv(ctx["split"] / f"{ctx['slug']}_matches.tsv", sep="\t", index=False)
    (ctx["split"] / f"{ctx['slug']}_removed_pdbs.txt").write_text(
        "\n".join(sorted(removed)) + "\n")

    write_loader_view(ctx, removed, audit)
    write_frozen_selection(ctx, removed, audit)
    if args.write_config:
        audit["config_written"] = str(write_config(ctx, audit).relative_to(ROOT))

    (ctx["split"] / f"{ctx['slug']}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"\nvalidate with: python dataset/plinder/validate_clean_corpus.py --corpus {tag}")


if __name__ == "__main__":
    main()

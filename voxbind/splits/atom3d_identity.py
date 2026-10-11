"""Sequence identity as ATOM3D defines it, for test-anchored filtering.

ATOM3D (atom3d/protein/sequence.py, find_similar) runs BLASTP of each test chain against all other
chains with default parameters and no cap on the number of hits, and calls a chain similar when one
HSP has identical residues / test-chain length >= cutoff. The CoDE_pretrain_v1_ID{30,60} corpora and
the *_ID{30,60} splits use this rule (dataset/plinder/02_make_clean_corpus.py, make_splits.py).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

# BLAST+ comes from the `blast` conda env unless $BLAST_BIN or PATH has it.
DEFAULT_BLAST_BIN = Path.home() / ".conda/envs/blast/bin"
COLUMNS = ["qseqid", "sseqid", "nident", "qlen", "slen", "pident", "length", "evalue", "bitscore"]


# The directory holding blastp and makeblastdb.
def blast_dir() -> Path:
    for candidate in (os.environ.get("BLAST_BIN"), shutil.which("blastp") and Path(shutil.which("blastp")).parent,
                      DEFAULT_BLAST_BIN):
        if candidate and (Path(candidate) / "blastp").is_file():
            return Path(candidate)
    raise FileNotFoundError("BLAST+ not found: set BLAST_BIN or create the `blast` conda env")


def blast_version() -> str:
    return subprocess.check_output([str(blast_dir() / "blastp"), "-version"], text=True).splitlines()[0]


# Unique sequences as u0, u1, ...: {sequence: id} and {id: [names]}.
def _dedupe(chains: dict[str, str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    ids, members = {}, {}
    for name, seq in chains.items():
        if seq:
            key = ids.setdefault(seq, f"u{len(ids)}")
            members.setdefault(key, []).append(name)
    return ids, members


# FASTA text read into {name: sequence}; the name is the header up to the first space.
def read_fasta(path: Path) -> dict[str, str]:
    chains, name = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            chains[name] = ""
        elif name:
            chains[name] += line.strip().upper()
    return chains


# Every (query, subject) pair with ATOM3D identity >= cutoff, one row per pair at its best HSP:
# query, subject, identity, nident, qlen, slen, evalue, bitscore. Queries are the test chains.
def similar_pairs(queries: dict[str, str], database: dict[str, str], cutoff: float, *,
                  threads: int = 8, workdir: str | None = None) -> pd.DataFrame:
    q_ids, q_members = _dedupe(queries)
    d_ids, d_members = _dedupe(database)
    with tempfile.TemporaryDirectory(prefix="atom3d_", dir=workdir) as tmp:
        tmp = Path(tmp)
        (tmp / "q.fasta").write_text("".join(f">{k}\n{s}\n" for s, k in q_ids.items()))
        (tmp / "db.fasta").write_text("".join(f">{k}\n{s}\n" for s, k in d_ids.items()))
        bin_dir = blast_dir()
        subprocess.run([str(bin_dir / "makeblastdb"), "-in", str(tmp / "db.fasta"), "-dbtype", "prot",
                        "-out", str(tmp / "db")], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["nice", "-n", "19", str(bin_dir / "blastp"), "-query", str(tmp / "q.fasta"),
                        "-db", str(tmp / "db"), "-outfmt", "6 " + " ".join(COLUMNS),
                        "-max_target_seqs", str(len(d_ids)), "-num_threads", str(threads),
                        "-out", str(tmp / "hits.tsv")], check=True)
        hits = pd.read_csv(tmp / "hits.tsv", sep="\t", header=None, names=COLUMNS)

    hits["identity"] = hits.nident / hits.qlen
    hits = hits[hits.identity >= cutoff]
    best = hits.sort_values(["identity", "bitscore"], ascending=False).drop_duplicates(["qseqid", "sseqid"])

    # Back from unique sequences to every chain that carries them.
    rows = [(qn, sn, r.identity, r.nident, r.qlen, r.slen, r.evalue, r.bitscore)
            for r in best.itertuples(index=False)
            for qn in q_members[r.qseqid] for sn in d_members[r.sseqid]]
    return pd.DataFrame(rows, columns=["query", "subject", "identity", "nident", "qlen", "slen",
                                       "evalue", "bitscore"])

"""The data every baseline is built from, and where its featurised copy goes.

The baselines read the same complexes, labels and splits as our encoders, straight from this
repository's data, and write what they featurise from them to base/data/<method>/:

    STRUCT_BASES  dataset/data/pdbbind/structures/{pbpp-2020,misato_qm_built}/<pid>/
                  <pid>_protein.pdb, <pid>_pocket.pdb, <pid>_ligand.sdf (first base that has the pid)
    LP_CSV        dataset/data/pdbbind/raw/LP_PDBBind.csv   pK (column value), SMILES, sequence
    SPLITS_DIR    voxbind/splits/<split>.csv                frozen pid,split manifests
    CASF_CSV      voxbind/splits/casf2016_eval.csv          CASF-2016 core set: pid, in_v2train, pK

Standard library only, so every baseline env can import it after
sys.path.insert(0, "<repo>/base").
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
PDBBIND = os.path.join(REPO, "dataset", "data", "pdbbind")
STRUCT_BASES = [os.path.join(PDBBIND, "structures", "pbpp-2020"),
                os.path.join(PDBBIND, "structures", "misato_qm_built")]
LP_CSV = os.path.join(PDBBIND, "raw", "LP_PDBBind.csv")
SPLITS_DIR = os.path.join(REPO, "voxbind", "splits")
CASF_CSV = os.path.join(SPLITS_DIR, "casf2016_eval.csv")
DATA = os.path.join(BASE, "data")


def split_csv(split: str) -> str:
    """voxbind/splits/<split>.csv"""
    return os.path.join(SPLITS_DIR, f"{split}.csv")


def data_dir(method: str) -> str:
    """base/data/<method>, where a baseline keeps its featurised data (created on first use)."""
    path = os.path.join(DATA, method)
    os.makedirs(path, exist_ok=True)
    return path


def structure_dir(pid: str):
    """<base>/<pid> for the first STRUCT_BASES entry that has the complex, else None."""
    for base in STRUCT_BASES:
        path = os.path.join(base, pid)
        if os.path.isdir(path):
            return path
    return None

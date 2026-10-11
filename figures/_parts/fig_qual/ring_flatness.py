"""PoseBusters `non-aromatic_ring_non-flatness`, re-derived ring by ring.

Shared by ring-nonflatness-nglview.ipynb and build_ring_nonflatness_site.py, so the notebook
and the local site always agree on which ring failed and why.

What the check tests (posebusters 0.6.5, config/dock.yml + modules/flatness.py):
  * only 6-membered non-aromatic rings with 0-2 double bonds. The 5-ring pattern is commented
    out upstream, so a flat cyclopentane can never fail it.
  * a ring passes if at least one atom sits >= 0.05 A from the ring's best-fit plane.
  * the distance is SIGNED: PoseBusters takes max(d), not max(|d|), and the sign of the SVD
    normal is arbitrary. A ring bent to one side only can fail with |d| = 0.08 A, so each ring
    gets one of three statuses:
      fail            fails whichever way the normal points (max|d| < 0.05)
      sign-dependent  fails for one sign, passes for the other
      pass            passes for both signs
"""

import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

RDLogger.DisableLog("rdApp.*")

# The repository this file sits in (figures/_parts/fig_qual/).
REPO = Path(__file__).resolve().parents[3]
E = REPO / "task2-ligand_generation" / "exps"
# Same roots as figures/_parts/00_core.py ARMS.
ARMS = {
    "CoDE":    E / "voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350",
    "VoxBind": E / "_vanilla_ep923/samples/full_eval_ep923",
}
P79 = sorted(json.load(open(E / "frozenenc_probes/p79_targets.json")))   # the paper's pocket set

# Copied verbatim from posebusters 0.6.5 config/dock.yml ("Ring non-flatness").
PB_NONFLAT_SMARTS = {
    "non-aromatic_6_membered_rings":        "[C,O,S,N;R1]~1[C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1]1",
    "non-aromatic_6_membered_rings_db03_0": "[C;R1]~1[C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1][C;R1]1",
    "non-aromatic_6_membered_rings_db03_1": "[C;R1]~1[C;R1][C;R1]~[C;R1][C,O,S,N;R1][C;R1]1",
    "non-aromatic_6_membered_rings_db02_0": "[C;R1]~1[C;R1][C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1]1",
    "non-aromatic_6_membered_rings_db02_1": "[C;R1]~1[C;R1][C,O,S,N;R1][C;R1]~[C;R1][C;R1]1",
}
PB_PATTS = [Chem.MolFromSmarts(s) for s in PB_NONFLAT_SMARTS.values()]
PB_THR = 0.05   # threshold_flatness, in A

# Common names for the ring cores that show up; anything else is shown as its SMILES.
RING_NAMES = {
    "C1CCCCC1": "cyclohexane", "C1=CCCCC1": "cyclohexene",
    "C1=CCC=CC1": "1,4-cyclohexadiene", "C1=CCCC=C1": "1,3-cyclohexadiene",
    "C1CCNCC1": "piperidine", "C1CNCCN1": "piperazine", "C1COCCN1": "morpholine",
    "C1CCOCC1": "tetrahydropyran", "C1COCCO1": "1,4-dioxane",
    "C1=CCN=CC1": "dihydropyridine (C1=CCN=CC1)", "C1=CCCN=C1": "dihydropyridine (C1=CCCN=C1)",
    "C1=CNC=CC1": "1,4-dihydropyridine", "C1=CCNC=C1": "1,2-dihydropyridine",
    "C1CCCC1": "cyclopentane", "C1=CCCC1": "cyclopentene", "C1CCNC1": "pyrrolidine",
    "C1CCOC1": "tetrahydrofuran", "C1CCSC1": "tetrahydrothiophene", "C1CNCN1": "imidazolidine",
}


def signed_plane_dist(X):
    """Signed distance of each point to the best-fit plane, exactly as PoseBusters computes it."""
    Y = X - X.mean(0)
    n = np.linalg.svd(Y)[2][-1]
    return Y @ n, n


def ring_core(m, ring):
    """Canonical SMILES of the ring atoms and ring bonds only (substituents and exocyclic C=O dropped)."""
    rw, idx = Chem.RWMol(), {}
    for i in ring:
        a = Chem.Atom(m.GetAtomWithIdx(i).GetSymbol()); a.SetIsAromatic(False); idx[i] = rw.AddAtom(a)
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx and j in idx:
            rw.AddBond(idx[i], idx[j], b.GetBondType())
    return Chem.MolToSmiles(rw.GetMol())


def load_target(tdir):
    """Molecules in the order metrics.json stores them: the evaluator that wrote it skips
    records that fail sanitization and multi-fragment molecules, so a naive zip against
    samples.sdf goes out of step after the first skipped record."""
    tdir = Path(tdir)
    rows = json.load(open(tdir / "metrics.json")).get("samples") or []
    mols = []
    for m in Chem.SDMolSupplier(str(tdir / "samples.sdf"), sanitize=True):
        if m is None:
            continue
        try:
            if "." in Chem.MolToSmiles(m):
                continue
        except Exception:
            continue
        mols.append(m)
    if len(mols) != len(rows):
        raise ValueError(f"{tdir.name}: {len(mols)} mols vs {len(rows)} metrics rows")
    return mols, rows


def ring_records(m):
    """One record per non-aromatic ring: size, core, max|d|, and PoseBusters' value for both normal signs."""
    X = m.GetConformer().GetPositions()
    checked = {frozenset(g) for p in PB_PATTS for g in m.GetSubstructMatches(p)}
    out = []
    for r in m.GetRingInfo().AtomRings():
        if any(m.GetAtomWithIdx(i).GetIsAromatic() for i in r):
            continue
        d, _ = signed_plane_dist(X[list(r)])
        hi, lo = d.max(), (-d).max()          # PB's value for the two possible normal signs
        pb = frozenset(r) in checked
        status = ("not checked" if not pb else
                  "fail" if max(hi, lo) < PB_THR else
                  "sign-dependent" if min(hi, lo) < PB_THR else "pass")
        core = ring_core(m, r)
        # sp2 ring atoms, counting exocyclic C=O / C=N too (which `core` does not show):
        # a ring with none is a true saturated ring that must pucker.
        n_sp2 = sum(m.GetAtomWithIdx(i).GetHybridization() != Chem.HybridizationType.SP3 for i in r)
        out.append(dict(size=len(r), core=core, name=RING_NAMES.get(core, core), atoms=tuple(r), n_sp2=n_sp2,
                        max_abs_dev=float(np.abs(d).max()), pb_value_lo=float(min(hi, lo)),
                        pb_checked=pb, status=status))
    return out


def relax(m):
    """MMFF-relaxed copy (UFF when MMFF has no parameters), started from the generated pose.
    Hydrogens are added for the minimisation and removed again, so heavy-atom indices match `m`."""
    mh = Chem.AddHs(m, addCoords=True)
    try:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh, maxIters=5000)
        else:
            AllChem.UFFOptimizeMolecule(mh, maxIters=5000)
    except Exception:
        return None
    return Chem.RemoveHs(mh)


def relaxed_copy(m, ring):
    """The relaxed molecule superposed on the generated ring atoms, for drawing."""
    r = relax(m)
    if r is not None:
        rdMolAlign.AlignMol(r, m, atomMap=[(i, i) for i in ring])
    return r


def relaxed_ring_dev(m, ring):
    r = relax(m)
    if r is None:
        return np.nan
    return float(np.abs(signed_plane_dist(r.GetConformer().GetPositions()[list(ring)])[0]).max())


def pocket_residues(pdb_path, lig_xyz, cutoff=5.0):
    """NGL selection string for residues with any atom within `cutoff` A of the ligand."""
    keep = set()
    for line in open(pdb_path):
        if line.startswith(("ATOM", "HETATM")):
            x = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if np.min(np.linalg.norm(lig_xyz - x, axis=1)) < cutoff:
                keep.add((int(line[22:26]), line[21].strip()))
    return " or ".join(f"{r}:{c}" if c else str(r) for r, c in sorted(keep)) or "none"

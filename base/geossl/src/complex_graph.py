"""Build a GeoSSL LBA-style complex graph from local PDBbind structures.

Faithfully replicates GeoSSL's `TransformLBA` + `DatasetLBA.process` featurisation
(chao1224/GeoSSL, Geom3D/datasets/datasets_LBA.py):

  * pocket = protein heavy atoms within `dist` (=6 A) of ANY ligand heavy atom,
    then capped to the `maxnum` (=500) minus n_ligand closest atoms,
  * ligand = ligand heavy atoms, shifted by +1000 A along x (`move_lig=True`) so the
    radius graph never connects ligand<->pocket (the ATOM3D-ENN trick),
  * each atom -> categorical index into
      possible_atomic_num_list = [1, 6, 7, 8, 9, 15, 16, 17, "unknown"]  (9 classes),
    H atoms dropped, any element outside the list -> "unknown" (index 8).

Returns (z_idx LongTensor[N], pos FloatTensor[N,3]) for the merged pocket+ligand graph,
ready to feed the vendored SchNet.
"""
import os
import numpy as np
from rdkit import Chem

# GeoSSL element vocabulary (Geom3D/datasets/datasets_utils.py)
POSSIBLE_ATOMIC_NUM = [1, 6, 7, 8, 9, 15, 16, 17]  # H C N O F P S Cl -> idx 0..7 ; else -> 8 ("unknown")
UNKNOWN_IDX = 8

# element symbol -> atomic number (for the few we care about; else falls to unknown)
_SYM2Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "CL": 17, "Cl": 17,
}
_WATER = {"HOH", "WAT", "DOD"}


def _z_to_idx(z):
    try:
        return POSSIBLE_ATOMIC_NUM.index(z)
    except ValueError:
        return UNKNOWN_IDX


def _parse_pdb_heavy(path):
    """Parse ATOM/HETATM heavy atoms from a PDB file -> (elements[str list], coords[N,3]).

    Drops hydrogens and water. Element taken from cols 77-78 when present, else inferred
    from the atom name.
    """
    elems, coords = [], []
    with open(path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if resname in _WATER:
                continue
            # element (cols 77-78, 1-indexed 77-78 -> py 76:78)
            el = line[76:78].strip()
            if not el:
                # infer from atom name (cols 13-16)
                name = line[12:16].strip()
                el = "".join(c for c in name if c.isalpha())[:2]
            el_u = el.upper()
            if el_u == "H" or el_u == "D":
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            elems.append(el_u)
            coords.append([x, y, z])
    return elems, np.asarray(coords, dtype=np.float32).reshape(-1, 3)


def _read_ligand(mol2_path, sdf_path):
    """Robust ligand read (mol2 -> sdf, sanitize fallback). Returns rdkit Mol with a conformer."""
    for path, reader in ((mol2_path, Chem.MolFromMol2File), (sdf_path, Chem.MolFromMolFile)):
        if path and os.path.exists(path):
            for san in (True, False):
                try:
                    m = reader(path, sanitize=san, removeHs=True)
                except TypeError:
                    m = reader(path, sanitize=san)
                if m is not None and m.GetNumConformers() > 0:
                    return m
    # last resort: SDF supplier
    if sdf_path and os.path.exists(sdf_path):
        for san in (True, False):
            supp = Chem.SDMolSupplier(sdf_path, sanitize=san, removeHs=True)
            for m in supp:
                if m is not None and m.GetNumConformers() > 0:
                    return m
    return None


def _ligand_heavy(mol):
    """(atomic_numbers[list], coords[N,3]) for heavy ligand atoms."""
    conf = mol.GetConformer()
    zs, coords = [], []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        zs.append(z)
        coords.append([p.x, p.y, p.z])
    return zs, np.asarray(coords, dtype=np.float32).reshape(-1, 3)


def build_complex_graph(protein_pdb, mol2_path, sdf_path, dist=6.0, maxnum=500, move_dx=1000.0):
    """Return (z_idx np.int64[N], pos np.float32[N,3]) or None on failure."""
    mol = _read_ligand(mol2_path, sdf_path)
    if mol is None:
        return None
    lig_z, lig_pos = _ligand_heavy(mol)
    if len(lig_z) == 0:
        return None

    pk_el, pk_pos = _parse_pdb_heavy(protein_pdb)
    if pk_pos.shape[0] == 0:
        return None

    # --- select pocket env by distance (<= dist of any ligand atom) ---
    d = np.linalg.norm(pk_pos[:, None, :] - lig_pos[None, :, :], axis=-1)  # (P, L)
    min_d = d.min(axis=1)                                                  # nearest lig atom
    keep = np.where(min_d <= dist)[0]
    if keep.size == 0:
        return None
    pk_pos_k = pk_pos[keep]
    pk_el_k = [pk_el[i] for i in keep]
    min_d_k = min_d[keep]

    # --- cap to maxnum - n_lig closest atoms ---
    num = int(max(1, maxnum - len(lig_z)))
    if pk_pos_k.shape[0] > num:
        order = np.argsort(min_d_k)[:num]
        order = np.sort(order)
        pk_pos_k = pk_pos_k[order]
        pk_el_k = [pk_el_k[i] for i in order]

    # --- element -> vocab index ---
    pk_idx = [_z_to_idx(_SYM2Z.get(e.capitalize(), _SYM2Z.get(e, -1))) for e in pk_el_k]
    lig_idx = [_z_to_idx(z) for z in lig_z]

    # --- move ligand far along x so radius graph keeps pocket/ligand disconnected ---
    lig_pos_moved = lig_pos.copy()
    lig_pos_moved[:, 0] += move_dx

    z_idx = np.asarray(pk_idx + lig_idx, dtype=np.int64)
    pos = np.concatenate([pk_pos_k, lig_pos_moved], axis=0).astype(np.float32)
    return z_idx, pos

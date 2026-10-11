"""PoseBusters (Buttenschoen et al., 2024): pose validity in `dock` mode, ligand plus protein.

Runs in the evaluation environment through pose_worker.py. `valid` requires every graded check to
pass; the two *_loaded columns only say the file was read and are not graded. With 0.6.5 that is 20
graded checks, so a release that adds or renames a check moves every validity number.
"""
from __future__ import annotations

from typing import Dict, List

INFORMATIONAL = frozenset({"mol_pred_loaded", "mol_cond_loaded"})


def _record(row) -> Dict:
    checks = {k: (None if v is None else bool(v)) for k, v in row.items()}
    graded = {k: v for k, v in checks.items() if k not in INFORMATIONAL and v is not None}
    n_pass = sum(1 for v in graded.values() if v)
    return {"valid": bool(graded and n_pass == len(graded)), "n_pass": n_pass,
            "n_checks": len(graded), "checks": checks}


def score(protein_pdb: str, mols: List) -> List[Dict]:
    """One record per ligand: one batch, falling back to ligand-by-ligand on failure."""
    from posebusters import PoseBusters

    try:
        df = PoseBusters(config="dock").bust(mols, None, protein_pdb, full_report=False)
        return [_record(df.iloc[i]) for i in range(len(mols))]
    except Exception:  # noqa: BLE001
        out: List[Dict] = []
        for m in mols:
            if m is None:
                out.append({"error": "unparsable mol"})
                continue
            try:
                df = PoseBusters(config="dock").bust([m], None, protein_pdb, full_report=False)
                out.append(_record(df.iloc[0]))
            except Exception as e:  # noqa: BLE001
                out.append({"error": "{}: {}".format(type(e).__name__, e)})
        return out

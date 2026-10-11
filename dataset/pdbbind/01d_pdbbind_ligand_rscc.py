"""01d_pdbbind_ligand_rscc.py — fetch per-ligand real-space CC (RSCC) for PDBbind complexes.

For the "LP-PDBBind ∩ ED-available with BOTH protein & ligand density" downstream dataset, we
need a ligand-resolution check analogous to the PLINDER pretraining filter (ligand RSCC ≥ 0.8).
PLINDER shipped RSCC in its annotation table; PDBbind does not, so we pull it from the wwPDB
validation report (RCSB mirror), which carries per-residue RSCC.

Per complex: the ligand = the largest non-standard / non-junk HETATM residue (drug-like, ≥6 atoms);
RSCC = best-resolved copy of that residue (max over copies). Protein-side density is already
guaranteed by has_density (the 2Fo-Fc map covers the pocket), and we additionally report the
pocket-residue median RSCC so callers can require BOTH ≥ threshold.

Concurrent, incremental (resumes by skipping pids already in the out CSV), polite retries.

Run from the repo root:
    python dataset/pdbbind/01d_pdbbind_ligand_rscc.py --workers 24 \
        --out dataset/data/pdbbind/ligand_rscc.csv
"""
import argparse, csv, gzip, os, sys, time, urllib.request, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PDB = Path(__file__).resolve().parents[1] / "data" / "pdbbind"
VAL_URL = "https://files.rcsb.org/pub/pdb/validation_reports/{ab}/{pid}/{pid}_validation.xml.gz"

AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO',
      'SER','THR','TRP','TYR','VAL','MSE','SEC','PYL','UNK'}
NUC = {'A','C','G','U','T','DA','DC','DG','DT','DU','DI','N'}
JUNK = {'HOH','DOD','SO4','PO4','GOL','EDO','PEG','PG4','PGE','1PE','ACT','DMS','MPD','TRS','EPE',
        'FMT','NA','CL','MG','ZN','CA','K','MN','FE','FE2','CU','NI','CO','CD','HG','BR','IOD','NO3',
        'CO3','IMD','BME','CIT','TLA','FLC','MES','CAC','NH4','ACE','NAG','BMA','MAN','FUC','GAL','BGC',
        'GLC','XYP','SIA','PO4','UNX','UNL','PLP','SF4','FES','OXY','PER','OH','O','MOH','EOH','BU3',
        'P6G','12P','15P','2PE','ACY','SCN','AZI','PG0','POP','SPM','SPD','BEN'}


def vurl(pid):
    pid = pid.lower()
    return VAL_URL.format(ab=pid[1:3], pid=pid)


def fetch_rscc(pid, retries=2, timeout=30):
    """→ dict(pid, status, lig_resname, lig_rscc, lig_natoms, n_copies, poc_rscc_median, n_poc)."""
    err = ""
    for k in range(retries + 1):
        try:
            raw = urllib.request.urlopen(vurl(pid), timeout=timeout).read()
            xml = gzip.decompress(raw)
            break
        except Exception as e:
            err = f"{type(e).__name__}"
            if k == retries:
                return dict(pid=pid, status=f"fetch_err:{err}", lig_resname="", lig_rscc="",
                            lig_natoms="", n_copies="", poc_rscc_median="", n_poc="")
            time.sleep(1.0 + k)
    try:
        root = ET.fromstring(xml)
    except Exception as e:
        return dict(pid=pid, status=f"parse_err:{type(e).__name__}", lig_resname="", lig_rscc="",
                    lig_natoms="", n_copies="", poc_rscc_median="", n_poc="")
    ligs, poc = {}, []
    for sg in root.iter("ModelledSubgroup"):
        a = sg.attrib
        rscc = a.get("rscc")
        if rscc is None:
            continue
        try:
            rscc = float(rscc)
        except ValueError:
            continue
        rn = a.get("resname")
        if rn in AA or rn in NUC:
            poc.append(rscc)
        elif rn and rn not in JUNK:
            na = a.get("natoms") or a.get("NatomsEDS") or 0
            try:
                na = int(na)
            except ValueError:
                na = 0
            ligs.setdefault(rn, []).append((rscc, na))
    poc_med = (sorted(poc)[len(poc) // 2] if poc else "")
    # ligand = non-junk HETATM resname with the most atoms (>=6), best-resolved copy
    drug = {rn: v for rn, v in ligs.items() if max(n for _, n in v) >= 6}
    if not drug:
        return dict(pid=pid, status="no_ligand", lig_resname="", lig_rscc="", lig_natoms="",
                    n_copies="", poc_rscc_median=poc_med, n_poc=len(poc))
    rn = max(drug, key=lambda r: max(n for _, n in drug[r]))
    copies = drug[rn]
    best = max(copies, key=lambda c: c[0])
    return dict(pid=pid, status="ok", lig_resname=rn, lig_rscc=round(best[0], 4),
                lig_natoms=best[1], n_copies=len(copies), poc_rscc_median=(round(poc_med, 4) if poc_med != "" else ""),
                n_poc=len(poc))


def pool_pids():
    """The LP-PDBBind complexes that have a usable density map.

    THE AVAILABILITY TABLE IS THE PREFERRED SOURCE AND IS NOT ALWAYS THERE. It is written
    by the v5 voxelizer, so a box that received the features but never ran the voxelizer --
    this is one -- has `voxels_v5/` without `availability.csv` and this used to die with a
    bare FileNotFoundError. `eds_cache.json` records exactly the same has_density decision
    (it IS what the voxelizer consulted), so it is the fallback; a box with neither gets a
    message naming both rather than a traceback."""
    lp = pd.read_csv(PDB / "raw" / "LP_PDBBind.csv")
    lp["pid"] = lp["Unnamed: 0"].astype(str).str.lower()
    av_csv = PDB / "voxels_v5" / "availability.csv"
    if av_csv.exists():
        av = pd.read_csv(av_csv)
        av["pid"] = av.pdb_id.astype(str).str.lower()
        m = lp.merge(av[["pid", "has_density", "filtered"]], on="pid", how="left")
        return m[(m.has_density == True) & (m.filtered != True)].pid.tolist()
    eds = PDB / "eds_cache.json"
    if not eds.exists():
        sys.exit(f"neither {av_csv} nor {eds} exists — cannot tell which complexes have a "
                 f"density map; pass --pids with an explicit list instead")
    import json
    with open(eds, encoding="utf-8") as fh:
        cache = json.load(fh)
    keep = {k.lower() for k, v in cache.items() if v.get("has_density")}
    print(f"  [pool] no availability.csv — using eds_cache.json ({len(keep):,} with density)")
    return [p for p in lp.pid.tolist() if p in keep]


def listed_pids(path):
    """pids from a file: one per line, or a CSV with a `pid` column.

    Added so this can be pointed at ONE split's test set (or at the CrossDocked receptors)
    instead of the whole 16k pool -- the validation reports are fetched one HTTP request per
    complex, so scoping the run is the difference between minutes and an hour."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = "pid" if "pid" in df.columns else df.columns[0]
        return [str(p).lower() for p in df[col]]
    return [ln.strip().lower() for ln in path.read_text().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PDB / "ligand_rscc.csv"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--pids", default=None,
                    help="File of pids to fetch (one per line, or a CSV with a `pid` column) "
                         "instead of the whole density-available pool.")
    args = ap.parse_args()

    pids = listed_pids(args.pids) if args.pids else pool_pids()
    if args.limit:
        pids = pids[:args.limit]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = set(pd.read_csv(out).pid.astype(str))
    todo = [p for p in pids if p not in done]
    print(f"pool={len(pids)} | done={len(done)} | todo={len(todo)} | workers={args.workers}", flush=True)

    cols = ["pid", "status", "lig_resname", "lig_rscc", "lig_natoms", "n_copies", "poc_rscc_median", "n_poc"]
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    if new:
        w.writeheader()
    t0 = time.time(); n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_rscc, p): p for p in todo}
        for fut in as_completed(futs):
            w.writerow(fut.result()); n += 1
            if n % 200 == 0:
                f.flush(); dt = time.time() - t0
                print(f"  {n}/{len(todo)}  ({n/dt:.1f}/s, ~{(len(todo)-n)/max(n/dt,1e-9)/60:.0f} min left)", flush=True)
    f.close()
    df = pd.read_csv(out)
    ok = df[df.status == "ok"]
    print(f"\n[done] {len(df)} rows | ok={len(ok)} | no_ligand={int((df.status=='no_ligand').sum())} | "
          f"err={int(df.status.str.contains('err').sum())}", flush=True)
    if len(ok):
        print(f"  lig_rscc>=0.8: {int((ok.lig_rscc>=0.8).sum())} | >=0.7: {int((ok.lig_rscc>=0.7).sum())}", flush=True)


if __name__ == "__main__":
    main()

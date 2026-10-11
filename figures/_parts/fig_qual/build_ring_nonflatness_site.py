#!/usr/bin/env python3
"""Build the static ring non-flatness site (the web version of ring-nonflatness-nglview.ipynb).

    python figures/fig-qual/build_ring_nonflatness_site.py [--arms CoDE VoxBind] [--out DIR]

Run it with the voxel-bind env python. It writes a self-contained static tree (default
.cache/ring-nonflatness-site/) that run_ring_nonflatness.sh serves:

    index.html                     copied from ring_nonflatness_site.html
    ngl.js                         NGL 2.3.1, the engine nglview wraps (downloaded once)
    data/arms.json                 which arms were built
    data/<arm>/index.json          summary tables + one row per non-aromatic ring
    data/<arm>/mols/<t>_<k>.json   molblock, pocket residues, per-ring plane + relaxed overlay
    data/<arm>/pockets/<t>.pdb     the pocket10 crop

Every path the page fetches is relative, so the tree also works behind the ncloud proxy.
"""

import argparse
import json
import shutil
import sys
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ring_flatness as rf  # noqa: E402

NGL_URL = "https://cdn.jsdelivr.net/npm/ngl@2.3.1/dist/ngl.js"
ROW_KEYS = ["target", "k", "ring", "size", "name", "core", "n_sp2", "max_abs_dev",
            "pb_value_lo", "status", "pb_metrics", "relaxed_max_abs_dev", "n_atoms"]
HIST_BINS = np.round(np.linspace(0, 0.8, 41), 3)
FAMILIES = {
    "all-sp3 6-rings (cyclohexane, piperidine, ...)":
        lambda r: r["size"] == 6 and r["n_sp2"] == 0,
    "6-rings with sp2 atoms (cyclohexenes, dienes, dihydro-azines, lactams)":
        lambda r: r["size"] == 6 and r["n_sp2"] > 0,
    "all-sp3 5-rings (cyclopentane, pyrrolidine, THF, ...)":
        lambda r: r["size"] == 5 and r["n_sp2"] == 0,
}


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def molecule_payload(job):
    """Everything the viewer needs for one molecule; runs in a worker process."""
    target, k, molblock, pocket_pdb = job
    m = Chem.MolFromMolBlock(molblock, sanitize=True)
    X = m.GetConformer().GetPositions()
    relaxed = rf.relax(m)
    recs, rings = rf.ring_records(m), []
    for rec in recs:
        ring = list(rec["atoms"])
        d, n = rf.signed_plane_dist(X[ring])
        item = dict(atoms=ring, d=[r3(v) for v in d], normal=[r3(v) for v in n],
                    centroid=[r3(v) for v in X[ring].mean(0)],
                    relaxed_xyz=None, relaxed_max_abs_dev=None)
        if relaxed is not None:
            rr = Chem.Mol(relaxed)
            rdMolAlign.AlignMol(rr, m, atomMap=[(i, i) for i in ring])
            Y = rr.GetConformer().GetPositions()
            item["relaxed_xyz"] = [[r3(v) for v in p] for p in Y]
            item["relaxed_max_abs_dev"] = r3(np.abs(rf.signed_plane_dist(Y[ring])[0]).max())
        rings.append(item)
    sel = rf.pocket_residues(pocket_pdb, X) if pocket_pdb else "none"
    return target, k, dict(molblock=molblock, pocket_sel=sel, smiles=Chem.MolToSmiles(m), rings=rings), recs


def hist(values):
    c, _ = np.histogram(np.clip(values, 0, HIST_BINS[-1] - 1e-9), HIST_BINS)
    return c.tolist()


def build_arm(arm, root, out, procs):
    print(f"[{arm}] {root}")
    adir = out / "data" / arm
    (adir / "mols").mkdir(parents=True, exist_ok=True)
    (adir / "pockets").mkdir(parents=True, exist_ok=True)

    jobs, meta, ref_rings, skipped = [], {}, [], []
    for t in rf.P79:
        tdir = root / t
        if not (tdir / "metrics.json").exists():
            skipped.append(f"{t}: no metrics.json"); continue
        try:
            mols, rows = rf.load_target(tdir)
        except ValueError as e:
            skipped.append(str(e)); continue
        pocket = next(tdir.glob("*_pocket10.pdb"), None)
        if pocket is not None:
            shutil.copyfile(pocket, adir / "pockets" / f"{t}.pdb")
        for f in tdir.glob("*.sdf"):
            if f.name != "samples.sdf":
                ref = next(iter(Chem.SDMolSupplier(str(f), sanitize=True)), None)
                if ref is not None:
                    ref_rings += rf.ring_records(ref)
        for k, (m, s) in enumerate(zip(mols, rows)):
            pb = ((s.get("posebusters") or {}).get("checks") or {}).get("non-aromatic_ring_non-flatness")
            meta[(t, k)] = dict(pb=pb, n_atoms=m.GetNumAtoms())
            jobs.append((t, k, Chem.MolToMolBlock(m), str(pocket) if pocket else None))
    for s in skipped:
        print("  skip:", s)

    rows, mol_status = [], []
    with Pool(procs) as pool:
        for i, (t, k, payload, recs) in enumerate(pool.imap_unordered(molecule_payload, jobs, chunksize=8), 1):
            json.dump(payload, open(adir / "mols" / f"{t}_{k}.json", "w"), separators=(",", ":"))
            st = {r["status"] for r in recs}
            mol_status.append((meta[(t, k)]["pb"], "fail" if "fail" in st else
                               "sign-dependent" if "sign-dependent" in st else "pass"))
            for j, (rec, geo) in enumerate(zip(recs, payload["rings"])):
                rows.append(dict(target=t, k=k, ring=j, size=rec["size"], name=rec["name"], core=rec["core"],
                                 n_sp2=rec["n_sp2"], max_abs_dev=r3(rec["max_abs_dev"]),
                                 pb_value_lo=r3(rec["pb_value_lo"]), status=rec["status"],
                                 pb_metrics=meta[(t, k)]["pb"], relaxed_max_abs_dev=geo["relaxed_max_abs_dev"],
                                 n_atoms=meta[(t, k)]["n_atoms"]))
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)} molecules")
    rows.sort(key=lambda r: (r["target"], r["k"], r["ring"]))

    # ── summary: the notebook's tables, precomputed ──
    def group(sel, key="name"):
        g = {}
        for r in rows:
            if sel(r):
                g.setdefault(r[key], []).append(r)
        return g

    by_name = []
    for name, rs in group(lambda r: r["status"] != "not checked").items():
        fail = sum(r["status"] == "fail" for r in rs)
        sd = sum(r["status"] == "sign-dependent" for r in rs)
        bad = [r for r in rs if r["status"] != "pass" and r["relaxed_max_abs_dev"] is not None]
        by_name.append(dict(name=name, size=rs[0]["size"], n_rings=len(rs), fail=fail, sign_dep=sd,
                            fail_rate=(fail + sd) / len(rs),
                            all_sp3=float(np.mean([r["n_sp2"] == 0 for r in rs])),
                            median_max_abs_dev=r3(np.median([r["max_abs_dev"] for r in rs])),
                            ff_also_flat=(float(np.mean([r["relaxed_max_abs_dev"] < rf.PB_THR for r in bad]))
                                          if bad else None)))
    by_name.sort(key=lambda d: (-d["fail"], -d["n_rings"]))

    five = []
    for name, rs in group(lambda r: r["size"] == 5).items():
        sp3 = [r["max_abs_dev"] for r in rs if r["n_sp2"] == 0]
        five.append(dict(name=name, n_rings=len(rs),
                         median_max_abs_dev=r3(np.median([r["max_abs_dev"] for r in rs])),
                         would_fail=float(np.mean([r["max_abs_dev"] < rf.PB_THR for r in rs])),
                         n_all_sp3=len(sp3),
                         would_fail_all_sp3=float(np.mean(np.array(sp3) < rf.PB_THR)) if sp3 else None,
                         median_all_sp3=r3(np.median(sp3)) if sp3 else None,
                         relaxed_median=r3(np.nanmedian([r["relaxed_max_abs_dev"] or np.nan for r in rs]))))
    five.sort(key=lambda d: -d["n_rings"])

    crosstab = {}
    for pb, ours in mol_status:
        crosstab.setdefault(str(pb), {}).setdefault(ours, 0)
        crosstab[str(pb)][ours] += 1
    bad = [r for r in rows if r["status"] in ("fail", "sign-dependent") and r["relaxed_max_abs_dev"] is not None]
    sp3_6 = [r for r in rows if r["status"] != "not checked" and r["n_sp2"] == 0]
    summary = dict(
        arm=arm, root=str(root), n_mols=len(mol_status), n_rings=len(rows),
        n_pockets=len({t for t, _ in meta}), skipped=skipped,
        fail_rate=float(np.mean([pb is False for pb, _ in mol_status])),
        crosstab=crosstab, by_name=by_name, five=five,
        ff_also_flat=float(np.mean([r["relaxed_max_abs_dev"] < rf.PB_THR for r in bad])) if bad else None,
        n_failing_rings=len(bad),
        all_sp3_6=dict(checked=len(sp3_6), failing=sum(r["status"] != "pass" for r in sp3_6)),
        hist=dict(bins=HIST_BINS.tolist(), families=[
            dict(label=lab,
                 gen=hist([r["max_abs_dev"] for r in rows if sel(r)]),
                 ref=hist([r["max_abs_dev"] for r in ref_rings if sel(r)]),
                 n_gen=sum(1 for r in rows if sel(r)), n_ref=sum(1 for r in ref_rings if sel(r)))
            for lab, sel in FAMILIES.items()]),
        threshold=rf.PB_THR,
    )
    index = dict(summary=summary, keys=ROW_KEYS, rows=[[r[k] for k in ROW_KEYS] for r in rows])
    json.dump(index, open(adir / "index.json", "w"), separators=(",", ":"))
    print(f"  {len(mol_status)} molecules, {len(rows)} rings, metrics fail rate {summary['fail_rate']:.1%}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(rf.ARMS), choices=list(rf.ARMS))
    ap.add_argument("--out", type=Path, default=rf.REPO / ".cache" / "ring-nonflatness-site")
    # The container is capped at 32 cores whatever nproc says; leave room for training.
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ngl = out / "ngl.js"
    if not ngl.exists():
        print("downloading", NGL_URL)
        urllib.request.urlretrieve(NGL_URL, ngl)
    shutil.copyfile(HERE / "ring_nonflatness_site.html", out / "index.html")

    built = [a for a in args.arms if build_arm(a, rf.ARMS[a], out, args.procs)]
    # Keep arms built by an earlier run, so `--arms VoxBind` alone does not hide CoDE.
    old = out / "data" / "arms.json"
    prev = json.load(open(old)) if old.exists() else []
    arms = [a for a in rf.ARMS if a in set(built) | {p for p in prev if (out / "data" / p / "index.json").exists()}]
    json.dump(arms, open(old, "w"))
    print("wrote", out, "arms:", arms)


if __name__ == "__main__":
    main()

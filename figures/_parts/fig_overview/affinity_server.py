"""Select high/low affinity examples from the same data used in the overview scatter."""
import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np
from rdkit import Chem, rdBase

import pocket_server as pocket

HERE, ROOT = pocket.HERE, pocket.ROOT
SOURCE = ROOT / "base/_casf/CDG_v2_casf2016_preds_seed0.csv"
INDEX = ROOT / "dataset/data/pdbbind/index.csv"
OUT = pocket.OUT
pocket.EXPORTS = OUT / "exports/affinity"
SCATTER_LOCK = threading.Lock()
SCATTER_STEM = "binding_affinity_scatter_selected"


def palette():
    with (ROOT / "figures/fig-palette/palette-methods.csv").open() as handle:
        return [dict(label=row["method"], color=row.get("color_hex") or row.get("palette_hex"))
                for row in csv.DictReader(handle) if not row.get("hatch")]


def last_exported_cases():
    result = {}
    for slot in ("high", "low"):
        paths = sorted(pocket.EXPORTS.glob(f"affinity-{slot}-*.json"), key=lambda p:p.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                pid = json.loads(path.read_text()).get("target")
                if pid in RECORDS:
                    result[slot] = pid
                    break
            except (ValueError, OSError):
                continue
    return result


def save_selected_scatter(body):
    selected = body.get("selected")
    if not isinstance(selected, list) or len(selected) != 2:
        raise ValueError("Select two cases with their colors")
    for item in selected:
        if not isinstance(item, dict) or item.get("pid") not in RECORDS:
            raise ValueError("Unknown PDB ID")
        if not isinstance(item.get("color"), str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", item["color"]):
            raise ValueError("Use a six-digit hex color")
    if selected[0]["pid"] == selected[1]["pid"]:
        raise ValueError("Choose different PDB IDs for the two panels")
    with SCATTER_LOCK:
        from binding_affinity import render_plot
        with tempfile.TemporaryDirectory(prefix="affinity-scatter-", dir=OUT) as temp:
            metadata = render_plot(selected, temp)
            metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            (Path(temp) / f"{SCATTER_STEM}.json").write_text(json.dumps(metadata, indent=2))
            for ext in ("png", "pdf", "json"):
                (Path(temp) / f"{SCATTER_STEM}.{ext}").replace(OUT / f"{SCATTER_STEM}.{ext}")
    return dict(**metadata, files=[dict(format=ext, url=f"/scatter/{SCATTER_STEM}.{ext}",
                                       path=str((OUT / f"{SCATTER_STEM}.{ext}").relative_to(ROOT)))
                                  for ext in ("png", "pdf")])


@lru_cache(maxsize=32)
def crystal_ligand(path):
    # Retain deposited bond types: rendering does not require chemical sanitization.
    with rdBase.BlockLogs():
        mol = next(iter(Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)), None)
    if mol is None or not mol.GetNumConformers() or not mol.GetNumHeavyAtoms():
        raise ValueError("Missing crystal ligand coordinates")
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError("Nonfinite ligand coordinates")
    edit = Chem.RWMol(mol)
    for i in reversed([a.GetIdx() for a in edit.GetAtoms() if a.GetAtomicNum() == 1]):
        edit.RemoveAtom(i)
    return edit.GetMol()


def build_catalog():
    metadata = {r["pdb_id"]:r for r in csv.DictReader(INDEX.open())}
    records, unavailable = {}, []
    for row in csv.DictReader(SOURCE.open()):
        pid = row["pid"]
        folder = ROOT / "dataset/data/pdbbind/structures/pbpp-2020" / pid
        ligand, protein = folder / f"{pid}_ligand.sdf", folder / f"{pid}_protein.pdb"
        if not protein.is_file():
            protein = folder / f"{pid}_pocket.pdb"
        try:
            mol = crystal_ligand(ligand)
            if not protein.is_file():
                raise ValueError("Missing receptor")
        except (ValueError, OSError):
            unavailable.append(pid)
            continue
        gt, pred = float(row["y"]), float(row["pred"])
        if not np.isfinite([gt, pred]).all():
            unavailable.append(pid)
            continue
        info = metadata.get(pid, {})
        label_consistent = abs(float(info.get("pK") or gt) - gt) < .02
        records[pid] = dict(pid=pid, gt=gt, predicted=pred, error=abs(pred-gt),
                            atoms=mol.GetNumHeavyAtoms(), protein_class=info.get("type", ""),
                            measurement=info.get("kd/ki", "") if label_consistent else "",
                            resolution=info.get("resolution", ""), covalent=info.get("covalent", ""),
                            source=dict(predictions=str(SOURCE.relative_to(ROOT)),
                                        ligand=str(ligand.relative_to(ROOT)), protein=str(protein.relative_to(ROOT))))
    return records, unavailable


RECORDS, UNAVAILABLE = build_catalog()


@lru_cache(maxsize=24)
def affinity_scene(pid, radius):
    if pid not in RECORDS or radius not in (5, 7, 10):
        raise ValueError("Unknown PDB ID or pocket radius")
    record = RECORDS[pid]
    mol = crystal_ligand(ROOT / record["source"]["ligand"])
    xyz = mol.GetConformer().GetPositions()
    center = xyz.mean(axis=0)
    pdb, coords = pocket.centered_pocket(ROOT / record["source"]["protein"], center, xyz, radius)
    centered = Chem.Mol(mol)
    for i, point in enumerate(xyz - center):
        centered.GetConformer().SetAtomPosition(i, point)
    sdf = Chem.MolToMolBlock(centered, kekulize=False) + "\n$$$$\n"
    return dict(task="affinity", target=pid, pocket_pdb=pdb, reference_sdf=sdf,
                direction=pocket.open_direction(coords), extent=float(max(np.abs(coords).max(), np.abs(xyz-center).max())+2),
                pocket_atoms=len(coords), reference_atoms=mol.GetNumHeavyAtoms(), samples=[],
                center=center.tolist(), source=record["source"], affinity=record)


class Handler(pocket.Handler):
    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                shared = (HERE / "pocket.html").read_text()
                html = (HERE / "affinity.html").read_text()
                html = html.replace("{{POCKET_STYLES}}", shared.split("<style>",1)[1].split("</style>",1)[0])
                template = '<template id="panel-template">' + shared.split('<template id="panel-template">',1)[1].split("</template>",1)[0] + "</template>"
                return self.send(html.replace("{{PANEL_TEMPLATE}}", template).encode(), "text/html; charset=utf-8")
            if url.path == "/affinity.js":
                return self.send((HERE / "affinity.js").read_bytes(), "application/javascript")
            if url.path == "/api/catalog":
                return self.send(dict(records=list(RECORDS.values()), unavailable=UNAVAILABLE,
                                      source=str(SOURCE.relative_to(ROOT)), model="CDG v2 · seed 0 · CASF-2016", palette=palette(), defaults=last_exported_cases()))
            if url.path.startswith("/scatter/"):
                name = url.path.removeprefix("/scatter/")
                types = {"png":"image/png", "svg":"image/svg+xml", "pdf":"application/pdf", "json":"application/json"}
                for ext, kind in types.items():
                    if name == f"{SCATTER_STEM}.{ext}" and (OUT/name).is_file():
                        return self.send((OUT/name).read_bytes(), kind)
                return self.send({"error":"Scatter plot not found"}, status=404)
            if url.path == "/api/scene":
                q = parse_qs(url.query)
                return self.send(affinity_scene(q.get("target", [""])[0], int(q.get("radius", [7])[0])))
            if url.path.startswith("/exports/") and url.path.endswith(".json"):
                name = url.path.split("/")[-1]
                if re.fullmatch(r"[A-Za-z0-9_-]+\.json", name) and (pocket.EXPORTS/name).is_file():
                    return self.send((pocket.EXPORTS/name).read_bytes())
            return super().do_GET()
        except (ValueError, KeyError, OSError) as e:
            self.send({"error":str(e)}, status=400)

    def do_POST(self):
        if self.path not in ("/api/save", "/api/selection", "/api/scatter"):
            return self.send({"error":"Not found"}, status=404)
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.send({"error":"Origin mismatch"}, status=403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 24 * 1024 * 1024 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON payload, up to 24 MB")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected an object")
            if self.path == "/api/scatter":
                return self.send(save_selected_scatter(body))
            if self.path == "/api/save":
                scene = body.get("scene", {})
                if not isinstance(scene, dict) or scene.get("target") not in RECORDS:
                    raise ValueError("Unknown capture target")
                scene["affinity"] = RECORDS[scene["target"]]
                return self.send(pocket.save_capture(body))
            pairs = body.get("panels", [])
            shortlist = body.get("shortlist", [])
            if not isinstance(pairs, list) or len(pairs) != 2 or not isinstance(shortlist, list) or len(shortlist) > len(RECORDS):
                raise ValueError("Expected two panels and a shortlist")
            for panel in pairs:
                if not isinstance(panel, dict) or panel.get("pid") not in RECORDS:
                    raise ValueError("Unknown selection")
                panel["record"] = RECORDS[panel["pid"]]
            if any(not isinstance(pid, str) or pid not in RECORDS for pid in shortlist):
                raise ValueError("Unknown shortlist entry")
            saved = dict(panels=pairs, shortlist=[RECORDS[p] for p in shortlist], source=str(SOURCE.relative_to(ROOT)))
            pocket.EXPORTS.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            path = pocket.EXPORTS / f"selection-{stamp}-{uuid4().hex[:6]}.json"
            path.write_text(json.dumps(saved, indent=2))
            self.send(dict(url=f"/exports/{path.name}", path=str(path.relative_to(ROOT))))
        except (ValueError, KeyError, OSError) as e:
            self.send({"error":str(e)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    server = pocket.ThreadingHTTPServer((args.host,args.port), Handler)
    print(f"Affinity example selector: http://{args.host}:{args.port} ({len(RECORDS)} cases)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

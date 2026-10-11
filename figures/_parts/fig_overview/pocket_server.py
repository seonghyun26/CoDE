"""Local downstream pocket figure editor; run with run_pocket.sh."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np
from PIL import Image
from rdkit import Chem, rdBase
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "figures" / "fig-overview"     # the code moved to _parts/; its outputs did not
EXPORTS = OUT / "exports" / "downstream"
NGL = HERE / "assets" / "ngl.js"
EXPORT_DPI = 300


def catalog():
    result = {"ligand": {}, "mcp": {}}
    root = ROOT / "results/task2-drugdesign/VoxBind-Ours/samples"
    for folder in sorted(root.glob("target_*")):
        pockets = list(folder.glob("*_pocket10.pdb"))
        refs = [p for p in folder.glob("*.sdf") if p.name != "samples.sdf"]
        generated = folder / "samples.sdf"
        if len(pockets) == len(refs) == 1 and generated.is_file():
            result["ligand"][folder.name] = dict(pocket=pockets[0], reference=refs[0], generated=generated)
    root = ROOT / "results/task3-mcp/Ours-receptorED/samples"
    for folder in sorted(root.glob("mcpp_*")):
        match = re.match(r"mcpp_([a-zA-Z0-9]{4})_", folder.name)
        if not match:
            continue
        pid = match[1].lower()
        refdir = ROOT / "dataset/data/pdbbind/structures/pbpp-2020" / pid
        pooled = sorted(folder.glob("pooled_*.sdf"), key=lambda p: int(p.stem.split("_")[-1]))
        record = dict(pocket=refdir / f"{pid}_pocket.pdb", reference=refdir / f"{pid}_ligand.sdf")
        if pooled and all(p.is_file() for p in record.values()):
            result["mcp"][folder.name] = dict(**record, generated=pooled[-1])
    return result


CATALOG = catalog()


@lru_cache(maxsize=8)
def molecules(path):
    valid = []
    with rdBase.BlockLogs():
        for index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=True)):
            if mol is not None and mol.GetNumConformers() and mol.GetNumHeavyAtoms():
                if np.isfinite(mol.GetConformer().GetPositions()).all():
                    valid.append((index, mol))
    if not valid:
        raise ValueError("No readable 3D molecules in this sample")
    return valid


def centered_mol(mol, center):
    copy = Chem.Mol(mol)
    conf = copy.GetConformer()
    for i, xyz in enumerate(conf.GetPositions() - center):
        conf.SetAtomPosition(i, xyz)
    return Chem.MolToMolBlock(copy) + "\n$$$$\n"


def centered_pocket(path, center, ligand_xyz, radius):
    """Select whole residues near the reference, keeping a common coordinate frame."""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[16:17] in (" ", "A"):
            if line[76:78].strip().upper() in ("H", "D") or line[17:20] in ("HOH", "WAT"):
                continue
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            rows.append((line, xyz, line[21:27]))
    keep = {key for _, xyz, key in rows if np.min(np.linalg.norm(ligand_xyz - xyz, axis=1)) <= radius}
    lines, coords = [], []
    for line, xyz, key in rows:
        if key not in keep:
            continue
        xyz = xyz - center
        lines.append(line[:30] + "".join(f"{x:8.3f}" for x in xyz) + line[54:])
        coords.append(xyz)
    if not coords:
        raise ValueError("Pocket crop contains no atoms")
    return "\n".join(lines) + "\nEND\n", np.asarray(coords)


def open_direction(coords):
    """Choose a camera direction with few pocket atoms projected over the ligand."""
    i = np.arange(256)
    z = 1 - 2 * (i + .5) / len(i)
    angle = i * np.pi * (3 - np.sqrt(5))
    directions = np.column_stack((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z))
    depth = coords @ directions.T
    perpendicular_sq = np.maximum(0, np.sum(coords*coords, axis=1)[:, None] - depth*depth)
    scores = (np.exp(-perpendicular_sq / 12) * (depth > 0)).sum(axis=0)
    return directions[np.argmin(scores)].tolist()


@lru_cache(maxsize=16)
def scene(task, target, radius):
    if task not in CATALOG or target not in CATALOG[task]:
        raise ValueError("Unknown task or target")
    if radius not in (5, 7, 10):
        raise ValueError("Pocket radius must be 5, 7 or 10 angstrom")
    record = CATALOG[task][target]
    ref = molecules(record["reference"])[0][1]
    xyz = ref.GetConformer().GetPositions()
    center = xyz.mean(axis=0)
    pocket, coords = centered_pocket(record["pocket"], center, xyz, radius)
    generated = molecules(record["generated"])
    return dict(task=task, target=target, pocket_pdb=pocket,
                reference_sdf=centered_mol(ref, center), direction=open_direction(coords),
                extent=float(max(np.abs(coords).max(), np.abs(xyz-center).max()) + 2),
                pocket_atoms=len(coords), reference_atoms=ref.GetNumHeavyAtoms(),
                samples=[dict(index=index, atoms=mol.GetNumHeavyAtoms()) for index, mol in generated],
                center=center.tolist(), source={k:str(v.relative_to(ROOT)) for k,v in record.items()})


def save_capture(body):
    name = body.get("name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,90}", name):
        raise ValueError("Invalid export name")
    data = body.get("image", "")
    if not isinstance(data, str) or not data.startswith("data:image/png;base64,"):
        raise ValueError("Expected a PNG image")
    raw = base64.b64decode(data.split(",", 1)[1], validate=True)
    with Image.open(BytesIO(raw)) as im:
        if im.format != "PNG" or max(im.size) > 4096:
            raise ValueError("PNG must be at most 4096 pixels per side")
        im.verify()
    # Embed the exact WebGL capture with its alpha channel, without a white fill.
    with Image.open(BytesIO(raw)) as im:
        width, height = (value * 72 / EXPORT_DPI for value in im.size)
        png_bytes = BytesIO()
        im.save(png_bytes, format="PNG", dpi=(EXPORT_DPI, EXPORT_DPI))
        raw = png_bytes.getvalue()
    pdf_bytes = BytesIO()
    pdf = Canvas(pdf_bytes, pagesize=(width, height), pageCompression=1)
    pdf.setTitle(name)
    pdf.drawImage(ImageReader(BytesIO(raw)), 0, 0, width=width, height=height, mask="auto")
    pdf.showPage()
    pdf.save()
    EXPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stem = f"{name}-{stamp}-{uuid4().hex[:6]}"
    (EXPORTS / f"{stem}.png").write_bytes(raw)
    (EXPORTS / f"{stem}.pdf").write_bytes(pdf_bytes.getvalue())
    (EXPORTS / f"{stem}.json").write_text(json.dumps(body.get("scene", {}), indent=2))
    files = [dict(format=ext, url=f"/exports/{stem}.{ext}",
                  path=str((EXPORTS / f"{stem}.{ext}").relative_to(ROOT))) for ext in ("png", "pdf")]
    return dict(url=files[0]["url"], path=files[0]["path"], files=files)


class Handler(BaseHTTPRequestHandler):
    def send(self, body, kind="application/json", status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            assets = {"/": (HERE / "pocket.html", "text/html; charset=utf-8"),
                      "/pocket.js": (HERE / "pocket.js", "application/javascript"),
                      "/ngl.js": (NGL, "application/javascript")}
            if url.path in assets:
                path, kind = assets[url.path]
                return self.send(path.read_bytes(), kind)
            if url.path == "/api/catalog":
                return self.send({task:list(targets) for task, targets in CATALOG.items()})
            if url.path in ("/api/scene", "/api/molecule"):
                task, target = q.get("task", [""])[0], q.get("target", [""])[0]
                data = scene(task, target, int(q.get("radius", [7])[0]))
                if url.path == "/api/scene":
                    return self.send(data)
                index = int(q.get("index", [-1])[0])
                mol = dict(molecules(CATALOG[task][target]["generated"])).get(index)
                if mol is None:
                    raise ValueError("Unknown generated sample index")
                return self.send(dict(sdf=centered_mol(mol, np.asarray(data["center"]))))
            if url.path.startswith("/exports/"):
                name = url.path.split("/")[-1]
                if re.fullmatch(r"[A-Za-z0-9_-]+\.(png|pdf)", name) and (EXPORTS/name).is_file():
                    return self.send((EXPORTS/name).read_bytes(), "application/pdf" if name.endswith(".pdf") else "image/png")
            if url.path == "/favicon.ico":
                return self.send(b"", "image/x-icon", 204)
            self.send({"error":"Not found"}, status=404)
        except (ValueError, KeyError, OSError) as e:
            self.send({"error":str(e)}, status=400)

    def do_POST(self):
        if self.path != "/api/save":
            return self.send({"error":"Not found"}, status=404)
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.send({"error":"Origin mismatch"}, status=403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 24 * 1024 * 1024:
                raise ValueError("Invalid capture size")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected JSON")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a capture object")
            self.send(save_capture(body))
        except (ValueError, OSError) as e:
            self.send({"error":str(e)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8768, type=int)
    args = parser.parse_args()
    if not all(CATALOG.values()):
        raise SystemExit("Both ligand and MCP sample datasets are required")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Pocket visualizer: http://{args.host}:{args.port}", flush=True)
    print(f"Targets: { {k:len(v) for k,v in CATALOG.items()} }; exports: {EXPORTS}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

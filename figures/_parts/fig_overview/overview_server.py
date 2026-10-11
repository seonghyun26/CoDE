"""Local, standalone overview viewer. Launch with run_overview.sh."""

import argparse
import base64
from datetime import datetime, timezone
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse
import uuid

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas
from plotly.offline import get_plotlyjs

from overview_scene import HERE, REPO_ROOT, DEFAULT_SAMPLE, SAMPLES, load_scene
from overview_ngl import payload as ngl_payload

OUTPUT_DIR = REPO_ROOT / "figures" / "fig-overview" / "exports"
CAPTURE_VIEWER = "pre-training"
LEGACY_CATEGORIES = ("overview", "ligand", "protein", "density", "gradient", "masking")
ASSETS = HERE / "assets"
MAX_BODY = 32 * 1024 * 1024


class SceneStore:
    """Bound memory while allowing independent browser sample selections."""
    def __init__(self):
        self.lock = threading.RLock()
        self.load = lru_cache(maxsize=2)(load_scene)
        self.figure = lru_cache(maxsize=6)(self._figure)
        self.density = lru_cache(maxsize=12)(self._density)
        self.ngl = lru_cache(maxsize=8)(self._ngl)

    def _ngl(self, sample, view, iso, proximity=True, expanded=False, density_style="mesh"):
        with self.lock:
            return ngl_payload(self.load(sample), view, iso, proximity, expanded, density_style)

    def config(self, sample):
        with self.lock:
            return self.load(sample).config()

    def _figure(self, sample, view):
        with self.lock:
            return self.load(sample).figure_json(view)

    def _density(self, sample, view, iso):
        with self.lock:
            scene = self.load(sample)
            if view == "gradient" or not 0.05 <= iso <= scene.ns["density_ceiling"]:
                raise ValueError("Invalid density threshold.")
            return scene.density_json(view, iso)


def save_capture(request, output_dir=OUTPUT_DIR):
    # This viewer always saves together, including requests from older pages.
    category = CAPTURE_VIEWER
    name = request.get("name", "overview")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Use 1–80 letters, numbers, hyphens, or underscores for the file name.")
    fmt = request.get("format", "png")
    if fmt not in ("png", "pdf", "both"):
        raise ValueError("Unsupported image format.")
    encoded = request.get("image", "")
    if not isinstance(encoded, str) or not encoded.startswith("data:image/png;base64,"):
        raise ValueError("A PNG capture is required.")
    raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid PNG image.")
    with Image.open(BytesIO(raw)) as check:
        if check.width > 4096 or check.height > 4096:
            raise ValueError("Image dimensions exceed 4096 pixels.")
        check.verify()
    with Image.open(BytesIO(raw)) as capture:
        capture.load()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        stem = f"{name}-{stamp}-{uuid.uuid4().hex[:8]}"
        output_dir = output_dir / category
        output_dir.mkdir(parents=True, exist_ok=True)
        files = []
        if fmt in ("png", "both"):
            target = output_dir / f"{stem}.png"
            target.write_bytes(raw)
            files.append(f"{category}/{target.name}")
        if fmt in ("pdf", "both"):
            # Embed RGBA with a PDF soft mask; do not flatten onto white.
            # Keep the same 144 dpi page dimensions as the PNG capture.
            rgba = capture.convert("RGBA")
            target = output_dir / f"{stem}.pdf"
            width, height = rgba.width / 2, rgba.height / 2
            pdf = pdf_canvas.Canvas(str(target), pagesize=(width, height), pdfVersion=(1, 4))
            pdf.drawImage(ImageReader(rgba), 0, 0, width=width, height=height, mask="auto")
            pdf.showPage()
            pdf.save()
            files.append(f"{category}/{target.name}")
    return files


class Handler(BaseHTTPRequestHandler):
    def send(self, body, content_type="application/json", status=200, compress=False):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        zipped = compress and "gzip" in self.headers.get("Accept-Encoding", "")
        if zipped:
            body = gzip.compress(body, compresslevel=3)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if zipped:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                return self.send((ASSETS / "overview.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/assets/overview.js":
                return self.send((ASSETS / "overview.js").read_bytes(), "application/javascript")
            if url.path in ("/assets/overview_ngl.js", "/assets/overview_clip.js", "/assets/ngl.js"):
                return self.send((ASSETS / url.path.rsplit("/", 1)[1]).read_bytes(), "application/javascript", compress=True)
            if url.path == "/assets/plotly.js":
                return self.send(self.server.plotly_js, "application/javascript", compress=True)
            if url.path == "/api/config":
                sample = query.get("sample", [DEFAULT_SAMPLE])[0]
                return self.send(self.server.scenes.config(sample))
            if url.path == "/api/ngl":
                sample = query.get("sample", [DEFAULT_SAMPLE])[0]
                view = query.get("view", ["overview"])[0]
                iso = float(query["iso"][0]) if "iso" in query else None
                proximity = query.get("proximity", ["1"])[0]
                if proximity not in ("0", "1"):
                    raise ValueError("Invalid proximity setting.")
                expanded = query.get("expanded", ["0"])[0]
                if expanded not in ("0", "1"):
                    raise ValueError("Invalid extent setting.")
                density_style = query.get("density_style", ["mesh"])[0]
                return self.send(self.server.scenes.ngl(sample, view, iso, proximity == "1", expanded == "1", density_style), compress=True)
            if url.path in ("/api/figure", "/api/density"):
                sample = query.get("sample", [DEFAULT_SAMPLE])[0]
                view = query.get("view", ["overview"])[0]
                if sample not in SAMPLES or view not in ("overview", "gradient", "masked"):
                    raise ValueError("Unknown view.")
                if url.path == "/api/figure":
                    return self.send(self.server.scenes.figure(sample, view), compress=True)
                iso = float(query.get("iso", ["0.1"])[0])
                return self.send(self.server.scenes.density(sample, view, round(iso, 3)), compress=True)
            if url.path.startswith("/captures/"):
                name = url.path.removeprefix("/captures/")
                parts = name.split("/")
                if len(parts) not in (1, 2) or not re.fullmatch(r"[A-Za-z0-9_-]+\.(png|pdf)", parts[-1]):
                    raise ValueError("Invalid image path.")
                if len(parts) == 2 and parts[0] not in (CAPTURE_VIEWER, *LEGACY_CATEGORIES):
                    raise ValueError("Invalid capture category.")
                # Resolve both old basename links and former layer-folder links.
                candidates = [OUTPUT_DIR / CAPTURE_VIEWER / parts[-1]]
                for path in candidates:
                    if path.is_file():
                        return self.send(path.read_bytes(), "image/png" if path.suffix == ".png" else "application/pdf")
            self.send({"error": "Not found"}, status=404)
        except (ValueError, KeyError) as error:
            self.send({"error": str(error)}, status=400)

    def do_POST(self):
        if self.path != "/api/save":
            return self.send({"error": "Not found"}, status=404)
        # Accept browser writes only from this site's own origin, including
        # localhost port forwarding. No arbitrary filesystem paths are accepted.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.send({"error": "Origin mismatch"}, status=403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("Invalid capture size.")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json.")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("Expected a capture object.")
            files = save_capture(request)
            self.send({"files": [{"path": f"figures/fig-overview/exports/{name}", "url": f"/captures/{name}"}
                                 for name in files]})
        except (ValueError, OSError) as error:
            self.send({"error": str(error)}, status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    print("Preparing overview scenes…", flush=True)
    scenes = SceneStore()
    scenes.config(DEFAULT_SAMPLE)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.scenes = scenes
    server.plotly_js = get_plotlyjs().encode()
    (OUTPUT_DIR / CAPTURE_VIEWER).mkdir(parents=True, exist_ok=True)
    print(f"Overview visualizer: http://{args.host}:{args.port}/", flush=True)
    print(f"Save directory: {OUTPUT_DIR / CAPTURE_VIEWER}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

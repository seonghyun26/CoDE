"""Run with: python -m unittest discover -s figures/_parts/fig_qual -p test_qual_capture.py"""

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from pypdf import PdfReader

import fig_qual_server as server
from qual_capture import COMPARISON_ORDER, comparison_pdf


class CaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.columns = ["Reference", *server.DEFAULT_METHODS]
        cls.files = {}
        cls.panels = []
        cls.metadata = dict(target="target_test", pocket_code="5LIU",
                            energy_key="vina_dock", energy_unit="kcal/mol",
                            updated_at="2026-09-16T00:00:00+00:00", methods={})
        for index, method in enumerate(cls.columns):
            image = Image.new("RGBA", (1600, 1600), (255, 255, 255, 0))
            ImageDraw.Draw(image).ellipse((400, 400, 1200, 1200), fill=(20 * index, 100, 180, 150))
            png = BytesIO()
            image.save(png, format="PNG")
            name = server._capture_stem("5LIU", method) + ".png"
            cls.files[name] = png.getvalue()
            cls.panels.append(dict(method=method, png=base64.b64encode(png.getvalue()).decode()))
            cls.metadata["methods"][method] = dict(status="captured", energy=-9.351 - index,
                                                    files={"png": name})

    def test_template_layout_and_lossless_images(self):
        pdf = PdfReader(BytesIO(comparison_pdf(self.files, self.metadata, image_dpi=None)))
        self.assertEqual(len(pdf.pages), 1)
        page = pdf.pages[0]
        self.assertEqual(list(page.mediabox), [0, 0, 324, 371])
        text = page.extract_text()
        labels = ["Reference 5LIU", *COMPARISON_ORDER[1:-1], "CoDE (Ours)"]
        positions = [text.index(label) for label in labels]
        self.assertEqual(positions, sorted(positions))
        for record in self.metadata["methods"].values():
            self.assertIn(f"{record['energy']:.2f}", text)
        self.assertEqual(len(page.images), 9)
        original = {Image.open(BytesIO(png)).tobytes() for png in self.files.values()}
        embedded = set()
        for image in page.images:
            self.assertEqual(image.image.size, (1600, 1600))
            embedded.add(image.image.convert("RGBA").tobytes())
        self.assertEqual(embedded, original)

    def test_compact_images_preserve_alpha_and_source_captures(self):
        originals = dict(self.files)
        compact = comparison_pdf(self.files, self.metadata)
        full = comparison_pdf(self.files, self.metadata, image_dpi=None)
        page = PdfReader(BytesIO(compact)).pages[0]
        self.assertEqual(list(page.mediabox), [0, 0, 324, 371])
        self.assertEqual(len(page.images), 9)
        self.assertEqual(page.extract_text(), PdfReader(BytesIO(full)).pages[0].extract_text())
        for image in page.images:
            self.assertEqual(image.image.size, (560, 560))
            self.assertEqual(image.image.mode, "RGBA")
            self.assertEqual(image.image.getpixel((0, 0))[3], 0)
        self.assertEqual(self.files, originals)
        # Flat synthetic shapes can grow after antialiasing; verify resolution
        # here, and measure file-size reduction on actual molecular captures.

    def test_unavailable_and_missing_score(self):
        metadata = deepcopy(self.metadata)
        metadata["methods"]["TargetDiff"] = dict(status="unavailable", reason="Awaiting results")
        metadata["methods"]["Reference"]["energy"] = None
        # An unavailable method must not reuse an older saved image or score.
        pdf = PdfReader(BytesIO(comparison_pdf(self.files, metadata)))
        self.assertEqual(len(pdf.pages[0].images), 8)
        text = pdf.pages[0].extract_text()
        self.assertIn("TargetDiff", text)
        self.assertIn("Awaiting results", text)
        self.assertIn("N/A", text)
        self.assertNotIn(f"{self.metadata['methods']['TargetDiff']['energy']:.2f}", text)

    def test_custom_method_subset(self):
        metadata = deepcopy(self.metadata)
        metadata["methods"] = {m: metadata["methods"][m] for m in ("Reference", "VoxBind + Ours")}
        pdf = PdfReader(BytesIO(comparison_pdf(self.files, metadata)))
        self.assertEqual(len(pdf.pages[0].images), 2)
        self.assertLess(float(pdf.pages[0].mediabox.width), 324)
        self.assertLess(float(pdf.pages[0].mediabox.height), 371)
        self.assertNotIn("TargetDiff", pdf.pages[0].extract_text())

    def _post(self, payload, root):
        # Exercise the real HTTP parser, capture route, PDF writer and file save
        # without loading the external molecular datasets or opening a socket.
        metadata = deepcopy(self.metadata)
        metadata["methods"] = {p["method"]: metadata["methods"][p["method"]] for p in payload["panels"]}
        viewer = SimpleNamespace(
            targets=["target_test"], methods=server.DEFAULT_METHODS,
            catalog=SimpleNamespace(reference=lambda target: {"label": "test · 5LIU"}),
            available_methods=lambda target: server.DEFAULT_METHODS,
            capture_metadata=lambda target, panels: metadata,
        )
        body = json.dumps(payload).encode()
        request = (f"POST /api/capture HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        response = BytesIO()
        connection = SimpleNamespace(makefile=lambda *args: BytesIO(request), sendall=response.write)
        with patch.object(server, "CAPTURE_ROOT", root), patch.object(server.ViewerHandler, "log_message"):
            server.ViewerHandler(connection, ("127.0.0.1", 1234), SimpleNamespace(viewer=viewer))
        headers, data = response.getvalue().split(b"\r\n\r\n", 1)
        self.assertIn(b"200 OK", headers, data)
        return json.loads(data)

    def test_capture_all_and_individual_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._post(dict(target="target_test", panels=self.panels), root)
            self.assertEqual(result["file_count"], 20)  # 9 PNG + 9 PDF + sheet + JSON
            self.assertEqual(result["comparison_pdf"], "drug-qual-5LIU.pdf")
            saved = root / "5LIU"
            self.assertEqual(sorted(p.name for p in saved.iterdir()), result["files"])
            sheet = saved / result["comparison_pdf"]
            original_sheet = sheet.read_bytes()
            for name, png in self.files.items():
                self.assertEqual((saved / name).read_bytes(), png)
                pdf = PdfReader(saved / name.replace(".png", ".pdf"))
                self.assertEqual(list(pdf.pages[0].mediabox), [0, 0, 384, 384])
            result = self._post(dict(target="target_test", method="Reference", panels=[self.panels[0]]), root)
            self.assertEqual(result["file_count"], 3)
            self.assertIsNone(result["comparison_pdf"])
            self.assertEqual(sheet.read_bytes(), original_sheet)
            metadata = json.loads((saved / "Pocket5LIU.json").read_text())
            self.assertEqual(set(metadata["methods"]), set(self.columns))


if __name__ == "__main__":
    unittest.main()

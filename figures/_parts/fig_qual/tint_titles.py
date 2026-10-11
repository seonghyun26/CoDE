import io, json, math, sys, pymupdf as fitz
from pathlib import Path
from PIL import Image
SRC, OUT, MIX = sys.argv[1], sys.argv[2], float(sys.argv[3])
EXTRA = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0   # uniform box = widest title + this
DPI = float(sys.argv[5]) if len(sys.argv) > 5 else None     # downsample images (alpha kept); None = untouched
pal = json.load(open(Path(__file__).resolve().parents[2] / 'fig-palette' / 'method-palette-approved.json'))
COL = {r['method']: r['color'] for r in pal['rows']}
ALIAS = {'Reference': 'Reference ligand', 'CoDE': 'CoDE'}
d = fitz.open(SRC); p = d[0]; H = p.rect.height
cols = [(0, 106), (106, 217), (217, 324)]
groups = {}
for b in p.get_text('dict')['blocks']:
    for l in b.get('lines', []):
        for s in l['spans']:
            if s['size'] < 10 or not s['text'].strip(): continue
            x0, y0, x1, y1 = s['bbox']; c = next(i for i, (a, bb) in enumerate(cols) if a <= x0 < bb)
            key = (c, round(y0 / 50)); g = groups.setdefault(key, [[], fitz.Rect(s['bbox'])])
            g[0].append(s['text']); g[1] |= fitz.Rect(s['bbox'])
ops = []
widest = max(r.width for _, r in groups.values())
for (c, _), (texts, r) in sorted(groups.items()):
    t = ''.join(texts).strip()
    m = 'Reference ligand' if t.startswith('Reference') else 'CoDE' if 'Ours' in t else t
    rgb = [int(COL[m][i:i+2], 16) / 255 for i in (1, 3, 5)]
    rgb = [1 - MIX * (1 - v) for v in rgb]
    padx, pady = 3.0, 0.6
    cx, half = (r.x0 + r.x1) / 2, widest / 2 + padx + EXTRA / 2
    x0, x1 = cx - half, cx + half
    y0, y1 = H - r.y1 - pady + 1.2, H - r.y0 + pady - 1.0   # glyph box has descender/ascender slack
    rad = min(2.5, (y1 - y0) / 2); k = 0.5523 * rad
    path = (f"{x0+rad} {y0} m {x1-rad} {y0} l {x1-rad+k} {y0} {x1} {y0+rad-k} {x1} {y0+rad} c "
            f"{x1} {y1-rad} l {x1} {y1-rad+k} {x1-rad+k} {y1} {x1-rad} {y1} c "
            f"{x0+rad} {y1} l {x0+rad-k} {y1} {x0} {y1-rad+k} {x0} {y1-rad} c "
            f"{x0} {y0+rad} l {x0} {y0+rad-k} {x0+rad-k} {y0} {x0+rad} {y0} c h f")
    ops.append(f"q {rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} rg {path} Q")
    print(m, COL[m], [round(v, 3) for v in r])
x = p.get_contents()[0]; s = d.xref_stream(x); i = s.index(b'h f Q') + len(b'h f Q')
d.update_stream(x, s[:i] + b'\n' + '\n'.join(ops).encode() + b'\n' + s[i:])
if DPI:   # rewrite each image XObject (and its SMask) in place: same names, same placement
    # An xref can be listed under two names (left by an earlier PyMuPDF edit); rewrite it once.
    for xref, smask in dict.fromkeys((info[0], info[1]) for info in p.get_images(full=True)):
        rects = p.get_image_rects(xref)
        if not rects: continue
        box = rects[0]
        px = math.ceil(max(box.width, box.height) * DPI / 72)
        for x, mode, cs in ((xref, 'RGB', '/DeviceRGB'), (smask, 'L', '/DeviceGray')):
            if not x: continue
            im = Image.open(io.BytesIO(fitz.Pixmap(d, x).tobytes('png'))).convert(mode)
            im.thumbnail((px, px), Image.LANCZOS)
            d.update_stream(x, im.tobytes(), compress=True)
            for k, v in (('Width', im.width), ('Height', im.height), ('BitsPerComponent', 8),
                         ('ColorSpace', cs), ('DecodeParms', 'null')):
                d.xref_set_key(x, k, str(v) if k != 'ColorSpace' else v)
d.save(OUT, garbage=3, deflate=True)

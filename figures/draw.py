#!/usr/bin/env python3
"""draw.py — every paper figure, one function each, from one command.

    python figures/draw.py --list
    python figures/draw.py fig-vina-per-pocket-dock posebusters     # ids and/or evaluations
    python figures/draw.py --all                                    # every figure, PNG + PDF
    bash   figures/draw_all.sh                                      # what a full refresh runs

LAYOUT. Each figure is ONE function registered with @figure under an id
`fig-<eval>-<name>`; its files land in `figures/fig-<folder>/` (plus the sub-folder an
evaluation keeps, e.g. `fig-posecheck/clash/`). The code for the figures of one folder is
ONE module in `_parts/`:

    figures/fig-vina/        <-  _parts/fig_vina.py
    figures/fig-uncertainty/ <-  _parts/fig_uncertainty/   (a package: many analyses)

THIS FILE holds only what figures of two or more folders share -- the house style, the
palette, the loaders, the registry, save() / write_csv(), the manuscript download sync
and the command line. Which statement lives where was decided by use, not by topic: a
helper only one folder's figures need lives in that folder's module.

WHAT THIS FILE DOES NOT DO. It draws; it does not re-score. The per-molecule inputs
(`metrics.json` under each run, the docking result JSONs, the PoseCheck/PoseBusters
exports) are produced by task2-ligand_generation/scripts/73_evaluate_samples.sh and the staging scripts,
and the figure-data builders, which are not part of this repository, hold the recompute paths that
made the exported CSV/JSON artifacts in figures/_data/. Those remain the source of truth for DATA; this file is the
source of truth for PICTURES.
"""
from __future__ import annotations

import argparse
import collections
import colorsys
import csv
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-voxbind")
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402
from matplotlib.ticker import MultipleLocator                         # noqa: E402

HERE = Path(__file__).resolve().parent                 # VoxBind/figures
sys.path.insert(0, str(HERE))
# `from draw import ...` in a _parts module must reach THIS module even when it runs as
# __main__ (python figures/draw.py), or the parts would import a second copy of it and
# register their figures into a registry the command line never sees.
sys.modules.setdefault("draw", sys.modules[__name__])

# ════════════════════════════════════════════════════════════════════════════════
# publication axis style — applied at the export boundary (was publication_style.py)
# ════════════════════════════════════════════════════════════════════════════════
# Shared axis proportions approved on the uncertainty combined figure (2026-09-22).
# Applied in save(), so family-specific plotting code cannot override them. Data, scales,
# limits, visibility, legends and annotations are left intact.
import re                                                             # noqa: E402
from itertools import combinations                                    # noqa: E402

from matplotlib.transforms import Bbox                                # noqa: E402

AXIS_WIDTH = 2.64
TICK_SIZE = 17.36
TITLE_SIZE = 19.36
GRID_WIDTH = 1.2
MAJOR_TICK_LENGTH_SCALE = 1.6
MINOR_TICK_LENGTH_SCALE = 1.2
Y_LABEL_MIN_PAD = 12.0  # points beyond the automatically measured tick-label extent


def _categorical(labels):
    words = [t.get_text() for t in labels if t.get_visible() and t.get_text()]
    return bool(words) and any(re.search(r'[A-Za-z]', w) and not w.startswith('$') for w in words)


def _visible_ticks(ax, which):
    axis = getattr(ax, which + 'axis')
    lo, hi = sorted(getattr(ax, 'get_' + which + 'lim')())
    return [t for t in axis.get_major_ticks() if lo <= t.get_loc() <= hi]


def _decorations(ax, renderer):
    artists = [ax.xaxis.label, ax.yaxis.label, ax.title]
    for which in ('x', 'y'):
        axis = getattr(ax, which + 'axis')
        if axis.get_visible():
            artists += [label for tick in _visible_ticks(ax, which)
                        for label in (tick.label1, tick.label2)]
    return [a.get_window_extent(renderer) for a in artists
            if a.get_visible() and a.get_text()]


def apply(fig, *, repair=True):
    """Set absolute print sizes once; expand cramped layouts without dropping labels."""
    if getattr(fig, '_publication_style_applied', False):
        return
    axes = [ax for ax in fig.axes if ax.get_visible() and ax.axison]
    if not axes:
        return
    # Per-figure opt-in scales (set as figure attributes by a family) for axis-title size,
    # tick-label size and axis weight; 1.0 everywhere else, so every other figure is unchanged.
    title_size = TITLE_SIZE * getattr(fig, '_publication_title_scale', 1.0)
    axis_width = AXIS_WIDTH * getattr(fig, '_publication_axis_scale', 1.0)
    tick_size = TICK_SIZE * getattr(fig, '_publication_tick_scale', 1.0)
    for ax in axes:
        ax.xaxis.label.set_fontsize(title_size)
        ax.yaxis.label.set_fontsize(title_size)
        ax.yaxis.labelpad = max(ax.yaxis.labelpad, Y_LABEL_MIN_PAD)
        ax.tick_params(which='both', width=axis_width, labelsize=tick_size)
        # Minor y ticks carry the grid weight, while major ticks retain the axis weight.
        ax.tick_params(axis='y', which='minor', width=GRID_WIDTH)
        # Scale marker length, not edge width. Keep zero-length ticks hidden.
        # The figure-level guard above prevents multiplying again on repeated exports.
        for axis in (ax.xaxis, ax.yaxis):
            for kind, factor in (('major', MAJOR_TICK_LENGTH_SCALE),
                                 ('minor', MINOR_TICK_LENGTH_SCALE)):
                ticks = (axis.get_major_ticks(1) if kind == 'major'
                         else axis.get_minor_ticks(1))
                axis.set_tick_params(which=kind,
                                     length=ticks[0].tick1line.get_markersize() * factor)
        # Category names play the role of x-axis titles in the reference bar chart.
        if _categorical(ax.get_xticklabels()):
            ax.tick_params(axis='x', labelsize=TITLE_SIZE)
        ax.xaxis.get_offset_text().set_fontsize(tick_size)
        ax.yaxis.get_offset_text().set_fontsize(tick_size)
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(axis_width)
        for line in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            line.set_linewidth(GRID_WIDTH)
    for attr in ('_supxlabel', '_supylabel'):
        artist = getattr(fig, attr, None)
        if artist is not None:
            artist.set_fontsize(TITLE_SIZE)
    if repair:
        _repair(fig, axes)
    trim = getattr(fig, '_publication_trim_bottom', None)
    if trim is not None:
        _trim_bottom(fig, trim, getattr(fig, '_publication_trim_keep', None))
    fig._publication_style_applied = True


def _trim_bottom(fig, pad, keep=None):
    """Cut empty canvas below the lowest ink down to `pad` inches (per-figure opt-in).

    `keep` (inches), if given, leaves at least that much canvas below the lowest axes, so
    panels placed side by side can share one bottom margin whatever their own ink needs.
    Only the canvas shrinks: every axes keeps its physical size and its distance from the
    top edge, so type, spines and plot areas are untouched."""
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())   # inches
    cut = tight.y0 - pad
    if keep is not None:
        h = fig.get_figheight()
        cut = min(cut, min(ax.get_position().y0 for ax in fig.axes) * h - keep)
    if cut <= 0:
        return
    w, h = fig.get_size_inches()
    positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
    fig.set_size_inches(w, h - cut)
    for ax, p in positions:
        ax.set_position([p.x0, (p.y0 * h - cut) / (h - cut), p.width, p.height * h / (h - cut)])


def _repair(fig, axes):
    # Keep manual panel positions; increasing the canvas creates room between
    # labels. Do not let tight_layout replace deliberately aligned or inset axes.
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        sx = sy = 1.
        for ax in axes:
            for which in ('x', 'y'):
                ticks = _visible_ticks(ax, which)
                labels = [t.label1 for t in ticks if t.label1.get_visible() and t.label1.get_text()]
                for a, b in zip(labels, labels[1:]):
                    aa, bb = a.get_window_extent(renderer), b.get_window_extent(renderer)
                    if aa.overlaps(bb):
                        if which == 'x':sx = max(sx, 1.12)
                        else:sy = max(sy, 1.12)
        for a, b in combinations(axes, 2):
            ra, rb = a.get_position(), b.get_position()
            # Twins, insets and overlaid panels intentionally occupy the same area.
            horizontal = ra.x1 <= rb.x0 + 1e-5 or rb.x1 <= ra.x0 + 1e-5
            vertical = ra.y1 <= rb.y0 + 1e-5 or rb.y1 <= ra.y0 + 1e-5
            if not (horizontal or vertical):continue
            aa, bb = _decorations(a, renderer), _decorations(b, renderer)
            if any(x.overlaps(y) for x in aa for y in bb):
                if horizontal:sx = max(sx, 1.12)
                if vertical:sy = max(sy, 1.12)
        if sx == sy == 1.:break
        fig.set_size_inches(fig.get_figwidth() * sx, fig.get_figheight() * sy)
    # Prevent outer clipping while preserving the resulting physical plot sizes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    # Matplotlib's layout bbox can suppress the along-axis extent of a long
    # rotated label. Include its full rendered extent to avoid export clipping.
    full_text = [box.transformed(fig.dpi_scale_trans.inverted())
                 for ax in axes for box in _decorations(ax, renderer)]
    tight = Bbox.union([tight, *full_text])
    w, h = fig.get_size_inches()
    extra = (max(0., .04 - tight.x0), max(0., tight.x1 + .04 - w),
             max(0., .04 - tight.y0), max(0., tight.y1 + .04 - h))
    if max(extra) > 0:
        left, right, bottom, top = extra
        positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        fig.set_size_inches(w + left + right, h + bottom + top)
        for ax, p in positions:
            ax.set_position([(p.x0 * w + left) / fig.get_figwidth(),
                             (p.y0 * h + bottom) / fig.get_figheight(),
                             p.width * w / fig.get_figwidth(), p.height * h / fig.get_figheight()])
    fig.canvas.draw()


def layout_issues(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, overlaps = [], []
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible() or not ax.axison:continue
        for artist in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            if not artist.get_visible() or not artist.get_text():continue
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(dict(axis=i, text=artist.get_text()))
        for which in ('x', 'y'):
            if not getattr(ax, which + 'axis').get_visible():continue
            labels = [t.label1 for t in _visible_ticks(ax, which) if t.label1.get_visible() and t.label1.get_text()]
            for a, b in zip(labels, labels[1:]):
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                    overlaps.append(dict(axis=i, direction=which, labels=[a.get_text(), b.get_text()]))
    return dict(outside_canvas=outside, overlapping_adjacent_ticks=overlaps)


def audit(fig):
    """Record the actual exported axis sizes for batch review."""
    rows = []
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:continue
        rows.append(dict(xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
                         xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                         tick_lengths={name + '_' + kind: sorted({t.tick1line.get_markersize() for t in getter()})
                                       for name, axis in (('x', ax.xaxis), ('y', ax.yaxis))
                                       for kind, getter in (('major', axis.get_major_ticks), ('minor', axis.get_minor_ticks))},
                         spines={k: v.get_linewidth() for k, v in ax.spines.items() if v.get_visible()},
                         xlabel_size=ax.xaxis.label.get_fontsize(), ylabel_size=ax.yaxis.label.get_fontsize(),
                         xtick_sizes=sorted({t.get_fontsize() for t in ax.get_xticklabels() if t.get_visible()}),
                         ytick_sizes=sorted({t.get_fontsize() for t in ax.get_yticklabels() if t.get_visible()})))
    return rows


def retained_formats(stem, formats=('png', 'pdf')):
    """An optional batch inventory prevents removed figures from being recreated."""
    inventory = os.environ.get('VOXBIND_FIGURE_RETAINED')
    if not inventory:return tuple(formats)
    allowed = set(json.loads(Path(inventory).read_text()))
    return tuple(ext for ext in formats if str(Path(stem).with_suffix('.' + ext).resolve()) in allowed)


def prepare(fig, stem):
    """Style a retained figure and append machine-readable rendering evidence."""
    if not retained_formats(stem):return False
    apply(fig)
    log = os.environ.get('VOXBIND_STYLE_AUDIT')
    if log:
        with Path(log).open('a') as f:
            f.write(json.dumps(dict(stem=str(Path(stem).resolve()),
                                   size_inches=list(fig.get_size_inches()), axes=audit(fig), **layout_issues(fig))) + '\n')
    return True


prepare_publication_figure = prepare     # the name save() and the uncertainty plots call

# ════════════════════════════════════════════════════════════════════════════════
# manuscript downloads — mirror the figures the paper uses into figures/_download
# ════════════════════════════════════════════════════════════════════════════════
# `python figures/draw.py --inventory <paper.zip>` reads the LaTeX in the uploaded paper
# archive, maps every \includegraphics it reaches to a local output and writes the map to
# figures/manuscript/iclr2027-figures.json; from then on save() copies each mapped figure
# into _download/ under the file name LaTeX uses, the moment it is redrawn.
# `--sync-downloads` copies them all and rewrites the lists. Without a map, nothing happens.
import hashlib                                                        # noqa: E402
import posixpath                                                      # noqa: E402
import shutil                                                         # noqa: E402
import tempfile                                                       # noqa: E402
import zipfile                                                        # noqa: E402
from pathlib import PurePosixPath                                     # noqa: E402

MANIFEST = HERE / 'manuscript/iclr2027-figures.json'
DOWNLOAD = HERE / '_download'
IMAGE_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg'}
# Reviewed aliases: the manuscript names predate the current plotting filenames.
DL_ALIASES = {
    'similarity_novelty.pdf': 'fig-similarity/novelty.pdf',
    'result-drug-posecheck-jsd.pdf': 'fig-posecheck/posecheck-combined.pdf',
}
# The user requested the five standalone panels instead of this composite.
EXCLUDED_DOWNLOADS = {'result-drug-posecheck-jsd.pdf'}

# Additional exports explicitly requested beyond the uploaded LaTeX snapshot.
ADDITIONAL_DOWNLOADS = [
    dict(source='fig-vina/baseline/dock-per-pocket-combined.pdf',
         destination='dock-per-pocket-combined.pdf', download=True),
    dict(source='fig-posecheck/strain-energy/ecdf-by-size-all.pdf',
         destination='ecdf-by-size-all.pdf', download=True),
    dict(source='fig-posecheck/strain-energy/per-atom-all-methods-median.pdf',
         destination='per-atom-all-methods-median.pdf', download=True),
    dict(source='fig-posecheck/clash/clash-box-by-method-horizontal.pdf',
         destination='clash-box-by-method-horizontal.pdf', download=True),
    dict(source='fig-posecheck/redock/median-all.pdf',
         destination='median-all.pdf', download=True),
    dict(source='fig-jsd/metric-summary.pdf',
         destination='metric-summary.pdf', download=True),
]


def download_rows(data):
    return [row for row in [*data['used'], *ADDITIONAL_DOWNLOADS]
            if row['destination'] not in EXCLUDED_DOWNLOADS]


COMMAND = re.compile(r'\\(input|include|includegraphics)\*?\s*(?:\[[^\]]*\]\s*)?\{([^{}]+)\}')


def blank(text):
    return ''.join('\n' if char == '\n' else ' ' for char in text)


def active_tex(text):
    """Keep line numbers while removing comments and literal false branches."""
    lines = []
    for line in text.splitlines(keepends=True):
        for i, char in enumerate(line):
            if char == '%':
                preceding = len(line[:i]) - len(line[:i].rstrip('\\'))
                if preceding % 2 == 0:
                    line = line[:i] + blank(line[i:])
                    break
        lines.append(line)
    text = ''.join(lines)
    text = re.sub(r'\\begin\{comment\}.*?\\end\{comment\}', lambda m: blank(m[0]), text, flags=re.S)
    tokens = re.compile(r'\\(iffalse|iftrue|else|fi)\b')
    parts, stack, start = [], [], 0
    for match in tokens.finditer(text):
        chunk = text[start:match.start()]
        parts.append(chunk if all(stack) else blank(chunk))
        command = match[1]
        if command in ('iftrue', 'iffalse'):
            stack.append(command == 'iftrue')
        elif command == 'else' and stack:
            stack[-1] = not stack[-1]
        elif command == 'fi' and stack:
            stack.pop()
        parts.append(blank(match[0]))
        start = match.end()
    parts.append(text[start:] if all(stack) else blank(text[start:]))
    text = ''.join(parts)
    stop = re.search(r'\\(?:end\{document\}|endinput\b)', text)
    if stop:
        text = text[:stop.start()] + blank(text[stop.start():])
    return text


def scan_archive(archive, main='ICLR2027/0_main.tex'):
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        project = PurePosixPath(main).parts[0]
        texts = {n: z.read(n).decode('utf-8-sig') for n in names
                 if n.startswith(project + '/') and n.endswith('.tex')}
        if main not in texts:
            raise ValueError(f'Main document absent: {main}')
        def resolve(raw, source, extensions):
            for candidate in (raw, posixpath.join(project, raw),
                              posixpath.join(posixpath.dirname(source), raw)):
                for ext in extensions:
                    path = posixpath.normpath(candidate + ext)
                    if path in names:
                        return path
            return None
        reachable, references, missing_inputs = [], [], []
        def walk(source, parents=()):
            if source in parents:
                raise ValueError(f'Cyclic input: {source}')
            if source not in reachable:
                reachable.append(source)
            clean = active_tex(texts[source])
            for match in COMMAND.finditer(clean):
                kind, raw = match.groups()
                extensions = ('', '.pdf', '.png', '.jpg', '.jpeg', '.eps') if kind == 'includegraphics' else ('', '.tex')
                path = resolve(raw.strip(), source, extensions)
                record = dict(path=path, raw=raw.strip(), tex=source,
                              line=clean[:match.start()].count('\n') + 1)
                if kind == 'includegraphics':
                    references.append(record)
                elif path:
                    walk(path, (*parents, source))
                else:
                    missing_inputs.append(record)
        walk(main)
        all_refs = {}
        for source, text in texts.items():
            for match in COMMAND.finditer(active_tex(text)):
                if match[1] != 'includegraphics':
                    continue
                path = resolve(match[2].strip(), source, ('', '.pdf', '.png', '.jpg', '.jpeg', '.eps'))
                if path:
                    all_refs.setdefault(path, []).append(source)
        images = sorted(n for n in names if n.startswith(project + '/')
                        and PurePosixPath(n).suffix.lower() in IMAGE_EXTENSIONS)
        used = sorted({r['path'] for r in references if r['path']})
        unused = []
        for path in images:
            if path in used:
                continue
            elsewhere = sorted(set(all_refs.get(path, [])))
            commented = [source for source, text in texts.items()
                         if PurePosixPath(path).name in text and source not in elsewhere]
            reason = ('Referenced only in .tex files not included by the main document' if elsewhere
                      else 'Only commented/inactive references' if commented else 'No LaTeX reference')
            unused.append(dict(path=path, reason=reason, tex=elsewhere or sorted(commented)))
        return dict(main=main, reachable_tex=reachable, references=references,
                    missing_inputs=missing_inputs, used=used, unused=unused,
                    external_references=[r for r in references if not r['path']])


def build_manifest(archive, main):
    archive = archive.resolve()
    scan = scan_archive(archive, main)
    if scan['missing_inputs']:
        raise ValueError(f"Unresolved inputs: {scan['missing_inputs']}")
    local = [p for p in HERE.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
             and not {'_download', 'manuscript', '.omc', '_parts', '_tools'}.intersection(
                 p.relative_to(HERE).parts)
             and 'motivation-layouts-20260921' not in p.parts and 'proposals' not in p.parts]
    filenames = [PurePosixPath(n).name for n in scan['used']]
    rows = []
    for path in scan['used']:
        name = PurePosixPath(path).name
        matches = [p for p in local if p.name == name]
        if name in DL_ALIASES:
            source = DL_ALIASES[name]
            if not (HERE / source).is_file():
                raise FileNotFoundError(source)
            mapping = 'reviewed filename alias'
        elif len(matches) == 1:
            source = matches[0].relative_to(HERE).as_posix()
            mapping = 'unique exact filename'
        elif len(matches) > 1:
            raise ValueError(f'Ambiguous local image: {path}: {matches}')
        else:
            source, mapping = None, 'external static asset; excluded from download'
        destination = name if filenames.count(name) == 1 else path
        rows.append(dict(latex_path=path, source=source, destination=destination, mapping=mapping, download=bool(source) and destination not in EXCLUDED_DOWNLOADS,
                         references=[r for r in scan['references'] if r['path'] == path]))
    data = dict(archive=os.path.relpath(archive, HERE), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                main=main, reachable_tex=scan['reachable_tex'], used=rows,
                unused=scan['unused'], external_references=scan['external_references'],
                limitations=['Literal input/include/includegraphics traversal; comments, comment environments, iftrue/iffalse and enddocument are handled. No TeX macro execution.'])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    return data


def safe_destination(name):
    path = (DOWNLOAD / name).resolve()
    if not path.is_relative_to(DOWNLOAD.resolve()):
        raise ValueError(f'Invalid download path: {name}')
    return path


def atomic_copy(source, destination, data=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.copy-', dir=destination.parent)
    os.close(fd)
    try:
        if data is not None:
            Path(tmp).write_bytes(data)
        else:
            shutil.copyfile(source, tmp)
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sync_saved_figure(path):
    """Called after a successful export; only selected source paths are mirrored."""
    if not MANIFEST.exists():
        return []
    path = Path(path).resolve()
    manifest = json.loads(MANIFEST.read_text())
    written = []
    for row in download_rows(manifest):
        if row.get('download', bool(row['source'])) and row['source'] and (HERE / row['source']).resolve() == path:
            destination = safe_destination(row['destination'])
            atomic_copy(path, destination)
            written.append(destination)
    return written


def write_lists(data):
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    rows = ['# ICLR 2027 — 사용 중인 그림', '',
            f"`{data['main']}`의 본문·부록에서 실제로 참조하는 이미지 **{len(data['used'])}개**입니다.", '',
            '| LaTeX 파일명 / 다운로드 | 최신 로컬 출력 또는 출처 | 참조 위치 |', '|---|---|---|']
    for row in data['used']:
        source = ('[최신 출력](../' + row['source'] + ')' if row['source'] else '외부에서 제작 — 다운로드 제외')
        if row['destination'] in EXCLUDED_DOWNLOADS:
            source += ' — 요청에 따라 합본 제외; 아래 개별 패널 5개로 제공'
        refs = '<br>'.join(f"`{r['tex']}:{r['line']}`" for r in row['references'])
        label = f"[{row['destination']}]({row['destination']})" if row.get('download', bool(row['source'])) else f"`{row['destination']}`"
        rows.append(f"| {label} | {source} | {refs} |")
    rows += ['', '## 추가 요청한 다운로드', '',
             '업로드된 LaTeX의 참조 목록과 별개로, 요청에 따라 함께 자동 갱신하는 그림입니다.', '']
    for row in ADDITIONAL_DOWNLOADS:
        rows.append(f"- [{row['destination']}]({row['destination']}) — [최신 출력](../{row['source']})")
    if data['external_references']:
        rows += ['', '## ZIP 밖의 참조 / placeholder', '']
        for row in data['external_references']:
            rows.append(f"- `{row['raw']}` — `{row['tex']}:{row['line']}`. ZIP에 포함되지 않아 복사하지 않았습니다.")
    (DOWNLOAD / 'used-figures.md').write_text('\n'.join(rows) + '\n')
    rows = ['# ICLR 2027 — 이미지 폴더의 미사용 파일', '',
            f"파일 **{len(data['unused'])}개**. 현재 main 문서에서 도달하는 참조가 없다는 뜻이며, 삭제하지 않았습니다.",
            '합본에 들어 있는 개별 패널도 직접 참조되지 않으면 이 목록에 포함됩니다.', '',
            '| 이미지 | 구분 | 관련 .tex |', '|---|---|---|']
    reasons = {'Referenced only in .tex files not included by the main document':'main에서 불러오지 않는 .tex에만 참조',
               'Only commented/inactive references':'주석 또는 비활성 참조만 있음', 'No LaTeX reference':'참조 없음'}
    for row in data['unused']:
        rows.append(f"| `{row['path']}` | {reasons[row['reason']]} | {'<br>'.join(row['tex']) or '—'} |")
    (DOWNLOAD / 'unused-images.md').write_text('\n'.join(rows) + '\n')
    mapped = sum(bool(r['source']) and r.get('download', True) for r in data['used'])
    (DOWNLOAD / 'README.md').write_text(f'''# ICLR 2027 figure downloads

- [사용 중인 그림 {len(data['used'])}개](used-figures.md)
- [이미지 폴더의 미사용 파일 {len(data['unused'])}개](unused-images.md)
- 원고: `{data['main']}`; ZIP: `{data['archive']}`

원고 참조 이미지 중 {mapped}개를 최신 로컬 출력에서 복사합니다. 외부 제작 그림과 개별 패널로 대체한 합본 {len(data['used'])-mapped}개는 사용 목록에만 기록하고 다운로드에서 제외합니다. LaTeX에서 사용하는 파일명으로 저장하므로 `ICLR2027/resource/image/`에 그대로 덮어쓸 수 있습니다. 미사용 파일은 복사하거나 삭제하지 않습니다.

추가 요청한 그림 **{len(ADDITIONAL_DOWNLOADS)}개**도 함께 저장합니다: `dock-per-pocket-combined.pdf` 및 PoseCheck/JSD 개별 패널 5개. `result-drug-posecheck-jsd.pdf` 합본은 다운로드하지 않습니다. 자세한 파일명은 [추가 다운로드 목록](used-figures.md#추가-요청한-다운로드)을 참고하세요. 이 설정은 원고 ZIP 목록을 다시 생성해도 유지됩니다.

`draw.py`의 저장 함수가 선택된 PDF를 저장한 직후 이 폴더에도 복사합니다. `draw_all.sh`도 마지막에 모든 선택 파일을 동기화합니다. 로컬 출력을 별도 도구로 수정한 경우 아래 명령을 실행하세요. ZIP의 정적 이미지는 동기화하지 않습니다.

```bash
python figures/draw.py --sync-downloads
```

원고 ZIP의 그림 구성이 바뀌면 목록을 다시 생성하세요.

```bash
python figures/draw.py --inventory figures/Density4Drug.zip
```

매핑 원본: [iclr2027-figures.json](../manuscript/iclr2027-figures.json). 이름이 다른 novelty와 PoseCheck 합본은 명시적 alias로 연결되어 있습니다. `%` 주석과 `\\iffalse`는 사용 목록에서 제외합니다. TeX를 실행하는 분석은 아니므로 새로운 매크로 기반 그림 삽입은 별도 확인이 필요합니다.
''')


def sync_all():
    data = json.loads(MANIFEST.read_text())
    selected = [r for r in download_rows(data) if r['source'] and r.get('download', True)]
    for row in selected:
        atomic_copy(HERE / row['source'], safe_destination(row['destination']))
    write_lists(data)
    print(f"Synced {len(selected)} selected local images to {DOWNLOAD}; {len(data['unused'])} unused images listed.")


# Shared partial-opacity levels for every figure family and standalone plot.
DATA_ALPHA = 0.75
BACKGROUND_ALPHA = 0.15
REPO = HERE.parent                                     # VoxBind
# Everything below is resolved FROM those two, never written out absolutely: this tree gets
# copied between boxes, and a server's home path baked into a constant is the thing that
# breaks first. The two sibling checkouts this needs -- base_drug (TargetDiff's own eval
# tree) and funcbind -- sit NEXT TO the repo, so they are REPO.parent-relative; both take
# an environment override for a box that arranges them differently.
SIBLING = Path(os.environ.get("VOXBIND_SIBLING_ROOT", REPO.parent))
BASEDRUG = Path(os.environ.get("VOXBIND_BASEDRUG", SIBLING / "base_drug"))
OUT_ROOT = HERE                                        # figures/fig-<eval>/

# ════════════════════════════════════════════════════════════════════════════════
# the layout — one folder per EVALUATION, named by it, files named by the figure
# ════════════════════════════════════════════════════════════════════════════════
# A figure id is `fig-<eval>-<name>` and selects one figure on the command line; the
# folder it lands in is its EVALUATION, so everything that answers the same question sits
# together and the folder name carries the part of the identity every file in it shares.
EVAL_FOLDER = {
    # fig-vina SPLITS (2026-09-22). One evaluation, but three questions that are read
    # separately -- affinity ranked per pocket, affinity against ligand size, and ours
    # against one baseline at a time -- and a flat folder of 90 files made the reader do
    # the sorting. Every vina figure names its own sub-folder on its @figure, so this entry
    # is the parent they all sit under rather than a folder anything is written to.
    "vina":        "fig-vina",
    "posebusters": "fig-posebusters",
    "posecheck":   "fig-posecheck",
    "consistency": "fig-rigid-consistency",
    # The interaction fingerprint IS a PoseCheck output -- ProLIF, run by PoseCheck over
    # the same pocket10 crop and read out of the same `posecheck` block of the same
    # metrics.json that strain and clashes come from (compute_baseline_interactions.py
    # imports PoseCheck directly to produce it for the baselines, which carry no posecheck
    # block of their own). So it shares PoseCheck's folder rather than taking one of its
    # own; its files keep an `interaction-` prefix to stay findable among the strain and
    # clash ones. `draw.py interaction` still selects exactly these six.
    "interaction": "fig-posecheck",
    "molweight":   "fig-molweight",
    "similarity":  "fig-similarity",
    "mcp":         "fig-mcp",
    "ensemble":    "fig-ensemble",
    "jsd":         "fig-jsd",
    # How far does Vina move a pose it is handed -- the generated pose against the same
    # molecule redocked. Its own evaluation rather than a panel inside fig-vina: fig-vina
    # compares AFFINITIES between arms, this compares two POSES of one molecule, and the
    # crystal ligand is a control here rather than a competitor.
    "redock":      "fig-posecheck/redock",
    # Does the gain track how well ORDERED the crystal is -- B-factor and occupancy of the
    # pocket the sampler was conditioned on? A property of the test set rather than of a
    # method, so it is its own evaluation and not a panel inside fig-vina.
    "uncertainty": "fig-uncertainty",
    # CoDE against ONE baseline per figure, rather than against all seven at once. A
    # sub-folder of fig-vina's and not a third variant beside `core` and `all`: those are two
    # figures, this is a SET of seven that are only readable side by side, and a set that
    # grows by one file per method does not belong in a folder whose names are read one at a
    # time. It keeps its own evaluation name, so `draw.py baseline` is the seven of them.
    "baseline":    "fig-vina/baseline",
}

# WHAT A FILE IS CALLED, once its folder already says which evaluation it belongs to.
# The part files still call save() with the name the original builder used, and the rename
# happens here, in one table, at the writing edge. So this table is also the map between
# this tree and the figure-data builders' outputs -- which is how the port was verified figure by
# figure -- and nothing in a family's code has to know about it.
#
# The rule is: drop whatever the folder already says (`rigid_`, `pb_`, `mw_`, `vina_`,
# `interaction_`, `similarity_`, `mcp_`), keep what distinguishes the figure -- including
# the metric, since `strain` and `clash` are not the evaluation's name -- and use hyphens.
STEMS = {
    # fig-vina
    "vina_dock_per_pocket_mean_core": "dock-per-pocket-mean-core",
    "vina_dock_per_pocket_mean_all": "dock-per-pocket-mean-all",
    "vina_dock_per_pocket_median_core": "dock-per-pocket-median-core",
    "vina_dock_per_pocket_median_all": "dock-per-pocket-median-all",
    "vina_dock_per_pocket_pair_core": "dock-per-pocket-pair-core",
    "vina_dock_per_pocket_pair_all": "dock-per-pocket-pair-all",
    "vina_score_per_pocket_mean_core": "score-per-pocket-mean-core",
    "vina_score_per_pocket_mean_all": "score-per-pocket-mean-all",
    "vina_score_per_pocket_median_core": "score-per-pocket-median-core",
    "vina_score_per_pocket_median_all": "score-per-pocket-median-all",
    "vina_score_per_pocket_pair_core": "score-per-pocket-pair-core",
    "vina_score_per_pocket_pair_all": "score-per-pocket-pair-all",
    "vina_min_per_pocket_mean_core": "min-per-pocket-mean-core",
    "vina_min_per_pocket_mean_all": "min-per-pocket-mean-all",
    "vina_min_per_pocket_median_core": "min-per-pocket-median-core",
    "vina_min_per_pocket_median_all": "min-per-pocket-median-all",
    "vina_min_per_pocket_pair_core": "min-per-pocket-pair-core",
    "vina_min_per_pocket_pair_all": "min-per-pocket-pair-all",
    "vina_dock_per_atom": "dock-per-atom",
    # THE VERSION TOKEN IS GONE with v1 and v2 (2026-09-22): one variant left, so `v3` in a
    # name distinguished nothing and only kept a reader looking for the other two.
    "vina_dock_per_atom_mean_core": "dock-per-atom-mean-core",
    "vina_dock_per_atom_mean_all": "dock-per-atom-mean-all",
    "vina_dock_per_atom_median_core": "dock-per-atom-median-core",
    "vina_dock_per_atom_median_all": "dock-per-atom-median-all",
    "vina_dock_per_atom_median_all_test": "dock-per-atom-median-all-test",
    "vina_dock_per_atom_median_all_grid": "dock-per-atom-median-all-grid",
    # fig-posebusters — `sucos` stays: it is a second metric, not the evaluation's name
    "pb_check_failures_all": "check-failures-all",
    "pb_check_failures_core": "check-failures-core",
    "pb_valid_per_atom_all": "valid-per-atom-all",
    "pb_valid_per_atom_core": "valid-per-atom-core",
    "pb_upset_all": "upset-all",
    "pb_upset_core": "upset-core",
    "posebusters_upset_sets": "upset-sets",
    # `pb-check-*`, not `all-checks-*` (2026-09-23, on request): the folder's other
    # multi-method sheets are pb-valid-heatmap and pb-group-bar, and this is the third.
    "pb_budget_all": "pb-budget-all",
    "pb_budget_core": "pb-budget-core",
    "posebusters_failure_budget": "failure-budget",
    "pb_all_checks_all": "pb-check-all",
    "pb_all_checks_core": "pb-check-core",
    "posebusters_all_checks": "pb-check",
    "posebusters_by_atom_range": "by-atom-range",
    "sucos_ecdf_all": "sucos-ecdf-all",
    "sucos_ecdf_core": "sucos-ecdf-core",
    "sucos_per_atom_mean_all": "sucos-per-atom-mean-all",
    "sucos_per_atom_mean_core": "sucos-per-atom-mean-core",
    "sucos_per_atom_median_all": "sucos-per-atom-median-all",
    "sucos_per_atom_median_core": "sucos-per-atom-median-core",
    # fig-posecheck
    "strain_per_atom_mean_all": "per-atom-mean-all",
    "strain_per_atom_mean_core": "per-atom-mean-core",
    "strain_per_atom_median_all": "per-atom-median-all",
    "strain_per_atom_median_core": "per-atom-median-core",
    # No `_all` stems: the clash per-atom figure is core-only (2026-09-14), and the whole
    # field is clash_per_atom_all_methods_*.
    "clash_per_atom_mean_core": "clash-per-atom-mean-core",
    "clash_per_atom_median_core": "clash-per-atom-median-core",
    "posecheck_per_atom": "per-atom",
    # the by-size set: the bin is the variant, so it reads as one family in a listing
    "strain_ecdf_le15": "ecdf-by-size-le15",
    "strain_ecdf_16_20": "ecdf-by-size-16-20",
    "strain_ecdf_21_25": "ecdf-by-size-21-25",
    "strain_ecdf_26_30": "ecdf-by-size-26-30",
    "strain_ecdf_gt30": "ecdf-by-size-gt30",
    "strain_ecdf_all": "ecdf-by-size-all",
    "clash_violin_le15": "clash-violin-by-size-le15",
    "clash_violin_16_20": "clash-violin-by-size-16-20",
    "clash_violin_21_25": "clash-violin-by-size-21-25",
    "clash_violin_26_30": "clash-violin-by-size-26-30",
    "clash_violin_gt30": "clash-violin-by-size-gt30",
    "clash_violin_all": "clash-violin-by-size-all",
    "posecheck_all_by_atom_range": "all-by-atom-range",
    "strain_per_rotbond_mean_all": "per-rotbond-mean-all",
    "strain_per_rotbond_mean_core": "per-rotbond-mean-core",
    "strain_per_rotbond_median_all": "per-rotbond-median-all",
    "strain_per_rotbond_median_core": "per-rotbond-median-core",
    "strain_box_per_rotbond_all": "per-rotbond-box-all",
    "strain_box_per_rotbond_core": "per-rotbond-box-core",
    "strain_per_rotbond": "per-rotbond",
    "strain_rotbond_all_methods_mean": "per-rotbond-all-methods-mean",
    "strain_rotbond_all_methods_median": "per-rotbond-all-methods-median",
    "strain_box_per_rotbond_all_methods": "per-rotbond-box-all-methods",
    "strain_box_per_rotbond_grid_generated": "per-rotbond-grid-generated",
    "rotbond_distribution_all_methods": "rotbond-distribution-all-methods",
    "strain_rotbond_all_methods": "per-rotbond-all-methods",
    "strain_per_atom_all_methods_mean": "per-atom-all-methods-mean",
    "strain_per_atom_all_methods_median": "per-atom-all-methods-median",
    "strain_per_atom_all_methods": "per-atom-all-methods",
    # fig-rigid-consistency
    "rigid_ecdf_pair_all": "ecdf-pair-all",
    "rigid_ecdf_pair_core": "ecdf-pair-core",
    "rigid_per_size_mean_all": "per-size-mean-all",
    "rigid_per_size_mean_core": "per-size-mean-core",
    "rigid_per_size_median_all": "per-size-median-all",
    "rigid_per_size_median_core": "per-size-median-core",
    "rigid_box_all": "rigid-consistent-box-all",
    "rigid_box_core": "rigid-consistent-box-core",
    # The same figure with the 14+ group taken apart, for the appendix.
    "rigid_box_expanded_all": "rigid-consistent-box-expanded-all",
    "rigid_box_expanded_core": "rigid-consistent-box-expanded-core",
    "rigid_size_distribution_core": "rigid-fragment-size-distribution-core",
    "rigid_size_distribution_all": "rigid-fragment-size-distribution-all",
    "rigid_size_distribution": "fragment-size-distribution",
    "rigid_size_distribution_expanded_core":
        "rigid-fragment-size-distribution-expanded-core",
    "rigid_size_distribution_expanded_all":
        "rigid-fragment-size-distribution-expanded-all",
    "rigid_size_distribution_expanded": "fragment-size-distribution-expanded",
    "rigid_violin_all": "violin-all",
    "rigid_violin_core": "violin-core",
    "rigid_violin_paperbins_all": "violin-paperbins-all",
    "rigid_violin_paperbins_core": "violin-paperbins-core",
    "rigid_fragment_per_size": "fragment-per-size",
    "rigid_fragment_by_bin": "fragment-by-bin",
    "rigid_fragment_by_bin_expanded": "fragment-by-bin-expanded",
    # fig-posecheck, interaction fingerprints (ProLIF, via PoseCheck)
    "interaction_contacts_all": "interaction-contacts-all",
    "interaction_contacts_core": "interaction-contacts-core",
    "interaction_contacts_pair_all": "interaction-contacts-pair-all",
    "interaction_hbonds_all": "interaction-hbonds-all",
    "interaction_hbonds_core": "interaction-hbonds-core",
    "interaction_hbonds_pair_all": "interaction-hbonds-pair-all",
    "interaction_pair_all": "interaction-pair-all",
    "interaction_legend": "interaction-legend",
    "posecheck_legend": "posecheck-legend",
    "interactions": "interactions",
    "interactions_pair": "interactions-pair",
    # fig-molweight
    "mw_distribution_all": "distribution-all",
    "mw_distribution_core": "distribution-core",
    "mw_ecdf_all": "ecdf-all",
    "mw_ecdf_core": "ecdf-core",
    "mw_histogram": "histogram",
    # fig-similarity — a/b/c/d is the paper's panel order, so it stays
    "similarity_a_dumbbell": "a-dumbbell",
    "similarity_b_bars": "b-bars",
    "similarity_c_scatter": "c-scatter",
    "similarity_d_table": "d-table",
    "similarity_novelty": "novelty",
    "similarity_reference_bars": "reference-bars",
    # fig-mcp
    "mcp_finetune_size": "finetune-size",
    # fig-ensemble
    "fig1_ensemble_6set_3metric": "6set-3metric",
    "fig2_ensemble_vs_alone": "vs-alone",
    # fig-redock
    "redock_ecdf_core": "ecdf-core",
    "redock_ecdf_all": "ecdf-all",
    "redock_median_core": "median-core",
    "redock_median_all": "median-all",
    "redock_summary": "summary",
    "redock_tail": "tail",
}


def filename(stem):
    """What a figure is called on disk: hyphens, minus whatever the folder already says.

    STEMS spells every current name out rather than deriving it, because the token to drop
    is different in each family (`rigid_`, `pb_`, `mw_`, ...) and a clever rule would have
    to special-case `sucos_`, `strain_` and `clash_`, which are metrics and must stay. The
    mechanical part -- underscores to hyphens -- is applied to anything not listed, so a
    newly added figure is named consistently from its first run; add it to STEMS, which is
    where the layout is documented."""
    return STEMS.get(stem, stem.replace("_", "-"))
# The figure-data builders' CSV/JSON artifacts; the builders are not part of this repository.
LEGACY = Path(os.environ.get("VOXBIND_FIGURE_DATA", REPO / "figures" / "_data"))
E = REPO / "task2-ligand_generation" / "exps"                          # the per-molecule source trees

# ════════════════════════════════════════════════════════════════════════════════
# palette — one colour per method, every figure, every spelling
# ════════════════════════════════════════════════════════════════════════════════
# A reader moving between the Vina, PoseBusters and PoseCheck figures should never have to
# re-learn the key, so the assignment lives here and nothing defines a colour locally.
# Fixed points (2026-09-09): CoDE is OUR blue #4363D8; VoxBind keeps its sand; DecompDiff
# green; TargetDiff violet (it used to be orange, a few degrees of hue from VoxBind's sand,
# which made two BASELINES read as one family). AR moved off purple onto teal-cyan because
# violet now sits where it was. Ours v2 is a DARKER shade of the CoDE blue rather than a
# hue of its own -- it is the same family -- and dark blue / mid blue / light violet is a
# lightness ladder that reads even where the hues are neighbours.
COLORS = {
    "Reference ligand": "#9AA0A6",   # grey, and dashed wherever it is drawn
    "AR":               "#0FA8C0",   # teal-cyan
    "Pocket2Mol":       "#ED76A3",   # pink
    "DiffSBDD":         "#EB4140",   # red
    "DecompDiff":       "#36BA46",   # green
    "FuncBind":         "#9F872B",   # olive — the saturated step of SOFT's #C6B46A
    "TargetDiff":       "#B58BDF",   # violet
    "VoxBind":          "#FBB178",   # sand
    "CoDE":             "#3C5FDF",   # blue — ours (LaTeX: \textsc{CoDE})
    "Ours v2":          "#2B3A8C",   # deep indigo — same family as CoDE
    # CoE uses the same pastel green in every figure (2026-09-21).
    "CoE":              "#80B993",
    # Affinity/uncertainty baselines: use their SOFT tints for filled bars.
    "DeepDTA":          "#C36C6C",   # coral
    "ProFSA":           "#939951",   # olive
    "Nesso-1":          "#8664B0",   # violet
}

# The MCP fine-tune arms are not four independent methods -- they are one model at four
# amounts of receptor-ED fine-tuning -- so they take an ordinal ramp darkening with training
# rather than four categorical hues.
#
# THE RAMP NO LONGER STARTS AT FuncBind'S OWN COLOUR (2026-09-14). It used to: both were the
# brown #A9744F, and the ramp read as "this IS FuncBind, trained further". FuncBind the
# METHOD has since moved to olive so that COLORS and SOFT agree on its hue family -- the two
# tables disagreed, and the eight-method figures draw through soft() while the interaction
# family draws through color(), so the same method came out olive in one and brown in the
# other. The ramp is left brown because it belongs to the macrocycle section, which is not
# part of that set and was not re-coloured with it. If those figures should track FuncBind
# again, re-base this ramp on #9A8430.
MCP_RAMP = {
    "FuncBind vanilla":   "#A9744F",
    "FuncBind ft 3.17M":  "#86593A",
    "FuncBind ft 8.21M":  "#684226",
    "FuncBind ft 26.1M":  "#4A2C12",
}
COLORS.update(MCP_RAMP)

ALIASES = {
    "Reference": "Reference ligand", "reference": "Reference ligand",
    "DecompDiff_ref_prior": "DecompDiff", "DecompDiff (ref-informed)": "DecompDiff",
    "targetdiff": "TargetDiff",
    "vanilla": "VoxBind", "VoxBind σ=0.9": "VoxBind", "VoxBind σ0.9": "VoxBind",
    "VoxBind sigma=0.9": "VoxBind", r"VoxBind$_{\sigma=0.9}$": "VoxBind",
    "ours_v1": "CoDE", "Ours": "CoDE", "Ours · v1": "CoDE", "Ours &middot; v1": "CoDE",
    "Ours v1": "CoDE", "VoxBind + Ours": "CoDE", "\\textsc{CoDE}": "CoDE", "CODE": "CoDE",
    "ours_v2": "Ours v2", "Ours · v2": "Ours v2", "Ours &middot; v2": "Ours v2",
    # The coords-only ablation, which every table spells differently: "Ours C" is the
    # similarity CSV's key (build_reference_similarity.py's PLAIN map), "Ours · coords" the
    # drug-design table's row name, "VoxBind+CoE" the results-bundle folder.
    "ours_c": "CoE", "Ours C": "CoE", "Ours · coords": "CoE", "Ours &middot; coords": "CoE",
    "VoxBind+CoE": "CoE", "VoxBind + CoE": "CoE", "\\oursC": "CoE", "COE": "CoE",
    "vanilla_mcp": "FuncBind vanilla", "fb_unified": "FuncBind vanilla",
    "ft_3.17M": "FuncBind ft 3.17M", "ft_8.21M": "FuncBind ft 8.21M",
    "ft_26.1M": "FuncBind ft 26.1M",
}

# Canonical baseline sequence shared by bars, boxes, panels, and legends.
METHOD_ORDER = ("AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff",
                "FuncBind", "VoxBind", "CoDE")


def method_order_key(label):
    """Reference first; optional CoE between VoxBind and CoDE; unknowns stable last."""
    key = ALIASES.get(label, label)
    if key == "Reference ligand":
        return -1
    if key == "CoE":
        return METHOD_ORDER.index("CoDE") - 0.5
    if key == "VoxBind σ=1.0":
        return METHOD_ORDER.index("VoxBind") + 0.1
    return METHOD_ORDER.index(key) if key in METHOD_ORDER else len(METHOD_ORDER)


def ordered_methods(items, key=None):
    """Stable ordering of labels or records without changing their data or membership."""
    return sorted(items, key=lambda item: method_order_key(key(item) if key else item))


# What a LEGEND shows, where that differs from the key the data is stored under. The key
# heads every exported CSV column and JSON object, and a column header containing
# matplotlib mathtext is not something a reader can join on -- so the exports carry plain
# "VoxBind" and only the drawn label picks up the sigma subscript.
DISPLAY = {"VoxBind": r"VoxBind$_{\sigma=0.9}$"}


def color(label):
    """Colour for a method, by any of its spellings. Raises rather than defaulting: a
    silently grey method in a figure is worse than a failed build."""
    key = ALIASES.get(label, label)
    if key not in COLORS:
        raise KeyError(f"no colour for method {label!r} (known: {', '.join(sorted(COLORS))})")
    return METHOD_COLOR.get(key, COLORS[key]) if not METHOD_HATCH.get(key) else COLORS[key]


def display(label):
    """The string to put in a legend for a method, by any of its spellings."""
    key = ALIASES.get(label, label)
    if key not in COLORS:
        raise KeyError(f"no method named {label!r}")
    return DISPLAY.get(key, key)


# LIGHTER STEPS OF THE PALETTE HUES, shared like COLORS so a pastel tone is defined once. They
# used to be hex literals repeated per family (CoDE's soft blue in three of them), which is how
# one figure's tint drifts from another's. Keyed like COLORS, reached by any spelling.
#   SOFT  one step lighter -- CoDE in the Vina per-atom family, the similarity dumbbell and the
#         eight-method strain figures, where the full #4363D8 was the one saturated hue.
#   PALE  two steps lighter -- VoxBind sigma=1.0's tint in the similarity figure, and the two
#         ends of the strain-grid colormap.
# The eight-method figures draw EVERY series through soft(), so the five published baselines
# now have tints of their own (2026-09-14): at full saturation they sat badly beside CoDE's and
# VoxBind's tints and TargetDiff's violet, which are already pale. Each keeps its hue family --
# teal, pink, red, green, brown -- so a method is the same colour across the paper, lifted to
# the tints' lightness. COLORS holds the stronger variants used by three-arm, Vina and PoseBusters
# figures; CoE intentionally uses the same pastel green in both tables.
# 2026-09-21: categorical saturation raised 10%, soft tints 25%, at fixed HSL lightness.
# Neutral reference grey and the ordinal MCP ramp remain unchanged.
SOFT = {
    "CoDE":       "#7588F5",
    "CoE":        "#80B993",   # pastel green, shared with color()
    "DeepDTA":    "#E78E8E",   # pastel coral
    "ProFSA":     "#C5C981",   # pastel olive
    "Nesso-1":    "#AC8CD5",   # pastel violet
    "AR":         "#75CBDB",   # was #17A2B8 teal-cyan
    "Pocket2Mol": "#F99DBD",   # was #E87BA4 pink
    "DiffSBDD":   "#FA8684",   # was #E34948 red
    "DecompDiff": "#85D38C",   # was #3CB44B green
    # FuncBind LEAVES ITS HUE FAMILY here (2026-09-14), the one tint that does: as a tan it was
    # the same warm family as VoxBind's sand and the two could only be told apart by weight.
    # Olive-gold is the one family none of the other eight occupy -- teal, pink, coral, green,
    # violet, sand, periwinkle, grey. COLORS keeps the brown, which is what the MCP fine-tune
    # ramp darkens off, so FuncBind is brown in the macrocycle figures and olive in these.
    # COLORS now carries the saturated olive #9A8430, so this is an ordinary lighter step of
    # its own hue like every other entry here -- not a hue change of its own. It used to be
    # the one tint that left its family, which put FuncBind in two colours depending on
    # which table a figure read.
    "FuncBind":   "#D2BB5E",   # was #A9744F brown, then #C4A083 tan
}
# Approved A Gentle (2026-09-22): keep each SOFT hue, reduce HLS saturation
# by 15%, and move lightness 12% toward white. The reference stays neutral.
# COLORS/SOFT remain intact for historical compatibility and candidate construction.
def _gentle_color(value):
    h, lightness, saturation = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(value))
    return matplotlib.colors.to_hex(colorsys.hls_to_rgb(
        h, lightness + (1 - lightness) * .12, saturation * .85)).upper()


# Preserve the previous palette for archived figures and candidate reproducibility.
LEGACY_PALE = {key: (value if key == "Reference ligand" else _gentle_color(SOFT.get(key, value)))
               for key, value in COLORS.items()}
PALE = dict(LEGACY_PALE)
# The approved sheet is now the default source, loaded before family constants.
# An explicit override remains useful for controlled palette comparisons.
_method_palette_path = os.environ.get("VOXBIND_METHOD_PALETTE") or str(
    HERE / "fig-palette" / "method-palette-approved.json")
METHOD_PALETTE_ROWS = []
if _method_palette_path != "legacy":
    with open(_method_palette_path, encoding="utf-8") as _palette_handle:
        METHOD_PALETTE_ROWS = json.load(_palette_handle)["rows"]
METHOD_COLOR = {r["method"]: r["color"] for r in METHOD_PALETTE_ROWS}
HIGHLIGHT = {r["method"]: r["highlight"] for r in METHOD_PALETTE_ROWS}
METHOD_HATCH = {r["method"]: r.get("hatch", "") for r in METHOD_PALETTE_ROWS}
for _palette_row in METHOD_PALETTE_ROWS:
    if not _palette_row.get("hatch"):
        PALE[_palette_row["method"]] = _palette_row["color"]


def highlight(label):
    """Approved Highlight tone, accepting the same method aliases as color()."""
    return HIGHLIGHT[ALIASES.get(label, label)]


def soft(label):
    """Legacy tint retained for historical compatibility and candidate construction."""
    key = ALIASES.get(label, label)
    return SOFT.get(key, COLORS[key])


def pale(label):
    """A method's pale tint. Raises rather than falling back: a pale step is chosen on purpose,
    and silently getting the full-strength colour instead would not look like an error."""
    key = ALIASES.get(label, label)
    if key not in PALE:
        raise KeyError(f"no pale tint for {label!r} (known: {', '.join(sorted(PALE))})")
    return PALE[key]


# ════════════════════════════════════════════════════════════════════════════════
# house style — no panel titles, warm near-black furniture, left+bottom spines only
# ════════════════════════════════════════════════════════════════════════════════
INK, GRID, AXIS = "#514F52", "#c2c6cd", "#514F52"
LEGEND_EDGE = "#b6bbc3"
SOLID, DASH = (0, ()), (0, (4, 2.6))
DOT = (0, (1, 2.6))

# ── the dash a method is drawn with, everywhere it is drawn ──────────────────────
# THE SECOND IDENTITY CHANNEL, and like the palette it is decided HERE and nowhere else.
# Nine series on one axis cannot be told apart by hue alone, so every eight-method figure
# carries dash as well -- and it only works if a method's dash is the same in all of them.
# It was not. Two tables had drifted out of step: the ECDF family's (in PCSZ_BASELINES /
# PCSZ_LOCAL) and the rotatable-bond family's (RB_BASE_DASH), and the rotatable-bond one was
# shifted by a whole method -- its Pocket2Mol carried the ECDF's DiffSBDD pattern, its
# DiffSBDD the ECDF's DecompDiff, and so on down the list. Worse, the standalone key
# (fig-posecheck-strain-legend) read the rotatable-bond table while the ECDF drew the other,
# so the legend was decoding the panel wrongly.
#
# AR AND POCKET2MOL USED TO SHARE (5, 2) in the ECDF, which left them separable by hue only
# -- exactly what the dash channel exists to prevent. Pocket2Mol takes the pattern that was
# spare, so all five baselines are now distinct.
#
# SOLID MEANS OURS. TargetDiff is dashed even though we run it locally: the rotatable-bond
# family gave it a solid line because its helper keyed on provenance (RB_LOCAL_LABELS, "we
# ran this") rather than on authorship, and solid reads as ours in every other figure.
METHOD_DASH = {
    "Reference ligand": SOLID,          # solid since 2026-09-14, as the ECDF draws it
    "AR":               (0, (5, 2)),
    "Pocket2Mol":       (0, (3, 1.4, 1, 1.4)),
    "DiffSBDD":         (0, (1, 1.6)),
    "DecompDiff":       (0, (6, 2, 1, 2)),
    "FuncBind":         (0, (9, 3)),
    "TargetDiff":       (0, (6, 2)),
    "VoxBind":          SOLID,
    "CoDE":             SOLID,
}


def dash(label):
    """The dash pattern for `label`, through the alias table.

    Raises rather than defaulting to solid: a method missing here is a rename to fix, and a
    silent solid would hand it the channel that means ours."""
    key = ALIASES.get(label, label)
    if key not in METHOD_DASH:
        raise KeyError(f"no dash for {label!r} (known: {', '.join(sorted(METHOD_DASH))})")
    return METHOD_DASH[key]
MODEL_LW, REF_LW = 2.35, 1.5
AXIS_LW, GRID_LW = 1.35, 1.1
AXIS_LABEL_FS = 15.5
DIST_LW, DIST_FILL = 1.7, BACKGROUND_ALPHA

# GEOMETRY, in inches, off the 3-line base figure of 7.25 x 2.77.
WIDE, TALL = 1.05, 1.155
FIG_W = 7.25 * WIDE
PANEL_H = 2.77 * TALL
STACK_H = PANEL_H + 1.62              # a full panel plus a distribution strip
HEIGHT_RATIOS = (2.62, 1.05)
# The gap between stacked panels goes through tight_layout's h_pad, NOT gridspec_kw's
# hspace: an explicit hspace makes tight_layout call the axes "not compatible", warn, and
# leave ~12% of the figure as dead white at the top.
H_PAD = 0.2
XTICK_STEP = 5
X_LABEL = "Number of heavy atoms in ligand"

RC = {
    "font.family": "DejaVu Sans", "font.size": 15,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3, which
                                     # journals reject and Illustrator mangles
}


def use_style():
    plt.rcParams.update(RC)


def furniture(ax, *, ylabel=None, xlabel=None, xlim=None, xloc=XTICK_STEP):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS, labelpad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FS, labelpad=9)
    if xlim:
        ax.set_xlim(*xlim)
    if xloc:
        ax.xaxis.set_major_locator(MultipleLocator(xloc))
    ax.tick_params(labelsize=14, direction="out", length=3.5, width=AXIS_LW, pad=4,
                   colors=AXIS)
    ax.grid(True, axis="y", color=GRID, lw=GRID_LW, ls=DOT)
    ax.grid(True, axis="x", color=GRID, lw=GRID_LW, ls=DOT)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
        ax.spines[sp].set_linewidth(AXIS_LW)


def legend(target, handles, loc="upper right", ncol=1, fontsize=12.5, **kw):
    """Opaque white with a rule in the axis pen, so the dotted grid does not run through
    the labels but the box itself stays quiet. `target` is an Axes OR a Figure -- a key
    that has to sit outside the data area belongs to the figure, not to the axes it would
    otherwise cover."""
    leg = target.legend(handles=handles, loc=loc, ncol=ncol, frameon=True,
                        fontsize=fontsize, handlelength=1.9, handletextpad=0.55,
                        labelspacing=0.3, borderpad=0.4, borderaxespad=0.39,
                        facecolor="white", edgecolor=LEGEND_EDGE, framealpha=1.0, **kw)
    leg.get_frame().set_boxstyle("square", pad=0.16)
    leg.get_frame().set_linewidth(AXIS_LW)
    leg.set_zorder(7)
    for text in leg.get_texts():
        text.set_color(INK)
    return leg


def row_major(handles, ncol):
    """Reorder for matplotlib's column-major legend fill, so the key reads ACROSS.

    A legend with ncol=4 lays its entries down column 1, then column 2 -- so the drawing
    order the eye expects along a row is not the order it gets. Interleaving here puts the
    entries back in reading order."""
    rows = -(-len(handles) // ncol)
    out = []
    for c in range(ncol):
        for r in range(rows):
            i = r * ncol + c
            if i < len(handles):
                out.append(handles[i])
    return out


def one_frame(fig, legs):
    """Take the frames off stacked keys and draw ONE around their union, so they read as a
    single box. Returns the patch, which is then the outermost thing the axes must clear.

    A legend's window extent already includes its own borderpad, so the union sits exactly
    where the individual frames did -- no padding is added here, or the pair would gain a
    ring the single-legend figures do not have."""
    fig.canvas.draw()
    boxes = [l.get_window_extent(fig.canvas.get_renderer()) for l in legs]
    for l in legs:
        l.set_frame_on(False)
    (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(
        [(min(b.x0 for b in boxes), min(b.y0 for b in boxes)),
         (max(b.x1 for b in boxes), max(b.y1 for b in boxes))])
    patch = matplotlib.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, figure=fig,
        facecolor="white", edgecolor=LEGEND_EDGE, linewidth=AXIS_LW, zorder=6)
    fig.add_artist(patch)
    return patch


def arm_handles(arms, include_ref=True, paint=None):
    """`paint` maps a method to its colour: pale() by default, pale() for the families
    drawn in the tints."""
    paint = paint or pale
    h = [Line2D([], [], color=pale(REF_LABEL), lw=REF_LW, ls=DASH, label=REF_LABEL)] \
        if include_ref else []
    return h + [Line2D([], [], color=paint(lab), lw=MODEL_LW, ls="-", label=display(lab))
                for lab, _, _ in arms]


def fit(fig, **kw):
    """tight_layout, then give back whatever it under-reserved.

    At this type size it can leave a rotated y label overrunning the figure edge even
    though it has just run, and the label is then silently SLICED OFF in the raster.
    get_tightbbox reports the overrun after the fact, so hand back exactly that much --
    re-running tight_layout with a rect would re-apply `pad` on top of the correction and
    cost plot area."""
    fig.tight_layout(**kw)
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    sp, adj = fig.subplotpars, {}
    if bb.x0 < 0:
        adj["left"] = sp.left + (-bb.x0) / w
    if bb.y0 < 0:
        adj["bottom"] = sp.bottom + (-bb.y0) / h
    if bb.x1 > w:
        adj["right"] = sp.right - (bb.x1 - w) / w
    if bb.y1 > h:
        adj["top"] = sp.top - (bb.y1 - h) / h
    if adj:
        fig.subplots_adjust(**adj)


# ════════════════════════════════════════════════════════════════════════════════
# the heat ramp and its colourbar — shared by every figure keyed to a continuous scale
# ════════════════════════════════════════════════════════════════════════════════
# HERE, NOT IN A PART (2026-09-15, on request). The ramp was born in posecheck_rotbond and
# fig-posebusters-valid-heatmap reached across for it by name at call time, which worked
# only because parts are pasted into one module: it read as one family borrowing another's
# private constant, and the borrow was easy to miss when either end changed. A colour scale
# used by two families is house furniture, so it sits beside INK, legend() and furniture()
# under the same rule the palette follows -- decided in ONE place, read everywhere else.
# THE HOUSE HEAT RAMP: one hue, light to dark (2026-09-15, on request; it was matplotlib's
# diverging coolwarm, then a purple). "Darker is more of the quantity the bar is named
# after" -- more strain in the rotbond grid, a higher pass rate in the PoseBusters map.
# Diverging was the wrong family for either: its white middle marked 400 kcal/mol, a number
# nothing in the data means.
#
# TWO STOPS, BOTH GIVEN (2026-09-23, on request: "#E8F2F7, #465CCF 이렇게 두 색깔 써서 공용
# color bar 다시 구성하고"). The ramp is the straight sRGB blend between those two colours and
# nothing else is chosen here -- no extension of a line through a pair, no derivation from a
# method's tint. A ramp keyed by its own colourbar carries no method identity, so hex literals
# are allowed here where the palette rule otherwise forbids them; the deep end is close to
# CoDE's blue (#4363D8) without being it.
#
# WHAT THIS REPLACED, in order, so nobody re-derives a rationale that has been retired twice:
# matplotlib's diverging coolwarm, whose white middle marked 400 kcal/mol -- a number nothing
# in the data means; a hand-placed five-stop version; then the extension of the line through
# #B4BEF0 and #8291E8, stretched 1.4x each way to #FAFDFB -> #3C52DD (2026-09-15) and lightened
# at the deep end to #5E70E3 the same day, on request ("100에 해당하는 수치를 덜 어둡게"); then
# a light grey to a darkened highlight("CoDE"), derived rather than written (earlier the same
# day as this). This is the version with the fewest decisions in it: two colours, given.
#
# WHAT IT SPANS, stated so nobody re-tunes it by accident: luminance 0.87 at the light end to
# 0.135 at the deep one. That is a wider range than the #FAFDFB -> #5E70E3 pair it replaces
# (0.98 -> 0.20) -- cells ten points apart separate further -- and the light end is a pale blue
# a reader can see against white paper rather than a near-white that disappears into it.
# The deep end was taken up in SATURATION only (2026-09-23, on request: "마지막 색깔 채도 높일래
# 조금만"): #465CCF -> #3C55D9 is HLS saturation 0.588 -> 0.676 at the hue and the lightness it
# was given with, so the ramp ends a little more colour and no darker (luminance 0.135 ->
# 0.125). Hue 230 and lightness 0.543 are the given colour's own and are not ours to move.
#
# AND THEN THE RAMP WAS MADE TO PASS THROUGH CoDE (2026-09-23, on request: "CoDE 색깔이
# 포함되도록"). The deep end is no longer a colour of its own: it is the line from the light end
# THROUGH CoDE continued to HEAT_CODE_AT of the way, so the deck's CoDE blue is a point ON the
# bar rather than something near it, and the end is what lies beyond it. 0.83 puts CoDE at 83%
# of the bar and leaves the end at luminance 0.108 -- 0.055 below CoDE, which is the separation
# the budget's darkest segment needs from its CoDE-coloured neighbour. The end lands at
# #2643F1, within a shade of the #3C55D9 it replaces (luminance 0.125 -> 0.108): the bar reads
# as the same bar, one point on it is now a method's colour.
# WHICH CoDE: highlight(), not color(). The palette's CoDE tones are #8092F6 and #4761F2, and
# the lighter one is too light to sit near the end of a ramp -- run the line through it and the
# bar bottoms out at luminance 0.247, where the PoseBusters map paints every cell from 94 to
# 100 the same periwinkle and white numbers lose their contrast on it. The highlight is the
# tone CoDE is emphasised in anyway, and it leaves the bar as deep as it was.
# WHY NOT END AT CoDE ITSELF: the bar would then have nothing below it, and every heat map in
# the deck spends its top on cells at 95-100% -- they would all end on a method's identity
# colour, which is the confusion the palette rule exists to prevent. Passing THROUGH it says
# `this much` where ending on it would say `this method`.
HEAT_CODE_AT = 0.83
HEAT_LIGHT = "#E8F2F7"
HEAT_STOPS = (HEAT_LIGHT, matplotlib.colors.to_hex(tuple(
    min(1.0, max(0.0, light + (code - light) / HEAT_CODE_AT))
    for light, code in zip(matplotlib.colors.to_rgb(HEAT_LIGHT),
                           matplotlib.colors.to_rgb(highlight("CoDE"))))))
HEAT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("voxbind_heat", HEAT_STOPS)
# The bar's SHAPE, which is the same in both figures and is what a reader recognises before
# reading either: vertical on the right, a little short of the block it keys (0.92) and thin
# (aspect 38), so it reads as a key rather than as a tenth panel.
HEAT_CBAR_SHRINK, HEAT_CBAR_ASPECT = 0.92, 38
# Gap from the panel block to the bar, doubled from 0.012 (2026-09-13) -- a share of the
# figure width, which is what colorbar(pad=) takes.
HEAT_CBAR_PAD = 0.024
# The caption's gap off the bar's numbers, DOUBLED (2026-09-14): measured off the PNG it was
# 19 px at 220 dpi, i.e. 6.2 pt on matplotlib's default labelpad of 4, so 10.5 buys about 12.4.
HEAT_CBAR_LABELPAD = 10.5


def heat_colorbar(fig, ax, norm, label, *, label_fs, tick_fs, extend="neither", ticks=None,
                  pad=HEAT_CBAR_PAD, labelpad=HEAT_CBAR_LABELPAD, cmap=None, cax=None,
                  orientation="vertical"):
    """The house heat colourbar: HEAT_CMAP under `norm`, drawn the same way every time.

    `ax` is whatever the bar should steal its space from -- one axes or an array of them.
    What is NOT shared is point size: the rotbond grid is nine panels on a 14.45 in canvas
    and carries 27/18 pt, the PoseBusters map is one 8.4 x 6.2 in block at 13/11.5, and a bar
    that copied one into the other would tower over or vanish beside the figure it keys. So
    the sizes are arguments and everything else -- ramp, shape, pad, the quarter-turned
    numbers, the missing outline -- is fixed here, which is the point of having the helper.

    The numbers are turned a quarter turn anticlockwise (2026-09-14) so they read along the
    bar like its name; anchored left and centred on the tick, which is where they sat upright.
    No frame round the bar (2026-09-14), as the boxes it keys lost theirs -- the extend arrow
    is part of that same outline path, so it loses its edge with it and reads as pure fill."""
    kw = {} if ticks is None else {"ticks": ticks}
    # `cax` is a bar the caller has ALREADY placed, for a figure whose geometry is arithmetic
    # rather than a layout engine's negotiation -- the 22-check table sets its own axes in
    # inches, and a bar that steals space from them would move the table off its own grid.
    # shrink/aspect/pad describe space taken OUT of `ax` and mean nothing in that case, so
    # they are not passed; everything else about the bar is the same one.
    if cax is not None:
        kw["cax"] = cax
    else:
        kw.update(ax=ax, shrink=HEAT_CBAR_SHRINK, aspect=HEAT_CBAR_ASPECT, pad=pad)
    # `cmap` is for trying a ramp on one figure without moving the house one under the other.
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap or HEAT_CMAP),
                      extend=extend, orientation=orientation, **kw)
    cb.set_label(label, fontsize=label_fs, color=INK, labelpad=labelpad)
    cb.ax.tick_params(labelsize=tick_fs, colors=AXIS, width=AXIS_LW)
    # The quarter turn is what makes the numbers read ALONG a vertical bar. A bar laid on its
    # side already reads that way, and turning its numbers would stand them on end.
    if orientation == "vertical":
        plt.setp(cb.ax.get_yticklabels(), rotation=90, va="center", ha="left")
    cb.outline.set_visible(False)
    return cb


# ════════════════════════════════════════════════════════════════════════════════
# the pose data — one load per run, shared by every figure that asks for it
# ════════════════════════════════════════════════════════════════════════════════
# label, key, run root — drawn in this order, so our method lands on top. The labels are
# DATA KEYS as well as labels (they head the exported CSV columns), so they stay plain;
# display() dresses them for a legend.
ARMS = [
    ("AR",         "ar",         f"{E}/baselines_pose/ar"),
    ("Pocket2Mol", "pocket2mol", f"{E}/baselines_pose/pocket2mol"),
    ("DiffSBDD",   "diffsbdd",   f"{E}/baselines_pose/diffsbdd"),
    ("DecompDiff", "decompdiff", f"{E}/baselines_pose/decompdiff"),
    ("FuncBind",   "funcbind",   f"{E}/baselines_pose/funcbind"),
    ("TargetDiff", "targetdiff", str(BASEDRUG / "eval" / "targetdiff")),
    ("VoxBind",    "vanilla",    f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    ("CoDE",       "ours_v1",    f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
]

ARMS = ordered_methods(ARMS, key=lambda row: row[0])
# `core` is the comparison this section makes: our method against the model it modifies,
# with the crystal ligand as the benchmark. `all` is every arm the figure has data for.
CORE = ("vanilla", "ours_v1")
REF_LABEL = "Reference ligand"
REF_COLOR = pale(REF_LABEL)
# The crystal ligand is the same molecule in every run that carries one, so any of OUR
# roots will do -- but it is resolved BY KEY, never by position: ARMS[2][2] silently became
# DiffSBDD the moment the baselines were prepended, and the staged baseline dirs carry no
# reference block at all.
REF_ROOT = next(root for _, key, root in ARMS if key == "ours_v1")

# THE POCKET SET IS THE 79, NOT EACH ARM'S OWN COVERAGE. TargetDiff holds 100 pockets, but
# CoDE only sampled the 79 with usable deposited electron density, and every arm covers
# those -- so every figure is like-for-like. target_71 is in: it is unscoreable on the
# DOCKING side only, both pose metrics work on it.
P79 = json.load(open(f"{E}/frozenenc_probes/p79_targets.json"))
EDGES = [0, 16, 21, 26, 31, 10 ** 6]
BIN_LABELS = ["≤15", "16–20", "21–25", "26–30", ">30"]

# A heavy-atom count is plotted only where EVERY drawn arm has this many molecules, so the
# curves start and stop together and none carries a tail the others cannot answer.
MIN_N = 25
# The reference window: +-REF_WIN atoms, kept where it pools at least MIN_REF ligands.
# There is one crystal ligand per pocket -- 79 over ~40 sizes -- so a per-count reference
# curve would be noise, and its share would be a picket fence of 5%-tall spikes.
REF_WIN, MIN_REF = 4, 12


def rows_of(target_dir, reference=False):
    """One record per molecule: heavy atoms, PoseCheck strain and clashes, its interaction
    fingerprint, PoseBusters validity and the checks it failed. `s=None` means the UFF
    relaxation did not converge -- kept as None rather than 0, because dropping a value is
    not the same as scoring it.

    `ifp` IS THE POSECHECK FINGERPRINT, NOT `m["interactions"]`. A molecule record carries
    two unrelated things under that name: `m["interactions"]` is the docking side's
    contact/clash geometry, `m["posecheck"]["interactions"]` is the ProLIF fingerprint. A
    type the molecule does not make is ABSENT from the dict, not zero in it, so read it
    with .get(type, 0) and never with len() or a sum over its keys."""
    path = os.path.join(target_dir, "metrics.json")
    if not os.path.exists(path):
        return []
    try:
        j = json.load(open(path, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    items = [j.get("reference")] if reference else (j.get("samples") or [])
    out = []
    for m in items:
        if not m or not m.get("n_atoms"):
            continue
        pc = m.get("posecheck") if isinstance(m.get("posecheck"), dict) else {}
        pb = m.get("posebusters") if isinstance(m.get("posebusters"), dict) else {}
        s = pc.get("strain")
        out.append({
            "n": int(m["n_atoms"]),
            "smi": m.get("smiles"),
            "s": float(s) if isinstance(s, (int, float)) and np.isfinite(s) else None,
            "c": float(pc["clashes"]) if pc.get("clashes") is not None else None,
            "ifp": pc["interactions"] if isinstance(pc.get("interactions"), dict) else None,
            "v": pb["valid"] if isinstance(pb.get("valid"), bool) else None,
            "f": sorted(k for k, ok in (pb.get("checks") or {}).items() if ok is False),
        })
    return out


_POSE_CACHE = {}


def pose_data():
    """(per-arm {target: rows}, p79 rows per arm, the 79 crystal-ligand rows).

    Cached for the life of the process. This is the expensive part of every pose figure --
    ~60k molecules across eight run trees -- and drawing six of them in one command used to
    mean reading it six times, once per builder."""
    if "d" not in _POSE_CACHE:
        t0 = time.time()
        data = {}
        for _, key, root in ARMS:
            names = sorted(d for d in os.listdir(root) if d.startswith("target_")
                           and os.path.isdir(os.path.join(root, d)))
            data[key] = {t: rows_of(os.path.join(root, t)) for t in names}
        p79 = {key: [r for t in P79 for r in data[key].get(t, [])] for _, key, _ in ARMS}
        ref = [r for t in P79 for r in rows_of(os.path.join(REF_ROOT, t), reference=True)]
        _POSE_CACHE["d"] = (data, p79, ref)
        n = sum(len(v) for v in p79.values())
        print(f"  [loaded {n:,} molecules over {len(ARMS)} arms in {time.time()-t0:.1f}s]")
    return _POSE_CACHE["d"]


def arms_for(field, data, arms=None):
    """The arms that carry `field` across the WHOLE 79-pocket set, in ARMS order.

    An arm with no data for a metric must not become an empty curve and a legend row that
    says it was measured and lost -- it was never measured. The baselines hold PoseBusters
    and not PoseCheck, so the two families end up with different arm lists from one ARMS.
    The test is per POCKET, not "has any value at all": a scoring run in progress leaves an
    arm with a handful of finished pockets, and pooling those would draw a complete-looking
    curve over 4% of the data."""
    arms = ARMS if arms is None else arms
    out = []
    for a in arms:
        pockets = [t for t in P79 if t in data.get(a[1], {})]
        if not pockets:
            continue
        scored = sum(any(r[field] is not None for r in data[a[1]][t]) for t in pockets)
        if scored == len(pockets) == len(P79):
            out.append(a)
    return out


def variants(field=None, data=None):
    """(name, arms) for each figure variant, core first."""
    arms = ARMS if field is None else arms_for(field, data)
    return (("core", [a for a in arms if a[1] in CORE]), ("all", arms))


# ── statistics ───────────────────────────────────────────────────────────────────
def by_size(rows, field, key="n"):
    """{x: [values]}, where x is the row's `key` -- heavy atoms by default. `key` exists so
    a builder can resolve the same metric against another per-molecule integer (rotatable
    bonds) through the same statistics and the same x_range/model_curve/reference_curve."""
    out = collections.defaultdict(list)
    for r in rows:
        if r[field] is not None and r.get(key) is not None:
            out[r[key]].append(r[field])
    return out


def bin_of(n):
    return min(int(np.searchsorted(EDGES, n, side="right")) - 1, len(BIN_LABELS) - 1)


def model_curve(per, xs, f, win=0):
    """Value at each exact heavy-atom count, or over a centred +-win window.

    A median of a continuous quantity (strain, clashes) is stable at one count -- each arm
    has 100-300 molecules there -- so those curves are drawn unrolled. A PROPORTION IS NOT:
    at n=150 a per-count validity rate carries about +-4 pp of sampling noise, enough that
    arms separated by 2-10 pp cross each other repeatedly on nothing, so those curves pass
    win=2 (~750 molecules per point, about +-1.8 pp). The window is the only smoothing
    anywhere; nothing is interpolated and no point is invented."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= win for v in vals] if win \
            else per.get(a, [])
        out.append(f(p) if p else None)
    return out


def reference_curve(per, xs, f):
    """The same over a centred +-REF_WIN window; None where the window is too thin to mean
    anything, which leaves a gap in the line rather than an invented value."""
    out = []
    for a in xs:
        p = [v for n, vals in per.items() if abs(n - a) <= REF_WIN for v in vals]
        out.append(f(p) if len(p) >= MIN_REF else None)
    return out


def x_range(per_arm, arms):
    """Counts where every drawn arm has at least MIN_N molecules."""
    common = set.intersection(*(set(per_arm[key]) for _, key, _ in arms))
    return [a for a in sorted(common) if all(len(per_arm[key][a]) >= MIN_N
                                             for _, key, _ in arms)]


def share(per, xs):
    total = sum(len(v) for v in per.values())
    return ([100.0 * len(per.get(a, ())) / total for a in xs] if total else []), total


def rolled(values, xs):
    """The reference's shares under the same +-REF_WIN window -- averaged, not summed, so
    it stays on the same scale as the models' per-count shares and all read on one axis."""
    index = {a: i for i, a in enumerate(xs)}
    return [st.mean([values[index[n]] for n in xs if abs(n - a) <= REF_WIN]) for a in xs]


# ════════════════════════════════════════════════════════════════════════════════
# per-molecule Vina, from the results bundle — the one source every method shares
# ════════════════════════════════════════════════════════════════════════════════
# Until 2026-09-13 the Vina figures read each run's own eval_docking_results_full79.json
# out of task2-ligand_generation/exps/, which only OUR arms have: the five published baselines were docked
# on the evaluation server and only their aggregates came back, so they could not be drawn at all. The
# bundle now carries <Method>/eval/vina_docking/per_molecule.csv for every method --
# target, n_atoms, vina_score, vina_min, vina_dock, and an in_density79 flag -- which is
# one uniform table per method instead of a per-run file format. Both Vina families read
# it, so "which methods are in this figure" is a question about which CSVs exist.
BUNDLE = REPO / "results" / "task2-drugdesign"


# ════════════════════════════════════════════════════════════════════════════════
# the registry — id -> function, and the folder that id names
# ════════════════════════════════════════════════════════════════════════════════
FIGURES = {}          # "fig-{eval}-{figname}" -> function(out_dir) -> None
_FORMATS = ("png", "pdf")
FIGURES_ONLY = False  # --figures-only: redraw pictures, leave every exported table as it is


def figure(fig_id, *, needs=(), folder=None, capture=False, isolate=False):
    """Register one figure.

    `fig_id` is the CLI name; the output folder is its EVALUATION (see EVAL_FOLDER), so
    everything answering the same question lands together. `folder` overrides that with a
    path under OUT_ROOT, which is how an evaluation that grows past a readable folder --
    PoseCheck has forty figures over three different measurements -- splits into
    sub-folders without the ids or the CLI changing.

    `needs` names the evaluation inputs the function reads, and is printed by --list so a
    missing input is a readable message rather than a traceback three call levels down.

    `capture=True` marks a figure laid out from saved browser captures (the 3D panels): it
    is drawn when named, or by `--all --with-3d`, but not by plain `--all`.

    `isolate=True` runs the figure in its own process whenever more than one figure is
    drawn: the standalone analysis modules (uncertainty, palette previews, overviews) set
    matplotlib's global rcParams, and used to be separate scripts, so they must not leak
    their style into the figures drawn after them."""
    def deco(fn):
        if fig_id in FIGURES:
            raise KeyError(f"duplicate figure id {fig_id!r}")
        if not fig_id.startswith("fig-"):
            raise ValueError(f"figure id must be fig-<eval>-<name>: {fig_id!r}")
        fn.fig_id, fn.needs, fn.folder = fig_id, tuple(needs), folder
        fn.capture, fn.isolate = capture, isolate
        FIGURES[fig_id] = fn
        return fn
    return deco


def figure_folder(fig_id):
    """Where a figure's files go, relative to OUT_ROOT."""
    override = getattr(FIGURES[fig_id], "folder", None)
    return override or EVAL_FOLDER[fig_id.split("-")[1]]


def save(fig, out, stem):
    """PNG to look at, PDF to place -- vector, with live text.

    `stem` is the name the ORIGINAL builder used; filename() turns it into the name this
    tree uses. Families pass the original name so STEMS stays a usable map back to
    the figure-data builders' outputs, which is what the port was verified against."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = filename(stem)
    if not prepare_publication_figure(fig, out / stem):
        plt.close(fig)
        return []
    written = []
    for ext in retained_formats(out / stem, _FORMATS):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, facecolor="white")
        sync_saved_figure(p)
        written.append(p)
    plt.close(fig)
    return written


def write_csv(out, stem, header, rows):
    """The numbers behind a figure, beside it. A figure whose values cannot be read back
    out is not reviewable."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{filename(stem)}.csv"
    if FIGURES_ONLY:                      # a redraw never rewrites the numbers behind it
        return p
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return p


def legacy(*parts):
    """A data artifact under figures/_data/ (VOXBIND_FIGURE_DATA), where the figure-data
    builders write. Raises with the path if it is not there -- an empty figure that silently
    dropped a method is the failure mode this exists to prevent."""
    p = LEGACY.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(
            f"missing input {p}\n"
            f"  rebuild it with the matching figure-data builder")
    return p
# Match summary-all's proportions: 11.5 pt numeric ticks, 16.1 pt axis names,
# 14.5 pt method names, and 2.2 pt axes, scaled to this figure's 14 pt ticks.
RF_BOX_TICK_FS = 14.0
RF_BOX_LEGEND_FS = RF_BOX_TICK_FS * (14.5 / 11.5)
RF_BOX_AXIS_LW = RF_BOX_TICK_FS * (2.2 / 11.5)


# ════════════════════════════════════════════════════════════════════════════════
# fig-mcp-finetune-size and fig-ensemble-* — drawn under matplotlib's own defaults
# ════════════════════════════════════════════════════════════════════════════════
# These three figures predate 260910's house style and never applied it: each ran as its own
# process, under stock rcParams. draw.py is ONE process, so a figure drawn after a styled
# one would silently inherit 15 pt DejaVu in warm near-black, and one drawn BEFORE a styled
# one would leave the style stripped for it. MISC_PLAIN_RC is rcParams as they stood when
# draw.py was imported -- before any use_style() can have run -- which is exactly the
# fresh-process state the originals drew in, and it is applied through rc_context so the
# reset lasts only as long as the figure. `backend` is left out: restoring it through
# rcParams is not how a backend is set, and rc_context drops it for the same reason.
MISC_PLAIN_RC = {k: v for k, v in plt.rcParams.copy().items() if k != "backend"}


def _misc_savefig(fig, out, stems, **kw):
    """save() writes png+svg+pdf at the figure's own dpi and no bbox; these families set
    their own dpi and a tight bbox and ship a different pair of formats, and changing that
    changes the image. So they keep their original savefig call."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not prepare_publication_figure(fig, (out / stems[0]).with_suffix('')):
        plt.close(fig)
        return
    for path in stems:
        if retained_formats((out / path).with_suffix(''), (Path(path).suffix[1:],)):
            fig.savefig(out / path, **kw)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-ecdf-by-size / fig-posecheck-clash-violin-by-size
# ════════════════════════════════════════════════════════════════════════════════
# Every method on one axis -- the five published baselines PLUS the three we ran here
# (TargetDiff, vanilla VoxBind, Ours v1) -- stratified by generated-molecule heavy-atom
# count. Six panels each: the five size bins plus `all sizes`.
#
# WHY THIS FAMILY DOES NOT USE pose_data(). It reads a different scoring of the same
# molecules, and a different pocket set, for two reasons:
#
#  1. prj-denovo/baselines is not on this box, so the five baselines are read from the
#     per-molecule exports the legacy folder ships (`fig-posecheck/posecheck_<Method>.json`).
#     The loader is checked on every run: the per-bin numbers it reproduces from the
#     exports must match the stored `posecheck_baselines_by_atom_range.json` on all five
#     methods x five bins x three statistics, or the build aborts instead of drawing.
#
#  2. THE RUNS' MAIN SAMPLES DIRECTORIES ARE THE WRONG PROTOCOL. There PoseCheck was
#     scored against the *pocket10 crop*; the five baselines were scored against the whole
#     `*_rec.pdb`. Clashes are receptor-dependent and the crop hides some of them --
#     pooled clash mean 5.25 crop vs 6.27 full for vanilla, 6.49 vs 7.55 for Ours v1,
#     10.39 vs 11.11 for TargetDiff (see posecheck_crop_vs_full.json). Mixing the two
#     would have handed our three a ~1-clash head start. Our three are therefore read from
#     `frozenenc_probes/posecheck_full/`, the whole-receptor re-scoring, whose pooled
#     numbers reproduce `ours_posecheck_full.json` exactly.
#
# POCKET SUBSET. The baselines cover all 79 electron-density pockets; the whole-receptor
# re-scoring of our three now covers all 79 as well: target_71 was missing only because
# scripts 70/71/72 default to `p78_targets.json`, a list that exists for a *docking*
# failure (pdb2pqr30 rejects that pocket's 10 A crop) which PoseCheck never had. Every
# series is still restricted to the pockets both sides share, computed at run time, so no
# method is scored on a pocket another one never saw; the console prints the shared count
# and anything dropped.
#
# THE REFERENCE LIGAND is drawn on the strain figure only. PoseCheck's strain
# (`calculate_strain_energy(mol, num_confs=50)`) never touches the receptor, so the
# crystal ligand's strain is protocol-independent and can be read from our local run. Its
# *clashes* are not: the only local reference scoring is against the crop, so it would sit
# ~0.3-1 clash low next to eight whole-receptor series, and it is left off the violin
# rather than quietly compared. (The whole-receptor reference clash means from the
# baselines' own run live on in the legacy posecheck_all_by_atom_range.json.)
#
# STRAIN IS NOISY BY CONSTRUCTION. `num_confs=50` random conformers means re-scoring the
# same pose gives a slightly different answer: over 7,349 molecules scored twice here the
# median relative difference is 1.0%, p90 8.9%. Fine for an ECDF, not for ranking two
# methods a few percent apart.
#
# COLOURS come from the shared palette, so the key carries across figures; only line
# weight and dash are decided here. Our two are drawn thick and solid; the five published
# baselines are context, so they are thin and each carries its own dash pattern -- with
# nine series on one axis hue alone cannot be the identity channel.
#
# STYLE. This family predates the house style at the top of this file and keeps its own:
# 12 pt type, BLACK spines and tick labels. The violin keeps its panel title -- the size bin
# has nowhere else to live inside the frame -- but the strain ECDF dropped its title on
# request (2026-09-13), so there the bin is named by the file only.

# key -> (label, linewidth, linestyle). Dash is a second identity channel for the five
# published baselines: they are drawn thin, five at a time, and colour alone is not enough
# at that weight.
# The dashes come from METHOD_DASH in 00_core.py now (2026-09-14), so this family and the
# rotatable-bond one cannot drift apart again. AR and Pocket2Mol both carried (5, 2) here,
# which left them separable by hue alone; dash() gives Pocket2Mol its own pattern.
PCSZ_BASELINES = [
    ("AR",         "AR",         1.6, dash("AR")),
    ("Pocket2Mol", "Pocket2Mol", 1.6, dash("Pocket2Mol")),
    ("DiffSBDD",   "DiffSBDD",   1.6, dash("DiffSBDD")),
    ("DecompDiff", "DecompDiff", 1.6, dash("DecompDiff")),
    ("FuncBind",   "FuncBind",   1.6, dash("FuncBind")),
]
# TargetDiff is violet and still dashed: it used to be orange, which put two BASELINES in
# the same hue family as each other, and the dash costs nothing now that the collision is
# gone. These labels are the drawn labels, not the ARMS keys -- they reach pale() through
# the alias table.
PCSZ_LOCAL = [
    ("TargetDiff",    f"{E}/frozenenc_probes/posecheck_full/targetdiff", 1.6,
     dash("TargetDiff")),
    ("VoxBind σ=0.9", f"{E}/frozenenc_probes/posecheck_full/vanilla",    3.4,
     dash("VoxBind σ=0.9")),
    ("CoDE",          f"{E}/frozenenc_probes/posecheck_full/ours_v1",    3.8, dash("CoDE")),
]


def _pcsz_ordered(labels):
    """`labels` in RB_ORDER — the sequence the rotbond grid and the shared legend draw in.

    THE LIST ORDER IS A LOADING ORDER, NOT A DISPLAY ORDER (2026-09-15). PCSZ_BASELINES is
    the five published baselines, which come out of one export, and PCSZ_LOCAL is the three
    arms scored from run trees; concatenating them put DecompDiff and FuncBind AHEAD of
    TargetDiff, which is nobody's row order. Sorting here makes this figure read in the same
    sequence as the rotbond grid and the legend that names it.

    SPELLINGS CANONICALISE FIRST. PCSZ_LOCAL calls it "VoxBind σ=0.9" and RB_ORDER calls it
    "VoxBind", so a raw membership test would report it missing and take the arm out of the
    figure. Anything the table does not name raises, exactly as _rb_order does: a method
    that quietly vanishes from a figure is worse than a failed build.

    RB_ORDER and ALIASES are read at CALL time -- RB_ORDER lives in a part assembled after
    this one, so it cannot be referenced while this module is being defined."""
    rank = {lab: i for i, lab in enumerate(RB_ORDER)}
    canon = {l: ALIASES.get(l, l) for l in labels}
    unknown = [l for l, c in canon.items() if c not in rank]
    if unknown:
        raise SystemExit(f"not in RB_ORDER: {unknown}")
    return sorted(labels, key=lambda l: rank[canon[l]])
# Weight per METHOD, keyed by the canonical name so the other family can read it: the two
# figures are a pair and a series that is thicker in one of them reads as a different series
# (2026-09-13). NOT one flat weight -- our two arms are the subject and the five published
# baselines are context, which is the same reason they are dashed and these are solid.
# ONE WEIGHT FOR EVERY SERIES THAT IS NOT OURS (2026-09-14): the five published baselines,
# TargetDiff and the crystal ligands all draw at 1.6, and dash alone tells them apart. Only
# VoxBind (3.4) and CoDE (3.8) are heavier, which is the whole point of the weight channel.
PCSZ_REF_LW = 1.6
PCSZ_LINE_LW = {**{ALIASES.get(lab, lab): lw for _, lab, lw, _ in PCSZ_BASELINES},
                **{ALIASES.get(lab, lab): lw for lab, _, lw, _ in PCSZ_LOCAL}}

# This family's own furniture: a white ground and BLACK spines/ticks (the house INK/AXIS
# are the warm near-black of the newer figures), with the shared GRID and LEGEND_EDGE.
PCSZ_BG, PCSZ_INK, PCSZ_AXIS = "#ffffff", "#000000", "#000000"
# 80 % of the original 8.6 x 5.6 in canvas both ways, then 1.2x wider again so the lower-right
# key leaves room for the curves' long right tails, then 0.9x both ways (2026-09-13). The
# per-atom all-methods figure reads its aspect from this, so keep scaling both sides together
# unless that one should change too.
# PCSZ_SHRINK pulls the canvas in once more (2026-09-13) WITHOUT touching the type sizes or
# the dpi, so the same ink fills a smaller frame -- that, not a font change, is what makes the
# panel read fuller. PCSZ_LW_SCALE thickens the curves by the same argument: at this canvas
# the nine series were drawn for a frame a fifth wider. The trailing 1.1x is WIDTH ONLY
# (2026-09-13): the lower-right key and the curves' right tails were tight against each other
# once the canvas came in.
PCSZ_SHRINK, PCSZ_LW_SCALE = 0.85, 1.3
PCSZ_ECDF_SIZE = (8.6 * 0.8 * 1.2 * 0.9 * PCSZ_SHRINK * 1.1,
                  5.6 * 0.8 * 0.9 * PCSZ_SHRINK * 0.9)   # trailing 0.9 is HEIGHT only
# Height-only adjustment for the cumulative-probability / median-strain pair.
PCSZ_STRAIN_HEIGHT_SCALE = 1.1 * 0.9 * 0.9  # Two successive 10% height reductions.
PCSZ_STRAIN_SIZE = (PCSZ_ECDF_SIZE[0], PCSZ_ECDF_SIZE[1] * PCSZ_STRAIN_HEIGHT_SCALE)
# Taller main ECDF and median-strain panels; other size bins and comparison panels keep their size.
# Use the prior exported heights (pt), including their outer margin repair,
# so the delivered PDFs themselves are exactly 1.1x taller.
PCSZ_MAIN_STRAIN_SIZE = (PCSZ_STRAIN_SIZE[0], 198.907915 / 72 * 1.1)
PCSZ_MAIN_ECDF_SIZE = (PCSZ_STRAIN_SIZE[0], 201.211915 / 72 * 1.1)
# Fixed axes bounds for the landscape clash / redocking-median comparison.
# Keep the same margins in both panels.
PCSZ_COMPARISON_MARGINS = dict(left=0.16, right=0.98, bottom=0.26, top=0.94)
# Axis-title size, tick-label size and axis (spine + tick) weight of those panels, as multiples
# of the shared publication_style TITLE_SIZE / TICK_SIZE / AXIS_WIDTH.
# MATCHED TO ONE fig-jsd-metric-summary PANEL, BY AXES HEIGHT (2026-09-24, on request).
# Title, tick-label and axis weight each stand in the same proportion to the plot area's
# height as in a metric-summary panel (axes 3.55 x 2.24 in after export; titles 19.36 pt =
# 0.120 of that height). Height, not width: these panels are ~2.8:1 against its 1.6:1, so
# width-matching would crowd them vertically. One factor for all four, off their mean axes
# height (the ECDF and per-atom panels are taller than the clash and redocking ones), so
# they still share one type size. An earlier round matched PRINTED size instead (the whole
# 22.66 in figure at full width vs these at half), which left the type small in-panel.
PCSZ_METRIC_AXES_H = 2.24
PCSZ_COMPARISON_AXES_H = (np.mean([PCSZ_MAIN_ECDF_SIZE[1], PCSZ_MAIN_STRAIN_SIZE[1],
                                   PCSZ_STRAIN_SIZE[1], PCSZ_STRAIN_SIZE[1]])
                          * (PCSZ_COMPARISON_MARGINS["top"] - PCSZ_COMPARISON_MARGINS["bottom"]))
PCSZ_PANEL_MATCH = PCSZ_COMPARISON_AXES_H / PCSZ_METRIC_AXES_H
PCSZ_COMPARISON_TITLE_SCALE = PCSZ_PANEL_MATCH
PCSZ_COMPARISON_TICK_SCALE = PCSZ_PANEL_MATCH
PCSZ_COMPARISON_AXIS_SCALE = PCSZ_PANEL_MATCH
# Spine-to-title distance in points, the same in all four panels (2026-09-24). Sized to
# clear the widest tick column among them (the per-atom log axis) with room to spare.
PCSZ_YTITLE_GAP = 50
PCSZ_XTITLE_GAP = 36
# Canvas left below the lowest ink when a figure's empty bottom margin is trimmed (inches).
PCSZ_TRIM_PAD = 0.04
# Canvas kept below the axes of a panel without an x title (inches): room for the
# redocking y title, whose lower end reaches below its axes.
PCSZ_NO_XTITLE_BOTTOM = 0.16
# Axis-name size for the ECDF, and for the per-atom all-methods figure drawn in its style.
# 1.4x the family's original 11.5 pt (2026-09-13).
PCSZ_LABEL_FS = 11.5 * 1.4
# The y name's gap off its tick labels, DOUBLED (2026-09-14). Measured off the drawn PNGs the
# matplotlib default (labelpad 4.0) buys 5.5 pt here and 6.8 pt on the per-atom panel; a pad of
# 10 is twice that on both. Shared, so the pair keeps one left margin.
PCSZ_YPAD = 10
PCSZ_RC = {
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": PCSZ_INK, "axes.labelcolor": PCSZ_INK,
    "xtick.color": PCSZ_AXIS, "ytick.color": PCSZ_AXIS,
    "svg.fonttype": "none",          # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,              # TrueType in the PDF, not matplotlib's Type 3
    # save() takes no dpi/bbox arguments, and this family is drawn at 170 dpi and cropped
    # to its own ink rather than sized to the figure. Both ride in on the savefig.* rc
    # keys, which Figure.savefig and print_figure fall back to when the kwargs are absent.
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
}


def _pcsz_style(ax, *, grid_axis="both"):
    """The vina figures' axis furniture: black spines and outward ticks, dotted grid.

    `grid_axis` is "y" for the violin -- its x is categorical, so a vertical rule through
    every category centre is a picket fence, not a reading aid."""
    ax.set_facecolor(PCSZ_BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PCSZ_AXIS)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(labelsize=11, direction="out", length=3.5, width=1.1, pad=4,
                   colors=PCSZ_AXIS)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.9, ls=(0, (1, 2.6)))
    ax.set_axisbelow(True)


def _pcsz_comparison_layout(fig, ax, *, xunit=None, yunit=None, tick_scale=None):
    """Identical physical axes and typography for the four PoseCheck comparison panels.

    `xunit` / `yunit` (2026-09-24, on request) put the unit INSIDE the frame at that axis's
    far end -- x at the bottom-right, y at the top-left -- instead of in the axis title,
    at tick-label size. Both corners are empty in every panel that has a unit.

    The axis titles sit a FIXED distance from their spine (PCSZ_YTITLE_GAP /
    PCSZ_XTITLE_GAP), not a labelpad off the tick labels, so the four panels -- whose tick
    labels differ in width -- carry their titles at the same place. `tick_scale` overrides
    PCSZ_COMPARISON_TICK_SCALE for one panel (the ECDF's larger ticks)."""
    fig.set_layout_engine("none")
    fig.subplots_adjust(**PCSZ_COMPARISON_MARGINS)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color(PCSZ_AXIS)
        ax.spines[side].set_zorder(6)
    ax.tick_params(which="major", labelsize=11, direction="out", length=3.5,
                   width=1.1, pad=4, colors=PCSZ_AXIS)
    ax.yaxis.label.set_fontsize(PCSZ_LABEL_FS)
    ax.yaxis.label.set_color(PCSZ_INK)
    # Let Matplotlib measure major/minor tick labels before positioning the title.
    ax.yaxis.labelpad = max(ax.yaxis.labelpad, PCSZ_YPAD)
    # Export-time overrides read by publication_style.apply (2026-09-24, on request): these
    # four panels take 0.8x the shared axis-title size, half its tick-label size and 0.8x the
    # shared axis weight.
    fig._publication_title_scale = PCSZ_COMPARISON_TITLE_SCALE
    fig._publication_tick_scale = (PCSZ_COMPARISON_TICK_SCALE if tick_scale is None
                                   else tick_scale)
    fig._publication_axis_scale = PCSZ_COMPARISON_AXIS_SCALE
    # Title position in points off the spine; a manual label position also makes matplotlib
    # (and publication_style) ignore labelpad from here on.
    to_pt = lambda dx, dy: matplotlib.transforms.ScaledTranslation(dx / 72, dy / 72,
                                                                   fig.dpi_scale_trans)
    ax.yaxis.set_label_coords(0, 0.5, transform=ax.transAxes + to_pt(-PCSZ_YTITLE_GAP, 0))
    ax.xaxis.set_label_coords(0.5, 0, transform=ax.transAxes + to_pt(0, -PCSZ_XTITLE_GAP))
    # No x title (clash, redocking): the bottom margin held for one is empty, so the export
    # trims the canvas below the lowest ink (2026-09-24, on request). Axes keep their size.
    # Both keep PCSZ_NO_XTITLE_BOTTOM below the axes, so the pair stays the same height.
    if not ax.get_xlabel():
        fig._publication_trim_bottom = PCSZ_TRIM_PAD
        fig._publication_trim_keep = PCSZ_NO_XTITLE_BOTTOM
    unit_kw = dict(transform=ax.transAxes, fontsize=TICK_SIZE * PCSZ_COMPARISON_TICK_SCALE,
                   color=PCSZ_INK, zorder=7,
                   bbox=dict(facecolor=PCSZ_BG, edgecolor="none", pad=1.0))
    if xunit:
        ax.text(0.99, 0.03, xunit, ha="right", va="bottom", **unit_kw)
    if yunit:
        ax.text(0.015, 0.98, yunit, ha="left", va="top", **unit_kw)
# handlelength 1.6 showed barely one repeat of a dash pattern, so the five baselines' second
# identity channel was unreadable in the key; 2.8 fits two to three repeats of every pattern.
PCSZ_KEY_KW = dict(fontsize=9.5, handlelength=2.8, handletextpad=0.5, labelspacing=0.32,
                   borderpad=0.4, columnspacing=1.2)
# The frame's face, translucent so a curve running under the key is still followable, still
# opaque enough to keep the dotted grid out of the text (2026-09-13). Set on the FACE, not as
# the artist's alpha, which would fade the grey rule with it.
PCSZ_KEY_FACE = (1.0, 1.0, 1.0, DATA_ALPHA)


# ── fig-posecheck-clash-box-by-method ──────────────────────────────────────────────────
# The rotatable-bond grid's box format on ONE panel: method along x, clashes up y. There is
# nothing to facet here -- this is every pose the method made, at every ligand size -- so the
# 3x3 block collapses to a single axes and the bins live in the violin above.
# THE FILL IS THE METHOD, not its median (2026-09-14). The grid needed a colourmap because a
# method owns nine panels there and the fill was the only place its median could be read; on
# one axes the eight medians are side by side already, so the fill goes back to carrying
# identity -- the same pale() tint the violin and the ECDF give that method -- and the
# colourbar that encoded the median comes off.
# 0.8x the grid's 2.0 (2026-09-14): a box here is far wider than one in a nine-panel grid,
# and the same weight read heavier across it.
PCSZ_CBOX_LAYOUT = "vertical"  # Canvas layout; set by --clash-layout.


# ════════════════════════════════════════════════════════════════════════════════
# fig-posecheck-strain-legend — the per-atom all-methods strain key, as its own image
# ════════════════════════════════════════════════════════════════════════════════
# Standalone key: one horizontal row or one vertical column, reference first.
# Both layouts use the shared method order, colours, and line styles.
#
# THE SWATCHES ARE THE PANEL'S. The dash comes from the shared METHOD_DASH table through
# dash(), the WEIGHT from the ECDF family (PCSZ_LINE_LW x PCSZ_LW_SCALE) and the colour
# through pale(), in RB_ORDER. PCSZ_LINE_LW is read at call time: it lives in a part
# assembled after this one.
#
# THIS KEY USED TO LIE. It read the rotatable-bond family's own dash table, which had drifted
# a whole method out of step with the ECDF's -- so the legend told a reader that Pocket2Mol
# was the dotted line when the panel drew DiffSBDD that way. One table now feeds both.
PLEG_LAYOUT = "horizontal"  # --legend-layout selects the standalone key layout.
# THE ROW ORDER OF 260827/table_drug_design.tex, which is canonical for this section, except
# that VoxBind is pulled down next to CoDE so the model our arm modifies sits immediately
# before it and the two read as a pair. Legends, panels and exports all read from this, so a
# method cannot sit in one order in the legend and another in the share panels.
RB_ORDER = list(METHOD_ORDER)
# The two lines are laid out as two legends and then wrapped in one frame; this is the gap
# left between them, in figure fractions, INSIDE that shared frame.
VPA_KEY_ROW_GAP = 0.002


# ════════════════════════════════════════════════════════════════════════════════
# viewers — the interactive pages the 3D panels are captured from
# ════════════════════════════════════════════════════════════════════════════════
# A 3D panel (fig-qual, fig-overview's pocket renders) is posed and captured by hand in a
# browser viewer; no script can choose the camera for you. The viewers register here so
# `draw.py --viewer <name>` starts one. What CAN be rebuilt from saved captures is a
# figure with capture=True: drawn by `--with-3d`, never by plain `--all`.
VIEWERS = {}          # name -> function(host, port) -> None (blocks while serving)


def viewer(name):
    def deco(fn):
        VIEWERS[name] = fn
        return fn
    return deco


# ════════════════════════════════════════════════════════════════════════════════
# the parts — one module (or package) per figure folder, imported to register
# ════════════════════════════════════════════════════════════════════════════════
def _load_parts():
    """Import every _parts/fig_* module and package; importing one registers its figures."""
    import importlib
    for path in sorted((HERE / "_parts").iterdir()):
        if path.name.startswith("fig_") and (path.suffix == ".py" or (path / "__init__.py").is_file()):
            importlib.import_module(f"_parts.{path.stem}")


_load_parts()


# ════════════════════════════════════════════════════════════════════════════════
# the runner
# ════════════════════════════════════════════════════════════════════════════════
def _by_eval():
    """{eval: [fig_id, ...]} in registration order — `fig-<eval>-<name>` splits on the
    SECOND hyphen, so a figure name may itself contain hyphens."""
    out = collections.OrderedDict()
    for fid in FIGURES:
        ev = fid.split("-")[1]
        out.setdefault(ev, []).append(fid)
    return out


def _list():
    print(f"{len(FIGURES)} figures in {len(_by_eval())} evaluations "
          f"-> {OUT_ROOT.relative_to(REPO)}/fig-<folder>/\n")
    for ev, ids in _by_eval().items():
        folders = sorted({figure_folder(f) for f in ids})
        print(f"  {ev}  ->  {', '.join(f + '/' for f in folders)}")
        for fid in ids:
            doc = (FIGURES[fid].__doc__ or "").strip().splitlines()
            here = figure_folder(fid)
            tag = f"[{here.split('/')[-1]}] " if len(folders) > 1 else ""
            tag += "[--with-3d] " if getattr(FIGURES[fid], "capture", False) else ""
            print(f"    {fid:44s} {tag}{doc[0] if doc else ''}")
        print()
    if VIEWERS:
        print(f"  viewers (draw.py --viewer <name>): {', '.join(sorted(VIEWERS))}\n")


def _resolve(names):
    """Figure ids for the names asked for. A name that is an EVALUATION expands to every
    figure in it, so `draw.py posecheck` is the whole PoseCheck set. An unknown name is a
    hard error listing the near misses -- silently drawing nothing is the failure mode
    this exists to prevent."""
    by_eval = _by_eval()
    out = []
    for n in names:
        # Older commands now draw the replacement box plot.
        if n == "fig-consistency-rigid-violin":
            n = "fig-consistency-rigid-box"
        if n in FIGURES:
            out.append(n)
        elif n in by_eval:
            out.extend(by_eval[n])
        else:
            near = [f for f in FIGURES if n in f] or [f"{e} (whole evaluation)"
                                                      for e in by_eval if n in e]
            raise SystemExit(f"unknown figure {n!r}"
                             + (f"\n  did you mean: {', '.join(near)}" if near else
                                f"\n  --list shows all {len(FIGURES)}"))
    seen, uniq = set(), []
    for f in out:                       # keep order, drop repeats
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="draw.py", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A name may be a figure id or a whole evaluation "
               f"({', '.join(_by_eval())}).")
    ap.add_argument("names", nargs="*", help="figure ids and/or evaluation names")
    ap.add_argument("-a", "--all", action="store_true",
                    help="draw every figure (those with capture=True only with --with-3d)")
    ap.add_argument("-l", "--list", action="store_true", help="list the figures and exit")
    ap.add_argument("-f", "--formats", default="png,pdf",
                    help="comma-separated: png,pdf (default both)")
    ap.add_argument("-o", "--out", default=str(OUT_ROOT),
                    help="output root; each figure lands in <root>/fig-<folder>/ "
                         f"(default {OUT_ROOT})")
    ap.add_argument("-k", "--keep-going", action="store_true",
                    help="carry on after a figure fails, and report at the end")
    ap.add_argument("--clash-layout", choices=("horizontal", "vertical"), default="vertical",
                    help="clash box-plot canvas: horizontal = methods along x; vertical = rows (default)")
    ap.add_argument("--legend-layout", choices=("horizontal", "vertical"), default="horizontal",
                    help="standalone strain legend: single horizontal row (default) or a single vertical column")
    ap.add_argument('--existing-only', action='store_true',
                    help='redraw only figure files already present below the output root '
                         '(a removed figure stays removed)')
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw pictures only: exported CSV/JSON/notes are left as they are")
    ap.add_argument("--with-3d", action="store_true",
                    help="with --all, also rebuild the figures laid out from saved 3D captures")
    ap.add_argument("--viewer", metavar="NAME",
                    help="start an interactive 3D viewer instead of drawing (see --list)")
    ap.add_argument("--host", default="127.0.0.1", help="--viewer: bind address")
    ap.add_argument("--port", type=int, default=None, help="--viewer: port (default per viewer)")
    ap.add_argument("--inventory", metavar="ZIP", type=Path,
                    help="map the paper archive's \\includegraphics to local outputs, then sync")
    ap.add_argument("--tex-main", default="ICLR2027/0_main.tex",
                    help="--inventory: the main .tex inside the archive")
    ap.add_argument("--sync-downloads", action="store_true",
                    help="copy every mapped figure into _download/ and rewrite its lists")
    ap.add_argument("--ids", action="store_true",
                    help="print the figure ids the names resolve to (all if none), one per line")
    ap.add_argument("--evals", action="store_true", help="print the evaluation names, one per line")
    ap.add_argument("--jobs", action="store_true",
                    help="print the batches draw_all.sh runs in parallel, one per line: an "
                         "evaluation's figures together, each isolate=True figure on its own "
                         "(capture=True figures only with --with-3d)")
    args = ap.parse_args(argv)

    global PCSZ_CBOX_LAYOUT, PLEG_LAYOUT
    PCSZ_CBOX_LAYOUT, PLEG_LAYOUT = args.clash_layout, args.legend_layout

    if args.viewer:
        if args.viewer not in VIEWERS:
            raise SystemExit(f"unknown viewer {args.viewer!r}; have: {', '.join(sorted(VIEWERS))}")
        VIEWERS[args.viewer](args.host, args.port)
        return 0
    if args.inventory or args.sync_downloads:
        if args.inventory:
            build_manifest(args.inventory, args.tex_main)
        if not MANIFEST.exists():
            raise SystemExit(f"no download map at {MANIFEST} -- run --inventory <paper.zip> first")
        sync_all()
        return 0

    if args.evals:
        print("\n".join(_by_eval()))
        return 0
    if args.jobs:
        for ev, fids in _by_eval().items():
            keep = [f for f in fids if args.with_3d or not getattr(FIGURES[f], "capture", False)]
            together = [f for f in keep if not getattr(FIGURES[f], "isolate", False)]
            if together:
                print(" ".join(together))
            for f in keep:
                if getattr(FIGURES[f], "isolate", False):
                    print(f)
        return 0
    if args.ids:
        print("\n".join(_resolve(args.names) if args.names else FIGURES))
        return 0
    if args.list or (not args.names and not args.all):
        _list()
        return 0

    global _FORMATS, FIGURES_ONLY
    _FORMATS = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    bad = [f for f in _FORMATS if f not in ("png", "pdf")]
    if bad:
        raise SystemExit(f"unsupported format(s): {', '.join(bad)}")
    FIGURES_ONLY = args.figures_only

    if args.all:
        ids = [f for f in FIGURES if args.with_3d or not getattr(FIGURES[f], "capture", False)]
    else:
        ids = _resolve(args.names)
    root = Path(args.out)
    if args.existing_only:
        # Historical layout snapshots are archives: never redrawn, even though they exist.
        inventory = [str(p.resolve()) for p in root.rglob('*') if p.suffix in ('.png', '.pdf')
                     and 'motivation-layouts-20260921' not in p.parts]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='voxbind-retained-', delete=False) as f:
            json.dump(inventory, f)
            os.environ['VOXBIND_FIGURE_RETAINED'] = f.name
    use_style()
    print(f"drawing {len(ids)} figure(s) as {'+'.join(_FORMATS)} -> {root}/\n")

    failed, skipped, t_all = [], [], time.time()
    for i, fid in enumerate(ids, 1):
        # The folder is the EVALUATION, not the figure: everything answering the same
        # question sits together, and the shared CSVs two figures both export (the pose
        # per-atom table, the interaction table) land once instead of once per figure.
        out = root / figure_folder(fid)
        if args.existing_only and not any(p.suffix in (".png", ".pdf") for p in out.glob("*")):
            skipped.append(fid)
            print(f"[{i}/{len(ids)}] {fid}: skipped (no retained figures)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(ids)}] {fid}  -> {out.name}/")
        t0 = time.time()
        if getattr(FIGURES[fid], "isolate", False) and len(ids) > 1:
            child = [sys.executable, str(Path(__file__).resolve()), fid, "--formats", args.formats,
                     "--out", args.out, "--clash-layout", args.clash_layout,
                     "--legend-layout", args.legend_layout]
            child += ["--figures-only"] if args.figures_only else []
            child += ["--existing-only"] if args.existing_only else []
            import subprocess
            code = subprocess.call(child)
            if code:
                if not args.keep_going:
                    raise SystemExit(code)
                failed.append((fid, f"exit {code} (own process)"))
                print(f"  FAILED in its own process (exit {code})")
                continue
            print(f"  done in {time.time() - t0:.1f}s (own process)")
            continue
        try:
            FIGURES[fid](out)
        except Exception as exc:                                  # noqa: BLE001
            if not args.keep_going:
                raise
            failed.append((fid, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED {type(exc).__name__}: {exc}")
            continue
        print(f"  done in {time.time() - t0:.1f}s")

    print(f"\n{len(ids) - len(failed) - len(skipped)}/{len(ids)} drawn, {len(skipped)} skipped in {time.time() - t_all:.1f}s")
    if failed:
        print(f"{len(failed)} failed:")
        for fid, why in failed:
            print(f"  {fid}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

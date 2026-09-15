"""Slice O.13 — rasterize the Kick wordmark SVG into a PNG once.

The CSS preview loads `/static/kick-logo.svg` directly so it gets
pixel-perfect Kick brand letterforms. The ffmpeg burn previously
rendered the word `KICK` via `drawtext` using a system font (Arial),
producing a visibly different glyph shape — operators saw a
"preview ≠ export" mismatch on every clip.

This module bridges the gap by rasterizing the same SVG path data
into a PNG (axis-aligned rectangles only, no curves, no anti-alias
needed beyond Pillow's default polygon fill), then ffmpeg can
`overlay=` the PNG onto the banner. Result: preview and burn use the
identical letterforms.

The SVG happens to be 4 simple closed polygons (K-I-C-K) on a 933×300
canvas. We don't need a full SVG parser — we hardcode the four polygons
the same way the SVG file does.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock

# Source SVG (kept in sync with /static/kick-logo.svg). Each Mxy ... Z
# is one filled polygon. We translate H/V commands to absolute
# coordinates and feed Pillow's polygon fill.
_KICK_SVG_PATH = (
    "M0 0H100V66.6667H133.333V33.3333H166.667V0H266.667V100H233.333"
    "V133.333H200V166.667H233.333V200H266.667V300H166.667V266.667H133.333"
    "V233.333H100V300H0V0Z "
    "M666.667 0H766.667V66.6667H800V33.3333H833.333V0H933.333V100H900"
    "V133.333H866.667V166.667H900V200H933.333V300H833.333V266.667H800"
    "V233.333H766.667V300H666.667V0Z "
    "M300 0H400V300H300V0Z "
    "M533.333 0H466.667V33.3333H433.333V266.667H466.667V300H533.333"
    "H633.333V200H533.333V100H633.333V0H533.333Z"
)

# Rendering resolution. The SVG viewBox is 933×300 (aspect 3.11:1).
# We render at 4× height = 1200 → 4× width = 3733, gives plenty of
# crispness when ffmpeg overlay scales it back to the band height.
_RENDER_SCALE = 4.0
_VIEWBOX_W = 933.333
_VIEWBOX_H = 300.0

# Lime color matching the SVG `fill="#53FC18"`. Stored as RGBA so the
# rest of the frame stays transparent — ffmpeg `overlay=` honors the
# alpha channel.
_KICK_GREEN: tuple[int, int, int, int] = (0x53, 0xFC, 0x18, 0xFF)

_lock = Lock()
_cached_png_path: Path | None = None


def _parse_svg_polygons(path_d: str) -> list[list[tuple[float, float]]]:
    """Walk the path data and produce a list of polygons (each a list
    of (x, y) vertices). Handles M / H / V / Z only — that's all this
    SVG uses. Coordinates are kept in SVG units; the caller scales.
    """
    tokens = re.findall(r"[MHVZmhvz]|-?\d+(?:\.\d+)?", path_d)
    polygons: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    x = y = 0.0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("M", "m"):
            # New subpath. If we were building one without a Z, close it.
            if current:
                polygons.append(current)
            i += 1
            x = float(tokens[i]); i += 1
            y = float(tokens[i]); i += 1
            current = [(x, y)]
        elif t in ("H", "h"):
            i += 1
            x = float(tokens[i]); i += 1
            current.append((x, y))
        elif t in ("V", "v"):
            i += 1
            y = float(tokens[i]); i += 1
            current.append((x, y))
        elif t in ("Z", "z"):
            i += 1
            if current:
                polygons.append(current)
                current = []
        else:
            # Bare coords after M get treated as L by SVG spec, but
            # the Kick path doesn't use that. Skip defensively.
            i += 1
    if current:
        polygons.append(current)
    return polygons


def _render_png(target_path: Path) -> None:
    """Render the Kick wordmark to `target_path` as a transparent PNG."""
    from PIL import Image, ImageDraw

    w = int(_VIEWBOX_W * _RENDER_SCALE)
    h = int(_VIEWBOX_H * _RENDER_SCALE)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for poly in _parse_svg_polygons(_KICK_SVG_PATH):
        # Scale SVG coords → pixel coords.
        scaled = [(px * _RENDER_SCALE, py * _RENDER_SCALE) for px, py in poly]
        draw.polygon(scaled, fill=_KICK_GREEN)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: the cached file is trusted by existence alone across boots,
    # so a crash mid-save must never leave a truncated PNG on the real name
    # (ffmpeg would fail to decode it on every burn until manually deleted).
    part_path = target_path.with_suffix(".part.png")
    img.save(part_path, format="PNG")
    os.replace(part_path, target_path)


def kick_logo_png_path() -> Path:
    """Return the on-disk path to a rasterized Kick wordmark PNG.

    Idempotent: the first call renders + caches, every subsequent call
    returns the same path. Cache lives next to the source SVG inside
    `chalybclip/api/static/` so the editor preview and the ffmpeg burn
    can both reference it (preview by URL, burn by absolute path).

    Render is conservative: a single PIL polygon-fill per glyph, no
    anti-alias needed (axis-aligned rects only). Takes <50ms on a
    modern machine and only happens once per process boot.
    """
    global _cached_png_path
    if _cached_png_path is not None and _cached_png_path.exists():
        return _cached_png_path
    with _lock:
        if _cached_png_path is not None and _cached_png_path.exists():
            return _cached_png_path
        here = Path(__file__).resolve()
        # chalybclip/clip/kick_logo_png.py → ../api/static/kick-logo.png
        static_dir = here.parent.parent / "api" / "static"
        png_path = static_dir / "kick-logo.png"
        if not png_path.exists():
            _render_png(png_path)
        _cached_png_path = png_path
        return png_path


__all__ = ["kick_logo_png_path"]

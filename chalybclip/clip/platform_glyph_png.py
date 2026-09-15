"""Rasterize simple white monochrome platform glyphs into PNGs once.

Sibling of `kick_logo_png` — but where Kick gets its full brand wordmark
(the repost-page variant), the OTHER source platforms (YouTube, Twitch,
X/Twitter) credit their source with a branded color band + a small white
monochrome glyph instead of reproducing each platform's exact trademarked
logo. A play triangle reads as YouTube, a chat bubble as Twitch, an X as
X — recognizable at a glance without shipping third-party brand art.

Each glyph is a handful of PIL primitives on a transparent square canvas,
rendered once per process and cached on disk next to the Kick wordmark in
`chalybclip/api/static/` so the editor preview (by URL) and the ffmpeg burn
(by absolute path) read the identical file.

The burn composites the glyph the same way the Kick wordmark does — via a
`__PNG_OVERLAY__:key=<platform>` sentinel that `overlay_burn.burn_overlays`
swaps for a second ffmpeg input + `overlay=` in a `-filter_complex` graph.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

# White, fully opaque. Everything else on the canvas stays transparent so
# ffmpeg `overlay=` (and the CSS background-image) let the colored band
# show through — including any knocked-out interior (e.g. the YouTube
# play triangle).
_WHITE: tuple[int, int, int, int] = (0xFF, 0xFF, 0xFF, 0xFF)
_CLEAR: tuple[int, int, int, int] = (0, 0, 0, 0)

# Square render canvas. ffmpeg scales the PNG down to the band glyph size,
# so render big enough that the downscale stays crisp.
_CANVAS = 480

# Which source platforms get a monochrome glyph. Kick is intentionally
# absent — it uses its full wordmark via `kick_logo_png`. `twitter`
# normalizes to `x` (same brand, same mark) before this lookup.
_SUPPORTED = frozenset({"youtube", "twitch", "x"})

_lock = Lock()
_cached: dict[str, Path] = {}


def _draw_youtube(draw: object) -> None:
    """White rounded-rect play button with a knocked-out play triangle —
    the triangle is transparent so the red band shows through it."""
    from PIL import ImageDraw

    assert isinstance(draw, ImageDraw.ImageDraw)
    c = _CANVAS
    # Rounded "button" body.
    draw.rounded_rectangle(
        [c * 0.18, c * 0.28, c * 0.82, c * 0.72],
        radius=int(c * 0.16),
        fill=_WHITE,
    )
    # Play triangle knocked out of the white body (points right).
    draw.polygon(
        [(c * 0.44, c * 0.40), (c * 0.44, c * 0.60), (c * 0.62, c * 0.50)],
        fill=_CLEAR,
    )


def _draw_twitch(draw: object) -> None:
    """White chat/speech bubble — rounded body + a down-left tail."""
    from PIL import ImageDraw

    assert isinstance(draw, ImageDraw.ImageDraw)
    c = _CANVAS
    draw.rounded_rectangle(
        [c * 0.20, c * 0.24, c * 0.80, c * 0.60],
        radius=int(c * 0.12),
        fill=_WHITE,
    )
    # Tail dropping from the lower-left of the bubble.
    draw.polygon(
        [(c * 0.30, c * 0.56), (c * 0.30, c * 0.78), (c * 0.48, c * 0.56)],
        fill=_WHITE,
    )


def _draw_x(draw: object) -> None:
    """White X — two thick crossing diagonal strokes."""
    from PIL import ImageDraw

    assert isinstance(draw, ImageDraw.ImageDraw)
    c = _CANVAS
    width = int(c * 0.15)
    draw.line([(c * 0.26, c * 0.26), (c * 0.74, c * 0.74)], fill=_WHITE, width=width)
    draw.line([(c * 0.74, c * 0.26), (c * 0.26, c * 0.74)], fill=_WHITE, width=width)


_DRAWERS = {
    "youtube": _draw_youtube,
    "twitch": _draw_twitch,
    "x": _draw_x,
}


def _render_png(platform: str, target_path: Path) -> None:
    """Render one platform's glyph to `target_path` as a transparent PNG."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (_CANVAS, _CANVAS), _CLEAR)
    draw = ImageDraw.Draw(img)
    _DRAWERS[platform](draw)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path, format="PNG")


def platform_glyph_png_path(platform: str) -> Path | None:
    """Return the on-disk path to a cached white glyph PNG for `platform`,
    or None when the platform has no glyph (kick uses its wordmark; upload
    / unknown get no logo overlay).

    Idempotent: first call per platform renders + caches, subsequent calls
    return the same path. The cache lives in `chalybclip/api/static/` so the
    editor preview and the ffmpeg burn reference the identical file.
    """
    key = (platform or "").strip().lower()
    if key == "twitter":
        key = "x"  # same brand + mark as X
    if key not in _SUPPORTED:
        return None
    cached = _cached.get(key)
    if cached is not None and cached.exists():
        return cached
    with _lock:
        cached = _cached.get(key)
        if cached is not None and cached.exists():
            return cached
        here = Path(__file__).resolve()
        # chalybclip/clip/platform_glyph_png.py -> ../api/static/glyph-<p>.png
        static_dir = here.parent.parent / "api" / "static"
        png_path = static_dir / f"glyph-{key}.png"
        if not png_path.exists():
            _render_png(key, png_path)
        _cached[key] = png_path
        return png_path


__all__ = ["platform_glyph_png_path"]

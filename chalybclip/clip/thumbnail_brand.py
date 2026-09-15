"""Brand-kit thumbnail compositing — voice-markers spec §3.3.b / 3.5 step 9.

Pipeline:
    raw frame (JPEG, from `pick_thumbnail`) → center-crop for aspect →
    Pillow overlay (brand-color frame + handle text) → JPEG variant

Three aspect ratios per clip:
    - 16:9  (1280x720)  — YouTube horizontal, X video, web embed
    - 9:16  (1080x1920) — TikTok / Shorts / Reels (matches the clip)
    - 1:1   (1080x1080) — Instagram feed, LinkedIn

Each variant is written to `<clip_dir>/thumb_<aspect>.jpg` alongside
the existing `thumbnail.jpg` (raw frame). The raw stays — it's what
the dashboard preview uses; the branded variants are what the
publisher adapters upload as the cover image.

Failures are non-fatal — if Pillow can't decode the source JPEG (e.g.
a placeholder test stub), the function logs + returns an empty list
and the clip ships without branded thumbnails. The renderer's existing
fallback is "no thumbnail," not "raise."
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal, NamedTuple

import structlog
from PIL import Image, ImageDraw, ImageFont

_log = structlog.get_logger(__name__)


# ---- aspect-ratio variants ----


ThumbnailAspect = Literal["16x9", "9x16", "1x1"]


class _Variant(NamedTuple):
    """One target rendering size."""

    aspect: ThumbnailAspect
    width: int
    height: int
    filename: str


_VARIANTS: tuple[_Variant, ...] = (
    _Variant("16x9", 1280, 720, "thumb_16x9.jpg"),
    _Variant("9x16", 1080, 1920, "thumb_9x16.jpg"),
    _Variant("1x1", 1080, 1080, "thumb_1x1.jpg"),
)


# ---- color helpers ----


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """`#FF3366` / `FF3366` / `#f36` → `(r, g, b)`. Falls back to black on
    junk input — branded thumbnails should never crash the clip step."""
    h = hex_color.lstrip("#").strip()
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (0, 0, 0)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return (0, 0, 0)


def _readable_text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text for `bg_rgb` based on perceived luminance.

    Used when the brand kit's text_color isn't supplied or when we're
    drawing onto a banner that uses the brand's primary color (which may
    be light or dark)."""
    r, g, b = bg_rgb
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return (0, 0, 0) if luminance > 0.6 else (255, 255, 255)


# ---- font resolution ----


def _load_handle_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best-effort: a system sans-serif at `size_px`, falling back to
    Pillow's bitmap default when no TTF is reachable.

    Return type is a union because `ImageFont.truetype` yields a
    `FreeTypeFont` (scalable, vector) while the fallback `load_default`
    returns a bitmap `ImageFont` — Pillow's stubs split them.

    We don't try to honor the brand-kit's font_family here — it's already
    burned into the clip via ffmpeg drawtext (slice D.1). The thumbnail
    just needs a legible sans face; matching the in-clip font is a
    polish task for slice F.2."""
    candidates = [
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        # macOS
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            continue
    # Pillow's stock bitmap font as the last resort. Worse-looking, but
    # the publisher upload still succeeds.
    return ImageFont.load_default()


# ---- compositing primitives ----


def _crop_to_aspect(img: Image.Image, *, target_w: int, target_h: int) -> Image.Image:
    """Center-crop `img` to the target aspect ratio, then resize.

    We crop before resizing so the output keeps detail (no letterboxing).
    Aspect-mismatch slack is taken off the longer side."""
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        # Source is wider — crop horizontal slack.
        new_w = int(src_h * target_ratio)
        x0 = (src_w - new_w) // 2
        cropped = img.crop((x0, 0, x0 + new_w, src_h))
    else:
        # Source is taller — crop vertical slack.
        new_h = int(src_w / target_ratio)
        y0 = (src_h - new_h) // 2
        cropped = img.crop((0, y0, src_w, y0 + new_h))
    # Pillow ≥ 10 moved LANCZOS under Resampling; older releases still
    # expose it at module level. Use getattr to stay compatible with both
    # without dragging in a version probe — mypy can't infer the attr
    # path on older stubs.
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return cropped.resize((target_w, target_h), resample)


def _draw_handle_banner(
    img: Image.Image,
    *,
    handle: str,
    primary_color: str,
    text_color: str | None,
) -> None:
    """Draw a bottom banner with the handle text. Mutates `img` in place.

    Banner height = 12% of image height; text is centered vertically
    within. Uses the brand-kit primary color as the band background and
    the brand-kit text color (or auto-readable) as the text fill."""
    if not handle:
        return
    w, h = img.size
    banner_h = max(48, int(h * 0.12))
    bg_rgb = _hex_to_rgb(primary_color)
    fg_rgb = _hex_to_rgb(text_color) if text_color else _readable_text_color(bg_rgb)
    draw = ImageDraw.Draw(img, mode="RGBA")
    # Semi-transparent so the underlying frame still reads through.
    draw.rectangle(
        ((0, h - banner_h), (w, h)),
        fill=(bg_rgb[0], bg_rgb[1], bg_rgb[2], 230),
    )
    font = _load_handle_font(max(24, int(banner_h * 0.45)))
    # Pillow ≥ 10: textbbox returns (x0, y0, x1, y1)
    try:
        bbox = draw.textbbox((0, 0), handle, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow stub used in some test envs.
        text_w, text_h = draw.textsize(handle, font=font)  # type: ignore[attr-defined]
    tx = (w - text_w) // 2
    ty = h - banner_h + (banner_h - text_h) // 2
    draw.text((tx, ty), handle, fill=fg_rgb, font=font)


def _draw_accent_corner(img: Image.Image, *, accent_color: str) -> None:
    """Top-right accent triangle — a tiny brand-color flag so the
    thumbnail reads as branded even when no handle is supplied."""
    w, _h = img.size
    flag_size = max(60, w // 12)
    rgb = _hex_to_rgb(accent_color)
    draw = ImageDraw.Draw(img, mode="RGBA")
    draw.polygon(
        [(w, 0), (w, flag_size), (w - flag_size, 0)],
        fill=(rgb[0], rgb[1], rgb[2], 235),
    )


# ---- public entry ----


def render_branded_thumbnails(
    *,
    source_jpeg: bytes,
    clip_dir: Path,
    handle: str | None,
    primary_color: str,
    accent_color: str,
    text_color: str | None = None,
) -> list[Path]:
    """Composite the three aspect-ratio variants. Returns the list of
    paths written (empty when decoding or rendering fails).

    `source_jpeg` is the raw frame bytes from `pick_thumbnail`. `handle`
    is the platform handle (e.g. "@aara_art") — we draw whichever the
    brand kit supplies first; the caller picks the right one. When
    `handle` is empty / None we still ship a branded variant but skip
    the bottom banner."""
    try:
        src = Image.open(io.BytesIO(source_jpeg)).convert("RGB")
    except Exception as e:
        # Source JPEG is unreadable (placeholder test bytes, truncated
        # download, etc.). Caller treats empty return as "no branded
        # thumbnails for this clip" — the raw `thumbnail.jpg` is still
        # available.
        _log.warning("thumbnail.brand_decode_failed", error=str(e))
        return []

    clip_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for variant in _VARIANTS:
        try:
            framed = _crop_to_aspect(
                src, target_w=variant.width, target_h=variant.height
            )
            # Re-mode to RGBA so the banner can use alpha-blended fill;
            # we flatten back to RGB before saving JPEG.
            framed = framed.convert("RGBA")
            _draw_accent_corner(framed, accent_color=accent_color)
            if handle:
                _draw_handle_banner(
                    framed,
                    handle=handle,
                    primary_color=primary_color,
                    text_color=text_color,
                )
            flat = Image.new("RGB", framed.size, (0, 0, 0))
            flat.paste(framed, mask=framed.split()[3])
            target = clip_dir / variant.filename
            flat.save(target, format="JPEG", quality=90, optimize=True)
            out.append(target)
        except Exception as e:
            _log.warning(
                "thumbnail.brand_variant_failed",
                aspect=variant.aspect,
                error=str(e),
            )
    return out


def pick_brand_kit_handle(
    *,
    handle_tiktok: str | None = None,
    handle_youtube: str | None = None,
    handle_instagram: str | None = None,
    handle_kick: str | None = None,
) -> str | None:
    """Resolve the handle the thumbnail should burn in.

    Priority: TikTok → Instagram → YouTube → Kick. Same order the
    in-clip drawtext (slice D.1) uses, so the thumbnail and the
    clip read as consistent."""
    for h in (handle_tiktok, handle_instagram, handle_youtube, handle_kick):
        if h and h.strip():
            return h.strip()
    return None

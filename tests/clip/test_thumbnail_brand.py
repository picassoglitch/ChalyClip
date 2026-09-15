"""Brand-kit thumbnail compositor — voice-markers spec slice D.4.

Generates three aspect-ratio variants from a raw frame plus brand-kit
colors + handle. Tests cover:
  * The center-crop is applied so each variant matches the target aspect.
  * The banner pixels show up at the bottom (we sample for the primary
    color rather than OCR'ing the handle text).
  * The accent corner is drawn (sample top-right pixel).
  * Handle priority TikTok > Instagram > YouTube > Kick.
  * Empty / None handle skips the banner.
  * Malformed input returns an empty path list instead of raising.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from chalybclip.clip.thumbnail_brand import (
    pick_brand_kit_handle,
    render_branded_thumbnails,
)


def _make_jpeg(*, width: int, height: int, fill: tuple[int, int, int]) -> bytes:
    """A simple solid-color JPEG to feed the compositor."""
    img = Image.new("RGB", (width, height), fill)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---- pick_brand_kit_handle ----


def test_handle_priority_prefers_tiktok() -> None:
    assert (
        pick_brand_kit_handle(
            handle_tiktok="@a",
            handle_youtube="@b",
            handle_instagram="@c",
            handle_kick="d",
        )
        == "@a"
    )


def test_handle_priority_falls_through_to_instagram() -> None:
    assert (
        pick_brand_kit_handle(
            handle_tiktok=None,
            handle_youtube="@b",
            handle_instagram="@c",
            handle_kick="d",
        )
        == "@c"
    )


def test_handle_priority_returns_none_when_all_empty() -> None:
    assert (
        pick_brand_kit_handle(
            handle_tiktok=None,
            handle_youtube="",
            handle_instagram="   ",
            handle_kick=None,
        )
        is None
    )


# ---- render_branded_thumbnails ----


def test_writes_all_three_aspect_variants(tmp_path: Path) -> None:
    """Happy path: every variant gets written with the right size."""
    src = _make_jpeg(width=1920, height=1080, fill=(50, 50, 50))
    paths = render_branded_thumbnails(
        source_jpeg=src,
        clip_dir=tmp_path,
        handle="@aara_art",
        primary_color="#FF3366",
        accent_color="#FFD700",
    )
    assert len(paths) == 3
    by_name = {p.name: p for p in paths}
    assert "thumb_16x9.jpg" in by_name
    assert "thumb_9x16.jpg" in by_name
    assert "thumb_1x1.jpg" in by_name

    # Each variant has the spec'd output size — no letterboxing.
    sizes = {p.name: Image.open(p).size for p in paths}
    assert sizes["thumb_16x9.jpg"] == (1280, 720)
    assert sizes["thumb_9x16.jpg"] == (1080, 1920)
    assert sizes["thumb_1x1.jpg"] == (1080, 1080)


def test_banner_uses_brand_primary_color(tmp_path: Path) -> None:
    """Sample a pixel from the bottom-center of the rendered output —
    it should be close to the brand-kit primary color (the banner fill)."""
    src = _make_jpeg(width=1920, height=1080, fill=(0, 0, 0))
    paths = render_branded_thumbnails(
        source_jpeg=src,
        clip_dir=tmp_path,
        handle="@brand",
        primary_color="#FF3366",
        accent_color="#FFD700",
    )
    assert paths
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        # Sample on either side of the banner, away from glyph strokes.
        for x in (int(w * 0.1), int(w * 0.9)):
            r, g, b = img.getpixel((x, h - 6))
            # Banner is 90% opaque over a black background → ~ (#FF * .9) etc.
            # Tolerate antialiasing slop.
            assert r > 180, f"x={x} R={r} too low; banner not bright red"
            assert g < 80, f"x={x} G={g} too high; banner bled through"
            assert 60 < b < 130, f"x={x} B={b} out of expected magenta range"


def test_accent_flag_drawn_in_top_right(tmp_path: Path) -> None:
    """Top-right pixel should pick up the accent color (gold)."""
    src = _make_jpeg(width=1920, height=1080, fill=(0, 0, 0))
    paths = render_branded_thumbnails(
        source_jpeg=src,
        clip_dir=tmp_path,
        handle=None,
        primary_color="#000000",
        accent_color="#FFD700",
    )
    assert paths
    p = next(x for x in paths if x.name == "thumb_16x9.jpg")
    img = Image.open(p).convert("RGB")
    w, _ = img.size
    r, g, b = img.getpixel((w - 4, 4))
    # Gold = #FFD700 ≈ (255, 215, 0); banner alpha pulls it slightly off.
    assert r > 200, f"top-right R={r} too low for gold"
    assert g > 150, f"top-right G={g} too low for gold"
    assert b < 80, f"top-right B={b} too high; gold should have ~0 blue"


def test_no_handle_still_produces_variants(tmp_path: Path) -> None:
    """`handle=None` is allowed — the variants ship without the banner."""
    src = _make_jpeg(width=1920, height=1080, fill=(120, 120, 120))
    paths = render_branded_thumbnails(
        source_jpeg=src,
        clip_dir=tmp_path,
        handle=None,
        primary_color="#FF3366",
        accent_color="#FFD700",
    )
    assert len(paths) == 3
    # Bottom-center pixel should NOT be brand red — there's no banner.
    img = Image.open(paths[0]).convert("RGB")
    w, h = img.size
    r, _g, _b = img.getpixel((w // 2, h - 6))
    # Source was grey (120,120,120); without the banner the bottom stays grey.
    assert r < 200, "banner appeared even though handle is None"


def test_malformed_jpeg_returns_empty_list(tmp_path: Path) -> None:
    """Junk bytes — the compositor logs and returns [] rather than raising."""
    paths = render_branded_thumbnails(
        source_jpeg=b"\x00\x01\x02 not a jpeg",
        clip_dir=tmp_path,
        handle="@x",
        primary_color="#FF3366",
        accent_color="#FFD700",
    )
    assert paths == []


def test_tall_source_is_center_cropped(tmp_path: Path) -> None:
    """A 9:16 source feeding the 16:9 variant should center-crop the
    vertical slack — the variant ends up 16:9 with no letterboxing."""
    src = _make_jpeg(width=1080, height=1920, fill=(100, 100, 100))
    paths = render_branded_thumbnails(
        source_jpeg=src,
        clip_dir=tmp_path,
        handle=None,
        primary_color="#000000",
        accent_color="#FFFFFF",
    )
    horiz = next(p for p in paths if p.name == "thumb_16x9.jpg")
    img = Image.open(horiz)
    assert img.size == (1280, 720)


def test_invalid_hex_falls_back_to_black(tmp_path: Path) -> None:
    """A junk hex color should not crash the compositor — it falls back
    to black so the variant still ships."""
    src = _make_jpeg(width=1920, height=1080, fill=(255, 255, 255))
    paths = render_branded_thumbnails(
        source_jpeg=src,
        clip_dir=tmp_path,
        handle="@x",
        primary_color="notahex",
        accent_color="alsonothex",
    )
    assert len(paths) == 3


def test_three_letter_hex_supported(tmp_path: Path) -> None:
    """`#f36` should expand to `#ff3366` rather than fall back to black."""
    src = _make_jpeg(width=1920, height=1080, fill=(0, 0, 0))
    paths = render_branded_thumbnails(
        source_jpeg=src,
        clip_dir=tmp_path,
        handle="@x",
        primary_color="#f36",
        accent_color="#fd7",
    )
    img = Image.open(paths[0]).convert("RGB")
    w, h = img.size
    r, _g, _b = img.getpixel((int(w * 0.1), h - 6))
    assert r > 180, "three-letter hex didn't expand to a bright red"

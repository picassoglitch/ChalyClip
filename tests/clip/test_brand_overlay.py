"""Brand-kit-driven drawtext overlay (slice D.1).

The full ffmpeg invocation is integration-tested by the pipeline tests
(they monkey-patch _run_ffmpeg). Here we cover the pure-logic helper
that builds the drawtext filter expression: the right handle is picked,
the accent color is used, Windows paths are escaped correctly, and
missing inputs gracefully degrade to None (renderer skips the overlay).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chalybclip.clip.service import _brand_kit_drawtext_filter
from chalybclip.db.models import BrandKitRow


def _make_kit(**overrides: object) -> BrandKitRow:
    base = {
        "id": "brk_t",
        "tenant_id": "t",
        "name": "Test Kit",
        "primary_color": "#FF3366",
        "accent_color": "#FFD700",
        "created_at": "2026-05-12T00:00:00+00:00",
        "updated_at": "2026-05-12T00:00:00+00:00",
    }
    base.update(overrides)
    return BrandKitRow.model_validate(base)


def test_none_kit_returns_no_overlay() -> None:
    assert _brand_kit_drawtext_filter(None, output_w=1080) is None


def test_kit_without_any_handle_returns_no_overlay() -> None:
    kit = _make_kit()  # no handle on any platform
    assert _brand_kit_drawtext_filter(kit, output_w=1080) is None


def test_tiktok_handle_wins_when_set() -> None:
    """Spec §3.5: handle priority tiktok > youtube > instagram > kick."""
    kit = _make_kit(
        handle_tiktok="@aara_art",
        handle_youtube="@should_not_win",
        handle_instagram="@nope",
    )
    with patch(
        "chalybclip.clip.service._find_system_font",
        return_value=Path("/fonts/arial.ttf"),
    ):
        out = _brand_kit_drawtext_filter(kit, output_w=1080)
    assert out is not None
    assert "@aara_art" in out
    assert "@should_not_win" not in out


def test_falls_through_to_kick_when_no_other_handle() -> None:
    kit = _make_kit(handle_kick="aldovillanueva")
    with patch(
        "chalybclip.clip.service._find_system_font",
        return_value=Path("/fonts/arial.ttf"),
    ):
        out = _brand_kit_drawtext_filter(kit, output_w=1080)
    assert out is not None
    assert "aldovillanueva" in out


def test_uses_accent_color_for_fontcolor() -> None:
    kit = _make_kit(accent_color="#00FF00", handle_tiktok="@x")
    with patch(
        "chalybclip.clip.service._find_system_font",
        return_value=Path("/fonts/arial.ttf"),
    ):
        out = _brand_kit_drawtext_filter(kit, output_w=1080)
    assert out is not None
    assert "fontcolor=#00FF00" in out


def test_no_font_found_returns_none() -> None:
    """If we can't resolve any system font, the overlay is skipped silently
    rather than producing an ffmpeg error."""
    kit = _make_kit(handle_tiktok="@x")
    with patch("chalybclip.clip.service._find_system_font", return_value=None):
        out = _brand_kit_drawtext_filter(kit, output_w=1080)
    assert out is None


def test_windows_font_path_is_colon_escaped() -> None:
    """The drive-letter colon in a Windows font path is a filter-graph
    delimiter unless escaped. Without escaping, ffmpeg would parse
    'C:/Windows/...' as setting filter option 'C' to '/Windows/...'."""
    kit = _make_kit(handle_tiktok="@x")
    with patch(
        "chalybclip.clip.service._find_system_font",
        return_value=Path(r"C:\Windows\Fonts\arial.ttf"),
    ):
        out = _brand_kit_drawtext_filter(kit, output_w=1080)
    assert out is not None
    assert "C\\:/Windows/Fonts/arial.ttf" in out


def test_fontsize_scales_with_output_width() -> None:
    """Larger output canvas → larger handle font (so handle stays
    proportional across 720p / 1080p / 4K)."""
    kit = _make_kit(handle_tiktok="@x")
    with patch(
        "chalybclip.clip.service._find_system_font",
        return_value=Path("/fonts/arial.ttf"),
    ):
        out_small = _brand_kit_drawtext_filter(kit, output_w=720)
        out_large = _brand_kit_drawtext_filter(kit, output_w=1920)
    assert out_small is not None and out_large is not None
    # Extract fontsize=N from each.
    def fs(s: str) -> int:
        for piece in s.split(":"):
            if piece.startswith("fontsize="):
                return int(piece.split("=", 1)[1])
        raise AssertionError("no fontsize in filter")

    assert fs(out_large) > fs(out_small)


def test_single_quote_in_handle_is_stripped() -> None:
    """drawtext uses single quotes as value delimiters; embedded quotes
    would break the filter. We strip them defensively."""
    kit = _make_kit(handle_tiktok="@o'connor")
    with patch(
        "chalybclip.clip.service._find_system_font",
        return_value=Path("/fonts/arial.ttf"),
    ):
        out = _brand_kit_drawtext_filter(kit, output_w=1080)
    assert out is not None
    assert "@oconnor" in out
    assert "'@oconnor'" in out  # The value delimiter quotes remain.


def test_kit_with_empty_string_handle_treated_as_missing() -> None:
    """Empty string handle from the form falls back to other handles or
    None — does NOT produce a drawtext with empty text."""
    kit = _make_kit(
        handle_tiktok="",
        handle_youtube="@yt",
    )
    with patch(
        "chalybclip.clip.service._find_system_font",
        return_value=Path("/fonts/arial.ttf"),
    ):
        out = _brand_kit_drawtext_filter(kit, output_w=1080)
    assert out is not None
    assert "@yt" in out

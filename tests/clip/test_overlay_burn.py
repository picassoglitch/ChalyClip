"""Renderer-side overlay burn-in (slice F.7-E) — pure-function tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from chalybclip.clip.overlay_burn import (
    PLATFORM_COLORS,
    _ass_ts,
    _ff_escape_path,
    _ff_escape_text,
    _hex_to_ff_color,
    _spec_from_overlay_config,
    _srt_ts,
    build_ass,
    build_filter_graph,
    build_srt,
    burn_overlays,
    captions_artifact_for_clip,
)
from chalybclip.clip.word_captions import chunk_words_to_lines

# ---- text + color escaping --------------------------------------


def test_ff_escape_text_handles_every_special_char() -> None:
    """ffmpeg drawtext eats: backslash, colon, single-quote, square
    brackets, comma, semicolon, percent. All must be escaped."""
    s = r"hi: 'world', [test]; 100% (\backslash)"
    out = _ff_escape_text(s)
    # Every special char is preceded by a backslash.
    assert "\\\\" in out  # backslash
    assert "\\:" in out
    assert "\\'" in out
    assert "\\[" in out and "\\]" in out
    assert "\\," in out
    assert "\\;" in out
    assert "\\%" in out


def test_ff_escape_text_leaves_normal_chars_alone() -> None:
    out = _ff_escape_text("Hello world")
    assert out == "Hello world"


def test_ff_escape_path_handles_windows_colons() -> None:
    p = Path(r"C:\Windows\Fonts\arial.ttf")
    out = _ff_escape_path(p)
    # Forward slashes + escaped colon for ffmpeg's filter parser.
    assert out == "C\\:/Windows/Fonts/arial.ttf"
    assert "\\\\" not in out  # backslashes flipped to forward


def test_ff_escape_path_unix_path_passthrough() -> None:
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    out = _ff_escape_path(p)
    assert out == "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def test_hex_to_ff_color_rejects_hash_prefix_returns_named_or_0x() -> None:
    assert _hex_to_ff_color("#FFD700") == "0xFFD700"
    assert _hex_to_ff_color("FFD700") == "0xFFD700"
    # 3-digit hex expands to 6.
    assert _hex_to_ff_color("#abc") == "0xAABBCC"
    # Junk input → fallback.
    assert _hex_to_ff_color("not a hex") == "white"
    assert _hex_to_ff_color("zzzzzz") == "white"


def test_hex_to_ff_color_custom_fallback() -> None:
    assert _hex_to_ff_color("not a hex", fallback="black") == "black"


# ---- _spec_from_overlay_config ---------------------------------


def test_spec_from_none_collapses_to_defaults() -> None:
    s = _spec_from_overlay_config(None)
    assert s.title_text is None
    assert s.banner_enabled is False
    assert s.banner_platform == "kick"
    assert s.captions_enabled is True  # default-ON to match the editor


def test_spec_strips_whitespace_only_title_to_none() -> None:
    s = _spec_from_overlay_config({"title_text": "   "})
    assert s.title_text is None


def test_spec_extracts_full_config_correctly() -> None:
    s = _spec_from_overlay_config({
        "title_text": "Hello world",
        "banner": {
            "enabled": True,
            "platform": "TWITCH",  # uppercase from form → lowercased
            "url": "twitch.tv/foo",
            "color": "#9146FF",
        },
        "captions": {"enabled": False},
    })
    assert s.title_text == "Hello world"
    assert s.banner_enabled is True
    assert s.banner_platform == "twitch"
    assert s.banner_url == "twitch.tv/foo"
    assert s.banner_color == "#9146FF"
    assert s.captions_enabled is False


def test_spec_tolerates_malformed_nested_blocks() -> None:
    """Missing or wrong-typed nested blocks don't raise — they fall
    back to defaults so the editor saves are non-destructive."""
    s = _spec_from_overlay_config({
        "title_text": None,
        "banner": "not a dict",  # type: ignore[dict-item]
        "captions": 42,           # type: ignore[dict-item]
    })
    assert s.title_text is None
    assert s.banner_enabled is False
    assert s.captions_enabled is True


# ---- build_srt --------------------------------------------------


def test_srt_ts_format() -> None:
    assert _srt_ts(0) == "00:00:00,000"
    assert _srt_ts(0.5) == "00:00:00,500"
    assert _srt_ts(61.234) == "00:01:01,234"
    assert _srt_ts(3661.999) == "01:01:01,999"


def test_srt_ts_clamps_negative_to_zero() -> None:
    assert _srt_ts(-5) == "00:00:00,000"


def test_srt_ts_handles_999ms_rounding_carry() -> None:
    """0.9999s rounds to 1000ms → must carry into the seconds field
    rather than stamp `00:00:00,1000` (which would break players)."""
    out = _srt_ts(0.9999)
    assert out == "00:00:01,000"


def test_build_srt_returns_empty_string_when_no_segments() -> None:
    assert build_srt([], clip_start_s=0, clip_end_s=10) == ""


def test_build_srt_filters_to_clip_window_and_shifts_timing() -> None:
    """A segment at stream-time 105-107 inside a clip starting at 100
    becomes SRT-time 5-7 (clip-relative)."""
    segs = [
        {"start": 50.0, "end": 55.0, "text": "before clip"},     # outside
        {"start": 105.0, "end": 107.0, "text": "inside clip"},   # inside
        {"start": 200.0, "end": 205.0, "text": "way after"},     # outside
    ]
    body = build_srt(segs, clip_start_s=100.0, clip_end_s=130.0)
    assert "before clip" not in body
    assert "way after" not in body
    assert "inside clip" in body
    # Timing shifted: segment at 105 → clip-relative 5s.
    assert "00:00:05,000" in body
    assert "00:00:07,000" in body


def test_build_srt_clamps_segments_at_window_edges() -> None:
    """A segment that straddles the clip end gets clipped to the end."""
    segs = [
        {"start": 95.0, "end": 105.0, "text": "straddles start"},
        {"start": 125.0, "end": 140.0, "text": "straddles end"},
    ]
    body = build_srt(segs, clip_start_s=100.0, clip_end_s=130.0)
    # First segment starts before the window — clamped to 0:00.
    assert "00:00:00,000 -->" in body
    # Second segment ends after the window — clamped to 30s.
    assert "00:00:30,000" in body


def test_build_srt_skips_segments_with_missing_text() -> None:
    segs = [
        {"start": 5.0, "end": 6.0, "text": ""},
        {"start": 6.0, "end": 7.0},  # no text key
        {"start": 7.0, "end": 8.0, "text": "real one"},
    ]
    body = build_srt(segs, clip_start_s=0, clip_end_s=10)
    assert body.count("-->") == 1
    assert "real one" in body


def test_build_srt_assigns_sequential_indices() -> None:
    segs = [
        {"start": 1.0, "end": 2.0, "text": "first"},
        {"start": 3.0, "end": 4.0, "text": "second"},
        {"start": 5.0, "end": 6.0, "text": "third"},
    ]
    body = build_srt(segs, clip_start_s=0, clip_end_s=10)
    lines = body.split("\n")
    # Indices 1, 2, 3 appear at the start of each block.
    assert lines[0] == "1"
    assert "second" in lines
    assert "third" in lines


# ---- build_filter_graph ----------------------------------------


def test_filter_graph_returns_empty_when_nothing_enabled(tmp_path: Path) -> None:
    """No title, no banner, captions on but no SRT → empty graph
    (caller skips the entire re-render in this case)."""
    spec = _spec_from_overlay_config({
        "title_text": None,
        "banner": {"enabled": False},
        "captions": {"enabled": True},
    })
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=None,
    )
    assert graph == ""


def test_filter_graph_includes_title_when_set(tmp_path: Path) -> None:
    spec = _spec_from_overlay_config({"title_text": "Hello world"})
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=None,
    )
    assert "drawtext=" in graph
    assert "Hello world" in graph
    # Slice L.4 — title sits at 18% from top (345 on a 1920 canvas) so
    # it clears the platform top-unsafe band.
    assert ":y=345" in graph
    # Background box, white, with the L.4 fat padding.
    assert "boxcolor=white" in graph
    assert "boxborderw=22" in graph


def test_filter_graph_includes_banner_when_enabled(tmp_path: Path) -> None:
    """Legacy band renderer (black_bar_classic variant) — colored band +
    platform name + URL. The default variant (kick_repost_page) renders
    the richer chrome covered by the kick companion test below."""
    spec = _spec_from_overlay_config({
        "banner": {
            "enabled": True,
            "platform": "kick",
            "variant": "kick_black_bar_classic",
            "url": "kick.com/clavicular",
            "color": "#53FC18",
        },
    })
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=None,
    )
    # black_bar_classic → the legacy band renderer: colored band +
    # platform name + URL (no wordmark PNG, no KICK.COM rewrite).
    assert "drawbox=" in graph
    assert "0x53FC18" in graph  # #-prefixed color converted
    assert "KICK" in graph  # platform name uppercased
    assert "kick.com/clavicular" in graph
    assert "__PNG_OVERLAY__" not in graph  # this variant has no logo overlay


def test_filter_graph_banner_uses_platform_default_color_when_blank(
    tmp_path: Path,
) -> None:
    """No explicit color → fall back to PLATFORM_COLORS[platform].

    Also pins the slice I.1 regression fix: the default banner variant
    (kick_repost_page) is Kick-branded chrome, so a non-kick platform
    must fall through to the legacy per-platform renderer — twitch
    band + TWITCH wordmark, never the Kick logo sentinel."""
    spec = _spec_from_overlay_config({
        "banner": {"enabled": True, "platform": "twitch"},
    })
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=None,
    )
    # Twitch default → 0x9146FF.
    twitch = PLATFORM_COLORS["twitch"].lstrip("#")
    assert f"0x{twitch.upper()}" in graph
    # Platform's own wordmark, not Kick chrome.
    assert "TWITCH" in graph
    assert "__PNG_OVERLAY__" not in graph
    assert "KICK.COM" not in graph


def test_filter_graph_kick_platform_keeps_repost_page_chrome(
    tmp_path: Path,
) -> None:
    """Kick-sourced clips DO get the repost-page chrome — black bar +
    the `__PNG_OVERLAY__:key=kick` wordmark sentinel burn_overlays swaps
    for the PNG overlay, with the URL normalized to KICK.COM/<handle>."""
    spec = _spec_from_overlay_config({
        "banner": {"enabled": True, "platform": "kick", "url": "clavicular"},
    })
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=None,
    )
    assert "__PNG_OVERLAY__:key=kick" in graph
    assert "color=black@1.0" in graph
    assert "KICK.COM/CLAVICULAR" in graph


def test_filter_graph_platform_band_youtube(tmp_path: Path) -> None:
    """A YouTube source banner (variant=platform_band) → YouTube-red band
    + the `__PNG_OVERLAY__:key=youtube` glyph sentinel + the domain URL.
    No Kick chrome, no KICK wordmark sentinel."""
    spec = _spec_from_overlay_config({
        "banner": {
            "enabled": True,
            "platform": "youtube",
            "variant": "platform_band",
            "url": "youtube.com/@somestreamer",
        },
    })
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=None,
    )
    yt = PLATFORM_COLORS["youtube"].lstrip("#")
    assert f"0x{yt.upper()}" in graph  # YouTube-red band
    assert "__PNG_OVERLAY__:key=youtube" in graph
    assert "YOUTUBE.COM/@SOMESTREAMER" in graph  # uppercased domain
    assert "key=kick" not in graph
    assert "KICK.COM" not in graph


def test_filter_graph_platform_band_twitch_and_x(tmp_path: Path) -> None:
    """Twitch → purple band + twitch glyph; X → its glyph sentinel."""
    for platform, url in (("twitch", "twitch.tv/ninja"), ("x", "x.com/elon")):
        spec = _spec_from_overlay_config({
            "banner": {
                "enabled": True,
                "platform": platform,
                "variant": "platform_band",
                "url": url,
            },
        })
        graph = build_filter_graph(
            spec,
            output_w=1080,
            output_h=1920,
            fontfile=tmp_path / "fake.ttf",
            srt_path=None,
        )
        assert f"__PNG_OVERLAY__:key={platform}" in graph
        assert url.upper() in graph
        color = PLATFORM_COLORS[platform].lstrip("#")
        assert f"0x{color.upper()}" in graph


def test_filter_graph_skips_drawtext_when_no_fontfile(tmp_path: Path) -> None:
    """Without a system font ffmpeg drawtext can't run — title +
    banner are skipped silently. Captions can still render via the
    subtitles filter (SRT carries its own ASS styling)."""
    spec = _spec_from_overlay_config({
        "title_text": "x",
        "banner": {"enabled": True, "platform": "kick"},
    })
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=None,
        srt_path=None,
    )
    assert graph == ""


def test_filter_graph_includes_captions_when_srt_exists(tmp_path: Path) -> None:
    srt = tmp_path / ".captions.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
    spec = _spec_from_overlay_config({"captions": {"enabled": True}})
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=srt,
    )
    assert "subtitles=" in graph
    assert "force_style" in graph


def test_filter_graph_skips_captions_when_disabled(tmp_path: Path) -> None:
    srt = tmp_path / ".captions.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
    spec = _spec_from_overlay_config({"captions": {"enabled": False}})
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=srt,
    )
    assert "subtitles=" not in graph


def test_filter_graph_chains_filters_with_commas(tmp_path: Path) -> None:
    """ffmpeg -vf joins filters with commas (NOT semicolons — that'd
    create a graph link, not a chain). The builder's contract is
    'comma-joined filter chunks'."""
    srt = tmp_path / ".captions.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
    spec = _spec_from_overlay_config({
        "title_text": "T",
        "banner": {"enabled": True, "platform": "kick"},
        "captions": {"enabled": True},
    })
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        srt_path=srt,
    )
    # All three families present.
    assert "drawtext=" in graph
    assert "subtitles=" in graph
    assert "drawbox=" in graph
    # Joined with commas, not semicolons.
    assert ";" not in graph
    assert "," in graph


# ---- burn_overlays driver --------------------------------------


def test_burn_overlays_returns_false_when_nothing_to_burn(
    tmp_path: Path,
) -> None:
    """Empty overlay_config + no transcript → nothing to render.
    Returns False so the caller falls back to the original clip."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    target = tmp_path / "clip_final.mp4"
    out = burn_overlays(
        source_path=src,
        target_path=target,
        overlay_config=None,
        transcript_segments=[],
        clip_start_s=0,
        clip_end_s=10,
        output_w=1080,
        output_h=1920,
    )
    assert out is False
    assert not target.exists()


def test_burn_overlays_raises_when_source_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="source clip missing"):
        burn_overlays(
            source_path=tmp_path / "nope.mp4",
            target_path=tmp_path / "out.mp4",
            overlay_config={"title_text": "x"},
            transcript_segments=[],
            clip_start_s=0,
            clip_end_s=10,
            output_w=1080,
            output_h=1920,
        )


def test_burn_overlays_invokes_ffmpeg_with_filter_graph(
    tmp_path: Path,
) -> None:
    """Happy path: source exists, title is set, fake ffmpeg run
    succeeds → SRT is cleaned up + we return True."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    target = tmp_path / "clip_final.mp4"
    captured: list[list[str]] = []

    def fake_run(cmd, capture_output, check, timeout=None):
        captured.append(cmd)
        # Pretend ffmpeg wrote the output (the scratch .part.mp4 path —
        # burn_overlays promotes it to target with os.replace on success).
        Path(cmd[-1]).write_bytes(b"burned")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    with patch("chalybclip.clip.overlay_burn.subprocess.run", fake_run):
        out = burn_overlays(
            source_path=src,
            target_path=target,
            overlay_config={
                "title_text": "Hello",
                # kick default (repost_page) → the two-input -filter_complex
                # path (KICK wordmark PNG overlay); title + captions ride
                # the main [0:v] chain inside it.
                "banner": {
                    "enabled": True,
                    "platform": "kick",
                    "color": "#53FC18",
                },
                "captions": {"enabled": True},
            },
            transcript_segments=[
                {"start": 1.0, "end": 2.0, "text": "hi"},
            ],
            clip_start_s=0,
            clip_end_s=10,
            output_w=1080,
            output_h=1920,
        )
    assert out is True
    assert target.exists()
    # Per-burn artifacts cleaned up: the (uniquely named) captions file
    # and the scratch .part.mp4 the encode was promoted from.
    assert not list(target.parent.glob(".captions*"))
    assert not list(target.parent.glob("*.part.mp4"))
    # Banner platform=kick routes through the two-input -filter_complex
    # graph (KICK wordmark PNG overlay); title + captions ride the main
    # [0:v] chain inside it.
    cmd = captured[0]
    assert "ffmpeg" in cmd[0]
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "Hello" in fc
    assert "subtitles=" in fc
    assert "overlay=" in fc
    # Audio is copied (no re-encode).
    assert cmd[cmd.index("-c:a") + 1] == "copy"


def test_burn_overlays_composites_platform_glyph_via_filter_complex(
    tmp_path: Path,
) -> None:
    """A platform_band banner drives the two-input `-filter_complex`
    path: main video + the platform glyph PNG overlaid. Kick's wordmark
    uses the same generalized path; here we prove it works for YouTube
    and that the `__PNG_OVERLAY__` sentinel is stripped before ffmpeg."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    target = tmp_path / "clip_final.mp4"
    captured: list[list[str]] = []

    def fake_run(cmd, capture_output, check, timeout=None):
        captured.append(cmd)
        # Encode writes the scratch .part.mp4; burn_overlays promotes it
        # to target with os.replace on success.
        Path(cmd[-1]).write_bytes(b"burned")
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=b"", stderr=b""
        )

    with patch("chalybclip.clip.overlay_burn.subprocess.run", fake_run):
        out = burn_overlays(
            source_path=src,
            target_path=target,
            overlay_config={
                "banner": {
                    "enabled": True,
                    "platform": "youtube",
                    "variant": "platform_band",
                    "url": "youtube.com/@aldo",
                },
                "captions": {"enabled": False},
            },
            transcript_segments=[],
            clip_start_s=0,
            clip_end_s=10,
            output_w=1080,
            output_h=1920,
        )
    assert out is True
    assert target.exists()
    cmd = captured[0]
    # Two inputs: the source clip + the glyph PNG.
    assert cmd.count("-i") == 2
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "overlay=" in fc
    assert "[outv]" in fc
    # Raw sentinel never reaches ffmpeg.
    assert "__PNG_OVERLAY__" not in fc
    # The second input is the YouTube glyph PNG.
    second_i = cmd.index("-i", cmd.index("-i") + 1)
    assert "glyph-youtube.png" in cmd[second_i + 1].replace("\\", "/")


def test_burn_overlays_propagates_ffmpeg_failure(tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    target = tmp_path / "clip_final.mp4"

    def fake_run(cmd, capture_output, check, timeout=None):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout=b"",
            stderr=b"Error: invalid filter graph",
        )

    with (
        patch("chalybclip.clip.overlay_burn.subprocess.run", fake_run),
        pytest.raises(RuntimeError, match="ffmpeg burn failed"),
    ):
        burn_overlays(
            source_path=src,
            target_path=target,
            overlay_config={"title_text": "x"},
            transcript_segments=[],
            clip_start_s=0,
            clip_end_s=10,
            output_w=1080,
            output_h=1920,
        )
    # Per-burn artifacts get cleaned up even on failure.
    assert not list(target.parent.glob(".captions*"))
    assert not list(target.parent.glob("*.part.mp4"))


# ---- ASS / word-level karaoke (slice F.7-F) ----------------------


def _make_words(specs: list[tuple[float, float, str]]) -> list:
    return chunk_words_to_lines(
        [{"ts": t, "end_ts": e, "text": txt} for t, e, txt in specs],
        clip_start_s=0,
        clip_end_s=60,
    )


def test_ass_ts_format() -> None:
    """ASS uses H:MM:SS.cs — centiseconds, NOT ms like SRT."""
    assert _ass_ts(0) == "0:00:00.00"
    assert _ass_ts(1.5) == "0:00:01.50"
    assert _ass_ts(61.23) == "0:01:01.23"
    assert _ass_ts(3661.99) == "1:01:01.99"


def test_ass_ts_clamps_negative_to_zero() -> None:
    assert _ass_ts(-5) == "0:00:00.00"


def test_build_ass_empty_lines_returns_empty_string() -> None:
    assert build_ass([]) == ""


def test_build_ass_includes_header_and_dialogue_block() -> None:
    lines = _make_words([(0.0, 0.5, "hello"), (0.5, 1.0, "world")])
    out = build_ass(lines)
    assert "[Script Info]" in out
    assert "[V4+ Styles]" in out
    assert "[Events]" in out
    assert "Dialogue: " in out
    # Margins reflect platform-safe positioning.
    assert "220" in out  # _ASS_BOTTOM_SAFE_MARGIN_V


def test_build_ass_inserts_karaoke_tag_per_word() -> None:
    """Every word gets a `\\kf<centiseconds>` tag for fill timing."""
    lines = _make_words([(0.0, 0.5, "hello"), (0.5, 1.0, "world")])
    out = build_ass(lines)
    # \kf50 for 0.5s words.
    assert "\\kf50" in out


def test_build_ass_uses_larger_font_for_shouts() -> None:
    """ALL-CAPS words get scaled up via the per-word `\\fs<size>` tag."""
    lines = _make_words([(0.0, 0.5, "EPIC")])
    out = build_ass(lines)
    # Shout scale 1.55 * baseline 56 ~ 87.
    assert "\\fs87" in out or "\\fs86" in out or "\\fs88" in out


def test_build_ass_uses_emphasis_color_per_word() -> None:
    """Shout words get the red fill; normal words get white."""
    shout_lines = _make_words([(0.0, 0.5, "WHAT")])
    out = build_ass(shout_lines)
    # Shout fill = &H003B3BFF.
    assert "&H003B3BFF" in out


def test_build_ass_escapes_curly_braces_in_word_text() -> None:
    """Words containing `{` / `}` (rare but Whisper can emit them)
    must not break ASS's override-tag parser."""
    lines = chunk_words_to_lines(
        [{"ts": 0.0, "end_ts": 0.5, "text": "weird{}word"}],
        clip_start_s=0,
        clip_end_s=10,
    )
    out = build_ass(lines)
    # Curly braces in the user text replaced; ASS still has its own
    # override braces around the karaoke tags. Caption words render
    # UPPERCASED (clip-culture styling), escaping preserved.
    assert "WEIRD()WORD" in out


def test_captions_artifact_prefers_ass_when_word_data_available() -> None:
    """The real DB shape carries word-level timings → ASS wins."""
    import json

    segments = [
        {
            "ts": 0.0,
            "end_ts": 1.0,
            "text": "hello world",
            "words": [
                {"ts": 0.0, "end_ts": 0.4, "text": "hello", "prob": 0.9},
                {"ts": 0.4, "end_ts": 1.0, "text": "world", "prob": 0.9},
            ],
        },
    ]
    body, ext = captions_artifact_for_clip(
        json.dumps(segments), clip_start_s=0, clip_end_s=10
    )
    assert ext == "ass"
    assert "[Events]" in body
    assert "\\kf" in body


def test_captions_artifact_returns_empty_when_no_transcript() -> None:
    body, _ext = captions_artifact_for_clip(None, clip_start_s=0, clip_end_s=10)
    assert body == ""


def test_build_srt_reads_real_db_shape_ts_end_ts() -> None:
    """REGRESSION GUARD for the production bug: build_srt USED to read
    only `start`/`end`, so it silently produced empty SRTs on real
    data (DB shape uses `ts`/`end_ts`). The fix reads either shape.

    This test feeds the real shape and proves the SRT body is
    non-empty + carries the expected text."""
    segments = [
        {"ts": 5.0, "end_ts": 6.5, "text": "actual production shape"},
    ]
    body = build_srt(segments, clip_start_s=0, clip_end_s=10)
    assert "actual production shape" in body
    assert "00:00:05" in body


def test_filter_graph_uses_subtitles_filter_for_ass_path(tmp_path) -> None:
    """`.ass` files don't pass force_style — the ASS file's Style
    line carries its own font / position / margins."""
    ass = tmp_path / ".captions.ass"
    ass.write_text("[Script Info]\n", encoding="utf-8")
    spec = _spec_from_overlay_config({"captions": {"enabled": True}})
    graph = build_filter_graph(
        spec,
        output_w=1080,
        output_h=1920,
        fontfile=tmp_path / "fake.ttf",
        captions_path=ass,
    )
    assert "subtitles=" in graph
    assert "force_style" not in graph  # ASS carries its own style

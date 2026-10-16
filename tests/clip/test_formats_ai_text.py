"""chalybclip.clip.formats (base videos per format, real ffmpeg when
available) and chalybclip.clip.ai_text (Spanish for the AI's sentences)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from chalybclip.clip import ai_decisions, publishability, scoring
from chalybclip.clip.ai_text import localize, localize_viral_why
from chalybclip.clip.formats import (
    available_formats,
    base_name,
    clip_format,
    ensure_format_base,
    pristine_window,
    render_dims,
)


def test_render_dims_and_format_choice() -> None:
    assert render_dims("vertical", 1080, 1920) == (1080, 1920)
    assert render_dims("horizontal", 1080, 1920) == (1920, 1080)
    assert render_dims("square", 1440, 2560) == (1440, 1440)
    assert clip_format(SimpleNamespace(overlay_config={"format": "square"})) == "square"
    assert clip_format(SimpleNamespace(overlay_config={"format": "x"})) == "vertical"
    assert clip_format(SimpleNamespace(overlay_config=None)) == "vertical"
    assert base_name("square", 1.5, 8.25) == "clip_square_1500_8250.mp4"


def test_pristine_window_follows_trim_bookkeeping() -> None:
    fresh = SimpleNamespace(start_s=3.0, end_s=9.0, original_start_s=None, original_end_s=None)
    trimmed = SimpleNamespace(start_s=4.0, end_s=7.0, original_start_s=3.0, original_end_s=9.0)
    assert pristine_window(fresh) == (3.0, 9.0)
    assert pristine_window(trimmed) == (3.0, 9.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
async def test_bases_are_cut_from_the_master_at_current_bounds(tmp_path: Path) -> None:
    master = tmp_path / "clip_landscape.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
            "-shortest", "-c:v", "libx264", "-c:a", "aac", str(master),
        ],
        check=True,
    )
    clip = SimpleNamespace(
        id="clp_x",
        path=str(tmp_path / "clip.mp4"),
        start_s=12.0,
        end_s=16.0,
        original_start_s=10.0,
        original_end_s=20.0,
        smart_crop_box={"x": 400.0, "y": 0.0, "w": 120.0, "h": 360.0},
        overlay_config={},
    )
    assert await available_formats(clip=clip, tenant_id="t", source_video=None) == [
        "vertical", "horizontal", "square",
    ]
    for fmt, size in (("horizontal", "1920,1080"), ("square", "1080,1080")):
        out = await ensure_format_base(clip=clip, fmt=fmt, tenant_id="t", source_video=None)
        assert out is not None and out.name == base_name(fmt, 12.0, 16.0)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert probe[0] == size
        assert float(probe[1]) == pytest.approx(4.0, abs=0.15)
    # New bounds → a new base; the stale one is dropped.
    clip.start_s, clip.end_s = 11.0, 17.0
    out = await ensure_format_base(clip=clip, fmt="square", tenant_id="t", source_video=None)
    assert out is not None and out.name == "clip_square_11000_17000.mp4"
    assert not (tmp_path / "clip_square_12000_16000.mp4").exists()
    # Outside the master's window and no source: not available.
    clip.start_s, clip.end_s = 5.0, 12.0
    assert await ensure_format_base(clip=clip, fmt="square", tenant_id="t", source_video=None) is None


def test_every_ai_sentence_has_spanish() -> None:
    """Run the real producers and check nothing comes back in English."""
    english_markers = ("words/sec", "Hook", "Captions", "Strong", "Face", "clip —", "Ready")
    texts: list[str] = []
    for kw in (
        dict(face_presence=0.9, speaking_intensity=2.0, reaction_confidence=0.9,
             heuristic_reason="voice", duration_s=20.0, face_zone="top", target_platform="kick"),
        dict(face_presence=0.0, speaking_intensity=0.1, reaction_confidence=None,
             heuristic_reason="visual", duration_s=3.0, face_zone="lower", target_platform=None),
        dict(face_presence=0.3, speaking_intensity=0.5, reaction_confidence=0.2,
             heuristic_reason="audio", duration_s=12.0, face_zone=None, target_platform="tiktok"),
    ):
        d = ai_decisions.decide(**kw)
        texts += [d.hook_reason, d.secondary_hook_reason, d.subtitle_reason,
                  d.caption_position_reason, *d.intel_lines]
    for wps in (None, 0.2, 0.6, 1.5, 3.0):
        b = SimpleNamespace(
            clip_id="c", duration_s=10.0, motion_score=0.5, face_presence=0.7,
            speaking_intensity=wps, reaction_confidence=0.7, rescore_delta=0.1,
            rescore_reason=None, heuristic_reason="voice", heuristic_score=0.8,
            integrity_issues=(),
        )
        s = scoring.compute_ai_scores(b)  # type: ignore[arg-type]
        texts += [s.hook_why, s.readability_why, s.dead_air_why]
        for ov in ({}, {"title_text": "x", "captions": {"enabled": False}}):
            v = publishability.compute_publishability(
                breakdown=b, overlay_config=ov, safe_zone_platform="tiktok"  # type: ignore[arg-type]
            )
            texts += [*v.reasons, *v.warnings, *v.blocking, v.recommended_action]
            texts += [x.text for x in publishability.director_lines(breakdown=b, verdict=v)]  # type: ignore[arg-type]
    leftovers = [t for t in texts if any(m in localize(t) for m in english_markers)]
    assert leftovers == []
    assert localize("Standard short-form placement (under the face)", "en").startswith("Standard")


def test_timeline_labels_and_viral_trail() -> None:
    assert localize("Audio peak · 12 words") == "Pico de audio · 12 palabras"
    assert localize("Reaction · 1 laugh token") == "Reacción · 1 risa"
    assert localize("Scene cut") == "Cambio de escena"
    assert localize("Face emotion → happy") == "Emoción en la cara → feliz"
    assert localize('🎯 Trigger phrase heard: "clip it"') == "🎯 La IA te escuchó: «clip it»"
    assert localize("silence: 6s without speech at 0-6s") == "silencio de 6 s (del 0 al 6 s)"
    assert "base 55" in localize_viral_why("baseline 55 · face 0.70 → +8")
    assert localize("Something new the engine says") == "Something new the engine says"

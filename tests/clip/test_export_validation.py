"""Unit tests for the preview_recorder's validation + manifest helpers.

These tests cover the pure-Python pieces that don't need a real browser:

  - `_validate_export` correctly classifies duration / fps / codec drift
  - `_write_manifest` writes a JSON file alongside the export
  - duration tolerance enforcement matches the documented contract

The end-to-end recorder (Playwright + ffmpeg) is exercised via a
separate manual test plan documented in the slice O.54 commit
message; CI runs the unit layer here.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from chalybclip.clip.preview_recorder import (
    _DURATION_TOLERANCE_MS,
    _validate_export,
    _write_manifest,
)


def _make_ffprobe_response(
    duration_s: float,
    *,
    fps: float = 30.0,
    video_codec: str = "h264",
    pix_fmt: str = "yuv420p",
    audio_codec: str | None = "aac",
    audio_sample_rate: str | None = "48000",
) -> str:
    """Build a JSON payload that mimics ffprobe -show_format -show_streams."""
    streams: list[dict[str, object]] = [
        {
            "codec_type": "video",
            "codec_name": video_codec,
            "pix_fmt": pix_fmt,
            "avg_frame_rate": f"{int(fps)}/1",
        }
    ]
    if audio_codec is not None:
        streams.append({
            "codec_type": "audio",
            "codec_name": audio_codec,
            "sample_rate": audio_sample_rate,
        })
    return json.dumps({
        "format": {"duration": str(duration_s)},
        "streams": streams,
    })


def _fake_ffprobe(stdout: str, returncode: int = 0):
    """Construct a CompletedProcess that subprocess.run would return."""
    return subprocess.CompletedProcess(
        args=["ffprobe"], returncode=returncode, stdout=stdout, stderr=""
    )


# ---- _validate_export -------------------------------------------------------


def test_validate_export_happy_path(tmp_path: Path) -> None:
    """A perfectly-encoded clip should validate ok=True with zero drift."""
    out = tmp_path / "clip_render_1080.mp4"
    out.write_bytes(b"fake mp4")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(40.8))
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is True
    checks = result["checks"]
    assert checks["duration_drift_ms"] == 0
    assert checks["actual_fps"] == 30.0
    assert checks["video_codec"] == "h264"
    assert checks["audio_codec"] == "aac"
    assert checks["audio_sample_rate"] == "48000"


def test_validate_export_duration_within_tolerance(tmp_path: Path) -> None:
    """Up to 150ms drift is acceptable (within _DURATION_TOLERANCE_MS)."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        # 40.8 expected, 40.85 actual = 50ms drift
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(40.85))
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is True
    assert result["checks"]["duration_drift_ms"] == 50


def test_validate_export_duration_drift_fails(tmp_path: Path) -> None:
    """Drift > 150ms must fail validation."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        # 40.8 expected, 41.5 actual = 700ms drift
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(41.5))
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert result["checks"]["duration_drift_ms"] == 700
    assert any("drift" in e for e in result["errors"])


def test_validate_export_fps_mismatch_fails(tmp_path: Path) -> None:
    """A 60fps output when we asked for 30 must fail validation."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(
            _make_ffprobe_response(40.8, fps=60.0)
        )
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert result["checks"]["actual_fps"] == 60.0
    assert any("fps" in e for e in result["errors"])


def test_validate_export_wrong_video_codec_fails(tmp_path: Path) -> None:
    """If somehow we emit something other than H.264, validation must fail."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(
            _make_ffprobe_response(40.8, video_codec="hevc")
        )
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert result["checks"]["video_codec"] == "hevc"


def test_validate_export_missing_audio_is_informational(tmp_path: Path) -> None:
    """Clips without audio still validate ok=True; codec is None in checks."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(
            _make_ffprobe_response(40.8, audio_codec=None)
        )
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is True
    assert result["checks"]["audio_codec"] is None


def test_validate_export_ffprobe_failure(tmp_path: Path) -> None:
    """A non-zero ffprobe exit must mark the export not-ok."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe("", returncode=1)
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert any("ffprobe" in e for e in result["errors"])


def test_validate_export_ffprobe_exception(tmp_path: Path) -> None:
    """If subprocess.run raises (binary missing, permission), surface it."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run", side_effect=FileNotFoundError("no ffprobe")):
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert any("exception" in e for e in result["errors"])


# ---- _write_manifest --------------------------------------------------------


def test_write_manifest_writes_sibling_json(tmp_path: Path) -> None:
    """Manifest lands at <output>.manifest.json with the full payload."""
    out = tmp_path / "clip_render_1080.mp4"
    out.write_bytes(b"x")
    payload: dict[str, object] = {
        "clip_id": "clp_abc",
        "input": {"duration_s": 40.8},
        "validation": {"ok": True, "errors": []},
    }
    _write_manifest(output_path=out, manifest=payload)

    manifest_path = out.with_suffix(".manifest.json")
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == payload


def test_write_manifest_survives_oserror(tmp_path: Path) -> None:
    """Manifest write failures must not raise — they only log."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
        # Must not raise.
        _write_manifest(output_path=out, manifest={"a": 1})


# ---- contract -------------------------------------------------------------


def test_duration_tolerance_documented() -> None:
    """The tolerance is part of the public contract — guard it from
    accidental loosening. The operator's spec requires <= 100ms
    end-to-end. If you change this, update the docstring + spec docs."""
    assert _DURATION_TOLERANCE_MS == 100


# ---- _ffprobe_duration_s (slice O.55) --------------------------------------


def test_ffprobe_duration_returns_seconds(tmp_path: Path) -> None:
    """The probe should parse ffprobe -show_format JSON into seconds."""
    from chalybclip.clip.preview_recorder import _ffprobe_duration_s

    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"x")
    with patch("subprocess.run") as mock_run, \
         patch("shutil.which", return_value="ffprobe"):
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(40.832))
        assert _ffprobe_duration_s(fake_file) == 40.832


def test_ffprobe_duration_no_binary_returns_none(tmp_path: Path) -> None:
    """Missing ffprobe binary must NOT raise — return None so callers
    fall back to the DB duration."""
    from chalybclip.clip.preview_recorder import _ffprobe_duration_s

    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"x")
    with patch("shutil.which", return_value=None):
        assert _ffprobe_duration_s(fake_file) is None


def test_ffprobe_duration_nonzero_rc_returns_none(tmp_path: Path) -> None:
    """An ffprobe error must NOT raise — same fall-back contract."""
    from chalybclip.clip.preview_recorder import _ffprobe_duration_s

    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"x")
    with patch("subprocess.run") as mock_run, \
         patch("shutil.which", return_value="ffprobe"):
        mock_run.return_value = _fake_ffprobe("", returncode=1)
        assert _ffprobe_duration_s(fake_file) is None


def test_ffprobe_duration_zero_returns_none(tmp_path: Path) -> None:
    """A 0-second file isn't usable as canonical duration — None
    forces the caller back to the DB value rather than producing a
    zero-frame export."""
    from chalybclip.clip.preview_recorder import _ffprobe_duration_s

    fake_file = tmp_path / "clip.mp4"
    fake_file.write_bytes(b"x")
    with patch("subprocess.run") as mock_run, \
         patch("shutil.which", return_value="ffprobe"):
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(0.0))
        assert _ffprobe_duration_s(fake_file) is None


# ---- 40s preview / 35s export root cause (slice O.55) ----------------------


def test_validate_export_uses_canonical_duration(tmp_path: Path) -> None:
    """If the file is 35s but caller expected 40s and DOESN'T pre-
    reconcile via _ffprobe_duration_s, validation must surface the
    mismatch as a failure (this is the bug the operator hit)."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        # File is 35s. Expected 40s. Drift = 5000ms.
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(35.0))
        result = _validate_export(
            output_path=out, expected_duration_s=40.0, expected_fps=30
        )

    assert result["ok"] is False
    assert result["checks"]["duration_drift_ms"] == 5000
    assert any("drift" in e for e in result["errors"])


def test_validate_export_passes_when_caller_pre_reconciles(tmp_path: Path) -> None:
    """The slice O.55 fix: caller probes the file first and passes
    file-derived duration into validate. With expected aligned to
    actual, drift is 0 -> ok=True."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(35.0))
        # Caller already probed source clip + got 35.0, passes that.
        result = _validate_export(
            output_path=out, expected_duration_s=35.0, expected_fps=30
        )

    assert result["ok"] is True
    assert result["checks"]["duration_drift_ms"] == 0


def test_audio_must_be_aac_48k_for_social(tmp_path: Path) -> None:
    """Operator spec: output must be H.264 + AAC + 48kHz for TikTok
    / Reels / Shorts compatibility. The validator records the audio
    codec + sample rate so the manifest can be diffed; this test
    pins the contract on what gets stored."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(40.0))
        result = _validate_export(
            output_path=out, expected_duration_s=40.0, expected_fps=30
        )

    assert result["checks"]["audio_codec"] == "aac"
    assert result["checks"]["audio_sample_rate"] == "48000"


# ---- banner safe zone (slice O.55 / Issue C) ------------------------------


def test_clip_render_banner_is_inside_safe_zone() -> None:
    """The Kick banner CSS must place the banner above the social UI
    bottom strip. Operator spec for 1080x1920:
      - bottom safe area is 360px (-> banner cannot live below y=1560)
      - we want banner top edge around y=1380 (kick variant)
      - we want banner top edge around y=1500 (default variant)

    These translate to cqh values: 28.125 / 21.875 respectively
    (28.125% of 1920 = 540px from bottom = banner top at y=1380).
    If anybody changes the banner CSS without updating this test
    they'll get a failure that points at the social-cropping bug.

    The banner CSS was merged into the shared _overlay_styles.html
    partial (clip_render.html now {% include %}s it), so that's the
    file the assertions read."""
    template_path = (
        Path(__file__).resolve().parents[2]
        / "chalybclip" / "api" / "templates" / "_overlay_styles.html"
    )
    css = template_path.read_text(encoding="utf-8")

    # Default banner must NOT be at bottom: 0 anymore.
    assert "bottom: 21.875cqh" in css, (
        "default .nc-pv-banner must sit at bottom: 21.875cqh "
        "(banner top edge y=1500 on 1920 canvas) so it stays "
        "above social-platform UI overlays"
    )
    # Kick variant must be higher still (it's a richer banner).
    assert "bottom: 28.125cqh" in css, (
        ".nc-pv-banner--kick_repost_page must sit at bottom: 28.125cqh "
        "(banner top edge y=1380 on 1920 canvas)"
    )
    # Left/right margins for the safe-zone (80px on a 1080 canvas = 7.4cqw).
    assert "left: 7.4cqw" in css, (
        "banner must have at least 80px left margin (7.4cqw) so the "
        "Kick logo never touches the canvas edge"
    )
    assert "right: 7.4cqw" in css, (
        "banner must have at least 80px right margin (7.4cqw) so the "
        "URL handle never touches the canvas edge"
    )

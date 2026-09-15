"""Tests for the end-of-clip chalybclip splash (`chalybclip.clip.outro`).

The outro is the bundled `assets/outro.mp4` concatenated after each
exported clip. These cover the (pure) command builder across the audio
combinations and the `append_outro` orchestration / failure handling.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from chalybclip.clip.outro import (
    append_outro,
    build_outro_command,
    outro_asset_path,
)


def _cmd(**over):
    base = dict(
        source_path=Path("clip.mp4"),
        asset_path=Path("outro.mp4"),
        out_path=Path("out.mp4"),
        width=1080,
        height=1920,
        fps=30,
        source_has_audio=True,
        asset_has_audio=True,
        source_duration_s=20.0,
        asset_duration_s=1.6,
    )
    base.update(over)
    return build_outro_command(**base)


# ---- the bundled asset ----------------------------------------------


def test_bundled_asset_exists() -> None:
    """The end card ships with the app — the whole feature is moot
    without it on disk. Guard against the 22-byte placeholder stub
    silently shipping again (it can't be probed, so append_outro
    no-ops and clips lose the outro): a real MP4 is comfortably > 1 KB."""
    asset = outro_asset_path()
    assert asset.exists(), "assets/outro.mp4 must be bundled"
    assert asset.stat().st_size > 1024, (
        "assets/outro.mp4 looks like a placeholder, not a real video — "
        "append_outro will silently no-op on an unprobeable file"
    )


# ---- build_outro_command (pure) -------------------------------------


def test_both_segments_have_audio() -> None:
    cmd = _cmd(source_has_audio=True, asset_has_audio=True)
    joined = " ".join(cmd)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1[v][a]" in graph
    # Real audio from both inputs, no synthesized silence.
    assert "[0:a]aformat" in graph
    assert "[1:a]aformat" in graph
    assert "anullsrc" not in joined
    assert ["-map", "[v]", "-map", "[a]"] == cmd[cmd.index("-map"):cmd.index("-map") + 4]
    assert "aac" in cmd
    # Both clips are the only real -i inputs.
    assert cmd.count("-i") == 2


def test_clip_audio_asset_silent_synthesizes_one_tail() -> None:
    cmd = _cmd(source_has_audio=True, asset_has_audio=False)
    joined = " ".join(cmd)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1[v][a]" in graph
    assert "[0:a]aformat" in graph  # clip's real audio
    # Exactly one synthesized silent track, of the asset's duration.
    assert joined.count("anullsrc") == 1
    assert "-t 1.6" in joined


def test_clip_silent_asset_audio_synthesizes_clip_silence() -> None:
    cmd = _cmd(source_has_audio=False, asset_has_audio=True)
    joined = " ".join(cmd)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1[v][a]" in graph
    assert "[1:a]aformat" in graph  # asset's real audio
    assert joined.count("anullsrc") == 1
    assert "-t 20.0" in joined  # silence spans the clip's duration


def test_neither_has_audio_is_video_only() -> None:
    cmd = _cmd(source_has_audio=False, asset_has_audio=False)
    joined = " ".join(cmd)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0[v]" in graph
    assert "anullsrc" not in joined
    assert "[a]" not in cmd
    assert "aac" not in cmd


def test_both_segments_normalized_to_clip_geometry() -> None:
    cmd = _cmd(width=720, height=1280, fps=24)
    graph = cmd[cmd.index("-filter_complex") + 1]
    # Both video segments get the same fps/scale/pad normalization.
    assert graph.count("scale=720:1280") == 2
    assert graph.count("fps=24") == 2


# ---- append_outro (orchestration) -----------------------------------


def test_append_outro_replaces_file_on_success(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "clip_final.mp4"
    video.write_bytes(b"original")

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "chalybclip.clip.outro.outro_asset_path", lambda: tmp_path / "asset.mp4"
    )
    (tmp_path / "asset.mp4").write_bytes(b"asset")
    monkeypatch.setattr(
        "chalybclip.clip.outro._ffprobe_meta",
        lambda _p: (1080, 1920, 30, True, 5.0),
    )

    captured: list[list[str]] = []

    def fake_run(cmd, capture_output, check):
        captured.append(cmd)
        Path(cmd[-1]).write_bytes(b"with-outro")  # pretend ffmpeg wrote it
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("chalybclip.clip.outro.subprocess.run", fake_run)

    assert append_outro(video) is True
    assert video.read_bytes() == b"with-outro"
    assert not (tmp_path / "clip_final.outro.mp4").exists()
    assert captured, "ffmpeg should have been invoked"


def test_append_outro_no_asset_is_noop(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"original")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "chalybclip.clip.outro.outro_asset_path", lambda: tmp_path / "missing.mp4"
    )

    assert append_outro(video) is False
    assert video.read_bytes() == b"original"


def test_append_outro_ffmpeg_failure_leaves_original(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"original")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "chalybclip.clip.outro.outro_asset_path", lambda: tmp_path / "asset.mp4"
    )
    (tmp_path / "asset.mp4").write_bytes(b"asset")
    monkeypatch.setattr(
        "chalybclip.clip.outro._ffprobe_meta",
        lambda _p: (1080, 1920, 30, True, 5.0),
    )

    def fake_run(cmd, capture_output, check):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout=b"", stderr=b"boom"
        )

    monkeypatch.setattr("chalybclip.clip.outro.subprocess.run", fake_run)

    assert append_outro(video) is False
    assert video.read_bytes() == b"original"
    assert not (tmp_path / "clip.outro.mp4").exists()


def test_append_outro_missing_file_is_noop(tmp_path: Path) -> None:
    assert append_outro(tmp_path / "nope.mp4") is False

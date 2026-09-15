"""R16 parity — the legacy fallback recorder must burn ASS captions.

The hybrid recorder burns captions via ffmpeg's libass filter
(`ass='<path>'` inside its composite filtergraph). When the hybrid
path raises and _clip_render.py falls back to the legacy seek-and-
shoot recorder, that encode previously had NO ass plumbing at all —
so the fallback shipped a caption-less MP4 (operator saw captions in
the preview; the published file had none). These tests pin the legacy
encode command shape: `-vf ass='<escaped path>'` present exactly when
an ASS file is provided AND exists, absent otherwise, using the SAME
escape helper the hybrid composite uses so the two can't drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.clip import preview_recorder
from chalybclip.clip.captions_ass import escape_ass_path_for_filter
from chalybclip.clip.preview_recorder import _encode_image_sequence


class _Proc:
    returncode = 0
    stderr = b""


def _install_fake_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, captured: dict,
) -> None:
    """Patch shutil.which + subprocess.run for the encode step. The
    patch on preview_recorder.subprocess.run is GLOBAL (it's the stdlib
    module) so the fake only writes inside the test sandbox — never to
    repo paths like the bundled outro asset (see the same guard in
    test_hybrid_recorder.py)."""
    monkeypatch.setattr(
        preview_recorder.shutil, "which", lambda _: "/usr/bin/ffmpeg",
    )

    def fake_run(cmd: list[str], *args, **kwargs):
        captured.setdefault("cmds", []).append(cmd)
        out = Path(cmd[-1])
        if tmp_path in out.parents:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00fake mp4")
        return _Proc()

    monkeypatch.setattr(preview_recorder.subprocess, "run", fake_run)


async def test_encode_burns_ass_filter_when_provided(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """With an ASS file on disk, the encode cmd carries `-vf ass='…'`
    with the shared escaping applied — the exact fix for the
    caption-less fallback renders."""
    captured: dict = {}
    _install_fake_ffmpeg(monkeypatch, tmp_path, captured)

    frames = tmp_path / "frames"
    frames.mkdir()
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"\x00src")
    ass = tmp_path / "captions.ass"
    ass.write_text("[Script Info]\n", encoding="utf-8")
    out = tmp_path / "clip_render_1080.mp4"

    await _encode_image_sequence(
        frames_dir=frames,
        audio_source_path=audio,
        output_path=out,
        duration_s=10.0,
        fps=30,
        ass_file_path=ass,
        # Command-shape test only; the outro step is covered by
        # tests/clip/test_outro.py.
        append_outro_enabled=False,
    )

    cmd = captured["cmds"][0]
    assert "-vf" in cmd, "legacy encode dropped the caption burn again"
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == f"ass='{escape_ass_path_for_filter(ass)}'"
    assert out.exists()


async def test_encode_skips_ass_filter_when_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No ASS file (captions disabled / no word-level data) → no -vf;
    the encode is byte-identical to the pre-fix command."""
    captured: dict = {}
    _install_fake_ffmpeg(monkeypatch, tmp_path, captured)

    frames = tmp_path / "frames"
    frames.mkdir()
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"\x00src")
    out = tmp_path / "clip_render_1080.mp4"

    await _encode_image_sequence(
        frames_dir=frames,
        audio_source_path=audio,
        output_path=out,
        duration_s=10.0,
        fps=30,
        ass_file_path=None,
        append_outro_enabled=False,
    )

    cmd = captured["cmds"][0]
    assert "-vf" not in cmd
    assert not any("ass=" in str(x) for x in cmd)


async def test_encode_skips_ass_filter_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An ASS path whose file vanished (build failed upstream, disk
    cleanup) degrades to a caption-less encode instead of handing
    ffmpeg a dangling filter path → hard rc!=0 failure."""
    captured: dict = {}
    _install_fake_ffmpeg(monkeypatch, tmp_path, captured)

    frames = tmp_path / "frames"
    frames.mkdir()
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"\x00src")
    out = tmp_path / "clip_render_1080.mp4"

    await _encode_image_sequence(
        frames_dir=frames,
        audio_source_path=audio,
        output_path=out,
        duration_s=10.0,
        fps=30,
        ass_file_path=tmp_path / "never_written.ass",
        append_outro_enabled=False,
    )

    assert "-vf" not in captured["cmds"][0]


def test_escape_helper_handles_windows_style_paths() -> None:
    """`\\` then `:` then `'` — the ffmpeg filter-arg escaping rules the
    hybrid composite documented; now shared so both recorders agree."""
    p = Path(r"C:\data\o'ut\captions.ass")
    assert escape_ass_path_for_filter(p) == r"C\:\\data\\o\'ut\\captions.ass"


def test_record_clip_to_mp4_accepts_ass_file_path() -> None:
    """Signature pin — the fallback call site in _clip_render.py passes
    ass_file_path=…; if the parameter is ever dropped the fallback
    render crashes with TypeError instead of shipping without captions,
    and this test names the regression first."""
    import inspect

    params = inspect.signature(
        preview_recorder.record_clip_to_mp4
    ).parameters
    assert "ass_file_path" in params
    assert params["ass_file_path"].default is None

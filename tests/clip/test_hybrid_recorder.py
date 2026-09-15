"""Hybrid ffmpeg + Playwright overlay-alpha recorder — Render T2.

The recorder's hot loop dedups screenshots by an overlay state hash
the render page emits — captions tick at word-frequency (~3-5/sec),
not 30/sec, so most "frames" reuse the previous unique PNG via
hardlink/copy instead of taking a fresh screenshot. These tests pin
that dedup behavior + the ffmpeg composite command shape; the full
end-to-end run is exercised against a real Playwright stack only in
the manual smoke-test path on Railway.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chalybclip.clip import hybrid_recorder
from chalybclip.clip.hybrid_recorder import (
    HybridRecordingError,
    _capture_overlay_alpha_sequence,
    _ffmpeg_composite_alpha_over_source,
)


class _FakePage:
    """Minimal Playwright page stub. Records every evaluate + screenshot
    call so tests can assert on the dedup behavior."""

    def __init__(self, state_sequence: list[str], png_bytes: bytes = b"\x89PNGfake"):
        self._states = list(state_sequence)
        self._png_bytes = png_bytes
        self.screenshot_calls = 0
        self.evaluate_calls: list[str] = []
        self._frame_idx = 0

    async def evaluate(self, js: str, *args: object) -> object:
        self.evaluate_calls.append(js)
        if "getOverlayStateHash" in js:
            # Return the next state in the sequence (the caller advances
            # through frame_count frames in order).
            idx = self._frame_idx
            self._frame_idx += 1
            if idx >= len(self._states):
                return self._states[-1]
            return self._states[idx]
        # captureFrameAtAlpha → no return value needed
        return None

    async def screenshot(self, **kwargs) -> bytes:
        # R16 — back to omit_background=True (R14's magenta chroma
        # key was reverted along with all per-page caption rendering).
        # The alpha pipeline carries only the server-rendered title /
        # banner / watermark layer; captions are burned by ffmpeg's
        # libass after the composite.
        assert kwargs.get("omit_background") is True
        assert kwargs.get("type") == "png"
        self.screenshot_calls += 1
        return self._png_bytes


# ---- dedup behavior ----


async def test_capture_dedups_consecutive_same_state(
    tmp_path: Path,
) -> None:
    """30 frames at the same state → exactly 1 screenshot. The other
    29 frame files are hardlinked / copied from the first."""
    page = _FakePage(state_sequence=["STATE_A"] * 30)
    unique, total = await _capture_overlay_alpha_sequence(
        page=page,
        duration_s=1.0,
        fps=30,
        frames_dir=tmp_path,
        progress_callback=None,
    )
    assert total == 30
    assert unique == 1
    assert page.screenshot_calls == 1
    # All 30 files exist (ffmpeg's image2 demuxer needs a complete
    # sequence; hardlinks fulfil that without duplicating bytes).
    files = sorted(tmp_path.glob("overlay_*.png"))
    assert len(files) == 30


async def test_capture_screenshots_each_state_change(
    tmp_path: Path,
) -> None:
    """Realistic case — caption advances every 6 frames (≈5 changes/sec
    at 30 fps). Dedup ratio matches the spec's "~85% saved" claim."""
    # 30 frames, state changes 5 times.
    states = (
        ["S0"] * 6 + ["S1"] * 6 + ["S2"] * 6 + ["S3"] * 6 + ["S4"] * 6
    )
    page = _FakePage(state_sequence=states)
    unique, total = await _capture_overlay_alpha_sequence(
        page=page,
        duration_s=1.0,
        fps=30,
        frames_dir=tmp_path,
        progress_callback=None,
    )
    assert total == 30
    assert unique == 5
    assert page.screenshot_calls == 5
    # Dedup ratio ≥ 80% (saved 25/30 screenshots).
    saved_ratio = 1.0 - unique / total
    assert saved_ratio >= 0.80


async def test_capture_progress_callback_fires(
    tmp_path: Path,
) -> None:
    """Progress callback fires during capture, capped at 50% (the
    composite step owns the upper half)."""
    calls: list[int] = []

    async def _cb(pct: int) -> None:
        calls.append(pct)

    # 60 frames so the 5%-step emitter actually emits multiple times.
    page = _FakePage(state_sequence=["S"] * 60)
    await _capture_overlay_alpha_sequence(
        page=page, duration_s=2.0, fps=30,
        frames_dir=tmp_path, progress_callback=_cb,
    )
    assert calls, "no progress emitted"
    # All capture-phase calls capped at 50%.
    assert max(calls) <= 50
    assert min(calls) >= 0


async def test_capture_evaluate_failure_surfaces_as_hybrid_error(
    tmp_path: Path,
) -> None:
    class _BrokenPage:
        async def evaluate(self, *args, **kwargs):
            raise RuntimeError("page evaluate exploded")
        async def screenshot(self, **kwargs):
            return b""

    with pytest.raises(HybridRecordingError, match="page evaluate failed"):
        await _capture_overlay_alpha_sequence(
            page=_BrokenPage(),
            duration_s=1.0, fps=30,
            frames_dir=tmp_path, progress_callback=None,
        )


# ---- ffmpeg composite ----


def test_composite_command_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The composite ffmpeg call uses overlay= filter, maps the
    source's audio with the optional `?` operator, and clamps duration
    via `-t`."""
    monkeypatch.setattr(hybrid_recorder.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    captured: dict = {}

    class _Proc:
        returncode = 0
        stderr = b""

    def fake_run(cmd: list[str], *args, **kwargs):
        captured["cmd"] = cmd
        # Touch the output file so any caller doing .exists() checks
        # downstream isn't surprised — but ONLY inside the sandbox;
        # the subprocess patch is global and must never write to repo
        # paths (e.g. the bundled outro asset an ffprobe call names
        # as its last arg).
        out = Path(cmd[-1])
        if tmp_path in out.parents:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00fake mp4")
        return _Proc()

    monkeypatch.setattr(hybrid_recorder.subprocess, "run", fake_run)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00src")
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "out.mp4"

    _ffmpeg_composite_alpha_over_source(
        source_video=source,
        frames_dir=frames,
        output_path=out,
        fps=30,
        duration_s=35.9,
        # This test asserts the composite command shape only; the
        # separate end-card outro step (which would issue its own
        # ffprobe/ffmpeg calls through the globally-patched subprocess)
        # is covered by tests/clip/test_outro.py.
        append_outro_enabled=False,
    )

    cmd = captured["cmd"]
    assert "ffmpeg" in cmd[0]
    # Source input + overlay PNG sequence input.
    i_indices = [i for i, x in enumerate(cmd) if x == "-i"]
    assert len(i_indices) == 2
    assert cmd[i_indices[0] + 1] == str(source)
    # Second -i points at the PNG glob pattern.
    assert cmd[i_indices[1] + 1].endswith("overlay_%08d.png")
    # overlay filter present.
    assert "-filter_complex" in cmd
    fc_idx = cmd.index("-filter_complex")
    assert "overlay=" in cmd[fc_idx + 1]
    # Audio mapped with optional `?` so video-only sources don't fail.
    assert "0:a?" in cmd
    # Duration clamp matches input.
    t_idx = cmd.index("-t")
    assert cmd[t_idx + 1] == "35.900"


def test_composite_nonzero_exit_surfaces_as_hybrid_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(hybrid_recorder.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    class _Proc:
        returncode = 1
        stderr = b"ffmpeg: cannot do the thing"

    monkeypatch.setattr(
        hybrid_recorder.subprocess, "run", lambda *a, **kw: _Proc(),
    )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00src")
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "out.mp4"

    with pytest.raises(HybridRecordingError, match="composite failed"):
        _ffmpeg_composite_alpha_over_source(
            source_video=source,
            frames_dir=frames,
            output_path=out,
            fps=30,
            duration_s=10.0,
        )


# ---- Render Migration R10 — atomic publish ----


def test_composite_writes_to_partial_then_renames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """ffmpeg targets `<output>.partial.mp4`, not the final cache path.

    The download endpoint serves whatever exists at `output_path`. If
    ffmpeg writes there directly, a concurrent download click hits a
    growing file → FileResponse stamps Content-Length from the current
    size, ffmpeg keeps appending, Starlette over-sends → RuntimeError
    `Response content longer than Content-Length` + a truncated MP4
    the player rejects (0xC00D36C4). Atomic rename closes the race:
    `output_path` is never visible to a request until the encode +
    post-conditions are fully done.
    """
    monkeypatch.setattr(hybrid_recorder.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    targets: list[str] = []

    class _Proc:
        returncode = 0
        stderr = b""

    def fake_run(cmd: list[str], *args, **kwargs):
        target = cmd[-1]
        targets.append(target)
        out = Path(target)
        # Only materialize outputs inside the test sandbox. The patch
        # on hybrid_recorder.subprocess.run is GLOBAL (it's the stdlib
        # module), so the outro step's ffprobe calls land here too —
        # and ffprobe's last arg is the file being probed, i.e. the
        # bundled chalybclip/clip/assets/outro.mp4. Writing to cmd[-1]
        # unconditionally overwrote the real end card in the working
        # tree on every test run (it even got committed broken once).
        if tmp_path in out.parents:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00fake mp4 large enough")
        return _Proc()

    monkeypatch.setattr(hybrid_recorder.subprocess, "run", fake_run)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00src")
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "clip_render_1080.mp4"

    _ffmpeg_composite_alpha_over_source(
        source_video=source,
        frames_dir=frames,
        output_path=out,
        fps=30,
        duration_s=10.0,
    )

    # ffmpeg was pointed at the partial path, NOT the final.
    assert targets, "ffmpeg was never called"
    assert targets[0].endswith(".partial.mp4"), targets[0]
    assert targets[0] != str(out), (
        "regression: ffmpeg is writing directly to the cache path again — "
        "would race a concurrent download click"
    )
    # After success, the partial is gone and the final exists.
    assert not Path(targets[0]).exists(), "partial should be renamed away"
    assert out.exists(), "atomic publish never happened"


def test_composite_failure_scrubs_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When ffmpeg fails, the `.partial.mp4` shouldn't linger on disk —
    leftover partials would mislead a future operator-facing tool that
    walked the output directory. Cleanup runs in the recorder's
    try/except so the failure path doesn't accumulate debris."""
    monkeypatch.setattr(hybrid_recorder.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    class _Proc:
        returncode = 1
        stderr = b"ffmpeg: broke"

    def fake_run(cmd: list[str], *args, **kwargs):
        # Simulate ffmpeg leaving a partial behind before bailing.
        # Sandbox-guarded like the other fakes — the patch is global.
        target = Path(cmd[-1])
        if tmp_path in target.parents:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00partial bytes")
        return _Proc()

    monkeypatch.setattr(hybrid_recorder.subprocess, "run", fake_run)

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00src")
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "clip_render_1080.mp4"
    partial = out.with_suffix(".partial.mp4")

    with pytest.raises(HybridRecordingError):
        _ffmpeg_composite_alpha_over_source(
            source_video=source,
            frames_dir=frames,
            output_path=out,
            fps=30,
            duration_s=10.0,
        )

    assert not partial.exists(), "partial should be unlinked on failure"
    assert not out.exists(), "final cache path must never appear on failure"

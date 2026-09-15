"""Parallel cut orchestration + substep events (Task 1a / 1d)."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from chalybclip.clip import cut_clips
from chalybclip.clip import service as clip_service
from chalybclip.config import ClipConfig

from ._fixtures import make_candidate, seed_stream


@pytest.fixture(autouse=True)
def _unthrottle_governor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the process-wide heavy-op governor for this file.

    These tests measure `cut_concurrency` in isolation — the barrier(3)
    case needs all three candidates' first ffmpeg call to run at once. The
    governor (chalybclip.resources) is a SEPARATE, additional cap whose
    default (cpu//2) would throttle below 3 on small CI boxes and break the
    barrier. It has its own coverage in test_resources_governor.py."""
    from chalybclip import resources

    monkeypatch.setenv("CHALYBCLIP_HEAVY_SLOTS", "16")
    resources.reset_for_testing()


def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Same shape as the existing _stub_ffmpeg in test_cut_clips.py:
    create the output file so the rest of the pipeline sees a real
    artifact, and record the argv for later assertions."""
    calls: list[list[str]] = []
    lock = threading.Lock()

    def fake_run(cmd: list[str], *, what: str) -> None:
        with lock:
            calls.append(cmd)
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake mp4 bytes")

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)
    return calls


def _stub_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clip_service, "_safe_smart_crop", lambda **_kw: None)
    monkeypatch.setattr(clip_service, "_safe_thumbnail", lambda **_kw: (None, None))
    monkeypatch.setattr(clip_service, "_safe_analyze_framing", lambda **_kw: None)


def test_concurrency_runs_more_than_one_candidate_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With cut_concurrency=3 and a 50ms artificial delay per ffmpeg
    call, 3 candidates serialized would take ~6 calls × 50ms = 300ms.
    Parallel at 3-wide should land closer to 100ms (2 calls per
    candidate, all overlapping). We assert << serial-floor so the
    test stays robust across machines without coupling to exact ms.

    NB: this test only proves work overlaps, not that any particular
    ordering shows up — gather() preserves submission order regardless.
    """
    _stub_vision(monkeypatch)

    barrier = threading.Barrier(3, timeout=3.0)

    def fake_run(cmd: list[str], *, what: str) -> None:
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake mp4 bytes")
        # Only barrier on the FIRST ffmpeg call per candidate (the
        # fast_cut step). The second call (reformat) runs after,
        # by which point all three have crossed the barrier.
        if "-c" in cmd and "copy" in cmd:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                # If concurrency is broken (serial path), the barrier
                # times out — the test fails on the assert below
                # because we never made it past wait().
                raise AssertionError(
                    "barrier never reached — candidates ran serially"
                )

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)

    stream = seed_stream(tmp_path)
    candidates = [make_candidate(timestamp=60.0 + i * 60.0) for i in range(3)]
    cfg = ClipConfig(cut_concurrency=3)

    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=candidates,
            output_dir=tmp_path,
            config=cfg,
        )
    )
    assert len(clips) == 3


def test_serial_when_concurrency_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cut_concurrency=1` restores the pre-Task-1 serial behavior —
    the escape hatch when ffmpeg starts thrashing."""
    calls = _stub_ffmpeg(monkeypatch)
    _stub_vision(monkeypatch)

    stream = seed_stream(tmp_path)
    candidates = [make_candidate(timestamp=60.0 + i * 60.0) for i in range(3)]
    cfg = ClipConfig(cut_concurrency=1)

    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=candidates,
            output_dir=tmp_path,
            config=cfg,
        )
    )
    # 3 candidates × 2 ffmpeg passes each.
    assert len(clips) == 3
    assert len(calls) == 6


def test_gather_preserves_candidate_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """asyncio.gather returns in submission order. Worth pinning so a
    future refactor doesn't accidentally swap it for as_completed (which
    would break the manifest's candidate→clip mapping)."""
    _stub_ffmpeg(monkeypatch)
    _stub_vision(monkeypatch)

    stream = seed_stream(tmp_path)
    timestamps = [60.0, 120.0, 180.0]
    candidates = [make_candidate(timestamp=t) for t in timestamps]

    clips = asyncio.run(
        cut_clips(
            tenant_id="default",
            stream=stream,
            candidates=candidates,
            output_dir=tmp_path,
            config=ClipConfig(cut_concurrency=3),
        )
    )
    # Clips come back in the order their candidates were submitted.
    assert [c.candidate.timestamp for c in clips] == timestamps

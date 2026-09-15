"""Auto-trim must keep working when the source VOD has been pruned.

The retention sweeper deletes the source video after a while, but the
clip's own 9:16 MP4 lives on. Since auto-trim only ever keeps a clean
*sub-window* of the clip, we can slice it straight out of the existing
clip MP4 — no source needed. These tests pin that fallback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from chalybclip.clip import service as clip_service
from chalybclip.clip import trim as trim_mod
from chalybclip.clip.trim import (
    auto_trim_around_integrity,
    revert_trim,
)
from chalybclip.errors import ClipError


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Insulate from the suite-wide structlog leak: an earlier test binds a
    PrintLogger to a pytest-captured stream that later closes, so any code
    here that calls `_log.info()` would crash. Reset to defaults first."""
    import structlog

    structlog.reset_defaults()


@dataclass
class _FakeClip:
    id: str
    stream_id: str
    path: Path
    start_s: float
    end_s: float
    duration_s: float
    original_start_s: float | None = None
    original_end_s: float | None = None


@dataclass
class _FakeStream:
    id: str
    source_video_path: Path


@dataclass
class _FakeClipsRepo:
    clip: _FakeClip
    trim_calls: list[dict] = field(default_factory=list)

    def __init__(self, _db: object) -> None:  # repo(db) ctor shape
        self.clip = _FAKE_STATE["clip"]
        self.trim_calls = _FAKE_STATE["trim_calls"]

    async def get(self, clip_id: str) -> _FakeClip | None:
        return self.clip if clip_id == self.clip.id else None

    async def set_trim_bounds(self, clip_id: str, **kwargs: object) -> None:
        self.trim_calls.append({"clip_id": clip_id, **kwargs})


@dataclass
class _FakeStreamsRepo:
    def __init__(self, _db: object) -> None:
        self.stream = _FAKE_STATE["stream"]

    async def get(self, stream_id: str) -> _FakeStream | None:
        return self.stream if stream_id == self.stream.id else None


# Module-level state the fake repos read from (they're instantiated inside
# trim.py as Repo(db), so we can't pass them in directly).
_FAKE_STATE: dict[str, object] = {}


def _wire_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    clip: _FakeClip,
    stream: _FakeStream,
) -> list[dict]:
    trim_calls: list[dict] = []
    _FAKE_STATE.clear()
    _FAKE_STATE.update(clip=clip, stream=stream, trim_calls=trim_calls)
    monkeypatch.setattr(trim_mod, "ClipsRepo", _FakeClipsRepo)
    monkeypatch.setattr(trim_mod, "StreamsRepo", _FakeStreamsRepo)
    return trim_calls


def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, what: str) -> None:
        calls.append(cmd)
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake mp4 bytes")

    monkeypatch.setattr(clip_service, "_run_ffmpeg", fake_run)
    # Force the encoder picker down the libx264 path so the unit test never
    # shells out to probe for NVENC (a real subprocess + log that's flaky
    # under the full-suite's shared structlog/encoder-cache state).
    monkeypatch.setattr("chalybclip.clip.encoders.has_nvenc", lambda: False)
    return calls


_ISSUES = [
    {"kind": "freeze", "start_s": 15.0, "end_s": 24.0, "label": "freeze"},
    {"kind": "freeze", "start_s": 25.0, "end_s": 40.0, "label": "freeze"},
]


def test_auto_trim_falls_back_to_clip_mp4_when_vod_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_dir = tmp_path / "clp_X"
    clip_dir.mkdir()
    clip_mp4 = clip_dir / "clip.mp4"
    clip_mp4.write_bytes(b"existing clip bytes")

    clip = _FakeClip(
        id="clp_X",
        stream_id="str_X",
        path=clip_mp4,
        start_s=0.0,
        end_s=40.0,
        duration_s=40.0,
    )
    # Source VOD path that does NOT exist — the sweeper pruned it.
    stream = _FakeStream(id="str_X", source_video_path=tmp_path / "gone" / "vod.mp4")

    trim_calls = _wire_fakes(monkeypatch, clip=clip, stream=stream)
    ffmpeg_calls = _stub_ffmpeg(monkeypatch)

    result = asyncio.run(
        auto_trim_around_integrity(
            object(), clip_id="clp_X", integrity_issues=_ISSUES
        )
    )

    # The clean window is [0, 15] — kept and persisted.
    assert result["outcome"] == "trimmed"
    assert result["new_start_s"] == pytest.approx(0.0)
    assert result["new_end_s"] == pytest.approx(15.0)

    # One ffmpeg pass (fast-cut only — already 9:16), reading the EXISTING
    # clip MP4, not the missing source VOD.
    assert len(ffmpeg_calls) == 1
    cmd = ffmpeg_calls[0]
    assert str(clip_mp4) == cmd[cmd.index("-i") + 1]
    assert cmd[cmd.index("-ss") + 1] == "0.000"
    assert cmd[cmd.index("-t") + 1] == "15.000"

    # Bounds were persisted with originals stashed for the revert column.
    assert len(trim_calls) == 1
    persisted = trim_calls[0]
    assert persisted["start_s"] == pytest.approx(0.0)
    assert persisted["end_s"] == pytest.approx(15.0)
    assert persisted["original_start_s"] == pytest.approx(0.0)
    assert persisted["original_end_s"] == pytest.approx(40.0)


def test_first_trim_saves_a_pretrim_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_dir = tmp_path / "clp_X"
    clip_dir.mkdir()
    clip_mp4 = clip_dir / "clip.mp4"
    clip_mp4.write_bytes(b"original clip bytes")

    clip = _FakeClip(
        id="clp_X", stream_id="str_X", path=clip_mp4,
        start_s=0.0, end_s=40.0, duration_s=40.0,
    )
    stream = _FakeStream(id="str_X", source_video_path=tmp_path / "gone.mp4")
    _wire_fakes(monkeypatch, clip=clip, stream=stream)
    _stub_ffmpeg(monkeypatch)

    asyncio.run(
        auto_trim_around_integrity(
            object(), clip_id="clp_X", integrity_issues=_ISSUES
        )
    )

    # A pristine pre-trim copy was stashed beside the clip so revert works
    # forever — its bytes are the ORIGINAL clip, not the re-cut output.
    pretrim = clip_dir / "clip.original.mp4"
    assert pretrim.exists()
    assert pretrim.read_bytes() == b"original clip bytes"


def test_revert_restores_from_pretrim_copy_without_vod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_dir = tmp_path / "clp_X"
    clip_dir.mkdir()
    clip_mp4 = clip_dir / "clip.mp4"
    clip_mp4.write_bytes(b"trimmed clip bytes")
    # The pristine pre-trim copy left behind by the first auto-trim.
    (clip_dir / "clip.original.mp4").write_bytes(b"original clip bytes")

    # Clip is in a trimmed state (originals stashed).
    clip = _FakeClip(
        id="clp_X", stream_id="str_X", path=clip_mp4,
        start_s=0.0, end_s=15.0, duration_s=15.0,
        original_start_s=0.0, original_end_s=40.0,
    )
    stream = _FakeStream(id="str_X", source_video_path=tmp_path / "gone.mp4")
    trim_calls = _wire_fakes(monkeypatch, clip=clip, stream=stream)
    ffmpeg_calls = _stub_ffmpeg(monkeypatch)

    result = asyncio.run(revert_trim(object(), clip_id="clp_X"))

    assert result["outcome"] == "reverted"
    assert result["new_start_s"] == pytest.approx(0.0)
    assert result["new_end_s"] == pytest.approx(40.0)
    # No ffmpeg, no source VOD — just the pristine copy swapped back in.
    assert ffmpeg_calls == []
    assert clip_mp4.read_bytes() == b"original clip bytes"
    # The copy was consumed (renamed back into place).
    assert not (clip_dir / "clip.original.mp4").exists()
    # Bounds restored + originals cleared.
    assert len(trim_calls) == 1
    assert trim_calls[0]["original_start_s"] is None
    assert trim_calls[0]["original_end_s"] is None


def test_auto_trim_raises_when_both_source_and_clip_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _FakeClip(
        id="clp_X",
        stream_id="str_X",
        path=tmp_path / "clp_X" / "clip.mp4",  # never created
        start_s=0.0,
        end_s=40.0,
        duration_s=40.0,
    )
    stream = _FakeStream(id="str_X", source_video_path=tmp_path / "gone" / "vod.mp4")
    _wire_fakes(monkeypatch, clip=clip, stream=stream)
    _stub_ffmpeg(monkeypatch)

    with pytest.raises(ClipError, match="clip MP4 missing"):
        asyncio.run(
            auto_trim_around_integrity(
                object(), clip_id="clp_X", integrity_issues=_ISSUES
            )
        )

"""Tests for the diarize service — focuses on graceful-skip semantics
and the cached-result handling. The actual pyannote call lives in
chalybclip.diarize._worker (subprocess) and is integration-tested
manually since it needs HF_TOKEN + a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chalybclip.config import DiarizationConfig
from chalybclip.diarize import Diarization, diarize, is_diarization_available
from chalybclip.diarize.models import DiarizationSegment, SpeakerEmbedding
from chalybclip.ingest import Stream


def _make_stream(tmp_path: Path) -> Stream:
    """A Stream whose audio file actually exists so we get past the
    pre-flight existence check and exercise the subprocess path."""
    audio = tmp_path / "source" / "audio.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"\x00fakeaudio")
    return Stream(
        id="str_diatest",
        tenant_id="default",
        vod_url="upload://x.mp4",
        platform="upload",
        title="t",
        duration_s=60.0,
        source_video_path=tmp_path / "source" / "video.mp4",
        source_audio_path=audio,
    )


async def test_disabled_config_skips_without_spawning(tmp_path: Path) -> None:
    """When detection.diarization.enabled=False, we never touch pyannote.
    Returns Diarization(skipped=True) with a clear reason."""
    stream = _make_stream(tmp_path)
    result = await diarize(
        tenant_id="default",
        stream=stream,
        config=DiarizationConfig(enabled=False),
    )
    assert result.skipped is True
    assert result.skip_reason == "diarization disabled in config"
    assert result.segments == []
    assert result.embeddings == []


async def test_missing_audio_skips_with_reason(tmp_path: Path) -> None:
    stream = Stream(
        id="str_noaudio",
        tenant_id="default",
        vod_url="upload://x.mp4",
        platform="upload",
        title="t",
        duration_s=60.0,
        source_video_path=tmp_path / "missing.mp4",
        source_audio_path=tmp_path / "missing.wav",
    )
    result = await diarize(
        tenant_id="default",
        stream=stream,
        config=DiarizationConfig(enabled=True),
    )
    assert result.skipped is True
    assert "audio file missing" in (result.skip_reason or "")


async def test_missing_hf_token_skips_with_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No HF_TOKEN -> immediate skip with a pointer to the license + setup."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    stream = _make_stream(tmp_path)
    result = await diarize(
        tenant_id="default",
        stream=stream,
        config=DiarizationConfig(enabled=True),
    )
    assert result.skipped is True
    assert "HF_TOKEN" in (result.skip_reason or "")


async def test_cached_result_returns_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A previous successful diarization is reused unless force=True.
    Pin HF_TOKEN so is_diarization_available's other branch (pyannote
    missing) isn't what's controlling the test."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake_for_tests")
    stream = _make_stream(tmp_path)
    cached = Diarization(
        stream_id=stream.id,
        tenant_id="default",
        segments=[
            DiarizationSegment(ts=0.0, end_ts=10.0, speaker_label="SPEAKER_00")
        ],
        embeddings=[
            SpeakerEmbedding(
                speaker_label="SPEAKER_00",
                embedding=[0.1, 0.2, 0.3],
                total_speech_s=10.0,
            )
        ],
        skipped=False,
    )
    cache_path = stream.source_audio_path.parent / "diarization.json"
    cache_path.write_text(cached.model_dump_json(), encoding="utf-8")

    result = await diarize(
        tenant_id="default",
        stream=stream,
        config=DiarizationConfig(enabled=True),
    )
    # Cached result returned (no subprocess spawned).
    assert result.skipped is False
    assert len(result.segments) == 1
    assert result.segments[0].speaker_label == "SPEAKER_00"
    assert len(result.embeddings) == 1


def test_overlap_speaker_picks_max_temporal_overlap() -> None:
    """For per-speaker trigger attribution: when a transcript segment
    straddles a turn boundary, assign the speaker with the most overlap."""
    d = Diarization(
        stream_id="str",
        tenant_id="default",
        segments=[
            DiarizationSegment(ts=0.0, end_ts=10.0, speaker_label="SPEAKER_00"),
            DiarizationSegment(ts=10.0, end_ts=30.0, speaker_label="SPEAKER_01"),
        ],
    )
    # Word @ 5-7s sits entirely in SPEAKER_00.
    assert d.overlap_speaker(5.0, 7.0) == "SPEAKER_00"
    # Word @ 9-12s straddles the boundary; SPEAKER_01 wins (2s vs 1s).
    assert d.overlap_speaker(9.0, 12.0) == "SPEAKER_01"
    # Word entirely outside any turn returns None.
    assert d.overlap_speaker(40.0, 41.0) is None


def test_speaker_label_at_returns_label_for_exact_point() -> None:
    d = Diarization(
        stream_id="str",
        tenant_id="default",
        segments=[
            DiarizationSegment(ts=0.0, end_ts=10.0, speaker_label="SPEAKER_00"),
        ],
    )
    assert d.speaker_label_at(5.0) == "SPEAKER_00"
    assert d.speaker_label_at(20.0) is None


def test_is_diarization_available_requires_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert is_diarization_available() is False


def test_is_diarization_available_pyannote_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When HF_TOKEN is set but pyannote isn't installed, still False.
    `find_spec` on Windows can raise ModuleNotFoundError for missing
    namespace packages — the service catches that defensively."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    import importlib.util

    try:
        spec = importlib.util.find_spec("pyannote.audio")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        pytest.skip("pyannote.audio is installed; skipping the absent-case test")
    assert is_diarization_available() is False


async def test_worker_timeout_degrades_to_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a wedged pyannote subprocess (CUDA stall) must degrade
    to skipped-with-reason instead of hanging the whole pipeline forever."""
    import subprocess as _subprocess

    from chalybclip.diarize import service as service_mod

    monkeypatch.setenv("HF_TOKEN", "hf_fake_for_tests")
    monkeypatch.setattr(service_mod, "is_diarization_available", lambda: True)

    def _hang_then_timeout(*args, **kwargs):
        raise _subprocess.TimeoutExpired(cmd="pyannote-worker", timeout=0.1)

    monkeypatch.setattr(service_mod.subprocess, "run", _hang_then_timeout)

    stream = _make_stream(tmp_path)
    result = await diarize(
        tenant_id="default",
        stream=stream,
        config=DiarizationConfig(enabled=True, worker_timeout_s=0.1),
    )
    assert result.skipped is True
    assert "killed after" in (result.skip_reason or "")

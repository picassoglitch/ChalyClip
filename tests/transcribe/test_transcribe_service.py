"""Tests for `transcribe()` with WhisperModel stubbed out.

Slice F.8 — `transcribe()` now routes through a `TranscribeProvider`.
For tests we still monkeypatch the in-process Whisper call, but the
hook moved into `chalybclip.transcribe.providers.local_whisper`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chalybclip.errors import TranscriptionError
from chalybclip.ingest import Stream
from chalybclip.transcribe import Transcript, transcribe

from ._fakes import FakeInfo, FakeSegment, FakeWhisperModel, FakeWord


def _make_stream(tmp_path: Path, *, tenant_id: str = "default") -> Stream:
    """Build a Stream pointing at fake source files inside `tmp_path`."""
    stream_id = "str_01TEST"
    stream_dir = tmp_path / stream_id
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    video_path = source_dir / "video.mp4"
    audio_path = source_dir / "audio.wav"
    video_path.write_bytes(b"v")
    audio_path.write_bytes(b"a")
    return Stream(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=120.0,
        source_video_path=video_path,
        source_audio_path=audio_path,
    )


@pytest.fixture(autouse=True)
def _patch_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    # Slice F.8 — patch `faster_whisper.WhisperModel` directly since
    # the local provider lazy-imports it inside `_run_inprocess`.
    # Force the provider's subprocess flag off so the fake intercepts.
    import sys
    import types

    fw = sys.modules.get("faster_whisper")
    if fw is None:
        fw = types.ModuleType("faster_whisper")
        sys.modules["faster_whisper"] = fw
    FakeWhisperModel.reset()
    monkeypatch.setattr(fw, "WhisperModel", FakeWhisperModel, raising=False)
    # Force the provider into in-process mode for the test.
    monkeypatch.setenv("CHALYBCLIP_TRANSCRIBE_INPROCESS", "1")
    # Defensive: invalidate the Settings singleton so the env-var pick-up
    # actually happens (tests share a cached Settings instance).
    from chalybclip.settings import get_settings

    get_settings.cache_clear()
    FakeWhisperModel.canned_info = FakeInfo(language="es", duration=10.5)
    FakeWhisperModel.canned_segments = [
        FakeSegment(
            start=0.0,
            end=2.5,
            text="hola mundo",
            words=[
                FakeWord(start=0.0, end=0.5, word="hola", probability=0.95),
                FakeWord(start=0.6, end=1.2, word="mundo", probability=0.91),
            ],
        ),
        FakeSegment(
            start=3.0,
            end=4.0,
            text="clipéalo",
            words=[FakeWord(start=3.0, end=4.0, word="clipéalo", probability=0.88)],
        ),
    ]


def test_transcribe_runs_and_writes_json(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    transcript = asyncio.run(
        transcribe(
            tenant_id="default",
            stream=stream,
            model_size="medium",
            device="cuda",
            compute_type="float16",
            language="es",
        )
    )

    assert isinstance(transcript, Transcript)
    assert transcript.stream_id == stream.id
    assert transcript.tenant_id == "default"
    assert transcript.language == "es"
    assert transcript.duration_s == pytest.approx(10.5)
    assert transcript.model == "medium"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].words[0].text == "hola"
    assert transcript.segments[0].words[0].prob == pytest.approx(0.95)

    json_path = stream.source_audio_path.parent / "transcript.json"
    assert json_path.exists()
    reloaded = Transcript.model_validate_json(json_path.read_text("utf-8"))
    assert reloaded == transcript

    # Verify whisper was called with the parameters we passed.
    assert len(FakeWhisperModel.constructor_calls) == 1
    ctor_args, ctor_kwargs = FakeWhisperModel.constructor_calls[0]
    assert ctor_args == ("medium",)
    assert ctor_kwargs == {"device": "cuda", "compute_type": "float16"}
    transcribe_args, transcribe_kwargs = FakeWhisperModel.transcribe_calls[0]
    assert transcribe_args == (str(stream.source_audio_path),)
    # The three tuning params below are the long-VOD survival kit
    # (VAD filter, no context accumulation, greedy decode). See
    # chalybclip.transcribe.service._run_whisper for the rationale.
    assert transcribe_kwargs == {
        "language": "es",
        "word_timestamps": True,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "beam_size": 1,
    }


def test_transcribe_is_idempotent(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    first = asyncio.run(transcribe(tenant_id="default", stream=stream))
    second = asyncio.run(transcribe(tenant_id="default", stream=stream))

    assert second == first
    # Whisper should have been instantiated only on the first call.
    assert len(FakeWhisperModel.constructor_calls) == 1
    assert len(FakeWhisperModel.transcribe_calls) == 1


def test_transcribe_force_reruns(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    asyncio.run(transcribe(tenant_id="default", stream=stream))
    asyncio.run(transcribe(tenant_id="default", stream=stream, force=True))

    assert len(FakeWhisperModel.constructor_calls) == 2
    assert len(FakeWhisperModel.transcribe_calls) == 2


def test_transcribe_rejects_tenant_mismatch(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path, tenant_id="alice")
    with pytest.raises(TranscriptionError, match="tenant mismatch"):
        asyncio.run(transcribe(tenant_id="bob", stream=stream))


def test_transcribe_rejects_missing_audio(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    stream.source_audio_path.unlink()
    with pytest.raises(TranscriptionError, match="audio file missing"):
        asyncio.run(transcribe(tenant_id="default", stream=stream))


def test_transcribe_returns_cache_when_audio_reclaimed(tmp_path: Path) -> None:
    """Cache check precedes the source-audio guard, so a re-run after the
    raw source has been reclaimed (delete-on-completion) still serves the
    cached transcript instead of raising. This is what keeps the pipeline
    idempotent once the source VOD is gone."""
    stream = _make_stream(tmp_path)
    first = asyncio.run(transcribe(tenant_id="default", stream=stream))

    # Simulate the pipeline's delete-on-completion: the audio is gone but
    # transcript.json (the durable output) remains.
    stream.source_audio_path.unlink()
    assert not stream.source_audio_path.exists()

    second = asyncio.run(transcribe(tenant_id="default", stream=stream))
    assert second == first
    # No second whisper run — served from cache.
    assert len(FakeWhisperModel.transcribe_calls) == 1


def test_transcribe_auto_language_passes_none(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    asyncio.run(transcribe(tenant_id="default", stream=stream, language=None))
    _, kwargs = FakeWhisperModel.transcribe_calls[0]
    assert kwargs["language"] is None


def test_transcribe_wraps_whisper_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = _make_stream(tmp_path)

    class BoomModel:
        def __init__(self, *_a: object, **_k: object) -> None:
            raise RuntimeError("CUDA out of memory")

    import sys

    fw = sys.modules["faster_whisper"]
    monkeypatch.setattr(fw, "WhisperModel", BoomModel, raising=False)
    with pytest.raises(TranscriptionError, match="Whisper failed to start"):
        asyncio.run(transcribe(tenant_id="default", stream=stream))


def test_transcribe_recovers_from_corrupt_cache(tmp_path: Path) -> None:
    """Regression: a truncated/corrupt transcript.json (crash mid-write,
    disk full) must fall through to a fresh transcription instead of
    permanently failing every later non-forced run with ValidationError."""
    stream = _make_stream(tmp_path)
    json_path = stream.source_audio_path.parent / "transcript.json"
    json_path.write_text('{"stream_id": "str_01TEST", "langua', encoding="utf-8")

    transcript = asyncio.run(
        transcribe(
            tenant_id="default",
            stream=stream,
            model_size="medium",
            device="cuda",
            compute_type="float16",
            language="es",
        )
    )

    assert isinstance(transcript, Transcript)
    assert len(transcript.segments) == 2
    # The corrupt cache was replaced with a valid one.
    assert Transcript.model_validate_json(json_path.read_text("utf-8"))

"""Shared fixtures for detect tests."""

from __future__ import annotations

from chalybclip.config import DetectionConfig, VoiceDetectorConfig
from chalybclip.ingest import Stream
from chalybclip.transcribe import Segment, Transcript, Word


def make_stream(stream_id: str = "str_01TEST", tenant_id: str = "default") -> Stream:
    """Build a Stream with placeholder paths (no files needed for detect tests)."""
    return Stream(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=600.0,
        source_video_path=f"out/{stream_id}/source/video.mp4",  # type: ignore[arg-type]
        source_audio_path=f"out/{stream_id}/source/audio.wav",  # type: ignore[arg-type]
    )


def make_transcript(
    *,
    stream_id: str = "str_01TEST",
    tenant_id: str = "default",
    words: list[tuple[float, float, str, float]] | None = None,
) -> Transcript:
    """Build a Transcript from `(ts, end_ts, text, prob)` word tuples.

    All words are placed in a single segment, which is enough for detect
    since the detector flattens across segments anyway.
    """
    word_models = [Word(ts=ts, end_ts=end, text=text, prob=prob) for ts, end, text, prob in (words or [])]
    if word_models:
        segment = Segment(
            ts=word_models[0].ts,
            end_ts=word_models[-1].end_ts,
            text=" ".join(w.text for w in word_models),
            words=word_models,
        )
        segments = [segment]
    else:
        segments = []
    return Transcript(
        stream_id=stream_id,
        tenant_id=tenant_id,
        language="es",
        duration_s=segments[-1].end_ts if segments else 0.0,
        model="tiny",
        segments=segments,
    )


def es_only_config(
    phrases: list[str] | None = None,
    *,
    fuzzy_distance: int = 2,
    weight: float = 1.0,
    merge_window_s: float = 30.0,
    retroactive_phrases: list[str] | None = None,
    retroactive_lookback_s: float = 60.0,
) -> DetectionConfig:
    """Voice detector config with a single Spanish phrase list.

    `retroactive_phrases` is optional — when set, the second phrase family
    (e.g. 'clipeaste eso') triggers a backward-looking clip window.
    """
    return DetectionConfig(
        voice=VoiceDetectorConfig(
            enabled=True,
            weight=weight,
            fuzzy_distance=fuzzy_distance,
            phrases={"es": phrases or ["clipéalo", "saca un clip"]},
            retroactive_phrases=(
                {"es": retroactive_phrases} if retroactive_phrases else {}
            ),
            retroactive_lookback_s=retroactive_lookback_s,
        ),
        merge_window_s=merge_window_s,
    )

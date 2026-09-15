"""Shared fixtures for clip tests."""

from __future__ import annotations

from pathlib import Path

from chalybclip.detect import Candidate
from chalybclip.ingest import Stream


def seed_stream(
    output_dir: Path, *, stream_id: str = "str_01CLIP", tenant_id: str = "default"
) -> Stream:
    """Create a stream directory + a placeholder source video on disk."""
    stream_dir = output_dir / stream_id
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    video_path = source_dir / "video.mp4"
    audio_path = source_dir / "audio.wav"
    video_path.write_bytes(b"\x00\x00fakevideo")
    audio_path.write_bytes(b"\x00\x00fakeaudio")
    return Stream(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=600.0,
        source_video_path=video_path,
        source_audio_path=audio_path,
    )


def make_candidate(
    *, timestamp: float, score: float = 0.9, phrase: str = "clipéalo"
) -> Candidate:
    return Candidate(
        timestamp=timestamp,
        score=score,
        reason="voice",
        evidence={
            "phrase": phrase,
            "language": "es",
            "transcript_snippet": phrase,
            "distance": 0,
            "confidence": score,
        },
    )

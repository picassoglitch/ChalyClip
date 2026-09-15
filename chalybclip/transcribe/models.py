"""Pydantic schemas for Whisper transcripts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    """One word with start/end times and Whisper-reported probability."""

    ts: float = Field(ge=0.0, description="Word start time, seconds.")
    end_ts: float = Field(ge=0.0, description="Word end time, seconds.")
    text: str
    prob: float = Field(ge=0.0, le=1.0)


class Segment(BaseModel):
    """A Whisper segment — a sentence-ish span containing word-level entries.

    `speaker` is an optional per-video diarization label populated by
    providers that emit speaker information natively (AssemblyAI's
    `speaker_labels=true`). Older providers + tests leave it None.
    Downstream consumers that don't care about diarization can ignore
    the field; the per-speaker persona/brand code reads it when present
    and falls back to the no-speaker path when absent. Migrating to
    AssemblyAI-based diarization without a schema break.
    """

    ts: float = Field(ge=0.0, description="Segment start time, seconds.")
    end_ts: float = Field(ge=0.0, description="Segment end time, seconds.")
    text: str
    words: list[Word] = Field(default_factory=list)
    speaker: str | None = None


class Transcript(BaseModel):
    """Full transcript for one stream.

    `speakers` is the deduped list of per-video speaker labels the
    provider emitted (e.g. ["A", "B"] for AssemblyAI, ["SPEAKER_00",
    "SPEAKER_01"] for pyannote-via-modal). Empty list when diarization
    wasn't requested or the provider doesn't support it.
    """

    stream_id: str
    tenant_id: str
    language: str
    duration_s: float = Field(ge=0.0)
    model: str = Field(description="Whisper model size, e.g. `medium`.")
    segments: list[Segment] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)

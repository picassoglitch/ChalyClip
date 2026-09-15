"""Pydantic schemas for the diarize module."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DiarizationSegment(BaseModel):
    """One contiguous turn by a single speaker within a VOD.

    `speaker_label` is the within-VOD label that pyannote assigns
    (SPEAKER_00, SPEAKER_01, ...). It's stable for a single
    diarization run but NOT stable across VODs — that's what the
    embedding match against the tenant's `speakers` table is for.
    """

    model_config = ConfigDict(extra="forbid")

    ts: float = Field(ge=0.0, description="Turn start in seconds.")
    end_ts: float = Field(ge=0.0, description="Turn end in seconds.")
    speaker_label: str = Field(min_length=1, description="e.g. 'SPEAKER_00'")


class SpeakerEmbedding(BaseModel):
    """One speaker's voice fingerprint, aggregated over a VOD.

    The vector is the pyannote embedding model's output averaged over
    all turns attributed to this label, weighted by turn duration. ~192
    floats for pyannote-3.1. Stored as a list[float] in JSON; cast to
    numpy at compare time.
    """

    model_config = ConfigDict(extra="forbid")

    speaker_label: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
    total_speech_s: float = Field(
        ge=0.0,
        description="Total time this speaker talked in this VOD. The "
        "embedding match step requires at least N seconds of speech for "
        "a confident ID (see DiarizationConfig.min_speech_for_id_s).",
    )


class Diarization(BaseModel):
    """All speaker info extracted from one VOD's audio."""

    model_config = ConfigDict(extra="forbid")

    stream_id: str
    tenant_id: str
    segments: list[DiarizationSegment] = Field(default_factory=list)
    embeddings: list[SpeakerEmbedding] = Field(default_factory=list)
    skipped: bool = Field(
        default=False,
        description="True when diarization was disabled, pyannote was "
        "missing, or the worker crashed. Downstream consumers must "
        "tolerate the empty case.",
    )
    skip_reason: str | None = Field(
        default=None,
        description="One-line explanation for `skipped=True`. Surfaces in "
        "the dashboard so the user knows why no speaker labels appeared.",
    )

    def speaker_label_at(self, timestamp: float) -> str | None:
        """Return the speaker_label whose turn contains `timestamp`.

        Returns None when no turn covers that moment (typical at the very
        start/end of a VOD or in silence the diarizer didn't attribute).
        """
        for seg in self.segments:
            if seg.ts <= timestamp <= seg.end_ts:
                return seg.speaker_label
        return None

    def overlap_speaker(self, ts: float, end_ts: float) -> str | None:
        """Return the speaker_label with the most temporal overlap on
        [ts, end_ts]. Used to attribute a transcript segment to a
        speaker even when the segment straddles a turn boundary.
        """
        best_label: str | None = None
        best_overlap = 0.0
        for seg in self.segments:
            overlap = max(0.0, min(end_ts, seg.end_ts) - max(ts, seg.ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = seg.speaker_label
        return best_label

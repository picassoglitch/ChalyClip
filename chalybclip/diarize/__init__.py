"""Speaker diarization — pyannote-audio 3.1 in a subprocess.

Splits a VOD's audio into per-speaker turns. Each turn carries a
within-VOD label (SPEAKER_00, SPEAKER_01, ...) and a 192-dim embedding
vector. The pipeline then matches embeddings against the tenant's
persistent `speakers` table to resolve who's actually talking.

Subprocess-isolated like Whisper: a hard pyannote crash (CUDA OOM,
torch driver mismatch) kills only the worker, not the dashboard.

Skippable: if HF_TOKEN is missing, pyannote isn't installed, or the
worker exits non-zero, `diarize()` returns an empty Diarization and the
pipeline carries on without speaker labels. Downstream code reads
`candidate.evidence.get('speaker_label')` defensively.
"""

from __future__ import annotations

from .from_transcript import diarization_from_transcript
from .identity import ResolutionOutcome, resolve_speakers
from .models import Diarization, DiarizationSegment, SpeakerEmbedding
from .service import diarize, is_diarization_available

__all__ = [
    "Diarization",
    "DiarizationSegment",
    "ResolutionOutcome",
    "SpeakerEmbedding",
    "diarization_from_transcript",
    "diarize",
    "is_diarization_available",
    "resolve_speakers",
]

"""Whisper transcription module — public surface."""

from .models import Segment, Transcript, Word
from .providers import (
    CloudWhisperProvider,
    LocalWhisperProvider,
    TranscribeProvider,
    TranscribeRequest,
    get_provider,
)
from .service import load_transcript, transcribe

__all__ = [
    "CloudWhisperProvider",
    "LocalWhisperProvider",
    "Segment",
    "Transcript",
    "TranscribeProvider",
    "TranscribeRequest",
    "Word",
    "get_provider",
    "load_transcript",
    "transcribe",
]

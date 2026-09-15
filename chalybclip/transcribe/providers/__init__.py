"""Transcription providers — slice F.8.

The pipeline calls one `TranscribeProvider` to turn audio into a
`Transcript`. Today the only working impl is `LocalWhisperProvider`
(faster-whisper on the local GPU). The `CloudWhisperProvider` stub
documents the shape an AssemblyAI / Deepgram / OpenAI-Whisper-API
implementation would take when we move pipeline work off the local
GPU — see docs/deployment.md for the Stage 1→2 migration story.

Selection is driven by `Settings.transcribe_provider`:

  - "local"     → LocalWhisperProvider (default; preserves existing behavior)
  - "assemblyai" / "deepgram" → CloudWhisperProvider stub (raises until wired)

Callers use the `get_provider()` factory; tests inject `FakeTranscribeProvider`
from `tests/_fakes/fake_transcribe.py`.
"""

from __future__ import annotations

from .assemblyai import AssemblyAIProvider
from .base import TranscribeProvider, TranscribeRequest
from .cloud_whisper import CloudWhisperProvider
from .factory import get_provider
from .local_whisper import LocalWhisperProvider

__all__ = [
    "AssemblyAIProvider",
    "CloudWhisperProvider",
    "LocalWhisperProvider",
    "TranscribeProvider",
    "TranscribeRequest",
    "get_provider",
]

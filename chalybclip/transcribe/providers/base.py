"""TranscribeProvider Protocol + the request payload it accepts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from chalybclip.transcribe.models import Transcript


@dataclass(frozen=True)
class TranscribeRequest:
    """Inputs a provider needs to turn audio into a Transcript.

    Keep this provider-agnostic — `model_size` / `device` / `compute_type`
    only mean something to the local Whisper provider, but cloud providers
    will need *different* knobs (AssemblyAI's `speech_model`, Deepgram's
    `tier`+`language_code`, etc.). Each impl reads the keys it cares about
    from its own constructor config, not from this request struct.
    """

    audio_path: Path
    stream_id: str
    tenant_id: str
    language: str | None = None  # ISO 639-1 or `None` for auto-detect


@runtime_checkable
class TranscribeProvider(Protocol):
    """A pluggable transcriber.

    Implementations must be safe to call from inside an `asyncio.to_thread`
    or as a true async coroutine — the pipeline awaits the call directly.
    Long-running blocking work (faster-whisper, network calls to AssemblyAI)
    should be wrapped inside the implementation.
    """

    async def transcribe(self, req: TranscribeRequest) -> Transcript:
        """Return a Transcript or raise `TranscriptionError`."""
        ...

    @property
    def name(self) -> str:
        """Short identifier for logs + telemetry, e.g. "local-whisper-medium"."""
        ...

"""CloudWhisperProvider — stub for AssemblyAI / Deepgram / OpenAI Whisper API.

NOT YET IMPLEMENTED. The class exists so the factory pattern is symmetric
and the integration shape (httpx upload → poll → map JSON to Transcript) is
documented for whoever wires it.

When we move pipeline workers off the local GPU (slice F.10+), the swap
is:

    settings.transcribe_provider = "assemblyai"
    settings.assemblyai_api_key = "..."

No change to pipeline.py — `get_provider()` returns this impl instead.

Implementation notes for the wiring slice:
  - AssemblyAI: POST /v2/upload (audio bytes) → POST /v2/transcript with
    `speech_model=universal`, `language_code=es`, `speaker_labels=true`,
    `word_boost=[...trigger phrases...]`. Poll /v2/transcript/{id} until
    `status=completed`. Map `words[].start/end/text/confidence` → our
    `Word`, group by `words[].speaker` to get diarization for free.
  - Deepgram: POST /v1/listen with multipart, `model=nova-2`, `diarize=true`,
    `punctuate=true`, `language=es`. Response is sync — no polling. Map
    `results.channels[0].alternatives[0].words` to our `Word`.
  - OpenAI Whisper API: POST /v1/audio/transcriptions with
    `response_format=verbose_json`, `timestamp_granularities=["word"]`.
    Cheapest, but no diarization — would still need pyannote separately.

The right pick for chalyb is AssemblyAI: cheapest combined STT +
diarization, decent Spanish quality, single API call.
"""

from __future__ import annotations

from chalybclip.errors import TranscriptionError
from chalybclip.transcribe.models import Transcript

from .base import TranscribeRequest


class CloudWhisperProvider:
    """Stub. Set `settings.transcribe_provider = "local"` until wired."""

    def __init__(self, *, vendor: str, api_key: str, language: str | None = None) -> None:
        self._vendor = vendor
        self._api_key = api_key
        self._default_language = language

    @property
    def name(self) -> str:
        return f"cloud-whisper-{self._vendor}"

    async def transcribe(self, req: TranscribeRequest) -> Transcript:
        raise TranscriptionError(
            f"CloudWhisperProvider({self._vendor!r}) is a stub. "
            f"Wire the {self._vendor} integration in slice F.10+ before "
            f"setting CHALYBCLIP_TRANSCRIBE_PROVIDER to a cloud vendor."
        )

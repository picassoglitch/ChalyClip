"""Factory — picks the TranscribeProvider based on Settings."""

from __future__ import annotations

import os

from chalybclip.errors import ChalybClipError
from chalybclip.settings import get_settings

from .base import TranscribeProvider
from .cloud_whisper import CloudWhisperProvider
from .local_whisper import LocalWhisperProvider


def get_provider() -> TranscribeProvider:
    """Return the configured transcribe provider.

    Selection lives entirely in `Settings.transcribe_provider`:

      "local"      → LocalWhisperProvider (faster-whisper on this host)
      "assemblyai" → CloudWhisperProvider(vendor="assemblyai")  [stub]
      "deepgram"   → CloudWhisperProvider(vendor="deepgram")    [stub]
      "openai"     → CloudWhisperProvider(vendor="openai")      [stub]

    Defaults to "local" so existing deployments don't change behavior.

    The legacy `CHALYBCLIP_TRANSCRIBE_INPROCESS=1` env override is still
    honored — it flips the local provider's subprocess isolation off.
    """
    settings = get_settings()
    choice = (settings.transcribe_provider or "local").strip().lower()

    if choice == "local":
        # Honor the legacy subprocess-isolation toggle. Off (subprocess
        # mode = on) is the default — see local_whisper.py for the
        # crash-isolation rationale.
        use_subprocess = (
            os.environ.get("CHALYBCLIP_TRANSCRIBE_INPROCESS", "").strip() != "1"
        )
        return LocalWhisperProvider(
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            use_subprocess=use_subprocess,
        )

    if choice == "modal":
        # Slice O.44 — Modal GPU Whisper. See modal_whisper.py for
        # the deploy + env-var contract.
        from .modal_whisper import ModalWhisperProvider

        public_url = (settings.public_url or "").strip()
        audio_via_object_storage = bool(
            getattr(settings, "transcribe_audio_via_object_storage", False)
        )
        if not settings.modal_endpoint_url or not settings.modal_token:
            raise ChalybClipError(
                "transcribe_provider='modal' requires CHALYBCLIP_MODAL_ENDPOINT_URL "
                "and CHALYBCLIP_MODAL_TOKEN. Run `modal deploy "
                "infra/modal_whisper_app.py` and paste the printed URL."
            )
        # Phase 2b — worker mode uploads the WAV to the object store and
        # hands Modal a presigned URL, so the signed-local-URL plumbing
        # (signing secret + public host) isn't required there.
        if not audio_via_object_storage:
            if not settings.internal_signing_secret:
                raise ChalybClipError(
                    "transcribe_provider='modal' requires "
                    "CHALYBCLIP_INTERNAL_SIGNING_SECRET (random string used to sign "
                    "the audio URL Modal pulls from)."
                )
            if not public_url:
                raise ChalybClipError(
                    "transcribe_provider='modal' requires CHALYBCLIP_PUBLIC_URL "
                    "(externally-reachable base URL Modal hits)."
                )
        return ModalWhisperProvider(
            endpoint_url=settings.modal_endpoint_url,
            bearer_token=settings.modal_token,
            signing_secret=settings.internal_signing_secret or "",
            public_base_url=public_url,
            model=settings.modal_model,
            audio_via_object_storage=audio_via_object_storage,
        )

    if choice == "assemblyai":
        # Task A1 — real AssemblyAI provider. The CloudWhisperProvider
        # stub stays available behind the legacy "deepgram"/"openai"
        # spellings until those vendors get their own concrete impls.
        from .assemblyai import AssemblyAIProvider

        if not settings.assemblyai_api_key:
            raise ChalybClipError(
                "transcribe_provider='assemblyai' requires "
                "CHALYBCLIP_ASSEMBLYAI_API_KEY in .env"
            )
        return AssemblyAIProvider(
            api_key=settings.assemblyai_api_key,
            language_mode=settings.assemblyai_language_mode,
            speaker_labels=settings.assemblyai_speaker_labels,
            speech_models=settings.assemblyai_speech_models,
        )

    if choice in {"deepgram", "openai"}:
        api_key = getattr(settings, f"{choice}_api_key", None)
        if not api_key:
            raise ChalybClipError(
                f"transcribe_provider={choice!r} requires "
                f"CHALYBCLIP_{choice.upper()}_API_KEY in .env"
            )
        return CloudWhisperProvider(vendor=choice, api_key=api_key)

    raise ChalybClipError(
        f"unknown transcribe_provider {choice!r}; expected one of "
        f"local / modal / assemblyai / deepgram / openai"
    )

"""ModalWhisperProvider — GPU Whisper via Modal (slice O.44).

Deploy plan:
  1. `modal deploy infra/modal_whisper_app.py` — once on the operator's
     box. Modal prints the endpoint URL.
  2. Set in Railway env:
        CHALYBCLIP_TRANSCRIBE_PROVIDER=modal
        CHALYBCLIP_MODAL_ENDPOINT_URL=<the URL Modal printed>
        CHALYBCLIP_MODAL_TOKEN=<long random string, same on both sides>
        CHALYBCLIP_INTERNAL_SIGNING_SECRET=<long random string>
        CHALYBCLIP_MODAL_MODEL=small  (optional; default small)
        CHALYBCLIP_PUBLIC_URL=https://chalybclip.chalyb.com  (already
            set; Modal pulls audio from this base + signed query)
  3. On Modal side, create the secret the app expects:
        modal secret create chalybclip-modal-token MODAL_BEARER_TOKEN=<same>

Once both sides agree on the bearer token, ChalybClip POSTs the audio
URL + signed token to Modal, Modal pulls + transcribes + returns
JSON, and the pipeline keeps going.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

import httpx
import structlog

from chalybclip.errors import TranscriptionError
from chalybclip.integrations.modal_http import (
    ModalPollDeadlineError,
    poll_until_terminal,
)
from chalybclip.transcribe.models import Segment, Transcript, Word

from .base import TranscribeRequest

_log = structlog.get_logger(__name__)

# Modal's web-endpoint protocol for long-running functions: the initial
# POST returns 303 See Other with Location: ?__modal_function_call_id=fc-...,
# and that polling URL keeps redirecting (302 / 303) until the function
# completes (200 + body) or fails (4xx / 5xx). The follow-the-redirects
# loop lives in `chalybclip.integrations.modal_http` (shared with the
# pipeline job dispatcher since Phase 2b); this module keeps its own
# cadence constant.
_POLL_INTERVAL_S = 5.0
"""How long to wait between poll requests. Modal's Whisper-small on T4
takes ~30-90s for a typical VOD, so a 5s cadence is ~6-18 polls per run."""


class ModalWhisperProvider:
    """Forwards transcribe to a Modal-hosted faster-whisper container.

    Behavior contract is identical to `LocalWhisperProvider`: returns
    a `Transcript` or raises `TranscriptionError`. The pipeline can
    swap between local and modal via CHALYBCLIP_TRANSCRIBE_PROVIDER
    without changing anything else.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        bearer_token: str,
        signing_secret: str,
        public_base_url: str,
        model: str = "small",
        request_timeout_s: float = 3600.0,
        audio_via_object_storage: bool = False,
    ) -> None:
        self._endpoint_url = endpoint_url.rstrip("/")
        self._bearer_token = bearer_token
        self._signing_secret = signing_secret
        self._public_base_url = public_base_url.rstrip("/")
        self._model = model
        self._timeout_s = request_timeout_s
        # Phase 2b — on the pipeline WORKER the WAV lives on ephemeral
        # disk and never existed on the web box, so a signed URL against
        # CHALYBCLIP_PUBLIC_URL would 410. Upload to the object store and
        # hand Modal a presigned URL instead.
        self._audio_via_object_storage = audio_via_object_storage

    @property
    def name(self) -> str:
        return f"modal-whisper-{self._model}"

    async def transcribe(self, req: TranscribeRequest) -> Transcript:
        if not self._endpoint_url or not self._bearer_token:
            raise TranscriptionError(
                "ModalWhisperProvider misconfigured: CHALYBCLIP_MODAL_ENDPOINT_URL "
                "and CHALYBCLIP_MODAL_TOKEN must be set."
            )
        work_audio_key: str | None = None
        if self._audio_via_object_storage:
            audio_url, work_audio_key = await self._upload_audio_for_pull(req)
        else:
            if not self._signing_secret:
                raise TranscriptionError(
                    "ModalWhisperProvider misconfigured: CHALYBCLIP_INTERNAL_SIGNING_SECRET "
                    "must be set so Modal can pull audio from this server."
                )
            if not self._public_base_url:
                raise TranscriptionError(
                    "ModalWhisperProvider misconfigured: CHALYBCLIP_PUBLIC_URL must "
                    "be set to the externally-reachable base URL (e.g. "
                    "https://chalybclip.chalyb.com) so Modal can "
                    "fetch the audio."
                )
            audio_url = self._build_signed_audio_url(req.stream_id, req.tenant_id)
        body: dict[str, Any] = {
            "audio_url": audio_url,
            "language": req.language,
            "model": self._model,
            "stream_id": req.stream_id,
            # Bearer token also in the payload — Modal's web_endpoint
            # quirks make header auth awkward; keeping it in the body
            # is simpler + still over HTTPS so it's encrypted in flight.
            "auth_token": self._bearer_token,
        }
        _log.info(
            "modal_whisper.dispatch",
            stream_id=req.stream_id,
            tenant_id=req.tenant_id,
            model=self._model,
            audio_url_host=audio_url.split("/")[2] if "//" in audio_url else "?",
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(self._endpoint_url, json=body)
                # Modal's async-function protocol — POST returns 303 with
                # a polling Location. Follow until the function completes
                # (200 + body) or the deadline expires. `_POLL_INTERVAL_S`
                # is read at call time so tests can shrink it.
                resp = await poll_until_terminal(
                    client=client,
                    initial=resp,
                    deadline_s=self._timeout_s,
                    poll_interval_s=_POLL_INTERVAL_S,
                    label="modal whisper",
                    ref=req.stream_id,
                )
                resp.raise_for_status()
                raw = resp.json()
        except ModalPollDeadlineError as e:
            # A whisper result that never lands blocks the pipeline —
            # deadline here IS a transcription failure.
            raise TranscriptionError(str(e)) from e
        except httpx.HTTPStatusError as e:
            raise _classify_modal_error(e) from e
        except httpx.HTTPError as e:
            raise TranscriptionError(
                f"modal whisper transport error: {e}"
            ) from e

        if work_audio_key is not None:
            await self._cleanup_work_audio(work_audio_key)
        return _modal_response_to_transcript(raw, req=req)

    def _build_signed_audio_url(self, stream_id: str, tenant_id: str) -> str:
        """HMAC-signed URL valid for 30 min so Modal can pull the audio.

        Shape: /api/internal/audio/<stream_id>?tenant=<tid>&exp=<unix>&sig=<hmac>
        Server side verifies sig + exp before serving the file.
        """
        exp = int(time.time()) + 30 * 60  # 30 min window
        msg = f"{stream_id}|{tenant_id}|{exp}".encode()
        sig = hmac.new(
            self._signing_secret.encode(), msg, hashlib.sha256
        ).hexdigest()
        return (
            f"{self._public_base_url}/api/internal/audio/{stream_id}"
            f"?tenant={tenant_id}&exp={exp}&sig={sig}"
        )

    async def _upload_audio_for_pull(
        self, req: TranscribeRequest
    ) -> tuple[str, str]:
        """Worker mode: put the local WAV in the object store and return
        `(presigned_url, key)` for Modal to pull. The key is transient —
        deleted best-effort once the transcript lands."""
        from pathlib import Path

        from chalybclip.integrations.storage import build_artifact_store
        from chalybclip.settings import get_settings

        store = build_artifact_store(get_settings())
        if store is None:
            raise TranscriptionError(
                "ModalWhisperProvider misconfigured: "
                "CHALYBCLIP_TRANSCRIBE_AUDIO_VIA_OBJECT_STORAGE=1 requires the "
                "CHALYBCLIP_OBJECT_STORAGE_* vars (there is no public host to "
                "serve the audio from on a worker)."
            )
        audio_path = Path(req.audio_path)
        if not audio_path.is_file():
            raise TranscriptionError(
                f"audio file missing on worker disk: {audio_path}"
            )
        key = f"work/{req.tenant_id}/{req.stream_id}/audio.wav"
        await store.upload(
            local_path=audio_path, key=key, content_type="audio/wav"
        )
        # 24h TTL: far beyond any transcription runtime, well under the
        # 7-day presign ceiling.
        url = await store.presigned_url(key=key, ttl_seconds=24 * 3600)
        _log.info(
            "modal_whisper.audio_via_object_storage",
            stream_id=req.stream_id, key=key,
        )
        return url, key

    async def _cleanup_work_audio(self, key: str) -> None:
        """Drop the transient work-audio object. Best-effort — a leaked
        object costs cents and the runbook documents a bucket lifecycle
        rule for `work/` as the backstop."""
        try:
            from chalybclip.integrations.storage import build_artifact_store
            from chalybclip.settings import get_settings

            store = build_artifact_store(get_settings())
            if store is not None:
                await store.delete(key=key)
        except Exception as e:  # cleanup must never fail the transcript
            _log.warning(
                "modal_whisper.work_audio_cleanup_failed",
                key=key, error=str(e),
            )


def _modal_response_to_transcript(
    raw: dict[str, Any], *, req: TranscribeRequest
) -> Transcript:
    """Map the Modal app's JSON response into our Transcript model.

    The Modal app's `transcribe()` shape is documented in
    infra/modal_whisper_app.py — see the docstring there.
    """
    try:
        segments = []
        for s in raw.get("segments", []):
            words = [
                Word(
                    text=w.get("word", ""),
                    ts=float(w.get("start", 0.0)),
                    end_ts=float(w.get("end", 0.0)),
                    prob=max(0.0, min(1.0, float(w.get("prob", 0.0)))),
                )
                for w in s.get("words", [])
            ]
            segments.append(
                Segment(
                    text=s.get("text", "").strip(),
                    ts=float(s.get("start", 0.0)),
                    end_ts=float(s.get("end", 0.0)),
                    words=words,
                )
            )
        return Transcript(
            stream_id=req.stream_id,
            tenant_id=req.tenant_id,
            language=str(raw.get("language") or req.language or "es"),
            duration_s=float(raw.get("duration_s", 0.0)),
            model=f"modal-{raw.get('model', '?')}",
            segments=segments,
        )
    except Exception as e:  # noqa: BLE001
        raise TranscriptionError(
            f"modal whisper response unparseable: {e}; raw keys: "
            f"{sorted(raw.keys())[:8]}"
        ) from e


# ---- error classification ----
#
# Modal's web endpoint surfaces several distinct failure modes through
# the same HTTP layer; the raw status code alone isn't actionable on
# the dashboard. The classifier below maps a couple of well-known cases
# (the operator-hit ones) to specific TranscriptionError messages that
# tell the operator what to do next. Other errors fall through to the
# generic "modal whisper returned NNN: …" shape.


def _classify_modal_error(exc: httpx.HTTPStatusError) -> TranscriptionError:
    """Map an HTTPStatusError from the Modal endpoint to a
    TranscriptionError whose message is actionable.

    Recognized cases:

      * 429 with "billing"/"spend"/"cycle" in the body — Modal's
        workspace billing cap. Telling the operator to flip
        `CHALYBCLIP_TRANSCRIBE_PROVIDER=assemblyai` is the right next
        step; that's the post-migration default anyway. Mentions the
        Modal billing page so they can also bump the cap if they want
        to stay on Modal.
      * 401 / 403 — bad bearer or expired token. Surface the env-var
        name they need to fix.
      * 5xx — Modal worker crashed. Generic "service down" hint.
      * Everything else — passthrough with status + body.
    """
    status = exc.response.status_code
    body = (exc.response.text or "")[:500]
    body_lower = body.lower()

    if status == 429 and any(
        kw in body_lower for kw in ("billing", "spend", "cycle", "quota")
    ):
        return TranscriptionError(
            "Modal whisper hit the workspace billing/spend cap "
            f"(HTTP 429). Two fixes, in order of least friction:\n"
            f"  1. Switch this deployment to AssemblyAI for transcription — "
            f"set CHALYBCLIP_TRANSCRIBE_PROVIDER=assemblyai and "
            f"CHALYBCLIP_ASSEMBLYAI_API_KEY=<key> on Railway, then redeploy. "
            f"AssemblyAI ~$0.17/hr does transcription + diarization in one "
            f"call with no GPU dep (Migration Tasks A1-A3 added this path).\n"
            f"  2. Raise the Modal workspace spend cap at "
            f"https://modal.com/settings/billing and re-run the pipeline.\n"
            f"Raw response: {body}"
        )

    if status in (401, 403):
        return TranscriptionError(
            f"Modal whisper rejected the bearer token (HTTP {status}). "
            f"Check CHALYBCLIP_MODAL_TOKEN on Railway matches the "
            f"MODAL_BEARER_TOKEN secret in the Modal workspace. Or "
            f"switch to AssemblyAI: set CHALYBCLIP_TRANSCRIBE_PROVIDER="
            f"assemblyai + CHALYBCLIP_ASSEMBLYAI_API_KEY=<key>. "
            f"Raw response: {body}"
        )

    if 500 <= status < 600:
        return TranscriptionError(
            f"Modal whisper service error (HTTP {status}) — Modal's "
            f"worker crashed or the function timed out. Retry the run; "
            f"if it persists, switch to AssemblyAI "
            f"(CHALYBCLIP_TRANSCRIBE_PROVIDER=assemblyai). "
            f"Raw response: {body}"
        )

    # Generic fallthrough — same shape we shipped before.
    return TranscriptionError(
        f"modal whisper returned {status}: {body}"
    )

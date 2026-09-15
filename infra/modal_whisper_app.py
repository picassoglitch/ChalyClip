"""Modal app: GPU-backed Whisper for ChalybClip.

Run once on the operator's machine to deploy:

    pip install modal
    modal token new           # one-time Modal auth
    modal deploy infra/modal_whisper_app.py

Modal prints the public endpoint URL after deploy — copy it into
Railway as CHALYBCLIP_MODAL_ENDPOINT_URL. Also set
CHALYBCLIP_MODAL_TOKEN to a long random string AND deploy this app
with the same value in MODAL_BEARER_TOKEN (set via Modal Secrets:
`modal secret create chalybclip-modal-token MODAL_BEARER_TOKEN=...`).

Cost model (Modal T4 GPU, ~$0.59/hour as of 2026):
  - faster-whisper `small` on T4 transcribes ~10x realtime
  - 1-hour VOD: ~6 minutes of GPU = ~$0.06
  - 3-hour streamer VOD: ~18 minutes of GPU = ~$0.18
Cold start adds ~30s — the container_idle_timeout keeps the box
warm for 5 minutes between requests so back-to-back uploads from
the same user pay it once.

Why this design:
  - ChalybClip serves the audio file at a short-lived HMAC-signed URL
    on its own Railway domain. Modal pulls from there over HTTPS.
  - Avoids the 32 MB Modal web-endpoint multipart limit that bites
    long VODs.
  - Removes the need for S3 / R2 / any third storage hop.
  - The HMAC URL is bound to (stream_id, expiry) so even if it leaks
    it only exposes one audio extract for at most 30 min.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import modal

# Image: Python + faster-whisper + curl for the audio fetch. Pinning
# faster-whisper here keeps the model behavior in sync with the local
# provider — bumping it on one side and forgetting the other would
# produce different transcripts between dev + prod.
_IMAGE = (
    # Slice O.44.4 — debian_slim doesn't ship libcublas / libcudnn, so
    # ctranslate2 (faster-whisper's runtime) crashes at first inference
    # with "Library libcublas.so.12 is not found". Use NVIDIA's CUDA
    # runtime image so the GPU libs are in /usr/local/cuda/lib64 where
    # ctranslate2 looks. cuDNN 9 + CUDA 12.4 matches ctranslate2 4.x.
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper==1.0.3",
        "httpx>=0.27",
        # Modal 1.x stopped auto-installing FastAPI for fastapi_endpoint
        # functions — we now have to pin it ourselves. `[standard]`
        # pulls in the useful middleware extras Modal recommends.
        "fastapi[standard]>=0.115",
        # faster-whisper 1.0.3's utils.py imports `requests` at the top
        # of the file but doesn't declare it as a dependency. Pin it
        # explicitly so the image build doesn't skip it.
        "requests>=2.32",
    )
    # Bake the `small` weights into the image at build time (CPU-billed,
    # once). Without this every cold container re-downloads ~460 MB from
    # HuggingFace on GPU-billed time before the first segment comes out.
    .run_commands(
        "python -c \"from faster_whisper import WhisperModel; "
        "WhisperModel('small', device='cpu', compute_type='int8')\""
    )
)

# Only sizes we deliberately pay for. A config typo on the caller side
# (CHALYBCLIP_MODAL_MODEL=large-v3) would otherwise silently multiply the
# per-VOD GPU cost — this is exactly how a credit balance dies quietly.
_ALLOWED_MODELS = {"tiny", "base", "small"}

app = modal.App("chalybclip-whisper")


@app.function(
    image=_IMAGE,
    gpu="T4",  # cheapest GPU that runs `small` at >10x realtime
    timeout=3600,  # 1h cap — multi-hour VODs need it
    scaledown_window=120,  # keep warm 2 min for back-to-back uploads
                            # (renamed from container_idle_timeout in Modal 1.x)
    # Credit-drain guard: at most 2 T4s alive no matter how many jobs
    # queue up (a reprocess sweep or channel-ingest burst fans out here).
    # No `retries=` — Modal web functions don't support retries at all,
    # so a crashed run never auto-retries on our dime; the pipeline's
    # recovery loop caps re-dispatch at 3 attempts.
    max_containers=2,
    secrets=[modal.Secret.from_name("chalybclip-modal-token")],
)
@modal.fastapi_endpoint(method="POST")
def transcribe(payload: dict) -> dict:
    """Transcribe the audio at `payload["audio_url"]`.

    Expected JSON body:
        {
            "audio_url": "https://chalybclip.../api/internal/audio/<stream_id>?sig=...",
            "language": "es",  # ISO 639-1 or null for auto-detect
            "model": "small",
            "stream_id": "...",  # echoed back for log correlation
        }

    Returns:
        {
            "language": "es",
            "duration_s": 7384.2,
            "segments": [
                {"start": 0.0, "end": 4.2, "text": "...",
                 "words": [{"start": 0.0, "end": 0.3, "word": "...", "prob": 0.92}, ...]},
                ...
            ],
            "model": "small",
            "elapsed_s": 412.3,
        }
    """
    import httpx
    from fastapi import HTTPException, Header
    from faster_whisper import WhisperModel

    # Bearer auth — Modal's web endpoint is otherwise public.
    # The function signature can include Header() params via FastAPI's
    # dependency system since Modal uses FastAPI under the hood.
    # We re-fetch the header from the request via Modal's web hook.
    # Simpler: read from os.environ which Modal Secret injects.
    expected_token = os.environ.get("MODAL_BEARER_TOKEN", "")
    # Modal exposes the incoming FastAPI request in modal.experimental
    # but a cleaner path is to use a fastapi.Header dependency. For
    # brevity we read the Modal-injected `MODAL_REQUEST_AUTH` header
    # via the request context. If this becomes fragile, switch to
    # the @asgi_app pattern with explicit FastAPI app.
    request_token = str((payload or {}).get("auth_token", ""))
    # Constant-time compare (bytes — str compare_digest rejects non-ASCII):
    # this public URL's only guard is the token, and `!=` leaks a timing
    # oracle byte by byte.
    import hmac as _hmac

    if not expected_token or not _hmac.compare_digest(
        request_token.encode("utf-8"), expected_token.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid bearer token")

    audio_url = (payload or {}).get("audio_url", "").strip()
    language = (payload or {}).get("language") or None
    model_size = (payload or {}).get("model") or "small"
    stream_id = (payload or {}).get("stream_id", "?")
    if not audio_url:
        raise HTTPException(status_code=400, detail="audio_url required")
    if model_size not in _ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model {model_size!r} not allowed; use one of "
            f"{sorted(_ALLOWED_MODELS)} (cost guard)",
        )

    started_at = time.time()

    # Download the audio from ChalybClip's signed URL.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with httpx.Client(timeout=600.0) as client:
            with client.stream("GET", audio_url) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)

        # Transcribe.
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        segments, info = model.transcribe(
            str(tmp_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        out_segments = []
        for seg in segments:
            out_segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word,
                        "prob": float(w.probability or 0.0),
                    }
                    for w in (seg.words or [])
                ],
            })
        return {
            "stream_id": stream_id,
            "language": info.language,
            "duration_s": float(info.duration),
            "segments": out_segments,
            "model": model_size,
            "elapsed_s": time.time() - started_at,
        }
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

"""ModalWhisperProvider — worker-mode audio via the object store (Phase 2b).

On the pipeline worker the WAV lives on ephemeral disk and never existed
on the web box, so the provider can't mint a CHALYBCLIP_PUBLIC_URL-signed
audio URL (it would 410). With `audio_via_object_storage=True` it uploads
the WAV to R2, hands Modal a presigned URL, and drops the transient object
once the transcript lands.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import chalybclip.integrations.storage as storage_mod
from chalybclip.errors import TranscriptionError
from chalybclip.transcribe.providers import modal_whisper
from chalybclip.transcribe.providers.base import TranscribeRequest

_REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(_s: float) -> None:
        return None
    monkeypatch.setattr(asyncio, "sleep", _noop)


class _FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    async def upload(
        self, *, local_path: Path, key: str, content_type: str | None = None
    ) -> None:
        self.objects[key] = Path(local_path).read_bytes()

    async def presigned_url(self, *, key: str, ttl_seconds: int) -> str:
        return f"https://bucket.example/{key}?exp={ttl_seconds}"

    async def delete(self, *, key: str) -> None:
        self.deleted.append(key)
        self.objects.pop(key, None)


_SUCCESS_BODY = {
    "language": "es",
    "duration_s": 12.0,
    "model": "small",
    "segments": [],
}


def _worker_provider() -> modal_whisper.ModalWhisperProvider:
    """Provider as the WORKER builds it: no signing secret, no public
    host — those are web-box-only plumbing."""
    return modal_whisper.ModalWhisperProvider(
        endpoint_url="https://modal.test",
        bearer_token="bear",
        signing_secret="",
        public_base_url="",
        model="small",
        request_timeout_s=2.0,
        audio_via_object_storage=True,
    )


def _req(tmp_path: Path) -> TranscribeRequest:
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF-fake-wav")
    return TranscribeRequest(
        audio_path=str(wav),
        stream_id="str_xyz",
        tenant_id="ten_xyz",
        language="es",
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    monkeypatch.setattr(
        modal_whisper.httpx, "AsyncClient",
        lambda *a, **kw: _REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(handler)
        ),
    )


async def test_uploads_audio_and_passes_presigned_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FakeStore()
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _s: store)
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_SUCCESS_BODY)

    _patch_client(monkeypatch, handler)

    transcript = await _worker_provider().transcribe(_req(tmp_path))

    assert transcript.language == "es"
    key = "work/ten_xyz/str_xyz/audio.wav"
    assert seen[0]["audio_url"].startswith(f"https://bucket.example/{key}")
    # The WAV actually landed in the bucket before the POST…
    assert store.deleted == [key]  # …and the transient object was dropped.


async def test_worker_mode_needs_object_storage_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _s: None)
    with pytest.raises(TranscriptionError, match="OBJECT_STORAGE"):
        await _worker_provider().transcribe(_req(tmp_path))


async def test_worker_mode_missing_audio_file_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        storage_mod, "build_artifact_store", lambda _s: _FakeStore()
    )
    req = TranscribeRequest(
        audio_path=str(tmp_path / "nope.wav"),
        stream_id="str_xyz",
        tenant_id="ten_xyz",
        language="es",
    )
    with pytest.raises(TranscriptionError, match="audio file missing"):
        await _worker_provider().transcribe(req)


async def test_failed_transcribe_keeps_work_object_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup only fires once a transcript lands — a 5xx leaves the
    object so a retry doesn't have to re-upload (bucket lifecycle rule
    is the backstop, see runbook)."""
    store = _FakeStore()
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _s: store)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="worker crashed")

    _patch_client(monkeypatch, handler)

    with pytest.raises(TranscriptionError, match="HTTP 500"):
        await _worker_provider().transcribe(_req(tmp_path))
    assert store.deleted == []
    assert "work/ten_xyz/str_xyz/audio.wav" in store.objects


async def test_web_box_mode_unchanged_without_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default mode still requires the signing secret — the worker flag
    must not loosen the web box's contract."""
    provider = modal_whisper.ModalWhisperProvider(
        endpoint_url="https://modal.test",
        bearer_token="bear",
        signing_secret="",
        public_base_url="https://chalybclip.test",
        model="small",
    )
    with pytest.raises(TranscriptionError, match="SIGNING_SECRET"):
        await provider.transcribe(_req(tmp_path))

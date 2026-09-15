"""AssemblyAI transcribe provider — Task A1.

We pin the request/response shape against mocked HTTP exchanges so the
karaoke-caption JSON stays bit-for-bit compatible with the existing
clip_render.html consumer. The mapper (`_to_transcript`) is the part
that matters most — the upload/submit/poll wiring is plumbing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from chalybclip.errors import TranscriptionError
from chalybclip.transcribe.models import Transcript
from chalybclip.transcribe.providers import assemblyai
from chalybclip.transcribe.providers.base import TranscribeRequest

# Capture before any monkeypatching so the deadline test (which needs a
# real micro-sleep to advance time.monotonic) doesn't recurse into the
# autouse no-op patch.
_REAL_SLEEP = asyncio.sleep
_REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(_s: float) -> None:
        return None
    monkeypatch.setattr(assemblyai.asyncio, "sleep", _noop)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    def _factory(*args, **kwargs):
        # Forward the provider's headers/timeout so the default
        # Authorization header reaches the MockTransport.
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(assemblyai.httpx, "AsyncClient", _factory)


def _make_provider(**overrides) -> assemblyai.AssemblyAIProvider:
    defaults = dict(
        api_key="aai_test_key",
        language_mode="auto",
        speaker_labels=True,
        speech_models=["universal-3-pro", "universal-2"],
        polling_interval_s=0.01,
        request_timeout_s=2.0,
    )
    defaults.update(overrides)
    return assemblyai.AssemblyAIProvider(**defaults)


def _req(tmp_path: Path, *, language: str | None = None) -> TranscribeRequest:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFFAKEWAVE")
    return TranscribeRequest(
        audio_path=audio,
        stream_id="str_test",
        tenant_id="ten_test",
        language=language,
    )


_COMPLETED_BODY = {
    "id": "tr_abc",
    "status": "completed",
    "language_code": "es",
    "audio_duration": 12.5,
    "text": "Hola mundo. Adiós.",
    "utterances": [
        {
            "start": 0,
            "end": 1500,
            "text": "Hola mundo.",
            "speaker": "A",
            "confidence": 0.95,
            "words": [
                {"text": "Hola", "start": 0, "end": 500, "confidence": 0.9, "speaker": "A"},
                {"text": "mundo.", "start": 500, "end": 1500, "confidence": 0.92, "speaker": "A"},
            ],
        },
        {
            "start": 2000,
            "end": 3500,
            "text": "Adiós.",
            "speaker": "B",
            "confidence": 0.91,
            "words": [
                {"text": "Adiós.", "start": 2000, "end": 3500, "confidence": 0.93, "speaker": "B"},
            ],
        },
    ],
}


def test_constructor_requires_api_key() -> None:
    with pytest.raises(TranscriptionError, match="api_key is required"):
        assemblyai.AssemblyAIProvider(api_key="")


def test_provider_name_reflects_primary_model() -> None:
    """The name reflects the FIRST model in the ladder — the one AAI
    tries first. Operators reading logs / events know which provider
    is the primary, regardless of which fallback ran on a given job."""
    p = _make_provider(speech_models=["universal-3-pro", "universal-2"])
    assert p.name == "assemblyai-universal-3-pro"
    p2 = _make_provider(speech_models=["nano"])
    assert p2.name == "assemblyai-nano"


@pytest.mark.asyncio
async def test_happy_path_returns_transcript_with_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full flow: upload → submit → poll once → completed body.
    Asserts the karaoke-relevant fields (segments[].words[].ts/end_ts/text)
    are populated correctly and speakers list dedupes."""
    state = {"phase": "upload"}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if state["phase"] == "upload":
            assert request.method == "POST"
            assert "/v2/upload" in url
            assert request.headers["authorization"] == "aai_test_key"
            state["phase"] = "submit"
            return httpx.Response(200, json={"upload_url": "https://cdn/audio.bin"})
        if state["phase"] == "submit":
            assert request.method == "POST"
            assert "/v2/transcript" in url
            body = request.read()
            import json as _json
            payload = _json.loads(body)
            assert payload["audio_url"] == "https://cdn/audio.bin"
            assert payload["speaker_labels"] is True
            # Default provider (language_mode="auto") uses
            # language_detection + code-switching prompt, NOT
            # language_code. AAI's Universal-3 Pro handles mixed
            # ES/EN audio natively when both are sent together.
            assert payload["language_detection"] is True
            assert payload["prompt"] == assemblyai._CODE_SWITCHING_PROMPT
            assert "language_code" not in payload
            # Model ladder — primary first, fallback second.
            assert payload["speech_models"] == [
                "universal-3-pro", "universal-2",
            ]
            state["phase"] = "poll"
            return httpx.Response(200, json={"id": "tr_abc", "status": "queued"})
        # poll
        assert request.method == "GET"
        assert "/v2/transcript/tr_abc" in url
        return httpx.Response(200, json=_COMPLETED_BODY)

    _patch_client(monkeypatch, handler)
    transcript = await _make_provider().transcribe(_req(tmp_path))
    assert isinstance(transcript, Transcript)
    assert transcript.language == "es"
    assert transcript.duration_s == pytest.approx(12.5)
    assert transcript.model == "assemblyai-universal-3-pro"
    assert transcript.speakers == ["A", "B"]

    assert len(transcript.segments) == 2
    s0 = transcript.segments[0]
    assert s0.ts == pytest.approx(0.0)
    assert s0.end_ts == pytest.approx(1.5)
    assert s0.text == "Hola mundo."
    assert s0.speaker == "A"
    assert [w.text for w in s0.words] == ["Hola", "mundo."]
    # Word timestamps in seconds (AssemblyAI returns ms — mapper converts).
    assert s0.words[1].ts == pytest.approx(0.5)
    assert s0.words[1].end_ts == pytest.approx(1.5)
    # Confidence preserved + clamped to [0,1].
    assert 0.0 <= s0.words[0].prob <= 1.0


def _make_capture_handler(captured: dict):
    """Tiny helper — mock transport that records the submit payload."""
    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            import json as _json
            captured.update(_json.loads(request.read()))
            return httpx.Response(200, json={"id": "tr_x", "status": "queued"})
        return httpx.Response(200, json=_COMPLETED_BODY)
    return handler


@pytest.mark.asyncio
async def test_auto_mode_sends_detection_plus_code_switching_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default `language_mode="auto"` (LATAM default) submits with
    language_detection=true PLUS the code-switching prompt. The prompt
    tells U3-Pro to preserve each phrase in its original language
    instead of "helpfully" translating English filler to Spanish."""
    captured: dict = {}
    _patch_client(monkeypatch, _make_capture_handler(captured))
    provider = _make_provider(language_mode="auto")
    await provider.transcribe(_req(tmp_path))
    assert captured["language_detection"] is True
    assert captured["prompt"] == assemblyai._CODE_SWITCHING_PROMPT
    assert "language_code" not in captured


@pytest.mark.asyncio
async def test_locked_language_drops_detection_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator locks the deployment to ES via `language_mode="es"`
    (e.g. dedicated Spanish-only channel) — drops detection + prompt
    for max-accuracy monolingual transcription. AAI rejects the
    combination of language_code + language_detection so we must
    send only one."""
    captured: dict = {}
    _patch_client(monkeypatch, _make_capture_handler(captured))
    provider = _make_provider(language_mode="es")
    await provider.transcribe(_req(tmp_path))
    assert captured["language_code"] == "es"
    assert "language_detection" not in captured
    assert "prompt" not in captured


@pytest.mark.asyncio
async def test_per_call_language_overrides_provider_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-call `language` (from the pipeline / API) wins over the
    deployment's default mode. So an operator with an EN-only stream
    on an otherwise auto-mode deployment locks to EN cleanly."""
    captured: dict = {}
    _patch_client(monkeypatch, _make_capture_handler(captured))
    provider = _make_provider(language_mode="auto")
    await provider.transcribe(_req(tmp_path, language="en"))
    assert captured["language_code"] == "en"
    assert "language_detection" not in captured
    assert "prompt" not in captured


@pytest.mark.asyncio
async def test_per_call_auto_switches_back_to_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-call `language="auto"` on a locked deployment switches back
    to detection mode for that specific stream — symmetric override."""
    captured: dict = {}
    _patch_client(monkeypatch, _make_capture_handler(captured))
    provider = _make_provider(language_mode="es")
    await provider.transcribe(_req(tmp_path, language="auto"))
    assert captured["language_detection"] is True
    assert captured["prompt"] == assemblyai._CODE_SWITCHING_PROMPT
    assert "language_code" not in captured


@pytest.mark.asyncio
async def test_speech_models_ladder_always_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model ladder is non-optional — every submit carries
    speech_models so AAI walks U3-Pro → U2 → … in order. Pinned here
    so a future refactor doesn't accidentally drop the field."""
    captured: dict = {}
    _patch_client(monkeypatch, _make_capture_handler(captured))
    provider = _make_provider(
        speech_models=["universal-3-pro", "universal-2", "nano"],
    )
    await provider.transcribe(_req(tmp_path))
    assert captured["speech_models"] == [
        "universal-3-pro", "universal-2", "nano",
    ]


@pytest.mark.asyncio
async def test_upload_failure_surfaces_as_transcription_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid key")

    _patch_client(monkeypatch, handler)
    with pytest.raises(TranscriptionError, match="upload failed.*401"):
        await _make_provider().transcribe(_req(tmp_path))


@pytest.mark.asyncio
async def test_error_status_surfaces_with_error_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AssemblyAI returns status=error with an error field when
    transcription fails server-side. Provider surfaces the message."""
    error_body = {"id": "tr_x", "status": "error", "error": "Audio too short"}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "tr_x", "status": "queued"})
        return httpx.Response(200, json=error_body)

    _patch_client(monkeypatch, handler)
    with pytest.raises(TranscriptionError, match="Audio too short"):
        await _make_provider().transcribe(_req(tmp_path))


@pytest.mark.asyncio
async def test_missing_audio_file_raises_before_any_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Don't waste an API call when the file is missing locally —
    surface the FS error first."""
    p = _make_provider()
    req = TranscribeRequest(
        audio_path=tmp_path / "nope.wav",
        stream_id="str_x", tenant_id="ten_x", language=None,
    )
    with pytest.raises(TranscriptionError, match="audio file missing"):
        await p.transcribe(req)


@pytest.mark.asyncio
async def test_no_utterances_falls_back_to_sentence_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When speaker_labels is off (or AssemblyAI doesn't emit utterances
    for whatever reason), the mapper builds segments from words by
    sentence-ending punctuation. Caption pipeline still works."""
    body = {
        "id": "tr_x", "status": "completed",
        "language_code": "es", "audio_duration": 3.0,
        "utterances": [],  # explicitly empty
        "words": [
            {"text": "Hola", "start": 0, "end": 500, "confidence": 0.9},
            {"text": "mundo.", "start": 500, "end": 1500, "confidence": 0.9},
            {"text": "Adiós.", "start": 2000, "end": 3000, "confidence": 0.9},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "tr_x", "status": "queued"})
        return httpx.Response(200, json=body)

    _patch_client(monkeypatch, handler)
    transcript = await _make_provider(speaker_labels=False).transcribe(_req(tmp_path))
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Hola mundo."
    assert transcript.segments[1].text == "Adiós."
    # No speaker info → field is None, speakers list empty.
    assert all(s.speaker is None for s in transcript.segments)
    assert transcript.speakers == []


@pytest.mark.asyncio
async def test_poll_respects_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck AssemblyAI job (queued forever) bails after deadline_s."""
    real_calls = {"n": 0}

    async def _short_sleep(_s: float) -> None:
        real_calls["n"] += 1
        await _REAL_SLEEP(0.001)
    monkeypatch.setattr(assemblyai.asyncio, "sleep", _short_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "tr_stuck", "status": "queued"})
        return httpx.Response(200, json={"id": "tr_stuck", "status": "queued"})

    _patch_client(monkeypatch, handler)
    provider = _make_provider(request_timeout_s=0.05, polling_interval_s=0.01)
    with pytest.raises(TranscriptionError, match="poll deadline"):
        await provider.transcribe(_req(tmp_path))


def test_mapper_handles_missing_fields_gracefully() -> None:
    """Direct unit-test of the mapper — a malformed AAI response (e.g.
    missing duration, garbage timestamps) shouldn't crash; we
    coerce to safe defaults."""
    minimal = {
        "id": "tr_x", "status": "completed",
        "language_code": "es",
        "utterances": [{"start": 0, "end": 1000, "text": "hi", "speaker": "A", "words": []}],
    }
    req = TranscribeRequest(
        audio_path=Path("/dev/null"), stream_id="s", tenant_id="t",
    )
    out = assemblyai._to_transcript(minimal, req=req, model_name="assemblyai-best")
    assert out.duration_s == 0.0  # missing audio_duration → 0
    assert len(out.segments) == 1
    assert out.segments[0].speaker == "A"

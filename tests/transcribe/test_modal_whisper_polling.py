"""Modal Whisper provider — polling-redirect handling.

Modal returns 303 with `Location: ?__modal_function_call_id=fc-...`
for any long-running web-endpoint call. The provider used to treat
that as a hard error (operator-reported on 06-03:
`modal whisper returned 303: `). These tests pin the new polling
behavior: follow the Location until a terminal status lands.

We monkeypatch `asyncio.sleep` to zero so the test polling completes
in <1ms instead of waiting `_POLL_INTERVAL_S` per iteration.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

# Capture the real async-sleep BEFORE any monkeypatch so test fixtures
# that need a tiny real sleep (for the deadline test) don't recurse
# into the autouse no-op patch.
_REAL_SLEEP = asyncio.sleep

from chalybclip.errors import TranscriptionError
from chalybclip.transcribe.models import Transcript
from chalybclip.transcribe.providers import modal_whisper
from chalybclip.transcribe.providers.base import TranscribeRequest


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the poll loop tick instantly. The loop lives in
    `chalybclip.integrations.modal_http` (shared since Phase 2b) and calls
    `asyncio.sleep` — patching the module attribute is global, so this
    covers it regardless of which module runs the loop."""
    async def _noop(_s: float) -> None:
        return None
    monkeypatch.setattr(asyncio, "sleep", _noop)


def _make_provider(endpoint: str = "https://modal.test") -> modal_whisper.ModalWhisperProvider:
    return modal_whisper.ModalWhisperProvider(
        endpoint_url=endpoint,
        bearer_token="bear",
        signing_secret="sigsig",
        public_base_url="https://chalybclip.test",
        model="small",
        # Short deadline so the polling-deadline tests don't have to
        # wait through a long timeout.
        request_timeout_s=2.0,
    )


def _req() -> TranscribeRequest:
    return TranscribeRequest(
        audio_path="/tmp/x.wav",
        stream_id="str_xyz",
        tenant_id="ten_xyz",
        language="es",
    )


_SUCCESS_BODY = {
    "language": "es",
    "duration_s": 12.0,
    "model": "small",
    "segments": [
        {
            "text": "hola",
            "start": 0.0,
            "end": 1.0,
            "words": [
                {"word": "hola", "start": 0.0, "end": 1.0, "prob": 0.99}
            ],
        }
    ],
}


_REAL_ASYNC_CLIENT = httpx.AsyncClient  # capture BEFORE any monkeypatch


def _mock_transport(handler):
    """Build an httpx MockTransport with a callable per request."""
    return httpx.MockTransport(handler)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Replace `modal_whisper.httpx.AsyncClient` with a factory that
    builds the REAL class against a MockTransport. We capture the real
    class above before any patching so the factory doesn't recurse on
    itself."""
    monkeypatch.setattr(
        modal_whisper.httpx, "AsyncClient",
        lambda *a, **kw: _REAL_ASYNC_CLIENT(transport=_mock_transport(handler))
    )


@pytest.mark.asyncio
async def test_direct_200_response_returns_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small/fast Modal app might return the JSON on the first POST
    without redirecting. That path stays unchanged: parse + return."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(200, json=_SUCCESS_BODY)

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    transcript = await provider.transcribe(_req())
    assert isinstance(transcript, Transcript)
    assert transcript.language == "es"
    assert transcript.segments[0].text == "hola"


@pytest.mark.asyncio
async def test_303_with_polling_url_eventually_lands_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator-hit case: POST returns 303 with a Modal polling URL,
    the poll URL keeps returning 303 for a few rounds, then a 200 with
    the body. Provider should walk through all of them and return the
    transcript."""
    poll_url = "https://modal.test/?__modal_function_call_id=fc-abc"
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["value"] += 1
        if request.method == "POST":
            return httpx.Response(303, headers={"location": poll_url})
        # GET path: 2 redirects then success
        if call_count["value"] <= 3:
            return httpx.Response(303, headers={"location": poll_url})
        return httpx.Response(200, json=_SUCCESS_BODY)

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    transcript = await provider.transcribe(_req())
    assert transcript.language == "es"
    # 1 POST + 3 GETs (two redirects + the final 200)
    assert call_count["value"] == 4


@pytest.mark.asyncio
async def test_303_without_location_surfaces_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed redirect (no Location header) is a protocol error;
    don't loop forever. Provider returns the 303 to the caller, whose
    raise_for_status surfaces it as a TranscriptionError with the
    original status code."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(303)  # no location

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    with pytest.raises(TranscriptionError, match="returned 303"):
        await provider.transcribe(_req())


@pytest.mark.asyncio
async def test_terminal_500_surfaces_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When polling lands on a 500, the caller's raise_for_status
    surfaces it as a normal TranscriptionError. Not a polling concern —
    just a sanity check the new flow keeps the failure path."""
    poll_url = "https://modal.test/?__modal_function_call_id=fc-abc"
    calls = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["value"] += 1
        if request.method == "POST":
            return httpx.Response(303, headers={"location": poll_url})
        return httpx.Response(500, text="modal crashed")

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    # Classifier surfaces 5xx as "service error (HTTP N)" instead of
    # the generic "returned N" — actionable hint to retry first then
    # fall back to AssemblyAI if Modal stays down.
    with pytest.raises(TranscriptionError, match="HTTP 500"):
        await provider.transcribe(_req())


@pytest.mark.asyncio
async def test_polling_respects_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Modal function that runs past the configured timeout should
    abort polling instead of looping forever."""
    poll_url = "https://modal.test/?__modal_function_call_id=fc-abc"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(303, headers={"location": poll_url})
        return httpx.Response(303, headers={"location": poll_url})

    # Re-enable real sleep so the deadline actually expires. Patch with
    # a tiny scaled value so we don't burn 2s in the test.
    real_calls = {"n": 0}

    async def _short_sleep(_s: float) -> None:
        # Use the captured real sleep so the deadline check (which reads
        # time.monotonic) sees actual wall clock advance. Patched-sleep
        # recursion is why this needs the captured ref, not the bound name.
        real_calls["n"] += 1
        await _REAL_SLEEP(0.001)

    monkeypatch.setattr(asyncio, "sleep", _short_sleep)
    # Shrink the per-poll interval so the deadline math triggers quickly.
    # (The provider reads its module-level constant at call time and
    # passes it into the shared poll loop, so this patch still bites.)
    monkeypatch.setattr(modal_whisper, "_POLL_INTERVAL_S", 0.01)
    _patch_client(monkeypatch, handler)

    provider = modal_whisper.ModalWhisperProvider(
        endpoint_url="https://modal.test",
        bearer_token="bear",
        signing_secret="sigsig",
        public_base_url="https://chalybclip.test",
        model="small",
        request_timeout_s=0.05,  # 50ms — polls expire fast
    )
    with pytest.raises(TranscriptionError, match="poll deadline"):
        await provider.transcribe(_req())


# ---- error classification ----


@pytest.mark.asyncio
async def test_429_billing_cap_surfaces_actionable_assemblyai_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator-hit on 06-03: Modal returned 429 with 'workspace billing
    cycle spend limit reached'. Pre-classifier the message was raw
    HTTP. Now we tell the operator how to switch to AssemblyAI in
    one env-var flip."""
    body = (
        "modal-http: Webhook failed: workspace billing cycle "
        "spend limit reached"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text=body)

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    with pytest.raises(TranscriptionError) as excinfo:
        await provider.transcribe(_req())
    msg = str(excinfo.value)
    # The classifier should mention BOTH options the operator has:
    # AssemblyAI swap + bumping the Modal spend cap.
    assert "billing/spend cap" in msg or "billing" in msg.lower()
    assert "CHALYBCLIP_TRANSCRIBE_PROVIDER=assemblyai" in msg
    assert "CHALYBCLIP_ASSEMBLYAI_API_KEY" in msg
    assert "modal.com/settings/billing" in msg
    # Raw body is preserved at the tail so the operator can see what
    # Modal actually said.
    assert body in msg


@pytest.mark.asyncio
async def test_401_surfaces_bearer_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid bearer")

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    with pytest.raises(TranscriptionError) as excinfo:
        await provider.transcribe(_req())
    msg = str(excinfo.value)
    assert "CHALYBCLIP_MODAL_TOKEN" in msg
    assert "MODAL_BEARER_TOKEN" in msg
    # AAI escape hatch still surfaces in the auth path.
    assert "CHALYBCLIP_TRANSCRIBE_PROVIDER=assemblyai" in msg


@pytest.mark.asyncio
async def test_503_surfaces_service_down_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Modal worker crashed")

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    with pytest.raises(TranscriptionError) as excinfo:
        await provider.transcribe(_req())
    msg = str(excinfo.value)
    assert "service error" in msg.lower() or "5" in msg
    assert "Retry" in msg or "retry" in msg


@pytest.mark.asyncio
async def test_unrecognized_400_falls_through_to_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 with no billing/auth keyword goes through the generic
    'modal whisper returned NNN: …' path — same shape we shipped
    pre-classifier, so existing log greps still work."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad input")

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    with pytest.raises(TranscriptionError) as excinfo:
        await provider.transcribe(_req())
    msg = str(excinfo.value)
    assert "modal whisper returned 400" in msg
    assert "bad input" in msg


@pytest.mark.asyncio
async def test_429_without_billing_keyword_falls_through_to_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 that's just a rate-limit (no billing) doesn't get the
    AssemblyAI swap pitch — that's a transient retryable, not a
    billing cliff."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="too many requests")

    _patch_client(monkeypatch, handler)
    provider = _make_provider()
    with pytest.raises(TranscriptionError) as excinfo:
        await provider.transcribe(_req())
    msg = str(excinfo.value)
    assert "modal whisper returned 429" in msg
    # NO billing-cliff hint when the body doesn't say "billing/spend".
    assert "CHALYBCLIP_TRANSCRIBE_PROVIDER=assemblyai" not in msg

"""Whisper transcription — orchestrator.

Holds the idempotence + tenant-validation + write-cache logic. The
actual audio→Transcript step is delegated to a `TranscribeProvider`
so deployments can pick local Whisper (default) or a cloud STT
vendor without touching pipeline.py.

The transcript is saved alongside the audio at
`<stream_dir>/source/transcript.json`. Idempotent: a second call
returns the cached `Transcript` unless `force=True`.

Task A1b — cost metering. When the provider exposes
`cost_rate_per_second_micros` (AssemblyAI does, LocalWhisperProvider
doesn't) and a `db` is passed in, the service:
  1. Pre-call: checks the tenant's daily LLM budget via
     BudgetGovernor.check_llm_spend → refuses with BudgetExceeded if
     today is already over cap.
  2. Post-call: records an LLMCallRow with provider/model/cost so
     the call shows up on /dashboard/llm-calls alongside Anthropic
     LLM calls. Same accounting → same governor → same credit pass-
     through.
Providers without a cost rate fall through the same code path with
a 0-cost record. Callers without a db skip both steps (CLI / tests).
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from chalybclip.errors import TranscriptionError
from chalybclip.ids import new_id
from chalybclip.ingest import Stream

from .models import Transcript
from .providers import TranscribeProvider, TranscribeRequest, get_provider

if TYPE_CHECKING:
    from chalybclip.db import Database

_log = structlog.get_logger(__name__)


def _transcript_path(stream: Stream) -> Path:
    return stream.source_audio_path.parent / "transcript.json"


async def transcribe(
    tenant_id: str,
    stream: Stream,
    *,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = "es",
    force: bool = False,
    provider: TranscribeProvider | None = None,
    db: "Database | None" = None,
) -> Transcript:
    """Run the configured TranscribeProvider on `stream.source_audio_path`
    and save `transcript.json`.

    Args:
        tenant_id: Tenant owning the stream.
        stream: The ingested stream.
        model_size: faster-whisper model name. Only honored when the
            configured provider is local Whisper; cloud providers
            ignore this and read their own knobs from Settings.
        device: `cuda` or `cpu`. Same caveat as `model_size`.
        compute_type: `float16` / `int8` / etc. Same caveat.
        language: ISO 639-1 code; pass `None` to auto-detect.
        force: Re-run even when `transcript.json` already exists.
        provider: Override the configured provider (tests inject
            `FakeTranscribeProvider` here).

    Note: `model_size` / `device` / `compute_type` are kept on the
    signature for backward compatibility with the pipeline call site.
    They override the Settings-driven defaults of the local provider
    when supplied; cloud providers ignore them entirely.
    """
    if tenant_id != stream.tenant_id:
        raise TranscriptionError(
            f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}"
        )

    # Cache check BEFORE the source-audio guard: a completed stream's
    # transcript.json is the durable output, and the raw source audio may
    # have been reclaimed after the pipeline finished (the source VOD is
    # deleted on completion — see pipeline.process_vod). A non-forced
    # re-run must still return the cached transcript without the audio
    # present, so the pipeline stays idempotent post-cleanup. We only
    # require the audio file when we actually have to transcribe.
    out_path = _transcript_path(stream)
    if not force and out_path.exists():
        try:
            return Transcript.model_validate_json(out_path.read_text("utf-8"))
        except Exception as e:  # corrupt cache must not wedge the stream
            # A truncated/corrupt transcript.json (crash mid-write, disk
            # full) must not permanently break every later non-forced run.
            # Fall through and re-transcribe; if the source audio was
            # already reclaimed, the audio-missing error below at least
            # names an actionable condition instead of a validation dump.
            _log.warning(
                "transcribe.cache_invalid",
                path=str(out_path),
                reason=str(e)[:300],
                stream_id=stream.id,
            )

    if not stream.source_audio_path.exists():
        raise TranscriptionError(f"audio file missing: {stream.source_audio_path}")

    # Pick the provider. Caller can inject (tests do); otherwise we
    # honor the per-call overrides for local Whisper to keep the old
    # pipeline.py call shape working without a config change.
    if provider is None:
        from .providers.factory import get_provider as _get_provider
        from .providers.local_whisper import LocalWhisperProvider

        # If the call site is passing per-call overrides for the local
        # Whisper knobs (which the pipeline does), build a fresh local
        # provider with those values. Otherwise use the configured one.
        configured = _get_provider()
        if isinstance(configured, LocalWhisperProvider):
            provider = LocalWhisperProvider(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                # Inherit subprocess-isolation default from the factory.
                use_subprocess=configured._use_subprocess,  # noqa: SLF001
            )
        else:
            provider = configured

    # Task A1b — pre-call budget check. Only enforced when:
    #   - caller supplied a db (CLI / tests skip)
    #   - the provider declares a marginal cost (LocalWhisperProvider
    #     doesn't define cost_for_duration_micros, so the hasattr
    #     fallback skips the gate)
    # The check itself reads `tenants.daily_llm_budget_usd_micros` and
    # today's running spend from the same llm_calls table; the
    # post-call record below will land there too.
    provider_has_cost = hasattr(provider, "cost_for_duration_micros")
    if db is not None and provider_has_cost:
        from chalybclip.governance.budget import BudgetGovernor
        await BudgetGovernor(db).check_llm_spend(tenant_id)

    transcript = await provider.transcribe(
        TranscribeRequest(
            audio_path=stream.source_audio_path,
            stream_id=stream.id,
            tenant_id=tenant_id,
            language=language,
        )
    )
    # Atomic write: transcript.json is the step's durable idempotency
    # cache and the source audio may be reclaimed after the run — a crash
    # mid-write must never leave a half-written cache behind.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, out_path)

    # Task A1b — post-call cost record. Same llm_calls table the
    # Anthropic router writes into, so /dashboard/llm-calls and the
    # daily-spend ceiling treat AssemblyAI calls identically. Cost is
    # computed from the transcript's actual duration_s, not the upfront
    # request, so an audio file that's shorter than the pre-pass
    # estimate doesn't get over-billed.
    if db is not None and provider_has_cost:
        await _record_transcribe_cost(
            db=db,
            tenant_id=tenant_id,
            provider=provider,
            transcript=transcript,
        )

    return transcript


async def _record_transcribe_cost(
    *,
    db: "Database",
    tenant_id: str,
    provider: TranscribeProvider,
    transcript: Transcript,
) -> None:
    """Append one llm_calls row capturing the transcribe spend.

    Best-effort: if the write fails (DB hiccup, tenancy not bound),
    we log + swallow — the transcript itself is already on disk and
    the operator's pipeline shouldn't fail because the audit row
    didn't land. Matches the LLM router's pattern.
    """
    from chalybclip.db import LLMCallsRepo
    from chalybclip.db.models import LLMCallRow
    from chalybclip.tenancy import bound_tenant

    cost_micros = provider.cost_for_duration_micros(transcript.duration_s)
    # Token T3 — attribute transcription cost to the run's stream, read
    # from the structlog contextvar the pipeline binds for the run.
    import structlog as _structlog
    _stream_id = _structlog.contextvars.get_contextvars().get("stream_id")
    row = LLMCallRow(
        id=new_id("llmc"),
        tenant_id=tenant_id,
        purpose="transcribe",
        provider=provider.name,
        stream_id=_stream_id,
        # No "model" in the LLM sense for STT; reuse the provider's
        # own model token (e.g. "best" / "nano" / "medium").
        model=getattr(provider, "_speech_model", None) or provider.name,
        quality="standard",
        input_tokens=0,
        output_tokens=0,
        cost_usd_micros=cost_micros,
        status="ok",
        error=None,
        attempts=1,
        ts=_dt.datetime.now(_dt.UTC).isoformat(),
    )
    try:
        with bound_tenant(tenant_id):
            await LLMCallsRepo(db).record(row)
    except Exception as e:  # noqa: BLE001 — observability never breaks pipeline
        import structlog
        structlog.get_logger(__name__).warning(
            "transcribe.cost_record.failed",
            tenant_id=tenant_id, error=str(e),
        )

    # Token T4 — also report this provider's usage to Chalyb so the
    # balance reflects ALL spend, not just Claude. Native amount =
    # transcript seconds (kind="transcription.seconds"); cost = the real
    # USD micros computed above. Fire-and-forget, same as the LLM path.
    try:
        from chalybclip.integrations.chalyb.reporter import schedule_usage

        schedule_usage(
            db,
            tenant_id=tenant_id,
            kind="transcription.seconds",
            amount=max(0, int(round(transcript.duration_s))),
            cost_usd_micros=cost_micros,
            source_id=row.id,
            occurred_at_iso=row.ts,
            # Explicit provider name ("assemblyai") for per-provider cost
            # rollups on the Chalyb side.
            provider=provider.name,
            operation="transcribe",
        )
    except Exception as e:  # noqa: BLE001 — never breaks the pipeline
        import structlog
        structlog.get_logger(__name__).warning(
            "transcribe.usage_report.failed",
            tenant_id=tenant_id, error=str(e),
        )


def load_transcript(stream_dir: Path) -> Transcript:
    """Load a previously-saved Transcript from `<stream_dir>/source/transcript.json`."""
    path = Path(stream_dir) / "source" / "transcript.json"
    if not path.exists():
        raise TranscriptionError(f"transcript not found at {path}")
    return Transcript.model_validate_json(path.read_text("utf-8"))


# Re-export for callers that imported these from service.py historically.
# (The implementations now live in providers/local_whisper.py; this is the
# minimal shim so we don't break in-flight imports.)
__all__ = ["load_transcript", "transcribe"]


# --- Provide get_provider in this namespace too. -----------------
# Defensive: anyone who imported `get_provider` from
# `chalybclip.transcribe.service` keeps working.
def __getattr__(name: str) -> object:
    if name == "get_provider":
        from .providers import get_provider as _gp
        return _gp
    raise AttributeError(name)

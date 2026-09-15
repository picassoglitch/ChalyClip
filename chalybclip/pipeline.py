"""End-to-end VOD-to-clips pipeline.

Wires the six Phase 0 steps in order:

    ingest → transcribe → detect → cut → variants(per clip) → manifest

Each step's service function is already idempotent on its own output file,
so the orchestrator just calls them in sequence; no extra resume bookkeeping
needed. `force=True` propagates down to all of them.

The final `manifest.json` written at `<output_dir>/<stream_id>/` summarizes the
full run — stream metadata, transcript stats, all candidates, every clip with
its variants, and a roll-up of LLM spend (read from `llm_calls.jsonl`).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from chalybclip.clip import Clip, cut_clips
from chalybclip.clip.overlay_defaults import default_overlay_config
from chalybclip.config import ChalybClipConfig, load_config
from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    PersonasRepo,
    StreamsRepo,
    TranscriptsRepo,
    VariantsRepo,
    VodSpeakersRepo,
    db_session,
)
from chalybclip.db.adapters import (
    candidate_to_row,
    clip_to_row,
    stream_to_row,
    transcript_to_row,
    variant_to_row,
)
from chalybclip.detect import (
    Candidate,
    CandidateBatch,
    detect_candidates,
    detect_viral_moments,
    fallback_interval_candidates,
    save_candidates,
)
from chalybclip.diarize import (
    Diarization,
    diarization_from_transcript,
    diarize,
    resolve_speakers,
)
from chalybclip.errors import ChalybClipError, DetectionError, VariantError
from chalybclip.events import (
    CLIP_READY_FOR_REVIEW,
    STREAM_CREATED,
    STREAM_PROCESSED,
    emit,
)
from chalybclip.ingest import Stream, ingest_vod, load_chat_replay
from chalybclip.llm import LLMConfig, LLMRouter, Variant, load_llm_config
from chalybclip.llm.config import Quality
from chalybclip.logging import get_logger
from chalybclip.settings import Settings, get_settings
from chalybclip.transcribe import Transcript, transcribe
from chalybclip.variants import Persona, load_personas
from chalybclip.vision import analyze_video as _analyze_video
from chalybclip.vision import load_visual_signals

_MANIFEST_SCHEMA_VERSION = 1
_log = get_logger("chalybclip.pipeline")


@dataclass
class _StepCtx:
    """Mutable handle yielded by `_step` so inner code can mark a step
    skipped + add a one-line `note`. Surfaces in the dashboard as
    "(skipped — reason)" rather than misleading "0.0s".

    The default state is `skipped=False, note=""` → behaves like the
    pre-handle version. Steps that hit cached / disabled / no-op paths
    set `ctx.skipped = True` and a short reason in `ctx.note`.
    """

    skipped: bool = False
    note: str = ""


@contextmanager
def _step(name: str, *, db: Database | None = None, **fields: Any) -> Iterator[_StepCtx]:
    """Time + log one pipeline step. Re-raises typed errors with the bound context.

    Each step is wrapped to:
        - emit `step.<name>.start` / `step.<name>.done` events with `duration_s`
        - on `ChalybClipError`, append the bound contextvars to the message so
          the user sees the stream_id when the CLI prints the error

    When `db` is provided, also writes `pipeline.step.start` / `.done` /
    `.failed` rows to the events table so the dashboard can poll and show
    live progress without scraping logs. The DB write is synchronous via
    sqlite3 to keep the context manager sync — the dashboard doesn't need
    millisecond freshness, just "this step is running" / "this step finished".

    Yields a `_StepCtx` that inner code can mark `.skipped = True` (with
    an optional `.note`) when a step is a cached / disabled / no-op
    run. The done event then carries those fields and the dashboard
    renders "(skipped — note)" instead of confusing "0.0s" durations.
    """
    _log.info(f"step.{name}.start", **fields)
    t0 = time.perf_counter()
    stream_id = structlog.contextvars.get_contextvars().get("stream_id")
    _record_step_event(db, "pipeline.step.start", name, stream_id, fields)
    ctx = _StepCtx()
    try:
        yield ctx
    except ChalybClipError as e:
        duration_s = time.perf_counter() - t0
        _log.error(
            f"step.{name}.failed",
            duration_s=duration_s,
            error_class=type(e).__name__,
            error_msg=str(e),
            **fields,
        )
        _record_step_event(
            db,
            "pipeline.step.failed",
            name,
            stream_id,
            {**fields, "duration_s": duration_s, "error": str(e)},
        )
        # Append every contextvar (stream_id, tenant_id, persona_id) into the
        # error message so the CLI / log readers see it without inspection.
        cv = structlog.contextvars.get_contextvars()
        if cv:
            ctx_str = " ".join(f"{k}={v}" for k, v in cv.items())
            raise type(e)(f"{e} [{ctx_str}]") from e
        raise
    duration_s = time.perf_counter() - t0
    _log.info(
        f"step.{name}.done",
        duration_s=duration_s,
        skipped=ctx.skipped,
        note=ctx.note or None,
        **fields,
    )
    _record_step_event(
        db,
        "pipeline.step.done",
        name,
        stream_id,
        {
            **fields,
            "duration_s": duration_s,
            "skipped": ctx.skipped,
            "note": ctx.note,
        },
    )


# Holds references to fire-and-forget event-emit tasks on the Postgres path
# so they aren't garbage-collected before they run (the done-callback clears
# each one). Best-effort telemetry — never awaited.
_PENDING_EVENT_TASKS: set[asyncio.Task[Any]] = set()


def _record_step_event(
    db: Database | None,
    event_type: str,
    step_name: str,
    stream_id: str | None,
    fields: dict[str, Any],
) -> None:
    """Best-effort sync write to the events table.

    Uses raw sqlite3 because the surrounding `_step` is a sync context
    manager and the pipeline runs the same coro. We don't want a missing
    DB or transient disk error to bring the whole step down — the dashboard
    progress view degrades gracefully when an event row is missing.
    """
    if db is None:
        return
    tenant_id = structlog.contextvars.get_contextvars().get("tenant_id")
    if not tenant_id:
        return
    payload = {"step": step_name, "stream_id": stream_id, **fields}
    # Strip non-JSON-serializable values defensively.
    clean = {
        k: v
        for k, v in payload.items()
        if isinstance(v, str | int | float | bool | type(None) | list | dict)
    }
    try:
        if getattr(db, "is_postgres", False):
            # asyncpg has no sync API. This helper runs in the pipeline's sync
            # `_step` context, so fire-and-forget the async emit onto the
            # running loop (contextvars — incl. the bound tenant — are copied
            # into the task). No running loop → skip; events are best-effort
            # and the progress view degrades gracefully when one is missing.
            from chalybclip.db import EventsRepo

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            # Hold a reference until done so the task isn't GC'd mid-flight.
            task = loop.create_task(EventsRepo(db).emit(type=event_type, payload=clean))
            _PENDING_EVENT_TASKS.add(task)
            task.add_done_callback(_PENDING_EVENT_TASKS.discard)
            # Register with the owning Database so its close() drains this
            # write — otherwise the run's teardown can close the pool while
            # the emit is still queued ("pool is closed" after pipeline.done).
            track = getattr(db, "track_background_task", None)
            if callable(track):
                track(task)
            return

        import sqlite3

        with sqlite3.connect(db.path) as conn:
            conn.execute(
                "INSERT INTO events (id, tenant_id, type, payload_json, ts) VALUES (?, ?, ?, ?, ?)",
                (
                    f"evt_{step_name}_{int(time.time() * 1000)}",
                    tenant_id,
                    event_type,
                    json.dumps(clean),
                    _dt.datetime.now(_dt.UTC).isoformat(),
                ),
            )
            conn.commit()
    except Exception as e:  # noqa: BLE001 - intentional best-effort
        _log.debug("pipeline.step.event_write_failed", error=str(e))


class TranscriptSummary(BaseModel):
    """Compact summary so manifest.json doesn't carry the full transcript."""

    stream_id: str
    language: str
    duration_s: float
    model: str
    segment_count: int
    word_count: int

    @classmethod
    def from_transcript(cls, transcript: Transcript) -> TranscriptSummary:
        return cls(
            stream_id=transcript.stream_id,
            language=transcript.language,
            duration_s=transcript.duration_s,
            model=transcript.model,
            segment_count=len(transcript.segments),
            word_count=sum(len(s.words) for s in transcript.segments),
        )


class ClipEntry(BaseModel):
    """One clip with its persona-specific variant batch."""

    clip: Clip
    persona_id: str
    variants: list[Variant] = Field(default_factory=list)


class LLMSpend(BaseModel):
    """Cumulative roll-up of `llm_calls.jsonl` for this stream."""

    total_calls: int = 0
    total_cost_usd_micros: int = 0


class StreamManifest(BaseModel):
    """Full pipeline output, saved to `<stream_dir>/manifest.json`."""

    schema_version: int = _MANIFEST_SCHEMA_VERSION
    started_at: str
    completed_at: str
    tenant_id: str
    persona_id: str
    persona_name: str
    language: str
    n_variants_requested: int
    stream: Stream
    transcript: TranscriptSummary
    candidates: list[Candidate] = Field(default_factory=list)
    clip_entries: list[ClipEntry] = Field(default_factory=list)
    llm_spend: LLMSpend = Field(default_factory=LLMSpend)


async def _merge_db_personas(
    yaml_personas: dict[str, Persona],
    tenant_id: str,
    db_path: str,
) -> dict[str, Persona]:
    """Slice O.26 — overlay DB personas onto a YAML-loaded map.

    The dashboard's New Persona form writes to the DB (PersonasRepo);
    the YAML at `config/personas.yaml` is a static baseline that
    survives DB resets. The pipeline previously only knew about YAML
    so dashboard-created personas raised `unknown persona; known: (none)`
    even though the operator saw them in the dropdown.

    Merge rule: YAML first, then DB rows overlay (DB wins on id
    collision — they reflect operator edits). Tenant-scoped: only
    rows belonging to `tenant_id`.

    Db read failure is the caller's problem (raises through); pipeline
    catches it and falls back to YAML-only with a logged warning.
    """
    from chalybclip.db import Database, PersonasRepo
    from chalybclip.tenancy import bound_tenant

    db = Database(db_path)
    await db.connect()
    with bound_tenant(tenant_id):
        db_rows = await PersonasRepo(db).list_for_tenant()

    merged = dict(yaml_personas)
    for row in db_rows:
        # PersonaRow has tenant_id + target_languages + created_at on
        # top of the YAML Persona shape; the variants pipeline only
        # needs id/name/primary_language/voice_prompt/routing_tags.
        merged[row.id] = Persona(
            id=row.id,
            name=row.name,
            primary_language=row.primary_language,
            voice_prompt=row.voice_prompt,
            routing_tags=list(row.routing_tags or []),
        )
    return merged


@dataclass
class PipelineDeps:
    """Injectable dependencies — the CLI uses defaults, tests override."""

    config: ChalybClipConfig | None = None
    llm_config: LLMConfig | None = None
    personas: dict[str, Persona] | None = None
    settings: Settings | None = None
    router_factory: Callable[[Path], LLMRouter] | None = None
    clock: Callable[[], _dt.datetime] = field(
        default_factory=lambda: lambda: _dt.datetime.now(_dt.UTC)
    )


async def process_vod(
    tenant_id: str,
    vod_url: str,
    output_dir: Path,
    *,
    persona_id: str,
    stream_id: str | None = None,
    language: str | None = None,
    n_variants: int = 5,
    quality: Quality | None = None,
    force: bool = False,
    db_path: str | None = None,
    chat_replay_source: Path | None = None,
    deps: PipelineDeps | None = None,
) -> StreamManifest:
    """Run the full Phase 0 pipeline against `vod_url`. Returns the manifest.

    Pass `stream_id` to resume a known stream; ingest will reuse the cached
    `stream.json` instead of minting a new ULID, and every downstream step
    finds its idempotency cache.

    Pass `db_path` to dual-write through the SQLite tenancy + repos. The
    tenant must already exist in the DB (use `chalybclip tenants add`).
    Without `db_path`, the pipeline runs filesystem-only — useful for
    tests and Phase 0-style invocations.

    Pass `chat_replay_source` (a path to a JSONL of `ChatMessage` rows)
    to feed the chat heat detector. Phase 1 doesn't fetch chat replay
    from platforms automatically — Phase 2 will add Kick / Twitch / YT
    fetchers behind the same import path.
    """
    deps = deps or PipelineDeps()
    config = deps.config or load_config()
    llm_config = deps.llm_config or load_llm_config()
    # Slice O.26 — merge DB personas with YAML personas.
    # Before: pipeline only read config/personas.yaml. The dashboard saved
    # operator-created personas to the DB; the dropdown showed them
    # (_merged_personas in dashboard.py) but the pipeline didn't know
    # about them → "unknown persona 'xxx'; known: (none)" on every
    # upload after a fresh deploy.
    # After: YAML first (deterministic baseline), then DB rows overlay
    # (any operator edits win). Only merges when deps.personas wasn't
    # explicitly passed — tests can still inject a closed set.
    if deps.personas is not None:
        personas = deps.personas
    else:
        personas = load_personas()
        if db_path:
            try:
                personas = await _merge_db_personas(personas, tenant_id, db_path)
            except Exception as merge_err:  # noqa: BLE001 — defensive
                # DB read failed: log + fall through to YAML-only. That
                # surfaces a clearer error downstream than masking it.
                _log.warning(
                    "personas.db_merge_failed",
                    error=str(merge_err)[:200],
                    tenant_id=tenant_id,
                )
    settings = deps.settings or get_settings()

    if persona_id not in personas:
        known = ", ".join(sorted(personas)) or "(none)"
        raise VariantError(f"unknown persona {persona_id!r}; known: {known}")
    persona = personas[persona_id]

    started_at = deps.clock().isoformat()
    output_dir = Path(output_dir).resolve()

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id, persona_id=persona.id)
    try:
        async with _maybe_db_session(tenant_id=tenant_id, db_path=db_path) as db:
            return await _run_pipeline(
                tenant_id=tenant_id,
                vod_url=vod_url,
                output_dir=output_dir,
                persona=persona,
                stream_id=stream_id,
                language=language,
                n_variants=n_variants,
                quality=quality,
                force=force,
                chat_replay_source=chat_replay_source,
                started_at=started_at,
                config=config,
                llm_config=llm_config,
                settings=settings,
                deps=deps,
                db=db,
            )
    finally:
        structlog.contextvars.clear_contextvars()


@asynccontextmanager
async def _maybe_db_session(
    *, tenant_id: str, db_path: str | None
) -> AsyncIterator[Database | None]:
    """Yield an open Database (with tenant bound) or None when no path is given."""
    if db_path is None:
        yield None
    else:
        async with db_session(tenant_id=tenant_id, db_path=db_path) as db:
            yield db


def _handsfree_clip_scores(clip_entries: list) -> list[tuple[str, float]]:
    """(clip_id, fused candidate score) pairs for the hands-free sweep.

    The fused detector score lives on each clip's source ``Candidate``
    (``clip.candidate.score``). We read it straight off the clip — the
    detect ``Candidate`` has no ``.id`` and the domain ``Clip`` has no
    ``.candidate_id`` (that's only on the DB row), so the earlier
    candidates-list join crashed with ``'Candidate' object has no
    attribute 'id'`` and silently blocked all hands-free auto-publish.
    """
    return [
        (e.clip.id, float(e.clip.candidate.score or 0.0)) for e in clip_entries
    ]


async def _resolve_brand_kit_url(db: Database, *, stream_id: str) -> str | None:
    """Best-effort: the operator's saved handle/URL for this stream's brand
    kit, so auto-correct can fill an empty banner URL. Mirrors the dashboard
    /apply-ai-fixes resolution. Returns None on any miss — the banner fix
    just won't fire."""
    try:
        from chalybclip.branding import resolve_brand_kit_for_candidate

        kit = await resolve_brand_kit_for_candidate(
            db, stream_id=stream_id, speaker_label=None
        )
        if kit is None:
            return None
        for attr in (
            "handle_kick",
            "handle_tiktok",
            "handle_youtube",
            "handle_instagram",
        ):
            v = getattr(kit, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:  # noqa: BLE001 — brand-kit URL is an optional hint
        return None
    return None


_SERIES_MAX_GAP_S = 20.0
"""Two clips whose windows sit within this many seconds of each other are
one long moment the cutter split — they publish as "Parte 1/N, 2/N, …" so
a viewer who lands on any part goes hunting for the rest of the series."""

_SERIES_TAG_RE = re.compile(r"\s+—\s+Part(?:e)?\s+\d+/\d+$")


def _series_parts(clips: list[Any]) -> dict[str, tuple[int, int, int]]:
    """Map clip.id -> (part_no, total, run_key) for temporally contiguous
    runs of >= 2 clips. Interval-fallback clips are evenly spaced by
    construction and never chain into a fake series."""
    ordered = sorted(clips, key=lambda c: float(getattr(c, "start_s", 0.0)))
    runs: list[list[Any]] = []
    for c in ordered:
        reason = str(getattr(c.candidate, "reason", "") or "")
        chainable = reason != "interval"
        if (
            chainable
            and runs
            and runs[-1]
            and str(getattr(runs[-1][-1].candidate, "reason", "") or "") != "interval"
            and float(c.start_s) <= float(runs[-1][-1].end_s) + _SERIES_MAX_GAP_S
        ):
            runs[-1].append(c)
        else:
            runs.append([c])
    parts: dict[str, tuple[int, int, int]] = {}
    for run_key, run in enumerate(runs):
        if len(run) >= 2:
            for i, c in enumerate(run, start=1):
                parts[c.id] = (i, len(run), run_key)
    return parts


def _series_hook(base: str, part: int, total: int, *, language: str) -> str:
    """`base — Parte {part}/{total}`, trimmed so the whole line stays
    within the 90-char burn budget."""
    word = "Parte" if (language or "es").startswith("es") else "Part"
    tag = f" — {word} {part}/{total}"
    room = 90 - len(tag)
    b = _SERIES_TAG_RE.sub("", base.strip())
    if len(b) > room:
        b = b[: max(0, room - 1)].rstrip() + "…"
    return b + tag


def _norm_hook(hook: str) -> str:
    return " ".join(hook.split()).casefold()


async def _auto_hook_for_clip(
    *,
    clip: Any,
    persona: Persona,
    language: str | None,
    tenant_id: str,
    router: Any,
    stream_title: str = "",
    avoid_hooks: tuple[str, ...] = (),
) -> str:
    """Generate ONE viral hook (title line) from the clip's transcript
    snippet. Best-effort — returns '' on any failure so a hook hiccup never
    breaks the pipeline. The snippet already lives on the clip's candidate
    evidence (set by detect), so no extra transcript work is needed.

    `stream_title` is passed as context so a no-speech clip (sports goal,
    reaction) still gets a real hook off the stream title instead of
    meta-commentary about a missing transcript.

    When the LLM path fails (billing cap, provider outage) the clip still
    gets a title via the deterministic generator — a hookless clip cascades
    into a degenerate one-word caption downstream, which is worse than a
    heuristic title."""
    from chalybclip.variants.deterministic import deterministic_hook
    from chalybclip.variants.hooks import generate_hooks

    evidence = getattr(clip.candidate, "evidence", None) or {}
    snippet = (
        str(evidence.get("transcript_snippet") or "")
        if isinstance(evidence, dict)
        else ""
    )
    context_bits: list[str] = []
    if (stream_title or "").strip():
        context_bits.append(f"Stream title: {stream_title.strip()}")
    reason = str(getattr(clip.candidate, "reason", "") or "").strip()
    if reason:
        context_bits.append(f"Why it was clipped: {reason}")
    hook = ""
    try:
        hooks = await generate_hooks(
            tenant_id=tenant_id,
            persona_voice=persona.voice_prompt,
            persona_language=language or persona.primary_language or "es",
            transcript_snippet=snippet,
            clip_context=" — ".join(context_bits),
            n=1,
            avoid_hooks=avoid_hooks,
            router=router,
        )
        hook = hooks[0].text.strip() if hooks else ""
    except Exception as e:
        _log.warning("pipeline.auto_hook_failed", clip_id=clip.id, error=str(e))
    # Cross-clip uniqueness is a hard rule: if the model converged on a
    # line a sibling clip already ships, discard it — the deterministic
    # generator below honors the avoid-set structurally.
    if hook and _norm_hook(hook) in {_norm_hook(h) for h in avoid_hooks}:
        _log.info("pipeline.auto_hook_duplicate_discarded", clip_id=clip.id)
        hook = ""
    if hook:
        return hook
    hook = deterministic_hook(
        transcript_snippet=snippet,
        stream_title=stream_title,
        language=language or persona.primary_language or "es",
        seed=str(clip.id),
        reason=reason,
        avoid=set(avoid_hooks),
    )
    _log.info("pipeline.auto_hook_deterministic_fallback", clip_id=clip.id)
    return hook


async def _run_pipeline(
    *,
    tenant_id: str,
    vod_url: str,
    output_dir: Path,
    persona: Persona,
    stream_id: str | None,
    language: str | None,
    n_variants: int,
    quality: Quality | None,
    force: bool,
    chat_replay_source: Path | None,
    started_at: str,
    config: ChalybClipConfig,
    llm_config: LLMConfig,
    settings: Settings,
    deps: PipelineDeps,
    db: Database | None,
) -> StreamManifest:
    # Bind stream_id BEFORE the ingest step so its events carry stream_id
    # in the payload and the dashboard's progress card can find them. (For
    # the URL path where stream_id is None up-front, we re-bind below
    # using the id ingest_vod minted.)
    if stream_id:
        structlog.contextvars.bind_contextvars(stream_id=stream_id)

    # 1) ingest
    with _step("ingest", db=db, vod_url=vod_url):
        stream = await ingest_vod(
            tenant_id=tenant_id,
            vod_url=vod_url,
            output_dir=output_dir,
            stream_id=stream_id,
            force=force,
            chat_replay_source=chat_replay_source,
            # Task 2a — wire the DB through so ingest emits
            # stream.download.started / .completed / audio_extracted
            # events the dashboard can break into substeps.
            db=db,
        )
        if db is not None:
            await StreamsRepo(db).upsert(stream_to_row(stream))
            # upsert is INSERT-OR-IGNORE: if this stream was pre-inserted as
            # a placeholder (URL-job flow → pending/0s/no-title), the upsert
            # above is a no-op. Backfill the real title/channel/duration and
            # flip pending→ingested so a successful run stops displaying as
            # 'pending / 0s' forever.
            await StreamsRepo(db).reconcile_metadata(
                stream.id,
                title=stream.title,
                channel=stream.channel,
                duration_s=stream.duration_s,
            )
            await emit(
                db,
                STREAM_CREATED,
                {
                    "stream_id": stream.id,
                    "vod_url": stream.vod_url,
                    "platform": stream.platform,
                    "duration_s": stream.duration_s,
                },
            )

    structlog.contextvars.bind_contextvars(stream_id=stream.id)
    stream_dir = output_dir / stream.id
    call_log_path = stream_dir / "llm_calls.jsonl"

    # Slice O.12 — adaptive timeouts. Long-form VODs need proportionally
    # more time for transcribe + analyze_video; a fixed 30-min whisper
    # cap that worked for 5-min spikes fails the first time someone
    # uploads a 3-hour stream. Each step's ceiling is now
    #   max(static_floor, duration_s * multiplier)
    # with admin tenants optionally bypassing the cap entirely. The
    # `_is_admin_tenant` lookup is best-effort + cached at process
    # boot via lru_cache so it doesn't hit the db per step.
    def _is_admin_tenant(tid: str) -> bool:
        raw = (settings.admin_tenant_ids or "").strip()
        if not raw:
            return False
        ids = {p.strip() for p in raw.split(",") if p.strip()}
        return tid in ids

    _admin_uncapped = settings.admin_uncapped_pipeline and _is_admin_tenant(tenant_id)
    _whisper_timeout = (
        None if _admin_uncapped
        else max(
            settings.whisper_timeout_s,
            stream.duration_s * settings.whisper_timeout_multiplier,
        )
    )
    _analyze_video_timeout = (
        None if _admin_uncapped
        else max(
            config.detection.visual.timeout_s,
            stream.duration_s * settings.analyze_video_timeout_multiplier,
        )
    )
    _log.info(
        "pipeline.timeouts_resolved",
        stream_id=stream.id,
        duration_s=stream.duration_s,
        whisper_timeout=_whisper_timeout,
        analyze_timeout=_analyze_video_timeout,
        admin_uncapped=_admin_uncapped,
    )

    # 2a) analyze video — local CV pipeline. Skip silently if:
    #   (a) the visual signals it produces aren't consumed by anything
    #       (visual detector disabled + no vision_rescore in the default
    #       pipeline) — pure wasted CPU per stream, and PySceneDetect on
    #       a multi-hour VOD on CPU can take longer than every other
    #       step combined.
    #   (b) the video file isn't decodable (test stubs, corrupted
    #       downloads — DetectionError) — log and continue.
    #   (c) the OpenCV work blows past the configured timeout — see
    #       VisualConfig.timeout_s. The pipeline moves on with no visual
    #       signals; downstream merges happen with whatever voice / audio
    #       / viral candidates were already found.
    with _step("analyze_video", db=db) as step_ctx:
        if not config.detection.visual.enabled:
            step_ctx.skipped = True
            step_ctx.note = "visual detector disabled in config"
            _log.info(
                "analyze_video.skipped",
                reason="visual detector disabled; nothing downstream consumes the output",
            )
        else:
            # Slice O.12 — duration-scaled timeout. `_analyze_video_timeout`
            # is None when admin bypass is on; we run unbounded in that
            # case (admin only — paying users always get a cap).
            timeout_s = _analyze_video_timeout
            try:
                if timeout_s is None:
                    await _analyze_video(
                        tenant_id=tenant_id,
                        stream=stream,
                        output_dir=output_dir,
                        db=db,
                        force=force,
                    )
                else:
                    await asyncio.wait_for(
                        _analyze_video(
                            tenant_id=tenant_id,
                            stream=stream,
                            output_dir=output_dir,
                            db=db,
                            force=force,
                        ),
                        timeout=timeout_s,
                    )
            except DetectionError as e:
                step_ctx.skipped = True
                step_ctx.note = f"video not decodable: {e}"
                _log.warning("analyze_video.skipped", reason=str(e))
            except asyncio.TimeoutError:
                step_ctx.skipped = True
                step_ctx.note = f"exceeded {timeout_s:.0f}s timeout"
                _log.warning(
                    "analyze_video.timeout",
                    timeout_s=timeout_s,
                    reason=(
                        f"PySceneDetect exceeded {timeout_s:.0f}s on this VOD. "
                        f"Continuing without visual signals. Raise "
                        f"detection.visual.timeout_s in chalybclip.yaml or set "
                        f"detection.visual.enabled=false to skip the step entirely."
                    ),
                )

    # 2a-bis) diarize — pyannote-3.1 speaker turns + per-speaker embeddings.
    # Skippable: returns Diarization(skipped=True) when disabled / HF_TOKEN
    # missing / pyannote not installed / worker crashes. Downstream code
    # reads candidate.evidence.get('speaker_label') defensively, so an
    # empty diarization just means no speaker labels on this VOD.
    #
    # Task A2 — when `detection.diarization.source="transcribe"` the
    # operator has opted into using the transcribe provider's
    # per-utterance speaker labels (AssemblyAI's speaker_labels=true)
    # instead of the GPU-bound pyannote pass. We mark the diarize
    # step as deferred here; after the transcribe step finishes we
    # materialize a Diarization from `transcript.segments` and
    # overwrite the placeholder. Cross-video identity (resolve_speakers)
    # is skipped in this mode — no embeddings to match against the
    # persistent speakers table.
    diarization: Diarization = Diarization(
        stream_id=stream.id, tenant_id=tenant_id, skipped=True
    )
    use_transcribe_speakers = (
        str(getattr(config.detection.diarization, "source", "pyannote")).strip().lower()
        == "transcribe"
    )
    with _step("diarize", db=db, model=config.detection.diarization.model) as step_ctx:
        if use_transcribe_speakers:
            # Deferred. The transcribe step below will produce the
            # speaker labels; we patch them in afterward. Mark the
            # step as skipped-with-reason so the dashboard's progress
            # surface explains why the diarize bar didn't fill.
            diarization = Diarization(
                stream_id=stream.id, tenant_id=tenant_id,
                skipped=True,
                skip_reason="deferred to transcribe step (AssemblyAI speaker labels)",
            )
            step_ctx.skipped = True
            step_ctx.note = "deferred to transcribe"
            _log.info(
                "diarize.deferred_to_transcribe",
                stream_id=stream.id,
            )
        else:
            diarization = await diarize(
                tenant_id=tenant_id,
                stream=stream,
                config=config.detection.diarization,
                force=force,
            )
            if diarization.skipped:
                step_ctx.skipped = True
                step_ctx.note = diarization.skip_reason or "diarization disabled"
                _log.info(
                    "diarize.skipped",
                    reason=diarization.skip_reason,
                    stream_id=stream.id,
                )
            else:
                _log.info(
                    "diarize.done",
                    segments=len(diarization.segments),
                    speakers=len({s.speaker_label for s in diarization.segments}),
                    embeddings=len(diarization.embeddings),
                    stream_id=stream.id,
                )
                # Resolve diarization output against the tenant's persistent
                # speakers table. Auto-create pending rows for unknown voices;
                # fold matches' embeddings into the existing identity. Safe
                # no-op when db=None (filesystem-only / test mode).
                if db is not None:
                    outcome = await resolve_speakers(
                        db=db,
                        stream_id=stream.id,
                        diarization=diarization,
                        config=config.detection.diarization,
                    )
                    _log.info(
                        "speakers.resolved",
                        stream_id=stream.id,
                        matched=outcome.matched,
                        created=outcome.created,
                        unresolved=outcome.unresolved,
                    )

    # 2) transcribe
    # Wrapped in asyncio.wait_for: faster-whisper occasionally stalls on
    # CUDA without raising — silent hang inside the worker thread. The
    # timeout (CHALYBCLIP_WHISPER_TIMEOUT_S, default 30 min) raises a
    # TranscriptionError so the dashboard's progress card flips to
    # "Pipeline failed" with an actionable message instead of staring at
    # a pulsing dot forever.
    #
    # Slice F.7-G — language is now AUTO-DETECTED by Whisper when the
    # caller didn't pin one. The previous hardcoded "es" fallback meant
    # English / Portuguese / silent test clips got transcribed AS IF
    # they were Spanish, producing garbage segments that VAD then
    # filtered to empty → "no captions" on the editor preview. Passing
    # `language=None` lets faster-whisper run its 30-sample language
    # detector and pick the right tokenizer.
    whisper_lang: str | None = language if (language and language != "auto") else None
    with _step("transcribe", db=db, model=settings.whisper_model, device=settings.whisper_device):
        try:
            # Slice O.12 — duration-scaled whisper timeout. Same admin
            # bypass as analyze_video — `_whisper_timeout=None` skips
            # the asyncio.wait_for wrapper entirely.
            if _whisper_timeout is None:
                transcript = await transcribe(
                    tenant_id=tenant_id,
                    stream=stream,
                    model_size=settings.whisper_model,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                    language=whisper_lang,
                    force=force,
                    # Task A1b — db wires cost metering through. The
                    # service consults BudgetGovernor pre-call and
                    # records an LLMCallRow post-call when the
                    # provider declares a marginal cost rate
                    # (AssemblyAI yes, LocalWhisperProvider no).
                    db=db,
                )
            else:
                transcript = await asyncio.wait_for(
                    transcribe(
                        tenant_id=tenant_id,
                        stream=stream,
                        model_size=settings.whisper_model,
                        device=settings.whisper_device,
                        compute_type=settings.whisper_compute_type,
                        language=whisper_lang,
                        force=force,
                        db=db,
                    ),
                    timeout=_whisper_timeout,
                )
        except asyncio.TimeoutError as e:
            from chalybclip.errors import TranscriptionError

            raise TranscriptionError(
                f"Whisper transcribe exceeded {_whisper_timeout:.0f}s "
                f"(scaled from {stream.duration_s:.0f}s VOD × "
                f"{settings.whisper_timeout_multiplier}× multiplier) "
                f"and was abandoned. Common causes on Windows: CUDA OOM (run "
                f"`nvidia-smi` — if VRAM is full, restart the dashboard); a "
                f"corrupted audio extract; or a stuck CTranslate2 worker. "
                f"Fix: restart `python run.py`, click 'Run pipeline' on the "
                f"stream again. If it hangs twice in a row, set "
                f"CHALYBCLIP_WHISPER_MODEL=base in .env (smaller model, faster, "
                f"less VRAM), or bump CHALYBCLIP_WHISPER_TIMEOUT_MULTIPLIER."
            ) from e
        if db is not None:
            await TranscriptsRepo(db).upsert(transcript_to_row(transcript))

    # Task A2 — materialize the diarization from the transcript's
    # speaker labels now that transcribe is done. Skipped when:
    #   * source != "transcribe" (pyannote already ran above), or
    #   * transcript carries no speaker labels (provider didn't emit
    #     them — adapter returns Diarization(skipped=True) with a
    #     clear skip_reason; downstream code already handles that).
    if use_transcribe_speakers:
        diarization = diarization_from_transcript(
            transcript, tenant_id=tenant_id, stream_id=stream.id,
        )
        _log.info(
            "diarize.from_transcript",
            stream_id=stream.id,
            speakers=len({s.speaker_label for s in diarization.segments}),
            segments=len(diarization.segments),
            skipped=diarization.skipped,
        )

    # 3) detect (also saves candidates.json). Build the router up-front so the
    # viral detector (LLM-based, runs inside this step when enabled) can use
    # it; the variants step at the bottom reuses the same router.
    router = (
        deps.router_factory(stream_dir)
        if deps.router_factory
        else LLMRouter(config=llm_config, call_log_path=call_log_path, db=db)
    )
    with _step("detect", db=db):
        chat_replay = load_chat_replay(stream_dir, stream_id=stream.id, tenant_id=tenant_id)
        visual_track = load_visual_signals(stream_dir)
        viral_cands: list[Candidate] = []
        if config.detection.viral.enabled:
            try:
                viral_cands = await detect_viral_moments(
                    tenant_id=tenant_id,
                    stream=stream,
                    transcript=transcript,
                    router=router,
                    config=config.detection.viral,
                )
            except Exception as e:  # noqa: BLE001 - belt-and-suspenders
                # detect_viral_moments already swallows expected errors;
                # this catches anything we missed so the pipeline keeps moving.
                _log.warning("viral.skipped", reason=str(e), stream_id=stream.id)

        # Build the per-speaker extra-phrases map for this VOD (slice C.2).
        # For each within-VOD speaker with a resolved persistent identity,
        # look up their brand kit's custom_trigger_phrases. Empty when
        # diarization was skipped, no kits exist, or no speakers have a
        # preferred kit — detect_candidates falls back to the tenant base.
        extra_phrases_by_speaker: dict[str, object] = {}
        if db is not None and not diarization.skipped:
            from chalybclip.branding import resolve_brand_kit_for_speaker

            for vs in await VodSpeakersRepo(db).list_for_stream(stream.id):
                kit = await resolve_brand_kit_for_speaker(
                    db, speaker_id=vs.resolved_speaker_id
                )
                if kit is None:
                    continue
                if kit.custom_trigger_phrases.forward or kit.custom_trigger_phrases.retroactive:
                    extra_phrases_by_speaker[vs.speaker_label] = (
                        kit.custom_trigger_phrases
                    )
            if extra_phrases_by_speaker:
                _log.info(
                    "detect.per_kit_phrases",
                    stream_id=stream.id,
                    speakers_with_extras=len(extra_phrases_by_speaker),
                )

        # Slice O.39 — solo-streamer fallback. When diarization was
        # skipped (no HF_TOKEN, pyannote not installed, CPU-only host
        # where it's too slow, etc.), the per-speaker map above is empty
        # by construction. The tenant default brand kit's custom phrases
        # would never fire, even though the operator clearly intended
        # them to. Apply them tenant-wide instead — no speaker label
        # means no restriction, so they hit anyone's words.
        extra_phrases_tenant_wide: object | None = None
        if db is not None and diarization.skipped:
            try:
                from chalybclip.db import BrandKitsRepo

                default_kit = await BrandKitsRepo(db).get_default()
                if default_kit is not None and (
                    default_kit.custom_trigger_phrases.forward
                    or default_kit.custom_trigger_phrases.retroactive
                ):
                    extra_phrases_tenant_wide = default_kit.custom_trigger_phrases
                    _log.info(
                        "detect.tenant_wide_kit_phrases",
                        stream_id=stream.id,
                        forward=len(default_kit.custom_trigger_phrases.forward),
                        retroactive=len(default_kit.custom_trigger_phrases.retroactive),
                    )
            except Exception as e:  # noqa: BLE001 — best-effort, never block detect
                _log.warning(
                    "detect.tenant_wide_kit_phrases_failed",
                    stream_id=stream.id,
                    reason=str(e),
                )

        candidates = detect_candidates(
            tenant_id=tenant_id,
            stream=stream,
            transcript=transcript,
            config=config.detection,
            chat_replay=chat_replay,
            visual_track=visual_track,
            viral_candidates=viral_cands,
            diarization=diarization,
            extra_phrases_by_speaker=extra_phrases_by_speaker or None,
            extra_phrases_tenant_wide=extra_phrases_tenant_wide,
        )
        # Always-give-clips fallback: a silent, static clip (no speech, no
        # audio peaks, no visual motion — e.g. some gameplay / ambient
        # streams) fires zero detectors. Rather than ship an empty result,
        # place evenly-spaced interval anchors so the user still gets clips
        # to review. Real signal-based candidates always rank above these.
        if not candidates:
            candidates = fallback_interval_candidates(
                tenant_id=tenant_id, stream=stream,
            )
            _log.info(
                "detect.fallback_interval",
                stream_id=stream.id,
                count=len(candidates),
            )
        save_candidates(
            stream_dir,
            CandidateBatch(stream_id=stream.id, tenant_id=tenant_id, candidates=candidates),
        )
        if db is not None:
            candidate_rows = [
                candidate_to_row(c, stream_id=stream.id, tenant_id=tenant_id) for c in candidates
            ]
            await CandidatesRepo(db).upsert_many(candidate_rows)
    _log.info("detect.candidates", count=len(candidates))

    # 4) cut clips — slice D.1: pre-resolve a brand kit per candidate so
    # the renderer can burn the right handle / colors into each clip.
    # Speaker-attributed candidates resolve via vod_speakers → speaker.
    # preferred_brand_kit_id → kit; un-attributed candidates fall back to
    # the tenant default kit; tenants without any kits get None and the
    # renderer skips the overlay.
    candidate_kits: list[object] | None = None
    if db is not None:
        from chalybclip.branding import resolve_brand_kit_for_candidate

        resolved: list[object] = []
        for c in candidates:
            ev = c.evidence or {}
            label = ev.get("speaker_label")
            label_str = label if isinstance(label, str) else None
            kit = await resolve_brand_kit_for_candidate(
                db, stream_id=stream.id, speaker_label=label_str
            )
            resolved.append(kit if kit is not None else None)
        candidate_kits = resolved
        used = sum(1 for k in resolved if k is not None)
        _log.info(
            "cut.brand_kits_resolved",
            stream_id=stream.id,
            with_kit=used,
            without_kit=len(resolved) - used,
        )

    with _step("cut", db=db, candidate_count=len(candidates)):
        # Slice O.33 — adaptive hard timeout. cut step previously had
        # no ceiling; a single hung ffmpeg subprocess (the smart-crop
        # OpenCV pass occasionally stalls on certain h264 profiles)
        # would freeze the whole pipeline indefinitely. The user saw
        # a 87s video stuck at "cut (running...) 22.9m elapsed".
        # Formula matches whisper/analyze_video: max(static_floor,
        # duration_s * multiplier). 6x realtime on CPU for ~3-4
        # concurrent clip cuts at preset=veryfast covers normal load;
        # the floor protects very short clips.
        _cut_timeout = (
            None if _admin_uncapped
            else max(300.0, stream.duration_s * 6.0)
        )
        # No-speech path: an empty transcript has no sentence boundaries to
        # snap to, so drop it and let cut use static pre/post-roll windows
        # around each candidate anchor (the dynamic windower needs words).
        cut_transcript = transcript if transcript.segments else None
        try:
            if _cut_timeout is None:
                clips = await cut_clips(
                    tenant_id=tenant_id,
                    stream=stream,
                    candidates=candidates,
                    output_dir=output_dir,
                    config=config.clip,
                    force=force,
                    brand_kits=candidate_kits,
                    transcript=cut_transcript,
                    # Task 1d — wire the DB through so cut_clips can
                    # emit clip.cut.started / .substep / .completed
                    # events into the existing event log. The dashboard
                    # reads these to drive the per-clip substep label
                    # + X/N progress.
                    db=db,
                )
            else:
                clips = await asyncio.wait_for(
                    cut_clips(
                        tenant_id=tenant_id,
                        stream=stream,
                        candidates=candidates,
                        output_dir=output_dir,
                        config=config.clip,
                        force=force,
                        brand_kits=candidate_kits,
                        # Slice G.1 — dynamic per-candidate windowing snaps each
                        # clip's start/end to transcript sentence boundaries.
                        # (None on a silent stream → static windows.)
                        transcript=cut_transcript,
                        db=db,
                    ),
                    timeout=_cut_timeout,
                )
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"cut step exceeded {_cut_timeout:.0f}s "
                f"(stream duration {stream.duration_s:.0f}s × 6x multiplier). "
                f"Common causes on CPU: a single ffmpeg subprocess stalled "
                f"(SIGTERM didn't propagate), smart_crop OpenCV pass stuck on "
                f"a broken h264 frame, or disk space pressure. "
                f"Restart `python run.py` and re-run; if it stalls a second "
                f"time on the same clip, the source mp4 may be the issue."
            ) from e
        if db is not None:
            await ClipsRepo(db).upsert_many([clip_to_row(c) for c in clips])

    # 4b) auto-correct each clip — slice G.6. Apply the AI's findings NOW,
    # while the source VOD still exists: auto-trim around freeze/silence and
    # apply the safe overlay fixes, so the clip arrives publish-ready and the
    # dashboard only reports what was corrected (no "fix it" button). Runs
    # BEFORE the hands-free sweep so corrected/promoted clips get published.
    # Strictly best-effort: one clip's failure must not break the batch or
    # the pipeline. No-op when no DB (offline/test cut).
    if db is not None and clips:
        from chalybclip.clip import auto_correct_clip

        _source_platform = getattr(stream, "platform", None)
        _bk_url = await _resolve_brand_kit_url(db, stream_id=stream.id)
        with _step("auto_correct", db=db, clip_count=len(clips)):
            for clip in clips:
                try:
                    await auto_correct_clip(
                        db,
                        clip_id=clip.id,
                        source_platform=_source_platform,
                        brand_kit_url=_bk_url,
                    )
                except Exception as e:  # noqa: BLE001 — never break the batch
                    _log.warning(
                        "pipeline.auto_correct_failed",
                        clip_id=clip.id,
                        error=str(e),
                    )

    # 4c) offload clip artifacts to object storage — Phase 2a. Runs AFTER
    # auto-correct because the trim re-cuts clip.mp4 in place; uploading
    # earlier would persist the pre-trim bytes. Best-effort and quiet (no
    # pipeline step event — the dashboard's six-step card stays as-is).
    if clips:
        try:
            # build_artifact_store is inside the guard too — a malformed
            # endpoint URL raises from boto3.client, and a config typo
            # must not kill a run whose clips already exist.
            from chalybclip.integrations.storage import build_artifact_store

            _store = build_artifact_store(settings)
            if _store is not None:
                from chalybclip.clip import offload_clip_artifacts

                _uploaded = await offload_clip_artifacts(
                    _store, tenant_id=tenant_id, clips=clips, force=force,
                )
                _log.info(
                    "pipeline.artifacts_offloaded",
                    stream_id=stream.id,
                    clip_count=len(clips),
                    objects_uploaded=_uploaded,
                )
        except Exception as e:  # never break the pipeline on offload
            _log.warning(
                "pipeline.artifact_offload_failed",
                stream_id=stream.id,
                error=str(e),
            )

    # 5) variants per clip — reuse the router built above for detect+viral.
    if db is not None:
        # Persona must exist in DB so variants can FK to it.
        await PersonasRepo(db).upsert(
            persona_id=persona.id,
            name=persona.name,
            primary_language=persona.primary_language,
            target_languages=persona.target_languages,
            voice_prompt=persona.voice_prompt,
            routing_tags=persona.routing_tags,
        )
    clip_entries: list[ClipEntry] = []
    # Each clip gets exactly ONE variant row: a deterministic stub the publish
    # flow reads its caption / hook / hashtags from (the LLM caption variant
    # generator was removed). The stub's caption comes from the clip's
    # overlay_config when set, else the stream title / hook. No LLM calls here.
    # Auto-hook: generate a viral title for EVERY clip so the editor, the
    # render burn, and auto-publish all get a hook without a manual "Generar 5".
    # (Was gated on the raw candidate score, which is uniformly low for YouTube
    # VODs with no chat heat — so those clips got no hook and shipped plain.)
    auto_hook_enabled = bool(getattr(settings, "auto_hook_enabled", True))
    hook_language = language or persona.primary_language or "es"
    # Contiguous clips are one long moment → "Parte 1/N" series titles;
    # every hook in the batch must be unique across the whole stream.
    series_parts = _series_parts(clips)
    series_base: dict[int, str] = {}
    used_hooks: list[str] = []
    used_norms: set[str] = set()
    with _step("variants", db=db, clip_count=len(clips), n=n_variants):
        # Time-ordered so a series' Parte 1 resolves its base hook before
        # Parte 2 needs it (and parts publish in watch order downstream).
        for clip in sorted(clips, key=lambda c: float(getattr(c, "start_s", 0.0))):
            # Idempotent re-runs must not re-pay the auto-hook LLM call:
            # the stub (and its hook) persists next to the clip, and a
            # resume without --force reuses it verbatim.
            variants_path = clip.path.parent / "variants.json"
            cached = None if force else _load_stub_variants(variants_path)
            part = series_parts.get(clip.id)
            if cached is not None:
                variants = cached
                auto_hook = variants[0].title_card_text or ""
                # A partially cached series still needs its base for the
                # uncached parts — recover it from the tagged hook.
                if part and auto_hook and part[2] not in series_base:
                    series_base[part[2]] = _SERIES_TAG_RE.sub("", auto_hook)
            else:
                auto_hook = ""
                if auto_hook_enabled:
                    if part and part[2] in series_base:
                        auto_hook = _series_hook(
                            series_base[part[2]], part[0], part[1],
                            language=hook_language,
                        )
                    else:
                        base = await _auto_hook_for_clip(
                            clip=clip, persona=persona, language=language,
                            tenant_id=tenant_id, router=router,
                            stream_title=stream.title or "",
                            avoid_hooks=tuple(used_hooks),
                        )
                        if part:
                            series_base[part[2]] = base
                            auto_hook = _series_hook(
                                base, part[0], part[1], language=hook_language,
                            )
                        else:
                            auto_hook = base
                # One stub variant per clip, captioned from the overlay or a
                # sensible fallback. The dashboard editor + the publish flow both
                # read .caption; the operator can edit it via the overlay form
                # before shipping.
                stub_caption = ""
                overlay = getattr(clip, "overlay_config", None)
                if isinstance(overlay, dict):
                    stub_caption = (
                        overlay.get("title_text")
                        or overlay.get("top_hook", {}).get("text", "")
                        if isinstance(overlay.get("top_hook"), dict)
                        else overlay.get("title_text") or ""
                    ) or ""
                if not stub_caption:
                    # NEVER the persona name — a tenant whose persona was named
                    # "viral" shipped 18 posts captioned literally "viral". The
                    # stream title is real context; the hook is already the
                    # caption's first block, so it's the last resort here.
                    stub_caption = (stream.title or "").strip() or auto_hook
                variants = [
                    Variant(
                        id="v_stub",
                        language=language or persona.primary_language or "es",
                        caption=str(stub_caption)[:240],
                        title_card_text=auto_hook,
                        hashtags=[],
                    )
                ]
                variants_path.write_text(
                    json.dumps(
                        {"variants": [v.model_dump() for v in variants]},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            # Feed the uniqueness guard — cached hooks count too, so a
            # partial re-run can't hand a fresh clip an already-used line.
            if auto_hook and _norm_hook(auto_hook) not in used_norms:
                used_norms.add(_norm_hook(auto_hook))
                used_hooks.append(auto_hook)
            # Stamp the viral default overlay so the clip ships non-plain even
            # when it never passes through the editor: captions on, the hook as
            # the top headline, and a source-credit "marca" banner carrying the
            # original streamer's name styled for the source platform (Kick /
            # YouTube / Twitch). The editor + auto-publish both burn this in;
            # the operator can still override it. (build_post reads the variant
            # title for the published caption text.)
            if db is not None:
                try:
                    await ClipsRepo(db).set_overlay_config(
                        clip.id,
                        overlay_config=default_overlay_config(
                            platform=stream.platform,
                            channel=stream.channel,
                            hook=auto_hook,
                        ),
                    )
                except Exception as e:
                    _log.warning(
                        "pipeline.overlay_default_persist_failed",
                        clip_id=clip.id, error=str(e),
                    )
            clip_entries.append(ClipEntry(clip=clip, persona_id=persona.id, variants=variants))
            if db is not None:
                variant_rows = [
                    variant_to_row(
                        v,
                        clip_id=clip.id,
                        tenant_id=tenant_id,
                        persona_id=persona.id,
                    )
                    for v in variants
                ]
                await VariantsRepo(db).replace_for_clip_persona(clip.id, persona.id, variant_rows)
                await emit(
                    db,
                    CLIP_READY_FOR_REVIEW,
                    {
                        "clip_id": clip.id,
                        "stream_id": stream.id,
                        "persona_id": persona.id,
                        "variant_count": len(variants),
                    },
                )

    # Auto-publish "manos libres" (Fase 2): fire-and-forget post of every
    # clip above the tenant's score threshold. Best-effort — must NEVER fail
    # the pipeline. No-op unless auto-publish is enabled in hands_free mode.
    if db is not None:
        try:
            from chalybclip.api.routers.zernio import autopublish_hands_free_sweep
            from chalybclip.settings import get_settings as _get_settings

            _clip_scores = _handsfree_clip_scores(clip_entries)
            await autopublish_hands_free_sweep(
                db=db,
                tenant_id=tenant_id,
                base_url=(_get_settings().public_url or "").rstrip("/"),
                clip_scores=_clip_scores,
            )
        except Exception as e:  # never break the pipeline on a publish hiccup
            _log.warning("pipeline.autopublish_handsfree_failed", error=str(e))

    # 6) manifest
    manifest = StreamManifest(
        started_at=started_at,
        completed_at=deps.clock().isoformat(),
        tenant_id=tenant_id,
        persona_id=persona.id,
        persona_name=persona.name,
        language=language or persona.primary_language,
        n_variants_requested=n_variants,
        stream=stream,
        transcript=TranscriptSummary.from_transcript(transcript),
        candidates=candidates,
        clip_entries=clip_entries,
        llm_spend=_read_llm_spend(call_log_path),
    )
    manifest_path = stream_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    _log.info(
        "pipeline.done",
        clips=len(clip_entries),
        llm_calls=manifest.llm_spend.total_calls,
        cost_usd_micros=manifest.llm_spend.total_cost_usd_micros,
    )
    if db is not None:
        await emit(
            db,
            STREAM_PROCESSED,
            {
                "stream_id": stream.id,
                "clip_count": len(clip_entries),
                "llm_calls": manifest.llm_spend.total_calls,
                "cost_usd_micros": manifest.llm_spend.total_cost_usd_micros,
            },
        )

    # Reclaim the raw source VOD now that the pipeline has run to completion.
    # The source (downloaded video + extracted audio) is the heaviest thing
    # on disk and is only needed during processing — clips are cut, the
    # transcript is persisted, and auto-correct already ran (step 4b runs
    # "while the source VOD still exists"). Deleting it here is what keeps
    # the volume flat as processed-stream count grows.
    #
    # Reaching this point means every step succeeded; an earlier failure
    # raises and skips this, leaving the source for a re-run. Re-runs of a
    # completed stream serve cached transcript + clips before touching the
    # source, so they stay no-ops even with it gone (a `force=True` re-run
    # re-downloads). Gated on `db is not None` so filesystem-only/CLI runs
    # keep the source for inspection. Best-effort — never fail the pipeline.
    if db is not None and getattr(settings, "delete_source_on_completion", True):
        from chalybclip.retention import reclaim_stream_source

        try:
            freed = reclaim_stream_source(
                stream_dir=stream_dir,
                source_video_path=stream.source_video_path,
                source_audio_path=stream.source_audio_path,
            )
            _log.info(
                "pipeline.source_reclaimed",
                stream_id=stream.id,
                bytes_freed=freed,
            )
        except Exception as e:
            # Cleanup must never break a successful run.
            _log.warning(
                "pipeline.source_reclaim_failed",
                stream_id=stream.id,
                error=str(e),
            )

    return manifest


def _load_stub_variants(path: Path) -> list[Variant] | None:
    """Cache read for the variants step. Returns None (regenerate) on any
    missing/corrupt file — the step is best-effort re-runnable either way."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        variants = [Variant.model_validate(v) for v in data.get("variants", [])]
    except Exception:
        return None
    return variants or None


def _read_llm_spend(call_log_path: Path) -> LLMSpend:
    """Sum cost across every row in `llm_calls.jsonl` for this stream."""
    if not call_log_path.exists():
        return LLMSpend()
    total_calls = 0
    total_cost = 0
    for line in call_log_path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total_calls += 1
        total_cost += int(row.get("cost_usd_micros", 0) or 0)
    return LLMSpend(total_calls=total_calls, total_cost_usd_micros=total_cost)


def load_manifest(stream_dir: Path) -> StreamManifest:
    """Read the manifest back from disk."""
    path = Path(stream_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return StreamManifest.model_validate_json(path.read_text("utf-8"))

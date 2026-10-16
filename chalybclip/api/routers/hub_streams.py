"""Hub API — start, jobs (streams), uploads, progress, clips list.

Mounted under /api/hub/v1 (see hub_api.hub_router). Every route reuses the
dashboard's repos, admission (consumption contract), pipeline kickoff and
view helpers; where a dashboard handler keeps its logic inline, the hub
version mirrors it and says so ("mirrors dashboard.X"), so both stay in
step until the dashboard pages are retired.

Media (video, thumbnails, the approved-clips zip) is never in a response
body: responses carry signed paths from api/hub_media.py.
"""

from __future__ import annotations

import base64
import contextlib
import datetime as _dt
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

from chalybclip.api._pipeline import PipelineKickoff
from chalybclip.api.deps import get_db
from chalybclip.api.hub_api import HubError, from_admission, hub_router, hub_tenant
from chalybclip.api.hub_media import media_kind, media_path
from chalybclip.api.status_gate import require_active_tenant
from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    EventsRepo,
    LLMCallsRepo,
    PersonasRepo,
    StreamsRepo,
    ZernioPublishesRepo,
)
from chalybclip.integrations.chalyb.admission import AdmissionRefused
from chalybclip.jobs.usage import admit_job, cancel_admission, readmit_job
from chalybclip.settings import get_settings

router = hub_router(tags=["hub-streams"])
_log = structlog.get_logger("chalybclip.api.hub.streams")

ALL_CLIP_STATUSES = ["cut", "ready_for_review", "approved", "published", "rejected"]


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _output_dir() -> Path:
    return Path(get_settings().default_output_dir)


def _refused(e: AdmissionRefused) -> HubError:
    return from_admission(e)


# ── view helpers ─────────────────────────────────────────────────────────


def _clip_json(tenant_id: str, clip: Any, candidate: Any | None = None) -> dict[str, Any]:
    from chalybclip.api.routers.dashboard import _viral_score

    return {
        "id": clip.id,
        "stream_id": clip.stream_id,
        "status": clip.status,
        "start_s": clip.start_s,
        "end_s": clip.end_s,
        "duration_s": clip.duration_s,
        "created_at": clip.created_at,
        "viral_score": _viral_score(clip, candidate),
        "why": _why(candidate),
        "publishability_status": clip.publishability_status,
        "title": _clip_title(clip),
        "format": _clip_format(clip),
        "media_url": media_path("clip", clip.id, tenant_id=tenant_id),
        "download_url": media_path("clip", clip.id, tenant_id=tenant_id, params={"dl": "1"}),
        "thumb_url": (
            media_path("thumb", clip.id, tenant_id=tenant_id)
            if clip.thumbnail_frame_path
            else None
        ),
    }


def _why(candidate: Any | None) -> dict[str, Any] | None:
    """Why the AI picked the moment, as data (the hub words it): the
    detector reason, the phrase it heard, the speaker, and whether a second
    AI pass confirmed it — the same facts dashboard._humanize_candidate
    turns into English bullets."""
    if candidate is None:
        return None
    ev = getattr(candidate, "evidence", None) or {}
    phrase = ev.get("phrase") or ev.get("transcript_snippet") if isinstance(ev, dict) else None
    speaker = ev.get("speaker_label") if isinstance(ev, dict) else None
    reason = (getattr(candidate, "reason", "") or "").lower()
    return {
        "reason": reason if reason in ("voice", "audio", "chat", "visual", "viral") else "other",
        "phrase": phrase.strip()[:120] if isinstance(phrase, str) and phrase.strip() else None,
        "speaker": speaker.strip()[:60] if isinstance(speaker, str) and speaker.strip() else None,
        "confirmed": bool(isinstance(ev, dict) and ev.get("rescore")),
    }


_FORMATS = ("vertical", "horizontal", "square")


def _clip_format(clip: Any) -> str:
    """The output format the editor saved in overlay_config["format"]
    (same rule as the editor's clip.formats.clip_format): vertical when
    unset or unknown."""
    cfg = clip.overlay_config if isinstance(clip.overlay_config, dict) else {}
    fmt = cfg.get("format")
    return fmt if isinstance(fmt, str) and fmt in _FORMATS else "vertical"


def _clip_title(clip: Any) -> str | None:
    """The hook the operator wrote in the editor, if any."""
    cfg = clip.overlay_config or {}
    for key in ("title", "hook", "title_hook"):
        v = cfg.get(key) if isinstance(cfg, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()[:200]
    return None


def _stream_json(stream: Any, *, active: bool, clip_count: int, ready_count: int) -> dict[str, Any]:
    return {
        "id": stream.id,
        "title": stream.title,
        "channel": stream.channel,
        "platform": stream.platform,
        "vod_url": stream.vod_url if not stream.vod_url.startswith("upload://") else None,
        "status": stream.status,
        "active": active,
        "duration_s": stream.duration_s,
        "created_at": stream.created_at,
        "is_live": stream.is_live,
        "clip_count": clip_count,
        "ready_count": ready_count,
    }


async def _personas_json(db: Database) -> list[dict[str, Any]]:
    from chalybclip.api.routers.dashboard import _merged_personas

    return [
        {
            "id": p.id,
            "name": getattr(p, "name", p.id),
            "primary_language": getattr(p, "primary_language", None),
        }
        for p in await _merged_personas(db)
    ]


def _failure_kind(error: str) -> str:
    """Bucket a top-level failure for creator-facing copy (the raw error is
    operator text — shown to engine admins only)."""
    low = error.lower()
    if "unknown persona" in low:
        return "persona"
    if "ingest" in low or "yt-dlp" in low or "private" in low or "unavailable" in low:
        return "source"
    if "too long" in low or "duration" in low:
        return "too_long"
    return "other"


STEP_ORDER = ["ingest", "analyze_video", "diarize", "transcribe", "detect", "cut"]
_ABANDONED_THRESHOLD_S = 60 * 60


async def stream_progress_view(
    db: Database, stream: Any, *, is_admin: bool
) -> dict[str, Any]:
    """Pipeline progress for one stream.

    Mirrors dashboard.stream_progress (step roll-up, elapsed time, the
    abandoned-run detector that trusts jobs.active, the top-level
    pipeline.failed surfacing) including the fixes in ChalyClip#88: a
    failure from a run BEFORE the latest Córrelo/recovery doesn't count.
    """
    from chalybclip.api.routers.dashboard import _rerun_available
    from chalybclip.jobs.active import active_stream_ids

    stream_id = stream.id
    candidates = await CandidatesRepo(db).list_for_stream(stream_id)
    clips = await ClipsRepo(db).list_for_stream(stream_id)
    all_events = [
        e
        for e in await EventsRepo(db).list_for_tenant(limit=2000)
        if e.payload.get("stream_id") == stream_id
    ]
    restarts = [
        e.ts
        for e in all_events
        if e.type in ("stream.rerun_requested", "stream.recovery_dispatched")
    ]
    last_restart = max(restarts) if restarts else None

    state: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            "status": "pending",
            "duration_s": None,
            "elapsed_s": None,
            "skipped": False,
            "note": "",
            "error_kind": None,
        }
        for name in STEP_ORDER
    }
    started: dict[str, str] = {}
    for ev in sorted(all_events, key=lambda e: e.ts):
        if not ev.type.startswith("pipeline.step."):
            continue
        name = str(ev.payload.get("step", ""))
        if name not in state:
            continue
        s = state[name]
        if ev.type == "pipeline.step.start":
            s["status"] = "running"
            started[name] = ev.ts
        elif ev.type == "pipeline.step.done":
            s["status"] = "done"
            s["duration_s"] = ev.payload.get("duration_s")
            s["skipped"] = bool(ev.payload.get("skipped"))
            s["note"] = str(ev.payload.get("note") or "")[:200]
        elif ev.type == "pipeline.step.failed":
            s["status"] = "failed"
            s["duration_s"] = ev.payload.get("duration_s")
            err = str(ev.payload.get("error") or "")
            s["error_kind"] = _failure_kind(err)
            if is_admin:
                s["error"] = err[:500]

    run_in_flight = stream_id in active_stream_ids()
    now = _dt.datetime.now(_dt.UTC)
    for name, s in state.items():
        if s["status"] == "running" and name in started:
            try:
                elapsed = (now - _dt.datetime.fromisoformat(started[name])).total_seconds()
            except ValueError:
                continue
            s["elapsed_s"] = elapsed
            if elapsed > _ABANDONED_THRESHOLD_S and not run_in_flight:
                s["status"] = "abandoned"

    steps = [state[n] for n in STEP_ORDER]

    failures = [
        e
        for e in all_events
        if e.type == "pipeline.failed" and (last_restart is None or e.ts >= last_restart)
    ]
    failure: dict[str, Any] | None = None
    if failures:
        latest = max(failures, key=lambda e: e.ts)
        err = str(latest.payload.get("error") or "pipeline aborted")
        failure = {"kind": _failure_kind(err)}
        if is_admin:
            failure["error"] = err[:500]
            failure["error_type"] = str(latest.payload.get("error_type") or "Exception")
        # Point at the first step that never ran — unless a step already
        # reports its own failure (the dashboard marks both; one red row
        # reads clearer).
        already = any(s["status"] == "failed" for s in steps)
        for s in [] if already else steps:
            if s["status"] == "pending":
                s["status"] = "failed"
                s["error_kind"] = failure["kind"]
                break

    is_running = any(s["status"] == "running" for s in steps) or (
        all(s["status"] == "pending" for s in steps) and failure is None
    )
    is_done = all(s["status"] == "done" for s in steps)
    has_failed = any(s["status"] == "failed" for s in steps)
    is_abandoned = any(s["status"] == "abandoned" for s in steps)
    can_rerun = (has_failed or is_abandoned) and _rerun_available(stream, _output_dir())

    return {
        "steps": steps,
        "is_running": is_running and not (is_done or has_failed or is_abandoned),
        "is_done": is_done,
        "has_failed": has_failed,
        "is_abandoned": is_abandoned,
        "failure": failure,
        "in_flight": run_in_flight,
        "candidate_count": len(candidates),
        "clip_count": len(clips),
        "can_rerun": bool(can_rerun),
    }


async def _run_cost_view(
    db: Database, request: Request, stream: Any, tenant_id: str
) -> dict[str, Any]:
    """Run cost in TOKENS (never USD for creators). Mirrors the token math
    in dashboard.stream_detail (Token T3/T5: per-provider floor 1, metered
    providers cost→tokens at MICROS_PER_TOKEN=4, plus the base charge)."""
    from math import ceil

    from chalybclip.llm.estimator import estimate_pipeline_run, estimate_runs_per_balance

    actual = None
    with contextlib.suppress(Exception):
        actual = await LLMCallsRepo(db).cost_for_stream(stream.id)

    micros_per_token = 4  # MUST match Chalyb getTokenBalance's MICROS_PER_TOKEN
    base_micros = int(getattr(get_settings(), "pipeline_base_charge_usd_micros", 0) or 0)
    base_tokens = ceil(base_micros / micros_per_token) if base_micros > 0 else 0

    used: int | None = None
    by_source: list[dict[str, Any]] = []
    if actual is not None and actual.total_calls:
        total = 0
        for p in actual.by_provider:
            t = p.tokens if p.tokens > 0 else ceil(p.cost_usd_micros / micros_per_token)
            t = max(1, int(t))
            by_source.append({"label": p.provider, "tokens": t})
            total += t
        if base_tokens > 0:
            by_source.append({"label": "base", "tokens": base_tokens})
            total += base_tokens
        used = total

    estimate = estimate_pipeline_run(duration_seconds=stream.duration_s, persona_count=1)
    est_total = (
        int(estimate.total_tokens)
        + (
            ceil(estimate.transcription_cost_usd_micros / micros_per_token)
            if estimate.transcription_cost_usd_micros > 0
            else 0
        )
        + base_tokens
    )
    balance = getattr(request.state, "token_balance", None) or {}
    unlimited = bool(balance.get("unlimited")) if isinstance(balance, dict) else False
    remaining = balance.get("remaining") if isinstance(balance, dict) and not unlimited else None
    runs = estimate_runs_per_balance(
        remaining_tokens=remaining, duration_seconds=stream.duration_s, persona_count=1
    )
    out: dict[str, Any] = {
        "tokens_used": used,
        "tokens_by_source": by_source,
        "estimate_tokens": est_total,
        "runs_available": runs,
        "unlimited": unlimited,
    }
    if getattr(request.state, "is_admin", False) and actual is not None:
        out["usd_micros"] = int(actual.total_cost_usd_micros)
    return out


# ── hub "settled" bookkeeping ────────────────────────────────────────────

# A finished run older than this is never announced by the hub (no "your
# clips are ready" notice for history it finds on first visit after cutover).
_SETTLE_WINDOW_S = 24 * 3600


def _settled_map(events: list[Any], streams: list[Any]) -> dict[str, bool]:
    """stream_id → whether the hub already handled its latest run: a
    `hub.settled` after the last Córrelo/recovery, or a latest run that
    started more than _SETTLE_WINDOW_S ago."""
    last_settle: dict[str, str] = {}
    last_start: dict[str, str] = {}
    for e in events:
        sid = e.payload.get("stream_id")
        if not isinstance(sid, str):
            continue
        if e.type == "hub.settled":
            last_settle[sid] = max(last_settle.get(sid, ""), e.ts)
        elif e.type in ("stream.rerun_requested", "stream.recovery_dispatched"):
            last_start[sid] = max(last_start.get(sid, ""), e.ts)
    cutoff = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(seconds=_SETTLE_WINDOW_S)).isoformat()
    out: dict[str, bool] = {}
    for s in streams:
        started = max(last_start.get(s.id, ""), s.created_at or "")
        out[s.id] = last_settle.get(s.id, "") >= started or started < cutoff
    return out


# ── start ───────────────────────────────────────────────────────────────


@router.get("/start")
async def hub_start(
    request: Request,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Everything the hub's start screen needs (mirrors dashboard.start_page)."""
    from chalybclip.ingest import is_ffmpeg_available
    from chalybclip.integrations.storage import build_artifact_store

    s = get_settings()
    personas = await PersonasRepo(db).list_for_tenant()
    rtmp = (getattr(s, "live_rtmp_base_url", "") or "").strip()
    return {
        "personas": await _personas_json(db),
        "default_persona_id": personas[0].id if personas else None,
        "has_persona": bool(personas),
        "ffmpeg_ok": is_ffmpeg_available(),
        "live_configured": bool(rtmp),
        "upload": {
            # Browser → bucket direct upload needs object storage.
            "enabled": build_artifact_store(s) is not None,
            "max_bytes": int(getattr(s, "max_upload_bytes", 0) or 0),
        },
        "tier": getattr(request.state, "tenant_tier", "free"),
        "status": getattr(request.state, "tenant_status", "active"),
    }


# ── URL jobs ────────────────────────────────────────────────────────────


class UrlJobIn(BaseModel):
    vod_url: str = Field(min_length=1, max_length=2000)
    persona_id: str | None = Field(default=None, max_length=64)
    boost: bool | None = None


async def _resolve_persona(db: Database, persona_id: str | None) -> Any:
    personas = await PersonasRepo(db).list_for_tenant()
    if not personas:
        raise HubError(
            "bad_request",
            "Primero crea una persona en Ajustes → Personas.",
            reason="no_persona",
        )
    if persona_id:
        for p in personas:
            if p.id == persona_id:
                return p
        raise HubError("bad_request", "Esa persona no existe.", reason="unknown_persona")
    return personas[0]


@router.post("/url-jobs", dependencies=[Depends(require_active_tenant)])
async def hub_url_job(
    body: UrlJobIn,
    request: Request,
    background_tasks: BackgroundTasks,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Create a URL job; the full pipeline runs in the background.
    Mirrors dashboard.url_job_create (validation, auto persona, admit
    before download, placeholder row, stream.created, dispatch)."""
    from chalybclip.db.models import StreamRow
    from chalybclip.ids import new_id
    from chalybclip.ingest import Stream as StreamModel
    from chalybclip.ingest import detect_platform

    clean = body.vod_url.strip()
    platform = detect_platform(clean)
    if platform == "unknown":
        raise HubError(
            "bad_request",
            "Solo funcionan enlaces de Kick, Twitch y YouTube.",
            reason="link_unsupported",
        )
    persona = await _resolve_persona(db, body.persona_id)

    output_dir = _output_dir()
    stream_id = new_id("str")
    source_dir = output_dir / stream_id / "source"
    try:
        admission = await admit_job(
            db, tenant_id=tenant_id, stream_id=stream_id, boost=body.boost
        )
    except AdmissionRefused as e:
        raise _refused(e) from e

    try:
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url=clean,
                platform=platform,
                title=None,
                channel=None,
                duration_s=0.0,
                source_video_path=str(source_dir / "video.mp4"),
                source_audio_path=str(source_dir / "audio.wav"),
                status="pending",
                created_at=_now_iso(),
            )
        )
        await EventsRepo(db).emit(
            type="stream.created",
            payload={
                "stream_id": stream_id,
                "vod_url": clean,
                "platform": platform,
                "duration_s": 0.0,
                "kind": "url_job",
                "via": "hub",
            },
        )
        stub = StreamModel(
            id=stream_id,
            tenant_id=tenant_id,
            vod_url=clean,
            platform=platform,
            duration_s=0.0,
            source_video_path=source_dir / "video.mp4",
            source_audio_path=source_dir / "audio.wav",
        )
        await request.app.state.job_dispatcher.dispatch_pipeline(
            PipelineKickoff(
                tenant_id=tenant_id,
                stream=stub,
                persona_id=persona.id,
                output_dir=output_dir,
            ).with_admission(admission),
            background_tasks=background_tasks,
        )
    except BaseException:
        await cancel_admission(db, admission)
        raise
    return {"stream_id": stream_id}


# ── uploads: browser → bucket directly ───────────────────────────────────

_UPLOAD_TTL_S = 30 * 60
_ALLOWED_UPLOAD_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}


def _ticket_key() -> bytes:
    secret = (get_settings().chalyb_sso_secret or "").encode()
    if not secret:
        raise HubError("unavailable", reason="sso_secret_missing")
    return hmac.new(secret, b"chalybclip-hub-upload-v1", hashlib.sha256).digest()


def _sign_ticket(data: dict[str, Any]) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode()
    raw = raw.rstrip("=")
    sig = hmac.new(_ticket_key(), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def _read_ticket(ticket: str, *, tenant_id: str, stream_id: str) -> dict[str, Any]:
    try:
        raw, sig = ticket.split(".", 1)
    except ValueError:
        raise HubError("bad_request", reason="bad_ticket") from None
    expected = hmac.new(_ticket_key(), raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HubError("bad_request", reason="bad_ticket")
    data = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
    if data.get("t") != tenant_id or data.get("sid") != stream_id:
        raise HubError("not_found")
    if int(data.get("exp", 0)) < int(time.time()):
        raise HubError("bad_request", "La subida venció. Vuelve a intentarlo.", reason="expired")
    return data


class UploadIn(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    content_type: str = Field(max_length=100)
    persona_id: str | None = Field(default=None, max_length=64)
    boost: bool | None = None


@router.post("/uploads", dependencies=[Depends(require_active_tenant)])
async def hub_upload_start(
    body: UploadIn,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Step 1 of a direct upload: admit with the declared size (refuse
    before a byte moves — same as dashboard.streams_upload), then hand back
    a short-lived presigned PUT to the bucket plus a signed ticket."""
    from chalybclip.api._admission import effective_cap
    from chalybclip.api.routers.streams import _global_upload_cap
    from chalybclip.ids import new_id
    from chalybclip.integrations.storage import build_artifact_store, upload_source_key

    store = build_artifact_store(get_settings())
    if store is None or not hasattr(store, "presigned_put_url"):
        raise HubError("unavailable", "Subir archivos no está disponible.", reason="no_storage")
    suffix = _ALLOWED_UPLOAD_TYPES.get(body.content_type.lower())
    if suffix is None:
        raise HubError("bad_request", "Sube un video MP4, MOV, MKV o WEBM.", reason="bad_type")
    persona = await _resolve_persona(db, body.persona_id)

    stream_id = new_id("str")
    try:
        admission = await admit_job(
            db,
            tenant_id=tenant_id,
            stream_id=stream_id,
            upload_bytes=body.size_bytes,
            boost=body.boost,
        )
    except AdmissionRefused as e:
        raise _refused(e) from e
    cap = effective_cap(admission.max_upload_bytes, _global_upload_cap())
    if cap is not None and body.size_bytes > cap:
        await cancel_admission(db, admission)
        raise HubError("too_large", reason="upload_too_large")

    key = upload_source_key(tenant_id, stream_id, suffix)
    try:
        url = await store.presigned_put_url(  # type: ignore[attr-defined]
            key=key, ttl_seconds=_UPLOAD_TTL_S, content_type=body.content_type.lower()
        )
    except Exception:
        await cancel_admission(db, admission)
        raise
    exp = int(time.time()) + _UPLOAD_TTL_S
    ticket = _sign_ticket(
        {
            "sid": stream_id,
            "t": tenant_id,
            "key": key,
            "job": admission.job_id,
            "persona": persona.id,
            "title": body.filename[:255],
            "boost": body.boost,
            "cap": cap,
            "exp": exp,
        }
    )
    return {
        "stream_id": stream_id,
        "upload_url": url,
        "method": "PUT",
        "headers": {"Content-Type": body.content_type.lower()},
        "expires_at": exp,
        "ticket": ticket,
    }


class UploadDoneIn(BaseModel):
    ticket: str = Field(min_length=10, max_length=4000)


def _probe_minutes_url(url: str) -> float:
    import subprocess

    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", url,
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
        return max(0.0, float(out.stdout.strip())) / 60.0
    except Exception:
        return 0.0


async def _load_admission(db: Database, job_id: str) -> Any:
    from chalybclip.db.usage_repos import UsageJobsRepo
    from chalybclip.jobs.usage import JobAdmission

    row = await UsageJobsRepo(db).get(job_id)
    if row is None:
        return None
    return JobAdmission(
        job_id=row.id,
        tenant_id=row.tenant_id,
        stream_id=row.stream_id,
        reservation_id=getattr(row, "reservation_id", None),
        lane=getattr(row, "lane", "standard") or "standard",
    )


@router.post("/uploads/{stream_id}/complete", dependencies=[Depends(require_active_tenant)])
async def hub_upload_complete(
    stream_id: str,
    body: UploadDoneIn,
    request: Request,
    background_tasks: BackgroundTasks,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Step 2: the browser finished the PUT. Check the real size, probe the
    length (ffprobe reads the bucket object over a presigned GET), re-admit
    with the real numbers, then start the pipeline exactly like
    dashboard.streams_upload → upload_pipeline_runner does."""
    import asyncio

    from chalybclip.db.models import StreamRow
    from chalybclip.integrations.storage import build_artifact_store

    data = _read_ticket(body.ticket, tenant_id=tenant_id, stream_id=stream_id)
    if await StreamsRepo(db).get(stream_id) is not None:
        return {"stream_id": stream_id}  # already completed (double submit)

    store = build_artifact_store(get_settings())
    if store is None:
        raise HubError("unavailable", reason="no_storage")
    key = str(data["key"])
    admission = await _load_admission(db, str(data["job"]))
    size = await store.size(key=key)  # type: ignore[attr-defined]
    if size is None or size <= 0:
        raise HubError(
            "bad_request", "No recibimos el archivo. Vuelve a subirlo.", reason="not_uploaded"
        )
    cap = data.get("cap")
    if cap is not None and size > int(cap):
        await store.delete(key=key)
        await cancel_admission(db, admission)
        raise HubError("too_large", reason="upload_too_large")

    get_url = await store.presigned_url(key=key, ttl_seconds=600)
    minutes = await asyncio.to_thread(_probe_minutes_url, get_url)
    try:
        admission = (
            await readmit_job(
                db,
                job_id=str(data["job"]),
                upload_bytes=size,
                source_minutes=minutes,
                boost=data.get("boost"),
            )
            or admission
        )
    except AdmissionRefused as e:
        await store.delete(key=key)
        raise _refused(e) from e

    title = str(data.get("title") or "upload.mp4")
    output_dir = _output_dir()
    source_dir = output_dir / stream_id / "source"
    pseudo_url = f"upload://{title}"
    await StreamsRepo(db).upsert(
        StreamRow(
            id=stream_id,
            tenant_id=tenant_id,
            vod_url=pseudo_url,
            platform="upload",
            title=title,
            channel=None,
            duration_s=0.0,
            source_video_path=str(source_dir / "video.mp4"),
            source_audio_path=str(source_dir / "audio.wav"),
            status="pending",
            created_at=_now_iso(),
        )
    )
    await EventsRepo(db).emit(
        type="stream.created",
        payload={
            "stream_id": stream_id,
            "vod_url": pseudo_url,
            "platform": "upload",
            "duration_s": 0.0,
            "kind": "upload_job",
            "via": "hub",
        },
    )
    await _dispatch_parked_upload(
        request=request,
        background_tasks=background_tasks,
        tenant_id=tenant_id,
        stream_id=stream_id,
        persona_id=str(data["persona"]),
        key=key,
        title=title,
        admission=admission,
        store=store,
    )
    return {"stream_id": stream_id}


async def _dispatch_parked_upload(
    *,
    request: Request,
    background_tasks: BackgroundTasks,
    tenant_id: str,
    stream_id: str,
    persona_id: str,
    key: str,
    title: str,
    admission: Any,
    store: Any,
) -> None:
    """The file is already in the bucket. With a remote worker (or a
    boost-lane run) hand it the object key — the worker downloads, deletes
    the bucket copy and runs the upload pipeline (_pipeline.
    remote_upload_runner). Otherwise pull it here and run in-process, as
    dashboard.streams_upload does."""
    from chalybclip.api import _pipeline as pipeline_mod
    from chalybclip.ingest import Stream as StreamModel
    from chalybclip.jobs.boost import boost_configured
    from chalybclip.jobs.modal import ModalJobDispatcher

    dispatcher = request.app.state.job_dispatcher
    output_dir = _output_dir()
    inner = getattr(dispatcher, "inner", dispatcher)
    lane = getattr(admission, "lane", "standard") if admission else "standard"
    remote = isinstance(inner, ModalJobDispatcher) or (
        lane == "boost" and boost_configured() and inner is not dispatcher
    )
    source_dir = output_dir / stream_id / "source"
    if remote:
        stub = StreamModel(
            id=stream_id,
            tenant_id=tenant_id,
            vod_url=f"upload://{title}",
            platform="upload",
            title=title,
            duration_s=0.0,
            source_video_path=source_dir / "video.mp4",
            source_audio_path=source_dir / "audio.wav",
        )
        await dispatcher.dispatch_pipeline(
            PipelineKickoff(
                tenant_id=tenant_id,
                stream=stub,
                persona_id=persona_id,
                output_dir=output_dir,
                source_object_key=key,
                title=title,
            ).with_admission(admission)
        )
        return

    suffix = Path(key).suffix or ".mp4"
    dest = output_dir / "_incoming" / f"{stream_id}{suffix}"
    got = await store.download(key=key, dest=dest)
    if got is None:
        raise HubError("internal", reason="upload_fetch_failed")
    await store.delete(key=key)
    background_tasks.add_task(
        pipeline_mod.upload_pipeline_runner,
        tenant_id=tenant_id,
        stream_id=stream_id,
        persona_id=persona_id,
        tmp_path=got,
        output_dir=output_dir,
        title=title,
        dispatcher=None,
        usage_job_id=getattr(admission, "job_id", None),
        reservation_id=getattr(admission, "reservation_id", None),
        lane=lane,
    )


@router.post("/uploads/{stream_id}/abort")
async def hub_upload_abort(
    stream_id: str,
    body: UploadDoneIn,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """The browser gave up (cancelled, network died): release the admission
    and drop whatever reached the bucket."""
    from chalybclip.integrations.storage import build_artifact_store

    data = _read_ticket(body.ticket, tenant_id=tenant_id, stream_id=stream_id)
    if await StreamsRepo(db).get(stream_id) is not None:
        raise HubError("conflict", reason="already_started")
    store = build_artifact_store(get_settings())
    if store is not None:
        await store.delete(key=str(data["key"]))
    await cancel_admission(db, await _load_admission(db, str(data["job"])))
    return {"ok": True}


# ── streams ─────────────────────────────────────────────────────────────


@router.get("/streams")
async def hub_streams(
    limit: int = 50,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """The tenant's videos, newest first (dashboard.streams_list)."""
    from chalybclip.api.routers.dashboard import _stream_overview
    from chalybclip.jobs.active import active_stream_ids

    limit = max(1, min(int(limit), 200))
    streams = (await StreamsRepo(db).list_for_tenant())[:limit]
    active = active_stream_ids()
    settled = _settled_map(await EventsRepo(db).list_for_tenant(limit=5000), streams)
    clips = await ClipsRepo(db).list_for_tenant_with_status(ALL_CLIP_STATUSES, limit=5000)
    by_stream: dict[str, list[Any]] = {}
    for c in clips:
        by_stream.setdefault(c.stream_id, []).append(c)
    out = []
    for s in streams:
        sc = by_stream.get(s.id, [])
        item = _stream_json(
            s,
            active=s.id in active,
            clip_count=len(sc),
            ready_count=sum(1 for c in sc if c.status in ("approved", "published")),
        )
        kind = _stream_overview(s, sc, [])["status_kind"]
        if kind in ("pending", "failed") and s.id in active:
            kind = "running"
        item["status_kind"] = kind
        item["settled"] = settled.get(s.id, False)
        first_thumb = next((c for c in sc if c.thumbnail_frame_path), None)
        item["thumb_url"] = (
            media_path("thumb", first_thumb.id, tenant_id=tenant_id) if first_thumb else None
        )
        out.append(item)
    return {"streams": out}


async def _get_stream(db: Database, stream_id: str) -> Any:
    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HubError("not_found")
    return stream


@router.get("/streams/{stream_id}")
async def hub_stream_detail(
    stream_id: str,
    request: Request,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Everything the stream page shows (dashboard.stream_detail)."""
    from chalybclip.api.routers.dashboard import _rerun_available, _stream_overview
    from chalybclip.jobs.active import active_stream_ids

    stream = await _get_stream(db, stream_id)
    candidates = await CandidatesRepo(db).list_for_stream(stream_id)
    clips = await ClipsRepo(db).list_for_stream(stream_id)
    overview = _stream_overview(stream, clips, candidates)
    active = stream_id in active_stream_ids()
    if overview["status_kind"] in ("pending", "failed") and active:
        overview["status_kind"] = "running"

    cand_by_id = {c.id: c for c in candidates}
    enriched = []
    for e in overview["enriched_clips"]:
        clip = e["clip"]
        j = _clip_json(tenant_id, clip, cand_by_id.get(clip.candidate_id))
        j["timeline_left_pct"] = e["timeline_left_pct"]
        j["timeline_width_pct"] = e["timeline_width_pct"]
        enriched.append(j)
    best = overview["best"]
    source_available = False
    with contextlib.suppress(OSError):
        source_available = Path(stream.source_video_path).exists()
    progress = await stream_progress_view(
        db, stream, is_admin=bool(getattr(request.state, "is_admin", False))
    )
    return {
        "stream": _stream_json(
            stream,
            active=active,
            clip_count=len(clips),
            ready_count=overview["ready_count"],
        ),
        "overview": {
            "status_kind": overview["status_kind"],
            "total_clips": overview["total_clips"],
            "ready_count": overview["ready_count"],
            "rejected_count": overview["rejected_count"],
            "moments_detected": overview["moments_detected"],
            "best_clip_id": best["clip"].id if best else None,
            "best_score": overview["best_score"],
            "perf_label": overview["perf_label"],
            "time_saved_min": overview["time_saved_min"],
            "suggested_platforms": overview["suggested_platforms"],
        },
        "clips": enriched,
        "pending_approval_count": sum(
            1 for c in clips if c.status in ("cut", "ready_for_review")
        ),
        "source_url": (
            media_path("source", stream_id, tenant_id=tenant_id) if source_available else None
        ),
        "can_rerun": _rerun_available(stream, _output_dir()),
        "personas": await _personas_json(db),
        "cost": await _run_cost_view(db, request, stream, tenant_id),
        "progress": progress,
    }


@router.get("/streams/{stream_id}/progress")
async def hub_stream_progress(
    stream_id: str,
    request: Request,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    stream = await _get_stream(db, stream_id)
    return await stream_progress_view(
        db, stream, is_admin=bool(getattr(request.state, "is_admin", False))
    )


class RerunIn(BaseModel):
    persona_id: str = Field(min_length=1, max_length=64)
    boost: bool | None = None


@router.post("/streams/{stream_id}/rerun", dependencies=[Depends(require_active_tenant)])
async def hub_stream_rerun(
    stream_id: str,
    body: RerunIn,
    request: Request,
    background_tasks: BackgroundTasks,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Córrelo — re-run the pipeline. Mirrors dashboard.streams_rerun
    (YAML persona upsert, on-disk stream or re-fetch stub for URL sources,
    409 when nothing can be re-run, persona language, admit, dispatch,
    stream.rerun_requested)."""
    from chalybclip.api.routers.dashboard import _NON_REFETCHABLE_PLATFORMS
    from chalybclip.ingest import Stream as StreamModel
    from chalybclip.ingest import load_stream
    from chalybclip.variants import load_personas

    output_dir = _output_dir()
    stream_row = await _get_stream(db, stream_id)
    persona_id = body.persona_id
    db_personas = {p.id: p for p in await PersonasRepo(db).list_for_tenant()}
    if persona_id not in db_personas:
        try:
            yaml_personas = load_personas()
        except Exception:
            yaml_personas = {}
        if persona_id not in yaml_personas:
            raise HubError("bad_request", "Esa persona no existe.", reason="unknown_persona")
        p = yaml_personas[persona_id]
        await PersonasRepo(db).upsert(
            persona_id=p.id,
            name=p.name,
            primary_language=p.primary_language,
            target_languages=p.target_languages,
            voice_prompt=p.voice_prompt,
            routing_tags=p.routing_tags,
        )
    try:
        stream = load_stream(output_dir / stream_id)
    except Exception as e:
        refetchable = (
            bool(stream_row.vod_url)
            and stream_row.platform not in _NON_REFETCHABLE_PLATFORMS
        )
        if not refetchable:
            raise HubError(
                "conflict",
                "Este video ya no está guardado. Súbelo otra vez para volver a procesarlo.",
                reason="source_gone",
            ) from e
        stream = StreamModel(
            id=stream_id,
            tenant_id=tenant_id,
            vod_url=stream_row.vod_url,
            platform=stream_row.platform,  # type: ignore[arg-type]
            duration_s=stream_row.duration_s or 0.0,
            source_video_path=Path(stream_row.source_video_path),
            source_audio_path=Path(stream_row.source_audio_path),
        )
        await StreamsRepo(db).set_status(stream_id, status="pending")

    persona_language = None
    persona_row = await PersonasRepo(db).get(persona_id)
    if persona_row is not None and persona_row.primary_language:
        persona_language = persona_row.primary_language
    try:
        admission = await admit_job(
            db,
            tenant_id=tenant_id,
            stream_id=stream_id,
            source_minutes=float(stream.duration_s or stream_row.duration_s or 0) / 60.0,
            boost=body.boost,
        )
    except AdmissionRefused as e:
        raise _refused(e) from e
    await request.app.state.job_dispatcher.dispatch_pipeline(
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
            language=persona_language,
        ).with_admission(admission),
        background_tasks=background_tasks,
    )
    await EventsRepo(db).emit(type="stream.rerun_requested", payload={"stream_id": stream_id})
    return {"stream_id": stream_id}


@router.post("/streams/{stream_id}/approve-all")
async def hub_stream_approve_all(
    stream_id: str,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Status flip only (no render, no publish) — dashboard.stream_approve_all_clips."""
    await _get_stream(db, stream_id)
    repo = ClipsRepo(db)
    approved = 0
    for clip in await repo.list_for_stream(stream_id):
        if clip.status in ("cut", "ready_for_review"):
            await repo.update_status(clip.id, status="approved")
            approved += 1
    await EventsRepo(db).emit(
        type="clips.bulk_approved", payload={"stream_id": stream_id, "count": approved}
    )
    return {"approved": approved}


@router.post("/streams/{stream_id}/download-approved")
async def hub_stream_download_approved(
    stream_id: str,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """A signed link to the zip of approved/published clips (built when the
    browser opens it — see the `approved-zip` media kind)."""
    await _get_stream(db, stream_id)
    clips = await ClipsRepo(db).list_for_stream(stream_id)
    if not any(c.status in ("approved", "published") for c in clips):
        raise HubError("not_found", "Aún no hay clips aprobados.", reason="none_approved")
    return {"url": media_path("approved-zip", stream_id, tenant_id=tenant_id, ttl_s=900)}


@media_kind("approved-zip")
async def _approved_zip(
    request: Request, db: Database, tenant_id: str, stream_id: str, params: dict[str, str]
) -> Any:
    """dashboard.stream_download_approved, written to a temp file instead of
    memory, and pulling a clip back from the bucket when the local copy is
    gone (worker-run pipelines keep clips in object storage)."""
    import tempfile
    import zipfile

    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    from chalybclip.api._artifacts import rehydrate_clip

    clips = await ClipsRepo(db).list_for_stream(stream_id)
    ready = [c for c in clips if c.status in ("approved", "published")]
    if not ready:
        raise HTTPException(status_code=404, detail="no approved clips")
    import os

    fd, tmp_name = tempfile.mkstemp(prefix="chalybclip_zip_", suffix=".zip")
    os.close(fd)
    written = 0
    with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, c in enumerate(ready, 1):
            original = Path(c.path)
            final = original.parent / "clip_final.mp4"
            if not final.exists() and not original.exists():
                with contextlib.suppress(Exception):
                    await rehydrate_clip(tenant_id=tenant_id, clip_id=c.id, clip_path=original)
            src = final if final.exists() else original
            if not src.exists():
                continue
            zf.write(src, arcname=f"{i:02d}_chalybclip_{c.id}.mp4")
            written += 1
    if written == 0:
        Path(tmp_name).unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail="clip files missing")
    return FileResponse(
        tmp_name,
        media_type="application/zip",
        filename=f"chalybclip_stream_{stream_id}.zip",
        headers={"Cache-Control": "no-store"},
        background=BackgroundTask(lambda: Path(tmp_name).unlink(missing_ok=True)),
    )


@router.delete("/streams/{stream_id}")
async def hub_stream_delete(
    stream_id: str,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Hard delete with the same guards as dashboard.stream_delete: refuse
    while running, refuse while scheduled Zernio posts still use its clips,
    then cascade rows, bucket copies and the stream folder."""
    import shutil

    repo = StreamsRepo(db)
    stream = await _get_stream(db, stream_id)
    if stream.status == "running":
        raise HubError(
            "conflict",
            "Este video se está procesando. Espera a que termine para borrarlo.",
            reason="running",
        )
    pending = await ZernioPublishesRepo(db).pending_for_stream(stream_id)
    if pending:
        raise HubError(
            "conflict",
            f"{len(pending)} publicación(es) programada(s) usan clips de este video. "
            "Cancélalas o deja que se publiquen antes de borrarlo.",
            reason="scheduled_posts",
        )
    stream_dir: Path | None = None
    with contextlib.suppress(Exception):
        parent = Path(stream.source_video_path).resolve().parent
        if parent.name == stream_id:
            stream_dir = parent
    clip_ids: list[str] = []
    with contextlib.suppress(Exception):
        clip_ids = [c.id for c in await ClipsRepo(db).list_for_stream(stream_id)]
    if not await repo.delete(stream_id):
        return {"deleted": True}
    if clip_ids:
        try:
            from chalybclip.integrations.storage import build_artifact_store, clip_key_family

            store = build_artifact_store(get_settings())
            if store is not None:
                for cid in clip_ids:
                    for key in clip_key_family(tenant_id, cid):
                        await store.delete(key=key)
        except Exception:
            _log.warning("hub.stream_delete.bucket_cleanup_failed", stream_id=stream_id)
    if stream_dir and stream_dir.exists():
        shutil.rmtree(stream_dir, ignore_errors=True)
    await EventsRepo(db).emit(
        type="stream.deleted",
        payload={"stream_id": stream_id, "fs_dir": str(stream_dir) if stream_dir else None},
    )
    return {"deleted": True}


@router.post("/streams/{stream_id}/settle")
async def hub_stream_settle(
    stream_id: str,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """The hub's once-per-run "your clips are ready" bookkeeping (its job
    policy notifies exactly once). `first` is True only for the call that
    recorded it for the stream's latest run."""
    stream = await _get_stream(db, stream_id)
    events = await EventsRepo(db).list_for_tenant(limit=5000)
    if _settled_map(events, [stream]).get(stream_id):
        return {"first": False}
    await EventsRepo(db).emit(type="hub.settled", payload={"stream_id": stream_id})
    return {"first": True}


# ── clips (Mis clips) ────────────────────────────────────────────────────


@router.get("/clips")
async def hub_clips(
    limit: int = 60,
    status: str | None = None,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """The tenant's clips across every video, newest first."""
    statuses = [s for s in (status or "").split(",") if s in ALL_CLIP_STATUSES]
    clips = await ClipsRepo(db).list_for_tenant_with_status(
        statuses or ALL_CLIP_STATUSES, limit=max(1, min(int(limit), 500))
    )
    return {"clips": [_clip_json(tenant_id, c) for c in clips]}

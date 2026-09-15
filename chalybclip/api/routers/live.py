"""Live RTMP ingest — phase L.1.

This router carries TWO surfaces:

  1. Dashboard pages under /dashboard/live for the operator: see the
     RTMP URL + stream key, rotate the key, see active + recent live
     streams. Auth via the standard tenant cookie middleware.

  2. Internal webhook endpoints under /api/internal/live for MediaMTX
     to call: authorize a publish attempt, signal stream start, signal
     stream end. Auth via shared `CHALYBCLIP_INTERNAL_SIGNING_SECRET`
     bearer (same secret the recorder uses for its signed audio URL —
     the secret is internal-trust-only).

Architecture reference: docs/phase_L_live.md.

What L.1 does NOT do:
  - No live transcription / clipping. That's L.2+.
  - No MediaMTX deployment automation. The operator deploys MediaMTX
    as a separate Railway service; this code just receives its
    webhooks. See docs/mediamtx_deploy.md.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from chalybclip.db import Database, LiveStreamKeysRepo, StreamsRepo
from chalybclip.db.repos import (
    TenantsRepo,
    _streams_repo_mark_live_ended,
    _streams_repo_mark_live_started,
    new_id_with_prefix,
)
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

from ..deps import get_db, require_full_scope, tenant_binder

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
from ..i18n import install_globals as _install_i18n  # noqa: E402
_install_i18n(templates)

router = APIRouter(prefix="", tags=["live"], include_in_schema=False)


# ---- Dashboard surface (tenant-scoped) ------------------------------------


@router.get("/dashboard/live", response_class=HTMLResponse)
async def live_dashboard(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Operator-facing live page. Shows RTMP URL + active key, rotate
    button, and the list of in-progress + recent live streams."""
    keys_repo = LiveStreamKeysRepo(db)
    streams_repo = StreamsRepo(db)
    active_key = await keys_repo.get_active_for_tenant()
    settings = get_settings()
    # The RTMP endpoint URL the operator pastes into OBS. Comes from
    # the same settings as the recorder's public URL, but with the
    # RTMP host the operator's MediaMTX deployment exposes.
    rtmp_url_base = (settings.live_rtmp_base_url or "").rstrip("/")
    # Filter all streams to the live-related ones (currently live OR
    # in retention window OR expired). Keeps the regular VOD list
    # page uncluttered.
    all_streams = await streams_repo.list_for_tenant()
    live_streams = [
        s for s in all_streams
        if s.is_live or (s.status in ("live_ended", "live"))
    ]
    # ChalybOBS is the streaming front-door now: the operator connects their
    # encoder there (multistream + preview), and ChalybOBS forwards the
    # recording to this pipeline for clipping. The live page links out to it
    # via Chalyb's cross-app SSO launcher so the session carries over (no
    # landing/login bounce). The launcher gates to ALL_ACCESS server-side.
    import os as _os

    from chalybclip.integrations.chalybobs import get_connection

    chalyb_base = _os.environ.get(
        "CHALYB_BASE_URL", "https://chalyb.com"
    ).rstrip("/")
    chalybobs_url = f"{chalyb_base}/auth/launch/chalybobs"

    # Bidirectional connection switch state + full-access gate.
    tenant = await TenantsRepo(db).get(tenant_id)
    is_full_access = bool(tenant and (tenant.tier or "").lower() == "all_access")
    external_user_id = tenant.external_user_id if tenant else None
    connection_on = (
        await get_connection(external_user_id) if external_user_id else False
    )
    return templates.TemplateResponse(
        request,
        "live_dashboard.html",
        {
            "active_key": active_key,
            "rtmp_url_base": rtmp_url_base,
            "live_streams": live_streams,
            "configured": bool(rtmp_url_base),
            "chalybobs_url": chalybobs_url,
            "is_full_access": is_full_access,
            "connection_on": connection_on,
            "has_external": bool(external_user_id),
        },
    )


@router.post(
    "/dashboard/live/connection",
    dependencies=[Depends(require_full_scope)],
)
async def live_set_connection(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Flip the bidirectional ChalybOBS↔ChalybClip connection switch from the
    ChalybClip side. Full-access only — mirror of the gate ChalybOBS enforces.
    The new state is read from the posted `enabled` field."""
    from chalybclip.integrations.chalybobs import set_connection

    tenant = await TenantsRepo(db).get(tenant_id)
    is_full_access = bool(tenant and (tenant.tier or "").lower() == "all_access")
    if not (is_full_access and tenant and tenant.external_user_id):
        return RedirectResponse(url="/dashboard/live", status_code=303)

    form = await request.form()
    enabled = str(form.get("enabled", "")).lower() in ("1", "true", "on", "yes")
    await set_connection(tenant.external_user_id, enabled)
    return RedirectResponse(url="/dashboard/live", status_code=303)


@router.post(
    "/dashboard/live/rotate-key",
    dependencies=[Depends(require_full_scope)],
)
async def live_rotate_key(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Generate a fresh RTMP key + revoke any existing active one.

    Mirrors the spec's idempotence note: calling twice in a row just
    rotates twice. The dashboard's rotate button shows a confirm
    dialog so accidental clicks are caught client-side."""
    await LiveStreamKeysRepo(db).rotate_for_tenant()
    return RedirectResponse(url="/dashboard/live?rotated=1", status_code=303)


# ---- Internal webhooks (MediaMTX -> ChalybClip) -----------------------------
#
# All three endpoints share the same auth model: shared
# CHALYBCLIP_INTERNAL_SIGNING_SECRET as a bearer. MediaMTX is configured
# with that secret in its hooks block (infra/mediamtx.yml). The
# secret is internal-trust-only — MediaMTX runs as a sibling service
# on the same Railway project, not user-facing.


def _verify_internal_bearer(authorization: str | None) -> None:
    settings = get_settings()
    expected = (settings.internal_signing_secret or "").strip()
    if not expected:
        # Misconfigured server — fail closed so MediaMTX rejects
        # the push rather than silently letting anyone in.
        raise HTTPException(
            status_code=503,
            detail="CHALYBCLIP_INTERNAL_SIGNING_SECRET not configured",
        )
    received = ""
    if authorization and authorization.lower().startswith("bearer "):
        received = authorization[len("bearer "):].strip()
    if received != expected:
        raise HTTPException(status_code=401, detail="bad bearer")


# ── authorize → started cross-check registry ────────────────────────────────
# When /authorize accepts an RTMP push it mints a fresh stream_id bound to
# the stream key's tenant. /started then arrives with both stream_id AND a
# tenant_id from MediaMTX's payload — if we trust the body verbatim, an
# attacker holding CHALYBCLIP_INTERNAL_SIGNING_SECRET can inject stream rows
# into ANY tenant by sending a (fresh stream_id, victim tenant_id) pair.
#
# This in-process registry binds the stream_id → tenant_id mapping issued
# by /authorize so /started can require an exact match. Entries auto-expire
# after _AUTHORIZE_TTL_S; expired entries are GC'd opportunistically on
# each write.
#
# Single-process only. ChalybClip currently runs as a single Railway replica;
# when this horizontally scales, swap for Redis (same Redis story as the
# rate-limit infra needed for engine-usage and contact-form endpoints).
_AUTHORIZE_REGISTRY: dict[str, tuple[str, float]] = {}
_AUTHORIZE_TTL_S = 60.0


def _gc_authorize_registry(now: float) -> None:
    """Drop expired entries. O(N) sweep; N stays tiny because the registry
    only holds in-flight authorize→started pairs (seconds-long lifespan)."""
    expired = [sid for sid, (_t, exp) in _AUTHORIZE_REGISTRY.items() if exp < now]
    for sid in expired:
        _AUTHORIZE_REGISTRY.pop(sid, None)


def _register_authorize(stream_id: str, tenant_id: str) -> None:
    now = _time.time()
    _gc_authorize_registry(now)
    _AUTHORIZE_REGISTRY[stream_id] = (tenant_id, now + _AUTHORIZE_TTL_S)


def _consume_authorize(stream_id: str) -> str | None:
    """Pop the registered tenant_id for stream_id, or None if no entry
    exists / it expired. Single-use: a subsequent /started call for the
    same stream_id without a fresh /authorize is rejected."""
    now = _time.time()
    _gc_authorize_registry(now)
    entry = _AUTHORIZE_REGISTRY.pop(stream_id, None)
    if entry is None:
        return None
    tenant_id, exp = entry
    if exp < now:
        return None
    return tenant_id


@router.post("/api/internal/live/authorize")
async def live_authorize(
    request: Request,
    payload: dict,
    authorization: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """MediaMTX calls this BEFORE accepting an RTMP push. We extract
    the key from the path and 200/401 based on whether it's a known
    active key.

    Expected payload shape (MediaMTX 1.x runOnConnect or
    authenticationURL webhook):
        {
          "path": "live/slk_01XXX",
          "action": "publish",
          "ip": "203.0.113.7"
        }

    Returns 200 with {tenant_id, stream_id, recording_path} on
    success. MediaMTX uses the recording_path field to know where to
    write the MP4 mirror.
    """
    _verify_internal_bearer(authorization)
    path = str((payload or {}).get("path") or "")
    action = str((payload or {}).get("action") or "")
    client_ip = str((payload or {}).get("ip") or "")
    if action and action != "publish":
        # MediaMTX uses this hook for read auth too; we only want
        # to gate publishes. Read is open (no live preview surface
        # yet — that's a future slice).
        return JSONResponse({"ok": True, "scope": "read"})

    # Extract the key. Expected path format: "live/<key>"
    parts = path.split("/")
    if len(parts) < 2 or parts[0] != "live":
        raise HTTPException(status_code=400, detail="bad path; expected live/<key>")
    key_value = parts[1].strip()
    if not key_value:
        raise HTTPException(status_code=400, detail="empty key")

    db: Database = request.app.state.db
    keys_repo = LiveStreamKeysRepo(db)
    key_row = await keys_repo.find_by_value(key_value)
    if key_row is None or key_row.revoked_at is not None:
        raise HTTPException(status_code=401, detail="unknown or revoked key")

    # Generate a stream_id we'll use for the streams row + the
    # MediaMTX recording path. MediaMTX writes to
    # /data/live/<stream_id>/source.mp4. We DO NOT create the streams
    # row here — that happens in /started so it only exists when the
    # push actually goes through.
    stream_id = new_id_with_prefix("str")

    # Bind stream_id to the tenant_id determined by the (authenticated)
    # stream key so /started can verify its body's tenant_id against
    # ours. See _register_authorize above for the threat model.
    _register_authorize(stream_id, key_row.tenant_id)

    # Update last_used_at so the dashboard can show "this key was
    # used 12s ago". Best-effort.
    try:
        await keys_repo.touch_last_used(key_row.id)
    except Exception:  # noqa: BLE001
        pass

    return JSONResponse({
        "ok": True,
        "tenant_id": key_row.tenant_id,
        "stream_id": stream_id,
        # MediaMTX will substitute %path into its recordPath template;
        # we hand back the absolute path so the operator's MediaMTX
        # config doesn't need to know our /data layout.
        "recording_path": f"/data/live/{stream_id}/source",
        "client_ip": client_ip,
    })


@router.post("/api/internal/live/started")
async def live_started(
    request: Request,
    payload: dict,
    authorization: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """MediaMTX calls this when the RTMP push is accepted + recording
    has started. We create the streams row + flip status to 'live'.

    Expected payload:
        {
          "stream_id": "str_01XXX",
          "tenant_id": "ten_01XXX",
          "recording_path": "/data/live/str_01XXX/source.mp4"
        }
    """
    _verify_internal_bearer(authorization)
    body = payload or {}
    stream_id = str(body.get("stream_id") or "").strip()
    tenant_id = str(body.get("tenant_id") or "").strip()
    recording_path = str(body.get("recording_path") or "").strip()
    if not (stream_id and tenant_id and recording_path):
        raise HTTPException(
            status_code=400,
            detail="stream_id, tenant_id, recording_path required",
        )

    # Verify the body's tenant_id matches what /authorize issued for
    # this stream_id. The bearer alone is not enough — anyone holding
    # it could otherwise pair a fresh stream_id with any tenant_id and
    # inject a bogus streams row into a victim tenant. See
    # _register_authorize for the threat model.
    expected_tenant = _consume_authorize(stream_id)
    if expected_tenant is None:
        raise HTTPException(
            status_code=401,
            detail="no authorize record for stream_id (or expired)",
        )
    if expected_tenant != tenant_id:
        raise HTTPException(
            status_code=401,
            detail="tenant_id does not match authorize record",
        )

    db: Database = request.app.state.db
    # Create the streams row bound to the tenant. Use the recording
    # path as source_video_path so the existing pipeline (Whisper,
    # detect, cut) reads the right file once the operator triggers
    # it post-stream.
    from chalybclip.db.adapters import _now as _adapters_now

    now = _adapters_now()
    audio_path = recording_path.rsplit(".", 1)[0] + ".audio.wav"
    with bound_tenant(tenant_id):
        streams_repo = StreamsRepo(db)
        # Construct minimal StreamRow; duration_s will be filled by
        # the /ended webhook once we know the final length.
        from chalybclip.db.models import StreamRow

        row = StreamRow(
            id=stream_id,
            tenant_id=tenant_id,
            vod_url=f"live://rtmp/{stream_id}",
            platform="live",
            title=f"Live stream {stream_id[:12]}",
            channel=None,
            duration_s=0.0,
            source_video_path=recording_path,
            source_audio_path=audio_path,
            status="live",
            created_at=now,
            is_live=True,
            live_started_at=now,
        )
        await streams_repo.upsert(row)
        # The upsert path uses INSERT OR IGNORE so the live-only
        # fields aren't covered. Flip them explicitly.
        await _streams_repo_mark_live_started(db, stream_id=stream_id)

    return JSONResponse({"ok": True})


@router.post("/api/internal/live/ended")
async def live_ended(
    request: Request,
    payload: dict,
    background_tasks: BackgroundTasks,
    authorization: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """MediaMTX calls this when the RTMP push ends (streamer hits
    Stop in OBS, or the TCP connection drops). We flip status to
    'live_ended' and — Phase L.2 — auto-launch the clip pipeline on
    the recording so the streamer gets publish-ready clips with zero
    dashboard interaction (gated by CHALYBCLIP_LIVE_AUTO_CLIP, default on).

    Expected payload:
        {
          "stream_id": "str_01XXX",
          "duration_s": 7384.2
        }
    """
    _verify_internal_bearer(authorization)
    body = payload or {}
    stream_id = str(body.get("stream_id") or "").strip()
    duration_s = body.get("duration_s")
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id required")
    db: Database = request.app.state.db
    final_duration = None
    if isinstance(duration_s, int | float):
        final_duration = float(duration_s)
    ended_row = await _streams_repo_mark_live_ended(
        db, stream_id=stream_id, duration_s=final_duration
    )

    # Phase L.2 — auto-clip. Claim the stream (idempotent against
    # duplicate webhooks) and schedule the pipeline on the recording.
    # Best-effort: a failure here must never 500 the webhook (MediaMTX
    # would keep retrying); the manual "Run pipeline" button is the
    # fallback.
    autoclip_scheduled = False
    try:
        from chalybclip.api._pipeline import (
            live_pipeline_runner,
            maybe_autoclip_after_live_end,
        )

        def _schedule(**kwargs: object) -> None:
            background_tasks.add_task(live_pipeline_runner, **kwargs)

        autoclip_scheduled = await maybe_autoclip_after_live_end(
            db, row=ended_row, schedule=_schedule
        )
    except Exception:  # noqa: BLE001 — observability only
        # Don't let an autoclip hiccup fail the webhook.
        pass

    return JSONResponse(
        {"ok": True, "stream_id": stream_id, "autoclip": autoclip_scheduled}
    )


# ---- ChalybOBS handoff (ChalybOBS -> ChalybClip) --------------------------------
#
# When the bidirectional connection switch is ON, ChalybOBS forwards each
# stream's lifecycle here. Unlike the relay-native /api/internal/live/*
# endpoints (which carry ChalybClip's own ten_ id), ChalybOBS only knows the
# Chalyb user id — so these endpoints MAP external_user_id -> ChalybClip
# tenant, then reuse the exact same live pipeline (create stream row →
# autoclip on end). The recording is fetched from object storage by
# stream_id (the id ChalybOBS minted + the relay uploaded under).


@router.post("/api/internal/chalybobs/started")
async def chalybobs_started(
    request: Request,
    payload: dict,
    authorization: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """ChalybOBS stream went live → create the streams row so it shows in the
    Live list. Maps external_user_id (Chalyb user id) to the ChalybClip
    tenant. Full-access only."""
    _verify_internal_bearer(authorization)
    body = payload or {}
    external_user_id = str(
        body.get("external_user_id") or body.get("tenant_id") or ""
    ).strip()
    stream_id = str(body.get("stream_id") or "").strip()
    recording_path = (
        str(body.get("recording_path") or "").strip()
        or f"/data/live/{stream_id}/source"
    )
    if not (external_user_id and stream_id):
        raise HTTPException(status_code=400, detail="external_user_id, stream_id required")

    db: Database = request.app.state.db
    tenant = await TenantsRepo(db).find_by_external_user_id(external_user_id)
    if tenant is None:
        return JSONResponse({"ok": False, "reason": "no_tenant"})
    if (tenant.tier or "").lower() != "all_access":
        return JSONResponse({"ok": False, "reason": "not_full_access"})

    from chalybclip.db.adapters import _now as _adapters_now
    from chalybclip.db.models import StreamRow

    now = _adapters_now()
    audio_path = recording_path.rsplit(".", 1)[0] + ".audio.wav"
    # Prefer the title the operator set for the stream in ChalybOBS (forwarded
    # in the webhook payload). Fall back to a per-session tag when ChalybOBS
    # didn't send one: stream_id is <tenant>__<random>, so the random suffix
    # is the per-session part (the tenant prefix is identical across a
    # tenant's streams and made every row look the same in the list).
    session_tag = stream_id.rsplit("__", 1)[-1][:10]
    title = str(body.get("title") or "").strip() or f"ChalybOBS live · {session_tag}"
    with bound_tenant(tenant.id):
        streams_repo = StreamsRepo(db)
        row = StreamRow(
            id=stream_id,
            tenant_id=tenant.id,
            vod_url=f"live://chalybobs/{stream_id}",
            platform="live",
            title=title,
            channel="ChalybOBS",
            duration_s=0.0,
            source_video_path=recording_path,
            source_audio_path=audio_path,
            status="live",
            created_at=now,
            is_live=True,
            live_started_at=now,
        )
        await streams_repo.upsert(row)
        await _streams_repo_mark_live_started(db, stream_id=stream_id)

    return JSONResponse({"ok": True, "tenant_id": tenant.id})


@router.post("/api/internal/chalybobs/ended")
async def chalybobs_ended(
    request: Request,
    payload: dict,
    background_tasks: BackgroundTasks,
    authorization: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """ChalybOBS stream ended → mark ended + auto-clip the recording (pulled
    from storage by stream_id). Full-access only."""
    _verify_internal_bearer(authorization)
    body = payload or {}
    external_user_id = str(
        body.get("external_user_id") or body.get("tenant_id") or ""
    ).strip()
    stream_id = str(body.get("stream_id") or "").strip()
    duration_s = body.get("duration_s")
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id required")

    db: Database = request.app.state.db
    tenant = (
        await TenantsRepo(db).find_by_external_user_id(external_user_id)
        if external_user_id
        else None
    )
    final_duration = (
        float(duration_s) if isinstance(duration_s, int | float) else None
    )
    ended_row = await _streams_repo_mark_live_ended(
        db, stream_id=stream_id, duration_s=final_duration
    )

    autoclip_scheduled = False
    if tenant is not None and (tenant.tier or "").lower() == "all_access":
        try:
            from chalybclip.api._pipeline import (
                live_pipeline_runner,
                maybe_autoclip_after_live_end,
            )

            def _schedule(**kwargs: object) -> None:
                background_tasks.add_task(live_pipeline_runner, **kwargs)

            autoclip_scheduled = await maybe_autoclip_after_live_end(
                db, row=ended_row, schedule=_schedule
            )
        except Exception:  # noqa: BLE001 — never fail the webhook
            pass

    return JSONResponse(
        {"ok": True, "stream_id": stream_id, "autoclip": autoclip_scheduled}
    )


__all__ = ["router"]

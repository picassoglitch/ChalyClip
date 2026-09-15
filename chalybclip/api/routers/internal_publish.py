"""Internal Publish API — /api/internal/v1/* (Hub phase 3).

The service-to-service entry point ChalybOBS Auto-Clip Mode and Chalyb
engines publish through. Auth is a bearer service token validated
against CHALYBCLIP_HUB_SERVICE_TOKENS (comma-separated name:token pairs)
— consumers address tenants by id and never see Zernio.

Routes are thin (parse → service → respond); the logic lives in
chalybclip.publish.hub. Lives under /api/internal/ so the bearer-cookie
middleware exemption applies (each route enforces its own auth here,
same pattern as the other /api/internal surfaces).

  POST /api/internal/v1/publish            — one clip, all modes
  GET  /api/internal/v1/publish/{job_id}   — live status (webhook-fed)
  GET  /api/internal/v1/accounts           — connected platforms
  POST /api/internal/v1/batch              — a session's clips, anti-spam spread
"""
from __future__ import annotations

import hmac
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from chalybclip.db import Database, HubPublishJobsRepo, TenantsRepo
from chalybclip.integrations.zernio.client import ZernioClient, ZernioError
from chalybclip.publish.hub import (
    HubPublishError,
    PublishOptions,
    hub_batch,
    hub_publish,
)
from chalybclip.settings import get_settings

from ..deps import get_db

_log = logging.getLogger("chalybclip.api.internal_publish")

router = APIRouter(prefix="/api/internal/v1", tags=["hub"], include_in_schema=False)


def require_hub_service(request: Request) -> str:
    """Validate the bearer service token; return the consumer name.

    503 when no tokens are configured (the API is OFF, not open), 401
    on a missing/unknown token. Constant-time comparison per candidate
    so token probing gains nothing from timing.
    """
    tokens = get_settings().hub_service_token_map()
    if not tokens:
        raise HTTPException(
            status_code=503,
            detail=(
                "CHALYBCLIP_HUB_SERVICE_TOKENS is not configured; the "
                "internal publish API is disabled."
            ),
        )
    header = request.headers.get("authorization") or ""
    scheme, _, provided = header.partition(" ")
    provided = provided.strip()
    if scheme.lower() != "bearer" or not provided:
        raise HTTPException(status_code=401, detail="missing bearer service token")
    for token, name in tokens.items():
        if hmac.compare_digest(token, provided):
            return name
    raise HTTPException(status_code=401, detail="unknown service token")


def _build_client() -> ZernioClient:
    settings = get_settings()
    if not settings.zernio_api_key:
        raise HTTPException(
            status_code=503, detail="CHALYBCLIP_ZERNIO_API_KEY is not configured",
        )
    return ZernioClient(
        api_key=settings.zernio_api_key, base_url=settings.zernio_base_url,
    )


def _error_response(e: HubPublishError) -> JSONResponse:
    """The structured-error contract: machine-readable `error` code +
    human `message`. plan_limit keeps HTTP 402 but NEVER a raw Zernio
    body."""
    return JSONResponse(
        {"ok": False, "error": e.code, "message": e.message},
        status_code=e.http_status,
    )


class ClipSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_url: str
    title: str | None = None
    caption_default: str | None = None
    duration_s: float | None = None  # informational; not validated


class PublishOptionsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    per_platform_captions: dict[str, Any] = Field(default_factory=dict)
    first_comment: str | None = None
    use_best_time: bool = False

    def to_options(self) -> PublishOptions:
        return PublishOptions(
            per_platform_captions=self.per_platform_captions,
            first_comment=self.first_comment,
            use_best_time=self.use_best_time,
        )


class PublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str
    clip: ClipSpec
    targets: list[str] | Literal["all_connected"]
    mode: Literal["now", "queue", "schedule", "draft"] = "now"
    scheduled_for: str | None = None
    options: PublishOptionsIn = Field(default_factory=PublishOptionsIn)
    source: Literal["chalybobs", "chalybclip", "chalyb"]
    idempotency_key: str | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str
    clips: list[ClipSpec] = Field(min_length=1, max_length=50)
    targets: list[str] | Literal["all_connected"]
    options: PublishOptionsIn = Field(default_factory=PublishOptionsIn)
    source: Literal["chalybobs", "chalybclip", "chalyb"]
    idempotency_key: str | None = None


@router.post("/publish")
async def internal_publish(
    body: PublishRequest,
    consumer: str = Depends(require_hub_service),
    db: Database = Depends(get_db),
) -> JSONResponse:
    """Publish one clip on behalf of a tenant. Idempotent on
    (tenant_id, idempotency_key)."""
    try:
        result = await hub_publish(
            db,
            body.tenant_id,
            video_url=body.clip.video_url,
            title=body.clip.title,
            caption=body.clip.caption_default,
            targets=body.targets,
            mode=body.mode,
            scheduled_for=body.scheduled_for,
            options=body.options.to_options(),
            source=body.source,
            idempotency_key=body.idempotency_key,
            client=_build_client(),
        )
    except HubPublishError as e:
        _log.info(
            "hub.publish.rejected consumer=%s tenant=%s error=%s",
            consumer, body.tenant_id, e.code,
        )
        return _error_response(e)
    job = result.job
    return JSONResponse(
        {
            "ok": True,
            "job_id": job.job_id,
            "zernio_post_id": job.zernio_post_id,
            "status": job.status,
            "mode": job.mode,
            "scheduled_for": job.scheduled_for,
            "platforms": result.platforms,
            "duplicate": result.duplicate,
        }
    )


@router.get("/publish/{job_id}")
async def internal_publish_status(
    job_id: str,
    tenant_id: str,
    _consumer: str = Depends(require_hub_service),
    db: Database = Depends(get_db),
) -> JSONResponse:
    """Live job status — fed by the phase-2 webhooks, no Zernio poll."""
    job = await HubPublishJobsRepo(db).get(tenant_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job for this tenant")
    return JSONResponse(
        {
            "ok": True,
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
            "source": job.source,
            "mode": job.mode,
            "status": job.status,
            "targets": [p for p in job.targets.split(",") if p],
            "zernio_post_id": job.zernio_post_id,
            "scheduled_for": job.scheduled_for,
            "platforms_json": job.platforms_json,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
    )


@router.get("/accounts")
async def internal_accounts(
    tenant_id: str,
    _consumer: str = Depends(require_hub_service),
    db: Database = Depends(get_db),
) -> JSONResponse:
    """Connected platforms for a tenant — what ChalybOBS shows before
    offering publish targets."""
    tenant = await TenantsRepo(db).get(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="unknown tenant")
    if not tenant.zernio_profile_id:
        return JSONResponse({"ok": True, "connected": [], "accounts": []})
    try:
        accounts = await _build_client().list_accounts(
            profile_id=tenant.zernio_profile_id,
        )
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": "zernio_error", "message": str(e)},
            status_code=502,
        )
    return JSONResponse(
        {
            "ok": True,
            "connected": sorted({a.platform.lower() for a in accounts}),
            "accounts": [
                {"platform": a.platform, "account_id": a.account_id}
                for a in accounts
            ],
        }
    )


@router.post("/batch")
async def internal_batch(
    body: BatchRequest,
    consumer: str = Depends(require_hub_service),
    db: Database = Depends(get_db),
) -> JSONResponse:
    """Publish a stream session's clips, distributed across queue/best
    times under the per-platform daily cap (default 4) — never fired
    all at once."""
    settings = get_settings()
    try:
        results = await hub_batch(
            db,
            body.tenant_id,
            clips=[c.model_dump() for c in body.clips],
            targets=body.targets,
            source=body.source,
            options=body.options.to_options(),
            idempotency_key=body.idempotency_key,
            cap_per_day=settings.hub_max_posts_per_platform_per_day,
            client=_build_client(),
        )
    except HubPublishError as e:
        _log.info(
            "hub.batch.rejected consumer=%s tenant=%s error=%s",
            consumer, body.tenant_id, e.code,
        )
        return _error_response(e)
    return JSONResponse(
        {
            "ok": True,
            "results": results,
            "scheduled": sum(1 for r in results if r.get("ok")),
            "failed": sum(1 for r in results if not r.get("ok")),
        }
    )


@router.get("/analytics")
async def internal_analytics_read(
    tenant_id: str,
    days: int = 30,
    _consumer: str = Depends(require_hub_service),
    db: Database = Depends(get_db),
) -> JSONResponse:
    """Performance read for Chalyb engines — so a ranking engine can
    see which VOD moments produced winning clips. Live Zernio analytics
    when available, latest stored snapshots as fallback. Metrics follow
    the no-fake-zeros rule (absent → null, never an invented 0)."""
    from chalybclip.publish.analytics_service import internal_analytics

    tenant = await TenantsRepo(db).get(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="unknown tenant")
    days = max(1, min(int(days), 366))
    settings = get_settings()
    client = _build_client() if settings.zernio_api_key else None
    result = await internal_analytics(db, tenant_id, days=days, client=client)
    return JSONResponse({"ok": True, **result})


__all__ = ["router"]

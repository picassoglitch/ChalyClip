"""Hub API — live (RTMP), channel watches, brand kits, personas,
calibration and the admin-only pages (LLM calls log, LLM settings, site
settings). Mounted under /api/hub/v1.

Same repos and rules as the dashboard handlers they replace; each route
names its dashboard counterpart.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import structlog
from fastapi import BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

from chalybclip.api.deps import get_db
from chalybclip.api.hub_api import HubError, hub_router, hub_tenant
from chalybclip.db import (
    BrandKitsRepo,
    ChannelWatchesRepo,
    Database,
    EventsRepo,
    LiveStreamKeysRepo,
    LLMCallsRepo,
    PersonasRepo,
    PlatformSettingsRepo,
    StreamsRepo,
    TenantsRepo,
)
from chalybclip.db.models import CustomTriggerPhrases
from chalybclip.settings import get_settings

router = hub_router(tags=["hub-settings"])
_log = structlog.get_logger("chalybclip.api.hub.settings")


def _require_admin(request: Request) -> None:
    """Operator-only pages answer 404 to everyone else, like the dashboard."""
    if not getattr(request.state, "is_admin", False):
        raise HubError("not_found")


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, _dt.datetime | _dt.date) else v


# ── live (dashboard /live) ───────────────────────────────────────────────


def _mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}" if len(key) > 10 else "…"


async def _live_view(db: Database, tenant_id: str) -> dict[str, Any]:
    from chalybclip.integrations.chalybobs import get_connection

    s = get_settings()
    rtmp = (s.live_rtmp_base_url or "").rstrip("/")
    key = await LiveStreamKeysRepo(db).get_active_for_tenant()
    streams = [
        st
        for st in await StreamsRepo(db).list_for_tenant()
        if st.is_live or st.status in ("live_ended", "live")
    ]
    tenant = await TenantsRepo(db).get(tenant_id)
    full_access = bool(tenant and (tenant.tier or "").lower() == "all_access")
    ext = tenant.external_user_id if tenant else None
    return {
        "configured": bool(rtmp),
        "rtmp_url": rtmp or None,
        "key": (
            {
                "value": key.key_value,
                "masked": _mask(key.key_value),
                "created_at": key.created_at,
                "last_used_at": key.last_used_at,
            }
            if key
            else None
        ),
        "chalybobs": {
            "full_access": full_access,
            "has_external": bool(ext),
            "connected": (await get_connection(ext)) if (ext and full_access) else False,
        },
        "streams": [
            {
                "id": st.id,
                "status": st.status,
                "title": st.title,
                "live_started_at": st.live_started_at,
                "live_ended_at": st.live_ended_at,
            }
            for st in streams[:50]
        ],
    }


@router.get("/live")
async def hub_live(tenant_id: str = hub_tenant(), db: Database = Depends(get_db)) -> dict[str, Any]:
    return await _live_view(db, tenant_id)


@router.post("/live/rotate-key")
async def hub_live_rotate(
    tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    """New key, old one revoked (dashboard.live_rotate_key)."""
    await LiveStreamKeysRepo(db).rotate_for_tenant()
    return await _live_view(db, tenant_id)


class LiveConnectionIn(BaseModel):
    enabled: bool


@router.post("/live/connection")
async def hub_live_connection(
    body: LiveConnectionIn, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    """The ChalyOBS ↔ Clips switch — full access only (dashboard.live_set_connection)."""
    from chalybclip.integrations.chalybobs import set_connection

    tenant = await TenantsRepo(db).get(tenant_id)
    full_access = bool(tenant and (tenant.tier or "").lower() == "all_access")
    if not (full_access and tenant and tenant.external_user_id):
        raise HubError("needs_plan", reason="paywall_top")
    await set_connection(tenant.external_user_id, body.enabled)
    return await _live_view(db, tenant_id)


# ── channel watches (dashboard /sources) ────────────────────────────────


def _watch_json(w: Any) -> dict[str, Any]:
    return {
        "id": w.id,
        "platform": w.platform,
        "channel_url": w.channel_url,
        "label": w.channel_label,
        "persona_id": w.persona_id,
        "language": w.language,
        "enabled": w.enabled,
        "polls_per_day": w.polls_per_day,
        "max_per_poll": w.max_per_poll,
        "seen_count": len(w.seen_video_ids or []),
        "last_polled_at": w.last_polled_at,
        "created_at": w.created_at,
    }


@router.get("/sources")
async def hub_sources(tenant_id: str = hub_tenant(), db: Database = Depends(get_db)) -> dict[str, Any]:
    watches = await ChannelWatchesRepo(db).list_for_tenant()
    personas = await PersonasRepo(db).list_for_tenant()
    return {
        "watches": [_watch_json(w) for w in watches],
        "personas": [{"id": p.id, "name": p.name} for p in personas],
    }


class SourceIn(BaseModel):
    channel_url: str = Field(min_length=4, max_length=500)
    persona_id: str = Field(min_length=1, max_length=64)
    platform: str = Field(default="", max_length=20)
    language: str = Field(default="", max_length=16)
    label: str = Field(default="", max_length=120)
    max_per_poll: int = Field(default=3, ge=1, le=25)
    polls_per_day: int = Field(default=1, ge=1, le=96)


@router.post("/sources")
async def hub_sources_add(
    body: SourceIn, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    """dashboard.sources_add — a duplicate channel is a 409, not a 500."""
    from chalybclip.db import IntegrityViolation
    from chalybclip.ingest.service import detect_platform

    if await PersonasRepo(db).get(body.persona_id) is None:
        raise HubError("bad_request", "Esa persona no existe.", reason="unknown_persona")
    url = body.channel_url.strip()
    resolved = body.platform.strip() or detect_platform(url)
    if resolved not in ("youtube", "twitch", "kick"):
        raise HubError(
            "bad_request", "Solo canales de YouTube, Twitch o Kick.", reason="link_unsupported"
        )
    try:
        w = await ChannelWatchesRepo(db).create(
            platform=resolved,
            channel_url=url,
            persona_id=body.persona_id,
            channel_label=body.label.strip() or None,
            language=body.language.strip() or None,
            max_per_poll=body.max_per_poll,
            polls_per_day=body.polls_per_day,
        )
    except IntegrityViolation:
        raise HubError(
            "conflict", "Ya estás vigilando ese canal.", reason="already_watching"
        ) from None
    await EventsRepo(db).emit(
        type="channel_watch.created", payload={"channel_url": url, "platform": resolved}
    )
    return {"watch": _watch_json(w)}


class SourcePatch(BaseModel):
    enabled: bool | None = None
    polls_per_day: int | None = Field(default=None, ge=1, le=96)


@router.patch("/sources/{watch_id}")
async def hub_sources_patch(
    watch_id: str, body: SourcePatch, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    """Pause/resume (dashboard.sources_toggle) and cadence (sources_schedule)."""
    repo = ChannelWatchesRepo(db)
    if await repo.get(watch_id) is None:
        raise HubError("not_found")
    if body.enabled is not None:
        await repo.set_enabled(watch_id, body.enabled)
    if body.polls_per_day is not None:
        await repo.set_polls_per_day(watch_id, max(1, min(body.polls_per_day, 96)))
    w = await repo.get(watch_id)
    return {"watch": _watch_json(w)}


@router.delete("/sources/{watch_id}")
async def hub_sources_delete(
    watch_id: str, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    repo = ChannelWatchesRepo(db)
    if await repo.get(watch_id) is None:
        raise HubError("not_found")
    await repo.delete(watch_id)
    return {"deleted": True}


@router.post("/sources/{watch_id}/poll")
async def hub_sources_poll(
    watch_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Scan now, ignoring the cadence (dashboard.sources_poll_now)."""
    from chalybclip.channels import make_channel_ingest_callback, poll_channel_watches

    if await ChannelWatchesRepo(db).get(watch_id) is None:
        raise HubError("not_found")
    app_db = request.app.state.db
    callback = make_channel_ingest_callback(
        app_db,
        request.app.state.job_dispatcher,
        output_dir=Path(get_settings().default_output_dir),
    )

    async def _run() -> None:
        await poll_channel_watches(
            app_db,
            ingest_callback=callback,
            tenant_id=tenant_id,
            watch_id=watch_id,
            respect_schedule=False,
        )

    background_tasks.add_task(_run)
    return {"scanning": True}


# ── personas (dashboard /personas) ──────────────────────────────────────


def _persona_json(p: Any) -> dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "primary_language": p.primary_language,
        "target_languages": list(p.target_languages or []),
        "voice_prompt": p.voice_prompt,
        "routing_tags": list(p.routing_tags or []),
        "created_at": p.created_at,
    }


@router.get("/personas")
async def hub_personas(tenant_id: str = hub_tenant(), db: Database = Depends(get_db)) -> dict[str, Any]:
    return {"personas": [_persona_json(p) for p in await PersonasRepo(db).list_for_tenant()]}


class PersonaIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    primary_language: str = Field(min_length=2, max_length=16)
    voice_prompt: str = Field(min_length=1, max_length=8000)
    target_languages: list[str] = Field(default_factory=list, max_length=20)
    routing_tags: list[str] = Field(default_factory=list, max_length=40)


class PersonaCreate(PersonaIn):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")


def _clean_list(values: list[str]) -> list[str]:
    return [v.strip()[:40] for v in values if isinstance(v, str) and v.strip()]


@router.post("/personas")
async def hub_personas_create(
    body: PersonaCreate, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    repo = PersonasRepo(db)
    if await repo.get(body.id) is not None:
        raise HubError("conflict", "Ya existe una persona con ese id.", reason="exists")
    await repo.create(
        persona_id=body.id,
        name=body.name,
        primary_language=body.primary_language,
        voice_prompt=body.voice_prompt,
        target_languages=_clean_list(body.target_languages),
        routing_tags=_clean_list(body.routing_tags),
    )
    return {"persona": _persona_json(await repo.get(body.id))}


@router.put("/personas/{persona_id}")
async def hub_personas_update(
    persona_id: str, body: PersonaIn, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    repo = PersonasRepo(db)
    if await repo.get(persona_id) is None:
        raise HubError("not_found")
    await repo.upsert(
        persona_id=persona_id,
        name=body.name,
        primary_language=body.primary_language,
        voice_prompt=body.voice_prompt,
        target_languages=_clean_list(body.target_languages),
        routing_tags=_clean_list(body.routing_tags),
    )
    return {"persona": _persona_json(await repo.get(persona_id))}


# ── brand kits (dashboard /brand-kits) ──────────────────────────────────


def _kit_logo_svg(kit_id: str) -> str | None:
    from chalybclip.api.routers.dashboard import _brand_kit_asset_dir

    path = _brand_kit_asset_dir(Path(get_settings().default_output_dir), kit_id) / "logo.svg"
    try:
        return path.read_text("utf-8") if path.exists() else None
    except OSError:
        return None


def _kit_json(kit: Any, *, full: bool = False) -> dict[str, Any]:
    from chalybclip.branding import caption_style_or_default
    from chalybclip.safety.policy import preset_for_overrides

    phrases = kit.custom_trigger_phrases
    out: dict[str, Any] = {
        "id": kit.id,
        "name": kit.name,
        "is_default": kit.is_default,
        "primary_color": kit.primary_color,
        "accent_color": kit.accent_color,
        "default_layout": kit.default_layout,
        "forward_count": len(phrases.forward),
        "retroactive_count": len(phrases.retroactive),
        "has_logo": bool(kit.logo_url),
    }
    if full:
        style = caption_style_or_default(kit.caption_style)
        out.update(
            {
                "handles": {
                    "tiktok": kit.handle_tiktok,
                    "youtube": kit.handle_youtube,
                    "instagram": kit.handle_instagram,
                    "kick": kit.handle_kick,
                },
                "caption_preset": style.preset,
                "caption_style": {
                    "font_family": style.font_family,
                    "font_weight": style.font_weight,
                    "font_size_px": style.font_size_px,
                    "position": style.position,
                    "animation": style.animation,
                    "censor_swears": bool(getattr(style, "censor_swears", False)),
                },
                "forward_phrases": list(phrases.forward),
                "retroactive_phrases": list(phrases.retroactive),
                "safe_schedule_enabled": kit.safe_schedule_enabled,
                "content_timezone": kit.content_timezone,
                "safety_preset": preset_for_overrides(kit.safety_policy),
                "show_chalybclip_outro": kit.show_chalybclip_outro,
                "logo_svg": _kit_logo_svg(kit.id) if kit.logo_url else None,
                "ai_prompt": kit.ai_prompt,
            }
        )
    return out


@router.get("/brand-kits")
async def hub_brand_kits(
    request: Request, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    from chalybclip.branding import preset_choices
    from chalybclip.safety.policy import SAFETY_PRESETS

    return {
        "kits": [_kit_json(k) for k in await BrandKitsRepo(db).list_for_tenant()],
        "caption_presets": [p for p, _label in preset_choices()],
        "safety_presets": list(SAFETY_PRESETS),
        "tier": getattr(request.state, "tenant_tier", "free"),
    }


@router.get("/brand-kits/{kit_id}")
async def hub_brand_kit(
    kit_id: str, request: Request, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    kit = await BrandKitsRepo(db).get(kit_id)
    if kit is None:
        raise HubError("not_found")
    return {"kit": _kit_json(kit, full=True), "tier": getattr(request.state, "tenant_tier", "free")}


class BrandKitIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    handle_tiktok: str = Field(default="", max_length=100)
    handle_youtube: str = Field(default="", max_length=100)
    handle_instagram: str = Field(default="", max_length=100)
    handle_kick: str = Field(default="", max_length=100)
    caption_preset: str = Field(default="karaoke_pop", max_length=40)
    forward_phrases: list[str] = Field(default_factory=list, max_length=200)
    retroactive_phrases: list[str] = Field(default_factory=list, max_length=200)
    safe_schedule_enabled: bool = False
    content_timezone: str = Field(default="UTC", max_length=64)
    safety_preset: str = Field(default="recommended", max_length=20)
    show_chalybclip_outro: bool = True
    is_default: bool = False


def _phrases(values: list[str]) -> list[str]:
    return [v.strip()[:200] for v in values if isinstance(v, str) and v.strip()]


def _validate_kit(body: BrandKitIn, tier: str) -> dict[str, Any]:
    from chalybclip.branding import preset_choices
    from chalybclip.branding.captions import _preset_by_id
    from chalybclip.safety.policy import SAFETY_PRESETS, policy_overrides_for_preset

    if body.caption_preset not in {p for p, _ in preset_choices()}:
        raise HubError("bad_request", reason="caption_preset")
    if body.safety_preset not in SAFETY_PRESETS:
        raise HubError("bad_request", reason="safety_preset")
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(body.content_timezone or "UTC")
    except Exception:
        raise HubError("bad_request", reason="timezone") from None
    return {
        "caption_style": _preset_by_id(body.caption_preset).model_dump(),  # type: ignore[arg-type]
        "safety_policy": policy_overrides_for_preset(body.safety_preset),
        # Free tier keeps the end card (the renderer forces it on as well).
        "outro": True if tier == "free" else body.show_chalybclip_outro,
    }


@router.post("/brand-kits")
async def hub_brand_kit_create(
    body: BrandKitIn, request: Request, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    """dashboard.brand_kits_create — colors/fonts/layout keep their defaults
    (the dashboard form never sends them either)."""
    v = _validate_kit(body, getattr(request.state, "tenant_tier", "free"))
    repo = BrandKitsRepo(db)
    kit = await repo.create(
        name=body.name,
        primary_color="#FF3366",
        accent_color="#FFD700",
        is_default=body.is_default,
        handle_tiktok=body.handle_tiktok.strip() or None,
        handle_youtube=body.handle_youtube.strip() or None,
        handle_instagram=body.handle_instagram.strip() or None,
        handle_kick=body.handle_kick.strip() or None,
        caption_style=v["caption_style"],
        custom_trigger_phrases=CustomTriggerPhrases(
            forward=_phrases(body.forward_phrases),
            retroactive=_phrases(body.retroactive_phrases),
        ),
        safe_schedule_enabled=body.safe_schedule_enabled,
        safety_policy=v["safety_policy"],
        content_timezone=body.content_timezone or "UTC",
        show_chalybclip_outro=v["outro"],
    )
    return {"kit": _kit_json(kit, full=True)}


@router.put("/brand-kits/{kit_id}")
async def hub_brand_kit_update(
    kit_id: str,
    body: BrandKitIn,
    request: Request,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """dashboard.brand_kits_edit_submit (fields its form sends)."""
    repo = BrandKitsRepo(db)
    if await repo.get(kit_id) is None:
        raise HubError("not_found")
    v = _validate_kit(body, getattr(request.state, "tenant_tier", "free"))
    await repo.update(
        kit_id,
        name=body.name,
        caption_style=v["caption_style"],
        handle_tiktok=body.handle_tiktok.strip(),
        handle_youtube=body.handle_youtube.strip(),
        handle_instagram=body.handle_instagram.strip(),
        handle_kick=body.handle_kick.strip(),
        custom_trigger_phrases=CustomTriggerPhrases(
            forward=_phrases(body.forward_phrases),
            retroactive=_phrases(body.retroactive_phrases),
        ),
        safe_schedule_enabled=body.safe_schedule_enabled,
        content_timezone=body.content_timezone or "UTC",
        safety_policy=v["safety_policy"],
        show_chalybclip_outro=v["outro"],
    )
    return {"kit": _kit_json(await repo.get(kit_id), full=True)}


@router.post("/brand-kits/{kit_id}/default")
async def hub_brand_kit_default(
    kit_id: str, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    from chalybclip.errors import ChalybClipError

    try:
        await BrandKitsRepo(db).set_default(kit_id)
    except ChalybClipError:
        raise HubError("not_found") from None
    return {"ok": True}


@router.delete("/brand-kits/{kit_id}")
async def hub_brand_kit_delete(
    kit_id: str, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    repo = BrandKitsRepo(db)
    if await repo.get(kit_id) is None:
        raise HubError("not_found")
    await repo.delete(kit_id)
    return {"deleted": True}


class LogoIn(BaseModel):
    style_hint: str = Field(default="Minimal mark / monogram", max_length=200)


@router.post("/brand-kits/{kit_id}/logo")
async def hub_brand_kit_logo(
    kit_id: str, body: LogoIn, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    """Generate a logo with the LLM (dashboard.brand_kits_generate_logo):
    sanitized SVG saved next to the kit, PNG best-effort, kit marked
    ai_generated. The kit isn't touched unless both the call and the write
    succeed."""
    from chalybclip.api.routers.dashboard import _brand_kit_asset_dir
    from chalybclip.branding import generate_logo, rasterize_svg_to_png
    from chalybclip.llm import LLMRouter, load_llm_config

    repo = BrandKitsRepo(db)
    kit = await repo.get(kit_id)
    if kit is None:
        raise HubError("not_found")
    output_dir = Path(get_settings().default_output_dir)
    asset_dir = _brand_kit_asset_dir(output_dir, kit_id)
    asset_dir.mkdir(parents=True, exist_ok=True)
    style = body.style_hint.strip() or "Minimal mark / monogram"
    router_ = LLMRouter(
        config=load_llm_config(),
        call_log_path=output_dir / "llm_calls_brand_kits.jsonl",
        db=db,
    )
    try:
        logo = await generate_logo(
            tenant_id=tenant_id,
            brand_name=kit.name,
            primary_color=kit.primary_color,
            accent_color=kit.accent_color,
            style=style,
            router=router_,
        )
    except Exception as e:
        _log.warning("hub.logo_failed", kit_id=kit_id, error=str(e)[:200])
        raise HubError(
            "unavailable", "No pudimos generar el logo. Inténtalo de nuevo.", reason="logo_failed"
        ) from e
    (asset_dir / "logo.svg").write_text(logo.svg, encoding="utf-8")
    rasterize_svg_to_png(logo.svg, output_path=asset_dir / "logo.png", size_px=1024)
    await repo.update(
        kit_id,
        logo_url=f"/dashboard/brand-kits/{kit_id}/logo.svg",
        ai_generated=True,
        ai_prompt=style,
        ai_provider="anthropic",
    )
    return {"kit": _kit_json(await repo.get(kit_id), full=True)}


# ── calibration (dashboard /calibration) ────────────────────────────────


@router.get("/calibration")
async def hub_calibration(tenant_id: str = hub_tenant(), db: Database = Depends(get_db)) -> dict[str, Any]:
    from chalybclip.metrics import compute_calibration

    reports = []
    for platform in ("youtube", "tiktok", "instagram", "buffer"):
        r = await compute_calibration(db, platform=platform)
        reports.append(
            {
                "platform": r.platform,
                "pearson_r": r.pearson_r,
                "n_paired": r.n_paired,
                "rows": [
                    {
                        "clip_id": row.clip_id,
                        "rescore_score": row.rescore_score,
                        "views": row.views,
                        "likes": row.likes,
                        "fetched_at": row.fetched_at,
                    }
                    for row in r.rows
                ],
            }
        )
    return {"reports": reports}


# ── admin only ──────────────────────────────────────────────────────────


@router.get("/admin/llm-calls")
async def hub_llm_calls(
    request: Request, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    """Raw USD economics — engine admins only (dashboard.llm_calls_view)."""
    from chalybclip.cost import compute_cost_projection

    _require_admin(request)
    calls = await LLMCallsRepo(db).list_for_tenant(limit=200)
    p = await compute_cost_projection(db)
    return {
        "calls": [
            {
                "id": c.id,
                "ts": c.ts,
                "purpose": c.purpose,
                "provider": c.provider,
                "model": c.model,
                "quality": c.quality,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "cost_usd_micros": c.cost_usd_micros,
                "status": c.status,
                "attempts": c.attempts,
                "stream_id": c.stream_id,
            }
            for c in calls
        ],
        "total_usd_micros": sum(c.cost_usd_micros for c in calls),
        "projection": {
            "today_micros": p.today_micros,
            "mtd_micros": p.mtd_micros,
            "projected_eom_micros": p.projected_eom_micros,
            "by_purpose": p.by_purpose,
            "by_model": p.by_model,
            "daily_budget_micros": p.daily_budget_micros,
            "budget_consumed_frac": p.budget_consumed_frac,
            "n_calls_mtd": p.n_calls_mtd,
        },
    }


@router.get("/admin/llm-settings")
async def hub_llm_settings(request: Request, tenant_id: str = hub_tenant()) -> dict[str, Any]:
    """Read-only view of config/llm.yaml (dashboard.llm_settings_view)."""
    from chalybclip.llm.config import load_llm_config

    _require_admin(request)
    cfg = load_llm_config()
    return {
        "providers": [
            {
                "name": name,
                "kind": p.kind,
                "standard": p.models.standard,
                "premium": getattr(p.models, "premium", None),
                "timeout_s": getattr(p, "timeout_s", None),
            }
            for name, p in cfg.providers.items()
        ],
        "routing": [
            {
                "purpose": purpose,
                "default_quality": rule.default_quality,
                "primary": rule.primary,
                "fallbacks": list(rule.fallbacks),
            }
            for purpose, rule in cfg.routing.items()
        ],
    }


_PRICE_KEYS = ("landing_price_es", "landing_price_en")


@router.get("/admin/site-settings")
async def hub_site_settings(
    request: Request, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    _require_admin(request)
    values = await PlatformSettingsRepo(db).get_many(list(_PRICE_KEYS))
    return {"price_es": values.get(_PRICE_KEYS[0], ""), "price_en": values.get(_PRICE_KEYS[1], "")}


class SiteSettingsIn(BaseModel):
    price_es: str = Field(default="", max_length=120)
    price_en: str = Field(default="", max_length=120)


@router.put("/admin/site-settings")
async def hub_site_settings_save(
    body: SiteSettingsIn,
    request: Request,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Landing price strings (dashboard.site_settings_save). Empty clears."""
    _require_admin(request)
    repo = PlatformSettingsRepo(db)
    await repo.set(_PRICE_KEYS[0], body.price_es.strip())
    await repo.set(_PRICE_KEYS[1], body.price_en.strip())
    return {"price_es": body.price_es.strip(), "price_en": body.price_en.strip()}

"""Hub API — the clip editor (/api/hub/v1/clips/{id}/…).

The Chalyb hub rebuilds the old `/dashboard/clips/{id}` editor as a React
screen. Everything here is a thin JSON face over the SAME handlers the
dashboard uses — no business rule is forked:

  * GET  editor            → `dashboard.clip_detail`'s computed context
                              (publishability, AI scores, decisions, …),
                              serialized, plus the editor's prefill chain
  * PUT  overlay           → `dashboard.clip_overlay_save`   (Guardar borrador)
  * POST finalize          → `dashboard.clip_overlay_finalize` (Aprobar y seguir)
  * POST reject            → `dashboard.clip_reject`
  * POST revert-trim       → `dashboard.clip_revert_trim`
  * POST hooks             → `dashboard.clip_generate_hooks`
  * POST prefs             → `dashboard.me_brand_kit_prefs`
  * GET  captions/waveform/intelligence → the dashboard JSON feeds
  * GET  growth            → `zernio.zernio_clip_growth_json`
  * POST render / GET render-status / POST render-retry / GET download
                            → the dashboard's background-render flow
  * GET  media / thumbnail → the dashboard media endpoints (Range-aware)

Auth: `hub_router()` binds the tenant from the hub's Bearer launch token
(see hub_api). Every repo read is tenant-filtered, so another tenant's clip
id is a 404. Dashboard URLs that leak into reused JSON bodies
(`download_url`, `retry_url`, `status_url`) are stripped: the hub never
calls /dashboard.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from chalybclip.api.deps import get_db
from chalybclip.api.hub_api import HubError, hub_router, hub_tenant
from chalybclip.api.routers import dashboard as dash
from chalybclip.db import ClipsRepo, Database, StreamsRepo

router = hub_router(tags=["hub-editor"])

# ---- small helpers ------------------------------------------------------

_QUALITIES = ("1080", "2k", "4k")
_TONES = ("default", "aggressive", "gen_z", "corporate", "curious")
_URL_KEYS = ("download_url", "retry_url", "status_url")

# Banner platform colors — the editor flips the banner color to these when
# the operator changes platform (clip_detail.html `platform_colors`).
PLATFORM_COLORS: dict[str, str] = {
    "kick": "#53FC18",
    "twitch": "#9146FF",
    "tiktok": "#000000",
    "youtube": "#FF0000",
    "instagram": "#E1306C",
    "x": "#000000",
    "twitter": "#000000",
}
_BANNER_PLATFORMS = ("kick", "twitch", "tiktok", "youtube", "instagram")
_BANNER_VARIANTS = (
    "kick_repost_page",
    "kick_black_bar_classic",
    "kick_green_block",
    "kick_minimal_url",
)
_CAPTION_POSITIONS = ("upper_third", "centered", "lower_third", "bottom")
_CAPTION_SIZES = ("small", "medium", "large", "xl")
_CAPTION_ANIMATIONS = ("pop", "slide", "typewriter", "fade")
_TOP_HOOK_STYLES = ("white_rounded", "black_solid", "subtle")
_FIT_MODES = ("smart_crop", "blurred_bg", "letterbox", "dynamic_zoom")
_TARGETS = ("tiktok", "reels", "shorts", "kick", "all")


def _quality(raw: str | None) -> str:
    q = (raw or "1080").strip().lower()
    if q not in _QUALITIES:
        raise HubError("bad_request", "Calidad no válida.", reason="bad_quality")
    return q


async def _require_clip(db: Database, clip_id: str) -> Any:
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HubError("not_found", "No encontramos este clip.", reason="clip_not_found")
    return clip


def _strip_urls(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in _URL_KEYS}


def _json_body(resp: Response) -> dict[str, Any]:
    try:
        data = json.loads(bytes(resp.body or b"{}"))
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {"value": data}


def _redirect_target(resp: Response) -> tuple[list[str], dict[str, list[str]]]:
    """Split a dashboard RedirectResponse's Location into path parts + query.
    The dashboard handlers encode their outcome in the redirect (next clip,
    trim flash); we translate it to JSON instead of re-deriving it."""
    loc = resp.headers.get("location", "")
    parts = urlsplit(loc)
    return [p for p in parts.path.split("/") if p], parse_qs(parts.query)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if hasattr(value, "_asdict"):  # NamedTuple
        return {k: _jsonable(v) for k, v in value._asdict().items()}
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    return str(value)


# ---- the editor's prefill chain (clip_detail.html, lines ~2800-2890) -----


def _pick(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """A <select> whose value isn't one of its options shows the first
    option; mirror that so the hub opens on what the old editor showed."""
    return value if isinstance(value, str) and value in allowed else default


def editor_form(
    clip: Any,
    inherited_overlay: dict[str, Any] | None,
    brand_kit: Any,
    caption_style: Any,
) -> dict[str, Any]:
    """The control values the old editor rendered on load.

    Fallback chain (Slice H.1): this clip's overlay_config → the newest
    edited sibling's (inherited_overlay) → brand-kit user prefs → defaults.
    Mirrors the Jinja block at the top of clip_detail.html field by field.
    """
    ov: dict[str, Any] = dict(clip.overlay_config or inherited_overlay or {})
    banner = dict(ov.get("banner") or {})
    captions = dict(ov.get("captions") or {})
    comments = dict(ov.get("comments") or {})
    top_hook = dict(ov.get("top_hook") or {})
    bk = brand_kit
    if bk is not None:
        if "enabled" not in banner and bk.banner_enabled_default:
            banner["enabled"] = True
        if "show_context" not in banner and bk.banner_show_context_default:
            banner["show_context"] = True
        if "show_safezones" not in banner and bk.banner_show_safezones_default:
            banner["show_safezones"] = True
        if not banner.get("platform") and bk.default_platform:
            banner["platform"] = bk.default_platform
        if not banner.get("variant") and bk.bottom_banner_style:
            banner["variant"] = bk.bottom_banner_style
        if "live_badge" not in banner and bk.banner_live_badge_default:
            banner["live_badge"] = True
        if "enabled" not in top_hook and bk.top_hook_enabled_default:
            top_hook["enabled"] = True
        if not top_hook.get("style") and bk.top_hook_style_default:
            top_hook["style"] = bk.top_hook_style_default
        bks = bk.caption_style or {}
        for key in ("position", "font_size", "animation"):
            if not captions.get(key) and bks.get(key):
                captions[key] = bks.get(key)
        if "lead_ms" not in captions and bks.get("lead_ms") is not None:
            captions["lead_ms"] = bks.get("lead_ms")

    cs = caption_style
    platform = _pick(banner.get("platform") or "kick", _BANNER_PLATFORMS, "kick")
    target = ov.get("target_platform") or (bk.target_platform if bk else None) or "all"
    preview_mode = ov.get("preview_mode") or "edited"
    if preview_mode not in ("raw", "edited"):
        preview_mode = "edited"
    try:
        lead_ms = int(captions.get("lead_ms", 120))
    except (TypeError, ValueError):
        lead_ms = 120
    try:
        likes = int(comments.get("fake_likes", 8))
    except (TypeError, ValueError):
        likes = 8
    return {
        "target_platform": _pick(target, _TARGETS, "all"),
        "clip_style": ov.get("clip_style")
        or (bk.clip_style if bk is not None and bk.clip_style else "repost_page_viral"),
        "title_text": ov.get("title_text") or "",
        "captions_enabled": bool(captions.get("enabled", True)),
        "captions_preset": captions.get("preset") or "",
        "captions_highlight_color": captions.get("highlight_color") or cs.highlight_color,
        "captions_position": _pick(captions.get("position") or cs.position, _CAPTION_POSITIONS, ""),
        "captions_font_size": _pick(captions.get("font_size") or "medium", _CAPTION_SIZES, "small"),
        "captions_animation": _pick(
            captions.get("animation") or cs.animation or "pop", _CAPTION_ANIMATIONS, "pop"
        ),
        "captions_lead_ms": max(0, min(500, lead_ms)),
        "banner_enabled": bool(banner.get("enabled", False)),
        "banner_platform": platform,
        "banner_url": banner.get("url") or "",
        "banner_color": banner.get("color") or PLATFORM_COLORS.get(platform, "#53FC18"),
        "banner_show_context": bool(banner.get("show_context", False)),
        "banner_show_safezones": bool(banner.get("show_safezones", False)),
        "banner_variant": _pick(
            banner.get("variant") or "kick_repost_page", _BANNER_VARIANTS, "kick_repost_page"
        ),
        "banner_live_badge": bool(banner.get("live_badge", False)),
        "platform_overlay_preview": ov.get("platform_overlay_preview") or "clean",
        "safe_zone_platform": ov.get("safe_zone_platform") or "tiktok",
        "preview_mode": preview_mode,
        "fit_mode": _pick(ov.get("fit_mode") or "smart_crop", _FIT_MODES, "smart_crop"),
        "top_hook_enabled": bool(top_hook.get("enabled", False)),
        "top_hook_text": top_hook.get("text") or "",
        "top_hook_style": _pick(
            top_hook.get("style") or "white_rounded", _TOP_HOOK_STYLES, "white_rounded"
        ),
        "comments_show": bool(comments.get("show_overlay", False)),
        "comments_fake_likes": max(0, likes),
    }


def _intel_lines(ctx: dict[str, Any]) -> list[str]:
    """The ✨ lines on the verdict card (clip_detail.html ~3003-3032)."""
    lines: list[str] = []
    dec = ctx.get("ai_decisions")
    dec_lines = list(getattr(dec, "intel_lines", None) or [])
    lines.extend(dec_lines)
    face_zone = ctx.get("face_zone")
    face_in_dec = "Face" in " ".join(dec_lines)
    if face_zone == "top" and (dec is None or not face_in_dec):
        lines.append("Tu cara va arriba del cuadro — el hook se bajó para no taparla")
    elif face_zone == "lower" and (dec is None or not face_in_dec):
        lines.append("Tu cara va abajo del cuadro — los subtítulos se subieron para no taparla")
    b = ctx["breakdown"]
    if b.heuristic_reason and b.heuristic_reason != "unknown":
        lines.append("Detectamos: " + b.heuristic_reason.replace("_", " "))
    if b.reaction_confidence is not None and b.reaction_confidence >= 0.6:
        lines.append("Hay una reacción fuerte en pantalla justo en el momento clave")
    if b.face_presence is not None and b.face_presence >= 0.55:
        lines.append(f"Se te ve en pantalla el {int(b.face_presence * 100)}% del clip")
    return lines[:2]


def _default_handle(bk: Any) -> str:
    if bk is None:
        return ""
    for attr in ("handle_kick", "handle_tiktok", "handle_instagram", "handle_youtube"):
        v = getattr(bk, attr, None)
        if v:
            return str(v)
    return ""


async def _next_clip_id(db: Database, clip: Any) -> str | None:
    """The next clip on the same stream still waiting for review — the same
    rule `clip_overlay_finalize` uses for "Aprobar y seguir"."""
    for c in sorted(
        await ClipsRepo(db).list_for_stream(clip.stream_id), key=lambda x: x.created_at
    ):
        if c.id != clip.id and c.status in ("cut", "ready_for_review"):
            return str(c.id)
    return None


def _export_qualities(request: Request) -> list[str]:
    tier = getattr(request.state, "tenant_tier", "free") or "free"
    return sorted(dash._TIER_RESOLUTIONS.get(tier, {"1080"}), key=_QUALITIES.index)


# ---- GET editor ---------------------------------------------------------


@router.get("/clips/{clip_id}/editor")
async def hub_clip_editor(
    request: Request,
    clip_id: str,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Everything the editor screen renders, computed by the dashboard's own
    `clip_detail` handler (same verdict, same cache write, same fallbacks)."""
    await _require_clip(db, clip_id)
    page = await dash.clip_detail(request, clip_id, tenant_id=tenant_id, db=db)
    ctx: dict[str, Any] = getattr(page, "context", None) or {}
    clip = ctx["clip"]
    bk = ctx.get("brand_kit")
    stream = await StreamsRepo(db).get(clip.stream_id)
    pub = ctx["publishability"]
    outcomes = [
        {
            "platform": o["job"].platform,
            "status": o["job"].status,
            "external_url": getattr(o["job"], "external_url", None),
            "external_id": getattr(o["job"], "external_id", None),
            "views": getattr(o["metric"], "views", None) if o["metric"] else None,
            "likes": getattr(o["metric"], "likes", None) if o["metric"] else None,
            "comments": getattr(o["metric"], "comments", None) if o["metric"] else None,
            "fetched_at": getattr(o["metric"], "fetched_at", None) if o["metric"] else None,
        }
        for o in ctx.get("outcomes") or []
    ]
    return {
        "clip": {
            "id": clip.id,
            "stream_id": clip.stream_id,
            "stream_title": (stream.title if stream else None) or "",
            "status": clip.status,
            "duration_s": float(clip.duration_s),
            "start_s": float(clip.start_s),
            "end_s": float(clip.end_s),
            "original_start_s": clip.original_start_s,
            "original_end_s": clip.original_end_s,
            "has_overlay": bool(clip.overlay_config),
            "has_thumbnail": bool(clip.thumbnail_frame_path),
            "render_state": clip.render_state,
            "created_at": clip.created_at,
        },
        "can_finalize": clip.status in ("cut", "ready_for_review", "approved"),
        "can_reject": clip.status != "rejected",
        "next_clip_id": await _next_clip_id(db, clip),
        "form": editor_form(clip, ctx.get("inherited_overlay"), bk, ctx["caption_style"]),
        "default_handle": _default_handle(bk),
        "caption_style": _jsonable(ctx["caption_style"]),
        "publishability": _jsonable(pub),
        "intel_lines": _intel_lines(ctx),
        "director_lines": _jsonable(ctx.get("director_lines") or []),
        "ai_scores": _jsonable(ctx.get("ai_scores")),
        "breakdown": _jsonable(ctx["breakdown"]),
        "ai_decisions": _jsonable(ctx.get("ai_decisions")),
        "auto_corrections": _jsonable(ctx.get("auto_corrections")),
        "candidate_trigger": _jsonable(ctx.get("candidate_trigger")),
        "face_zone": ctx.get("face_zone"),
        "outcomes": outcomes,
        "choices": {
            "caption_presets": [p[0] for p in ctx.get("caption_preset_choices") or []],
            "clip_styles": [s[0] for s in ctx.get("clip_style_choices") or []],
            "targets": [t[0] for t in ctx.get("platform_target_choices") or []],
            "platform_sims": [p[0] for p in ctx.get("platform_choices") or []],
            "zone_specs": _jsonable(ctx.get("platform_zone_specs") or {}),
            "export_qualities": _export_qualities(request),
        },
    }


# ---- overlay save / finalize ---------------------------------------------


class OverlayIn(BaseModel):
    """The editor's controls — the same fields the old form posted, typed.
    `dashboard._parse_overlay_form` still does the canonical normalization."""

    model_config = ConfigDict(extra="ignore")

    title_text: str = Field(default="", max_length=500)
    banner_enabled: bool = False
    banner_platform: str = Field(default="kick", max_length=32)
    banner_url: str = Field(default="", max_length=300)
    banner_color: str = Field(default="", max_length=32)
    banner_show_context: bool = False
    banner_show_safezones: bool = False
    captions_enabled: bool = False
    captions_preset: str = Field(default="", max_length=40)
    captions_highlight_color: str = Field(default="", max_length=32)
    captions_position: str = Field(default="", max_length=32)
    captions_font_size: str = Field(default="", max_length=16)
    captions_animation: str = Field(default="", max_length=16)
    captions_lead_ms: int = Field(default=120, ge=0, le=500)
    clip_style: str = Field(default="", max_length=40)
    banner_variant: str = Field(default="", max_length=40)
    banner_live_badge: bool = False
    top_hook_enabled: bool = False
    top_hook_text: str = Field(default="", max_length=120)
    top_hook_style: str = Field(default="", max_length=32)
    platform_overlay_preview: str = Field(default="", max_length=32)
    safe_zone_platform: str = Field(default="", max_length=32)
    preview_mode: str = Field(default="", max_length=16)
    target_platform: str = Field(default="", max_length=16)
    comments_show: bool = False
    comments_fake_likes: int = Field(default=0, ge=0, le=999_999)

    def form_kwargs(self) -> dict[str, Any]:
        """As the HTML form submitted them: checkboxes are "1" or absent."""
        out: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            out[name] = ("1" if value else "") if isinstance(value, bool) else value
        return out


@router.put("/clips/{clip_id}/overlay")
async def hub_clip_overlay_save(
    request: Request,
    clip_id: str,
    body: OverlayIn,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Guardar borrador: save the overlay (+ brand-kit mirror), drop cached
    renders so the next download matches."""
    await _require_clip(db, clip_id)
    await dash.clip_overlay_save(request, clip_id, **body.form_kwargs(), tenant_id=tenant_id, db=db)
    clip = await _require_clip(db, clip_id)
    return {"ok": True, "status": clip.status, "overlay": clip.overlay_config}


@router.post("/clips/{clip_id}/finalize")
async def hub_clip_finalize(
    request: Request,
    background_tasks: BackgroundTasks,
    clip_id: str,
    body: OverlayIn,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Aprobar y seguir: save + approve + background 1080 pre-render +
    auto-publish hook, then point at the next clip waiting for review."""
    clip = await _require_clip(db, clip_id)
    resp = await dash.clip_overlay_finalize(
        request,
        background_tasks,
        clip_id,
        **body.form_kwargs(),
        tenant_id=tenant_id,
        db=db,
    )
    parts, _query = _redirect_target(resp)
    next_id = parts[2] if len(parts) >= 3 and parts[1] == "clips" else None
    fresh = await _require_clip(db, clip_id)
    return {
        "ok": True,
        "status": fresh.status,
        "stream_id": clip.stream_id,
        "next_clip_id": next_id,
    }


@router.post("/clips/{clip_id}/reject")
async def hub_clip_reject(
    request: Request,
    clip_id: str,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Descartar y cerrar."""
    clip = await _require_clip(db, clip_id)
    await dash.clip_reject(request, clip_id, tenant_id=tenant_id, db=db)
    fresh = await _require_clip(db, clip_id)
    return {"ok": True, "status": fresh.status, "stream_id": clip.stream_id}


@router.post("/clips/{clip_id}/revert-trim")
async def hub_clip_revert_trim(
    request: Request,
    clip_id: str,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Deshacer the pipeline's auto-trim. outcome: reverted | noop | error."""
    await _require_clip(db, clip_id)
    resp = await dash.clip_revert_trim(request, clip_id, tenant_id=tenant_id, db=db)
    _parts, query = _redirect_target(resp)
    outcome = (query.get("trim") or ["error"])[0]
    if outcome not in ("reverted", "noop"):
        outcome = "error"
    return {"ok": outcome != "error", "outcome": outcome}


# ---- AI hooks + brand-kit prefs -----------------------------------------


class HooksIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tone: str = "default"
    n: int = Field(default=5, ge=1, le=10)


@router.post("/clips/{clip_id}/hooks")
async def hub_clip_hooks(
    request: Request,
    clip_id: str,
    body: HooksIn,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Generar 5: hook titles (LLM, deterministic fallback when it's down)."""
    await _require_clip(db, clip_id)
    tone = body.tone if body.tone in _TONES else "default"
    resp = await dash.clip_generate_hooks(
        request, clip_id, tone=tone, n=body.n, persona_id="", tenant_id=tenant_id, db=db
    )
    return _json_body(resp)


class PrefIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, max_length=40)
    value: bool | int | float | str | None = None


@router.post("/brand-kit/prefs")
async def hub_brand_kit_prefs(
    request: Request,
    body: PrefIn,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Editor auto-save of user-level setup into the default brand kit.
    The dashboard handler reads the same `{field, value}` JSON body and
    owns the field whitelist (unknown field → 400)."""
    if isinstance(body.value, str) and len(body.value) > 500:
        raise HubError("bad_request", "Valor demasiado largo.", reason="value_too_long")
    await dash.me_brand_kit_prefs(request, tenant_id=tenant_id, db=db)
    return {"ok": True}


# ---- read-only feeds -----------------------------------------------------


@router.get("/clips/{clip_id}/captions")
async def hub_clip_captions(
    clip_id: str, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> Response:
    await _require_clip(db, clip_id)
    return await dash.clip_captions(clip_id, tenant_id=tenant_id, db=db)


@router.get("/clips/{clip_id}/waveform")
async def hub_clip_waveform(
    clip_id: str, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> Response:
    await _require_clip(db, clip_id)
    return await dash.clip_waveform(clip_id, tenant_id=tenant_id, db=db)


@router.get("/clips/{clip_id}/intelligence")
async def hub_clip_intelligence(
    clip_id: str, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> Response:
    await _require_clip(db, clip_id)
    return await dash.clip_intelligence(clip_id, tenant_id=tenant_id, db=db)


@router.get("/clips/{clip_id}/growth")
async def hub_clip_growth(
    clip_id: str, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> dict[str, Any]:
    """Growth Score card; `{scored: false}` until the Growth Engine ran."""
    from chalybclip.api.routers.zernio import zernio_clip_growth_json

    await _require_clip(db, clip_id)
    resp = await zernio_clip_growth_json(clip_id, tenant_id=tenant_id, db=db)
    if resp.status_code != 200:
        return {"scored": False}
    data = _json_body(resp)
    data.pop("ok", None)
    return {"scored": True, **data}


# ---- render / download ---------------------------------------------------


def _render_state(resp: Response) -> dict[str, Any]:
    """Normalize the dashboard's download/render-status answers to
    `{state, progress_pct, error}` (no dashboard URLs)."""
    if not isinstance(resp, JSONResponse):
        # A file response = the cached MP4 is ready.
        return {"state": "ready", "progress_pct": 100, "error": None}
    data = _strip_urls(_json_body(resp))
    return {
        "state": data.get("state") or "idle",
        "progress_pct": int(data.get("progress_pct") or 0),
        "error": data.get("error"),
    }


@router.post("/clips/{clip_id}/render")
async def hub_clip_render(
    request: Request,
    background_tasks: BackgroundTasks,
    clip_id: str,
    quality: str = "1080",
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Start (or report) the export render at `quality`. Same path as the
    dashboard's Download button: cache hit → ready; idle → schedule;
    rendering → progress; failed → error (retry with render-retry)."""
    q = _quality(quality)
    await _require_clip(db, clip_id)
    resp = await dash.clip_download(
        request, background_tasks, clip_id, quality=q, tenant_id=tenant_id, db=db
    )
    if not isinstance(resp, JSONResponse):
        # Don't stream the file on a POST — the hub fetches GET download.
        return {"state": "ready", "progress_pct": 100, "error": None, "quality": q}
    return {**_render_state(resp), "quality": q}


@router.get("/clips/{clip_id}/render-status")
async def hub_clip_render_status(
    request: Request,
    clip_id: str,
    quality: str = "1080",
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    q = _quality(quality)
    await _require_clip(db, clip_id)
    resp = await dash.clip_render_status(request, clip_id, quality=q, tenant_id=tenant_id, db=db)
    return {**_render_state(resp), "quality": q}


@router.post("/clips/{clip_id}/render-retry")
async def hub_clip_render_retry(
    request: Request,
    clip_id: str,
    quality: str = "1080",
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    q = _quality(quality)
    await _require_clip(db, clip_id)
    await dash.clip_render_retry(request, clip_id, quality=q, tenant_id=tenant_id, db=db)
    return {"state": "idle", "progress_pct": 0, "error": None, "quality": q}


@router.get("/clips/{clip_id}/download")
async def hub_clip_download(
    request: Request,
    background_tasks: BackgroundTasks,
    clip_id: str,
    quality: str = "1080",
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> Response:
    """The rendered MP4 as an attachment once it's ready; 409 otherwise
    (the hub starts renders with POST render and polls render-status)."""
    q = _quality(quality)
    await _require_clip(db, clip_id)
    status = await dash.clip_render_status(request, clip_id, quality=q, tenant_id=tenant_id, db=db)
    if _render_state(status)["state"] != "ready":
        raise HubError(
            "conflict", "El video todavía se está preparando.", reason="render_not_ready"
        )
    resp = await dash.clip_download(
        request, background_tasks, clip_id, quality=q, tenant_id=tenant_id, db=db
    )
    if isinstance(resp, JSONResponse):
        raise HubError(
            "conflict", "El video todavía se está preparando.", reason="render_not_ready"
        )
    return resp


@router.api_route("/clips/{clip_id}/media", methods=["GET", "HEAD"])
async def hub_clip_media(
    request: Request,
    clip_id: str,
    tenant_id: str = hub_tenant(),
    db: Database = Depends(get_db),
) -> Response:
    """The cut MP4 (no burns) for the preview player. Range-aware; may
    redirect to the bucket copy when the local file was reclaimed."""
    await _require_clip(db, clip_id)
    return await dash.clip_media(request, clip_id, source="original", tenant_id=tenant_id, db=db)


@router.get("/clips/{clip_id}/thumbnail")
async def hub_clip_thumbnail(
    clip_id: str, tenant_id: str = hub_tenant(), db: Database = Depends(get_db)
) -> Response:
    await _require_clip(db, clip_id)
    return await dash.clip_thumbnail(clip_id, tenant_id=tenant_id, db=db)


__all__ = ["PLATFORM_COLORS", "OverlayIn", "editor_form", "router"]

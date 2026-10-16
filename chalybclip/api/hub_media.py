"""Signed media links for the hub: /api/hub/media/{kind}/{resource_id}.

The hub renders every Clips screen, but video and image bytes are too big
to proxy through it (Vercel caps function bodies). So hub API responses
carry SHORT-LIVED SIGNED PATHS to this engine instead; the hub turns them
into absolute URLs with the engine base it already calls, and the browser
loads them directly (<video src>, <img src>, download links).

    path = media_path("clip", clip.id, tenant_id=tenant_id)
    # → /api/hub/media/clip/clp_123?t=ten_1&exp=1700000000&sig=…

A signature binds (kind, resource_id, tenant, exp, extra params) with a key
derived from CHALYB_SSO_SECRET (domain-separated from launch tokens), so a
leaked link exposes one object for at most its TTL and can't be re-pointed
at another tenant's resource. The route binds that tenant, then serves the
local file (Range-capable) or 302s to the object-storage copy.

Kinds are a registry: core registers clip / thumb / source here; the editor
and publishing workstreams register their own (e.g. render) from their
modules with `@media_kind("name")`.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from chalybclip.db import Database
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

MEDIA_PREFIX = "/api/hub/media"
DEFAULT_TTL_S = 3600
# Upper bound on how far in the future a signed link may claim to expire —
# defense in depth if the key ever leaks.
MAX_TTL_S = 24 * 3600

MediaHandler = Callable[[Request, Database, str, str, dict[str, str]], Awaitable[Response]]
_KINDS: dict[str, MediaHandler] = {}


def media_kind(name: str) -> Callable[[MediaHandler], MediaHandler]:
    """Register a handler: `(request, db, tenant_id, resource_id, params)`.
    It runs with the tenant bound; raise HTTPException(404) when missing."""

    def deco(fn: MediaHandler) -> MediaHandler:
        if name in _KINDS and _KINDS[name] is not fn:
            raise ValueError(f"media kind {name!r} already registered")
        _KINDS[name] = fn
        return fn

    return deco


def _key() -> bytes:
    secret = (get_settings().chalyb_sso_secret or "").encode()
    if not secret:
        raise RuntimeError("CHALYB_SSO_SECRET not configured")
    return hmac.new(secret, b"chalybclip-hub-media-v1", hashlib.sha256).digest()


def _sig(kind: str, resource_id: str, tenant: str, exp: int, extra: dict[str, str]) -> str:
    extra_s = "&".join(f"{k}={extra[k]}" for k in sorted(extra))
    msg = f"{kind}|{resource_id}|{tenant}|{exp}|{extra_s}".encode()
    return hmac.new(_key(), msg, hashlib.sha256).hexdigest()


def media_path(
    kind: str,
    resource_id: str,
    *,
    tenant_id: str,
    ttl_s: int = DEFAULT_TTL_S,
    params: dict[str, str] | None = None,
    now: int | None = None,
) -> str:
    """Relative signed path for `kind/resource_id`. `params` (e.g.
    {"dl": "1"}) are signed too, so they can't be altered."""
    if kind not in _KINDS:
        raise ValueError(f"unknown media kind {kind!r}")
    extra = {k: str(v) for k, v in (params or {}).items()}
    exp = int(now if now is not None else time.time()) + min(int(ttl_s), MAX_TTL_S)
    sig = _sig(kind, resource_id, tenant_id, exp, extra)
    query = urlencode({**extra, "t": tenant_id, "exp": exp, "sig": sig})
    return f"{MEDIA_PREFIX}/{kind}/{resource_id}?{query}"


def verify_media(
    kind: str, resource_id: str, query: dict[str, str], *, now: int | None = None
) -> tuple[str, dict[str, str]]:
    """Return (tenant_id, extra params) or raise HTTPException(403/404)."""
    params = dict(query)
    tenant = params.pop("t", "")
    sig = params.pop("sig", "")
    exp_raw = params.pop("exp", "")
    try:
        exp = int(exp_raw)
    except ValueError:
        raise HTTPException(status_code=403, detail="bad link") from None
    current = int(now if now is not None else time.time())
    if not tenant or not sig or exp < current or exp > current + MAX_TTL_S:
        raise HTTPException(status_code=403, detail="link expired")
    try:
        expected = _sig(kind, resource_id, tenant, exp, params)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="not configured") from None
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=403, detail="bad link")
    return tenant, params


# ── core kinds ──────────────────────────────────────────────────────────


async def _bucket_redirect(key: str) -> Response | None:
    from fastapi.responses import RedirectResponse

    from chalybclip.api._artifacts import artifact_redirect_url

    url = await artifact_redirect_url(key)
    return RedirectResponse(url=url, status_code=302) if url else None


@media_kind("clip")
async def _clip_media(
    request: Request, db: Database, tenant_id: str, clip_id: str, params: dict[str, str]
) -> Response:
    """The cut original (no burned overlays) — same file the dashboard's
    /clips/{id}/media serves by default. `dl=1` → attachment."""
    from pathlib import Path

    from chalybclip.api._video_response import video_file_response
    from chalybclip.db import ClipsRepo
    from chalybclip.integrations.storage import clip_media_key

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="not found")
    original = Path(clip.path)
    final = original.parent / "clip_final.mp4"
    path = original if original.exists() else final
    download = params.get("dl") == "1"
    if path.exists():
        headers = (
            {"Content-Disposition": f'attachment; filename="chalybclip_{clip_id}.mp4"'}
            if download
            else None
        )
        return video_file_response(request, path, filename=None, headers=headers)
    redirect = await _bucket_redirect(clip_media_key(tenant_id, clip_id))
    if redirect is None:
        raise HTTPException(status_code=404, detail="not found")
    return redirect


@media_kind("thumb")
async def _clip_thumb(
    request: Request, db: Database, tenant_id: str, clip_id: str, params: dict[str, str]
) -> Response:
    from pathlib import Path

    from fastapi.responses import FileResponse

    from chalybclip.db import ClipsRepo
    from chalybclip.integrations.storage import clip_thumbnail_key

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None or not clip.thumbnail_frame_path:
        raise HTTPException(status_code=404, detail="not found")
    thumb = Path(clip.thumbnail_frame_path)
    if thumb.exists():
        return FileResponse(path=thumb, media_type="image/jpeg")
    redirect = await _bucket_redirect(clip_thumbnail_key(tenant_id, clip_id))
    if redirect is None:
        raise HTTPException(status_code=404, detail="not found")
    return redirect


@media_kind("source")
async def _stream_source(
    request: Request, db: Database, tenant_id: str, stream_id: str, params: dict[str, str]
) -> Response:
    """The stream's source video, while it is still on disk (retention
    reclaims it once processing finishes — the hub shows a note then)."""
    from pathlib import Path

    from chalybclip.api._video_response import video_file_response
    from chalybclip.db import StreamsRepo

    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="not found")
    src = Path(stream.source_video_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail="not found")
    return video_file_response(request, src, filename=None)


router = APIRouter(prefix=MEDIA_PREFIX, include_in_schema=False)


@router.api_route("/{kind}/{resource_id}", methods=["GET", "HEAD"])
async def hub_media(request: Request, kind: str, resource_id: str) -> Response:
    handler = _KINDS.get(kind)
    if handler is None:
        raise HTTPException(status_code=404, detail="not found")
    tenant_id, params = verify_media(kind, resource_id, dict(request.query_params))
    db: Database = request.app.state.db
    with bound_tenant(tenant_id):
        response = await handler(request, db, tenant_id, resource_id, params)
    response.headers.setdefault("Cache-Control", "private, max-age=300")
    return response


__all__ = ["MEDIA_PREFIX", "media_kind", "media_path", "router", "verify_media"]

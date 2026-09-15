"""Internal endpoints — slice O.44 + publisher (Zernio) fetch target.

Both routes here share an HMAC-signed-URL auth model that's separate
from the rest of the API:

  - No tenant cookie / bearer required.
  - URL is signed with CHALYBCLIP_INTERNAL_SIGNING_SECRET.
  - Signature binds (resource_id, tenant_id, expiry_unix_ts).
  - Expiry is bounded; anything past TTL 403s.

Why: external callers (Modal Whisper, Zernio) can't carry our
operator session cookie. Short-lived signed URLs are bounded — if
one leaks, the worst case is one media file is exposed for the
remaining TTL window. The signing secret never crosses the wire.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from chalybclip.db import ClipsRepo, Database, StreamsRepo
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

router = APIRouter(prefix="/api/internal", tags=["internal"], include_in_schema=False)

# A scheduled post fires at `scheduled_for`, then the publishing vendor's
# per-platform pipeline downloads the media minutes later — and clocks
# drift. The signed clip URL must outlive that whole tail, so we pad the
# gap-until-scheduled with this margin. Without it a URL minted now with a
# 1h TTL expires long before a post scheduled hours/days out is fetched,
# and the platform gets a 403 ("could not download the video").
_SCHEDULED_FETCH_MARGIN_S = 6 * 3600

# Verify-side ceiling for the signed CLIP route (distinct from the 24h
# audio ceiling). Scheduled publishing legitimately needs URLs that live
# until the post fires — best-time-slot spread can place posts several
# days out — so the clip route allows a longer plausible horizon. The
# leak tradeoff is small: the resource is a single short-form clip that's
# about to be published to the world anyway.
_SIGNED_CLIP_MAX_TTL_S = 35 * 24 * 3600

# Floor TTL for an IMMEDIATE ("now") publish. The signed URL is baked
# into the vendor (Zernio) post payload exactly once — it can't be
# refreshed afterwards — and the vendor fetches the media server-side on
# its own retry schedule, and re-fetches it again on every operator
# "Reintentar"/"Reagendar" days later. A 1h floor expired before a
# queued/rate-limited fetch retried; a 24h floor then *also* proved too
# short — a prod auto-publish run on 2026-06-16/17 kept getting YouTube
# "Failed to fetch ... 403" and TikTok "could not download" for ~10 DAYS
# as the vendor + reschedules retried against URLs that expired 24h in.
# Every retry past the floor 403s forever and the post never ships. We
# now floor at 14 days so the baked URL outlives the full retry +
# reschedule tail. Same leak tradeoff as the route ceiling above (and
# well under it): the clip is about to be published to the world anyway.
_IMMEDIATE_PUBLISH_TTL_S = 14 * 24 * 3600


def signed_clip_ttl_for_schedule(
    scheduled_for: str | _dt.datetime | None,
    *,
    now: int | None = None,
    default_ttl_s: int = _IMMEDIATE_PUBLISH_TTL_S,
) -> int:
    """Pick a signed-clip-URL TTL that is still valid when the post is
    actually downloaded.

    For an immediate post the 24h default floor covers the vendor's
    server-side fetch + retry tail (see `_IMMEDIATE_PUBLISH_TTL_S`). For a
    SCHEDULED post,
    size the TTL to cover the gap until `scheduled_for` plus a margin for
    the per-platform pipeline + clock skew, clamped to the route ceiling.
    Accepts an ISO string or a datetime (naive is treated as UTC); an
    unparseable value, or one far enough in the past that gap+margin
    underflows, falls back to the default TTL."""
    if scheduled_for is None:
        return default_ttl_s
    when = scheduled_for
    if isinstance(when, str):
        try:
            when = _dt.datetime.fromisoformat(when)
        except ValueError:
            return default_ttl_s
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.UTC)
    now_ts = now if now is not None else int(time.time())
    ttl = int(when.timestamp()) - now_ts + _SCHEDULED_FETCH_MARGIN_S
    return max(default_ttl_s, min(ttl, _SIGNED_CLIP_MAX_TTL_S))


def _verify_signed_params(
    *,
    resource_id: str,
    tenant: str,
    exp: int,
    sig: str,
    max_ttl_s: int,
) -> None:
    """Shared HMAC + expiry check. Raises HTTPException on any failure.

    `max_ttl_s` upper-bounds the future expiry an attacker can claim
    — defense in depth against a leaked secret minting essentially-
    permanent URLs.
    """
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="server not configured for signed-URL access",
        )
    if not tenant or not exp or not sig:
        raise HTTPException(status_code=400, detail="missing signature params")

    now = int(time.time())
    if int(exp) < now:
        raise HTTPException(status_code=403, detail="signed URL expired")
    if int(exp) > now + max_ttl_s:
        raise HTTPException(status_code=403, detail="signed URL expiry implausible")

    msg = f"{resource_id}|{tenant}|{int(exp)}".encode()
    expected = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=403, detail="signature mismatch")


def mint_signed_clip_url(
    *,
    clip_id: str,
    tenant_id: str,
    base_url: str,
    ttl_seconds: int = 3600,
) -> str:
    """Helper used by the publish router to build the URL we hand to
    the publishing vendor (Zernio). The signature binds (clip_id,
    tenant_id, exp) so a leaked URL only exposes one clip for at most
    `ttl_seconds`."""
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise RuntimeError(
            "CHALYBCLIP_INTERNAL_SIGNING_SECRET is not configured; "
            "cannot mint signed clip URL for the publisher"
        )
    exp = int(time.time()) + int(ttl_seconds)
    msg = f"{clip_id}|{tenant_id}|{exp}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    base = base_url.rstrip("/")
    return (
        f"{base}/api/internal/clip/{clip_id}"
        f"?tenant={tenant_id}&exp={exp}&sig={sig}"
    )


def artifact_key_for_clip(tenant_id: str, clip_id: str) -> str:
    """Bucket key for a clip's burned-in publish render. Tenant-namespaced.

    Delegates to the shared key family in `chalybclip.integrations.storage`
    (Phase 2a) so the pipeline/retention and this router can never drift.
    """
    from chalybclip.integrations.storage import clip_render_key

    return clip_render_key(tenant_id, clip_id)


async def resolve_publish_media_url(
    *,
    clip_id: str,
    tenant_id: str,
    base_url: str,
    rendered_path: Path,
    ttl_seconds: int,
) -> str:
    """The media URL handed to the publisher (Zernio) for a clip.

    With object storage configured (Phase 1): upload the rendered MP4 to the
    bucket once, then return a STABLE public URL (or a presigned ≤7d url as a
    fallback). The vendor fetches a durable object — no load on our box, and
    no time-boxed local endpoint to 403 on its multi-day retry tail.

    Without object storage: the local signed-URL path (`mint_signed_clip_url`),
    unchanged — fully backward compatible.
    """
    from chalybclip.integrations.storage import build_artifact_store

    store = build_artifact_store(get_settings())
    if store is None:
        return mint_signed_clip_url(
            clip_id=clip_id, tenant_id=tenant_id, base_url=base_url,
            ttl_seconds=ttl_seconds,
        )
    key = artifact_key_for_clip(tenant_id, clip_id)
    if not await store.exists(key=key):
        await store.upload(
            local_path=rendered_path, key=key, content_type="video/mp4",
        )
    return store.public_url(key) or await store.presigned_url(
        key=key, ttl_seconds=ttl_seconds,
    )


def sign_render_query(
    *, clip_id: str, tenant_id: str, ttl_seconds: int = 600
) -> str:
    """Just the signed query string (`tenant=..&exp=..&sig=..`) for the
    `/render` page — the `auth_query` the background recorder appends to
    authenticate without a cookie. Same HMAC scheme as the signed clip URL."""
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise RuntimeError(
            "CHALYBCLIP_INTERNAL_SIGNING_SECRET is not configured; cannot sign "
            "a render URL for background auto-publish"
        )
    exp = int(time.time()) + int(ttl_seconds)
    msg = f"{clip_id}|{tenant_id}|{exp}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return f"tenant={tenant_id}&exp={exp}&sig={sig}"


def mint_signed_render_url(
    *,
    clip_id: str,
    tenant_id: str,
    base_url: str,
    ttl_seconds: int = 600,
) -> str:
    """Build a signed `/dashboard/clips/{id}/render` URL the headless
    recorder can open when there is NO operator cookie — i.e. auto-publish
    hands-free, which renders from the background pipeline.

    Same HMAC scheme as `mint_signed_clip_url` (binds clip_id, tenant_id,
    exp). Short TTL by default — the render starts within seconds. The
    `/render` handler verifies it via `_verify_signed_params` and binds
    that tenant, an alternative to the cookie path."""
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise RuntimeError(
            "CHALYBCLIP_INTERNAL_SIGNING_SECRET is not configured; cannot mint "
            "a signed render URL for background auto-publish"
        )
    exp = int(time.time()) + int(ttl_seconds)
    msg = f"{clip_id}|{tenant_id}|{exp}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    base = base_url.rstrip("/")
    return (
        f"{base}/dashboard/clips/{clip_id}/render"
        f"?capture=1&tenant={tenant_id}&exp={exp}&sig={sig}"
    )


@router.get("/audio/{stream_id}")
async def fetch_audio_for_transcribe(
    stream_id: str,
    request: Request,
    tenant: str = "",
    exp: int = 0,
    sig: str = "",
) -> FileResponse:
    """Serve the stream's source audio if (stream_id, tenant, exp) HMAC checks."""
    _verify_signed_params(
        resource_id=stream_id,
        tenant=tenant,
        exp=exp,
        sig=sig,
        max_ttl_s=24 * 3600,
    )

    # The dashboard binds tenants via middleware — for this admin-less
    # path we bind manually to the tenant claim in the signed URL.
    db: Database = request.app.state.db
    with bound_tenant(tenant):
        stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")

    audio_path = Path(stream.source_audio_path)
    if not audio_path.exists():
        raise HTTPException(
            status_code=410,
            detail=f"audio extract missing from disk: {audio_path}",
        )

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=f"chalybclip_{stream_id}.wav",
    )


# GET + HEAD: Zernio probes the media URL with HEAD before downloading
# it — a GET-only route 405s the probe. FileResponse handles HEAD
# natively (headers only, no body).
@router.api_route("/clip/{clip_id}", methods=["GET", "HEAD"])
async def fetch_clip_for_publisher(
    clip_id: str,
    request: Request,
    tenant: str = "",
    exp: int = 0,
    sig: str = "",
) -> FileResponse:
    """Serve the rendered clip MP4 to the publishing vendor (Zernio) or
    any caller with a valid signed URL.

    Used by the publish router: when we call Zernio's `POST /posts`,
    each `mediaItems[].url` points here. Zernio downloads the file,
    re-hosts it, then publishes to each target platform. TTL is wider
    than the audio path (1h ceiling + 24h cap) because the per-platform
    pipeline can take several minutes for a 4K clip across 5 platforms.

    Serves ONLY the FINAL rendered MP4 (overlays + libass captions
    burned in) from the 1080-cache path. It deliberately does NOT fall
    back to the raw pre-render source: that fallback was the bug where
    a published clip went out missing its hooks + captions while the
    operator's download had them. The publish path
    (`zernio._publish_clip` → `ensure_clip_rendered`) renders this file
    before it hands us the URL, so by the time Zernio fetches it's on
    disk; if it somehow isn't, we 409 so the post errors loudly rather
    than shipping the unedited clip.
    """
    _verify_signed_params(
        resource_id=clip_id,
        tenant=tenant,
        exp=exp,
        sig=sig,
        max_ttl_s=_SIGNED_CLIP_MAX_TTL_S,
    )

    db: Database = request.app.state.db
    with bound_tenant(tenant):
        clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    from chalybclip.api._render_validation import is_servable_cached_mp4

    rendered = Path(clip.path).parent / "clip_render_1080.mp4"
    # Gate on is_servable_cached_mp4 (size floor + ISO BMFF magic) so a
    # partial/0-byte file from an aborted encode is treated as "not
    # ready" rather than served as a corrupt download.
    if not is_servable_cached_mp4(rendered):
        raise HTTPException(
            status_code=409,
            detail=(
                "clip render not ready (hooks + captions not burned in "
                "yet). Re-publish from the dashboard — the render runs "
                "before the post is sent."
            ),
        )

    return FileResponse(
        path=rendered,
        media_type="video/mp4",
        filename=f"chalybclip_{clip_id}.mp4",
    )

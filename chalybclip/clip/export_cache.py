"""Bucket-backed cache for the editor's "Descargar video" export.

The export used to live only in the rendering instance's /tmp, so a status
poll or download that landed on another API instance never saw it. Now the
finished MP4 is uploaded to `clip_export_key(...)` and every instance reads
the same object.

An object is only served when its `fp` metadata matches the clip's current
fingerprint, so an edit after the export (overlay, trim, format) or a plan
change (watermark, end-card outro) is a cache miss without touching every place that edits a clip.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from chalybclip.db import Database
    from chalybclip.integrations.storage import ArtifactStore

_log = structlog.get_logger(__name__)

# Bump when the renderer's output changes for the same inputs, so exports
# rendered by older code stop being served.
EXPORT_RENDER_VERSION = "2"

# Download links are used right away; short-lived and never a cacheable public URL.
_REDIRECT_TTL_S = 3600

_FP_META = "fp"


def export_fingerprint(
    clip: Any, *, quality: str, outro: bool | None = None, watermark: bool | None = None
) -> str:
    """Identity of what an export of `clip` at `quality` would contain."""
    payload = {
        "v": EXPORT_RENDER_VERSION,
        "q": quality,
        "ov": getattr(clip, "overlay_config", None) or {},
        "start": getattr(clip, "start_s", None),
        "end": getattr(clip, "end_s", None),
        "dur": getattr(clip, "duration_s", None),
        "outro": outro,
        # Plan branding: the corner mark is free-tier only, so a plan change flips it.
        "wm": watermark,
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


async def current_fingerprint(db: Database, *, tenant_id: str, clip: Any, quality: str) -> str | None:
    """Fingerprint including the plan-branding decisions the renderer makes
    (corner watermark, end-card outro), decided by the same helpers. None when
    the plan can't be read: then nothing cached can be proven current."""
    from chalybclip.branding import (
        TierUnknownError,
        outro_enabled_for_clip,
        watermark_enabled_for_clip,
    )

    try:
        watermark = await watermark_enabled_for_clip(db, tenant_id=tenant_id)
        outro = await outro_enabled_for_clip(
            db, tenant_id=tenant_id, stream_id=getattr(clip, "stream_id", "") or ""
        )
    except TierUnknownError:
        return None
    return export_fingerprint(clip, quality=quality, outro=outro, watermark=watermark)


def export_download_name(clip_id: str) -> str:
    return f"chalybclip_{clip_id}.mp4"


async def upload_export(
    store: ArtifactStore,
    *,
    tenant_id: str,
    clip_id: str,
    quality: str,
    fingerprint: str,
    local_path: Any,
) -> None:
    """Upload a validated export; served as an attachment so the browser saves it."""
    from chalybclip.integrations.storage import clip_export_key

    await store.upload(
        local_path=local_path,
        key=clip_export_key(tenant_id, clip_id, resolution=quality),
        content_type="video/mp4",
        content_disposition=f'attachment; filename="{export_download_name(clip_id)}"',
        metadata={_FP_META: fingerprint},
    )


async def export_in_bucket(
    store: ArtifactStore | None, db: Database, *, tenant_id: str, clip: Any, quality: str
) -> bool:
    """True when the bucket holds an export of the clip's CURRENT state.
    Bucket errors other than not-found propagate."""
    if store is None:
        return False
    from chalybclip.integrations.storage import clip_export_key

    meta = await store.metadata(
        key=clip_export_key(tenant_id, clip.id, resolution=quality)
    )
    if meta is None:
        return False
    expected = await current_fingerprint(db, tenant_id=tenant_id, clip=clip, quality=quality)
    return expected is not None and meta.get(_FP_META) == expected


async def export_redirect_url(store: ArtifactStore, *, tenant_id: str, clip_id: str, quality: str) -> str:
    from chalybclip.integrations.storage import clip_export_key

    return await store.presigned_url(
        key=clip_export_key(tenant_id, clip_id, resolution=quality), ttl_seconds=_REDIRECT_TTL_S
    )


async def drop_export(store: ArtifactStore | None, *, tenant_id: str, clip_id: str, quality: str) -> None:
    if store is None:
        return
    from chalybclip.integrations.storage import clip_export_key

    await store.delete(key=clip_export_key(tenant_id, clip_id, resolution=quality))


__all__ = [
    "EXPORT_RENDER_VERSION",
    "current_fingerprint",
    "drop_export",
    "export_download_name",
    "export_fingerprint",
    "export_in_bucket",
    "export_redirect_url",
    "upload_export",
]

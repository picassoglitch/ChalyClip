"""Phase 2a — dashboard-side helpers for bucket-backed clip artifacts.

Two access shapes, chosen by what the caller does with the bytes:

- pure serving (`<video>`/`<img>` src): `artifact_redirect_url` — a 302 to
  the bucket's stable public URL (or a presigned one). No bytes through
  the web box, no R2 round-trip on the request path (the redirect is
  minted blind; a missing object 404s at the bucket exactly like a local
  miss 404s here).
- local processing (ffmpeg render, waveform): `rehydrate_clip` — pull the
  cut original back into `clip.path` so the tool can run against a real
  file.

Both are no-ops returning None/False when object storage isn't configured,
so local-only deploys behave exactly as before.
"""

from __future__ import annotations

from pathlib import Path

from chalybclip.settings import get_settings


async def artifact_redirect_url(key: str) -> str | None:
    """Browser-facing URL for a bucket object, or None when object storage
    is off. Prefers the stable public base; falls back to a presigned URL
    (capped at the S3 7-day limit by the store)."""
    from chalybclip.integrations.storage import build_artifact_store

    settings = get_settings()
    store = build_artifact_store(settings)
    if store is None:
        return None
    url = store.public_url(key)
    if url is not None:
        return url
    ttl = int(getattr(settings, "object_storage_presign_ttl_s", 3600) or 3600)
    return await store.presigned_url(key=key, ttl_seconds=ttl)


async def rehydrate_clip(
    *, tenant_id: str, clip_id: str, clip_path: Path
) -> bool:
    """Pull the cut original back onto local disk from the bucket when
    `clip_path` is missing. True iff the file exists locally afterwards."""
    from chalybclip.clip import ensure_local_clip
    from chalybclip.integrations.storage import build_artifact_store

    store = build_artifact_store(get_settings())
    return await ensure_local_clip(
        store, tenant_id=tenant_id, clip_id=clip_id, clip_path=clip_path
    )


__all__ = ["artifact_redirect_url", "rehydrate_clip"]

"""Bucket-key builders for clip artifacts (Phase 2a).

One key family per clip, tenant-namespaced under `clips/`:

    clips/<tenant_id>/<clip_id>/clip.mp4              — the cut original
    clips/<tenant_id>/<clip_id>/thumbnail.jpg          — inbox-card thumbnail
    clips/<tenant_id>/<clip_id>/clip_render_1080.mp4   — burned publish render

Lives here (not in `chalybclip/api`) so the pipeline and retention can build
keys without importing the API layer. `api/routers/internal.py` re-exports
`artifact_key_for_clip` for its existing callers.
"""

from __future__ import annotations


def clip_media_key(tenant_id: str, clip_id: str) -> str:
    """Key for the cut original `clip.mp4` (no burned overlays)."""
    return f"clips/{tenant_id}/{clip_id}/clip.mp4"


def clip_thumbnail_key(tenant_id: str, clip_id: str) -> str:
    """Key for the clip-card thumbnail JPEG."""
    return f"clips/{tenant_id}/{clip_id}/thumbnail.jpg"


def clip_render_key(tenant_id: str, clip_id: str, *, resolution: str = "1080") -> str:
    """Key for a burned-in publish render at `resolution` (1080/2k/4k)."""
    return f"clips/{tenant_id}/{clip_id}/clip_render_{resolution}.mp4"


def clip_key_family(tenant_id: str, clip_id: str) -> list[str]:
    """Every key retention must drop when the clip row is deleted."""
    return [
        clip_media_key(tenant_id, clip_id),
        clip_thumbnail_key(tenant_id, clip_id),
        clip_render_key(tenant_id, clip_id),
    ]


__all__ = [
    "clip_key_family",
    "clip_media_key",
    "clip_render_key",
    "clip_thumbnail_key",
]

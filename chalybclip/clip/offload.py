"""Phase 2a — R2 offload for clip artifacts.

After the cut (and auto-correct, which may re-cut the file), each clip's
`clip.mp4` + `thumbnail.jpg` uploads to the object store so the local
volume stops being the only copy. The dashboard serves from disk when the
file is warm and falls back to the bucket when it isn't — which is what
lets the pipeline run on a worker whose disk evaporates after the run
(Phase 2b), and lets retention treat local files as reclaimable cache.

Everything here is best-effort: an R2 hiccup logs and moves on. Local
serving still works while the file is on disk, and the publish path's
`resolve_publish_media_url` re-uploads its render independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from chalybclip.integrations.storage import (
    clip_media_key,
    clip_thumbnail_key,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chalybclip.integrations.storage import ArtifactStore

_log = structlog.get_logger(__name__)


async def offload_clip_artifacts(
    store: ArtifactStore,
    *,
    tenant_id: str,
    clips: Sequence[object],
    force: bool = False,
) -> int:
    """Upload each clip's MP4 + thumbnail to the bucket. Returns how many
    objects were uploaded (skips count as zero).

    Idempotent: an object already in the bucket is skipped unless `force`
    (a re-cut with `force=True` produces new bytes under the same key, so
    the caller passes it through). Per-clip failures log and continue —
    one bad upload must not break the batch or the pipeline.
    """
    uploaded = 0
    for clip in clips:
        clip_id = str(getattr(clip, "id", "") or "")
        if not clip_id:
            continue
        # The pipeline's Clip model calls it `thumbnail_path`; the DB row
        # (ClipRow) calls the same file `thumbnail_frame_path`. Accept
        # either so both shapes offload correctly.
        thumbnail = (
            getattr(clip, "thumbnail_path", None)
            or getattr(clip, "thumbnail_frame_path", None)
        )
        targets = [
            (
                getattr(clip, "path", None),
                clip_media_key(tenant_id, clip_id),
                "video/mp4",
            ),
            (
                thumbnail,
                clip_thumbnail_key(tenant_id, clip_id),
                "image/jpeg",
            ),
        ]
        for local, key, content_type in targets:
            path = Path(local) if local else None
            if path is None or not path.is_file():
                continue
            try:
                if not force and await store.exists(key=key):
                    continue
                await store.upload(
                    local_path=path, key=key, content_type=content_type
                )
                uploaded += 1
            except Exception as e:  # offload is best-effort
                _log.warning(
                    "clip.offload_failed",
                    clip_id=clip_id,
                    key=key,
                    error=str(e),
                )
    return uploaded


async def ensure_local_clip(
    store: ArtifactStore | None,
    *,
    tenant_id: str,
    clip_id: str,
    clip_path: Path,
) -> bool:
    """Guarantee `clip_path` (the cut original) exists locally, pulling the
    bucket copy back into place when the local file was reclaimed.

    Returns True when the file is on disk after the call. The byte-needing
    paths (waveform, download render, publish render) call this before
    running ffmpeg against `clip.path`; the pure-serving endpoints redirect
    to the bucket instead and never rehydrate.
    """
    if clip_path.is_file():
        return True
    if store is None:
        return False
    # Download to a unique temp name + atomic rename: two concurrent
    # byte-needing requests (waveform + download) for the same reclaimed
    # clip must never observe a torn half-written file.
    import os
    import uuid

    tmp = clip_path.with_name(
        f"{clip_path.name}.rehydrating-{uuid.uuid4().hex[:8]}"
    )
    got = await store.download(
        key=clip_media_key(tenant_id, clip_id), dest=tmp
    )
    if got is None:
        tmp.unlink(missing_ok=True)
        _log.warning(
            "clip.rehydrate_miss", clip_id=clip_id, dest=str(clip_path)
        )
        return False
    os.replace(tmp, clip_path)
    _log.info("clip.rehydrated", clip_id=clip_id, dest=str(clip_path))
    return True


__all__ = ["ensure_local_clip", "offload_clip_artifacts"]

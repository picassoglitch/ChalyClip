"""Reprocess the scheduled queue — cancel + reset so clips can be redone.

Clips scheduled under the old pipeline shipped plain (no hook/marca, no real
outro) and on the wrong cadence (flat drip). This module does the destructive
CLEANUP half of "reprocess the queue": cancel the live scheduled Zernio posts,
hard-delete their local records, and reset each source clip so a fresh
rulebook-aware schedule can pick it up —

  * status → 'approved' (re-eligible for the auto-program scheduler),
  * render cache invalidated (LOCAL file + object-storage key both removed) so
    the next render re-bakes the new overlays + the real outro instead of
    re-serving the stale plain render,
  * overlay_config re-stamped with the viral default (captions + freshly
    generated hook + source-credit marca from the stream's channel).

The RE-SCHEDULE half (render + plan + publish through the rulebook) is the
existing auto-program flow — the CLI invokes it after this, or the operator
clicks "Auto-programar todo". Kept here as cleanup-only so this module has no
dependency on the API layer (no import cycle).

Best-effort per clip: one clip's failure is logged and skipped, never aborts
the batch.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chalybclip.clip.overlay_defaults import default_overlay_config
from chalybclip.db import (
    ClipsRepo,
    Database,
    StreamsRepo,
    ZernioPublishesRepo,
)
from chalybclip.integrations.zernio.client import ZernioError
from chalybclip.settings import Settings
from chalybclip.tenancy import bound_tenant

_log = logging.getLogger("chalybclip.publish.reprocess")

# The burned-in publish render lives next to the source clip under this name
# (see chalybclip.api._clip_render.ensure_clip_rendered).
_RENDER_NAME = "clip_render_1080.mp4"


@dataclass
class ReprocessSummary:
    scheduled_found: int = 0
    canceled: int = 0
    cancel_failed: int = 0
    clips_reset: int = 0
    clip_ids: list[str] = field(default_factory=list)
    # Clips whose SOURCE file is gone (retention reclaimed it) — left fully
    # untouched (post still scheduled, render still cached) because a reset
    # could never re-render and would strand them with nothing servable.
    skipped_missing_source: list[str] = field(default_factory=list)
    dry_run: bool = False


async def _invalidate_render(clip: Any, tenant_id: str, settings: Settings) -> None:
    """Drop the cached render so the next one re-bakes overlays + outro: delete
    the LOCAL render file and the object-storage key (uploaded once with an
    exists() guard, so a stale key would re-serve the old plain render)."""
    with contextlib.suppress(Exception):
        src = Path(getattr(clip, "path", "") or "")
        if src.name:
            (src.parent / _RENDER_NAME).unlink(missing_ok=True)
    with contextlib.suppress(Exception):
        from chalybclip.api.routers.internal import artifact_key_for_clip
        from chalybclip.integrations.storage import build_artifact_store

        store = build_artifact_store(settings)
        if store is not None:
            await store.delete(key=artifact_key_for_clip(tenant_id, clip.id))


async def _restamp_overlay(db: Database, tenant_id: str, clip: Any) -> None:
    """Re-stamp the viral default overlay: a freshly generated hook + captions
    + a source-credit marca built from the clip's stream channel/platform."""
    from chalybclip.publish.compose import generate_hook_line

    hook = ""
    with contextlib.suppress(Exception):
        hook = await generate_hook_line(db, tenant_id, clip.id)
    platform = channel = None
    with bound_tenant(tenant_id):
        stream = await StreamsRepo(db).get(clip.stream_id)
    if stream is not None:
        platform = getattr(stream, "platform", None)
        channel = getattr(stream, "channel", None)
    cfg = default_overlay_config(platform=platform, channel=channel, hook=hook)
    with bound_tenant(tenant_id):
        await ClipsRepo(db).set_overlay_config(clip.id, overlay_config=cfg)


async def reprocess_scheduled_queue(
    db: Database,
    tenant_id: str,
    *,
    client: Any,
    settings: Settings,
    dry_run: bool = False,
) -> ReprocessSummary:
    """Cancel the tenant's scheduled posts and reset their clips for a fresh
    schedule. Returns a summary (and the reset clip ids, for the caller to
    re-schedule). `dry_run` reports what would happen and changes nothing."""
    pubs = ZernioPublishesRepo(db)
    clips_repo = ClipsRepo(db)
    with bound_tenant(tenant_id):
        rows = await pubs.list_for_tenant(limit=1000, status="scheduled")

    # Distinct source clips behind the scheduled posts (order-preserving).
    all_clip_ids: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r.clip_id and r.clip_id not in seen:
            seen.add(r.clip_id)
            all_clip_ids.append(r.clip_id)

    # Re-render feasibility gate BEFORE anything destructive: reprocessing
    # deletes the cached render AND the object-storage key, so a clip whose
    # source file retention already reclaimed (or whose row is gone) would
    # hard-fail ensure_clip_rendered later and be stranded with NOTHING
    # servable. Those clips (and their posts) are left completely untouched —
    # the stale-but-live scheduled post beats a dead clip.
    clip_ids: list[str] = []
    skipped: list[str] = []
    for clip_id in all_clip_ids:
        with bound_tenant(tenant_id):
            clip = await clips_repo.get(clip_id)
        src = Path(getattr(clip, "path", "") or "") if clip is not None else None
        if src is None or not src.exists():
            skipped.append(clip_id)
        else:
            clip_ids.append(clip_id)
    skip_set = set(skipped)
    scheduled_found = len(rows)  # every scheduled post found, incl. skipped
    rows = [r for r in rows if r.clip_id not in skip_set]

    summary = ReprocessSummary(
        scheduled_found=scheduled_found, clip_ids=clip_ids,
        skipped_missing_source=skipped, dry_run=dry_run,
    )
    if dry_run or not rows:
        return summary

    # 1. Cancel each scheduled post on Zernio (best-effort) + hard-delete the
    #    local record so the clip is re-schedulable.
    for r in rows:
        try:
            with contextlib.suppress(ZernioError):
                await client.delete_post(r.post_id)
            await pubs.delete(r.post_id)
            summary.canceled += 1
        except Exception as e:  # one cancel failing must not stop the rest
            summary.cancel_failed += 1
            _log.warning(
                "reprocess.cancel_failed tenant=%s post=%s err=%s",
                tenant_id, r.post_id, e,
            )

    # 2. Reset + re-stamp + invalidate each clip.
    for clip_id in clip_ids:
        try:
            with bound_tenant(tenant_id):
                clip = await clips_repo.get(clip_id)
            if clip is None:
                continue
            with bound_tenant(tenant_id):
                await clips_repo.update_status(clip_id, status="approved")
                await clips_repo.reset_render_state(clip_id)
            await _invalidate_render(clip, tenant_id, settings)
            await _restamp_overlay(db, tenant_id, clip)
            summary.clips_reset += 1
        except Exception as e:  # one clip failing must not stop the batch
            _log.warning(
                "reprocess.clip_reset_failed tenant=%s clip=%s err=%s",
                tenant_id, clip_id, e,
            )
    _log.info(
        "reprocess.done tenant=%s scheduled=%d canceled=%d failed=%d reset=%d "
        "skipped_missing_source=%d",
        tenant_id, summary.scheduled_found, summary.canceled,
        summary.cancel_failed, summary.clips_reset,
        len(summary.skipped_missing_source),
    )
    return summary

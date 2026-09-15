"""Retention sweeper — voice-markers spec slice E.1 / §9 locked defaults.

Daily cron deletes VOD, clip, and transcript artifacts past their
per-tenant retention windows. Hard-delete only — no soft-delete /
trash bin per spec.

System defaults (when a tenant's column is NULL):
    retention_vod_days        = 7
    retention_clip_days       = 90
    retention_transcript_days = 365

The raw source VOD is normally deleted the moment the pipeline finishes
(see `Settings.delete_source_on_completion`), so the 7-day VOD window is a
backstop for streams whose pipeline crashed before that cleanup. A stream's
row + per-stream dir are torn down only once no clips (90d) or transcripts
(365d) survive under it — reclaiming the source at the VOD window never
takes longer-lived artifacts down early.

Tenants override via `TenantsRepo.set_retention(...)`.

Run via:
    chalybclip retention sweep              # all tenants
    chalybclip retention sweep --tenant <id>
    chalybclip retention sweep --dry-run    # report only, no deletes

The sweeper is idempotent — re-running it after a partial failure picks
up where it left off (DB rows beyond the window get re-considered;
already-deleted files are skipped via `missing_ok=True`).
"""

from __future__ import annotations

from .service import (
    DEFAULT_RETENTION_CLIP_DAYS,
    DEFAULT_RETENTION_TRANSCRIPT_DAYS,
    DEFAULT_RETENTION_VOD_DAYS,
    RetentionPolicy,
    RetentionReport,
    reclaim_orphan_dirs,
    reclaim_sources_until_free,
    reclaim_stream_source,
    sweep_retention,
)

__all__ = [
    "DEFAULT_RETENTION_CLIP_DAYS",
    "DEFAULT_RETENTION_TRANSCRIPT_DAYS",
    "DEFAULT_RETENTION_VOD_DAYS",
    "RetentionPolicy",
    "RetentionReport",
    "reclaim_orphan_dirs",
    "reclaim_sources_until_free",
    "reclaim_stream_source",
    "sweep_retention",
]

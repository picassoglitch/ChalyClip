"""Retention sweeper — deletes clips, VODs and transcripts past the
tenant's plan window (free 7 days, paid 30). See `service.py`.

Run via:
    chalybclip retention sweep              # all tenants
    chalybclip retention sweep --tenant <id>
    chalybclip retention sweep --dry-run    # report only, no deletes

In prod a Cloud Run job (`chalybclip-retention-sweep`, chalyb Terraform)
runs it daily. Idempotent: a re-run picks up whatever is still past the
window.
"""

from __future__ import annotations

from .service import (
    RetentionPolicy,
    RetentionReport,
    reclaim_orphan_dirs,
    reclaim_sources_until_free,
    reclaim_stream_source,
    sweep_retention,
)

__all__ = [
    "RetentionPolicy",
    "RetentionReport",
    "reclaim_orphan_dirs",
    "reclaim_sources_until_free",
    "reclaim_stream_source",
    "sweep_retention",
]

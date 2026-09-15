"""ULID-based entity ID generation.

Per CLAUDE.md hard rule #7: IDs are ULIDs with a per-entity prefix.
ULIDs sort by creation time, which makes pagination and debugging nicer.
"""

from __future__ import annotations

from typing import Literal

from ulid import ULID

EntityKind = Literal[
    "str",
    "clp",
    "var",
    "job",
    "ten",
    "usr",
    "tok",
    "evt",
    "cnd",
    "llm",
    "acc",  # connected accounts (Buffer / future platforms)
    "pjb",  # publish jobs
    "whk",  # webhook subscriptions (Phase 2)
    "met",  # publish_metrics (Phase 3)
    "spk",  # speakers — persistent voice identities (voice-markers spec slice B.2)
    "vsp",  # vod_speakers — per-VOD speaker resolution
    "brk",  # brand_kits (voice-markers spec slice C.1)
    "drv",  # drive_watches (voice-markers spec slice E.4)
    "chw",  # channel_watches — auto-ingest from YouTube/Twitch/Kick channels
    "zev",  # zernio_events — inbound webhook rows missing payload.id (Hub phase 2)
    "hpj",  # hub_publish_jobs — internal service API publishes (Hub phase 3)
    "snp",  # zernio_publish_snapshots — daily per-post metrics (Hub phase 7)
    "bcl",  # zernio_broadcast_log — per-tenant daily broadcast cap (Hub phase 10)
]


def new_id(kind: EntityKind) -> str:
    """Mint a new prefixed ULID for the given entity kind."""
    return f"{kind}_{ULID()}"

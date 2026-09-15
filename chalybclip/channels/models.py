"""Types for channel auto-ingest."""

from __future__ import annotations

from typing import NamedTuple


class ChannelVOD(NamedTuple):
    """One video listed off a creator's channel (no media downloaded yet)."""

    video_id: str
    url: str
    title: str | None


class ChannelPollReport(NamedTuple):
    """Per-watch outcome from one channel-poll pass."""

    watch_id: str
    tenant_id: str
    channel_url: str
    videos_seen: int
    videos_ingested: int
    videos_failed: int
    skipped_disabled: bool
    # True when the watch was skipped because its scheduled cadence
    # (polls_per_day) hasn't elapsed yet — no yt-dlp call was made.
    skipped_not_due: bool = False

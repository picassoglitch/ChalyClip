"""Channel auto-ingest — watch YouTube / Twitch / Kick channels for new VODs.

Connect a creator's channel and the poller detects new uploads, ingests
each VOD, and runs the full pipeline (transcribe → detect → clip →
variants → auto-publish, the last gated by the safe trap). Mirrors the
`chalybclip.drive` folder-watch pattern.

Public surface:
  * `ChannelVOD`, `ChannelPollReport` — models
  * `list_channel_vods` — yt-dlp channel lister (flat, no download)
  * `poll_channel_watches` — the poll loop
  * `make_channel_ingest_callback` — production ingest → pipeline wiring
"""

from __future__ import annotations

from .ingest_bridge import make_channel_ingest_callback
from .models import ChannelPollReport, ChannelVOD
from .service import (
    ChannelIngestCallback,
    ChannelVODLister,
    list_channel_vods,
    poll_channel_watches,
)

__all__ = [
    "ChannelIngestCallback",
    "ChannelPollReport",
    "ChannelVOD",
    "ChannelVODLister",
    "list_channel_vods",
    "make_channel_ingest_callback",
    "poll_channel_watches",
]

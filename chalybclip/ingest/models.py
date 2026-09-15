"""Pydantic schemas for ingested streams."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Platform = Literal["kick", "twitch", "youtube", "upload", "unknown"]


class ChatMessage(BaseModel):
    """One chat message from the VOD's chat replay.

    `ts` is seconds since stream start (NOT wall-clock); the platform-
    specific fetcher normalizes whatever the source format uses.
    """

    model_config = ConfigDict(extra="ignore")

    ts: float = Field(ge=0.0)
    user: str
    text: str


class ChatReplay(BaseModel):
    """Sorted-by-ts chat messages for one stream."""

    stream_id: str
    tenant_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class Stream(BaseModel):
    """A VOD that has been downloaded and audio-extracted."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(description="ULID with `str_` prefix")
    tenant_id: str
    vod_url: str
    platform: Platform
    title: str | None = None
    channel: str | None = None
    duration_s: float = Field(ge=0.0)
    source_video_path: Path
    source_audio_path: Path
    source_chat_path: Path | None = None

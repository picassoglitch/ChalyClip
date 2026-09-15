"""Pydantic schemas for the Drive watcher.

`DriveFile` is the listing shape returned by the DriveClient protocol;
`DriveWatchRow` is the persisted row and lives canonically in
`chalybclip.db.models` (same module as every other repo row model). We
re-export it here so the drive package presents a clean self-contained
import surface — but the type identity is the DB one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chalybclip.db.models import DriveWatchRow

__all__ = ["DriveFile", "DriveWatchRow"]


class DriveFile(BaseModel):
    """One file listed from a Drive folder.

    Lean shape — only the fields the watcher actually needs. Real
    Google API responses carry far more (`mimeType`, `parents`,
    `webViewLink`, ...); we drop the rest at the protocol boundary.
    """

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    name: str
    mime_type: str
    # Drive uses RFC 3339 timestamps; we keep them as strings so the
    # poller can compare against `last_polled_at` without a tz round-trip.
    modified_time: str
    size_bytes: int | None = None

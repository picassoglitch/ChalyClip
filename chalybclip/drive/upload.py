"""DriveUploadClient protocol + a Fake impl for tests / local dev.

The mirror image of `protocol.DriveClient`: that one is the INPUT side
(watch a folder, download VODs to ingest). This is the OUTPUT side —
push a finished clip MP4 UP to the tenant's Drive (mid-tier "save each
clip to Drive" feature, task #31).

Same design discipline as the input side: the export service takes a
`DriveUploadClient` rather than calling the Google API directly, so:

  * Tests — `FakeDriveUploadClient` exercises the export service with
    zero network / OAuth.
  * Local dev — the Fake doubles as "copy the clip into a local
    folder", same code path as production-against-Google.
  * Future destinations — Dropbox, S3, a synced folder — each one impl
    of this protocol.

The real `GoogleDriveUploadClient` (deferred follow-up, needs a Google
Cloud OAuth app) implements `upload_file` via Drive's resumable upload
(`POST /upload/drive/v3/files?uploadType=resumable` to init, then
`PUT <session_uri>` with the bytes), authorized with the tenant's
Bearer access token. Token refresh reuses publish/oauth.py's pattern
(same `oauth2.googleapis.com/token` endpoint YouTube already uses).
Its token lifecycle is internal to the impl — protocol consumers never
see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DriveUploadResult:
    """What the destination returns after a successful upload."""

    file_id: str
    # A user-openable link, when the destination provides one (Google
    # returns webViewLink). None for destinations that don't, or when
    # the impl didn't request the field.
    web_view_link: str | None = None
    # Bytes the destination acknowledged storing. Lets the caller
    # cross-check against the local file size.
    size_bytes: int | None = None


class DriveUploadClient(Protocol):
    """One client per tenant Drive destination. Stateless beyond OAuth
    setup (the impl holds the tenant's access token internally)."""

    async def upload_file(
        self,
        *,
        local_path: Path,
        folder_id: str,
        name: str,
        mime_type: str = "video/mp4",
    ) -> DriveUploadResult:
        """Upload `local_path` into `folder_id`, named `name`. Returns
        the created file's id (+ link/size when available). May raise on
        network / quota / auth errors; the export service catches and
        records them per-clip so one failure never blocks the render."""
        ...


# ---- FakeDriveUploadClient — used by tests AND the dev "copy to folder" mode ----


class FakeDriveUploadClient:
    """A `DriveUploadClient` over an on-disk folder.

    `upload_file` copies the clip into `dest_dir/<folder_id>/<name>` and
    returns a stable fake file_id (SHA1 of the destination path), so the
    export service's dedup / record-keeping is exercised exactly as it
    would be against Google. No network, no OAuth.
    """

    def __init__(self, dest_dir: Path):
        self._dest_dir = Path(dest_dir).resolve()
        # Recorded uploads, for test assertions: list of (folder_id, name,
        # bytes). Lets a test confirm WHAT got pushed without re-reading
        # the filesystem.
        self.uploads: list[tuple[str, str, int]] = []

    async def upload_file(
        self,
        *,
        local_path: Path,
        folder_id: str,
        name: str,
        mime_type: str = "video/mp4",
    ) -> DriveUploadResult:
        import hashlib
        import shutil

        src = Path(local_path)
        if not src.is_file():
            raise FileNotFoundError(
                f"FakeDriveUploadClient: source missing: {src}"
            )
        target = self._dest_dir / folder_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        size = target.stat().st_size
        self.uploads.append((folder_id, name, size))
        file_id = "fakedrv_" + hashlib.sha1(
            str(target).encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:16]
        return DriveUploadResult(
            file_id=file_id,
            web_view_link=f"https://drive.fake/{file_id}",
            size_bytes=size,
        )


__all__ = ["DriveUploadClient", "DriveUploadResult", "FakeDriveUploadClient"]

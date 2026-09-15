"""DriveClient protocol + a Fake impl for tests / local dev.

The watcher (`service.poll_drive_watches`) takes a `DriveClient` rather
than calling the Google API directly. This keeps:

  * Tests — `FakeDriveClient` lets us drive the polling loop without any
    network access or OAuth flow.
  * Local dev — `FakeDriveClient` doubles as a "watch this directory on
    disk" stub: you can drop files into a folder and the same code path
    that production runs against Google picks them up.
  * Future sources — Dropbox, S3, a synced folder on the operator's
    machine. Each is one impl of the same protocol.

The real `GoogleDriveClient` (deferred follow-up) implements
`list_video_files` via `drive.files.list` (query
`mimeType contains 'video/' and '<folder>' in parents and trashed=false`)
and `download_file` via `drive.files.get_media`. Its refresh_token
lifecycle is internal to the impl — protocol consumers never see it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import DriveFile


class DriveClient(Protocol):
    """One client per drive_watches row. Stateless beyond OAuth setup."""

    async def list_video_files(
        self,
        *,
        folder_id: str,
        modified_after: str | None = None,
    ) -> list[DriveFile]:
        """Return every video file in `folder_id` modified after the
        given RFC 3339 timestamp (None = no filter)."""
        ...

    async def download_file(
        self,
        *,
        file_id: str,
        target_path: Path,
    ) -> None:
        """Stream `file_id` to disk at `target_path`. Parent dirs are
        created by the caller. May raise on network / quota errors;
        the watcher catches and logs them per-file."""
        ...


# ---- FakeDriveClient — used by tests AND by the dev "drop folder" mode ----


class FakeDriveClient:
    """A `DriveClient` over an on-disk folder.

    Each video file in `source_dir` is reported as a DriveFile. The
    file's `file_id` is the SHA1 of its path, which is stable across
    polls so the deduplication-by-seen-file-ids logic works the same
    as it would against Google.

    Downloads = filesystem copies. The watcher's downstream
    `ingest_callback` doesn't care where the file actually came from.
    """

    def __init__(self, source_dir: Path):
        self._source_dir = Path(source_dir).resolve()

    async def list_video_files(
        self,
        *,
        folder_id: str,
        modified_after: str | None = None,
    ) -> list[DriveFile]:
        # We don't honor folder_id in the fake — there's exactly one
        # source dir. The poller still passes a folder_id; we'd map
        # them per-tenant in a real multi-watch deployment.
        _ = folder_id
        if not self._source_dir.is_dir():
            return []
        out: list[DriveFile] = []
        for path in sorted(self._source_dir.iterdir()):
            if not path.is_file():
                continue
            if not _looks_like_video(path):
                continue
            mtime_iso = _mtime_iso(path)
            if modified_after is not None and mtime_iso <= modified_after:
                continue
            out.append(
                DriveFile(
                    file_id=_stable_id(path),
                    name=path.name,
                    mime_type=_mime_for(path),
                    modified_time=mtime_iso,
                    size_bytes=path.stat().st_size,
                )
            )
        return out

    async def download_file(
        self,
        *,
        file_id: str,
        target_path: Path,
    ) -> None:
        import shutil

        # Find the source path whose stable id matches.
        for path in self._source_dir.iterdir():
            if path.is_file() and _stable_id(path) == file_id:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target_path)
                return
        raise FileNotFoundError(
            f"FakeDriveClient: no file with id {file_id!r} in {self._source_dir}"
        )


# ---- helpers ----


_VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".flv", ".avi"})


def _looks_like_video(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTS


def _mime_for(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return f"video/{ext}" if ext else "application/octet-stream"


def _mtime_iso(path: Path) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(
        path.stat().st_mtime, tz=_dt.UTC
    ).isoformat()


def _stable_id(path: Path) -> str:
    """Deterministic file id derived from the absolute path.

    Drive returns its own opaque ids; we mimic that shape so the
    deduplication logic in the poller is exercised the same way."""
    import hashlib

    return "fake_" + hashlib.sha1(
        str(path.resolve()).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]

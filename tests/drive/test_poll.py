"""Drive poll loop — slice E.4.

Drives the polling loop end-to-end through `FakeDriveClient` (one file
per .mp4 in a source dir). Asserts:

  * Each new file is downloaded + handed to the ingest callback.
  * `seen_file_ids` is appended; re-polling doesn't reingest.
  * `last_polled_at` is updated.
  * Disabled watches are skipped without touching the DB.
  * A failing download is reported (`files_failed`) but the rest of
    the batch continues.
  * Per-tenant isolation: polling Alice doesn't touch Bob's folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.db import (
    Database,
    DriveWatchesRepo,
    TenantsRepo,
)
from chalybclip.db.models import DriveWatchRow
from chalybclip.drive import (
    DriveClient,
    FakeDriveClient,
    PollReport,
    poll_drive_watches,
)
from chalybclip.drive.models import DriveFile
from chalybclip.tenancy import bound_tenant

# ---- ingest callback collector ----


class _IngestRecorder:
    """Captures (tenant_id, target_path, original_name) for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, str]] = []

    async def __call__(
        self, tenant_id: str, local_path: Path, original_name: str
    ) -> None:
        self.calls.append((tenant_id, local_path, original_name))


# ---- helpers ----


def _put_video(folder: Path, name: str, size_bytes: int = 1024) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"\x00" * size_bytes)
    return path


async def _create_watch(
    db: Database, *, tenant_id: str, folder_id: str
) -> DriveWatchRow:
    await TenantsRepo(db).create(tenant_id=tenant_id, name=f"T-{tenant_id}")
    with bound_tenant(tenant_id):
        return await DriveWatchesRepo(db).create(
            folder_id=folder_id, folder_name=None, refresh_token="rt"
        )


# ---- FakeDriveClient behavioral tests ----


async def test_fake_client_lists_only_videos(tmp_path: Path) -> None:
    _put_video(tmp_path, "a.mp4")
    _put_video(tmp_path, "b.mov")
    (tmp_path / "notes.txt").write_text("not a video", encoding="utf-8")
    client = FakeDriveClient(tmp_path)
    files = await client.list_video_files(folder_id="ignored")
    names = {f.name for f in files}
    assert names == {"a.mp4", "b.mov"}


async def test_fake_client_honors_modified_after(tmp_path: Path) -> None:
    p1 = _put_video(tmp_path, "old.mp4")
    p2 = _put_video(tmp_path, "new.mp4")
    # Force p1 into the past so the cutoff actually separates them.
    import os

    past = p2.stat().st_mtime - 3600
    os.utime(p1, (past, past))

    # Cutoff between the two.
    cutoff = (
        p1.stat().st_mtime + (p2.stat().st_mtime - p1.stat().st_mtime) / 2
    )
    import datetime as _dt

    cutoff_iso = _dt.datetime.fromtimestamp(cutoff, tz=_dt.UTC).isoformat()
    client = FakeDriveClient(tmp_path)
    files = await client.list_video_files(
        folder_id="x", modified_after=cutoff_iso
    )
    names = {f.name for f in files}
    assert names == {"new.mp4"}


async def test_fake_client_download_copies_bytes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    payload = b"vid bytes" * 64
    p = src / "x.mp4"
    p.write_bytes(payload)
    client = FakeDriveClient(src)
    listed = (await client.list_video_files(folder_id="x"))[0]
    target = tmp_path / "out.mp4"
    await client.download_file(file_id=listed.file_id, target_path=target)
    assert target.read_bytes() == payload


# ---- poll_drive_watches ----


async def test_poll_no_watches_returns_empty(drive_db: Database, tmp_path: Path) -> None:
    out = await poll_drive_watches(
        drive_db,
        output_dir=tmp_path,
        drive_client_factory=lambda _w: FakeDriveClient(tmp_path),
        ingest_callback=_IngestRecorder(),
    )
    assert out == []


async def test_poll_ingests_each_new_file_once(
    drive_db: Database, tmp_path: Path
) -> None:
    src = tmp_path / "drive_folder"
    src.mkdir()
    _put_video(src, "stream_a.mp4")
    _put_video(src, "stream_b.mp4")

    watch = await _create_watch(
        drive_db, tenant_id="aldo", folder_id="ignored"
    )
    recorder = _IngestRecorder()

    reports = await poll_drive_watches(
        drive_db,
        output_dir=tmp_path / "out",
        drive_client_factory=lambda _w: FakeDriveClient(src),
        ingest_callback=recorder,
    )
    assert len(reports) == 1
    r = reports[0]
    assert r.files_seen == 2
    assert r.files_ingested == 2
    assert r.files_failed == 0
    assert len(recorder.calls) == 2
    # Each callback got a tenant_id + a real on-disk target file.
    for tid, path, name in recorder.calls:
        assert tid == "aldo"
        assert path.exists()
        assert name.endswith(".mp4")

    # Re-polling doesn't reingest (seen_file_ids persisted).
    recorder2 = _IngestRecorder()
    r2 = (
        await poll_drive_watches(
            drive_db,
            output_dir=tmp_path / "out",
            drive_client_factory=lambda _w: FakeDriveClient(src),
            ingest_callback=recorder2,
        )
    )[0]
    assert r2.files_ingested == 0
    assert recorder2.calls == []
    _ = watch  # marker that the row is still around


async def test_poll_picks_up_new_files_on_subsequent_run(
    drive_db: Database, tmp_path: Path
) -> None:
    """A second poll after a new file is dropped ingests just the delta."""
    import os
    import time

    src = tmp_path / "drive_folder"
    src.mkdir()
    _put_video(src, "first.mp4")

    await _create_watch(drive_db, tenant_id="aldo", folder_id="x")
    recorder = _IngestRecorder()
    await poll_drive_watches(
        drive_db,
        output_dir=tmp_path / "out",
        drive_client_factory=lambda _w: FakeDriveClient(src),
        ingest_callback=recorder,
    )
    # Wall-clock granularity on Windows mtime can collapse the
    # "first poll vs second file" boundary into the same second. Force
    # the second file's mtime well after `last_polled_at` was written.
    time.sleep(0.05)
    second = _put_video(src, "second.mp4")
    future = time.time() + 1.0
    os.utime(second, (future, future))

    reports = await poll_drive_watches(
        drive_db,
        output_dir=tmp_path / "out",
        drive_client_factory=lambda _w: FakeDriveClient(src),
        ingest_callback=recorder,
    )
    assert reports[0].files_ingested == 1
    names_ingested = [name for _tid, _p, name in recorder.calls]
    assert sorted(names_ingested) == ["first.mp4", "second.mp4"]


async def test_poll_skips_disabled_watches(
    drive_db: Database, tmp_path: Path
) -> None:
    src = tmp_path / "drive_folder"
    src.mkdir()
    _put_video(src, "x.mp4")

    watch = await _create_watch(
        drive_db, tenant_id="aldo", folder_id="x"
    )
    with bound_tenant("aldo"):
        await DriveWatchesRepo(drive_db).set_enabled(watch.id, False)
    recorder = _IngestRecorder()
    reports = await poll_drive_watches(
        drive_db,
        output_dir=tmp_path / "out",
        drive_client_factory=lambda _w: FakeDriveClient(src),
        ingest_callback=recorder,
    )
    assert reports[0].skipped_disabled is True
    assert reports[0].files_ingested == 0
    assert recorder.calls == []


async def test_poll_continues_when_download_fails(
    drive_db: Database, tmp_path: Path
) -> None:
    """If one file errors, the rest of the batch still proceeds and
    `seen_file_ids` records only the successes."""
    src = tmp_path / "drive_folder"
    src.mkdir()
    good = _put_video(src, "good.mp4")
    bad = _put_video(src, "bad.mp4")
    _ = good

    class _FlakyClient:
        def __init__(self, root: Path):
            self._inner = FakeDriveClient(root)

        async def list_video_files(
            self, *, folder_id: str, modified_after: str | None = None
        ) -> list[DriveFile]:
            return await self._inner.list_video_files(
                folder_id=folder_id, modified_after=modified_after
            )

        async def download_file(self, *, file_id: str, target_path: Path) -> None:
            listed = await self._inner.list_video_files(folder_id="x")
            for f in listed:
                if f.file_id == file_id and f.name == "bad.mp4":
                    raise RuntimeError("network ka-boom")
            await self._inner.download_file(
                file_id=file_id, target_path=target_path
            )

    await _create_watch(drive_db, tenant_id="aldo", folder_id="x")
    recorder = _IngestRecorder()
    flaky: DriveClient = _FlakyClient(src)
    reports = await poll_drive_watches(
        drive_db,
        output_dir=tmp_path / "out",
        drive_client_factory=lambda _w: flaky,
        ingest_callback=recorder,
    )
    r = reports[0]
    assert r.files_seen == 2
    assert r.files_ingested == 1
    assert r.files_failed == 1
    assert len(recorder.calls) == 1
    # Re-polling tries the failed file again.
    reports2 = await poll_drive_watches(
        drive_db,
        output_dir=tmp_path / "out",
        drive_client_factory=lambda _w: FakeDriveClient(src),
        ingest_callback=recorder,
    )
    assert reports2[0].files_ingested == 1
    _ = bad


async def test_poll_isolated_per_tenant(drive_db: Database, tmp_path: Path) -> None:
    """tenant_id='alice' doesn't list bob's watches."""
    alice_src = tmp_path / "alice"
    bob_src = tmp_path / "bob"
    alice_src.mkdir()
    bob_src.mkdir()
    _put_video(alice_src, "a.mp4")
    _put_video(bob_src, "b.mp4")

    await _create_watch(drive_db, tenant_id="alice", folder_id="alice_folder")
    await _create_watch(drive_db, tenant_id="bob", folder_id="bob_folder")

    def factory(w: DriveWatchRow) -> DriveClient:
        return FakeDriveClient(
            alice_src if w.tenant_id == "alice" else bob_src
        )

    recorder = _IngestRecorder()
    reports = await poll_drive_watches(
        drive_db,
        output_dir=tmp_path / "out",
        drive_client_factory=factory,
        ingest_callback=recorder,
        tenant_id="alice",
    )
    assert len(reports) == 1
    assert reports[0].tenant_id == "alice"
    # Recorder only got alice's file.
    assert [name for _tid, _p, name in recorder.calls] == ["a.mp4"]


async def test_poll_list_failure_is_reported(
    drive_db: Database, tmp_path: Path
) -> None:
    """If the client raises on list, we still produce a PollReport
    with files_failed=1 and the watch is left untouched."""

    class _BrokenClient:
        async def list_video_files(
            self, *, folder_id: str, modified_after: str | None = None
        ) -> list[DriveFile]:
            raise RuntimeError("API down")

        async def download_file(self, *, file_id: str, target_path: Path) -> None:
            ...  # never reached

    await _create_watch(drive_db, tenant_id="aldo", folder_id="x")
    broken: DriveClient = _BrokenClient()
    reports = await poll_drive_watches(
        drive_db,
        output_dir=tmp_path / "out",
        drive_client_factory=lambda _w: broken,
        ingest_callback=_IngestRecorder(),
    )
    r = reports[0]
    assert r.files_seen == 0
    assert r.files_failed == 1
    assert r.files_ingested == 0
    assert isinstance(r, PollReport)


# Pytest plumbing — make sure deferred 'pytest' import in test fns lands.
_ = pytest

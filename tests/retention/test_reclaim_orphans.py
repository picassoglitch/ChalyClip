"""Orphan-dir sweep — reclaim_orphan_dirs (dirs on disk with no streams row)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from chalybclip.db import Database, StreamsRepo, TenantsRepo
from chalybclip.db.models import StreamRow
from chalybclip.jobs.active import pipeline_active
from chalybclip.retention import reclaim_orphan_dirs
from chalybclip.tenancy import bound_tenant

from .conftest import days_ago_iso


def _make_stream_dir(output_dir: Path, stream_id: str, *, age_s: float) -> Path:
    """Stream-shaped dir with one file, both backdated `age_s` seconds."""
    d = output_dir / stream_id / "source"
    d.mkdir(parents=True)
    f = d / "video.mp4"
    f.write_bytes(b"\x00" * 2048)
    old = time.time() - age_s
    os.utime(f, (old, old))
    os.utime(d, (old, old))
    os.utime(output_dir / stream_id, (old, old))
    return output_dir / stream_id


async def _seed_row(db: Database, *, tenant_id: str, stream_id: str) -> None:
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id=tenant_id,
                vod_url="https://kick.com/x/videos/1", platform="kick",
                title="t", channel="c", duration_s=60.0,
                source_video_path="", source_audio_path="",
                status="done", created_at=days_ago_iso(1),
            )
        )


async def test_deletes_orphan_keeps_row_backed(
    retention_db: Database, tmp_path: Path
) -> None:
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_row(retention_db, tenant_id=t.id, stream_id="str_tracked")
    tracked = _make_stream_dir(tmp_path, "str_tracked", age_s=7 * 86400)
    orphan = _make_stream_dir(tmp_path, "str_orphan", age_s=7 * 86400)

    deleted, freed = await reclaim_orphan_dirs(retention_db, output_dir=tmp_path)

    assert deleted == 1
    assert freed > 0
    assert not orphan.exists()
    assert tracked.exists()


async def test_spares_recently_written_orphan(
    retention_db: Database, tmp_path: Path
) -> None:
    """A fresh dir may be an in-flight ingest whose row hasn't landed (or
    whose row was deleted mid-download) — mtime grace must protect it."""
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_row(retention_db, tenant_id=t.id, stream_id="str_any")
    fresh = _make_stream_dir(tmp_path, "str_fresh_orphan", age_s=60)

    deleted, _ = await reclaim_orphan_dirs(retention_db, output_dir=tmp_path)

    assert deleted == 0
    assert fresh.exists()


async def test_refuses_when_streams_table_empty(
    retention_db: Database, tmp_path: Path
) -> None:
    """Empty table + populated volume = mispointed DATABASE_URL until proven
    otherwise. Deleting the whole volume on that signal would be catastrophic."""
    orphan = _make_stream_dir(tmp_path, "str_orphan", age_s=7 * 86400)

    deleted, freed = await reclaim_orphan_dirs(retention_db, output_dir=tmp_path)

    assert (deleted, freed) == (0, 0)
    assert orphan.exists()


async def test_spares_in_flight_run_without_row(
    retention_db: Database, tmp_path: Path
) -> None:
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_row(retention_db, tenant_id=t.id, stream_id="str_any")
    running = _make_stream_dir(tmp_path, "str_running", age_s=7 * 86400)

    with pipeline_active("str_running"):
        deleted, _ = await reclaim_orphan_dirs(retention_db, output_dir=tmp_path)

    assert deleted == 0
    assert running.exists()


async def test_ignores_non_stream_entries(
    retention_db: Database, tmp_path: Path
) -> None:
    """Only str_* dirs are candidates — the DB file, hf_cache, cookies.txt
    and stray files all live next to /data/out contents in prod."""
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_row(retention_db, tenant_id=t.id, stream_id="str_any")
    other_dir = tmp_path / "hf_cache"
    other_dir.mkdir()
    old = time.time() - 7 * 86400
    os.utime(other_dir, (old, old))
    stray = tmp_path / "str_looks_like_a_file"
    stray.write_bytes(b"\x00")
    os.utime(stray, (old, old))

    deleted, _ = await reclaim_orphan_dirs(retention_db, output_dir=tmp_path)

    assert deleted == 0
    assert other_dir.exists()
    assert stray.exists()

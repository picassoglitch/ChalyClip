"""Disk-budget enforcer — reclaim_sources_until_free (volume self-regulation)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chalybclip.db import Database, StreamsRepo, TenantsRepo
from chalybclip.db.models import StreamRow
from chalybclip.retention import reclaim_sources_until_free
from chalybclip.tenancy import bound_tenant

from .conftest import days_ago_iso


async def _seed_source(
    db: Database, *, tenant_id: str, stream_id: str, output_dir: Path, age_days: int
) -> Path:
    src = output_dir / stream_id / "source"
    src.mkdir(parents=True, exist_ok=True)
    video = src / "video.mp4"
    audio = src / "audio.wav"
    video.write_bytes(b"\x00" * 4096)
    audio.write_bytes(b"\x00" * 1024)
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id=tenant_id,
                vod_url="https://kick.com/x/videos/1", platform="kick",
                title="t", channel="c", duration_s=60.0,
                source_video_path=str(video), source_audio_path=str(audio),
                status="failed", created_at=days_ago_iso(age_days),
            )
        )
    return video


def _fake_usage(free_sequence: list[int]):
    """disk_usage stub returning the next free value each call (last repeats)."""
    seq = list(free_sequence)

    def _usage(_p: object) -> object:
        free = seq.pop(0) if len(seq) > 1 else seq[0]
        return shutil._ntuple_diskusage(100 * 1024**3, 0, free)

    return _usage


async def test_noop_when_already_above_target(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = await TenantsRepo(retention_db).create(name="Aldo")
    video = await _seed_source(
        retention_db, tenant_id=t.id, stream_id="str_a",
        output_dir=tmp_path, age_days=5,
    )
    monkeypatch.setattr(shutil, "disk_usage", _fake_usage([50 * 1024**3]))
    freed, reached = await reclaim_sources_until_free(
        retention_db, output_dir=tmp_path, target_free_bytes=10 * 1024**3,
    )
    assert (freed, reached) == (0, True)
    assert video.exists()  # nothing touched


async def test_evicts_oldest_first_and_stops_at_target(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = await TenantsRepo(retention_db).create(name="Aldo")
    old = await _seed_source(
        retention_db, tenant_id=t.id, stream_id="str_old",
        output_dir=tmp_path, age_days=10,
    )
    new = await _seed_source(
        retention_db, tenant_id=t.id, stream_id="str_new",
        output_dir=tmp_path, age_days=1,
    )
    # Low at the top check + first loop check, then plenty after one reclaim.
    monkeypatch.setattr(
        shutil, "disk_usage",
        _fake_usage([1024, 1024, 50 * 1024**3, 50 * 1024**3]),
    )
    freed, reached = await reclaim_sources_until_free(
        retention_db, output_dir=tmp_path, target_free_bytes=10 * 1024**3,
        active_grace_s=0.0,  # treat nothing as in-flight in this test
    )
    assert reached is True
    assert freed > 0
    assert not old.exists()  # oldest evicted
    assert new.exists()      # newest spared — target already met


async def test_skips_actively_downloading_source(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = await TenantsRepo(retention_db).create(name="Aldo")
    # Old row, but its source file was just written (active download).
    video = await _seed_source(
        retention_db, tenant_id=t.id, stream_id="str_live",
        output_dir=tmp_path, age_days=10,
    )
    monkeypatch.setattr(shutil, "disk_usage", _fake_usage([1024]))  # always low
    freed, reached = await reclaim_sources_until_free(
        retention_db, output_dir=tmp_path, target_free_bytes=10 * 1024**3,
        active_grace_s=600.0,
    )
    # In-flight source (fresh mtime) must NOT be yanked.
    assert video.exists()
    assert freed == 0
    assert reached is False


async def test_skips_active_hls_download_with_only_fragments(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an in-flight HLS/DASH download has only .part-Frag* files
    and NO video.mp4 yet. The reclaimer must NOT evict it — doing so deleted
    fragments mid-download and flooded yt-dlp 'Unable to rename ...part-Frag'.
    """
    t = await TenantsRepo(retention_db).create(name="Aldo")
    # Seed an OLD row whose source is mid-HLS-download: fragments, no video.mp4.
    sid = "str_hls"
    src = tmp_path / sid / "source"
    src.mkdir(parents=True)
    (src / "video.mp4.part-Frag2302.part").write_bytes(b"\x00" * 4096)
    (src / "video.mp4.part-Frag2303").write_bytes(b"\x00" * 4096)
    with bound_tenant(t.id):
        await StreamsRepo(retention_db).upsert(
            StreamRow(
                id=sid, tenant_id=t.id,
                vod_url="https://twitch.tv/videos/1", platform="twitch",
                title="t", channel="c", duration_s=0.0,
                source_video_path=str(src / "video.mp4"),  # doesn't exist yet
                source_audio_path=str(src / "audio.wav"),
                status="pending", created_at=days_ago_iso(30),
            )
        )
    monkeypatch.setattr(shutil, "disk_usage", _fake_usage([1024]))  # always low
    freed, _reached = await reclaim_sources_until_free(
        retention_db, output_dir=tmp_path, target_free_bytes=10 * 1024**3,
    )
    # The actively-downloading fragments survive.
    assert (src / "video.mp4.part-Frag2302.part").exists()
    assert (src / "video.mp4.part-Frag2303").exists()
    assert freed == 0


async def test_skips_stream_with_pipeline_in_flight(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RUNNING pipeline only READS its source during long transcribe/LLM
    stretches, so the mtime check can't see it — the in-flight registry
    must protect it. Regression: the reclaimer ate a source mid-run and
    the run died at the next step that read the file."""
    from chalybclip.jobs import pipeline_active

    t = await TenantsRepo(retention_db).create(name="Aldo")
    running = await _seed_source(
        retention_db, tenant_id=t.id, stream_id="str_running",
        output_dir=tmp_path, age_days=10,
    )
    idle = await _seed_source(
        retention_db, tenant_id=t.id, stream_id="str_idle",
        output_dir=tmp_path, age_days=5,
    )
    monkeypatch.setattr(shutil, "disk_usage", _fake_usage([1024]))  # always low
    with pipeline_active("str_running"):
        await reclaim_sources_until_free(
            retention_db, output_dir=tmp_path, target_free_bytes=10 * 1024**3,
            active_grace_s=0.0,
        )
    assert running.exists()      # in-flight source survives
    assert not idle.exists()     # the idle one was fair game


async def test_disabled_when_target_zero(
    retention_db: Database, tmp_path: Path
) -> None:
    out = await reclaim_sources_until_free(
        retention_db, output_dir=tmp_path, target_free_bytes=0,
    )
    assert out == (0, True)

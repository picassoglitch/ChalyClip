"""Retention sweeper — per-tenant deletes past the window (slice E.1)."""

from __future__ import annotations

from pathlib import Path

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
    TenantsRepo,
    TranscriptsRepo,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
    TranscriptRow,
)
from chalybclip.retention import (
    DEFAULT_RETENTION_CLIP_DAYS,
    DEFAULT_RETENTION_TRANSCRIPT_DAYS,
    DEFAULT_RETENTION_VOD_DAYS,
    RetentionPolicy,
    sweep_retention,
)
from chalybclip.tenancy import bound_tenant

from .conftest import days_ago_iso


async def _seed_stream(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str,
    output_dir: Path,
    age_days: int,
) -> None:
    """Create a stream row + on-disk source files at `age_days` old."""
    stream_dir = output_dir / stream_id
    src_dir = stream_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    video = src_dir / "video.mp4"
    audio = src_dir / "audio.wav"
    video.write_bytes(b"\x00" * 1024)
    audio.write_bytes(b"\x00" * 512)

    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url="https://kick.com/x/videos/1",
                platform="kick",
                title="t",
                channel="c",
                duration_s=60.0,
                source_video_path=str(video),
                source_audio_path=str(audio),
                status="ingested",
                created_at=days_ago_iso(age_days),
            )
        )


async def _seed_clip(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str,
    clip_id: str,
    output_dir: Path,
    age_days: int,
) -> Path:
    """Create a candidate + clip row + on-disk clip dir at `age_days` old."""
    clip_dir = output_dir / stream_id / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.mp4"
    thumb_path = clip_dir / "thumbnail.jpg"
    clip_path.write_bytes(b"\xff" * 4096)
    thumb_path.write_bytes(b"\xff\xd8\xff\xe0")

    with bound_tenant(tenant_id):
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id=f"cnd_{clip_id[4:]}",
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.9,
                    reason="voice",
                    evidence={},
                    created_at=days_ago_iso(age_days),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id,
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    candidate_id=f"cnd_{clip_id[4:]}",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path=str(clip_path),
                    thumbnail_frame_path=str(thumb_path),
                    status="cut",
                    created_at=days_ago_iso(age_days),
                )
            ]
        )
    return clip_path


async def _seed_transcript(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str,
    age_days: int,
) -> None:
    with bound_tenant(tenant_id):
        await TranscriptsRepo(db).upsert(
            TranscriptRow(
                stream_id=stream_id,
                tenant_id=tenant_id,
                language="es",
                duration_s=60.0,
                model="whisper-medium",
                segments_json="[]",
                created_at=days_ago_iso(age_days),
            )
        )


# ---- RetentionPolicy.from_tenant ----


def test_policy_falls_back_to_defaults_on_null_columns() -> None:
    p = RetentionPolicy.from_tenant(
        vod_days=None, clip_days=None, transcript_days=None
    )
    assert p.vod_days == DEFAULT_RETENTION_VOD_DAYS
    assert p.clip_days == DEFAULT_RETENTION_CLIP_DAYS
    assert p.transcript_days == DEFAULT_RETENTION_TRANSCRIPT_DAYS


def test_policy_honors_tenant_overrides() -> None:
    p = RetentionPolicy.from_tenant(vod_days=7, clip_days=14, transcript_days=30)
    assert p == (7, 14, 30)


def test_policy_zero_is_an_override_not_a_fallback() -> None:
    """`vod_days=0` means 'delete immediately', NOT 'use the default 30'."""
    p = RetentionPolicy.from_tenant(vod_days=0, clip_days=None, transcript_days=None)
    assert p.vod_days == 0


# ---- sweep_retention end-to-end ----


async def test_sweep_no_tenants_returns_empty(
    retention_db: Database, tmp_path: Path
) -> None:
    out = await sweep_retention(retention_db, output_dir=tmp_path)
    assert out == []


async def test_sweep_keeps_artifacts_inside_window(
    retention_db: Database, tmp_path: Path
) -> None:
    """A stream younger than the VOD window is left alone."""
    out_dir = tmp_path / "out"
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream(
        retention_db,
        tenant_id=t.id,
        stream_id="str_recent",
        output_dir=out_dir,
        age_days=5,  # well inside 7-day VOD default
    )
    reports = await sweep_retention(retention_db, output_dir=out_dir)
    assert len(reports) == 1
    r = reports[0]
    assert r.vods_deleted == 0
    assert r.bytes_freed == 0
    with bound_tenant(t.id):
        assert await StreamsRepo(retention_db).get("str_recent") is not None


async def test_sweep_deletes_aged_vod_files_and_row(
    retention_db: Database, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream(
        retention_db,
        tenant_id=t.id,
        stream_id="str_old",
        output_dir=out_dir,
        age_days=45,  # past 7-day VOD default
    )
    stream_dir = out_dir / "str_old"
    assert stream_dir.exists()

    reports = await sweep_retention(retention_db, output_dir=out_dir)
    r = reports[0]
    assert r.vods_deleted == 1
    assert r.bytes_freed >= 1024 + 512
    assert not stream_dir.exists()
    with bound_tenant(t.id):
        assert await StreamsRepo(retention_db).get("str_old") is None


async def test_sweep_deletes_aged_clips_and_unlinks_clip_dir(
    retention_db: Database, tmp_path: Path
) -> None:
    """A clip past 90 days is removed but its (younger) stream stays."""
    out_dir = tmp_path / "out"
    t = await TenantsRepo(retention_db).create(name="Aldo")
    # Stream is still inside the VOD window. Only the clip ages out.
    await _seed_stream(
        retention_db,
        tenant_id=t.id,
        stream_id="str_keep",
        output_dir=out_dir,
        age_days=3,  # inside the 7-day VOD window → stream untouched
    )
    clip_path = await _seed_clip(
        retention_db,
        tenant_id=t.id,
        stream_id="str_keep",
        clip_id="clp_old",
        output_dir=out_dir,
        age_days=120,
    )
    assert clip_path.exists()

    reports = await sweep_retention(retention_db, output_dir=out_dir)
    r = reports[0]
    assert r.clips_deleted == 1
    assert r.vods_deleted == 0
    assert not clip_path.exists()
    assert not clip_path.parent.exists()
    # Stream is still here.
    with bound_tenant(t.id):
        assert await StreamsRepo(retention_db).get("str_keep") is not None


async def test_sweep_past_vod_window_preserves_in_window_clips(
    retention_db: Database, tmp_path: Path
) -> None:
    """A stream past the VOD window (30d) keeps clips that are still inside
    the clip window (90d). Only the raw source is reclaimed.

    Regression: the sweeper used to nuke the whole stream dir + cascade-
    delete the row at the VOD cutoff, which destroyed in-window clips (and
    their rows) early. With the daily sweep loop now active, that would be
    live data loss.
    """
    out_dir = tmp_path / "out"
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream(
        retention_db,
        tenant_id=t.id,
        stream_id="str_mixed",
        output_dir=out_dir,
        age_days=45,  # past 30-day VOD window
    )
    # Clip at 45 days — past VOD window but well inside the 90-day clip one.
    clip_path = await _seed_clip(
        retention_db,
        tenant_id=t.id,
        stream_id="str_mixed",
        clip_id="clp_keep",
        output_dir=out_dir,
        age_days=45,
    )
    source_dir = out_dir / "str_mixed" / "source"
    assert source_dir.exists()

    reports = await sweep_retention(retention_db, output_dir=out_dir)
    r = reports[0]

    # Source reclaimed...
    assert r.vods_deleted == 1
    assert not source_dir.exists()
    # ...but the clip and its stream survive.
    assert r.clips_deleted == 0
    assert clip_path.exists()
    with bound_tenant(t.id):
        assert await StreamsRepo(retention_db).get("str_mixed") is not None
        assert await ClipsRepo(retention_db).get("clp_keep") is not None


async def test_sweep_deletes_aged_transcripts(
    retention_db: Database, tmp_path: Path
) -> None:
    """Transcripts past 365 days are dropped; the stream survives."""
    out_dir = tmp_path / "out"
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream(
        retention_db,
        tenant_id=t.id,
        stream_id="str_for_tx",
        output_dir=out_dir,
        age_days=10,
    )
    await _seed_transcript(
        retention_db,
        tenant_id=t.id,
        stream_id="str_for_tx",
        age_days=400,
    )

    reports = await sweep_retention(retention_db, output_dir=out_dir)
    r = reports[0]
    assert r.transcripts_deleted == 1
    with bound_tenant(t.id):
        assert await TranscriptsRepo(retention_db).get("str_for_tx") is None


async def test_sweep_honors_per_tenant_override(
    retention_db: Database, tmp_path: Path
) -> None:
    """A tenant with retention_vod_days=7 nukes streams older than a week."""
    out_dir = tmp_path / "out"
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await TenantsRepo(retention_db).set_retention(t.id, retention_vod_days=7)
    await _seed_stream(
        retention_db,
        tenant_id=t.id,
        stream_id="str_w",
        output_dir=out_dir,
        age_days=10,  # past 7-day override
    )
    reports = await sweep_retention(retention_db, output_dir=out_dir)
    assert reports[0].vods_deleted == 1


async def test_sweep_dry_run_reports_but_does_not_delete(
    retention_db: Database, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream(
        retention_db,
        tenant_id=t.id,
        stream_id="str_dry",
        output_dir=out_dir,
        age_days=45,
    )
    stream_dir = out_dir / "str_dry"
    reports = await sweep_retention(
        retention_db, output_dir=out_dir, dry_run=True
    )
    r = reports[0]
    assert r.dry_run is True
    assert r.vods_deleted == 1  # reported
    # ... but the files + row are still present.
    assert stream_dir.exists()
    with bound_tenant(t.id):
        assert await StreamsRepo(retention_db).get("str_dry") is not None


async def test_sweep_is_per_tenant_isolated(
    retention_db: Database, tmp_path: Path
) -> None:
    """Sweeping with tenant_id=alice doesn't touch bob's old VOD."""
    out_dir = tmp_path / "out"
    alice = await TenantsRepo(retention_db).create(name="Alice")
    bob = await TenantsRepo(retention_db).create(name="Bob")
    await _seed_stream(
        retention_db,
        tenant_id=alice.id,
        stream_id="str_a",
        output_dir=out_dir,
        age_days=60,
    )
    await _seed_stream(
        retention_db,
        tenant_id=bob.id,
        stream_id="str_b",
        output_dir=out_dir,
        age_days=60,
    )
    reports = await sweep_retention(
        retention_db, output_dir=out_dir, tenant_id=alice.id
    )
    assert len(reports) == 1
    assert reports[0].tenant_id == alice.id
    # Bob's stream + files survive.
    assert (out_dir / "str_b").exists()
    with bound_tenant(bob.id):
        assert await StreamsRepo(retention_db).get("str_b") is not None


async def test_sweep_unknown_tenant_returns_empty(
    retention_db: Database, tmp_path: Path
) -> None:
    out = await sweep_retention(
        retention_db, output_dir=tmp_path, tenant_id="ten_nope"
    )
    assert out == []


async def test_sweep_idempotent_safe_to_rerun(
    retention_db: Database, tmp_path: Path
) -> None:
    """Running the sweeper twice in a row doesn't double-delete or error."""
    out_dir = tmp_path / "out"
    t = await TenantsRepo(retention_db).create(name="Aldo")
    await _seed_stream(
        retention_db,
        tenant_id=t.id,
        stream_id="str_x",
        output_dir=out_dir,
        age_days=45,
    )
    first = await sweep_retention(retention_db, output_dir=out_dir)
    second = await sweep_retention(retention_db, output_dir=out_dir)
    assert first[0].vods_deleted == 1
    assert second[0].vods_deleted == 0  # nothing left to delete

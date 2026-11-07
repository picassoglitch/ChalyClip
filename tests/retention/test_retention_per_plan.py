"""Per-plan retention (owner 2026-11-07): free 7 days, paid 30, for clips,
VODs and transcripts — DB rows AND bucket objects.

Regression: every prod tenant had NULL retention columns, so the sweep fell
back to 90-day clips / 365-day transcripts and deleted nothing
(`clips_deleted=0` for all 7 tenants on 2026-11-07) while the oldest clip
was 34 days old and the plan cards promise 7 / 30.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pytest

import chalybclip.integrations.storage as storage_mod
from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
    TenantsRepo,
    TranscriptsRepo,
)
from chalybclip.db.models import CandidateRow, ClipRow, StreamRow, TranscriptRow
from chalybclip.integrations.storage import S3ArtifactStore
from chalybclip.jobs import active as active_mod
from chalybclip.retention import RetentionPolicy, sweep_retention
from chalybclip.tenancy import bound_tenant
from chalybclip.tiers import LONGEST_RETENTION_DAYS


def _iso(days: float, *, fmt: str = "iso") -> str:
    ts = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=days)
    if fmt == "z":
        return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if fmt == "space":
        return ts.isoformat(sep=" ")
    if fmt == "naive":
        return ts.replace(tzinfo=None).isoformat()
    return ts.isoformat()


class _FakeS3:
    """In-memory S3 client: paginated list (2 keys/page) + delete."""

    def __init__(self, keys: list[str] | None = None, *, fail_delete: bool = False) -> None:
        self.objects: set[str] = set(keys or [])
        self.fail_delete = fail_delete
        self.list_calls = 0

    def list_objects_v2(self, **kw: Any) -> dict[str, Any]:
        self.list_calls += 1
        keys = sorted(k for k in self.objects if k.startswith(kw["Prefix"]))
        start = int(kw.get("ContinuationToken") or 0)
        page = keys[start : start + 2]
        more = start + 2 < len(keys)
        out: dict[str, Any] = {
            "Contents": [{"Key": k} for k in page],
            "IsTruncated": more,
        }
        if more:
            out["NextContinuationToken"] = str(start + 2)
        return out

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        if self.fail_delete:
            raise RuntimeError("bucket down")
        self.objects.discard(Key)


def _use_store(monkeypatch: pytest.MonkeyPatch, client: _FakeS3) -> None:
    store = S3ArtifactStore(client=client, bucket="chalyb-media", prefix="artifacts")
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _s: store)


async def _tenant(db: Database, tier: str) -> str:
    t = await TenantsRepo(db).create(name=f"t-{tier}")
    conn = await db.connect()
    await conn.execute("UPDATE tenants SET tier = ? WHERE id = ?", (tier, t.id))
    await conn.commit()
    return t.id


async def _stream(db: Database, tid: str, sid: str, out: Path, created_at: str) -> None:
    src = out / sid / "source"
    src.mkdir(parents=True, exist_ok=True)
    (src / "video.mp4").write_bytes(b"\x00" * 100)
    with bound_tenant(tid):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=sid,
                tenant_id=tid,
                vod_url="https://kick.com/x/videos/1",
                platform="kick",
                title="t",
                channel="c",
                duration_s=60.0,
                source_video_path=str(src / "video.mp4"),
                source_audio_path=str(src / "audio.wav"),
                status="ingested",
                created_at=created_at,
            )
        )


async def _clip(db: Database, tid: str, sid: str, cid: str, out: Path, created_at: str) -> Path:
    clip_dir = out / sid / "clips" / cid
    clip_dir.mkdir(parents=True, exist_ok=True)
    path = clip_dir / "clip.mp4"
    path.write_bytes(b"\xff" * 100)
    with bound_tenant(tid):
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id=f"cnd_{cid}",
                    stream_id=sid,
                    tenant_id=tid,
                    ts=1.0,
                    score=0.9,
                    reason="voice",
                    evidence={},
                    created_at=created_at,
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=cid,
                    stream_id=sid,
                    tenant_id=tid,
                    candidate_id=f"cnd_{cid}",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path=str(path),
                    thumbnail_frame_path=str(clip_dir / "thumbnail.jpg"),
                    status="approved",
                    created_at=created_at,
                )
            ]
        )
    return path


async def _transcript(db: Database, tid: str, sid: str, created_at: str) -> None:
    with bound_tenant(tid):
        await TranscriptsRepo(db).upsert(
            TranscriptRow(
                stream_id=sid,
                tenant_id=tid,
                language="es",
                duration_s=60.0,
                model="m",
                segments_json="[]",
                created_at=created_at,
            )
        )


async def _clip_exists(db: Database, tid: str, cid: str) -> bool:
    with bound_tenant(tid):
        return await ClipsRepo(db).get(cid) is not None


# ---- policy ----


@pytest.mark.parametrize(
    ("tier", "days", "known"),
    [
        ("free", 7, True),
        ("pro", 30, True),
        ("all_access", 30, True),
        ("partner", 30, True),
        ("VIP", 30, True),
        ("gold", LONGEST_RETENTION_DAYS, False),
        ("", LONGEST_RETENTION_DAYS, False),
    ],
)
def test_policy_window_per_plan(tier: str, days: int, known: bool) -> None:
    p = RetentionPolicy.from_tenant(tier=tier)
    assert (p.vod_days, p.clip_days, p.transcript_days) == (days, days, days)
    assert p.tier_known is known


def test_unknown_tier_never_gets_free_window() -> None:
    assert RetentionPolicy.from_tenant(tier="gold").clip_days > 7


# ---- per-tier windows end to end ----


@pytest.mark.parametrize(
    ("tier", "age", "deleted"),
    [
        ("free", 8, True),
        ("free", 6, False),
        ("pro", 8, False),
        ("pro", 31, True),
        ("all_access", 29, False),
        ("all_access", 31, True),
        ("gold", 8, False),  # unknown tier: longest window, not free's
        ("gold", 31, True),
    ],
)
async def test_clip_window_per_tier(
    retention_db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: str,
    age: int,
    deleted: bool,
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, tier)
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    path = await _clip(retention_db, tid, "str_1", "clp_1", tmp_path, _iso(age))

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert r.clips_deleted == int(deleted)
    assert await _clip_exists(retention_db, tid, "clp_1") is not deleted
    assert path.exists() is not deleted


async def test_prod_shape_all_access_tenant(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prod tenant that reported clips_deleted=0: all_access, NULL
    columns, clips 35 and 10 days old, transcript + stream 35 days old."""
    client = _FakeS3()
    _use_store(monkeypatch, client)
    tid = await _tenant(retention_db, "all_access")
    await _stream(retention_db, tid, "str_old", tmp_path, _iso(35))
    await _transcript(retention_db, tid, "str_old", _iso(35))
    await _clip(retention_db, tid, "str_old", "clp_a", tmp_path, _iso(35))
    await _clip(retention_db, tid, "str_old", "clp_b", tmp_path, _iso(34))
    await _stream(retention_db, tid, "str_new", tmp_path, _iso(10))
    await _clip(retention_db, tid, "str_new", "clp_c", tmp_path, _iso(10))
    client.objects.update(
        {
            f"artifacts/clips/{tid}/clp_a/clip.mp4",
            f"artifacts/clips/{tid}/clp_a/clip_render_4k.mp4",
            f"artifacts/clips/{tid}/clp_b/thumbnail.jpg",
            f"artifacts/clips/{tid}/clp_c/clip.mp4",
            f"artifacts/uploads/{tid}/str_old/source.mp4",
            f"artifacts/work/{tid}/str_old/audio.wav",
        }
    )

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (r.clips_deleted, r.transcripts_deleted, r.streams_deleted) == (2, 1, 1)
    assert r.objects_deleted == 5
    assert client.objects == {f"artifacts/clips/{tid}/clp_c/clip.mp4"}
    assert not (tmp_path / "str_old").exists()
    assert await _clip_exists(retention_db, tid, "clp_c")
    with bound_tenant(tid):
        assert await StreamsRepo(retention_db).get("str_old") is None
        assert await StreamsRepo(retention_db).get("str_new") is not None


# ---- TEXT timestamps ----


@pytest.mark.parametrize("fmt", ["iso", "z", "space", "naive"])
async def test_text_timestamp_formats_compare_as_datetimes(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: str
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    await _clip(retention_db, tid, "str_1", "clp_old", tmp_path, _iso(8, fmt=fmt))
    await _clip(retention_db, tid, "str_1", "clp_new", tmp_path, _iso(6, fmt=fmt))

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert r.clips_deleted == 1
    assert not await _clip_exists(retention_db, tid, "clp_old")
    assert await _clip_exists(retention_db, tid, "clp_new")


async def test_unparseable_timestamp_is_kept_and_counted(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    await _clip(retention_db, tid, "str_1", "clp_bad", tmp_path, "not-a-date")

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert r.clips_deleted == 0
    assert r.skipped_bad_timestamp == 1
    assert await _clip_exists(retention_db, tid, "clp_bad")


# ---- in-flight clips ----


async def test_rendering_clip_is_skipped(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    await _clip(retention_db, tid, "str_1", "clp_r", tmp_path, _iso(20))
    conn = await retention_db.connect()
    await conn.execute("UPDATE clips SET render_state = 'rendering' WHERE id = 'clp_r'")
    await conn.commit()

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (r.clips_deleted, r.skipped_in_flight) == (0, 1)
    assert await _clip_exists(retention_db, tid, "clp_r")


@pytest.mark.parametrize(
    ("status", "kept"),
    [("scheduled", True), ("publishing", True), ("published", False), ("failed", False)],
)
async def test_clip_with_pending_publish_is_skipped(
    retention_db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    kept: bool,
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    await _clip(retention_db, tid, "str_1", "clp_p", tmp_path, _iso(20))
    conn = await retention_db.connect()
    await conn.execute(
        "INSERT INTO zernio_publishes (post_id, tenant_id, clip_id, platforms, "
        "created_at, status) VALUES (?, ?, ?, ?, ?, ?)",
        ("post_1", tid, "clp_p", "[]", _iso(1), status),
    )
    await conn.commit()

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert await _clip_exists(retention_db, tid, "clp_p") is kept
    assert r.skipped_in_flight == int(kept)


async def test_running_pipeline_keeps_its_stream_and_clips(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_run", tmp_path, _iso(20))
    await _clip(retention_db, tid, "str_run", "clp_1", tmp_path, _iso(20))
    monkeypatch.setattr(active_mod, "_active", active_mod.Counter({"str_run": 1}))

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (r.clips_deleted, r.streams_deleted) == (0, 0)
    assert r.skipped_in_flight == 2
    assert (tmp_path / "str_run" / "source" / "video.mp4").exists()


# ---- bucket failures, idempotency, dry run, batches ----


async def test_bucket_failure_keeps_row_for_retry(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tid = await _tenant(retention_db, "free")
    client = _FakeS3(fail_delete=True)
    _use_store(monkeypatch, client)
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    path = await _clip(retention_db, tid, "str_1", "clp_1", tmp_path, _iso(20))
    client.objects.add(f"artifacts/clips/{tid}/clp_1/clip.mp4")

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (r.clips_deleted, r.failed) == (0, 1)
    assert await _clip_exists(retention_db, tid, "clp_1")
    assert path.exists()

    client.fail_delete = False
    [r2] = await sweep_retention(retention_db, output_dir=tmp_path)
    assert (r2.clips_deleted, r2.failed) == (1, 0)
    assert client.objects == set()


async def test_broken_store_config_deletes_nothing(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(_s: object) -> None:
        raise RuntimeError("bad endpoint")

    monkeypatch.setattr(storage_mod, "build_artifact_store", _boom)
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(20))
    await _clip(retention_db, tid, "str_1", "clp_1", tmp_path, _iso(20))

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (r.clips_deleted, r.streams_deleted, r.failed) == (0, 0, 2)
    assert await _clip_exists(retention_db, tid, "clp_1")


async def test_second_run_is_a_no_op(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _FakeS3()
    _use_store(monkeypatch, client)
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(20))
    await _clip(retention_db, tid, "str_1", "clp_1", tmp_path, _iso(20))
    await _transcript(retention_db, tid, "str_1", _iso(20))
    client.objects.add(f"artifacts/clips/{tid}/clp_1/clip.mp4")

    [first] = await sweep_retention(retention_db, output_dir=tmp_path)
    calls_after_first = client.list_calls
    [second] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (first.clips_deleted, first.transcripts_deleted, first.streams_deleted) == (1, 1, 1)
    assert (
        second.clips_deleted,
        second.transcripts_deleted,
        second.streams_deleted,
        second.vods_deleted,
        second.objects_deleted,
    ) == (0, 0, 0, 0, 0)
    assert client.list_calls == calls_after_first


async def test_dry_run_counts_per_tier_and_touches_nothing(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[bool] = []
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _s: built.append(True))
    free = await _tenant(retention_db, "free")
    pro = await _tenant(retention_db, "pro")
    await _stream(retention_db, free, "str_f", tmp_path, _iso(10))
    await _clip(retention_db, free, "str_f", "clp_f", tmp_path, _iso(10))
    await _stream(retention_db, pro, "str_p", tmp_path, _iso(10))
    await _clip(retention_db, pro, "str_p", "clp_p", tmp_path, _iso(10))

    reports = {
        r.tenant_id: r
        for r in await sweep_retention(retention_db, output_dir=tmp_path, dry_run=True)
    }

    assert (reports[free].clips_deleted, reports[free].streams_deleted) == (1, 1)
    assert (reports[pro].clips_deleted, reports[pro].streams_deleted) == (0, 0)
    assert all(r.dry_run for r in reports.values())
    assert built == []
    assert await _clip_exists(retention_db, free, "clp_f")
    assert (tmp_path / "str_f" / "clips" / "clp_f" / "clip.mp4").exists()


async def test_batches_commit_every_n_rows(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    for i in range(5):
        await _clip(retention_db, tid, "str_1", f"clp_{i}", tmp_path, _iso(20))

    [r] = await sweep_retention(retention_db, output_dir=tmp_path, batch_size=2)

    assert r.clips_deleted == 5
    for i in range(5):
        assert not await _clip_exists(retention_db, tid, f"clp_{i}")


async def test_tenant_isolation_with_plan_windows(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _FakeS3()
    _use_store(monkeypatch, client)
    a = await _tenant(retention_db, "free")
    b = await _tenant(retention_db, "free")
    await _stream(retention_db, a, "str_a", tmp_path, _iso(1))
    await _clip(retention_db, a, "str_a", "clp_a", tmp_path, _iso(20))
    await _stream(retention_db, b, "str_b", tmp_path, _iso(1))
    await _clip(retention_db, b, "str_b", "clp_b", tmp_path, _iso(20))
    client.objects.add(f"artifacts/clips/{b}/clp_b/clip.mp4")

    await sweep_retention(retention_db, output_dir=tmp_path, tenant_id=a)

    assert not await _clip_exists(retention_db, a, "clp_a")
    assert await _clip_exists(retention_db, b, "clp_b")
    assert client.objects == {f"artifacts/clips/{b}/clp_b/clip.mp4"}


# ---- bucket prefix safety ----


@pytest.mark.parametrize(("tenant", "clip"), [("", "c"), ("t", ""), ("t/x", "c"), ("t", "..")])
def test_prefix_builders_reject_unsafe_ids(tenant: str, clip: str) -> None:
    from chalybclip.integrations.storage import clip_prefix

    with pytest.raises(ValueError):
        clip_prefix(tenant, clip)


async def test_delete_prefix_paginates_and_stays_inside_prefix() -> None:
    keys = [f"artifacts/clips/t/c/{i}.mp4" for i in range(5)]
    client = _FakeS3([*keys, "artifacts/clips/t/c2/clip.mp4"])
    store = S3ArtifactStore(client=client, bucket="b", prefix="artifacts")

    n = await store.delete_prefix(prefix="clips/t/c/")

    assert n == 5
    assert client.objects == {"artifacts/clips/t/c2/clip.mp4"}
    assert client.list_calls == 3


async def test_kept_stream_counts_its_vod_once(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream kept for a young clip is revisited every run; its source is
    only reported the run that actually removed it."""
    client = _FakeS3()
    _use_store(monkeypatch, client)
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(20))
    await _clip(retention_db, tid, "str_1", "clp_young", tmp_path, _iso(3))
    client.objects.add(f"artifacts/uploads/{tid}/str_1/source.mp4")

    [first] = await sweep_retention(retention_db, output_dir=tmp_path)
    [second] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (first.vods_deleted, first.streams_deleted, first.objects_deleted) == (1, 0, 1)
    assert (second.vods_deleted, second.objects_deleted) == (0, 0)
    assert not (tmp_path / "str_1" / "source").exists()
    assert await _clip_exists(retention_db, tid, "clp_young")


# ---- review follow-ups ----


@pytest.mark.parametrize("busy", ["usage_job", "pending_stream"])
async def test_stream_busy_in_db_is_skipped_from_another_process(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, busy: str
) -> None:
    """The scheduled job runs in its own process, where the in-memory
    registry is empty — the DB has to say the stream is busy."""
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(20))
    await _clip(retention_db, tid, "str_1", "clp_1", tmp_path, _iso(20))
    conn = await retention_db.connect()
    if busy == "usage_job":
        await conn.execute(
            "INSERT INTO usage_jobs (id, tenant_id, stream_id, operation, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ujob_1", tid, "str_1", "clip", "running", _iso(0.1), _iso(0.1)),
        )
    else:
        await conn.execute("UPDATE streams SET status = 'pending' WHERE id = 'str_1'")
    await conn.commit()

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (r.clips_deleted, r.streams_deleted) == (0, 0)
    assert await _clip_exists(retention_db, tid, "clp_1")


async def test_old_usage_job_does_not_block(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(20))
    await _clip(retention_db, tid, "str_1", "clp_1", tmp_path, _iso(20))
    conn = await retention_db.connect()
    await conn.execute(
        "INSERT INTO usage_jobs (id, tenant_id, stream_id, operation, status, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("ujob_1", tid, "str_1", "clip", "settled", _iso(20), _iso(19)),
    )
    await conn.commit()

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert (r.clips_deleted, r.streams_deleted) == (1, 1)


async def test_odd_clip_path_never_removes_a_shared_dir(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "free")
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    await _clip(retention_db, tid, "str_1", "clp_keep", tmp_path, _iso(1))
    await _clip(retention_db, tid, "str_1", "clp_odd", tmp_path, _iso(20))
    shared = tmp_path / "str_1" / "clips"
    conn = await retention_db.connect()
    await conn.execute(
        "UPDATE clips SET path = ? WHERE id = 'clp_odd'", (str(shared / "clip.mp4"),)
    )
    await conn.commit()

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert r.clips_deleted == 1
    assert (shared / "clp_keep" / "clip.mp4").exists()


# ---- overrides capped at the bucket's delete age ----


def test_override_above_bucket_age_is_clamped_and_logged() -> None:
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        p = RetentionPolicy.from_tenant(
            tier="pro", vod_days=45, clip_days=30, transcript_days=None, tenant_id="ten_x"
        )

    assert (p.vod_days, p.clip_days, p.transcript_days) == (30, 30, 30)
    clamped = [e for e in logs if e["event"] == "retention.override_clamped"]
    assert clamped == [
        {
            "event": "retention.override_clamped",
            "log_level": "warning",
            "tenant_id": "ten_x",
            "scope": "vod",
            "override_days": 45,
            "effective_days": 30,
        }
    ]


def test_plan_windows_never_exceed_bucket_age() -> None:
    from chalybclip.tiers import BUCKET_RETENTION_DAYS, RETENTION_DAYS

    assert BUCKET_RETENTION_DAYS == 30
    assert RETENTION_DAYS == {"free": 7, "pro": 30, "all_access": 30}
    assert max(RETENTION_DAYS.values()) <= BUCKET_RETENTION_DAYS


async def test_stored_override_above_bucket_age_sweeps_at_30(
    retention_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_store(monkeypatch, _FakeS3())
    tid = await _tenant(retention_db, "pro")
    conn = await retention_db.connect()
    await conn.execute("UPDATE tenants SET retention_clip_days = 90 WHERE id = ?", (tid,))
    await conn.commit()
    await _stream(retention_db, tid, "str_1", tmp_path, _iso(1))
    await _clip(retention_db, tid, "str_1", "clp_old", tmp_path, _iso(31))

    [r] = await sweep_retention(retention_db, output_dir=tmp_path)

    assert r.policy.clip_days == 30
    assert r.clips_deleted == 1

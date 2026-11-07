"""metered_run — every exit path meters compute.seconds at the lane's rate
and settles the reservation with the matching outcome."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chalybclip.db import ClipsRepo, Database, StreamsRepo, TenantsRepo, apply_migrations
from chalybclip.db.models import ClipRow, StreamRow
from chalybclip.db.usage_repos import UsageJobsRepo, UsageOutboxRepo
from chalybclip.integrations.chalyb import outbox as outbox_mod
from chalybclip.jobs.usage import COMPUTE_MICROS_PER_S, metered_run, settle_job
from chalybclip.tenancy import bound_tenant


@pytest.fixture
async def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    path = tmp_path / "t.db"
    db = Database(path)
    await apply_migrations(db)
    await TenantsRepo(db).create(tenant_id="ten_a", name="A")
    for job, res, lane in (("j_ok", "r_ok", "boost"), ("j_bad", "r_bad", "standard"),
                           ("j_cxl", "r_cxl", "standard")):
        await UsageJobsRepo(db).upsert(
            job_id=job, tenant_id="ten_a", stream_id="str_1", operation="clips.pipeline",
            job_class="job", reservation_id=res, lane=lane, boost=None,
            upload_mb=0, source_minutes=0,
        )
    await db.close()

    # No hub in tests: keep the job-end drain from doing network work.
    async def no_drain(_db: Database, *, timeout_s: float = 30.0) -> None:
        return None

    monkeypatch.setattr("chalybclip.jobs.usage.drain_until_idle", no_drain)
    return str(path)


async def _rows(db_path: str) -> tuple[dict, dict]:
    db = Database(db_path)
    repo = UsageOutboxRepo(db)
    out = {}
    conn = await db.connect()
    for i in ("compute_j_ok", "compute_j_bad", "compute_j_cxl"):
        # One compute row per run attempt: compute_<job>_<attempt id>.
        cur = await conn.execute(
            "SELECT id FROM usage_outbox WHERE id LIKE ?", (f"{i}_%",)
        )
        ids = [r["id"] for r in await cur.fetchall()]
        if ids:
            assert len(ids) == 1
            out[i] = (await repo.get(ids[0])).payload
    settles = {}
    for res in ("r_ok", "r_bad", "r_cxl"):
        r = await repo.get(f"settle_{res}")
        if r:
            settles[res] = r.payload["outcome"]
    await db.close()
    return out, settles


async def test_success_meters_boost_rate_and_settles_succeeded(
    db_path: str, tmp_path: Path
) -> None:
    stream_dir = tmp_path / "out" / "str_1"
    stream_dir.mkdir(parents=True)
    (stream_dir / "clip.mp4").write_bytes(b"x" * 2048)
    async with metered_run(
        db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_ok",
        reservation_id="r_ok", lane="boost", stream_dir=stream_dir,
    ):
        # Inside the run, events default to the run's reservation.
        assert outbox_mod.current_reservation() == "r_ok"
    assert outbox_mod.current_reservation() is None
    compute, settles = await _rows(db_path)
    ev = compute["compute_j_ok"]
    assert ev["kind"] == "compute.seconds" and ev["provider"] == "gcp"
    assert ev["reservation_id"] == "r_ok"
    assert ev["cost_usd_micros"] == ev["amount"] * COMPUTE_MICROS_PER_S["boost"] == ev["amount"] * 208
    assert settles["r_ok"] == "succeeded"
    db = Database(db_path)
    job = await UsageJobsRepo(db).get("j_ok")
    await db.close()
    assert job is not None and job.status == "succeeded"


async def test_failure_meters_standard_rate_and_settles_failed(db_path: str) -> None:
    with pytest.raises(RuntimeError):
        async with metered_run(
            db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_bad",
            reservation_id="r_bad", lane="standard",
        ):
            raise RuntimeError("ffmpeg died")
    compute, settles = await _rows(db_path)
    assert compute["compute_j_bad"]["cost_usd_micros"] == compute["compute_j_bad"]["amount"] * 88
    assert settles["r_bad"] == "failed"


async def test_cancellation_settles_cancelled(db_path: str) -> None:
    started = asyncio.Event()

    async def run() -> None:
        async with metered_run(
            db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_cxl",
            reservation_id="r_cxl", lane="standard",
        ):
            started.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    compute, settles = await _rows(db_path)
    assert settles["r_cxl"] == "cancelled"
    assert "compute_j_cxl" in compute


async def test_rerunning_the_same_job_meters_each_attempts_compute(db_path: str) -> None:
    """The same usage job can run twice (a dispatch retried after a lost
    response, a boost start that ran and also fell back to standard). Both
    attempts used compute; the second must not collide with the first's
    source_id and be dropped as a duplicate."""
    for _ in range(2):
        async with metered_run(
            db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_ok",
            reservation_id="r_ok", lane="standard",
        ):
            pass
    db = Database(db_path)
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT id FROM usage_outbox WHERE id LIKE 'compute_j_ok_%'"
    )
    ids = {r["id"] for r in await cur.fetchall()}
    await db.close()
    assert len(ids) == 2


# ---- outputs: the hub frees a failed run only when it left no clips ---------


async def _settle_payload(db_path: str, reservation_id: str) -> dict:
    db = Database(db_path)
    row = await UsageOutboxRepo(db).get(f"settle_{reservation_id}")
    await db.close()
    assert row is not None
    return row.payload


async def _add_clips(db_path: str, stream_id: str, n: int) -> None:
    db = Database(db_path)
    with bound_tenant("ten_a"):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id="ten_a", vod_url="upload://v.mp4",
                platform="upload", title="v", channel=None, duration_s=0.0,
                source_video_path="/tmp/v.mp4", source_audio_path="/tmp/v.wav",
                status="pending", created_at="2026-11-01T00:00:00+00:00",
            )
        )
        await ClipsRepo(db).upsert_many([
            ClipRow(id=f"clp_{stream_id}_{i}", stream_id=stream_id, tenant_id="ten_a",
                    start_s=0.0, end_s=10.0, duration_s=10.0, width=1080, height=1920,
                    path="/tmp/c.mp4", created_at="2026-11-01T00:00:00+00:00")
            for i in range(n)
        ])
    await db.close()


async def test_failure_without_clips_settles_outputs_zero(db_path: str) -> None:
    with pytest.raises(RuntimeError):
        async with metered_run(
            db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_bad",
            reservation_id="r_bad", lane="standard",
        ):
            raise RuntimeError("transcription failed")
    assert await _settle_payload(db_path, "r_bad") == {
        "reservation_id": "r_bad", "outcome": "failed", "outputs": 0,
    }


async def test_failure_after_making_clips_settles_their_count(db_path: str) -> None:
    with pytest.raises(RuntimeError):
        async with metered_run(
            db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_bad",
            reservation_id="r_bad", lane="standard",
        ):
            await _add_clips(db_path, "str_1", 3)
            raise RuntimeError("upload of the last clip failed")
    payload = await _settle_payload(db_path, "r_bad")
    assert payload["outcome"] == "failed" and payload["outputs"] == 3


async def test_clips_of_another_stream_or_tenant_do_not_count(db_path: str) -> None:
    await _add_clips(db_path, "str_other", 2)
    with pytest.raises(RuntimeError):
        async with metered_run(
            db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_bad",
            reservation_id="r_bad", lane="standard",
        ):
            raise RuntimeError("boom")
    assert (await _settle_payload(db_path, "r_bad"))["outputs"] == 0


async def test_success_and_cancel_send_no_outputs(db_path: str) -> None:
    async with metered_run(
        db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_ok",
        reservation_id="r_ok", lane="boost",
    ):
        pass
    assert "outputs" not in await _settle_payload(db_path, "r_ok")


async def test_unreadable_clip_count_sends_no_outputs(
    db_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-safe: no count → no field → the hub counts the run."""

    async def broken(*_a: object, **_k: object) -> int:
        raise RuntimeError("db locked")

    monkeypatch.setattr("chalybclip.jobs.usage.count_stream_clips", broken)
    with pytest.raises(RuntimeError):
        async with metered_run(
            db_path, tenant_id="ten_a", stream_id="str_1", job_id="j_bad",
            reservation_id="r_bad", lane="standard",
        ):
            raise RuntimeError("boom")
    payload = await _settle_payload(db_path, "r_bad")
    assert payload["outcome"] == "failed" and "outputs" not in payload


async def test_dispatcher_side_failure_settles_its_streams_clip_count(db_path: str) -> None:
    db = Database(db_path)
    try:
        await settle_job(
            db, tenant_id="ten_a", job_id="j_bad", reservation_id="r_bad", outcome="failed"
        )
        await settle_job(
            db, tenant_id="ten_a", job_id=None, reservation_id="r_cxl", outcome="failed"
        )
    finally:
        await db.close()
    assert (await _settle_payload(db_path, "r_bad"))["outputs"] == 0
    # No job row to find the stream by: no count sent.
    assert "outputs" not in await _settle_payload(db_path, "r_cxl")


async def test_dispatcher_failure_still_settles_when_the_job_lookup_fails(
    db_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metering hiccup must not lose the settle: it goes out without outputs."""

    async def broken(self: UsageJobsRepo, job_id: str) -> None:
        raise RuntimeError("db locked")

    monkeypatch.setattr(UsageJobsRepo, "get", broken)
    db = Database(db_path)
    try:
        await settle_job(
            db, tenant_id="ten_a", job_id="j_bad", reservation_id="r_bad", outcome="failed"
        )
    finally:
        await db.close()
    payload = await _settle_payload(db_path, "r_bad")
    assert payload["outcome"] == "failed" and "outputs" not in payload

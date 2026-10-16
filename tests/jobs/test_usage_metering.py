"""metered_run — every exit path meters compute.seconds at the lane's rate
and settles the reservation with the matching outcome."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chalybclip.db import Database, TenantsRepo, apply_migrations
from chalybclip.db.usage_repos import UsageJobsRepo, UsageOutboxRepo
from chalybclip.integrations.chalyb import outbox as outbox_mod
from chalybclip.jobs.usage import COMPUTE_MICROS_PER_S, metered_run


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
    for i in ("compute_j_ok", "compute_j_bad", "compute_j_cxl"):
        r = await repo.get(i)
        if r:
            out[i] = r.payload
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

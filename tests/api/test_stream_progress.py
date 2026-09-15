"""GET /dashboard/streams/{id}/progress — HTMX fragment used to surface
live pipeline progress on the stream detail page."""

from __future__ import annotations

import datetime as _dt
import json

import httpx

from chalybclip.db import (
    Database,
    StreamsRepo,
)
from chalybclip.db.models import StreamRow
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_stream(db: Database, *, tenant_id: str, stream_id: str) -> None:
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url=f"upload://{stream_id}.mp4",
                platform="upload",
                title="Test",
                channel=None,
                duration_s=120.0,
                source_video_path=f"/tmp/{stream_id}.mp4",
                source_audio_path=f"/tmp/{stream_id}.wav",
                status="ingested",
                created_at=_now(),
            )
        )


async def _emit_step_event(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str,
    event_type: str,
    step: str,
    duration_s: float | None = None,
    error: str | None = None,
) -> None:
    """Insert a pipeline.step.* event directly via raw SQL (mirrors what the
    sync hook in `chalybclip.pipeline._record_step_event` does)."""
    payload: dict[str, object] = {"step": step, "stream_id": stream_id}
    if duration_s is not None:
        payload["duration_s"] = duration_s
    if error is not None:
        payload["error"] = error
    conn = await db.connect()
    await conn.execute(
        "INSERT INTO events (id, tenant_id, type, payload_json, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            f"evt_{step}_{event_type}",
            tenant_id,
            event_type,
            json.dumps(payload),
            _now(),
        ),
    )
    await conn.commit()


async def test_progress_pending_when_no_events_yet(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """Fresh stream with no step events: every step renders as pending."""
    await _seed_stream(db, tenant_id=tenants["alice"]["id"], stream_id="str_p1")
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_p1/progress")
    assert r.status_code == 200
    body = r.text
    # All six steps appear by name.
    for step in ("ingest", "analyze_video", "diarize", "transcribe", "detect", "cut", "variants"):
        assert step in body
    # No "running" / "done" badges yet.
    assert "(running…)" not in body


async def test_progress_running_step_shows_elapsed_time(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """The running step row shows 'X s elapsed' / 'X m elapsed' so the user
    can see something is actively happening, not just a frozen spinner."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_pE")
    # Backdate the start event so elapsed > 0.
    payload = {"step": "transcribe", "stream_id": "str_pE"}
    started_at = (
        _dt.datetime.now(_dt.UTC) - _dt.timedelta(seconds=42)
    ).isoformat()
    conn = await db.connect()
    await conn.execute(
        "INSERT INTO events (id, tenant_id, type, payload_json, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            "evt_elapsed_test",
            tenant_id,
            "pipeline.step.start",
            json.dumps(payload),
            started_at,
        ),
    )
    await conn.commit()
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_pE/progress")
    assert r.status_code == 200
    assert "elapsed" in r.text


async def test_progress_running_step_polls_every_3s(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """A step.start without a matching step.done shows as running and the
    fragment includes the HTMX poll attribute so the page keeps refreshing."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_p2")
    await _emit_step_event(
        db,
        tenant_id=tenant_id,
        stream_id="str_p2",
        event_type="pipeline.step.done",
        step="ingest",
        duration_s=2.0,
    )
    await _emit_step_event(
        db,
        tenant_id=tenant_id,
        stream_id="str_p2",
        event_type="pipeline.step.start",
        step="transcribe",
    )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_p2/progress")
    assert r.status_code == 200
    body = r.text
    assert "(running…)" in body
    assert "transcribe" in body
    # Auto-refresh while running.
    assert 'hx-trigger="every 3s"' in body


async def test_progress_done_when_every_step_finished(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """Once every step has a `done` event, the fragment shows complete and
    drops the HTMX polling attribute (no more refresh)."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_p3")
    for step in ("ingest", "analyze_video", "diarize", "transcribe", "detect", "cut", "variants"):
        await _emit_step_event(
            db,
            tenant_id=tenant_id,
            stream_id="str_p3",
            event_type="pipeline.step.done",
            step=step,
            duration_s=1.5,
        )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_p3/progress")
    assert r.status_code == 200
    body = r.text
    assert "Pipeline complete" in body
    assert 'hx-trigger="every 3s"' not in body


async def test_progress_running_step_flips_to_abandoned_after_an_hour(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """A step.start that's >1h old with no matching done/failed is treated as
    abandoned — the original pipeline process died with a dashboard restart
    or Ctrl+C, and the dashboard should say so explicitly instead of
    showing a fake 'running' state forever."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_aban")

    # Backdate the start event by 2 hours so it crosses the abandonment
    # threshold (1h) regardless of test machine clock skew.
    started_at = (
        _dt.datetime.now(_dt.UTC) - _dt.timedelta(hours=2)
    ).isoformat()
    conn = await db.connect()
    await conn.execute(
        "INSERT INTO events (id, tenant_id, type, payload_json, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            "evt_abandoned_test",
            tenant_id,
            "pipeline.step.start",
            json.dumps({"step": "transcribe", "stream_id": "str_aban"}),
            started_at,
        ),
    )
    await conn.commit()

    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_aban/progress")
    assert r.status_code == 200
    body = r.text
    # Banner reflects abandonment, not 'Pipeline running'.
    assert "Pipeline abandoned" in body
    assert "Pipeline running" not in body
    # And we stop polling — no point hitting the server every 3s when
    # nothing can change without user action.
    assert 'hx-trigger="every 3s"' not in body


async def test_progress_long_step_stays_running_while_run_in_flight(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """A step.start >1h old is NOT abandoned while jobs.active holds a run
    for the stream — the pipeline scales per-step timeouts with VOD duration
    (analyze_video on a 5h VOD legitimately runs for hours), so wall clock
    alone must not flip a live run to 'se quedó a medias' + a Córrelo button
    that would stack a duplicate run on top of it."""
    from chalybclip.jobs.active import pipeline_active

    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_live")

    started_at = (
        _dt.datetime.now(_dt.UTC) - _dt.timedelta(hours=2)
    ).isoformat()
    conn = await db.connect()
    await conn.execute(
        "INSERT INTO events (id, tenant_id, type, payload_json, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            "evt_long_alive_test",
            tenant_id,
            "pipeline.step.start",
            json.dumps({"step": "analyze_video", "stream_id": "str_live"}),
            started_at,
        ),
    )
    await conn.commit()

    with pipeline_active("str_live"):
        r = await client.get(
            "/dashboard/streams/str_live/progress",
            headers=auth(tenants["alice"]["token"]),
        )
    assert r.status_code == 200
    body = r.text
    assert "se quedó a medias" not in body
    assert "(en proceso…)" in body
    # Elapsed renders in hours past the 60-minute mark.
    assert "lleva 2h" in body
    # Still polling — the run is alive and the card must keep refreshing.
    assert 'hx-trigger="every 3s"' in body

    # Same stale event with NO run in flight (registry released) → the
    # original abandoned behavior stands.
    r = await client.get(
        "/dashboard/streams/str_live/progress",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    assert "se quedó a medias" in r.text


async def test_progress_failed_step_surfaces_error(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """A step.failed event shows the error text inline and stops polling."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_p4")
    await _emit_step_event(
        db,
        tenant_id=tenant_id,
        stream_id="str_p4",
        event_type="pipeline.step.failed",
        step="transcribe",
        error="CUDA out of memory",
    )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_p4/progress")
    assert r.status_code == 200
    body = r.text
    assert "Pipeline failed" in body
    assert "CUDA out of memory" in body
    assert 'hx-trigger="every 3s"' not in body


async def test_rerun_kicks_off_pipeline_for_existing_stream(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """POST /dashboard/streams/{id}/rerun reschedules the pipeline runner
    using the on-disk stream.json — useful when the original background
    task died with a server restart."""
    import httpx as _httpx
    import pytest as _pytest

    from chalybclip.api import PipelineKickoff, create_app
    from chalybclip.ingest.models import Stream

    _ = _pytest  # silence unused-import warning

    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_rr1")

    # Persist a stream.json on disk for load_stream() to read.
    stream_dir = tmp_path / "out" / "str_rr1"
    stream_dir.mkdir(parents=True)
    Stream(
        id="str_rr1",
        tenant_id=tenant_id,
        vod_url="upload://x.mp4",
        platform="upload",
        title="x",
        duration_s=120.0,
        source_video_path=stream_dir / "source" / "video.mp4",
        source_audio_path=stream_dir / "source" / "audio.wav",
    ).model_dump_json(indent=2)
    (stream_dir / "stream.json").write_text(
        Stream(
            id="str_rr1",
            tenant_id=tenant_id,
            vod_url="upload://x.mp4",
            platform="upload",
            title="x",
            duration_s=120.0,
            source_video_path=stream_dir / "source" / "video.mp4",
            source_audio_path=stream_dir / "source" / "audio.wav",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type("S", (), {"default_output_dir": str(tmp_path / "out")})(),
    )

    captured: list[PipelineKickoff] = []

    async def fake_runner(kickoff: PipelineKickoff) -> None:
        captured.append(kickoff)

    # Add a persona — the rerun form requires one.
    from chalybclip.db import PersonasRepo
    from chalybclip.tenancy import bound_tenant as _bound

    with _bound(tenant_id):
        await PersonasRepo(db).create(
            persona_id="p1",
            name="P",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="v",
        )

    app = create_app(db=db, pipeline_runner=fake_runner)
    transport = _httpx.ASGITransport(app=app)
    async with _httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as c:
        await c.post(
            "/dashboard/login", data={"token": tenants["alice"]["token"]}
        )
        r = await c.post(
            "/dashboard/streams/str_rr1/rerun",
            data={"persona_id": "p1"},
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/streams/str_rr1"
    assert len(captured) == 1
    assert captured[0].stream.id == "str_rr1"
    assert captured[0].persona_id == "p1"


async def test_rerun_reingests_failed_url_stream_without_artifacts(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
) -> None:
    """A URL stream whose ingest failed (no on-disk stream.json — e.g. a
    transient YouTube bot-gate) re-attempts a fresh download from the stored
    VOD URL instead of 409'ing, and its row is reset to 'pending' so a
    successful re-download reconciles back to 'ingested'. (The shared client's
    pipeline runner is a no-op, so nothing actually downloads.)"""
    from chalybclip.db import PersonasRepo

    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    url = "https://youtu.be/5ZY5-5FcpMo"
    with bound_tenant(tid):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_yt", tenant_id=tid, vod_url=url, platform="youtube",
                title=None, channel=None, duration_s=0.0,
                # An output dir under tmp_path that has NO stream.json → the
                # rerun handler's load_stream() fails → re-ingest fallback.
                source_video_path=str(tmp_path / "str_yt" / "source" / "video.mp4"),
                source_audio_path=str(tmp_path / "str_yt" / "source" / "audio.wav"),
                status="failed", created_at=_now(),
            )
        )
        await PersonasRepo(db).create(
            persona_id="p1", name="P", primary_language="es",
            target_languages=["es"], voice_prompt="v",
        )
    r = await client.post(
        "/dashboard/streams/str_yt/rerun",
        data={"persona_id": "p1"},
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )
    # 303 (re-ingest dispatched), NOT 409 (the old "can't resume" behavior).
    assert r.status_code == 303, r.text
    with bound_tenant(tid):
        row = await StreamsRepo(db).get("str_yt")
    # Reset so a successful re-download reconciles back to 'ingested'.
    assert row is not None and row.status == "pending"


async def test_default_pipeline_runner_passes_db_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: the dashboard's pipeline runner must forward db_path so
    the pipeline's _step() can write events the progress card reads back.

    Earlier this was broken — process_vod was called without db_path, so
    every step ran with db=None, _record_step_event short-circuited, and
    the progress endpoint had no events to surface (the panel stayed on
    'all pending' forever even while the real pipeline was running).
    """
    from chalybclip.api import PipelineKickoff
    from chalybclip.api._pipeline import default_pipeline_runner
    from chalybclip.ingest.models import Stream

    captured: dict[str, object] = {}

    async def fake_process_vod(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("chalybclip.pipeline.process_vod", fake_process_vod)
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: type(
            "S",
            (),
            {
                "db_path": str(tmp_path / "fake.db"),
                # _pipeline now resolves the target via db_target() (Postgres
                # migration seam); mirror it so the runner forwards the path.
                "db_target": lambda self: str(tmp_path / "fake.db"),
            },
        )(),
    )

    stream = Stream(
        id="str_x1",
        tenant_id="t1",
        vod_url="upload://x.mp4",
        platform="upload",
        title="x",
        duration_s=60.0,
        source_video_path=tmp_path / "v.mp4",
        source_audio_path=tmp_path / "a.wav",
    )
    await default_pipeline_runner(
        PipelineKickoff(
            tenant_id="t1",
            stream=stream,
            persona_id="p1",
            output_dir=tmp_path,
        )
    )

    assert captured["db_path"] == str(tmp_path / "fake.db")
    assert captured["stream_id"] == "str_x1"
    assert captured["persona_id"] == "p1"


async def test_progress_isolates_events_per_stream(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """Step events from another stream in the same tenant don't bleed in."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_pA")
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_pB")
    # str_pA has finished transcribe; str_pB hasn't started anything.
    await _emit_step_event(
        db,
        tenant_id=tenant_id,
        stream_id="str_pA",
        event_type="pipeline.step.done",
        step="transcribe",
        duration_s=42.0,
    )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_pB/progress")
    assert r.status_code == 200
    # str_pB's transcribe should still show as pending, not done.
    assert "Pipeline complete" not in r.text

"""Pipeline-recovery sweeper tests.

Two layers:
  * `classify_streams` — pure threshold logic, exercised against
    hand-built rows/events with a fixed `now`.
  * `recover_orphaned_pipelines` — end-to-end against a migrated DB +
    a fake dispatcher, asserting it re-dispatches the right streams,
    writes the audit event, and gives up after the attempt cap.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from pathlib import Path

from chalybclip.db import Database, EventsRepo, PersonasRepo, StreamsRepo, TenantsRepo
from chalybclip.db.models import Event, StreamRow
from chalybclip.events.log import STREAM_PROCESSED
from chalybclip.recovery import classify_streams, recover_orphaned_pipelines
from chalybclip.recovery.service import (
    IN_FLIGHT_SILENCE_S,
    MAX_ATTEMPTS,
    NEVER_STARTED_GRACE_S,
    RECOVERY_DISPATCHED,
)
from chalybclip.tenancy import bound_tenant

_NOW = _dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=_dt.UTC)


def _ago(seconds: float) -> str:
    return (_NOW - _dt.timedelta(seconds=seconds)).isoformat()


def _stream(
    stream_id: str,
    *,
    created_s_ago: float,
    vod_url: str = "https://www.youtube.com/watch?v=abc",
    platform: str = "youtube",
    is_live: bool = False,
    status: str = "pending",
) -> StreamRow:
    return StreamRow(
        id=stream_id,
        tenant_id="ten_a",
        vod_url=vod_url,
        platform=platform,
        duration_s=0.0,
        source_video_path=f"/out/{stream_id}/source/video.mp4",
        source_audio_path=f"/out/{stream_id}/source/audio.wav",
        status=status,
        created_at=_ago(created_s_ago),
        is_live=is_live,
    )


def _event(stream_id: str, type_: str, *, s_ago: float) -> Event:
    return Event(
        id=f"evt_{stream_id}_{type_}_{int(s_ago)}",
        tenant_id="ten_a",
        type=type_,
        payload={"stream_id": stream_id},
        ts=_ago(s_ago),
    )


# ---------- classify_streams (pure) ----------


def test_never_started_old_url_job_is_recovered() -> None:
    # The headline bug: a URL job orphaned before any step ran.
    s = _stream("str_1", created_s_ago=NEVER_STARTED_GRACE_S + 60)
    ev = [_event("str_1", "stream.created", s_ago=NEVER_STARTED_GRACE_S + 60)]
    out = classify_streams([s], ev, now=_NOW)
    assert [d.stream_id for d in out] == ["str_1"]
    assert out[0].give_up is False


def test_fresh_submission_is_left_alone() -> None:
    # Younger than the grace window — could just be mid-download.
    s = _stream("str_1", created_s_ago=60)
    out = classify_streams([s], [], now=_NOW)
    assert out == []


def test_completed_stream_skipped() -> None:
    s = _stream("str_1", created_s_ago=NEVER_STARTED_GRACE_S + 600)
    ev = [_event("str_1", STREAM_PROCESSED, s_ago=10)]
    assert classify_streams([s], ev, now=_NOW) == []


def test_surfaced_failure_skipped() -> None:
    s = _stream("str_1", created_s_ago=NEVER_STARTED_GRACE_S + 600)
    ev = [_event("str_1", "pipeline.failed", s_ago=10)]
    assert classify_streams([s], ev, now=_NOW) == []


def test_live_and_upload_streams_skipped() -> None:
    live = _stream("str_live", created_s_ago=99999, vod_url="live://rtmp/x", is_live=True)
    upload = _stream("str_up", created_s_ago=99999, vod_url="upload://clip.mp4")
    assert classify_streams([live, upload], [], now=_NOW) == []


def test_ancient_stream_left_for_retention() -> None:
    s = _stream("str_1", created_s_ago=20 * 24 * 3600)  # 20 days
    assert classify_streams([s], [], now=_NOW) == []


def test_in_flight_recent_step_protected_but_long_silence_recovered() -> None:
    # A started-then-silent run: protected under the silence ceiling
    # (a slow Whisper looks like this), recovered past it.
    alive = _stream("str_alive", created_s_ago=IN_FLIGHT_SILENCE_S)
    alive_ev = [_event("str_alive", "pipeline.step.start", s_ago=IN_FLIGHT_SILENCE_S - 600)]
    assert classify_streams([alive], alive_ev, now=_NOW) == []

    dead = _stream("str_dead", created_s_ago=IN_FLIGHT_SILENCE_S + 4000)
    dead_ev = [_event("str_dead", "pipeline.step.start", s_ago=IN_FLIGHT_SILENCE_S + 600)]
    out = classify_streams([dead], dead_ev, now=_NOW)
    assert [d.stream_id for d in out] == ["str_dead"]


def test_in_flight_registry_stream_skipped() -> None:
    """A stream queued on the dispatcher's semaphore (or mid-run in a long
    silent step) looks timing-eligible — no step events for hours — but is
    NOT an orphan. The jobs.active registry is authoritative: skip it
    entirely, so it is neither re-dispatched (duplicate concurrent runs)
    nor given up on (bogus RecoveryExhausted over a healthy run)."""
    queued = _stream("str_q", created_s_ago=NEVER_STARTED_GRACE_S + 600)
    silent = _stream("str_s", created_s_ago=IN_FLIGHT_SILENCE_S + 4000)
    silent_ev = [
        _event("str_s", "pipeline.step.start", s_ago=IN_FLIGHT_SILENCE_S + 600)
    ]
    # Both are eligible on timing alone…
    assert len(classify_streams([queued, silent], silent_ev, now=_NOW)) == 2
    # …but excluded while registered in flight.
    out = classify_streams(
        [queued, silent], silent_ev, now=_NOW,
        active_ids=frozenset({"str_q", "str_s"}),
    )
    assert out == []


def test_attempts_cap_flags_give_up() -> None:
    s = _stream("str_1", created_s_ago=NEVER_STARTED_GRACE_S + 600)
    ev = [
        _event("str_1", RECOVERY_DISPATCHED, s_ago=NEVER_STARTED_GRACE_S + i)
        for i in range(MAX_ATTEMPTS)
    ]
    out = classify_streams([s], ev, now=_NOW)
    assert len(out) == 1
    assert out[0].give_up is True


# ---------- recover_orphaned_pipelines (integration) ----------


class _FakeDispatcher:
    """Records kickoffs instead of running the pipeline."""

    name = "fake"

    def __init__(self) -> None:
        self.kickoffs: list[object] = []

    async def dispatch_pipeline(self, kickoff: object, *, background_tasks: object = None) -> None:
        self.kickoffs.append(kickoff)


async def _seed_tenant_with_persona(db: Database) -> None:
    await TenantsRepo(db).create(tenant_id="ten_a", name="A")
    with bound_tenant("ten_a"):
        await PersonasRepo(db).create(
            persona_id="per_1",
            name="Default",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="x",
        )


async def _insert_event_at(
    db: Database, *, stream_id: str, type_: str, s_ago: float
) -> None:
    """Insert an event row with a controlled `ts` (EventsRepo.emit stamps
    wall-clock, which we can't pin to the fixed `_NOW` the test asserts on)."""
    conn = await db.connect()
    await conn.execute(
        "INSERT INTO events (id, tenant_id, type, payload_json, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            f"evt_{stream_id}_{type_}_{int(s_ago)}",
            "ten_a",
            type_,
            json.dumps({"stream_id": stream_id}),
            _ago(s_ago),
        ),
    )
    await conn.commit()


async def test_recover_dispatches_orphan_and_writes_audit_event(
    recovery_db: Database, tmp_path: Path
) -> None:
    await _seed_tenant_with_persona(recovery_db)
    with bound_tenant("ten_a"):
        await StreamsRepo(recovery_db).upsert(
            _stream("str_orphan", created_s_ago=NEVER_STARTED_GRACE_S + 600)
        )
    # A backdated stream.created event — the orphan's only trace, well past
    # the grace window.
    await _insert_event_at(
        recovery_db,
        stream_id="str_orphan",
        type_="stream.created",
        s_ago=NEVER_STARTED_GRACE_S + 600,
    )

    dispatcher = _FakeDispatcher()
    registry: set[asyncio.Task[None]] = set()
    recovered = await recover_orphaned_pipelines(
        recovery_db,
        dispatcher=dispatcher,
        output_dir=tmp_path,
        now=_NOW,
        task_registry=registry,
    )
    if registry:
        await asyncio.gather(*registry)

    assert recovered == ["str_orphan"]
    assert len(dispatcher.kickoffs) == 1
    ko = dispatcher.kickoffs[0]
    assert ko.stream.id == "str_orphan"  # type: ignore[attr-defined]
    assert ko.persona_id == "per_1"  # type: ignore[attr-defined]

    with bound_tenant("ten_a"):
        dispatched = await EventsRepo(recovery_db).list_for_tenant(type=RECOVERY_DISPATCHED)
    assert len(dispatched) == 1
    assert dispatched[0].payload["stream_id"] == "str_orphan"


async def test_recover_gives_up_after_attempt_cap(
    recovery_db: Database, tmp_path: Path
) -> None:
    await _seed_tenant_with_persona(recovery_db)
    with bound_tenant("ten_a"):
        await StreamsRepo(recovery_db).upsert(
            _stream("str_stuck", created_s_ago=NEVER_STARTED_GRACE_S + 600)
        )
    # Backdated past the grace window so the stream is eligible again, but
    # with MAX_ATTEMPTS recovery dispatches already on record → give up.
    for i in range(MAX_ATTEMPTS):
        await _insert_event_at(
            recovery_db,
            stream_id="str_stuck",
            type_=RECOVERY_DISPATCHED,
            s_ago=NEVER_STARTED_GRACE_S + 300 + i,
        )

    dispatcher = _FakeDispatcher()
    recovered = await recover_orphaned_pipelines(
        recovery_db, dispatcher=dispatcher, output_dir=tmp_path, now=_NOW
    )

    assert recovered == []
    assert dispatcher.kickoffs == []
    with bound_tenant("ten_a"):
        failed = await EventsRepo(recovery_db).list_for_tenant(type="pipeline.failed")
    assert len(failed) == 1
    assert failed[0].payload["error_type"] == "RecoveryExhausted"

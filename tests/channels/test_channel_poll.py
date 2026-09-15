"""Channel auto-ingest poll loop — back-catalog guard, dedup, caps,
failure handling. Lister + ingest callback are faked (no yt-dlp, no DB
pipeline)."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.channels import ChannelVOD, poll_channel_watches
from chalybclip.channels.service import _is_due, _poll_one_watch
from chalybclip.db import ChannelWatchesRepo, Database, TenantsRepo, apply_migrations
from chalybclip.db.models import ChannelWatchRow
from chalybclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "ch.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


def _lister(vods: list[ChannelVOD]):
    async def _list(platform: str, channel_url: str, *, limit: int) -> list[ChannelVOD]:
        return list(vods[:limit])
    return _list


def _failing_lister():
    async def _list(platform: str, channel_url: str, *, limit: int) -> list[ChannelVOD]:
        raise RuntimeError("yt-dlp boom")
    return _list


class _Recorder:
    """Captures every (tenant, url, video_id) the poller hands us."""

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.ingested: list[str] = []
        self.fail_ids = fail_ids or set()

    async def __call__(
        self, tenant_id: str, url: str, video_id: str, persona_id: str,
        language: str | None,
    ) -> None:
        if video_id in self.fail_ids:
            raise RuntimeError("ingest boom")
        self.ingested.append(video_id)


def _vods(*ids: str) -> list[ChannelVOD]:
    return [ChannelVOD(video_id=i, url=f"https://yt/watch?v={i}", title=i) for i in ids]


@pytest_asyncio.fixture
async def watch_tenant(db: Database) -> str:
    t = await TenantsRepo(db).create(name="Streamer")
    return t.id


@pytest.mark.asyncio
async def test_first_poll_ingests_only_newest_and_marks_backcatalog_seen(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        watch = await ChannelWatchesRepo(db).create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", max_per_poll=2,
        )
        rec = _Recorder()
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_lister(_vods("a", "b", "c", "d")),
        )
        # Only the newest 2 ingested; the rest of the window is marked seen.
        assert rec.ingested == ["a", "b"]
        assert report.videos_ingested == 2
        refreshed = await ChannelWatchesRepo(db).get(watch.id)
    assert refreshed is not None
    assert set(refreshed.seen_video_ids) == {"a", "b", "c", "d"}  # back-catalog suppressed
    assert refreshed.last_polled_at is not None


@pytest.mark.asyncio
async def test_steady_state_ingests_only_new_capped(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", max_per_poll=2,
        )
        # Simulate a prior poll: a,b,c,d already seen, last_polled set.
        await repo.mark_polled(
            watch.id, seen_video_ids=["a", "b", "c", "d"],
            last_polled_at="2026-06-10T10:00:00+00:00",
        )
        watch = await repo.get(watch.id)
        assert watch is not None
        rec = _Recorder()
        # Two NEW uploads (e, f) + the old ones still listed.
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec,
            list_vods=_lister(_vods("f", "e", "d", "c")),
        )
    assert set(rec.ingested) == {"f", "e"}  # only the new ones, capped at 2
    assert report.videos_ingested == 2


@pytest.mark.asyncio
async def test_disabled_watch_skipped(db: Database, watch_tenant: str) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me", persona_id="per_1",
        )
        await repo.set_enabled(watch.id, enabled=False)
        watch = await repo.get(watch.id)
        assert watch is not None
        rec = _Recorder()
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_lister(_vods("a")),
        )
    assert report.skipped_disabled is True
    assert rec.ingested == []


@pytest.mark.asyncio
async def test_list_failure_does_not_advance_polled_at(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        watch = await ChannelWatchesRepo(db).create(
            platform="twitch", channel_url="https://twitch/me", persona_id="per_1",
        )
        rec = _Recorder()
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_failing_lister(),
        )
    assert report.videos_failed == 1
    assert report.videos_ingested == 0


@pytest.mark.asyncio
async def test_ingest_failure_keeps_video_unseen_for_retry(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", max_per_poll=3,
        )
        await repo.mark_polled(
            watch.id, seen_video_ids=[], last_polled_at="2026-06-10T10:00:00+00:00",
        )
        watch = await repo.get(watch.id)
        assert watch is not None
        rec = _Recorder(fail_ids={"b"})
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_lister(_vods("a", "b", "c")),
        )
        refreshed = await repo.get(watch.id)
    assert report.videos_ingested == 2  # a, c
    assert report.videos_failed == 1   # b
    assert refreshed is not None
    # b stays unseen → retried next poll. last_polled_at DOES advance now:
    # the listing succeeded, so the poll ran; a single failed video must not
    # freeze the watch as "nunca" / permanently due.
    assert "b" not in refreshed.seen_video_ids
    assert {"a", "c"} <= set(refreshed.seen_video_ids)
    assert refreshed.last_polled_at != "2026-06-10T10:00:00+00:00"


@pytest.mark.asyncio
async def test_failed_video_is_parked_after_retry_ceiling(
    db: Database, watch_tenant: str
) -> None:
    """A permanently-unavailable video (yt-dlp 'This video is not available')
    is retried a few times then PARKED into the seen-set, so it stops failing
    on every poll. Mirrors the real k0fGua3TA8s loop."""
    from chalybclip.channels.service import _MAX_INGEST_ATTEMPTS

    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", max_per_poll=3,
        )
        # Steady state: a good video `a` already seen, a dead video `b` keeps
        # failing. Each poll lists both; `a` is deduped, `b` is retried.
        await repo.mark_polled(
            watch.id, seen_video_ids=["a"],
            last_polled_at="2026-06-10T10:00:00+00:00",
        )
        rec = _Recorder(fail_ids={"b"})

        # Poll up to (ceiling - 1) times: `b` stays unseen, count climbs.
        for attempt in range(1, _MAX_INGEST_ATTEMPTS):
            watch = await repo.get(watch.id)
            assert watch is not None
            await _poll_one_watch(
                db=db, watch=watch, ingest_callback=rec,
                list_vods=_lister(_vods("b", "a")),
            )
            refreshed = await repo.get(watch.id)
            assert refreshed is not None
            assert "b" not in refreshed.seen_video_ids
            assert refreshed.failed_video_attempts == {"b": attempt}

        # The ceiling-th failure parks `b`: into seen, out of the retry map.
        watch = await repo.get(watch.id)
        assert watch is not None
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec,
            list_vods=_lister(_vods("b", "a")),
        )
        refreshed = await repo.get(watch.id)
    assert report.videos_failed == 1
    assert refreshed is not None
    assert "b" in refreshed.seen_video_ids       # parked
    assert "b" not in refreshed.failed_video_attempts  # retry map cleared


@pytest.mark.asyncio
async def test_successful_retry_clears_failure_count(
    db: Database, watch_tenant: str
) -> None:
    """A video that fails once then succeeds is ingested and its failure
    count is cleared — a transient error must not count against it forever."""
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", max_per_poll=3,
        )
        await repo.mark_polled(
            watch.id, seen_video_ids=[],
            last_polled_at="2026-06-10T10:00:00+00:00",
        )

        # Poll 1: `b` fails → count {b: 1}, unseen.
        watch = await repo.get(watch.id)
        assert watch is not None
        rec = _Recorder(fail_ids={"b"})
        await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_lister(_vods("b")),
        )
        mid = await repo.get(watch.id)
        assert mid is not None
        assert mid.failed_video_attempts == {"b": 1}

        # Poll 2: ingest now succeeds → `b` seen, count cleared.
        rec_ok = _Recorder()
        await _poll_one_watch(
            db=db, watch=mid, ingest_callback=rec_ok, list_vods=_lister(_vods("b")),
        )
        refreshed = await repo.get(watch.id)
    assert rec_ok.ingested == ["b"]
    assert refreshed is not None
    assert "b" in refreshed.seen_video_ids
    assert refreshed.failed_video_attempts == {}


# ---- poll schedule (polls_per_day) ----


def _watch(last_polled_at: str | None, polls_per_day: int) -> ChannelWatchRow:
    return ChannelWatchRow(
        id="chw_1", tenant_id="ten_1", platform="youtube",
        channel_url="https://yt/@me", persona_id="per_1",
        last_polled_at=last_polled_at, seen_video_ids=[], max_per_poll=3,
        enabled=True, polls_per_day=polls_per_day,
        created_at="2026-06-10T00:00:00+00:00", updated_at="2026-06-10T00:00:00+00:00",
    )


def test_is_due_never_polled() -> None:
    now = _dt.datetime(2026, 6, 10, 12, 0, tzinfo=_dt.UTC)
    assert _is_due(_watch(None, 1), now) is True


def test_is_due_daily_respects_24h() -> None:
    now = _dt.datetime(2026, 6, 10, 12, 0, tzinfo=_dt.UTC)
    # polled 2h ago, 1/day → not due
    assert _is_due(_watch("2026-06-10T10:00:00+00:00", 1), now) is False
    # polled 25h ago → due
    assert _is_due(_watch("2026-06-09T11:00:00+00:00", 1), now) is True


def test_is_due_four_per_day_is_every_6h() -> None:
    now = _dt.datetime(2026, 6, 10, 12, 0, tzinfo=_dt.UTC)
    # polled 3h ago, 4/day (6h interval) → not due
    assert _is_due(_watch("2026-06-10T09:00:00+00:00", 4), now) is False
    # polled 7h ago → due
    assert _is_due(_watch("2026-06-10T05:00:00+00:00", 4), now) is True


@pytest.mark.asyncio
async def test_respect_schedule_skips_recently_polled(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", polls_per_day=1,
        )
        # Polled 1h ago → with 1/day cadence, not due yet.
        await repo.mark_polled(
            watch.id, seen_video_ids=[],
            last_polled_at=(_dt.datetime.now(_dt.UTC) - _dt.timedelta(hours=1)).isoformat(),
        )
    rec = _Recorder()
    reports = await poll_channel_watches(
        db, ingest_callback=rec, list_vods=_lister(_vods("a")),
        tenant_id=watch_tenant, respect_schedule=True,
    )
    assert len(reports) == 1
    assert reports[0].skipped_not_due is True
    assert rec.ingested == []  # lister never ran — resource saved


@pytest.mark.asyncio
async def test_respect_schedule_polls_when_due(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", polls_per_day=1,
        )
        await repo.mark_polled(
            watch.id, seen_video_ids=[],
            last_polled_at=(_dt.datetime.now(_dt.UTC) - _dt.timedelta(hours=25)).isoformat(),
        )
    rec = _Recorder()
    reports = await poll_channel_watches(
        db, ingest_callback=rec, list_vods=_lister(_vods("a")),
        tenant_id=watch_tenant, respect_schedule=True,
    )
    assert reports[0].skipped_not_due is False
    assert rec.ingested == ["a"]


@pytest.mark.asyncio
async def test_manual_poll_ignores_schedule(
    db: Database, watch_tenant: str
) -> None:
    """`channel poll` (respect_schedule=False) forces a poll regardless
    of cadence."""
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", polls_per_day=1,
        )
        await repo.mark_polled(
            watch.id, seen_video_ids=[],
            last_polled_at=_dt.datetime.now(_dt.UTC).isoformat(),  # just now
        )
    rec = _Recorder()
    reports = await poll_channel_watches(
        db, ingest_callback=rec, list_vods=_lister(_vods("a")),
        tenant_id=watch_tenant,  # respect_schedule defaults False
    )
    assert reports[0].skipped_not_due is False
    assert rec.ingested == ["a"]


@pytest.mark.asyncio
async def test_set_polls_per_day_clamps_and_persists(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me", persona_id="per_1",
        )
        assert watch.polls_per_day == 1  # default
        updated = await repo.set_polls_per_day(watch.id, 4)
        assert updated.polls_per_day == 4
        clamped = await repo.set_polls_per_day(watch.id, 0)  # < 1 clamps to 1
        assert clamped.polls_per_day == 1


@pytest.mark.asyncio
async def test_poll_all_watches_isolates_per_tenant(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        await ChannelWatchesRepo(db).create(
            platform="youtube", channel_url="https://yt/@me", persona_id="per_1",
        )
    rec = _Recorder()
    reports = await poll_channel_watches(
        db, ingest_callback=rec, list_vods=_lister(_vods("x")),
        tenant_id=watch_tenant,
    )
    assert len(reports) == 1
    assert reports[0].videos_ingested == 1


# ---- on-demand single-watch poll (dashboard "scan now") ----


@pytest.mark.asyncio
async def test_poll_watch_id_filter_targets_one_watch(
    db: Database, watch_tenant: str
) -> None:
    """`watch_id` restricts the poll to a single watch — the on-demand
    'scan now' button must not sweep the tenant's other channels."""
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        a = await repo.create(
            platform="youtube", channel_url="https://yt/@a", persona_id="per_1",
        )
        await repo.create(
            platform="youtube", channel_url="https://yt/@b", persona_id="per_1",
        )

    def _per_channel_lister(mapping: dict[str, list[ChannelVOD]]):
        async def _list(platform: str, channel_url: str, *, limit: int):
            return list(mapping.get(channel_url, [])[:limit])
        return _list

    rec = _Recorder()
    reports = await poll_channel_watches(
        db,
        ingest_callback=rec,
        list_vods=_per_channel_lister(
            {"https://yt/@a": _vods("a1"), "https://yt/@b": _vods("b1")}
        ),
        tenant_id=watch_tenant,
        watch_id=a.id,
        respect_schedule=False,
    )

    assert rec.ingested == ["a1"]  # only watch A, not B
    assert len(reports) == 1
    assert reports[0].watch_id == a.id

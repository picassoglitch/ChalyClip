"""Per-platform abuse/rate-limit backoff: parsing + the cooldown repo."""

from __future__ import annotations

import datetime as _dt

import pytest

from chalybclip.db import Database, PlatformCooldownsRepo, apply_migrations
from chalybclip.integrations.zernio.errors import (
    cooldowns_from_failed_post,
    failure_anchor,
    parse_cooldown,
)


def test_parse_cooldown_minutes() -> None:
    assert parse_cooldown("Rate limit hit. Please wait 1438 minutes before posting again.") == _dt.timedelta(minutes=1438)


def test_parse_cooldown_hours_and_daily() -> None:
    assert parse_cooldown("try again in 2 hours") == _dt.timedelta(hours=2)
    assert parse_cooldown("daily upload limit reached. try again tomorrow.") == _dt.timedelta(hours=24)


def test_parse_cooldown_default_when_unparseable() -> None:
    assert parse_cooldown("something went wrong") == _dt.timedelta(hours=6)
    assert parse_cooldown(None) == _dt.timedelta(hours=6)


def test_cooldowns_from_failed_post_only_user_abuse() -> None:
    post = {
        "platforms": [
            {"platform": "youtube", "status": "failed",
             "errorCategory": "user_abuse",
             "errorMessage": "Rate limit hit. Please wait 60 minutes."},
            {"platform": "tiktok", "status": "failed",
             "errorCategory": "auth_expired", "errorMessage": "token expired"},
            {"platform": "instagram", "status": "published"},
        ]
    }
    cds = cooldowns_from_failed_post(post)
    assert set(cds) == {"youtube"}  # only the abuse one
    assert cds["youtube"] == _dt.timedelta(minutes=60)


def test_failure_anchor_parses_and_prefers_failure_time() -> None:
    """Cooldowns run from WHEN the post failed, not from when a sweep
    re-reads the (never-shrinking) failed list — anchoring at read time
    re-armed every cooldown forever. failedAt wins over createdAt; a Z
    suffix parses; the result is aware UTC."""
    post = {
        "createdAt": "2026-07-01T10:00:00Z",
        "failedAt": "2026-07-02T08:30:00Z",
    }
    anchor = failure_anchor(post)
    assert anchor == _dt.datetime(2026, 7, 2, 8, 30, tzinfo=_dt.UTC)

    # Falls back through updatedAt/createdAt when failedAt is absent.
    assert failure_anchor({"createdAt": "2026-07-01T10:00:00Z"}) == _dt.datetime(
        2026, 7, 1, 10, 0, tzinfo=_dt.UTC
    )


def test_failure_anchor_none_when_unparseable() -> None:
    """No timestamp → None. The caller must SKIP the post (guessing
    "now" recreates the eternal re-arm this fixes)."""
    assert failure_anchor({}) is None
    assert failure_anchor({"failedAt": "not-a-date", "updatedAt": ""}) is None


@pytest.mark.asyncio
async def test_cooldown_repo_active_and_expiry(tmp_path) -> None:
    db = Database(tmp_path / "x.db")
    await apply_migrations(db)
    conn = await db.connect()
    await conn.execute("INSERT INTO tenants (id, name, created_at) VALUES ('t1','T','2026-01-01')")
    await conn.commit()
    repo = PlatformCooldownsRepo(db)
    now = _dt.datetime.now(_dt.UTC)

    # Fresh cooldown (future) shows as active; canonicalizes x→twitter.
    await repo.set_cooldown("t1", "x", until=(now + _dt.timedelta(hours=2)).isoformat())
    await repo.set_cooldown("t1", "youtube", until=(now - _dt.timedelta(hours=1)).isoformat())
    active = await repo.active("t1")
    assert "twitter" in active  # x normalized + still active
    assert "youtube" not in active  # lapsed → pruned
    await db.close()


async def _seeded_db(tmp_path):
    db = Database(tmp_path / "x.db")
    await apply_migrations(db)
    conn = await db.connect()
    await conn.execute(
        "INSERT INTO tenants (id, name, created_at) VALUES ('t1','T','2026-01-01')"
    )
    await conn.commit()
    return db


@pytest.mark.asyncio
async def test_clear_stale_by_reason_prunes_unsupported_rows(tmp_path) -> None:
    """The self-heal primitive: rate_limit rows outside the keep-set go,
    kept platforms and OTHER reasons survive (canonical comparison)."""
    db = await _seeded_db(tmp_path)
    repo = PlatformCooldownsRepo(db)
    future = (_dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=12)).isoformat()
    await repo.set_cooldown("t1", "youtube", until=future, reason="rate_limit")
    await repo.set_cooldown("t1", "x", until=future, reason="rate_limit")
    await repo.set_cooldown("t1", "tiktok", until=future, reason="manual")

    deleted = await repo.clear_stale_by_reason(
        "t1", reason="rate_limit", keep={"twitter"},  # x row stored canonically
    )
    assert deleted == 1  # youtube dropped
    active = await repo.active("t1")
    assert set(active) == {"twitter", "tiktok"}  # kept + other-reason survive

    # Empty keep-set clears every rate_limit row, still sparing other reasons.
    await repo.clear_stale_by_reason("t1", reason="rate_limit", keep=set())
    assert set(await repo.active("t1")) == {"tiktok"}
    await db.close()


class _StubZernio:
    def __init__(self, failed: list[dict]) -> None:
        self._failed = failed

    async def list_failed(self, *, profile_id=None, limit=50):
        return self._failed


@pytest.mark.asyncio
async def test_cooled_down_platforms_self_heals_stale_rows(
    tmp_path, monkeypatch
) -> None:
    """A stale rate_limit row the PRE-FIX code wrote (re-armed from `now`
    forever) must clear as soon as the failed-post evidence no longer
    supports it — not sit parking the platform until its bogus `until`
    passes. Evidence without a parseable timestamp keeps its row (bounded
    by that row's own until); anchored evidence rewrites its row."""
    from chalybclip.api.routers import zernio as zr

    db = await _seeded_db(tmp_path)
    repo = PlatformCooldownsRepo(db)
    now = _dt.datetime.now(_dt.UTC)
    bogus_until = (now + _dt.timedelta(hours=20)).isoformat()  # pre-fix debris
    await repo.set_cooldown("t1", "youtube", until=bogus_until, reason="rate_limit")
    await repo.set_cooldown("t1", "tiktok", until=bogus_until, reason="rate_limit")
    await repo.set_cooldown("t1", "instagram", until=bogus_until, reason="rate_limit")

    def _abuse(platform: str, msg: str, **extra) -> dict:
        return {
            "platforms": [{
                "platform": platform, "status": "failed",
                "errorCategory": "user_abuse", "errorMessage": msg,
            }],
            **extra,
        }

    failed = [
        # tiktok: anchored + still waiting → row rewritten, stays active.
        _abuse("tiktok", "rate limit. wait 120 minutes.",
               failedAt=(now - _dt.timedelta(minutes=10)).isoformat()),
        # instagram: abuse evidence but NO parseable timestamp → row kept.
        _abuse("instagram", "rate limit. wait 120 minutes."),
        # youtube: no abuse evidence at all → stale row must be dropped.
    ]
    monkeypatch.setattr(zr, "_build_client", lambda: _StubZernio(failed))

    cooled = await zr._cooled_down_platforms(db, "t1", profile_id="prof_1")
    assert "tiktok" in cooled       # live, evidence-backed
    assert "instagram" in cooled    # unanchored evidence → bounded row stands
    assert "youtube" not in cooled  # pre-fix debris self-healed away
    await db.close()


@pytest.mark.asyncio
async def test_cooled_down_platforms_keeps_rows_when_fetch_fails(
    tmp_path, monkeypatch
) -> None:
    """No evidence ≠ evidence of nothing: a Zernio outage must not wipe
    real cooldowns (fail-open would feed a throttling platform)."""
    from chalybclip.api.routers import zernio as zr

    db = await _seeded_db(tmp_path)
    repo = PlatformCooldownsRepo(db)
    until = (_dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=2)).isoformat()
    await repo.set_cooldown("t1", "tiktok", until=until, reason="rate_limit")

    class _Down:
        async def list_failed(self, *, profile_id=None, limit=50):
            raise RuntimeError("zernio 503")

    monkeypatch.setattr(zr, "_build_client", lambda: _Down())
    cooled = await zr._cooled_down_platforms(db, "t1", profile_id="prof_1")
    assert "tiktok" in cooled  # untouched
    await db.close()

"""Unified calendar: external-post webhook processing + store (phase 8)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.db import Database, ZernioCalendarRepo, ZernioEventsRepo, apply_migrations
from chalybclip.integrations.zernio.events import process_zernio_event

_ACC = "acct_gbp_1"


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "cal.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


def _external_payload(
    event_id: str, event: str, post_id: str, **post_over: object
) -> dict:
    post = {
        "id": post_id,
        "platform": "googlebusiness",
        "accountId": _ACC,
        "url": "https://g.co/p/" + post_id,
        "content": "Oferta de la semana",
        "mediaType": "image",
        "mediaItems": [],
        "thumbnailUrl": "https://g.co/thumb.jpg",
        "publishedAt": "2026-06-05T10:00:00Z",
        "source": "external",
    }
    post.update(post_over)
    return {
        "id": event_id,
        "event": event,
        "post": post,
        "account": {"id": _ACC, "platform": "googlebusiness", "username": "Mi Negocio"},
        "timestamp": "2026-06-05T10:05:00Z",
    }


async def _deliver(db: Database, payload: dict) -> None:
    await ZernioEventsRepo(db).insert_dedup(
        event_id=payload["id"], type=payload["event"], payload=json.dumps(payload),
    )
    await process_zernio_event(db, payload["id"])


@pytest.mark.asyncio
async def test_external_created_upserts_into_calendar(db: Database) -> None:
    await _deliver(db, _external_payload("ev1", "post.external.created", "gp1"))
    rows = await ZernioCalendarRepo(db).list_for_accounts([_ACC])
    assert len(rows) == 1
    assert rows[0]["post_id"] == "gp1"
    assert rows[0]["platform"] == "googlebusiness"
    assert rows[0]["status"] == "active"
    assert rows[0]["content"] == "Oferta de la semana"


@pytest.mark.asyncio
async def test_first_sync_created_is_idempotent(db: Database) -> None:
    # Zernio's first sync delivers every native post as `created`;
    # a re-delivered created must not duplicate.
    await _deliver(db, _external_payload("ev1", "post.external.created", "gp1"))
    await _deliver(db, _external_payload("ev2", "post.external.created", "gp1"))
    rows = await ZernioCalendarRepo(db).list_for_accounts([_ACC])
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_external_updated_overwrites_content(db: Database) -> None:
    await _deliver(db, _external_payload("ev1", "post.external.created", "gp1"))
    await _deliver(
        db,
        _external_payload(
            "ev2", "post.external.updated", "gp1", content="Texto editado",
        ),
    )
    rows = await ZernioCalendarRepo(db).list_for_accounts([_ACC])
    assert len(rows) == 1
    assert rows[0]["content"] == "Texto editado"


@pytest.mark.asyncio
async def test_external_deleted_marks_not_drops(db: Database) -> None:
    await _deliver(db, _external_payload("ev1", "post.external.created", "gp1"))
    await _deliver(
        db,
        _external_payload(
            "ev2", "post.external.deleted", "gp1",
            deletedAt="2026-06-06T08:00:00Z",
        ),
    )
    # Default list includes deleted (greyed in UI).
    rows = await ZernioCalendarRepo(db).list_for_accounts([_ACC])
    assert len(rows) == 1
    assert rows[0]["status"] == "deleted"
    assert rows[0]["deleted_at"] == "2026-06-06T08:00:00Z"
    # active-only filter excludes it.
    active = await ZernioCalendarRepo(db).list_for_accounts(
        [_ACC], include_deleted=False
    )
    assert active == []


@pytest.mark.asyncio
async def test_deleted_post_can_reappear(db: Database) -> None:
    await _deliver(db, _external_payload("ev1", "post.external.created", "gp1"))
    await _deliver(db, _external_payload("ev2", "post.external.deleted", "gp1"))
    await _deliver(db, _external_payload("ev3", "post.external.created", "gp1"))
    rows = await ZernioCalendarRepo(db).list_for_accounts([_ACC])
    assert rows[0]["status"] == "active"
    assert rows[0]["deleted_at"] is None


@pytest.mark.asyncio
async def test_list_for_accounts_date_range_and_empty(db: Database) -> None:
    await _deliver(db, _external_payload(
        "ev1", "post.external.created", "gp1", publishedAt="2026-06-05T10:00:00Z"))
    await _deliver(db, _external_payload(
        "ev2", "post.external.created", "gp2", publishedAt="2026-07-01T10:00:00Z"))
    in_june = await ZernioCalendarRepo(db).list_for_accounts(
        [_ACC], date_from="2026-06-01", date_to="2026-06-30",
    )
    assert {r["post_id"] for r in in_june} == {"gp1"}
    # No accounts → empty (no SQL on empty IN).
    assert await ZernioCalendarRepo(db).list_for_accounts([]) == []


@pytest.mark.asyncio
async def test_missing_account_id_is_skipped(db: Database) -> None:
    payload = _external_payload("ev1", "post.external.created", "gp1")
    del payload["post"]["accountId"]
    await _deliver(db, payload)
    assert await ZernioCalendarRepo(db).list_for_accounts([_ACC]) == []

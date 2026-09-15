"""Auto-retry-once on transient failures (Hub phase 6).

The processor fires ONE automatic retry when a post.failed / post.partial
event has ALL failed platforms in a transient error class, then stops.
The ledger claim guarantees once-only across at-least-once redeliveries.
Delay is set to 0 in tests so the retry runs inline (observable).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import respx

from chalybclip.db import (
    Database,
    ZernioAutoRetriesRepo,
    ZernioEventsRepo,
    apply_migrations,
)
from chalybclip.integrations.zernio.events import process_zernio_event
from chalybclip.settings import get_settings

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def retry_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_retry")
    monkeypatch.setenv("CHALYBCLIP_HUB_AUTO_RETRY_DELAY_S", "0")  # inline
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "retry.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _store_failed(
    db: Database, *, event_id: str, post_id: str, category: str,
    event_type: str = "post.failed",
) -> None:
    payload = {
        "id": event_id,
        "event": event_type,
        "post": {
            "id": post_id,
            "status": "failed",
            "platforms": [
                {"platform": "tiktok", "status": "failed",
                 "errorCategory": category, "errorMessage": "boom"},
            ],
        },
    }
    await ZernioEventsRepo(db).insert_dedup(
        event_id=event_id, type=event_type, payload=json.dumps(payload),
    )


@pytest.mark.asyncio
async def test_transient_failure_auto_retries_once(
    retry_env: None, db: Database
) -> None:
    await _store_failed(
        db, event_id="evt_t1", post_id="post_t1", category="platform_error",
    )
    with respx.mock(assert_all_called=True) as mock:
        retry = mock.post(f"{_ZBASE}/posts/post_t1/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "post_t1"}})
        )
        await process_zernio_event(db, "evt_t1")
    assert retry.called
    ledger = await ZernioAutoRetriesRepo(db).get("post_t1")
    assert ledger is not None
    assert ledger["outcome"] == "ok"


@pytest.mark.asyncio
async def test_redelivery_does_not_retry_twice(
    retry_env: None, db: Database
) -> None:
    with respx.mock() as mock:
        retry = mock.post(f"{_ZBASE}/posts/post_t2/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "post_t2"}})
        )
        # First delivery → one retry.
        await _store_failed(db, event_id="evt_t2a", post_id="post_t2",
                            category="system_error")
        await process_zernio_event(db, "evt_t2a")
        # Second delivery of the SAME post (different event id) → claim
        # already taken, no second retry.
        await _store_failed(db, event_id="evt_t2b", post_id="post_t2",
                            category="system_error")
        await process_zernio_event(db, "evt_t2b")
    assert len(retry.calls) == 1


@pytest.mark.asyncio
async def test_non_transient_failure_does_not_auto_retry(
    retry_env: None, db: Database
) -> None:
    await _store_failed(
        db, event_id="evt_t3", post_id="post_t3", category="auth_expired",
    )
    with respx.mock(assert_all_called=False) as mock:
        retry = mock.post(f"{_ZBASE}/posts/post_t3/retry").mock(
            return_value=httpx.Response(200, json={"post": {"_id": "post_t3"}})
        )
        await process_zernio_event(db, "evt_t3")
    assert not retry.called
    # No ledger row — nothing was claimed.
    assert await ZernioAutoRetriesRepo(db).get("post_t3") is None


@pytest.mark.asyncio
async def test_auto_retry_disabled_when_delay_negative(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_x")
    monkeypatch.setenv("CHALYBCLIP_HUB_AUTO_RETRY_DELAY_S", "-1")
    get_settings.cache_clear()
    try:
        await _store_failed(
            db, event_id="evt_t4", post_id="post_t4", category="platform_error",
        )
        with respx.mock(assert_all_called=False) as mock:
            retry = mock.post(f"{_ZBASE}/posts/post_t4/retry").mock(
                return_value=httpx.Response(200, json={"post": {"_id": "x"}})
            )
            await process_zernio_event(db, "evt_t4")
        assert not retry.called
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_failed_retry_records_failed_outcome(
    retry_env: None, db: Database
) -> None:
    await _store_failed(
        db, event_id="evt_t5", post_id="post_t5", category="platform_error",
    )
    with respx.mock() as mock:
        mock.post(f"{_ZBASE}/posts/post_t5/retry").mock(
            return_value=httpx.Response(500, text="still broken")
        )
        await process_zernio_event(db, "evt_t5")
    ledger = await ZernioAutoRetriesRepo(db).get("post_t5")
    assert ledger is not None
    assert ledger["outcome"] == "failed"

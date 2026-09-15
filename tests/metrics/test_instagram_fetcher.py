"""Instagram Insights metric fetcher."""

from __future__ import annotations

import datetime as _dt

import httpx
import respx

from chalybclip.db.models import ConnectedAccount, PublishJob
from chalybclip.metrics import fetch_instagram_metric


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _job(*, external_id: str | None) -> PublishJob:
    return PublishJob(
        id="pjb_x",
        tenant_id="t",
        clip_id="c",
        variant_id="v",
        account_id="acc",
        platform="instagram",
        status="sent",
        attempts=1,
        last_error=None,
        scheduled_for=None,
        external_id=external_id,
        created_at=_now(),
        external_url=None,
        platform_metadata=None,
    )


def _account() -> ConnectedAccount:
    return ConnectedAccount(
        id="acc",
        tenant_id="t",
        platform="instagram",
        external_id="17841400000000000",
        oauth_blob={"access_token": "tok"},
        created_at=_now(),
    )


_INSIGHTS_URL = "https://graph.facebook.com/v22.0/media_omega/insights"


@respx.mock
async def test_instagram_fetcher_happy_path() -> None:
    """Insights API returns one row per metric; we take the latest value."""
    respx.get(_INSIGHTS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"name": "plays", "values": [{"value": 1500}]},
                    {"name": "likes", "values": [{"value": 88}]},
                    {"name": "comments", "values": [{"value": 5}]},
                    {"name": "shares", "values": [{"value": 12}]},
                ]
            },
        )
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_instagram_metric(
            _job(external_id="media_omega"), _account(), http, "tok"
        )
    assert m.views == 1500
    assert m.likes == 88
    assert m.comments == 5
    assert m.shares == 12
    assert m.retention_pct is None  # IG doesn't expose retention here
    assert m.raw_metadata is not None


@respx.mock
async def test_instagram_fetcher_4xx_records_audit_row() -> None:
    respx.get(_INSIGHTS_URL).mock(return_value=httpx.Response(403, text="forbidden"))
    async with httpx.AsyncClient() as http:
        m = await fetch_instagram_metric(
            _job(external_id="media_omega"), _account(), http, "tok"
        )
    assert m.views is None
    assert m.raw_metadata is not None
    assert "error" in m.raw_metadata


@respx.mock
async def test_instagram_fetcher_skips_when_no_external_id() -> None:
    async with httpx.AsyncClient() as http:
        m = await fetch_instagram_metric(
            _job(external_id=None), _account(), http, "tok"
        )
    assert m.views is None
    assert m.raw_metadata == {"skipped": "no external_id"}


@respx.mock
async def test_instagram_fetcher_empty_data_array() -> None:
    """Sandbox apps see 200 with empty `data` - record the audit row."""
    respx.get(_INSIGHTS_URL).mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_instagram_metric(
            _job(external_id="media_omega"), _account(), http, "tok"
        )
    assert m.views is None
    assert m.raw_metadata == {"empty_response": True}


@respx.mock
async def test_instagram_fetcher_handles_missing_metric_rows() -> None:
    """Some accounts only return a subset; the others stay None."""
    respx.get(_INSIGHTS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"name": "plays", "values": [{"value": 200}]},
                    # likes/comments/shares not returned at all
                ]
            },
        )
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_instagram_metric(
            _job(external_id="media_omega"), _account(), http, "tok"
        )
    assert m.views == 200
    assert m.likes is None
    assert m.comments is None
    assert m.shares is None

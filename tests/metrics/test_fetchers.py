"""Per-platform fetchers — happy path + missing-stat NULLs + 4xx fallback."""

from __future__ import annotations

import datetime as _dt

import httpx
import respx

from chalybclip.db.models import ConnectedAccount, PublishJob
from chalybclip.metrics import (
    fetch_buffer_metric,
    fetch_tiktok_metric,
    fetch_youtube_metric,
)


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _job(*, external_id: str | None, platform: str) -> PublishJob:
    return PublishJob(
        id="pjb_x",
        tenant_id="t",
        clip_id="c",
        variant_id="v",
        account_id="acc",
        platform=platform,
        status="sent",
        attempts=1,
        last_error=None,
        scheduled_for=None,
        external_id=external_id,
        created_at=_now(),
        external_url=None,
        platform_metadata=None,
    )


def _account(platform: str) -> ConnectedAccount:
    return ConnectedAccount(
        id="acc",
        tenant_id="t",
        platform=platform,
        external_id="x",
        oauth_blob={"access_token": "tok"},
        created_at=_now(),
    )


# ---- YouTube ----


@respx.mock
async def test_youtube_fetcher_happy_path() -> None:
    respx.get("https://www.googleapis.com/youtube/v3/videos").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "yt_abc",
                        "statistics": {
                            "viewCount": "1500",
                            "likeCount": "42",
                            "commentCount": "7",
                        },
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_youtube_metric(
            _job(external_id="yt_abc", platform="youtube"),
            _account("youtube"),
            http,
            "tok",
        )
    assert m.views == 1500
    assert m.likes == 42
    assert m.comments == 7
    assert m.shares is None  # YT Data API doesn't expose shares
    assert m.retention_pct is None  # requires the YT Analytics API
    assert m.raw_metadata is not None


@respx.mock
async def test_youtube_fetcher_4xx_returns_null_metric_with_audit() -> None:
    respx.get("https://www.googleapis.com/youtube/v3/videos").mock(
        return_value=httpx.Response(403, text="forbidden")
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_youtube_metric(
            _job(external_id="yt_abc", platform="youtube"),
            _account("youtube"),
            http,
            "tok",
        )
    assert m.views is None
    assert m.raw_metadata is not None
    assert "error" in m.raw_metadata


@respx.mock
async def test_youtube_fetcher_skips_when_no_external_id() -> None:
    """Job without an external_id can't be looked up - return empty."""
    async with httpx.AsyncClient() as http:
        m = await fetch_youtube_metric(
            _job(external_id=None, platform="youtube"),
            _account("youtube"),
            http,
            "tok",
        )
    assert m.views is None
    assert m.raw_metadata == {"skipped": "no external_id"}


@respx.mock
async def test_youtube_fetcher_handles_string_view_counts() -> None:
    """Data API returns viewCount as a string; we coerce to int."""
    respx.get("https://www.googleapis.com/youtube/v3/videos").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "x",
                        "statistics": {
                            "viewCount": "999999",
                            "likeCount": "not-a-number",  # garbage from API
                        },
                    }
                ]
            },
        )
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_youtube_metric(
            _job(external_id="x", platform="youtube"),
            _account("youtube"),
            http,
            "tok",
        )
    assert m.views == 999999
    assert m.likes is None  # garbage falls through to None


# ---- TikTok ----


@respx.mock
async def test_tiktok_fetcher_happy_path() -> None:
    respx.post("https://open.tiktokapis.com/v2/research/video/query/").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "videos": [
                        {
                            "id": "tt_pub",
                            "view_count": 1200,
                            "like_count": 88,
                            "comment_count": 5,
                            "share_count": 3,
                        }
                    ]
                }
            },
        )
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_tiktok_metric(
            _job(external_id="tt_pub", platform="tiktok"),
            _account("tiktok"),
            http,
            "tok",
        )
    assert m.views == 1200
    assert m.shares == 3
    assert m.retention_pct is None
    assert m.raw_metadata is not None


@respx.mock
async def test_tiktok_fetcher_403_records_null_metric() -> None:
    """Sandbox-tier accounts get 403 from the Research API — we record but don't crash."""
    respx.post("https://open.tiktokapis.com/v2/research/video/query/").mock(
        return_value=httpx.Response(403, text="research api not authorized")
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_tiktok_metric(
            _job(external_id="tt_pub", platform="tiktok"),
            _account("tiktok"),
            http,
            "tok",
        )
    assert m.views is None
    assert m.raw_metadata is not None
    assert "error" in m.raw_metadata


@respx.mock
async def test_tiktok_fetcher_empty_response() -> None:
    respx.post("https://open.tiktokapis.com/v2/research/video/query/").mock(
        return_value=httpx.Response(200, json={"data": {"videos": []}})
    )
    async with httpx.AsyncClient() as http:
        m = await fetch_tiktok_metric(
            _job(external_id="tt_pub", platform="tiktok"),
            _account("tiktok"),
            http,
            "tok",
        )
    assert m.views is None
    assert m.raw_metadata == {"empty_response": True}


# ---- Buffer ----


async def test_buffer_fetcher_returns_null_metric() -> None:
    """Buffer doesn't expose analytics on our tier; the fetcher records the no-op."""
    async with httpx.AsyncClient() as http:
        m = await fetch_buffer_metric(
            _job(external_id="b_x", platform="buffer"),
            _account("buffer"),
            http,
            "tok",
        )
    assert m.views is None
    assert m.raw_metadata == {"skipped": "buffer_no_analytics"}

"""Zernio API client tests.

respx-mocks every endpoint we care about and pins the auth header,
URL shape, body shape, and response parsing. These are the contract
tests — when Zernio changes their API, these break first.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from chalybclip.integrations.zernio.client import ZernioClient, ZernioError

_BASE = "https://zernio.com/api/v1"


def _client(http: httpx.AsyncClient) -> ZernioClient:
    return ZernioClient(api_key="sk_test_abc", http=http)


async def _no_sleep(_seconds: float) -> None:
    """Injected sleep for backoff tests — never waits in real time."""
    return None


# ---- auth + ctor ----


def test_constructor_refuses_empty_api_key() -> None:
    with pytest.raises(ZernioError, match="ZERNIO_API_KEY"):
        ZernioClient(api_key="")


# ---- 429 backoff (phase 13 hardening) ----


@pytest.mark.asyncio
async def test_429_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def _responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rl"})
        return httpx.Response(200, json={"accounts": []})

    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/accounts").mock(side_effect=_responder)
            c = ZernioClient(api_key="sk_test_abc", http=http, sleep=_no_sleep)
            accts = await c.list_accounts(profile_id="ten_alice")
    assert accts == []
    assert calls["n"] == 2  # one 429, one success


@pytest.mark.asyncio
async def test_429_gives_up_after_max_retries() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            route = mock.get(f"{_BASE}/accounts").mock(
                return_value=httpx.Response(429, json={"error": "rl"})
            )
            c = ZernioClient(
                api_key="sk_test_abc", http=http, sleep=_no_sleep, max_retries=2,
            )
            with pytest.raises(ZernioError) as ei:
                await c.list_accounts(profile_id="ten_alice")
    assert ei.value.status_code == 429
    # initial + 2 retries = 3 attempts.
    assert len(route.calls) == 3


@pytest.mark.asyncio
async def test_non_429_error_does_not_retry() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            route = mock.get(f"{_BASE}/accounts").mock(
                return_value=httpx.Response(500, text="boom")
            )
            c = ZernioClient(api_key="sk_test_abc", http=http, sleep=_no_sleep)
            with pytest.raises(ZernioError):
                await c.list_accounts(profile_id="ten_alice")
    assert len(route.calls) == 1  # 500 is not retried


# ---- create_profile ----


@pytest.mark.asyncio
async def test_create_profile_sends_name_and_parses_id() -> None:
    body = {"profile": {"_id": "prof_abc123", "name": "My Brand"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/profiles").mock(
                return_value=httpx.Response(201, json=body)
            )
            profile = await _client(http).create_profile(
                name="My Brand", description="Testing the Zernio API",
            )
    assert profile.profile_id == "prof_abc123"
    assert profile.name == "My Brand"
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer sk_test_abc"
    payload = json.loads(sent.content.decode())
    assert payload["name"] == "My Brand"
    assert payload["description"] == "Testing the Zernio API"


@pytest.mark.asyncio
async def test_create_profile_missing_id_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/profiles").mock(
                return_value=httpx.Response(201, json={"profile": {"name": "X"}})
            )
            with pytest.raises(ZernioError, match="missing _id"):
                await _client(http).create_profile(name="X")


# ---- connect_url ----


@pytest.mark.asyncio
async def test_connect_url_returns_authurl_with_profile_id() -> None:
    body = {"authUrl": "https://zernio.com/oauth/tiktok?state=xyz"}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/connect/tiktok").mock(
                return_value=httpx.Response(200, json=body)
            )
            link = await _client(http).connect_url(
                "tiktok",
                profile_id="ten_alice",
                redirect_url="https://chalybclip.test/dashboard/publish/zernio/connected",
            )
    assert link.auth_url.startswith("https://zernio.com/oauth/tiktok")
    sent = route.calls.last.request
    # Bearer auth (NOT Apikey).
    assert sent.headers["Authorization"] == "Bearer sk_test_abc"
    # profileId scopes the connection to the tenant.
    assert sent.url.params.get("profileId") == "ten_alice"
    # redirect_url sends the post-OAuth popup back to OUR page instead
    # of Zernio's dashboard (white-label flow).
    assert sent.url.params.get("redirect_url") == (
        "https://chalybclip.test/dashboard/publish/zernio/connected"
    )


@pytest.mark.asyncio
async def test_connect_url_missing_authurl_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/connect/tiktok").mock(
                return_value=httpx.Response(200, json={"oops": True})
            )
            with pytest.raises(ZernioError, match="authUrl"):
                await _client(http).connect_url("tiktok", profile_id="ten_alice")


@pytest.mark.asyncio
async def test_connect_url_headless_sets_param() -> None:
    """headless=true makes Zernio's redirect carry the selection state
    (tempToken & co) so WE render the page/board picker."""
    body = {"authUrl": "https://facebook.com/oauth?x=1"}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/connect/facebook").mock(
                return_value=httpx.Response(200, json=body)
            )
            await _client(http).connect_url(
                "facebook",
                profile_id="ten_alice",
                redirect_url="https://chalybclip.test/dashboard/publish/zernio/connected?platform=facebook",
                headless=True,
            )
    sent = route.calls.last.request
    assert sent.url.params.get("headless") == "true"
    # The platform rides on the redirect_url query string, percent-
    # encoding intact, so /connected knows which chip just connected.
    assert sent.url.params.get("redirect_url") == (
        "https://chalybclip.test/dashboard/publish/zernio/connected?platform=facebook"
    )


@pytest.mark.asyncio
async def test_connect_url_default_omits_headless() -> None:
    body = {"authUrl": "https://tiktok.com/oauth"}
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            route = mock.get(f"{_BASE}/connect/tiktok").mock(
                return_value=httpx.Response(200, json=body)
            )
            await _client(http).connect_url("tiktok", profile_id="ten_alice")
    assert "headless" not in route.calls.last.request.url.params


# ---- list_accounts ----


@pytest.mark.asyncio
async def test_list_accounts_parses_rows() -> None:
    body = {
        "accounts": [
            {"platform": "tiktok", "_id": "acct_tt_1"},
            {"platform": "youtube", "_id": "acct_yt_2"},
            {"platform": "x", "no_id": True},  # dropped (no _id)
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/accounts").mock(
                return_value=httpx.Response(200, json=body)
            )
            accts = await _client(http).list_accounts(profile_id="ten_alice")
    assert {(a.platform, a.account_id) for a in accts} == {
        ("tiktok", "acct_tt_1"),
        ("youtube", "acct_yt_2"),
    }
    assert route.calls.last.request.url.params.get("profileId") == "ten_alice"


@pytest.mark.asyncio
async def test_list_accounts_filters_by_profile_id_when_present() -> None:
    body = {
        "accounts": [
            {"platform": "tiktok", "_id": "a1", "profileId": "ten_alice"},
            {"platform": "youtube", "_id": "a2", "profileId": "ten_bob"},
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/accounts").mock(
                return_value=httpx.Response(200, json=body)
            )
            accts = await _client(http).list_accounts(profile_id="ten_alice")
    assert [a.account_id for a in accts] == ["a1"]


@pytest.mark.asyncio
async def test_list_accounts_keeps_row_when_profileid_is_object() -> None:
    """Regression: a profileId returned as a nested object (or absent)
    must NOT be dropped — that bug made connected accounts vanish."""
    body = {
        "accounts": [
            {"platform": "tiktok", "_id": "a1", "profileId": {"_id": "ten_alice"}},
            {"platform": "instagram", "_id": "a2"},  # no profileId field at all
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/accounts").mock(
                return_value=httpx.Response(200, json=body)
            )
            accts = await _client(http).list_accounts(profile_id="ten_alice")
    assert {(a.platform, a.account_id) for a in accts} == {
        ("tiktok", "a1"),
        ("instagram", "a2"),
    }


# ---- transport errors wrap into ZernioError ----


@pytest.mark.asyncio
async def test_transport_error_wraps_into_zernio_error() -> None:
    """Timeouts / connect failures must surface as ZernioError so
    callers' `except ZernioError` degrades gracefully (a raw httpx
    exception 500ed the dashboard when Zernio was slow)."""
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/accounts").mock(
                side_effect=httpx.ReadTimeout("read timed out")
            )
            with pytest.raises(ZernioError, match="request failed"):
                await _client(http).list_accounts(profile_id="ten_alice")


# ---- disconnect_account ----


@pytest.mark.asyncio
async def test_disconnect_account_tolerates_empty_204() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.delete(f"{_BASE}/accounts/acc_1").mock(
                return_value=httpx.Response(204)
            )
            # Must NOT raise on an empty 204 body.
            await _client(http).disconnect_account("acc_1")
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk_test_abc"


@pytest.mark.asyncio
async def test_disconnect_account_raises_on_error() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.delete(f"{_BASE}/accounts/acc_x").mock(
                return_value=httpx.Response(404, json={"error": "not found"})
            )
            with pytest.raises(ZernioError) as ei:
                await _client(http).disconnect_account("acc_x")
    assert ei.value.status_code == 404


# ---- headless Facebook page selection ----


@pytest.mark.asyncio
async def test_list_facebook_pages_parses_rows_and_never_keeps_tokens() -> None:
    body = {
        "pages": [
            {
                "id": "123", "name": "My Brand Page", "username": "mybrand",
                "access_token": "EAAxxxxx", "category": "Brand",
                "tasks": ["MANAGE"],
            },
            {"id": "456", "name": "Side Page"},
            {"name": "no id — dropped"},
            "not-a-dict",
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(200, json=body)
            )
            pages = await _client(http).list_facebook_pages(
                profile_id="ten_alice", temp_token="EAAtmp",
            )
    sent = route.calls.last.request
    assert sent.url.params.get("profileId") == "ten_alice"
    assert sent.url.params.get("tempToken") == "EAAtmp"
    assert [(p.page_id, p.name, p.username, p.category) for p in pages] == [
        ("123", "My Brand Page", "mybrand", "Brand"),
        ("456", "Side Page", None, None),
    ]
    # The page-scoped access_token must not survive parsing — we never
    # store or log platform tokens.
    assert not any(hasattr(p, "access_token") for p in pages)


@pytest.mark.asyncio
async def test_list_facebook_pages_missing_pages_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(200, json={"oops": True})
            )
            with pytest.raises(ZernioError, match="missing pages"):
                await _client(http).list_facebook_pages(
                    profile_id="ten_alice", temp_token="EAAtmp",
                )


@pytest.mark.asyncio
async def test_select_facebook_page_posts_selection_and_parses_account() -> None:
    body = {
        "message": "Facebook page connected successfully",
        "account": {
            "accountId": "acct_fb_1", "platform": "facebook",
            "username": "mybrand", "displayName": "My Brand Page",
        },
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(200, json=body)
            )
            acct = await _client(http).select_facebook_page(
                profile_id="ten_alice",
                page_id="123",
                temp_token="EAAtmp",
                user_profile={"id": "987", "name": "Alice"},
            )
    assert acct.platform == "facebook"
    assert acct.account_id == "acct_fb_1"
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload == {
        "profileId": "ten_alice",
        "pageId": "123",
        "tempToken": "EAAtmp",
        "userProfile": {"id": "987", "name": "Alice"},
    }


@pytest.mark.asyncio
async def test_select_facebook_page_missing_account_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(200, json={"message": "ok"})
            )
            with pytest.raises(ZernioError, match="missing account"):
                await _client(http).select_facebook_page(
                    profile_id="ten_alice", page_id="123",
                    temp_token="EAAtmp", user_profile={"id": "987"},
                )


@pytest.mark.asyncio
async def test_select_facebook_page_404_unknown_page_raises_with_status() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(
                    404, json={"error": "Selected page not found"}
                )
            )
            with pytest.raises(ZernioError) as ei:
                await _client(http).select_facebook_page(
                    profile_id="ten_alice", page_id="999",
                    temp_token="EAAtmp", user_profile={"id": "987"},
                )
    assert ei.value.status_code == 404


# ---- create_post ----


@pytest.mark.asyncio
async def test_create_post_sends_media_url_accounts_and_tiktok_consent() -> None:
    body = {"success": True, "post": {"_id": "post_123"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/posts").mock(
                return_value=httpx.Response(201, json=body)
            )
            result = await _client(http).create_post(
                profile_id="ten_alice",
                content="My caption",
                media_url="https://chalybclip.test/api/internal/clip/clp_x?sig=...",
                platforms=[("tiktok", "acct_tt_1"), ("youtube", "acct_yt_2")],
                publish_now=True,
                tiktok_settings={
                    "content_preview_confirmed": True,
                    "express_consent_given": True,
                },
            )
    assert result.post_id == "post_123"

    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer sk_test_abc"
    assert sent.headers["Content-Type"].startswith("application/json")
    payload = json.loads(sent.content.decode())
    # Media referenced BY URL (no presigned upload in the common path).
    assert payload["mediaItems"] == [
        {"type": "video", "url": "https://chalybclip.test/api/internal/clip/clp_x?sig=..."}
    ]
    # Per-platform accountId is required on every platforms[] entry.
    assert payload["platforms"] == [
        {"platform": "tiktok", "accountId": "acct_tt_1"},
        {"platform": "youtube", "accountId": "acct_yt_2"},
    ]
    assert payload["publishNow"] is True
    assert "scheduledFor" not in payload
    # TikTok legal consent flags must be present.
    assert payload["tiktokSettings"]["content_preview_confirmed"] is True
    assert payload["tiktokSettings"]["express_consent_given"] is True


@pytest.mark.asyncio
async def test_create_post_scheduled_sets_scheduledfor_not_publishnow() -> None:
    body = {"success": True, "post": {"_id": "post_sched"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/posts").mock(
                return_value=httpx.Response(201, json=body)
            )
            await _client(http).create_post(
                profile_id="ten_alice",
                content="Later",
                media_url="https://x.test/v.mp4",
                platforms=[("tiktok", "acct_tt_1")],
                scheduled_for="2026-12-31T23:00:00Z",
                timezone="America/New_York",
            )
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["scheduledFor"] == "2026-12-31T23:00:00Z"
    assert payload["timezone"] == "America/New_York"
    assert "publishNow" not in payload


@pytest.mark.asyncio
async def test_create_post_5xx_raises_with_status() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/posts").mock(
                return_value=httpx.Response(500, text="oops")
            )
            with pytest.raises(ZernioError) as ei:
                await _client(http).create_post(
                    profile_id="ten_alice",
                    content="x",
                    media_url="https://x.test/v.mp4",
                    platforms=[("tiktok", "acct_tt_1")],
                )
    assert ei.value.status_code == 500


# ---- get_post ----


@pytest.mark.asyncio
async def test_get_post_parses_status_and_platforms() -> None:
    body = {
        "post": {
            "_id": "post_123",
            "status": "published",
            "platforms": [
                {"platform": "tiktok", "status": "published", "url": "https://tiktok.com/x"}
            ],
        }
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{_BASE}/posts/post_123").mock(
                return_value=httpx.Response(200, json=body)
            )
            s = await _client(http).get_post("post_123")
    assert s.post_id == "post_123"
    assert s.status == "published"
    assert isinstance(s.platforms, list)
    assert s.platforms[0]["url"].startswith("https://tiktok.com/")


# ---- list_posts ----


@pytest.mark.asyncio
async def test_list_posts_passes_pagination() -> None:
    body = {"posts": [{"_id": "p1"}], "total": 1}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/posts").mock(
                return_value=httpx.Response(200, json=body)
            )
            out = await _client(http).list_posts(page=2, limit=50)
    assert out["posts"][0]["_id"] == "p1"
    sent = route.calls.last.request
    assert sent.url.params.get("page") == "2"
    assert sent.url.params.get("limit") == "50"


@pytest.mark.asyncio
async def test_list_posts_status_and_profile_filters() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/posts").mock(
                return_value=httpx.Response(200, json={"posts": []})
            )
            await _client(http).list_posts(
                status="scheduled", profile_id="ten_alice", sort_by="scheduled-asc",
            )
    sent = route.calls.last.request
    assert sent.url.params.get("status") == "scheduled"
    assert sent.url.params.get("profileId") == "ten_alice"
    assert sent.url.params.get("sortBy") == "scheduled-asc"


# ---- queue slots ----


@pytest.mark.asyncio
async def test_list_queues_returns_schedules() -> None:
    body = {
        "queues": [
            {"_id": "q1", "name": "Morning", "isDefault": True,
             "timezone": "UTC", "slots": [{"dayOfWeek": 1, "time": "09:00"}],
             "active": True},
        ],
        "count": 1,
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/queue/slots").mock(
                return_value=httpx.Response(200, json=body)
            )
            queues = await _client(http).list_queues(profile_id="ten_alice")
    assert [q["_id"] for q in queues] == ["q1"]
    sent = route.calls.last.request
    assert sent.url.params.get("profileId") == "ten_alice"
    assert sent.url.params.get("all") == "true"


@pytest.mark.asyncio
async def test_list_queues_empty_when_none() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/queue/slots").mock(
                return_value=httpx.Response(200, json={"queues": [], "count": 0})
            )
            assert await _client(http).list_queues(profile_id="x") == []


@pytest.mark.asyncio
async def test_upsert_default_queue_sends_slots() -> None:
    body = {
        "success": True,
        "schedule": {"_id": "q9", "name": "ChalyClip Queue", "slots": []},
    }
    slots = [{"dayOfWeek": 1, "time": "09:00"}, {"dayOfWeek": 5, "time": "18:30"}]
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.put(f"{_BASE}/queue/slots").mock(
                return_value=httpx.Response(200, json=body)
            )
            sched = await _client(http).upsert_default_queue(
                profile_id="ten_alice", slots=slots, timezone="America/Mexico_City",
            )
    assert sched["_id"] == "q9"
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["profileId"] == "ten_alice"
    assert payload["timezone"] == "America/Mexico_City"
    assert payload["slots"] == slots
    assert payload["active"] is True
    # No queueId → Zernio targets the DEFAULT queue.
    assert "queueId" not in payload


@pytest.mark.asyncio
async def test_delete_queue_passes_ids() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.delete(f"{_BASE}/queue/slots").mock(
                return_value=httpx.Response(200, json={"success": True})
            )
            await _client(http).delete_queue(profile_id="ten_alice", queue_id="q1")
    sent = route.calls.last.request
    assert sent.url.params.get("profileId") == "ten_alice"
    assert sent.url.params.get("queueId") == "q1"


# ---- failed posts + retry ----


@pytest.mark.asyncio
async def test_list_failed_uses_status_query_not_missing_path() -> None:
    """Regression: there is NO /posts/failed endpoint — failed posts
    come from GET /posts?status=failed (the old path 404'd)."""
    body = {"posts": [{"_id": "pf1", "status": "failed"}, "skip-non-dict"]}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/posts").mock(
                return_value=httpx.Response(200, json=body)
            )
            failed = await _client(http).list_failed(profile_id="ten_alice")
    assert [p["_id"] for p in failed] == ["pf1"]
    sent = route.calls.last.request
    assert sent.url.params.get("status") == "failed"
    assert sent.url.params.get("profileId") == "ten_alice"


@pytest.mark.asyncio
async def test_retry_post_posts_to_retry_endpoint() -> None:
    body = {"message": "Post published successfully",
            "post": {"_id": "pf1", "status": "published"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/posts/pf1/retry").mock(
                return_value=httpx.Response(200, json=body)
            )
            result = await _client(http).retry_post("pf1")
    assert result.post_id == "pf1"
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk_test_abc"


# ---- analytics ----


@pytest.mark.asyncio
async def test_post_analytics_list_passes_dates_and_profile() -> None:
    body = {"overview": {"totalPosts": 1}, "posts": [{"_id": "p1"}]}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/analytics").mock(
                return_value=httpx.Response(200, json=body)
            )
            out = await _client(http).post_analytics_list(
                profile_id="ten_alice", from_date="2026-05-01", to_date="2026-06-01",
            )
    assert out["posts"][0]["_id"] == "p1"
    sent = route.calls.last.request
    assert sent.url.params.get("profileId") == "ten_alice"
    assert sent.url.params.get("fromDate") == "2026-05-01"
    assert sent.url.params.get("toDate") == "2026-06-01"


@pytest.mark.asyncio
async def test_post_analytics_one_passes_post_id() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/analytics").mock(
                return_value=httpx.Response(200, json={"postId": "p1"})
            )
            await _client(http).post_analytics_one("p1")
    assert route.calls.last.request.url.params.get("postId") == "p1"


# ---- inbox ----


@pytest.mark.asyncio
async def test_reply_to_comment_body() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/inbox/comments/pp1").mock(
                return_value=httpx.Response(200, json={"success": True})
            )
            await _client(http).reply_to_comment(
                "pp1", account_id="acc1", message="hi", comment_id="c1",
            )
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload == {"accountId": "acc1", "message": "hi", "commentId": "c1"}


@pytest.mark.asyncio
async def test_send_message_with_and_without_attachment() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/inbox/conversations/conv1/messages").mock(
                return_value=httpx.Response(200, json={"success": True})
            )
            await _client(http).send_message(
                "conv1", account_id="acc1", message="hola",
                attachment_url="https://x/y.jpg",
            )
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["attachmentUrl"] == "https://x/y.jpg"


@pytest.mark.asyncio
async def test_set_conversation_status_archives() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.put(f"{_BASE}/inbox/conversations/conv1").mock(
                return_value=httpx.Response(200, json={"success": True})
            )
            await _client(http).set_conversation_status(
                "conv1", account_id="acc1", status="archived",
            )
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload == {"accountId": "acc1", "status": "archived"}


# ---- growth layer ----


@pytest.mark.asyncio
async def test_create_comment_automation_body() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/comment-automations").mock(
                return_value=httpx.Response(200, json={"id": "auto1"})
            )
            await _client(http).create_comment_automation(
                profile_id="prof", account_id="acc1", name="Clip link",
                dm_message="Aquí 👉 {url}", keywords=["CLIP", "LINK"],
                platform_post_id="pp1", comment_reply="¡Revisa tu DM!",
            )
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["accountId"] == "acc1"
    assert payload["keywords"] == ["CLIP", "LINK"]
    assert payload["platformPostId"] == "pp1"
    assert payload["dmMessage"] == "Aquí 👉 {url}"
    assert payload["isActive"] is True


@pytest.mark.asyncio
async def test_create_sequence_sends_steps() -> None:
    steps = [
        {"order": 1, "delayMinutes": 0, "message": {"text": "Bienvenida"}},
        {"order": 2, "delayMinutes": 1440, "message": {"text": "Tips"}},
    ]
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/sequences").mock(
                return_value=httpx.Response(200, json={"id": "seq1"})
            )
            await _client(http).create_sequence(
                profile_id="prof", account_id="acc1", platform="instagram",
                name="Bienvenida Chalyb", steps=steps,
            )
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["steps"] == steps
    assert payload["platform"] == "instagram"


@pytest.mark.asyncio
async def test_enroll_in_sequence() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/sequences/seq1/enroll").mock(
                return_value=httpx.Response(200, json={"enrolled": 2, "skipped": 0})
            )
            out = await _client(http).enroll_in_sequence(
                "seq1", contact_ids=["c1", "c2"],
            )
    assert out["enrolled"] == 2
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload == {"contactIds": ["c1", "c2"]}


@pytest.mark.asyncio
async def test_broadcast_create_recipients_send_flow() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            create = mock.post(f"{_BASE}/broadcasts").mock(
                return_value=httpx.Response(200, json={"id": "b1"})
            )
            recips = mock.post(f"{_BASE}/broadcasts/b1/recipients").mock(
                return_value=httpx.Response(200, json={"added": 3, "skipped": 0})
            )
            send = mock.post(f"{_BASE}/broadcasts/b1/send").mock(
                return_value=httpx.Response(200, json={"status": "sending", "sent": 3})
            )
            await _client(http).create_broadcast(
                profile_id="prof", account_id="acc1", platform="instagram",
                name="Promo", message_text="¡Nuevo clip!",
            )
            await _client(http).add_broadcast_recipients("b1", contact_ids=["c1"])
            out = await _client(http).send_broadcast("b1")
    assert create.called and recips.called and send.called
    assert out["status"] == "sending"
    create_body = json.loads(create.calls.last.request.content.decode())
    assert create_body["message"] == {"text": "¡Nuevo clip!"}


# ---- ads (phase 12) ----


@pytest.mark.asyncio
async def test_boost_post_body() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/ads/boost").mock(
                return_value=httpx.Response(200, json={"campaignId": "c1"})
            )
            await _client(http).boost_post(
                account_id="acc1", ad_account_id="act_1", name="Boost",
                goal="engagement", budget_amount=20.0, budget_type="daily",
                platform_post_id="pp1",
            )
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["adAccountId"] == "act_1"
    assert payload["budget"] == {"amount": 20.0, "type": "daily"}
    assert payload["platformPostId"] == "pp1"
    assert "postId" not in payload


@pytest.mark.asyncio
async def test_list_ad_campaigns() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/ads/campaigns").mock(
                return_value=httpx.Response(200, json={"campaigns": [{"id": "c1"}]})
            )
            out = await _client(http).list_ad_campaigns(profile_id="prof")
    assert out[0]["id"] == "c1"
    assert route.calls.last.request.url.params.get("profileId") == "prof"

"""HTMX dashboard - login flow + page rendering + status PATCH."""

from __future__ import annotations

import datetime as _dt

import httpx

from chalybclip.db import (
    ClipsRepo,
    Database,
    StreamsRepo,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
)
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def test_dashboard_redirects_to_login_when_unauthed(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get("/dashboard/streams", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/login"


async def test_login_form_renders(client: httpx.AsyncClient) -> None:
    r = await client.get("/dashboard/login")
    assert r.status_code == 200
    assert "API token" in r.text
    assert 'name="token"' in r.text


async def test_login_with_unknown_token_shows_error(client: httpx.AsyncClient) -> None:
    r = await client.post("/dashboard/login", data={"token": "tok_nope"})
    assert r.status_code == 401
    assert "unknown token" in r.text


async def test_login_with_valid_token_sets_cookie(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.post(
        "/dashboard/login",
        data={"token": tenants["alice"]["token"]},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/streams"
    assert "chalybclip_token" in r.headers.get("set-cookie", "")


async def test_streams_page_renders_with_cookie_auth(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    tenant_id = tenants["alice"]["id"]
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_dash1",
                tenant_id=tenant_id,
                vod_url="https://example.com/v",
                platform="kick",
                title="Live show",
                channel="aldo",
                duration_s=600.0,
                source_video_path="/tmp/x.mp4",
                source_audio_path="/tmp/x.wav",
                status="ingested",
                created_at=_now(),
            )
        )
    # Sign in.
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams")
    assert r.status_code == 200
    assert "Live show" in r.text
    # Both ingest paths render: upload (primary) and URL (advanced).
    assert "Upload a video" in r.text
    assert "Advanced: process from a VOD URL" in r.text


async def test_dashboard_html_is_no_store(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    # Per-tenant dashboard pages must never be cached — a stale browser /
    # bfcache / edge copy makes freshly uploaded streams and generated clips
    # appear to "not show up". The auth middleware stamps no-store on
    # authenticated HTML responses.
    r = await client.get(
        "/dashboard/streams",
        headers={"Authorization": f"Bearer {tenants['alice']['token']}"},
    )
    assert r.status_code == 200
    assert "no-store" in r.headers.get("cache-control", "")


async def test_personas_page_renders_and_creates(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/personas")
    assert r.status_code == 200
    assert "No personas yet" in r.text

    r = await client.post(
        "/dashboard/personas",
        data={
            "id": "aldo",
            "name": "Aldo",
            "primary_language": "es",
            "voice_prompt": "direct",
            "target_languages": "es, en",
            "routing_tags": "mindset,irl",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303

    r = await client.get("/dashboard/personas")
    assert "Aldo" in r.text
    assert "aldo" in r.text


async def test_logout_clears_cookie(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.post("/dashboard/logout", follow_redirects=False)
    assert r.status_code == 303
    # Subsequent dashboard request should redirect to login again.
    r = await client.get("/dashboard/streams", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/login"


# Slice H.1 — the dashboard's manual `PATCH /dashboard/clips/{id}/status`
# endpoint was removed (status is now derived from lifecycle handlers:
# Finalize → approved, Reject → rejected, publish worker → published).
# The two former tests for that endpoint were deleted along with it.
# Headless automations that still need manual transitions can hit the
# JSON API `PATCH /api/clips/{id}/status` — see tests/api/test_clips.py.

"""Task 3 — Drop-a-URL self-serve ingest endpoints.

NB: dashboard cookie-login flow is pre-broken in the test fixtures
(slice O.22 redirected login to chalyb; tests never got updated).
We use bearer-header auth here — the auth middleware accepts either
mechanism for /dashboard/* routes, and bearer is what the other
passing API tests use.
"""

from __future__ import annotations

import httpx

from chalybclip.db import Database, PersonasRepo, StreamsRepo
from chalybclip.tenancy import bound_tenant


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _seed_persona(
    db: Database, tenant_id: str, persona_id: str = "p1"
) -> None:
    """Insert a persona directly via the repo — sidesteps the broken
    dashboard form path."""
    with bound_tenant(tenant_id):
        await PersonasRepo(db).upsert(
            persona_id=persona_id,
            name="Test Persona",
            primary_language="es",
            voice_prompt="direct",
            target_languages=["es"],
            routing_tags=[],
        )


async def test_legacy_url_job_form_redirects_to_unified_start(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The old single-input /dashboard/url-jobs/new now 303s to
    /dashboard/start — bookmarked links from the Task 3 era keep
    landing the operator somewhere useful."""
    r = await client.get(
        "/dashboard/url-jobs/new",
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/start"


async def test_unified_start_page_renders_all_entry_points(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The unified start page surfaces every ingest method as a card:
    paste URL, upload, live RTMP, Drive watch. Persona is auto-resolved
    so there's no picker UI."""
    tenant_id = tenants["alice"]["id"]
    await _seed_persona(db, tenant_id)

    r = await client.get(
        "/dashboard/start", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200, r.text
    # All four entry-point cards present.
    assert 'data-entry="url"' in r.text
    assert 'data-entry="upload"' in r.text
    assert 'data-entry="live"' in r.text
    assert 'data-entry="drive"' in r.text
    # Same single-input pattern as Task 3 — no persona picker.
    assert 'name="vod_url"' in r.text
    assert 'name="persona_id"' not in r.text or 'type="hidden"' in r.text
    # Auto-resolved persona name surfaces (with the hidden field on the
    # upload form so the legacy endpoint accepts the post).
    assert "Test Persona" in r.text


async def test_start_page_warns_when_no_persona(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    """Operator needs to make a persona first — surface that on the
    unified page, don't fail at submit time."""
    r = await client.get(
        "/dashboard/start", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200, r.text
    assert "You need a persona first" in r.text


async def test_url_job_submit_creates_placeholder_stream(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Successful submit inserts a placeholder StreamRow with the URL +
    detected platform, then redirects to the existing stream detail page.
    """
    tenant_id = tenants["alice"]["id"]
    await _seed_persona(db, tenant_id)

    r = await client.post(
        "/dashboard/url-jobs",
        data={"vod_url": "https://kick.com/aldovillanueva/videos/abc"},
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    location = r.headers["location"]
    assert location.startswith("/dashboard/streams/str_")
    stream_id = location.removeprefix("/dashboard/streams/")

    # Placeholder row landed with the right tenant + URL + platform.
    with bound_tenant(tenant_id):
        row = await StreamsRepo(db).get(stream_id)
    assert row is not None
    assert row.vod_url == "https://kick.com/aldovillanueva/videos/abc"
    assert row.platform == "kick"
    assert row.tenant_id == tenant_id
    # duration_s starts at 0; the pipeline will UPSERT real metadata once
    # ingest finishes.
    assert row.duration_s == 0.0


async def test_url_job_rejects_unknown_platform(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed_persona(db, tenant_id)

    r = await client.post(
        "/dashboard/url-jobs",
        data={"vod_url": "https://example.com/some-video.mp4"},
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )
    assert r.status_code == 400, r.text
    assert "supported" in r.text.lower()


async def test_url_job_rejects_empty_url(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed_persona(db, tenant_id)

    r = await client.post(
        "/dashboard/url-jobs",
        data={"vod_url": "   "},
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )
    assert r.status_code == 400, r.text


async def test_url_job_rejects_when_no_persona(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    """Direct POST (curl / scripted) with no persona configured returns
    400 with an actionable hint — the form already disables submit, but
    the API surface still needs to fail loudly."""
    r = await client.post(
        "/dashboard/url-jobs",
        data={"vod_url": "https://kick.com/x/videos/1"},
        headers=auth(tenants["alice"]["token"]),
        follow_redirects=False,
    )
    assert r.status_code == 400, r.text
    assert "persona" in r.text.lower()


async def test_url_job_accepts_twitch_and_youtube(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """All three target platforms produce a placeholder row + 303 redirect.
    Parameterized as a single test so the platform-detect contract gets
    pinned to its actual call site here, not separately in the ingest
    module."""
    tenant_id = tenants["alice"]["id"]
    await _seed_persona(db, tenant_id)

    for url, expected_platform in [
        ("https://www.twitch.tv/foo/videos/1", "twitch"),
        ("https://youtu.be/abc123", "youtube"),
    ]:
        r = await client.post(
            "/dashboard/url-jobs",
            data={"vod_url": url},
            headers=auth(tenants["alice"]["token"]),
            follow_redirects=False,
        )
        assert r.status_code == 303, (url, r.text)
        stream_id = r.headers["location"].removeprefix("/dashboard/streams/")
        with bound_tenant(tenant_id):
            row = await StreamsRepo(db).get(stream_id)
        assert row is not None
        assert row.platform == expected_platform, (url, row.platform)


async def test_start_page_appears_in_nav(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The '+ New clip' nav entry is THE entry point for new operators
    — pin its presence so a future nav cleanup doesn't accidentally
    hide the first-run aha."""
    await _seed_persona(db, tenants["alice"]["id"])
    r = await client.get(
        "/dashboard/start", headers=auth(tenants["alice"]["token"])
    )
    assert "/dashboard/start" in r.text
    assert "+ New clip" in r.text


async def test_start_page_returns_303_when_unauth(
    client: httpx.AsyncClient,
) -> None:
    """No auth → bounce. Same contract every dashboard page has."""
    r = await client.get("/dashboard/start", follow_redirects=False)
    assert r.status_code in (303, 401), r.text



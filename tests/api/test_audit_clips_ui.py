"""Audit 2026-10-08 — hub users (Spanish-speaking) on the Clips start page.

Pins: the dashboard renders Spanish even when the browser says English,
`?lang=` persists a choice, the tier chip shows a human label (not the raw
`all_access` enum), operator-only setup text (env var names) never reaches
non-admins, and the live panel never hands out a fake stream key.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from chalybclip.db import Database, EventsRepo
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

from .conftest import auth

_EN = {"Accept-Language": "en-US,en;q=0.9"}


@pytest.fixture
def clean_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    monkeypatch.delenv("CHALYBCLIP_ADMIN_TENANT_IDS", raising=False)
    monkeypatch.delenv("CHALYBCLIP_COOKIES_FILE", raising=False)
    monkeypatch.delenv("CHALYBCLIP_LIVE_RTMP_BASE_URL", raising=False)
    get_settings.cache_clear()
    try:
        yield monkeypatch
    finally:
        get_settings.cache_clear()


async def _start(client: httpx.AsyncClient, token: str, **headers: str) -> str:
    r = await client.get("/dashboard/start", headers={**auth(token), **headers})
    assert r.status_code == 200, r.text[:300]
    return r.text


async def test_dashboard_is_spanish_even_for_english_browser(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    body = await _start(client, tenants["alice"]["token"], **_EN)
    assert "Pega una URL" in body
    assert "Nuevo clip" in body
    assert "En vivo" in body
    assert "Paste a URL" not in body
    assert ">Live<" not in body


async def test_lang_param_overrides_and_is_remembered(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    token = tenants["alice"]["token"]
    r = await client.get("/dashboard/start?lang=en", headers=auth(token))
    assert "Paste a URL" in r.text
    assert "chalybclip_lang=en" in r.headers.get("set-cookie", "")
    client.cookies.set("chalybclip_lang", "en")
    assert "Paste a URL" in await _start(client, token)


async def test_tier_chip_shows_human_label_not_raw_enum(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    await sync_tenant_tier(db, tenant_id=tenants["alice"]["id"], tier="vip")
    body = await _start(client, tenants["alice"]["token"])
    assert "All_access" not in body
    assert "<span>VIP</span>" in body


async def test_operator_setup_text_hidden_from_creators(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    body = await _start(client, tenants["alice"]["token"])
    assert "CHALYBCLIP_COOKIES_FILE" not in body
    assert "CHALYBCLIP_LIVE_RTMP_BASE_URL" not in body
    assert "cookies.txt" not in body
    # The plain-language alternative is there instead.
    assert "usa Subir" in body


async def test_operator_setup_text_still_shown_to_admins(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    clean_settings.setenv("CHALYBCLIP_ADMIN_TENANT_IDS", tenants["alice"]["id"])
    get_settings.cache_clear()
    body = await _start(client, tenants["alice"]["token"])
    assert "CHALYBCLIP_COOKIES_FILE" in body


async def test_live_panel_never_shows_placeholder_key(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    from chalybclip.db import LiveStreamKeysRepo

    clean_settings.setenv("CHALYBCLIP_LIVE_RTMP_BASE_URL", "rtmp://live.example/live")
    get_settings.cache_clear()
    token = tenants["alice"]["token"]
    body = await _start(client, token)
    assert "nx_live_placeholder_key" not in body
    assert "Todavía no tienes clave" in body

    with bound_tenant(tenants["alice"]["id"]):
        await LiveStreamKeysRepo(db).rotate_for_tenant()
        key = await LiveStreamKeysRepo(db).get_active_for_tenant()
    assert key is not None
    body = await _start(client, token)
    assert key.key_value in body


async def test_llm_settings_page_is_admin_only(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    r = await client.get(
        "/dashboard/settings/llm", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 404


async def test_balance_refresh_returns_to_same_host_referer(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ok(_db: object, *, tenant_id: str) -> bool:
        return True

    monkeypatch.setattr(
        "chalybclip.integrations.chalyb.balance.fetch_balance_now", _ok
    )
    token = tenants["alice"]["token"]
    r = await client.get(
        "/dashboard/_balance/refresh",
        headers={**auth(token), "Referer": "http://api.test/dashboard/start?x=1"},
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/start?x=1"
    r = await client.get(
        "/dashboard/_balance/refresh",
        headers={**auth(token), "Referer": "https://evil.example/dashboard/start"},
    )
    assert r.headers["location"] == "/dashboard/streams"


async def test_events_list_for_stream_reaches_old_events(
    db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """A stream's events stay reachable no matter how many newer events
    the tenant has (the progress card + recovery sweeper rely on it)."""
    with bound_tenant(tenants["alice"]["id"]):
        repo = EventsRepo(db)
        await repo.emit(type="stream.processed", payload={"stream_id": "str_old"})
        for i in range(30):
            await repo.emit(type="noise", payload={"stream_id": f"str_new{i}"})
        assert all(
            e.payload.get("stream_id") != "str_old"
            for e in await repo.list_for_tenant(limit=10)
        )
        evs = await repo.list_for_stream("str_old")
    assert [e.type for e in evs] == ["stream.processed"]


async def _seed_upload_stream(db: Database, tenant_id: str, stream_id: str) -> None:
    import datetime as _dt

    from chalybclip.db import StreamsRepo
    from chalybclip.db.models import StreamRow

    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id=tenant_id,
                vod_url=f"upload://{stream_id}.mp4", platform="upload",
                title="T", channel=None, duration_s=60.0,
                source_video_path=f"/tmp/{stream_id}.mp4",
                source_audio_path=f"/tmp/{stream_id}.wav",
                status="ingested",
                created_at=_dt.datetime.now(_dt.UTC).isoformat(),
            )
        )


async def test_progress_hides_raw_exception_from_creators(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tid = tenants["alice"]["id"]
    await _seed_upload_stream(db, tid, "str_raw")
    with bound_tenant(tid):
        await EventsRepo(db).emit(
            type="pipeline.failed",
            payload={
                "stream_id": "str_raw",
                "error_type": "ModalDispatchError",
                "error": "modal pipeline returned HTTP 500: <internal body>",
            },
        )
    r = await client.get(
        "/dashboard/streams/str_raw/progress", headers=auth(tenants["alice"]["token"])
    )
    assert r.status_code == 200
    assert "ModalDispatchError" not in r.text
    assert "internal body" not in r.text
    assert "No pudimos procesar tu video" in r.text


async def test_progress_ignores_failure_from_before_a_rerun(
    clean_settings: pytest.MonkeyPatch,
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tid = tenants["alice"]["id"]
    await _seed_upload_stream(db, tid, "str_rerun")
    with bound_tenant(tid):
        await EventsRepo(db).emit(
            type="pipeline.failed",
            payload={"stream_id": "str_rerun", "error_type": "X", "error": "old"},
        )
        await EventsRepo(db).emit(
            type="stream.rerun_requested", payload={"stream_id": "str_rerun"}
        )
    r = await client.get(
        "/dashboard/streams/str_rerun/progress", headers=auth(tenants["alice"]["token"])
    )
    assert "No pudimos procesar tu video" not in r.text
    assert 'hx-trigger="every 3s"' in r.text

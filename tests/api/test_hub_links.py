"""Hops from Clips back to the hub keep the language the hub launched with
(Aldo, 2026-11-07, "Visible text only"): English under /en, Spanish unchanged."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from chalybclip.api.hub_links import hub_url
from chalybclip.api.status_gate import require_paid_tier, require_top_tier
from chalybclip.db import Database
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.settings import get_settings

_SECRET = "sso_shared_secret_value"


@pytest.mark.parametrize(
    "url,want",
    [
        ("https://www.chalyb.com/app/clips", "https://www.chalyb.com/en/app/clips"),
        ("https://www.chalyb.com/app/clips/videos/s1", "https://www.chalyb.com/en/app/clips/videos/s1"),
        ("https://chalyb.com/login", "https://chalyb.com/en/login"),
        ("https://chalyb.com", "https://chalyb.com/en"),
        ("https://chalyb.com/", "https://chalyb.com/en"),
        ("https://chalyb.com/en/app/clips", "https://chalyb.com/en/app/clips"),
        ("https://chalyb.com/es/sign-in", "https://chalyb.com/es/sign-in"),
        ("https://chalyb.com/app/x?a=1", "https://chalyb.com/en/app/x?a=1"),
        ("https://chalyb.com/auth/launch/chalybobs", "https://chalyb.com/auth/launch/chalybobs?lang=en"),
        ("https://chalyb.com/auth/launch/x?via=hub&lang=es", "https://chalyb.com/auth/launch/x?via=hub&lang=en"),
        ("/app/clips", "/en/app/clips"),
        ("https://chalyb.com/english", "https://chalyb.com/en/english"),
    ],
)
def test_hub_url_english(url: str, want: str) -> None:
    assert hub_url(url, "en") == want


@pytest.mark.parametrize("lang", ["es", None, "fr", ""])
@pytest.mark.parametrize(
    "url", ["https://www.chalyb.com/app/clips", "https://chalyb.com/login", "/auth/launch/chalybobs"]
)
def test_hub_url_unchanged_unless_english(url: str, lang: str | None) -> None:
    assert hub_url(url, lang) == url


# ── HubRedirectMiddleware ──────────────────────────────────────────────────


@pytest.fixture
def redirect_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHALYBCLIP_HUB_REDIRECT", "1")
    monkeypatch.setenv("CHALYB_BASE_URL", "https://www.chalyb.com")


async def test_redirect_spanish_by_default(client: httpx.AsyncClient, redirect_on: None) -> None:
    r = await client.get("/dashboard/streams/str_9", follow_redirects=False)
    assert r.headers["location"] == "https://www.chalyb.com/app/clips/videos/str_9"


async def test_redirect_english_from_query(client: httpx.AsyncClient, redirect_on: None) -> None:
    r = await client.get("/dashboard/start?lang=en", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "https://www.chalyb.com/en/app/clips"


async def test_redirect_english_from_cookie(client: httpx.AsyncClient, redirect_on: None) -> None:
    client.cookies.set("chalybclip_lang", "en")
    r = await client.get("/dashboard/brand-kits/brk_1", follow_redirects=False)
    assert r.headers["location"] == "https://www.chalyb.com/en/app/clips/ajustes/estilos/brk_1"
    r = await client.get("/dashboard?lang=es", follow_redirects=False)
    assert r.headers["location"] == "https://www.chalyb.com/app/clips"


# ── /auth/sso keeps the hub's language for the next hops ──────────────────


@pytest.fixture
def sso_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    get_settings.cache_clear()
    try:
        yield _SECRET
    finally:
        get_settings.cache_clear()


def _token(tenant_id: str) -> str:
    return sign_sso_token(
        user_id="chalyb_user_1", email="alice@example.com", tenant_id=tenant_id, secret=_SECRET
    )


@pytest.mark.parametrize(
    "lang,cookie,landing",
    [
        ("en", "en", "https://www.chalyb.com/en/app/clips"),
        ("es", "es", "https://www.chalyb.com/app/clips"),
        (None, None, "https://www.chalyb.com/app/clips"),
        ("fr", None, "https://www.chalyb.com/app/clips"),
    ],
)
@pytest.mark.parametrize("stale", [None, "en"])
async def test_sso_then_hub_redirect_keeps_language(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
    redirect_on: None,
    lang: str | None,
    cookie: str | None,
    landing: str,
    stale: str | None,
) -> None:
    # The hub sends `lang` only for English: a Spanish launch must undo an old en choice.
    if stale:
        client.cookies.set("chalybclip_lang", stale, domain="api.test")
    q = f"&lang={lang}" if lang else ""
    r = await client.get(f"/auth/sso?token={_token(tenants['alice']['id'])}{q}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/dashboard/start"
    assert client.cookies.get("chalybclip_lang") == cookie
    # The real next hop: the browser follows to /dashboard/start with the cookies.
    r2 = await client.get("/dashboard/start", follow_redirects=False)
    assert r2.status_code == 303
    assert r2.headers["location"] == landing


# ── signed-out bounce, logout, upgrade links ──────────────────────────────


async def test_signed_out_dashboard_bounces_to_english_login(client: httpx.AsyncClient) -> None:
    client.cookies.clear()
    r = await client.get("/dashboard/streams?lang=en", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].endswith("/en/login")
    r = await client.get("/dashboard/streams", follow_redirects=False)
    assert r.headers["location"] == "https://chalyb.com/login"


async def test_logout_goes_to_english_login(client: httpx.AsyncClient) -> None:
    client.cookies.set("chalybclip_lang", "en")
    r = await client.post("/dashboard/logout", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("/en/login")


def _req(tier: str, lang: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(tenant_tier=tier),
        query_params={"lang": lang} if lang else {},
        cookies={},
    )


@pytest.mark.parametrize("gate", [require_paid_tier, require_top_tier])
@pytest.mark.parametrize(
    "lang,want",
    [("en", "https://chalyb.com/en/app/subscription"), (None, "https://chalyb.com/app/subscription")],
)
def test_upgrade_link_language(gate, lang: str | None, want: str) -> None:
    with pytest.raises(HTTPException) as ei:
        gate(_req("free", lang))
    assert ei.value.detail["upgrade_url"] == want


async def test_live_page_links_to_en_vivo_in_english(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await sync_tenant_tier(db, tenant_id=tenants["alice"]["id"], tier="all_access")
    monkeypatch.delenv("CHALYBCLIP_HUB_REDIRECT", raising=False)
    monkeypatch.setenv("CHALYB_BASE_URL", "https://www.chalyb.com")
    h = {"Authorization": f"Bearer {tenants['alice']['token']}"}
    en = await client.get("/dashboard/live?lang=en", headers=h)
    es = await client.get("/dashboard/live?lang=es", headers=h)
    assert en.status_code == 200 and es.status_code == 200
    assert "https://www.chalyb.com/auth/launch/chalybobs?lang=en" in en.text
    assert 'href="https://www.chalyb.com/auth/launch/chalybobs"' in es.text


@pytest.mark.parametrize(
    "lang,href", [("en", "https://chalyb.com/en/login"), (None, "https://chalyb.com/login")]
)
async def test_sso_failure_page_login_link(
    client: httpx.AsyncClient,
    sso_env: str,
    monkeypatch: pytest.MonkeyPatch,
    lang: str | None,
    href: str,
) -> None:
    # Earlier suites leave structlog bound to a closed stdout; the log line is not under test.
    import structlog

    quiet = SimpleNamespace(warning=lambda *a, **k: None)
    monkeypatch.setattr(structlog, "get_logger", lambda *a, **k: quiet)
    q = f"&lang={lang}" if lang else ""
    r = await client.get(f"/auth/sso?token=bad.token{q}", follow_redirects=False)
    assert r.status_code == 401
    assert f'href="{href}"' in r.text


@pytest.mark.parametrize(
    "lang,want", [("en", "https://chalyb.com/en/app/planes"), (None, "https://chalyb.com/app/planes")]
)
async def test_autopilot_upgrade_link_language(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], lang: str | None, want: str
) -> None:
    q = f"?lang={lang}" if lang else ""
    h = {"Authorization": f"Bearer {tenants['alice']['token']}"}
    r = await client.post(f"/dashboard/publish/zernio/autopublish/save{q}", headers=h, json={"enabled": True})
    assert r.status_code == 402, r.text
    assert r.json()["detail"]["upgrade_url"] == want

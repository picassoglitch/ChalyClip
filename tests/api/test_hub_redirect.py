"""CHALYBCLIP_HUB_REDIRECT — user pages move to the hub, nothing else does."""

from __future__ import annotations

import httpx
import pytest

from chalybclip.api.hub_redirect import hub_target


def test_page_map() -> None:
    assert hub_target("/dashboard") == "/app/clips"
    assert hub_target("/") == "/app/clips"
    assert hub_target("/agents") == "/app/clips"
    assert hub_target("/dashboard/start") == "/app/clips"
    assert hub_target("/dashboard/streams/str_1") == "/app/clips/videos/str_1"
    assert hub_target("/dashboard/brand-kits/new") == "/app/clips/ajustes/estilos/nuevo"
    assert hub_target("/dashboard/brand-kits/brk_9") == "/app/clips/ajustes/estilos/brk_9"
    for kept in (
        "/dashboard/clips/clp_1/media",
        "/dashboard/clips/clp_1/render",
        "/dashboard/streams/str_1/progress",
        "/dashboard/_health/queue",
        "/dashboard/_diag/chalyb",
        "/auth/sso",
        "/overlay/kick",
        "/api/hub/v1/me",
        "/dashboard/streams/../x",
        "/robots.txt",
        "/sitemap.xml",
        "/llms.txt",
        "/healthz",
    ):
        assert hub_target(kept) is None, kept


async def test_off_by_default(client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHALYBCLIP_HUB_REDIRECT", raising=False)
    r = await client.get("/dashboard/start", follow_redirects=False)
    assert "www.chalyb.com/app/clips" not in r.headers.get("location", "")


async def test_on_redirects_get_pages_only(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHALYBCLIP_HUB_REDIRECT", "1")
    monkeypatch.setenv("CHALYB_BASE_URL", "https://hub.test/")
    r = await client.get("/dashboard/streams/str_9?x=1", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "https://hub.test/app/clips/videos/str_9"
    # A form POST to the same path is not redirected (old pages keep working
    # until their tab reloads).
    r = await client.post("/dashboard/streams", follow_redirects=False)
    assert not r.headers.get("location", "").startswith("https://hub.test")
    assert (await client.get("/healthz")).status_code == 200

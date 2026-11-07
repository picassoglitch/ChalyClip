"""Paid brand kits start with the Chalyb end card OFF (owner, 2026-11-07);
Free keeps it whatever is posted. Covers the dashboard form (checkbox
default + create without the box) and the hub API default."""

from __future__ import annotations

import re

import httpx

from chalybclip.db import BrandKitsRepo, Database
from chalybclip.tenancy import bound_tenant

from .conftest import auth
from .test_hub_core import hdr, hub_env  # noqa: F401 — autouse SSO secret

_BOX = re.compile(r'<input type="checkbox" name="show_chalybclip_outro"[^>]*>')


async def _set_tier(db: Database, tenant_id: str, tier: str) -> None:
    conn = await db.connect()
    await conn.execute("UPDATE tenants SET tier = ? WHERE id = ?", (tier, tenant_id))
    await conn.commit()


async def _only_kit(db: Database, tenant_id: str):  # type: ignore[no-untyped-def]
    with bound_tenant(tenant_id):
        [kit] = await BrandKitsRepo(db).list_for_tenant()
    return kit


async def test_paid_form_box_starts_unchecked_and_says_chalyb(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], db: Database
) -> None:
    alice = tenants["alice"]
    await _set_tier(db, alice["id"], "pro")
    r = await client.get("/dashboard/brand-kits/new", headers=auth(alice["token"]))
    assert r.status_code == 200
    box = _BOX.search(r.text)
    assert box is not None and "checked" not in box.group(0)
    assert "Cierre Chalyb<" in r.text and "Cierre ChalyClip" not in r.text


async def test_paid_form_create_without_box_is_off(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], db: Database
) -> None:
    alice = tenants["alice"]
    await _set_tier(db, alice["id"], "pro")
    r = await client.post(
        "/dashboard/brand-kits", data={"name": "K"}, headers=auth(alice["token"]),
    )
    assert r.status_code in (200, 303), r.text
    assert (await _only_kit(db, alice["id"])).show_chalybclip_outro is False


async def test_paid_form_create_with_box_is_on(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], db: Database
) -> None:
    alice = tenants["alice"]
    await _set_tier(db, alice["id"], "pro")
    r = await client.post(
        "/dashboard/brand-kits",
        data={"name": "K", "show_chalybclip_outro": "1"},
        headers=auth(alice["token"]),
    )
    assert r.status_code in (200, 303), r.text
    assert (await _only_kit(db, alice["id"])).show_chalybclip_outro is True


async def test_free_form_keeps_end_card(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], db: Database
) -> None:
    alice = tenants["alice"]
    r = await client.get("/dashboard/brand-kits/new", headers=auth(alice["token"]))
    assert _BOX.search(r.text) is None
    assert '<input type="hidden" name="show_chalybclip_outro" value="1">' in r.text


async def test_hub_api_paid_default_off_free_on(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], db: Database
) -> None:
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    r = await client.post("/api/hub/v1/brand-kits", json={"name": "K"}, headers=hdr(alice))
    assert r.status_code == 200, r.text
    assert r.json()["kit"]["show_chalybclip_outro"] is False
    r = await client.post(
        "/api/hub/v1/brand-kits", json={"name": "K"}, headers=hdr(bob, tier="free")
    )
    assert r.status_code == 200, r.text
    assert r.json()["kit"]["show_chalybclip_outro"] is True


async def test_hub_put_without_field_keeps_opt_in(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]], db: Database
) -> None:
    alice = tenants["alice"]["id"]
    r = await client.post(
        "/api/hub/v1/brand-kits", json={"name": "K", "show_chalybclip_outro": True},
        headers=hdr(alice),
    )
    kid = r.json()["kit"]["id"]
    r = await client.put(f"/api/hub/v1/brand-kits/{kid}", json={"name": "K2"}, headers=hdr(alice))
    assert r.status_code == 200, r.text
    assert r.json()["kit"]["show_chalybclip_outro"] is True
    r = await client.put(
        f"/api/hub/v1/brand-kits/{kid}", json={"name": "K2", "show_chalybclip_outro": False},
        headers=hdr(alice),
    )
    assert r.json()["kit"]["show_chalybclip_outro"] is False

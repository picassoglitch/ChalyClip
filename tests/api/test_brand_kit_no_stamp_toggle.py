"""The "chalybclip.com" stamp toggle is gone from the brand-kit form: no
handler ever read it (Aldo, 2026-10-15). The end-card toggle stays."""

from __future__ import annotations

import httpx

from .conftest import auth


async def test_form_has_no_stamp_toggle(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.get("/dashboard/brand-kits/new", headers=auth(tenants["alice"]["token"]))
    assert r.status_code == 200
    assert "show_chalybclip_credit" not in r.text
    assert "show_chalybclip_outro" in r.text

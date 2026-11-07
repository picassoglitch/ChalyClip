"""Export render page (what the hybrid recorder screenshots for
"Descargar video") must not paint platform branding a paid user never
configured. QA 2026-11-06: a Pro account with no brand kit got a
"CHALYBCLIP" corner mark burned into its export."""

from __future__ import annotations

import httpx
import pytest

from chalybclip.db import Database

from .conftest import auth
from .test_clip_render_overlay_drift import _seed_clip_for_render

_WATERMARK_DIV = '<div class="nc-pv-watermark">'


async def _set_tier(db: Database, tenant_id: str, tier: str) -> None:
    conn = await db.connect()
    await conn.execute("UPDATE tenants SET tier = ? WHERE id = ?", (tier, tenant_id))
    await conn.commit()


async def _render(client: httpx.AsyncClient, token: str) -> str:
    r = await client.get("/dashboard/clips/clp_overlay_drift/render", headers=auth(token))
    assert r.status_code == 200
    return r.text


async def test_free_export_paints_chalyb_mark_not_legacy_name(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    alice = tenants["alice"]
    await _set_tier(db, alice["id"], "free")
    await _seed_clip_for_render(db, alice["id"])
    text = await _render(client, alice["token"])
    assert f"{_WATERMARK_DIV}CHALYB</div>" in text
    assert "CHALYBCLIP</div>" not in text


@pytest.mark.parametrize("with_kit", [False, True])
@pytest.mark.parametrize("tier", ["pro", "vip"])
async def test_paid_export_has_no_watermark(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    tier: str,
    with_kit: bool,
) -> None:
    alice = tenants["alice"]
    await _set_tier(db, alice["id"], tier)
    await _seed_clip_for_render(
        db, alice["id"], brand_kit_target_platform="tiktok" if with_kit else None,
    )
    text = await _render(client, alice["token"])
    assert _WATERMARK_DIV not in text

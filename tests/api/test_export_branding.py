"""Export render page (what the hybrid recorder screenshots for
"Descargar video") must not paint platform branding a paid user never
configured. QA 2026-11-06: a Pro account with no brand kit got a
"CHALYBCLIP" corner mark burned into its export."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from chalybclip.branding import TierUnknownError
from chalybclip.db import ClipsRepo, Database
from chalybclip.tenancy import bound_tenant

from .._db_backend import pg_enabled
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


# ---- plan unknown: fail retryable, never guess free vs paid ----


def _tier_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    import chalybclip.branding.service as svc

    async def boom(db, tenant_id):  # type: ignore[no-untyped-def]
        raise TierUnknownError("tier lookup failed")

    monkeypatch.setattr(svc, "_tenant_is_free", boom)


async def test_render_page_503_when_tier_unknown(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alice = tenants["alice"]
    await _seed_clip_for_render(db, alice["id"])
    _tier_unknown(monkeypatch)
    r = await client.get(
        "/dashboard/clips/clp_overlay_drift/render", headers=auth(alice["token"])
    )
    assert r.status_code == 503
    assert _WATERMARK_DIV not in r.text


async def test_burn_fallback_fails_when_tier_unknown(
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chalybclip.api.routers.dashboard import _burn_overlays_for_clip

    alice = tenants["alice"]
    await _seed_clip_for_render(db, alice["id"])
    src = tmp_path / "c.mp4"
    src.write_bytes(b"\x00" * 1000)
    conn = await db.connect()
    await conn.execute("UPDATE clips SET path = ? WHERE id = 'clp_overlay_drift'", (str(src),))
    await conn.commit()
    _tier_unknown(monkeypatch)
    with bound_tenant(alice["id"]):
        out = await _burn_overlays_for_clip(
            db=db, clip_id="clp_overlay_drift", overlay_config={}
        )
    assert out == "failed:tier_unknown"
    assert not (tmp_path / "clip_final.mp4").exists()


@pytest.mark.skipif(pg_enabled(), reason="render task opens its own SQLite db_path")
async def test_background_export_fails_retryable_when_tier_unknown(
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chalybclip.api._clip_render import render_clip_in_background

    from .test_render_outer_guard import _seed_clip

    alice = tenants["alice"]
    source = await _seed_clip(db, alice["id"], clip_id="clp_tier", output_dir=tmp_path)
    recorded: list[str] = []

    async def fake_hybrid(**kwargs: object) -> None:
        recorded.append("hybrid")

    monkeypatch.setattr("chalybclip.clip.hybrid_recorder.record_clip_hybrid", fake_hybrid)
    _tier_unknown(monkeypatch)
    await render_clip_in_background(
        clip_id="clp_tier", tenant_id=alice["id"], duration_s=30.0,
        audio_source_path=source,
        output_path=tmp_path / "clips" / "clp_tier" / "clip_render_1080.mp4",
        base_url="http://localhost:8000", auth_cookie_value=None,
        width=1080, height=1920, db_path=str(tmp_path / "api.db"),
    )
    with bound_tenant(alice["id"]):
        clip = await ClipsRepo(db).get("clp_tier")
        assert clip is not None
        assert clip.render_state == "failed"
        assert "plan" in (clip.render_error or "")
        assert recorded == []  # nothing rendered with guessed branding
        await ClipsRepo(db).mark_render_started("clp_tier")  # Retry path
        clip = await ClipsRepo(db).get("clp_tier")
    assert clip is not None and clip.render_state == "rendering"

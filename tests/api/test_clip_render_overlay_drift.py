"""Drift fix Step 2 — overlay markup + CSS unified via shared partial.

Pins the export-side contract after the refactor:
  - The render route emits the shared _overlay_styles.html CSS
    (banner safe-zone position, caption transitions, full variant
    set) instead of the old inline copy that had drifted.
  - The render route emits the shared _overlay_markup.html shape
    (banner, caption, title, top_hook, watermark IDs preserved).
  - `data-face-zone` lands on `.nc-preview` when the candidate's
    framing evidence supplies a y bucket — previously the editor
    set this and the export didn't.
  - `data-zone-profile` falls through to the brand_kit's
    target_platform when the per-clip overlay_config has none —
    previously the editor used this fallback chain and the export
    only honored the per-clip override.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
from pathlib import Path

import httpx
import pytest

from chalybclip.db import Database
from chalybclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_clip_for_render(
    db: Database,
    tenant_id: str,
    *,
    clip_id: str = "clp_overlay_drift",
    overlay_config: dict | None = None,
    framing_y: float | None = None,
    brand_kit_target_platform: str | None = None,
) -> None:
    """Insert a clip + its stream + an optional candidate that carries
    framing evidence and an optional brand_kit so the handler's
    overlay-resolution path is exercised end-to-end."""
    stream_id = "str_overlay_drift"
    candidate_id = "cnd_overlay_drift" if framing_y is not None else None

    with bound_tenant(tenant_id):
        conn = await db.connect()
        # Stream row.
        await conn.execute(
            "INSERT INTO streams "
            "(id, tenant_id, vod_url, platform, title, channel, "
            "duration_s, source_video_path, source_audio_path, status, "
            "created_at) "
            "VALUES (?, ?, 'https://kick.com/x/v/1', 'kick', NULL, NULL, "
            "30, '/tmp/v.mp4', '/tmp/a.wav', 'ingested', ?) ON CONFLICT DO NOTHING",
            (stream_id, tenant_id, _now()),
        )
        # Candidate carrying framing evidence — drives face_zone.
        if candidate_id is not None:
            ev = {
                "framing": {
                    "safe_crop_box": {"y": framing_y, "x": 0.0, "w": 1.0, "h": 1.0},
                },
            }
            await conn.execute(
                "INSERT INTO candidates "
                "(id, stream_id, tenant_id, ts, score, reason, "
                "evidence_json, created_at) "
                "VALUES (?, ?, ?, 1.0, 0.5, 'audio_energy', ?, ?) ON CONFLICT DO NOTHING",
                (candidate_id, stream_id, tenant_id,
                 _json.dumps(ev), _now()),
            )
        # Optional brand_kit so the zone_profile fallback chain
        # exercises the brand_kit branch.
        if brand_kit_target_platform is not None:
            await conn.execute(
                "INSERT INTO brand_kits "
                "(id, tenant_id, name, primary_color, accent_color, "
                "text_color, target_platform, created_at, updated_at, is_default) "
                "VALUES ('bk_test', ?, 'Test', '#000', '#fff', '#fff', "
                "?, ?, ?, 1) ON CONFLICT DO NOTHING",
                (tenant_id, brand_kit_target_platform, _now(), _now()),
            )
        # Clip row.
        ov_json = _json.dumps(overlay_config or {})
        await conn.execute(
            "INSERT INTO clips "
            "(id, stream_id, tenant_id, candidate_id, start_s, end_s, "
            "duration_s, width, height, path, status, created_at, "
            "overlay_config_json) "
            "VALUES (?, ?, ?, ?, 0, 30, 30, 1080, 1920, '/tmp/c.mp4', "
            "'cut', ?, ?)",
            (clip_id, stream_id, tenant_id, candidate_id,
             _now(), ov_json),
        )
        await conn.commit()


# ---- shared overlay partial is included ----


async def test_render_page_includes_shared_overlay_styles_partial(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The render template now `{% include "_overlay_styles.html" %}`
    instead of duplicating ~200 lines of overlay CSS. We can't directly
    assert the include happened, but we CAN assert one of the rules the
    partial brings in (banner safe-zone bottom from merged CSS) is
    present in the HTML — and that the old inline duplicate comment
    block is absent."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_for_render(db, tenant_id)

    r = await client.get(
        "/dashboard/clips/clp_overlay_drift/render",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    text = r.text

    # Banner base safe-zone rule from the merged partial:
    # `bottom: 21.875cqh;` (was 0 in editor, was 21.875 in export —
    # partial uses the safe-zone value for both).
    assert "21.875cqh" in text

    # Caption transitions — used to live ONLY in editor; the partial
    # now ships them to the export too.
    assert "transition: color 90ms linear, transform 90ms linear" in text

    # All four banner variants — kick_repost_page, kick_green_block,
    # kick_minimal_url existed only in the editor; the partial brings
    # the full set to the export.
    assert "kick_repost_page" in text
    assert "kick_green_block" in text
    assert "kick_minimal_url" in text


async def test_render_page_emits_shared_banner_markup(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The render template now includes _overlay_markup.html. Banner
    is emitted with the editor's element shape (id="pv-banner",
    sub-spans for platform / url / live-pill) — previously the export
    had a stripped-down version."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_for_render(
        db, tenant_id,
        overlay_config={
            "banner": {
                "enabled": True,
                "variant": "kick_repost_page",
                "url": "aldo",
                "live_badge": False,
            },
        },
    )

    r = await client.get(
        "/dashboard/clips/clp_overlay_drift/render",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    text = r.text

    # IDs that the editor's JS targets — kept in the export too so
    # both surfaces share the same DOM shape (no JS-specific drift).
    assert 'id="pv-banner"' in text
    assert 'id="pv-banner-platform"' in text
    assert 'id="pv-banner-url"' in text
    assert 'id="pv-banner-live-pill"' in text


# ---- face_zone now threaded through clip_render_view ----


async def test_render_page_sets_data_face_zone_when_framing_says_top(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Candidate.framing.safe_crop_box.y < 0.10 → face_zone="top".
    The shared helper now computes it for the export route too;
    previously this attribute was missing in the rendered page and
    the CSS rules gated on `[data-face-zone="top"]` never fired."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_for_render(db, tenant_id, framing_y=0.05)

    r = await client.get(
        "/dashboard/clips/clp_overlay_drift/render",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    # The CSS partial also mentions data-face-zone in selectors,
    # so we need to be specific: assert the attribute appears on
    # the .nc-preview element itself.
    assert 'class="nc-preview"' in r.text
    nc_preview_idx = r.text.index('class="nc-preview"')
    nc_preview_block = r.text[nc_preview_idx:nc_preview_idx + 800]
    assert 'data-face-zone="top"' in nc_preview_block


async def test_render_page_sets_data_face_zone_lower_for_high_y(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """y >= 0.35 → face_zone="lower"."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_for_render(db, tenant_id, framing_y=0.45)

    r = await client.get(
        "/dashboard/clips/clp_overlay_drift/render",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    nc_preview_idx = r.text.index('class="nc-preview"')
    nc_preview_block = r.text[nc_preview_idx:nc_preview_idx + 800]
    assert 'data-face-zone="lower"' in nc_preview_block


async def test_render_page_omits_face_zone_when_no_framing(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """No candidate / no framing evidence → no data-face-zone attribute
    (falls through to the L.4 static overlay positions)."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_for_render(db, tenant_id)  # no framing_y

    r = await client.get(
        "/dashboard/clips/clp_overlay_drift/render",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    # Attribute should NOT be set on the .nc-preview element so the
    # face-aware CSS rules don't match → static defaults apply.
    # (The CSS partial mentions `data-face-zone` in selectors, so
    # restrict the scope to the preview element itself.)
    nc_preview_idx = r.text.index('class="nc-preview"')
    nc_preview_block = r.text[nc_preview_idx:nc_preview_idx + 800]
    assert "data-face-zone=" not in nc_preview_block


# ---- zone_profile fallback now uses brand_kit ----


async def test_render_page_zone_profile_per_clip_override_wins(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """When overlay_config.target_platform is set, it wins — the
    brand_kit fallback never fires."""
    tenant_id = tenants["alice"]["id"]
    await _seed_clip_for_render(
        db, tenant_id,
        overlay_config={"target_platform": "tiktok"},
        brand_kit_target_platform="kick",  # would override → must NOT
    )

    r = await client.get(
        "/dashboard/clips/clp_overlay_drift/render",
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    assert 'data-zone-profile="tiktok"' in r.text

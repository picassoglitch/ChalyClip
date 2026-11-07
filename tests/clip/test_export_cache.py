"""Export fingerprint: what makes a cached export stale."""

from __future__ import annotations

from types import SimpleNamespace

from chalybclip.clip.export_cache import export_fingerprint


def _clip(**over: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "id": "clp_1", "start_s": 10.0, "end_s": 40.0, "duration_s": 30.0,
        "overlay_config": {"top_hook": {"text": "hola"}, "format": "vertical"},
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_same_clip_same_fingerprint_regardless_of_key_order() -> None:
    a = _clip(overlay_config={"a": 1, "b": {"x": 1, "y": 2}})
    b = _clip(overlay_config={"b": {"y": 2, "x": 1}, "a": 1})
    assert export_fingerprint(a, quality="1080") == export_fingerprint(b, quality="1080")


def test_overlay_trim_format_and_quality_change_it() -> None:
    fp = export_fingerprint(_clip(), quality="1080")
    assert export_fingerprint(_clip(), quality="4k") != fp
    assert export_fingerprint(_clip(start_s=11.0), quality="1080") != fp
    assert export_fingerprint(_clip(end_s=41.0, duration_s=31.0), quality="1080") != fp
    assert export_fingerprint(
        _clip(overlay_config={"top_hook": {"text": "adiós"}, "format": "vertical"}), quality="1080"
    ) != fp
    assert export_fingerprint(
        _clip(overlay_config={"top_hook": {"text": "hola"}, "format": "square"}), quality="1080"
    ) != fp


def test_missing_overlay_is_stable() -> None:
    assert export_fingerprint(_clip(overlay_config=None), quality="1080") == export_fingerprint(
        _clip(overlay_config={}), quality="1080"
    )


def test_render_version_bump_invalidates_old_exports(monkeypatch) -> None:
    """2026-11-07 real 2K/4K: exports cropped by the old renderer must miss."""
    from chalybclip.clip import export_cache

    new = export_fingerprint(_clip(), quality="4k")
    monkeypatch.setattr(export_cache, "EXPORT_RENDER_VERSION", "1")
    assert export_fingerprint(_clip(), quality="4k") != new


def test_plan_branding_changes_it() -> None:
    base = export_fingerprint(_clip(), quality="1080", outro=False, watermark=False)
    assert export_fingerprint(_clip(), quality="1080", outro=False, watermark=True) != base
    assert export_fingerprint(_clip(), quality="1080", outro=True, watermark=False) != base


async def test_current_fingerprint_follows_the_tenant_plan(tmp_path) -> None:
    """Same helpers the renderer uses: Free → watermark + outro, Pro → neither."""
    from chalybclip.clip.export_cache import current_fingerprint
    from chalybclip.db import TenantsRepo

    from .._db_backend import migrated_database

    async for db in migrated_database(tmp_path, name="fp.db"):
        tenant = await TenantsRepo(db).create(name="Plan Co")
        clip = _clip(stream_id="str_none")
        free_fp = await current_fingerprint(db, tenant_id=tenant.id, clip=clip, quality="1080")
        assert free_fp == export_fingerprint(clip, quality="1080", outro=True, watermark=True)
        conn = await db.connect()
        await conn.execute("UPDATE tenants SET tier = 'pro' WHERE id = ?", (tenant.id,))
        await conn.commit()
        pro_fp = await current_fingerprint(db, tenant_id=tenant.id, clip=clip, quality="1080")
        assert pro_fp == export_fingerprint(clip, quality="1080", outro=False, watermark=False)
        assert await current_fingerprint(db, tenant_id="ten_missing", clip=clip, quality="1080") is None

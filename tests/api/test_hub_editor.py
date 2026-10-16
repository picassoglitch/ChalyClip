"""/api/hub/v1 clip editor — the hub's JSON face over the dashboard editor.

Pins:
  * only the hub launch token reaches these routes; another tenant's clip
    is a 404 on every route (no tenant leakage through the editor);
  * GET editor returns the dashboard's own verdict + the editor's prefill
    chain (clip overlay → newest edited sibling → brand-kit prefs);
  * save / finalize / reject / revert-trim drive the SAME handlers as the
    dashboard (status walk, next clip, cache invalidation, brand-kit mirror);
  * render / render-status / render-retry never leak a /dashboard URL; a
    ready render hands out a signed `render` media link (hub_media) that
    serves only a servable file, for that tenant, as an attachment;
  * the editor payload's media links play the clip (Range) and are bound
    to the tenant that signed them;
  * a hub-initiated render (no operator cookie) signs the /render page.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from chalybclip.db import BrandKitsRepo, ClipsRepo, Database, StreamsRepo
from chalybclip.db.models import ClipRow, StreamRow
from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

_SECRET = "hub_editor_secret_value"
_MP4_HEADER = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2mp41"


@pytest.fixture
def hub_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[str]:
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    monkeypatch.setenv("CHALYBCLIP_INTERNAL_SIGNING_SECRET", "sign_secret_value")
    monkeypatch.setenv("CHALYBCLIP_DEFAULT_OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    try:
        yield _SECRET
    finally:
        get_settings.cache_clear()


def _hub(tenant_id: str, **kw: Any) -> dict[str, str]:
    token = sign_sso_token(
        user_id=str(kw.pop("user_id", "chalyb_user_1")),
        email=str(kw.pop("email", "alice@example.com")),
        tenant_id=tenant_id,
        secret=_SECRET,
        **kw,
    )
    return {"Authorization": f"Bearer {token}"}


def _now(offset_s: int = 0) -> str:
    return (_dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=offset_s)).isoformat()


async def _seed(
    db: Database,
    tenant_id: str,
    tmp_path: Path,
    *,
    clip_id: str = "clp_h1",
    stream_id: str = "str_h",
    status: str = "cut",
    overlay: dict[str, Any] | None = None,
    created_offset: int = 0,
    original: tuple[float, float] | None = None,
) -> Path:
    clip_dir = tmp_path / stream_id / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.mp4"
    clip_path.write_bytes(_MP4_HEADER + b"\x00" * 4096)
    with bound_tenant(tenant_id):
        if await StreamsRepo(db).get(stream_id) is None:
            await StreamsRepo(db).upsert(
                StreamRow(
                    id=stream_id,
                    tenant_id=tenant_id,
                    vod_url="https://kick.com/x",
                    platform="kick",
                    title="Mi stream",
                    channel="c",
                    duration_s=60.0,
                    source_video_path="/tmp/v",
                    source_audio_path="/tmp/a",
                    status="ingested",
                    created_at=_now(),
                )
            )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id,
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path=str(clip_path),
                    status=status,
                    created_at=_now(created_offset),
                    overlay_config=overlay,
                    original_start_s=original[0] if original else None,
                    original_end_s=original[1] if original else None,
                )
            ]
        )
        if overlay is not None:
            await ClipsRepo(db).set_overlay_config(clip_id, overlay_config=overlay)
        if original is not None:
            await ClipsRepo(db).set_trim_bounds(
                clip_id,
                start_s=2.0,
                end_s=10.0,
                duration_s=8.0,
                new_path=str(clip_path),
                original_start_s=original[0],
                original_end_s=original[1],
            )
    return clip_path


def _overlay_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "title_text": "Le pidió la revancha",
        "banner_enabled": True,
        "banner_platform": "kick",
        "banner_url": "kick.com/aldo",
        "banner_color": "#53FC18",
        "captions_enabled": True,
        "captions_position": "lower_third",
        "captions_font_size": "large",
        "captions_animation": "pop",
        "captions_lead_ms": 150,
        "clip_style": "repost_page_viral",
        "banner_variant": "kick_repost_page",
        "target_platform": "tiktok",
    }
    body.update(over)
    return body


def _err(r: httpx.Response, status: int, code: str) -> None:
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code


# ── auth + tenant isolation ───────────────────────────────────────────────

_ROUTES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("GET", "/api/hub/v1/clips/clp_h1/editor", None),
    ("PUT", "/api/hub/v1/clips/clp_h1/overlay", {}),
    ("POST", "/api/hub/v1/clips/clp_h1/finalize", {}),
    ("POST", "/api/hub/v1/clips/clp_h1/reject", None),
    ("POST", "/api/hub/v1/clips/clp_h1/revert-trim", None),
    ("POST", "/api/hub/v1/clips/clp_h1/hooks", {"tone": "default"}),
    ("GET", "/api/hub/v1/clips/clp_h1/captions", None),
    ("GET", "/api/hub/v1/clips/clp_h1/waveform", None),
    ("GET", "/api/hub/v1/clips/clp_h1/intelligence", None),
    ("GET", "/api/hub/v1/clips/clp_h1/growth", None),
    ("POST", "/api/hub/v1/clips/clp_h1/render", None),
    ("GET", "/api/hub/v1/clips/clp_h1/render-status", None),
    ("POST", "/api/hub/v1/clips/clp_h1/render-retry", None),
]


async def test_every_route_needs_the_hub_token(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]
    await _seed(db, alice["id"], tmp_path)
    for method, url, body in _ROUTES:
        # No credential, the tenant's api_token, and a forged token all fail.
        for headers in (
            {},
            {"Authorization": f"Bearer {alice['token']}"},
            {"Authorization": "Bearer " + "x" * 40},
        ):
            r = await client.request(method, url, json=body, headers=headers)
            _err(r, 401, "unauthorized")
        # The dashboard session cookie doesn't open hub paths either.
        client.cookies.set("chalybclip_token", alice["token"])
        r = await client.request(method, url, json=body)
        client.cookies.clear()
        _err(r, 401, "unauthorized")


async def test_another_tenants_clip_is_not_found_everywhere(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    await _seed(db, alice, tmp_path)
    for method, url, body in _ROUTES:
        r = await client.request(method, url, json=body, headers=_hub(bob))
        _err(r, 404, "not_found")
    # And nothing changed on Alice's clip.
    with bound_tenant(alice):
        clip = await ClipsRepo(db).get("clp_h1")
    assert clip is not None
    assert clip.status == "cut"
    assert clip.overlay_config is None


# ── GET editor ────────────────────────────────────────────────────────────


async def test_editor_payload_has_verdict_and_defaults(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path)
    r = await client.get("/api/hub/v1/clips/clp_h1/editor", headers=_hub(alice))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["clip"]["id"] == "clp_h1"
    assert body["clip"]["stream_title"] == "Mi stream"
    assert body["clip"]["has_overlay"] is False
    assert body["can_finalize"] is True and body["can_reject"] is True
    assert body["publishability"]["status"] in ("publish_ready", "needs_edit", "reject")
    assert 0 <= body["publishability"]["score"] <= 100
    assert "viral_score" in body["ai_scores"]
    assert body["breakdown"]["clip_id"] == "clp_h1"
    form = body["form"]
    assert form["target_platform"] == "all"
    assert form["clip_style"] == "repost_page_viral"
    assert form["captions_enabled"] is True
    assert form["banner_enabled"] is False
    assert form["banner_variant"] == "kick_repost_page"
    assert form["captions_animation"] in ("pop", "slide", "typewriter", "fade")
    assert form["captions_lead_ms"] == 120
    choices = body["choices"]
    assert set(choices["targets"]) == {"tiktok", "reels", "shorts", "kick", "all"}
    assert "tiktok" in choices["zone_specs"]
    assert choices["export_qualities"] == ["1080"]
    # Nothing in the payload points the hub at the dashboard.
    assert "/dashboard/" not in r.text
    # The verdict got cached on the clip row, like the dashboard does.
    with bound_tenant(alice):
        clip = await ClipsRepo(db).get("clp_h1")
    assert clip is not None and clip.publishability_status == body["publishability"]["status"]


async def test_editor_prefill_inherits_sibling_then_brand_kit(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]["id"]
    await _seed(
        db,
        alice,
        tmp_path,
        clip_id="clp_old",
        status="approved",
        overlay={"banner": {"enabled": True, "url": "kick.com/aldo"}, "title_text": "viejo"},
        created_offset=-60,
    )
    await _seed(db, alice, tmp_path, clip_id="clp_h1")
    with bound_tenant(alice):
        await BrandKitsRepo(db).create(
            name="Default",
            primary_color="#53FC18",
            accent_color="#FFD700",
            is_default=True,
            caption_style={"position": "bottom", "lead_ms": 80},
            handle_kick="aldov",
        )
    r = await client.get("/api/hub/v1/clips/clp_h1/editor", headers=_hub(alice))
    assert r.status_code == 200, r.text
    form = r.json()["form"]
    assert form["banner_enabled"] is True
    assert form["banner_url"] == "kick.com/aldo"
    assert form["title_text"] == "viejo"
    assert form["captions_position"] == "bottom"
    assert form["captions_lead_ms"] == 80
    assert r.json()["default_handle"] == "aldov"


# ── save / finalize / reject / revert ─────────────────────────────────────


async def test_save_draft_persists_and_drops_cached_renders(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]["id"]
    clip_path = await _seed(db, alice, tmp_path)
    stale = clip_path.parent / "clip_render_1080.mp4"
    stale.write_bytes(b"old")
    r = await client.put(
        "/api/hub/v1/clips/clp_h1/overlay",
        json=_overlay_body(unknown_field="ignored"),
        headers=_hub(alice),
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "cut"
    assert not stale.exists()
    with bound_tenant(alice):
        clip = await ClipsRepo(db).get("clp_h1")
        kit = await BrandKitsRepo(db).get_default()
    cfg = clip.overlay_config if clip else None
    assert cfg is not None
    assert cfg["title_text"] == "Le pidió la revancha"
    assert cfg["banner"]["enabled"] is True
    assert cfg["captions"]["lead_ms"] == 150
    assert cfg["target_platform"] == "tiktok"
    assert "unknown_field" not in cfg
    # Branding mirrored to the brand kit, like the dashboard save.
    assert kit is not None and kit.handle_kick == "kick.com/aldo"


async def test_save_draft_rejects_out_of_range_values(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path)
    for bad in (
        {"captions_lead_ms": 9000},
        {"top_hook_text": "x" * 500},
        {"comments_fake_likes": -1},
    ):
        r = await client.put(
            "/api/hub/v1/clips/clp_h1/overlay", json=_overlay_body(**bad), headers=_hub(alice)
        )
        _err(r, 422, "bad_request")


async def test_finalize_approves_and_points_at_next_clip(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _fake_render(**kw: Any) -> None:
        calls.append(kw)

    monkeypatch.setattr("chalybclip.api._clip_render.render_clip_in_background", _fake_render)
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path, clip_id="clp_h1", created_offset=-10)
    await _seed(db, alice, tmp_path, clip_id="clp_h2")
    r = await client.post(
        "/api/hub/v1/clips/clp_h1/finalize", json=_overlay_body(), headers=_hub(alice)
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "approved"
    assert body["next_clip_id"] == "clp_h2"
    assert body["stream_id"] == "str_h"
    # Pre-render scheduled for the 1080 preset.
    assert len(calls) == 1 and calls[0]["clip_id"] == "clp_h1"
    # Last clip in the queue → no next clip.
    r = await client.post(
        "/api/hub/v1/clips/clp_h2/finalize", json=_overlay_body(), headers=_hub(alice)
    )
    assert r.status_code == 200, r.text
    assert r.json()["next_clip_id"] is None


async def test_finalize_refuses_a_published_clip(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path, status="published")
    r = await client.post(
        "/api/hub/v1/clips/clp_h1/finalize", json=_overlay_body(), headers=_hub(alice)
    )
    _err(r, 409, "conflict")
    with bound_tenant(alice):
        clip = await ClipsRepo(db).get("clp_h1")
    assert clip is not None and clip.status == "published" and clip.overlay_config is None


async def test_reject_and_revert_trim(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path)
    r = await client.post("/api/hub/v1/clips/clp_h1/revert-trim", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "outcome": "noop"}
    r = await client.post("/api/hub/v1/clips/clp_h1/reject", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "rejected"
    assert r.json()["stream_id"] == "str_h"
    r = await client.get("/api/hub/v1/clips/clp_h1/editor", headers=_hub(alice))
    assert r.json()["can_reject"] is False


# ── hooks + prefs ─────────────────────────────────────────────────────────


async def test_hooks_fall_back_to_deterministic_titles(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _llm_down(**_kw: Any) -> list[Any]:
        raise RuntimeError("provider down")

    monkeypatch.setattr("chalybclip.variants.generate_hooks", _llm_down)
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path)
    r = await client.post(
        "/api/hub/v1/clips/clp_h1/hooks",
        json={"tone": "not-a-tone", "n": 3},
        headers=_hub(alice),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tone"] == "default"
    assert body["source"] == "deterministic"
    assert 1 <= len(body["hooks"]) <= 3
    assert all(isinstance(h["text"], str) and h["text"] for h in body["hooks"])
    r = await client.post("/api/hub/v1/clips/clp_h1/hooks", json={"n": 50}, headers=_hub(alice))
    _err(r, 422, "bad_request")


async def test_brand_kit_prefs(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
) -> None:
    alice = tenants["alice"]["id"]
    r = await client.post(
        "/api/hub/v1/brand-kit/prefs",
        json={"field": "captions_lead_ms", "value": 200},
        headers=_hub(alice),
    )
    assert r.status_code == 200, r.text
    r = await client.post(
        "/api/hub/v1/brand-kit/prefs",
        json={"field": "target_platform", "value": "kick"},
        headers=_hub(alice),
    )
    assert r.status_code == 200, r.text
    with bound_tenant(alice):
        kit = await BrandKitsRepo(db).get_default()
    assert kit is not None
    assert (kit.caption_style or {}).get("lead_ms") == 200
    assert kit.target_platform == "kick"
    r = await client.post(
        "/api/hub/v1/brand-kit/prefs",
        json={"field": "is_default", "value": True},
        headers=_hub(alice),
    )
    _err(r, 400, "bad_request")
    # Bob never sees Alice's kit.
    with bound_tenant(tenants["bob"]["id"]):
        assert await BrandKitsRepo(db).get_default() is None


# ── feeds ─────────────────────────────────────────────────────────────────


async def test_read_only_feeds(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path)
    r = await client.get("/api/hub/v1/clips/clp_h1/captions", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert r.json()["lines"] == []
    assert r.json()["diagnostics"]["reason"] == "no_transcript"
    r = await client.get("/api/hub/v1/clips/clp_h1/intelligence", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["markers"], list)
    r = await client.get("/api/hub/v1/clips/clp_h1/waveform", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)
    r = await client.get("/api/hub/v1/clips/clp_h1/growth", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert r.json() == {"scored": False}


async def test_media_links_play_and_stay_in_their_tenant(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    await _seed(db, alice, tmp_path)
    body = (await client.get("/api/hub/v1/clips/clp_h1/editor", headers=_hub(alice))).json()
    link = body["media"]["preview"]
    assert link.startswith("/api/hub/media/clip/clp_h1?")
    assert body["media"]["thumbnail"] is None  # no thumbnail on this clip
    # No credential needed: the link is the credential (a <video src>).
    r = await client.get(link, headers={"Range": "bytes=0-15"})
    assert r.status_code == 206, r.text
    assert r.content == _MP4_HEADER[:16]
    # Re-pointing the link at another tenant breaks the signature.
    r = await client.get(link.replace(f"t={alice}", f"t={bob}"))
    assert r.status_code == 403
    r = await client.get(link.replace("clp_h1", "clp_other"))
    assert r.status_code == 403


# ── render / download ─────────────────────────────────────────────────────


async def test_render_flow_without_dashboard_urls(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def _fake_render(**kw: Any) -> None:
        calls.append(kw)

    monkeypatch.setattr("chalybclip.api._clip_render.render_clip_in_background", _fake_render)
    alice = tenants["alice"]["id"]
    clip_path = await _seed(db, alice, tmp_path)
    h = _hub(alice)

    r = await client.get("/api/hub/v1/clips/clp_h1/render-status", headers=h)
    assert r.json() == {
        "state": "idle",
        "progress_pct": 0,
        "error": None,
        "quality": "1080",
        "download": None,
    }

    r = await client.post("/api/hub/v1/clips/clp_h1/render?quality=4k", headers=h)
    assert r.status_code == 200, r.text
    # Free tier: 4k silently downgrades to the 1080 render, as on the dashboard.
    assert r.json()["state"] == "rendering"
    assert "/dashboard" not in r.text
    assert len(calls) == 1 and calls[0]["width"] == 1080
    # No operator cookie on a hub call: the runner signs the /render page itself.
    assert calls[0]["auth_cookie_value"] is None

    r = await client.post("/api/hub/v1/clips/clp_h1/render?quality=8k", headers=h)
    _err(r, 400, "bad_request")

    # A signed render link for a render that isn't there yet is a 404,
    # never the unedited source.
    from chalybclip.api.hub_media import media_path

    early = media_path("render", "clp_h1", tenant_id=alice, params={"q": "1080"})
    assert (await client.get(early)).status_code == 404

    # Render lands on disk → ready → a signed download link (asked 4k on a
    # free plan → the 1080 file, like the dashboard).
    (clip_path.parent / "clip_render_1080.mp4").write_bytes(
        _MP4_HEADER + b"\x00" * (1024 * 1024 + 16)
    )
    r = await client.get("/api/hub/v1/clips/clp_h1/render-status?quality=4k", headers=h)
    st = r.json()
    assert st["state"] == "ready"
    assert "/dashboard" not in r.text
    assert st["download"].startswith("/api/hub/media/render/clp_h1?")
    assert "q=1080" in st["download"]
    r = await client.get(st["download"])
    assert r.status_code == 200
    assert r.headers["content-disposition"].startswith("attachment")
    assert r.content[: len(_MP4_HEADER)] == _MP4_HEADER
    # The signed resolution can't be swapped for another one.
    assert (await client.get(st["download"].replace("q=1080", "q=4k"))).status_code == 403

    r = await client.post("/api/hub/v1/clips/clp_h1/render-retry", headers=h)
    assert r.json()["state"] == "idle"
    assert not (clip_path.parent / "clip_render_1080.mp4").exists()


async def test_render_failed_state_is_reported_not_raised(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path)
    with bound_tenant(alice):
        await ClipsRepo(db).mark_render_started("clp_h1")
        await ClipsRepo(db).mark_render_failed("clp_h1", error="boom")
    r = await client.post("/api/hub/v1/clips/clp_h1/render", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "failed"
    assert r.json()["error"] == "boom"
    assert "retry_url" not in r.json()


async def test_runner_signs_render_page_without_cookie(
    monkeypatch: pytest.MonkeyPatch, db: Database, tmp_path: Path, hub_env: str
) -> None:
    """render_clip_in_background with no cookie and no auth_query mints a
    signed render query for the recorder (hub-initiated renders); a cookie
    caller keeps the old behaviour (no query)."""
    from chalybclip.api import _clip_render
    from chalybclip.clip.hybrid_recorder import HybridRecordingError

    seen: list[dict[str, Any]] = []

    async def _fake_record(**kw: Any) -> None:
        seen.append(kw)
        raise HybridRecordingError("stop here")

    async def _fake_legacy(**kw: Any) -> None:
        raise RuntimeError("stop here too")

    monkeypatch.setattr("chalybclip.clip.hybrid_recorder.record_clip_hybrid", _fake_record)
    monkeypatch.setattr("chalybclip.clip.preview_recorder.record_clip_to_mp4", _fake_legacy)
    src = tmp_path / "clip.mp4"
    src.write_bytes(_MP4_HEADER)
    db_target = str(tmp_path / "api.db")
    for cookie in (None, "operator_cookie"):
        await _clip_render.render_clip_in_background(
            clip_id="clp_x",
            tenant_id="ten_x",
            duration_s=1.0,
            audio_source_path=src,
            output_path=tmp_path / "out.mp4",
            base_url="http://engine.test",
            auth_cookie_value=cookie,
            width=1080,
            height=1920,
            db_path=db_target,
        )
    assert len(seen) == 2
    q = str(seen[0]["auth_query"])
    assert "tenant=ten_x" in q and "sig=" in q and "exp=" in q
    # The /render page's own check (auth._verify_render_signature, 900s
    # ceiling) must accept what the runner minted.
    from urllib.parse import parse_qs

    from chalybclip.api.routers.internal import _verify_signed_params

    p = {k: v[0] for k, v in parse_qs(q).items()}
    _verify_signed_params(
        resource_id="clp_x", tenant=p["tenant"], exp=int(p["exp"]), sig=p["sig"], max_ttl_s=900
    )
    assert seen[1]["auth_query"] == ""
    assert seen[1]["auth_cookie_value"] == "operator_cookie"


async def test_revert_trim_restores_original_bounds(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    hub_env: str,
    tmp_path: Path,
) -> None:
    from chalybclip.clip.trim import _pretrim_original_path

    alice = tenants["alice"]["id"]
    clip_path = await _seed(db, alice, tmp_path, original=(0.0, 20.0))
    _pretrim_original_path(clip_path).write_bytes(_MP4_HEADER + b"pristine")
    r = await client.get("/api/hub/v1/clips/clp_h1/editor", headers=_hub(alice))
    assert r.json()["clip"]["original_end_s"] == 20.0
    r = await client.post("/api/hub/v1/clips/clp_h1/revert-trim", headers=_hub(alice))
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "outcome": "reverted"}
    with bound_tenant(alice):
        clip = await ClipsRepo(db).get("clp_h1")
    assert clip is not None
    assert clip.original_start_s is None
    assert clip.end_s == 20.0

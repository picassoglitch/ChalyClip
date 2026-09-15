"""Auto-program + enrichment tests (Publish Center).

Covers the new whole-queue scheduler and the post-text composer:

  * `build_post` stitches viral hook (variant title card) + caption +
    AI hashtags + the tenant's fixed handle/hashtag suffix.
  * POST /compose/{clip_id} returns that composed title + caption.
  * POST /schedule/auto spreads every approved clip across best-time
    slots and ships the ENRICHED content to Zernio (asserted on the
    create_post payloads).
  * tag_suffix round-trips through /autopublish/save + /autopublish.json.

Zernio is respx-mocked; clip rendering is stubbed (the render path is
covered by the render tests, not here).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

import chalybclip.api._clip_render as _clip_render
from chalybclip.db import (
    AutopublishSettingsRepo,
    CandidatesRepo,
    ClipsRepo,
    Database,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
    VariantsRepo,
    ZernioPublishesRepo,
)
from chalybclip.db.models import CandidateRow, ClipRow, StreamRow, VariantRow
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def publish_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYBCLIP_ZERNIO_API_KEY", "sk_test_ap2")
    monkeypatch.setenv("CHALYBCLIP_INTERNAL_SIGNING_SECRET", "sign_me_ap2")
    get_settings.cache_clear()

    async def _rendered(**_kw: Any) -> None:
        return None

    monkeypatch.setattr(_clip_render, "ensure_clip_rendered", _rendered)
    try:
        yield
    finally:
        get_settings.cache_clear()


async def _seed_clip(
    db: Database, tid: str, *, clip_id: str, persona_id: str,
    caption: str, hook: str, hashtags: list[str], now: str,
    candidate_score: float = 0.7,
    publishability_score: int | None = None,
    publishability_status: str | None = None,
) -> None:
    """Approved clip + a single variant carrying caption/hook/hashtags."""
    with bound_tenant(tid):
        await CandidatesRepo(db).upsert_many([
            CandidateRow(
                id=f"cnd_{clip_id}", stream_id="str_ap", tenant_id=tid,
                ts=10.0, score=candidate_score, reason="voice", evidence={},
                created_at=now,
            )
        ])
        await ClipsRepo(db).upsert_many([
            ClipRow(
                id=clip_id, stream_id="str_ap", tenant_id=tid,
                candidate_id=f"cnd_{clip_id}", start_s=0.0, end_s=10.0,
                duration_s=10.0, width=1080, height=1920,
                path="/tmp/c.mp4", status="approved", created_at=now,
            )
        ])
        if publishability_score is not None and publishability_status is not None:
            await ClipsRepo(db).set_publishability(
                clip_id, score=publishability_score, status=publishability_status,
            )
        await VariantsRepo(db).replace_for_clip_persona(
            clip_id, persona_id,
            [VariantRow(
                id=f"var_{clip_id}", clip_id=clip_id, tenant_id=tid,
                persona_id=persona_id, language="es", caption=caption,
                title_card_text=hook, hashtags=hashtags, model="test",
                created_at=now,
            )],
        )


@pytest_asyncio.fixture
async def alice(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Alice: all-access, Zernio profile, persona, two approved clips with
    variants (clp_1, clp_2)."""
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(
        tid, profile_id="prof_alice", profile_name="Alice",
    )
    now = _dt.datetime.now(_dt.UTC).isoformat()
    with bound_tenant(tid):
        await StreamsRepo(db).upsert(StreamRow(
            id="str_ap", tenant_id=tid, vod_url="x", platform="kick",
            title=None, channel=None, duration_s=60.0,
            source_video_path="/tmp/v", source_audio_path="/tmp/a",
            status="ingested", created_at=now,
        ))
        await PersonasRepo(db).create(
            persona_id="psn_ap", name="Voz", primary_language="es",
            target_languages=["es"], voice_prompt="energético",
        )
    await _seed_clip(
        db, tid, clip_id="clp_1", persona_id="psn_ap",
        caption="Cuerpo uno", hook="Hook uno", hashtags=["gaming", "clips"],
        now=now,
    )
    await _seed_clip(
        db, tid, clip_id="clp_2", persona_id="psn_ap",
        caption="Cuerpo dos", hook="Hook dos", hashtags=["stream"], now=now,
    )
    return tenants["alice"]


def _mock_accounts(mock: respx.Router, *platforms: str) -> None:
    mock.get(f"{_ZBASE}/accounts").mock(return_value=httpx.Response(
        200,
        json={"accounts": [
            {"platform": p, "_id": f"acct_{p}", "profileId": "prof_alice"}
            for p in platforms
        ]},
    ))


# ---- composer ----


@pytest.mark.asyncio
async def test_build_post_stitches_hook_body_hashtags_suffix(
    db: Database, alice: dict[str, str]
) -> None:
    from chalybclip.publish.compose import build_post

    with bound_tenant(alice["id"]):
        post = await build_post(
            db, "clp_1", handle_suffix="@minombre #marca",
        )
    assert post.hook == "Hook uno"
    assert post.title == "Hook uno"
    assert post.hashtags == ["#gaming", "#clips"]
    assert post.caption == "Hook uno\n\nCuerpo uno\n\n#gaming #clips\n\n@minombre #marca"


@pytest.mark.asyncio
async def test_build_post_no_variant_is_empty(
    db: Database, alice: dict[str, str]
) -> None:
    from chalybclip.publish.compose import build_post

    with bound_tenant(alice["id"]):
        post = await build_post(db, "clp_missing", handle_suffix="@x")
    # No variant → only the fixed suffix survives; no spurious title.
    assert post.hook == ""
    assert post.title is None
    assert post.caption == "@x"
    assert post.is_degenerate  # nothing real to publish


@pytest.mark.asyncio
async def test_build_post_flags_degenerate_one_word_caption(
    db: Database, alice: dict[str, str]
) -> None:
    """A hookless variant whose caption is a stray token (the persona-name
    fallback that shipped literal "viral" posts) must be flagged so the
    automated publish paths refuse it."""
    from chalybclip.publish.compose import build_post

    now = _dt.datetime.now(_dt.UTC).isoformat()
    await _seed_clip(
        db, alice["id"], clip_id="clp_degen", persona_id="psn_ap",
        caption="viral", hook="", hashtags=[], now=now,
    )
    with bound_tenant(alice["id"]):
        post = await build_post(db, "clp_degen", handle_suffix="@doomscroll")
    assert post.is_degenerate
    # ...while a post with a real hook is not degenerate even with a short body.
    with bound_tenant(alice["id"]):
        healthy = await build_post(db, "clp_1", handle_suffix="")
    assert not healthy.is_degenerate


# ---- /compose/{clip_id} ----


@pytest.mark.asyncio
async def test_compose_endpoint_returns_enriched_text(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=False, mode="on_approve", targets="tiktok",
        post_mode="queue", daily_cap=10, score_threshold=0.6,
        tag_suffix="@minombre #marca",
    )
    resp = await client.post(
        "/dashboard/publish/zernio/compose/clp_1", headers=auth(alice["token"]),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["title"] == "Hook uno"
    assert body["hashtags"] == ["#gaming", "#clips"]
    assert "Hook uno" in body["caption"]
    assert "@minombre #marca" in body["caption"]


# ---- /schedule/auto ----


@pytest.mark.asyncio
async def test_schedule_auto_schedules_all_approved_with_enrichment(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="on_approve", targets="tiktok",
        post_mode="queue", daily_cap=10, score_threshold=0.6,
        tag_suffix="@minombre #marca",
    )
    posted: list[str] = []

    def _create(request: httpx.Request) -> httpx.Response:
        pid = f"post_auto_{len(posted)}"
        posted.append(pid)
        return httpx.Response(201, json={"success": True, "post": {"_id": pid}})

    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        # Fresh tenant (no publish history) → engagement scheduling is
        # gated off, so best-time is NOT fetched at all and the planner
        # uses the fixed fallback hours. (If the gate regressed and the
        # code called best-time, respx would 404 the un-mocked route.)
        post_route = mock.post(f"{_ZBASE}/posts").mock(side_effect=_create)
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/auto", headers=auth(alice["token"]),
        )
        # The run now executes in the background; the POST returns immediately.
        assert resp.status_code == 200, resp.text
        assert resp.json()["state"] == "running"
        # Poll progress until the background task finishes — keep the respx
        # mock open so the task's POST /posts calls hit it.
        prog: dict[str, Any] = {}
        for _ in range(500):
            prog = (
                await client.get(
                    "/dashboard/publish/zernio/schedule/auto/progress",
                    headers=auth(alice["token"]),
                )
            ).json()
            if prog.get("state") in ("done", "error"):
                break
            await asyncio.sleep(0.02)

    assert prog.get("state") == "done", prog
    assert prog["scheduled"] == 2
    assert prog["failed"] == 0
    assert post_route.call_count == 2

    # Every scheduled post carries a future scheduledFor and the enriched
    # caption (hook + body + hashtags + fixed handle suffix).
    payloads = [json.loads(c.request.content.decode()) for c in post_route.calls]
    for p in payloads:
        assert p.get("scheduledFor")
        assert p["timezone"] == "UTC"
        assert "publishNow" not in p
    contents = {p["content"] for p in payloads}
    assert any("Hook uno" in c and "@minombre #marca" in c for c in contents)
    assert any("Hook dos" in c for c in contents)


@pytest.mark.asyncio
async def test_schedule_auto_409s_while_a_run_is_fresh(
    publish_env: None, client: httpx.AsyncClient, db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A held cross-worker lock blocks a second click — no double-schedule."""
    from chalybclip.db import AutoprogLocksRepo

    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(tid, profile_id="prof_alice", profile_name="A")
    # Simulate a run already in progress on another worker by holding the lock.
    token = await AutoprogLocksRepo(db).acquire(tid)
    assert token is not None
    try:
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/auto",
            headers=auth(tenants["alice"]["token"]),
        )
        assert resp.status_code == 409
        assert "en curso" in resp.json()["error"]
    finally:
        await AutoprogLocksRepo(db).release(tid, token)


@pytest.mark.asyncio
async def test_schedule_auto_overrides_a_stale_run(
    publish_env: None, client: httpx.AsyncClient, db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A crashed run leaves a STALE DB lock; it's reclaimed so the tenant isn't
    wedged at 'en curso'. (The lock's own staleness math is unit-tested in
    tests/db/test_autoprog_locks.py — here we prove the endpoint isn't blocked
    by an old lock.)"""
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(tid, profile_id="prof_alice", profile_name="A")
    # Plant a STALE lock (claimed long ago) — a crashed run that never released.
    conn = await db.connect()
    await conn.execute(
        "INSERT INTO autoprog_locks (tenant_id, token, claimed_at) VALUES (?, ?, ?)",
        (tid, "dead", "2020-01-01T00:00:00+00:00"),
    )
    await conn.commit()
    resp = await client.post(
        "/dashboard/publish/zernio/schedule/auto",
        headers=auth(tenants["alice"]["token"]),
    )
    # Stale lock reclaimed → not wedged at 409. (No approved clips → clean done.)
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_schedule_auto_no_clips_is_clean_ok(
    publish_env: None, client: httpx.AsyncClient, db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(tid, profile_id="prof_alice", profile_name="A")
    resp = await client.post(
        "/dashboard/publish/zernio/schedule/auto", headers=auth(tenants["alice"]["token"]),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["scheduled"] == 0


# ---- hands-free best-time spread ----


@pytest.mark.asyncio
async def test_handsfree_queue_spreads_and_enriches(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """A finished VOD's clips, in hands_free + queue mode, schedule across
    best-time slots (scheduledFor set, not immediate) with enriched text."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep

    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="queue", daily_cap=10, score_threshold=0.5,
        tag_suffix="@minombre #marca",
    )
    posted: list[str] = []

    def _create(request: httpx.Request) -> httpx.Response:
        pid = f"post_hf_{len(posted)}"
        posted.append(pid)
        return httpx.Response(201, json={"success": True, "post": {"_id": pid}})

    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        # Fresh tenant → engagement gated off → best-time not fetched;
        # queue mode still spreads, just on the fallback hours.
        post_route = mock.post(f"{_ZBASE}/posts").mock(side_effect=_create)
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_1", 0.9), ("clp_2", 0.8)],
        )

    assert n == 2
    assert post_route.call_count == 2
    payloads = [json.loads(c.request.content.decode()) for c in post_route.calls]
    for p in payloads:
        assert p.get("scheduledFor")  # queue mode → spread, not immediate
        assert p.get("publishNow") in (None, False)
    assert any("@minombre #marca" in p["content"] for p in payloads)


@pytest.mark.asyncio
async def test_handsfree_first_clip_scheduled_immediately(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """The single, first eligible clip schedules ~now (min_gap spaces the rest,
    it shouldn't delay the first) — the rulebook path always uses scheduledFor."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep

    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="now", daily_cap=10, score_threshold=0.5, tag_suffix="",
    )
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            side_effect=lambda r: httpx.Response(
                201, json={"success": True, "post": {"_id": "post_now"}}
            )
        )
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_1", 0.9)],
        )
    assert n == 1
    from chalybclip.publish.pacing import DEFAULT_PLATFORM_RULES
    rule = DEFAULT_PLATFORM_RULES["tiktok"]
    payload = json.loads(post_route.calls.last.request.content.decode())
    # Scheduled ~now (within the jitter window), NOT a full min_gap (180m) out.
    when = _dt.datetime.fromisoformat(payload["scheduledFor"])
    assert when <= _dt.datetime.now(_dt.UTC) + _dt.timedelta(minutes=rule.jitter_minutes + 2)


@pytest.mark.asyncio
async def test_handsfree_publishes_low_signal_but_publish_ready_clip(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """Auto-publish gates on the PUBLISHABILITY verdict, not the raw detector
    score. A YouTube-style clip with a tiny candidate score (no chat heat) but
    a publish_ready render must still go out — the old detector-score gate
    silently dropped every such clip."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep

    now = _dt.datetime.now(_dt.UTC).isoformat()
    await _seed_clip(
        db, alice["id"], clip_id="clp_yt", persona_id="psn_ap",
        caption="Cuerpo yt", hook="Hook yt", hashtags=["yt"], now=now,
        candidate_score=0.1, publishability_score=55,
        publishability_status="publish_ready",
    )
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="now", daily_cap=10, score_threshold=0.5, tag_suffix="",
    )
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            side_effect=lambda r: httpx.Response(
                201, json={"success": True, "post": {"_id": "post_yt"}}
            )
        )
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_yt", 0.1)],  # low detector score — must NOT gate
        )
    assert n == 1
    assert post_route.call_count == 1


@pytest.mark.asyncio
async def test_handsfree_skips_reject_clip_despite_high_detector_score(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """A clip the publishability verdict marks `reject` is never auto-posted,
    even when its detector score and raw publishability score are high (a
    blocking integrity issue forces reject)."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep

    now = _dt.datetime.now(_dt.UTC).isoformat()
    await _seed_clip(
        db, alice["id"], clip_id="clp_bad", persona_id="psn_ap",
        caption="Cuerpo", hook="Hook", hashtags=["x"], now=now,
        candidate_score=0.9, publishability_score=80,
        publishability_status="reject",
    )
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="now", daily_cap=10, score_threshold=0.5, tag_suffix="",
    )
    with respx.mock(assert_all_called=False) as mock:
        _mock_accounts(mock, "tiktok")
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            side_effect=lambda r: httpx.Response(
                201, json={"success": True, "post": {"_id": "post_bad"}}
            )
        )
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_bad", 0.9)],
        )
    assert n == 0
    assert post_route.call_count == 0


@pytest.mark.asyncio
async def test_handsfree_skips_degenerate_caption_clip(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """Hands-free must never ship a contentless post (no hook, one-word
    body) — the exact failure that published 18 shorts titled "viral"."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep

    now = _dt.datetime.now(_dt.UTC).isoformat()
    await _seed_clip(
        db, alice["id"], clip_id="clp_viral", persona_id="psn_ap",
        caption="viral", hook="", hashtags=[], now=now,
        candidate_score=0.9, publishability_score=80,
        publishability_status="publish_ready",
    )
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="now", daily_cap=10, score_threshold=0.5, tag_suffix="@doomscroll",
    )
    with respx.mock(assert_all_called=False) as mock:
        _mock_accounts(mock, "tiktok")
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            side_effect=lambda r: httpx.Response(
                201, json={"success": True, "post": {"_id": "post_degen"}}
            )
        )
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_viral", 0.9)],
        )
    assert n == 0
    assert post_route.call_count == 0


@pytest.mark.asyncio
async def test_handsfree_empty_targets_falls_back_to_all_connected(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """Regression: hands_free with NO target platforms picked must post to all
    connected accounts (it used to silently no-op on empty targets, so
    channel-auto clips never published)."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep

    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="",  # EMPTY
        post_mode="now", daily_cap=10, score_threshold=0.5, tag_suffix="",
    )
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            side_effect=lambda r: httpx.Response(
                201, json={"success": True, "post": {"_id": "post_ft"}}
            )
        )
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_1", 0.9)],
        )
    # Empty targets → all connected. The rulebook path posts one per platform
    # (per-platform assets/scheduling), so one clip cross-posts to both
    # connected accounts: 2 posts.
    assert n == 2
    assert post_route.call_count == 2


@pytest.mark.asyncio
async def test_handsfree_spaces_by_platform_min_gap(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """Two clips to TikTok → consecutive scheduled posts respect TikTok's
    min_gap from the rulebook (180 min), not a flat 30-min drip. The first
    fires ~now; the second a gap later (± the jitter window)."""
    from chalybclip.api.routers.zernio import autopublish_hands_free_sweep
    from chalybclip.publish.pacing import DEFAULT_PLATFORM_RULES

    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="queue", daily_cap=10, score_threshold=0.5, tag_suffix="",
    )
    payloads: list[dict[str, Any]] = []

    def _create(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content.decode()))
        pid = f"post_{len(payloads)}"
        return httpx.Response(201, json={"success": True, "post": {"_id": pid}})

    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        mock.post(f"{_ZBASE}/posts").mock(side_effect=_create)
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_1", 0.9), ("clp_2", 0.8)],
        )
    assert n == 2
    rule = DEFAULT_PLATFORM_RULES["tiktok"]
    whens = sorted(_dt.datetime.fromisoformat(p["scheduledFor"]) for p in payloads)
    gap = whens[1] - whens[0]
    assert gap >= _dt.timedelta(minutes=rule.min_gap_minutes - 2 * rule.jitter_minutes)


# ---- tag_suffix round-trip ----


@pytest.mark.asyncio
async def test_tag_suffix_roundtrips_through_save_and_json(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/autopublish/save",
        json={
            "enabled": True, "mode": "on_approve", "targets": ["tiktok"],
            "post_mode": "queue", "daily_cap": 7,
            "tag_suffix": "  @minombre #marca  ",
        },
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    s = await AutopublishSettingsRepo(db).get(alice["id"])
    assert s is not None
    assert s["tag_suffix"] == "@minombre #marca"  # trimmed

    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        got = await client.get(
            "/dashboard/publish/zernio/autopublish.json", headers=auth(alice["token"]),
        )
    assert got.status_code == 200
    assert got.json()["settings"]["tag_suffix"] == "@minombre #marca"


# ---- per-platform daily counting (caps hold across sweeps) ----


@pytest.mark.asyncio
async def test_count_by_platform_today(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """Counts today's active posts per platform: splits the csv, canonicalizes
    x→twitter, and excludes failed/deleted rows."""
    tid = alice["id"]
    pubs = ZernioPublishesRepo(db)
    today = _dt.datetime.now(_dt.UTC).isoformat()
    await pubs.record(post_id="p1", tenant_id=tid, clip_id="c1",
                      platforms=["tiktok"], content="a", status="scheduled",
                      scheduled_for=today)
    await pubs.record(post_id="p2", tenant_id=tid, clip_id="c2",
                      platforms=["tiktok", "x"], content="b", status="scheduled",
                      scheduled_for=today)
    await pubs.record(post_id="p3", tenant_id=tid, clip_id="c3",
                      platforms=["tiktok"], content="c", status="failed",
                      scheduled_for=today)  # failed → not counted
    await pubs.record(post_id="p4", tenant_id=tid, clip_id="c4",
                      platforms=["youtube"], content="d", status="scheduled",
                      scheduled_for="2020-01-01T00:00:00+00:00")  # old day → not counted

    counts = await pubs.count_by_platform_today(tid)
    assert counts.get("tiktok") == 2  # p1 + p2 (p3 failed excluded)
    assert counts.get("twitter") == 1  # p2's "x" canonicalized
    assert "youtube" not in counts  # p4 is an old day


# ---- reprocess failed ----


async def _seed_failed(db: Database, tid: str, post_id: str, clip_id: str) -> None:
    """A local zernio_publishes row in the 'failed' state (what the
    expired-URL bug left behind)."""
    await ZernioPublishesRepo(db).record(
        post_id=post_id, tenant_id=tid, clip_id=clip_id,
        platforms=["tiktok"], content="Cuerpo " + clip_id, status="failed",
    )


@pytest.mark.asyncio
async def test_reprocess_failed_reschedules_via_rulebook(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    """Reprocess re-creates each failed post (fresh signed media URL) and
    re-schedules it under the rulebook (TikTok min_gap, not a flat drip); the
    old failed rows are tombstoned."""
    await _seed_failed(db, alice["id"], "post_f1", "clp_1")
    await _seed_failed(db, alice["id"], "post_f2", "clp_2")
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="queue", daily_cap=10,
    )
    created: list[dict[str, Any]] = []

    def _create(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        created.append(body)
        return httpx.Response(
            201, json={"success": True, "post": {"_id": f"new_{len(created)}"}}
        )

    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        mock.post(f"{_ZBASE}/posts").mock(side_effect=_create)
        mock.delete(url__regex=rf"{_ZBASE}/posts/.+").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/reprocess-failed", headers=auth(alice["token"]),
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["reprocessed"] == 2

    # Both new posts are SCHEDULED, spaced by TikTok's rulebook min_gap (180
    # min, ± jitter) — not a flat 60-min drip — each with a fresh signed URL.
    from chalybclip.publish.pacing import DEFAULT_PLATFORM_RULES
    rule = DEFAULT_PLATFORM_RULES["tiktok"]
    whens = sorted(_dt.datetime.fromisoformat(p["scheduledFor"]) for p in created)
    assert whens[1] - whens[0] >= _dt.timedelta(minutes=rule.min_gap_minutes - 2 * rule.jitter_minutes)
    for p in created:
        url = p.get("mediaUrl") or (p.get("mediaItems") or [{}])[0].get("url", "")
        assert "/api/internal/clip/" in url and "sig=" in url

    # Old failed rows tombstoned so they leave the failed list.
    old1 = await ZernioPublishesRepo(db).get_by_post_id("post_f1")
    old2 = await ZernioPublishesRepo(db).get_by_post_id("post_f2")
    assert old1 is not None and old1.status == "deleted"
    assert old2 is not None and old2.status == "deleted"


@pytest.mark.asyncio
async def test_reprocess_drops_disconnected_platform(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    """A tiktok+youtube failure, with only youtube reconnected, re-schedules
    to youtube ONLY — the gone platform is dropped, not re-targeted (which
    would just fail again)."""
    await ZernioPublishesRepo(db).record(
        post_id="post_mix", tenant_id=alice["id"], clip_id="clp_1",
        platforms=["tiktok", "youtube"], content="Cuerpo", status="failed",
    )
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="youtube",
        post_mode="queue", daily_cap=10,
    )
    created: list[dict[str, Any]] = []

    def _create(request: httpx.Request) -> httpx.Response:
        created.append(json.loads(request.content.decode()))
        return httpx.Response(201, json={"success": True, "post": {"_id": "new_1"}})

    with respx.mock() as mock:
        _mock_accounts(mock, "youtube")  # tiktok no longer connected
        mock.post(f"{_ZBASE}/posts").mock(side_effect=_create)
        mock.delete(url__regex=rf"{_ZBASE}/posts/.+").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        resp = await client.post(
            "/dashboard/publish/zernio/reprocess-failed", headers=auth(alice["token"]),
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reprocessed"] == 1
    result = body["results"][0]
    assert result["ok"] is True
    assert result["platforms"] == ["youtube"]
    assert result["dropped"] == ["tiktok"]
    # The new post went out targeting youtube only — no tiktok account in it.
    assert len(created) == 1
    platforms_sent = {
        e["platform"].lower() for e in created[0]["platforms"]
    }
    assert platforms_sent == {"youtube"}


@pytest.mark.asyncio
async def test_reprocess_skips_when_no_platform_connected(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    """When none of a failed post's platforms are still connected, the clip
    is skipped with a clear reason rather than crashing the batch."""
    await ZernioPublishesRepo(db).record(
        post_id="post_gone", tenant_id=alice["id"], clip_id="clp_2",
        platforms=["tiktok"], content="Cuerpo", status="failed",
    )

    with respx.mock() as mock:
        _mock_accounts(mock, "youtube")  # only youtube; the failure was tiktok
        resp = await client.post(
            "/dashboard/publish/zernio/reprocess-failed", headers=auth(alice["token"]),
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reprocessed"] == 0
    result = body["results"][0]
    assert result["ok"] is False
    assert "conectada" in result["error"]


@pytest.mark.asyncio
async def test_reprocess_one_unknown_post_404s(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/reprocess/post_nope", headers=auth(alice["token"]),
    )
    assert resp.status_code == 404
    assert resp.json()["ok"] is False

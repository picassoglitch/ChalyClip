"""Per-clip overlay editor (slice F.6).

End-to-end through the dashboard:
  * GET /dashboard/clips/{id} renders the editor with the live-preview
    surface + control panels.
  * POST /dashboard/clips/{id}/overlay saves the overlay config but
    leaves status untouched.
  * POST /dashboard/clips/{id}/finalize saves AND walks the status to
    'approved' (the existing pre-publish standby).
  * Per-tenant isolation: Bob can't touch Alice's clip.
  * Idempotence: re-saving overwrites cleanly.
  * Status validation: a `published` clip can't be re-finalized.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import httpx

from chalybclip.db import (
    ClipsRepo,
    Database,
    StreamsRepo,
)
from chalybclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
)
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_clip(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str = "str_e",
    clip_id: str = "clp_e",
    status: str = "cut",
) -> None:
    from chalybclip.db import CandidatesRepo

    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url="https://kick.com/x",
                platform="kick",
                title="t",
                channel="c",
                duration_s=60.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_e",
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.9,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id,
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    candidate_id="cnd_e",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status=status,
                    created_at=_now(),
                )
            ]
        )


async def _login(client: httpx.AsyncClient, token: str) -> None:
    await client.post("/dashboard/login", data={"token": token})


# ---- Page render -------------------------------------------------


async def test_clip_editor_page_renders_live_preview_surface(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The editor page renders with the preview <video>, the four
    control panels, and the Complete + Save-draft buttons."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e")
    assert r.status_code == 200
    body = r.text

    # Preview surface + the four overlay layers.
    assert 'id="preview"' in body
    assert 'id="pv-title"' in body
    assert 'id="pv-banner"' in body
    assert 'id="pv-caption"' in body
    assert 'id="pv-rail"' in body

    # The four control panels.
    assert 'name="title_text"' in body
    assert 'name="banner_enabled"' in body
    assert 'name="banner_platform"' in body
    assert 'name="banner_url"' in body
    assert 'name="captions_preset"' in body
    assert 'name="captions_highlight_color"' in body
    assert 'name="comments_show"' in body

    # The two action buttons live on the same form via formaction.
    # CTA copy was rewritten in slice F.7 (creator-OS positioning):
    # "Complete & stage for publish" -> "Ship to platforms".
    assert "Ship to platforms" in body
    assert "Save draft" in body
    assert "/dashboard/clips/clp_e/overlay" in body
    assert "/dashboard/clips/clp_e/finalize" in body


async def test_clip_editor_pre_populates_from_saved_overlay(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A previously-saved overlay config populates the form on next render."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    with bound_tenant(tid):
        await ClipsRepo(db).set_overlay_config(
            "clp_e",
            overlay_config={
                "title_text": "Adin Ross asked Clav",
                "banner": {
                    "enabled": True,
                    "platform": "kick",
                    "url": "kick.com/clavicular",
                    "color": "#53FC18",
                },
                "captions": {
                    "enabled": True,
                    "preset": "bold_block",
                    "highlight_color": "#FF00FF",
                },
                "comments": {"show_overlay": True, "fake_likes": 42},
            },
        )
    r = await client.get("/dashboard/clips/clp_e")
    assert r.status_code == 200
    body = r.text
    assert "Adin Ross asked Clav" in body
    assert "kick.com/clavicular" in body
    assert "#FF00FF" in body
    assert "42" in body  # fake_likes


# ---- Save (draft) ----------------------------------------------


async def test_overlay_save_persists_config_only(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """POST /clips/{id}/overlay writes the config but leaves status
    where it was."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="cut")
    r = await client.post(
        "/dashboard/clips/clp_e/overlay",
        data={
            "title_text": "Hello world",
            "banner_enabled": "1",
            "banner_platform": "twitch",
            "banner_url": "twitch.tv/foo",
            "banner_color": "#9146FF",
            "captions_enabled": "1",
            "captions_preset": "karaoke_pop",
            "captions_highlight_color": "#FFD700",
            "comments_show": "1",
            "comments_fake_likes": "100",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/clips/clp_e"
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "cut"  # unchanged
    cfg = clip.overlay_config
    assert cfg is not None
    assert cfg["title_text"] == "Hello world"
    banner = cfg["banner"]
    assert isinstance(banner, dict)
    assert banner["enabled"] is True
    assert banner["platform"] == "twitch"
    assert banner["url"] == "twitch.tv/foo"
    captions = cfg["captions"]
    assert isinstance(captions, dict)
    assert captions["preset"] == "karaoke_pop"
    comments = cfg["comments"]
    assert isinstance(comments, dict)
    assert comments["fake_likes"] == 100


async def test_overlay_save_blank_title_clears_to_none(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A blank title_text form field collapses to None in storage —
    the editor is additive, not destructive."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.post(
        "/dashboard/clips/clp_e/overlay",
        data={
            "title_text": "",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    cfg = clip.overlay_config
    assert cfg is not None
    assert cfg["title_text"] is None


async def test_overlay_save_404_for_unknown_clip(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.post(
        "/dashboard/clips/clp_nope/overlay",
        data={"title_text": "x"},
        follow_redirects=False,
    )
    assert r.status_code == 404


# ---- Finalize ---------------------------------------------------


async def test_finalize_runs_overlay_burn_when_overlays_enabled(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch,
    tmp_path,
) -> None:
    """Slice F.7-E: finalize triggers a renderer-side burn pass.
    Patch ClipsRepo.get to return a clip whose path lives under
    tmp_path so the burn helper can find it on disk; patch
    burn_overlays to confirm it gets called with the right config."""
    from chalybclip.clip import overlay_burn

    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])

    # Seed a clip whose path is a real file on disk so the burn
    # helper's "source missing" guard doesn't short-circuit.
    clip_dir = tmp_path / "clips" / "clp_e"
    clip_dir.mkdir(parents=True)
    clip_path = clip_dir / "clip.mp4"
    clip_path.write_bytes(b"fake mp4")

    with bound_tenant(tid):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_e", tenant_id=tid, vod_url="x", platform="kick",
                title="t", channel="c", duration_s=60.0,
                source_video_path="/tmp/v", source_audio_path="/tmp/a",
                status="ingested", created_at=_now(),
            )
        )
        from chalybclip.db import CandidatesRepo
        await CandidatesRepo(db).upsert_many([CandidateRow(
            id="cnd_e", stream_id="str_e", tenant_id=tid, ts=10.0,
            score=0.9, reason="voice", evidence={}, created_at=_now(),
        )])
        await ClipsRepo(db).upsert_many([ClipRow(
            id="clp_e", stream_id="str_e", tenant_id=tid,
            candidate_id="cnd_e",
            start_s=0.0, end_s=10.0, duration_s=10.0,
            width=1080, height=1920, path=str(clip_path),
            status="cut", created_at=_now(),
        )])

    captured: dict[str, object] = {}

    def fake_burn(**kwargs):
        captured.update(kwargs)
        # Pretend we wrote the final MP4.
        Path(kwargs["target_path"]).write_bytes(b"burned")
        return True

    # Patch BOTH the module-of-origin AND the re-exported name on
    # chalybclip.clip — the dashboard handler imports
    # `from chalybclip.clip import burn_overlays` at call time, so
    # the re-export binding is what it actually grabs.
    import chalybclip.clip
    monkeypatch.setattr(overlay_burn, "burn_overlays", fake_burn)
    monkeypatch.setattr(chalybclip.clip, "burn_overlays", fake_burn)

    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "Adin Ross asked Clav for RETA",
            "banner_enabled": "1",
            "banner_platform": "kick",
            "banner_url": "kick.com/clavicular",
            "banner_color": "#53FC18",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303

    # Burn was called with the operator's overlay config.
    assert captured["source_path"] == clip_path
    assert captured["target_path"] == clip_dir / "clip_final.mp4"
    cfg = captured["overlay_config"]
    assert isinstance(cfg, dict)
    assert cfg["title_text"] == "Adin Ross asked Clav for RETA"
    assert cfg["banner"]["platform"] == "kick"
    # And the burned file lives on disk now — publishers will pick it up.
    assert (clip_dir / "clip_final.mp4").exists()


async def test_finalize_tolerates_burn_failure(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch,
    tmp_path,
) -> None:
    """If ffmpeg fails, finalize still succeeds (status moves,
    overlay_config persists). Publishers fall back to the original
    clip.mp4 — the operator can retry from the editor."""
    from chalybclip.clip import overlay_burn

    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])

    clip_dir = tmp_path / "clips" / "clp_e"
    clip_dir.mkdir(parents=True)
    clip_path = clip_dir / "clip.mp4"
    clip_path.write_bytes(b"fake")
    with bound_tenant(tid):
        await StreamsRepo(db).upsert(StreamRow(
            id="str_e", tenant_id=tid, vod_url="x", platform="kick",
            title="t", channel="c", duration_s=60.0,
            source_video_path="/tmp/v", source_audio_path="/tmp/a",
            status="ingested", created_at=_now(),
        ))
        from chalybclip.db import CandidatesRepo
        await CandidatesRepo(db).upsert_many([CandidateRow(
            id="cnd_e", stream_id="str_e", tenant_id=tid, ts=10.0,
            score=0.9, reason="voice", evidence={}, created_at=_now(),
        )])
        await ClipsRepo(db).upsert_many([ClipRow(
            id="clp_e", stream_id="str_e", tenant_id=tid,
            candidate_id="cnd_e",
            start_s=0.0, end_s=10.0, duration_s=10.0,
            width=1080, height=1920, path=str(clip_path),
            status="cut", created_at=_now(),
        )])

    def boom(**kwargs):
        raise RuntimeError("ffmpeg burn failed: pretend stderr")

    import chalybclip.clip
    monkeypatch.setattr(overlay_burn, "burn_overlays", boom)
    monkeypatch.setattr(chalybclip.clip, "burn_overlays", boom)

    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "x",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    # Status still moves even though the burn failed.
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "approved"
    assert clip.overlay_config is not None
    # No burned file → publishers fall back to clip.mp4.
    assert not (clip_dir / "clip_final.mp4").exists()


async def test_finalize_walks_cut_to_approved(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A `cut` clip → `ready_for_review` → `approved` in one POST."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="cut")
    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "Final",
            "banner_enabled": "1",
            "banner_platform": "kick",
            "banner_url": "kick.com/me",
            "banner_color": "#53FC18",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "1",
            "comments_fake_likes": "8",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "approved"
    assert clip.overlay_config is not None
    assert clip.overlay_config["title_text"] == "Final"


async def test_finalize_from_ready_for_review_goes_to_approved(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="ready_for_review")
    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "x",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "approved"


async def test_finalize_idempotent_when_already_approved(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Re-finalizing an already-approved clip overwrites the overlay
    but leaves the status alone (no 409)."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="approved")
    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "v2",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "approved"
    assert clip.overlay_config is not None
    assert clip.overlay_config["title_text"] == "v2"


async def test_finalize_409_on_published_clip(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A `published` clip is terminal — can't be re-finalized."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="published")
    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "x",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 409


async def test_finalize_404_for_unknown_clip(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.post(
        "/dashboard/clips/clp_nope/finalize",
        data={"title_text": "x"},
        follow_redirects=False,
    )
    assert r.status_code == 404


# ---- Tenant isolation ------------------------------------------


async def test_clip_editor_renders_waveform_scrubber(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The preview now has a click-to-seek scrubber + SVG waveform
    placeholder under it. The waveform peaks load async via fetch
    against /dashboard/clips/{id}/waveform.json."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e")
    body = r.text
    assert 'id="waveform"' in body
    assert 'id="scrubber-bar"' in body
    assert 'id="scrubber-progress"' in body
    assert 'id="scrubber-handle"' in body
    assert "/dashboard/clips/clp_e/waveform.json" in body
    # Total time renders mm:ss from clip.duration_s (10s in fixture).
    assert "0:10" in body


async def test_captions_endpoint_returns_word_level_lines(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Slice F.7-F: /captions.json serves word-level caption lines
    chunked + emphasis-classified, ready for the editor's live
    preview to drive word-by-word animation."""
    import datetime as _dt
    import json

    from chalybclip.db import TranscriptsRepo
    from chalybclip.db.models import TranscriptRow

    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    # Seed a transcript with the REAL DB shape (ts/end_ts +
    # word-level data) — what production actually stores.
    segs = [
        {
            "ts": 0.0,
            "end_ts": 2.0,
            "text": "hello WORLD test",
            "words": [
                {"ts": 0.0, "end_ts": 0.5, "text": "hello", "prob": 0.9},
                {"ts": 0.5, "end_ts": 1.0, "text": "WORLD", "prob": 0.9},
                {"ts": 1.0, "end_ts": 2.0, "text": "test!", "prob": 0.9},
            ],
        },
    ]
    with bound_tenant(tid):
        await TranscriptsRepo(db).upsert(
            TranscriptRow(
                stream_id="str_e",
                tenant_id=tid,
                language="en",
                duration_s=60.0,
                model="medium",
                segments_json=json.dumps(segs),
                created_at=_dt.datetime.now(_dt.UTC).isoformat(),
            )
        )

    r = await client.get("/dashboard/clips/clp_e/captions.json")
    assert r.status_code == 200
    body = r.json()
    assert "lines" in body
    assert body["duration_s"] == 10.0  # fixture clip duration
    # Should have multiple chunked lines (WORLD breaks out as shout,
    # test! as emphasis, hello as its own).
    assert len(body["lines"]) >= 2
    # Emphasis tags actually flow through.
    emphases = {ln["emphasis"] for ln in body["lines"]}
    assert "shout" in emphases or "emphasis" in emphases


async def test_captions_endpoint_404_for_unknown_clip(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.get("/dashboard/clips/clp_nope/captions.json")
    assert r.status_code == 404


async def test_captions_endpoint_returns_empty_when_no_transcript(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """No transcript yet → empty lines (the editor's fallback)."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e/captions.json")
    assert r.status_code == 200
    body = r.json()
    assert body["lines"] == []


async def test_waveform_endpoint_returns_empty_when_clip_missing_on_disk(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Fixture clip points at /tmp/c.mp4 which doesn't exist; the
    endpoint must return [] (200) instead of 500ing so the editor
    JS gracefully degrades."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e/waveform.json")
    assert r.status_code == 200
    assert r.json() == []


async def test_waveform_endpoint_404_for_unknown_clip(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.get("/dashboard/clips/clp_nope/waveform.json")
    assert r.status_code == 404


async def test_clip_editor_renders_ai_insights_strip(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Slice F.7: the clip editor surfaces the four AI scores
    (viral, hook strength, caption readability, dead-air risk) as
    a strip above the editor split-pane."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e")
    assert r.status_code == 200
    body = r.text
    # The AI-strip wrapper + the four labels.
    assert 'class="nc-ai-strip"' in body
    assert ">AI insights<" in body
    assert "Viral score" in body
    assert "Hook strength" in body
    assert "Caption readability" in body
    assert "Dead-air risk" in body
    # The viral-score progress bar fills from 0-100.
    assert 'class="nc-ai-score__bar-fill"' in body
    # One of the three label families must appear.
    assert any(
        label in body
        for label in ("HIGH", "MEDIUM", "DEVELOPING")
    )


async def test_clip_editor_right_panel_uses_step_numbered_sections(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Right-panel hierarchy: Viral hook → Captions → Branding →
    Advanced (collapsible). Each section header carries a numbered
    step badge so the operator reads them as ranked steps."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e")
    body = r.text
    assert ">Viral hook<" in body
    assert ">Captions<" in body
    assert ">Branding<" in body
    assert ">Advanced<" in body
    # Numbered step badges 1-4 — the visual hierarchy of the panel.
    for n in (1, 2, 3, 4):
        assert (
            f'class="nc-panel__step">{n}<'
        ) in body, f"missing step {n}"
    # Advanced section is wrapped in <details> so it collapses.
    assert 'id="advanced-section"' in body


async def test_clip_editor_renders_social_context_toggle(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Branding section has a 'Show platform context' checkbox and
    the preview surface carries the (initially hidden) LIVE badge +
    fake chat overlays."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e")
    body = r.text
    assert 'name="banner_show_context"' in body
    assert 'id="ctl-context-on"' in body
    assert 'id="pv-live"' in body
    assert 'id="pv-chat"' in body


async def test_overlay_save_persists_show_context_flag(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The new banner.show_context flag round-trips through
    save → load."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.post(
        "/dashboard/clips/clp_e/overlay",
        data={
            "title_text": "x",
            "banner_enabled": "1",
            "banner_platform": "kick",
            "banner_url": "kick.com/me",
            "banner_color": "#53FC18",
            "banner_show_context": "1",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    cfg = clip.overlay_config
    assert cfg is not None
    banner = cfg["banner"]
    assert isinstance(banner, dict)
    assert banner["show_context"] is True


async def test_clip_row_loader_tolerates_unknown_columns(
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Defense-in-depth regression: if a future migration adds a
    column to `clips` that the ClipRow Pydantic model doesn't yet
    declare, `_clip_from_row` must still load existing rows. Without
    this defense an out-of-sync deploy (DB ahead of code) crashes
    every dashboard page that reads a clip.

    We simulate this by injecting a fake column directly via SQL,
    then verifying the loader strips it without raising.
    """
    from chalybclip.db.repos import _clip_from_row

    tid = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id=tid)
    conn = await db.connect()
    # Add a hypothetical future column. SQLite ALTER TABLE is cheap.
    await conn.execute("ALTER TABLE clips ADD COLUMN imaginary_v10_col TEXT")
    await conn.execute(
        "UPDATE clips SET imaginary_v10_col = ? WHERE id = ?",
        ("future-data", "clp_e"),
    )
    await conn.commit()
    cur = await conn.execute(
        "SELECT * FROM clips WHERE id = ?",
        ("clp_e",),
    )
    row = await cur.fetchone()
    assert row is not None
    # The unknown column is in the row dict, but the loader filters
    # it out before model_validate — no ValidationError raised.
    clip = _clip_from_row(row)
    assert clip.id == "clp_e"


async def test_clip_editor_renders_intelligence_timeline(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The scrubber strip ships with an empty marker rail + legend
    that get populated async from /clips/{id}/intelligence.json."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e")
    body = r.text
    assert 'id="timeline-rail"' in body
    assert 'id="timeline-legend"' in body
    assert "/dashboard/clips/clp_e/intelligence.json" in body


async def test_intelligence_endpoint_returns_shape(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e/intelligence.json")
    assert r.status_code == 200
    body = r.json()
    assert "markers" in body
    assert "duration_s" in body
    assert isinstance(body["markers"], list)
    # Fixture clip has no transcript / visual_signals / chat → empty markers.
    assert body["markers"] == []


async def test_intelligence_endpoint_404_for_unknown_clip(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.get("/dashboard/clips/clp_nope/intelligence.json")
    assert r.status_code == 404


async def test_intelligence_endpoint_emits_voice_trigger_marker(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Pins the candidate→trigger-marker path that 500'd in prod with
    `AttributeError: 'CandidateRow' object has no attribute 'timestamp'`.

    Two parallel models almost shadow each other:
      - `chalybclip.detect.models.Candidate` (in-memory detector output,
        field name `timestamp`)
      - `chalybclip.db.models.CandidateRow` (DB row, field name `ts`)

    `CandidatesRepo.list_for_stream` returns the DB shape, so the
    endpoint MUST read `.ts`. The pre-fix code used `.timestamp` →
    AttributeError → 500 on every clip whose candidate carried a
    voice-trigger phrase. Re-seed with that exact shape and assert the
    response is 200 with the expected clip-relative ts.
    """
    from chalybclip.db import CandidatesRepo

    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    # Re-seed the candidate with a non-empty `evidence` so the endpoint
    # enters the trigger-marker branch. ts=10.0 (stream-absolute) and
    # the seeded clip starts at 0.0 → clip-relative ts should be 10.0.
    with bound_tenant(tid):
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_e",
                    stream_id="str_e",
                    tenant_id=tid,
                    ts=10.0,
                    score=0.9,
                    reason="voice",
                    evidence={
                        "phrase": "que clipearon",
                        "trigger_kind": "live",
                        "confidence": 0.87,
                    },
                    created_at=_now(),
                )
            ]
        )

    r = await client.get("/dashboard/clips/clp_e/intelligence.json")
    assert r.status_code == 200, r.text
    body = r.json()
    voice_triggers = [m for m in body["markers"] if m["kind"] == "voice_trigger"]
    assert len(voice_triggers) == 1, body
    marker = voice_triggers[0]
    assert marker["ts"] == 10.0  # ts=10 stream-abs minus clip.start_s=0
    assert marker["score"] == 0.87
    assert "que clipearon" in marker["label"]


async def test_clip_editor_renders_hook_generator_ui(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The Viral hook section now ships with an AI tone picker + a
    'Generate 5' button that hits /clips/{id}/generate-hooks."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e")
    body = r.text
    assert 'id="hook-tone"' in body
    assert 'id="hook-gen-btn"' in body
    assert 'id="hook-results"' in body
    assert "/dashboard/clips/clp_e/generate-hooks" in body
    # All five tone presets in the dropdown.
    for tone in ("default", "aggressive", "gen_z", "corporate", "curious"):
        assert f'value="{tone}"' in body


async def test_generate_hooks_endpoint_returns_json(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch,
    tmp_path,
) -> None:
    """End-to-end: POST /clips/{id}/generate-hooks → patches the
    real Anthropic factory with a FakeProvider → returns the canned
    hooks as JSON shaped {hooks: [{text}], tone, n}."""
    from chalybclip.llm import router as router_module
    from chalybclip.settings import get_settings

    monkeypatch.setenv("CHALYBCLIP_DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_settings.cache_clear()

    from tests.llm._fakes import FakeProvider  # type: ignore[import]

    fake = FakeProvider("anthropic")
    fake.queue_success({
        "hooks": [
            {"text": "title one"},
            {"text": "title two"},
            {"text": "title three"},
        ],
    })

    def factory(name, _config, _api_key):
        return fake if name == "anthropic" else None

    monkeypatch.setattr(router_module, "_default_provider_factory", factory)

    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.post(
        "/dashboard/clips/clp_e/generate-hooks",
        data={"tone": "aggressive", "n": "3"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tone"] == "aggressive"
    assert body["n"] == 3
    assert [h["text"] for h in body["hooks"]] == [
        "title one",
        "title two",
        "title three",
    ]
    assert body["source"] == "llm"
    get_settings.cache_clear()


async def test_generate_hooks_endpoint_falls_back_when_llm_fails(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch,
    tmp_path,
) -> None:
    """LLM down / out of credits → the button still returns usable titles,
    built deterministically from the transcript/stream title (no 502)."""
    from chalybclip.llm import router as router_module
    from chalybclip.settings import get_settings

    monkeypatch.setenv("CHALYBCLIP_DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_settings.cache_clear()

    from tests.llm._fakes import FakeProvider  # type: ignore[import]

    fake = FakeProvider("anthropic")
    fake.queue_fatal("anthropic 400: Your credit balance is too low")

    def factory(name, _config, _api_key):
        return fake if name == "anthropic" else None

    monkeypatch.setattr(router_module, "_default_provider_factory", factory)
    router_module.reset_billing_lockouts()

    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    try:
        r = await client.post(
            "/dashboard/clips/clp_e/generate-hooks",
            data={"tone": "default", "n": "5"},
        )
    finally:
        router_module.reset_billing_lockouts()
        get_settings.cache_clear()
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "deterministic"
    assert body["hooks"]  # never empty — templates as the last resort
    assert all(h["text"].strip() for h in body["hooks"])


async def test_generate_hooks_endpoint_404_for_unknown_clip(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.post(
        "/dashboard/clips/clp_nope/generate-hooks",
        data={"tone": "default", "n": "5"},
    )
    assert r.status_code == 404


async def test_generate_hooks_endpoint_clamps_invalid_inputs(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch,
    tmp_path,
) -> None:
    """Bogus tone falls back to 'default'; n outside [1, 10] gets clamped."""
    from chalybclip.llm import router as router_module
    from chalybclip.settings import get_settings

    monkeypatch.setenv("CHALYBCLIP_DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_settings.cache_clear()

    from tests.llm._fakes import FakeProvider  # type: ignore[import]

    fake = FakeProvider("anthropic")
    fake.queue_success({"hooks": [{"text": "x"}]})

    def factory(name, _config, _api_key):
        return fake if name == "anthropic" else None

    monkeypatch.setattr(router_module, "_default_provider_factory", factory)

    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.post(
        "/dashboard/clips/clp_e/generate-hooks",
        data={"tone": "BOGUS", "n": "999"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tone"] == "default"  # bogus → default
    assert body["n"] == 10  # clamped from 999
    get_settings.cache_clear()


async def test_overlay_endpoints_isolated_per_tenant(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Bob can't save / finalize Alice's clip — both endpoints 404."""
    await _seed_clip(db, tenant_id=tenants["alice"]["id"])
    await _login(client, tenants["bob"]["token"])
    for path in (
        "/dashboard/clips/clp_e/overlay",
        "/dashboard/clips/clp_e/finalize",
    ):
        r = await client.post(
            path,
            data={"title_text": "hijack"},
            follow_redirects=False,
        )
        assert r.status_code == 404, f"{path} returned {r.status_code}"
    # And Alice's clip is untouched.
    with bound_tenant(tenants["alice"]["id"]):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.overlay_config is None

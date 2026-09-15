"""BrandKitsRepo + branding.service — slice C.1.

Covers:
  * CRUD round-trip
  * 'only one default per tenant' invariant (partial unique index in 006)
  * Resolution priority: speaker preferred → tenant default → None
  * Per-kit custom_trigger_phrases additive merge with the tenant base
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from chalybclip.branding import (
    merged_trigger_phrases_for_speaker,
    outro_enabled_for_clip,
    resolve_brand_kit_for_candidate,
    resolve_brand_kit_for_speaker,
)
from chalybclip.db import (
    BrandKitsRepo,
    Database,
    SpeakersRepo,
    StreamsRepo,
    TenantsRepo,
    VodSpeakersRepo,
    apply_migrations,
)
from chalybclip.db.models import CustomTriggerPhrases, StreamRow
from chalybclip.errors import ChalybClipError
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest.fixture
async def migrated_db(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    await apply_migrations(db)
    yield db
    await db.close()


async def _seed_tenant(db: Database, name: str = "Aldo") -> str:
    return (await TenantsRepo(db).create(name=name)).id


# ---- CRUD ----


async def test_create_get_roundtrip(migrated_db: Database) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        kit = await BrandKitsRepo(migrated_db).create(
            name="AARA",
            primary_color="#FF3366",
            accent_color="#FFD700",
            handle_tiktok="@aara_art",
            auto_publish_platforms=["tiktok", "shorts"],
            custom_trigger_phrases=CustomTriggerPhrases(
                forward=["córtalo"], retroactive=["monchi eso"]
            ),
        )
        fetched = await BrandKitsRepo(migrated_db).get(kit.id)

    assert fetched is not None
    assert fetched.name == "AARA"
    assert fetched.primary_color == "#FF3366"
    assert fetched.handle_tiktok == "@aara_art"
    assert fetched.auto_publish_platforms == ["tiktok", "shorts"]
    assert fetched.custom_trigger_phrases.forward == ["córtalo"]
    assert fetched.custom_trigger_phrases.retroactive == ["monchi eso"]
    assert fetched.is_default is False


async def _set_tier(db: Database, tenant_id: str, tier: str) -> None:
    conn = await db.connect()
    await conn.execute(
        "UPDATE tenants SET tier = ? WHERE id = ?", (tier, tenant_id)
    )
    await conn.commit()


# ---- end-card outro tier gating (migration 050) ----


async def test_outro_free_tier_always_on(migrated_db: Database) -> None:
    """Free tier always gets the end card — even if a kit somehow has
    the toggle off, the resolver forces it on."""
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        await BrandKitsRepo(migrated_db).create(
            name="K", primary_color="#fff", accent_color="#000",
            is_default=True, show_chalybclip_outro=False,
        )
    assert await outro_enabled_for_clip(
        migrated_db, tenant_id=tenant_id, stream_id="str_x"
    ) is True


async def test_outro_paid_tier_respects_toggle_off(migrated_db: Database) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    await _set_tier(migrated_db, tenant_id, "pro")
    with bound_tenant(tenant_id):
        await BrandKitsRepo(migrated_db).create(
            name="K", primary_color="#fff", accent_color="#000",
            is_default=True, show_chalybclip_outro=False,
        )
    assert await outro_enabled_for_clip(
        migrated_db, tenant_id=tenant_id, stream_id="str_x"
    ) is False


async def test_outro_paid_tier_default_on(migrated_db: Database) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    await _set_tier(migrated_db, tenant_id, "pro")
    with bound_tenant(tenant_id):
        await BrandKitsRepo(migrated_db).create(
            name="K", primary_color="#fff", accent_color="#000",
            is_default=True,
        )
    assert await outro_enabled_for_clip(
        migrated_db, tenant_id=tenant_id, stream_id="str_x"
    ) is True


async def test_outro_paid_tier_no_kit_defaults_on(migrated_db: Database) -> None:
    """No default kit → nothing to read a toggle from → keep the card."""
    tenant_id = await _seed_tenant(migrated_db)
    await _set_tier(migrated_db, tenant_id, "pro")
    assert await outro_enabled_for_clip(
        migrated_db, tenant_id=tenant_id, stream_id="str_x"
    ) is True


async def test_set_default_demotes_previous(migrated_db: Database) -> None:
    """Spec invariant: at most one brand_kit per tenant is the default."""
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        a = await BrandKitsRepo(migrated_db).create(
            name="A", primary_color="#000", accent_color="#FFF", is_default=True
        )
        b = await BrandKitsRepo(migrated_db).create(
            name="B", primary_color="#111", accent_color="#EEE", is_default=False
        )
        # Promote B; A should be demoted in the same transaction.
        await BrandKitsRepo(migrated_db).set_default(b.id)

        a_after = await BrandKitsRepo(migrated_db).get(a.id)
        b_after = await BrandKitsRepo(migrated_db).get(b.id)
        default = await BrandKitsRepo(migrated_db).get_default()

    assert a_after is not None and a_after.is_default is False
    assert b_after is not None and b_after.is_default is True
    assert default is not None and default.id == b.id


async def test_create_with_is_default_clears_existing(migrated_db: Database) -> None:
    """Creating a new kit with is_default=True demotes any prior default."""
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        a = await BrandKitsRepo(migrated_db).create(
            name="A", primary_color="#000", accent_color="#FFF", is_default=True
        )
        b = await BrandKitsRepo(migrated_db).create(
            name="B", primary_color="#111", accent_color="#EEE", is_default=True
        )
        a_after = await BrandKitsRepo(migrated_db).get(a.id)
        default = await BrandKitsRepo(migrated_db).get_default()

    assert a_after is not None and a_after.is_default is False
    assert default is not None and default.id == b.id


async def test_update_partial_fields_only(migrated_db: Database) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        kit = await BrandKitsRepo(migrated_db).create(
            name="A",
            primary_color="#FF0000",
            accent_color="#00FF00",
            auto_publish_enabled=False,
        )
        updated = await BrandKitsRepo(migrated_db).update(
            kit.id, primary_color="#0000FF", auto_publish_enabled=True
        )
    assert updated.primary_color == "#0000FF"
    assert updated.accent_color == "#00FF00"  # untouched
    assert updated.auto_publish_enabled is True


async def test_list_for_tenant_isolated(migrated_db: Database) -> None:
    """Cross-tenant brand kits MUST NOT leak."""
    t1 = await _seed_tenant(migrated_db, name="Alice")
    t2 = await _seed_tenant(migrated_db, name="Bob")
    with bound_tenant(t1):
        await BrandKitsRepo(migrated_db).create(
            name="Alice Kit", primary_color="#000", accent_color="#FFF"
        )
    with bound_tenant(t2):
        await BrandKitsRepo(migrated_db).create(
            name="Bob Kit", primary_color="#111", accent_color="#EEE"
        )
        kits = await BrandKitsRepo(migrated_db).list_for_tenant()
    assert len(kits) == 1
    assert kits[0].name == "Bob Kit"


async def test_delete_removes_kit(migrated_db: Database) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        kit = await BrandKitsRepo(migrated_db).create(
            name="K", primary_color="#000", accent_color="#FFF"
        )
        await BrandKitsRepo(migrated_db).delete(kit.id)
        gone = await BrandKitsRepo(migrated_db).get(kit.id)
    assert gone is None


async def test_set_default_unknown_kit_raises(migrated_db: Database) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id), pytest.raises(ChalybClipError):
        await BrandKitsRepo(migrated_db).set_default("brk_nonexistent")


# ---- speaker.preferred_brand_kit_id FK ----


async def test_speaker_preferred_brand_kit_persists(migrated_db: Database) -> None:
    """Migration 006 made preferred_brand_kit_id a real FK; assigning works."""
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        kit = await BrandKitsRepo(migrated_db).create(
            name="AARA", primary_color="#FF3366", accent_color="#FFD700"
        )
        speaker = await SpeakersRepo(migrated_db).create(
            display_name="Aldo",
            embedding=[1.0, 0.0],
            total_speech_s=120.0,
        )
        updated = await SpeakersRepo(migrated_db).set_preferred_brand_kit(
            speaker.id, kit.id
        )
    assert updated is not None
    assert updated.preferred_brand_kit_id == kit.id


# ---- resolution priority chain ----


async def test_resolve_uses_speaker_preferred_kit(migrated_db: Database) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        default_kit = await BrandKitsRepo(migrated_db).create(
            name="Default",
            primary_color="#000",
            accent_color="#FFF",
            is_default=True,
        )
        speaker_kit = await BrandKitsRepo(migrated_db).create(
            name="Aldo's Kit", primary_color="#FF3366", accent_color="#FFD700"
        )
        speaker = await SpeakersRepo(migrated_db).create(
            display_name="Aldo", embedding=[1.0, 0.0], total_speech_s=120.0
        )
        await SpeakersRepo(migrated_db).set_preferred_brand_kit(
            speaker.id, speaker_kit.id
        )
        resolved = await resolve_brand_kit_for_speaker(
            migrated_db, speaker_id=speaker.id
        )

    assert resolved is not None
    assert resolved.id == speaker_kit.id
    # Sanity: the default kit exists; we just didn't choose it.
    assert default_kit.is_default is True


async def test_resolve_falls_back_to_tenant_default(migrated_db: Database) -> None:
    """No preferred kit on the speaker → tenant default wins."""
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        default_kit = await BrandKitsRepo(migrated_db).create(
            name="Default",
            primary_color="#000",
            accent_color="#FFF",
            is_default=True,
        )
        speaker = await SpeakersRepo(migrated_db).create(
            display_name="Unknown", embedding=[1.0, 0.0], total_speech_s=60.0
        )
        resolved = await resolve_brand_kit_for_speaker(
            migrated_db, speaker_id=speaker.id
        )

    assert resolved is not None
    assert resolved.id == default_kit.id


async def test_resolve_returns_none_when_no_default(migrated_db: Database) -> None:
    """No speaker preference + no tenant default → None (renderer falls
    back to system defaults). Spec's third tier."""
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        speaker = await SpeakersRepo(migrated_db).create(
            display_name="X", embedding=[1.0, 0.0], total_speech_s=60.0
        )
        resolved = await resolve_brand_kit_for_speaker(
            migrated_db, speaker_id=speaker.id
        )
    assert resolved is None


async def test_resolve_for_candidate_via_vod_speakers(
    migrated_db: Database,
) -> None:
    """End-to-end: candidate.evidence['speaker_label'] resolves through
    vod_speakers → speaker → preferred kit."""
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        await StreamsRepo(migrated_db).upsert(
            StreamRow(
                id="str_rcand",
                tenant_id=tenant_id,
                vod_url="upload://x.mp4",
                platform="upload",
                title="t",
                channel=None,
                duration_s=120.0,
                source_video_path="/tmp/v.mp4",
                source_audio_path="/tmp/a.wav",
                status="ingested",
                created_at=_now(),
            )
        )
        kit = await BrandKitsRepo(migrated_db).create(
            name="Aldo Kit",
            primary_color="#FF3366",
            accent_color="#FFD700",
        )
        speaker = await SpeakersRepo(migrated_db).create(
            display_name="Aldo", embedding=[1.0, 0.0], total_speech_s=120.0
        )
        await SpeakersRepo(migrated_db).set_preferred_brand_kit(speaker.id, kit.id)
        await VodSpeakersRepo(migrated_db).upsert(
            stream_id="str_rcand",
            speaker_label="SPEAKER_00",
            resolved_speaker_id=speaker.id,
            confidence=0.95,
            total_speech_s=100.0,
            embedding=[1.0, 0.0],
        )
        resolved = await resolve_brand_kit_for_candidate(
            migrated_db, stream_id="str_rcand", speaker_label="SPEAKER_00"
        )
    assert resolved is not None
    assert resolved.id == kit.id


# ---- per-kit custom_trigger_phrases merge ----


async def test_merged_phrases_appends_kit_extras(migrated_db: Database) -> None:
    """Spec §1.6 hard rule: per-kit phrases ADD to the tenant base list,
    not replace it. set() dedup is case-insensitive."""
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        kit = await BrandKitsRepo(migrated_db).create(
            name="K",
            primary_color="#000",
            accent_color="#FFF",
            custom_trigger_phrases=CustomTriggerPhrases(
                forward=["córtalo", "Clip This"],  # 'Clip This' dupes the base
                retroactive=["monchi eso"],
            ),
        )
        speaker = await SpeakersRepo(migrated_db).create(
            display_name="X", embedding=[1.0, 0.0], total_speech_s=60.0
        )
        await SpeakersRepo(migrated_db).set_preferred_brand_kit(speaker.id, kit.id)

        fwd, retro = await merged_trigger_phrases_for_speaker(
            migrated_db,
            speaker_id=speaker.id,
            base_forward=["clipea esto", "clip this"],
            base_retroactive=["clipeaste eso"],
        )
    assert "clipea esto" in fwd
    assert "córtalo" in fwd
    # Dedupe ignores case: 'Clip This' was a duplicate of 'clip this' in base.
    assert sum(1 for p in fwd if p.lower() == "clip this") == 1
    assert "clipeaste eso" in retro
    assert "monchi eso" in retro


async def test_merged_phrases_no_kit_returns_base_unchanged(
    migrated_db: Database,
) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        speaker = await SpeakersRepo(migrated_db).create(
            display_name="X", embedding=[1.0, 0.0], total_speech_s=60.0
        )
        fwd, retro = await merged_trigger_phrases_for_speaker(
            migrated_db,
            speaker_id=speaker.id,
            base_forward=["clipea esto"],
            base_retroactive=["clipeaste eso"],
        )
    assert fwd == ["clipea esto"]
    assert retro == ["clipeaste eso"]


async def test_merged_phrases_speaker_none_skips_lookup(
    migrated_db: Database,
) -> None:
    tenant_id = await _seed_tenant(migrated_db)
    with bound_tenant(tenant_id):
        fwd, retro = await merged_trigger_phrases_for_speaker(
            migrated_db,
            speaker_id=None,
            base_forward=["a"],
            base_retroactive=["b"],
        )
    assert fwd == ["a"]
    assert retro == ["b"]

"""Brand kit resolution at clip-render time.

Implements the spec's priority chain (§3.1):

    speaker.preferred_brand_kit_id  →  tenant.default_brand_kit  →  None

All resolution paths tolerate missing data — None-returning is a legal
outcome that the renderer interprets as "use system defaults" (no logo,
no custom captions, Pico styles).
"""

from __future__ import annotations

import structlog

from chalybclip.db import (
    BrandKitsRepo,
    Database,
    SpeakersRepo,
    TenantsRepo,
    VodSpeakersRepo,
)
from chalybclip.db.models import BrandKitRow, CustomTriggerPhrases

_log = structlog.get_logger(__name__)


async def outro_enabled_for_clip(
    db: Database, *, tenant_id: str, stream_id: str
) -> bool:
    """Whether the chalybclip end-card outro should be appended to this
    clip's export.

    Free tier ALWAYS gets it (returns True regardless of any stored
    flag). Paid tiers respect the resolved brand kit's
    `show_chalybclip_outro` (default True). Best-effort — any failure
    defaults to True so a lookup hiccup never silently strips the
    free-tier credit.
    """
    from chalybclip.tenancy import bound_tenant

    try:
        tenant = await TenantsRepo(db).get(tenant_id)
        tier = (tenant.tier if tenant else "free") or "free"
        if tier == "free":
            return True
        with bound_tenant(tenant_id):
            kit = await resolve_brand_kit_for_candidate(
                db, stream_id=stream_id, speaker_label=None
            )
        if kit is None:
            return True
        return bool(getattr(kit, "show_chalybclip_outro", True))
    except Exception:  # noqa: BLE001 — best-effort; default to on
        return True


async def resolve_brand_kit_for_speaker(
    db: Database,
    *,
    speaker_id: str | None,
) -> BrandKitRow | None:
    """Spec §3.1 priority: speaker.preferred_brand_kit_id → tenant default → None.

    Returns the resolved kit, or None when neither layer produced one.
    Safe to call with speaker_id=None (skips straight to tenant default).
    """
    if speaker_id is not None:
        speakers_repo = SpeakersRepo(db)
        speaker = await speakers_repo.get(speaker_id)
        if speaker is not None and speaker.preferred_brand_kit_id is not None:
            kit = await BrandKitsRepo(db).get(speaker.preferred_brand_kit_id)
            if kit is not None:
                return kit
    return await BrandKitsRepo(db).get_default()


async def resolve_brand_kit_for_candidate(
    db: Database,
    *,
    stream_id: str,
    speaker_label: str | None,
) -> BrandKitRow | None:
    """Resolve a brand kit for a candidate / clip given its within-VOD label.

    Two-stage lookup:
        1. vod_speakers row whose (stream_id, speaker_label) matches —
           gives us the persistent speaker_id.
        2. Pass that to `resolve_brand_kit_for_speaker`.

    When `speaker_label` is None (no diarization, or trigger that
    couldn't be attributed to a turn), fall straight to the tenant
    default kit.
    """
    if speaker_label is None:
        return await BrandKitsRepo(db).get_default()

    vod_speakers = await VodSpeakersRepo(db).list_for_stream(stream_id)
    for vs in vod_speakers:
        if vs.speaker_label == speaker_label:
            return await resolve_brand_kit_for_speaker(
                db, speaker_id=vs.resolved_speaker_id
            )
    # Speaker label exists on the candidate but no vod_speakers row —
    # happens if diarization persistence failed but in-memory attribution
    # succeeded. Fall back to tenant default rather than failing the render.
    return await BrandKitsRepo(db).get_default()


async def merged_trigger_phrases_for_speaker(
    db: Database,
    *,
    speaker_id: str | None,
    base_forward: list[str],
    base_retroactive: list[str],
) -> tuple[list[str], list[str]]:
    """Merge the tenant base trigger list with the speaker's kit additions.

    Spec §1.6 + §9 hard rule: per-kit phrases are ADDITIVE, not overrides.
    `set()` dedupes case-insensitively-equivalent entries (after a simple
    lowercase pass to catch 'Clip This' vs 'clip this'). Order: base
    phrases first (so existing config keeps precedence in any list-order-
    sensitive consumers), then any unique kit additions appended.
    """
    base_f_lower = {p.lower(): p for p in base_forward}
    base_r_lower = {p.lower(): p for p in base_retroactive}
    merged_f = list(base_forward)
    merged_r = list(base_retroactive)

    if speaker_id is None:
        return merged_f, merged_r

    kit = await resolve_brand_kit_for_speaker(db, speaker_id=speaker_id)
    if kit is None:
        return merged_f, merged_r

    extras: CustomTriggerPhrases = kit.custom_trigger_phrases
    for p in extras.forward:
        if p.lower() not in base_f_lower:
            merged_f.append(p)
            base_f_lower[p.lower()] = p
    for p in extras.retroactive:
        if p.lower() not in base_r_lower:
            merged_r.append(p)
            base_r_lower[p.lower()] = p

    if extras.forward or extras.retroactive:
        _log.debug(
            "branding.merged_trigger_phrases",
            speaker_id=speaker_id,
            kit_id=kit.id,
            added_forward=len(merged_f) - len(base_forward),
            added_retroactive=len(merged_r) - len(base_retroactive),
        )
    return merged_f, merged_r

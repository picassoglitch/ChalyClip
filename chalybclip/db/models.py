"""Pydantic models that mirror the DB tables 1:1.

These are distinct from the service-level models (e.g. `chalybclip.ingest.Stream`,
`chalybclip.detect.Candidate`) because the DB row carries persistence-only
fields (`created_at`, JSON-encoded blobs) that the service models
shouldn't have to know about.

Service models are converted to/from DB models inside the repo methods
(Task 1 wires that through). For Task 0 the DB models stand alone.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------- Tenancy ----------


class Tenant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    created_at: str
    # Phase 2 budget governor knobs. NULL == unlimited.
    daily_llm_budget_usd_micros: int | None = None
    daily_publish_limit: int | None = None
    rescore_concurrency_cap: int = 4
    # Retention windows (voice-markers spec slice E.1 / §9 locked defaults).
    # NULL means the tenant inherits the system default:
    #   - retention_vod_days        → 7
    #   - retention_clip_days       → 90
    #   - retention_transcript_days → 365
    # The sweeper resolves NULL → default at scan time; storing NULL keeps
    # "factory default" distinguishable from "tenant explicitly chose this
    # number" for ops debugging.
    retention_vod_days: int | None = None
    retention_clip_days: int | None = None
    retention_transcript_days: int | None = None
    # Slice O.1 — subscription tier. Drives watermark presence on
    # exported clips + (future J.2) token allowances and billing.
    # Existing tenants land on "free" via migration 013 default.
    tier: str = "free"
    # Slice NX.2 — Chalyb mutual exclusion. When PRO subscribers swap
    # their live engine, Chalyb POSTs /api/admin/tenants/{id}/status to
    # set this to 'paused' here. The pipeline + publish workers gate on
    # status='active' before doing real work; paused tenants surface a
    # clear "paused by Chalyb" message on the dashboard instead of jobs
    # silently failing.
    status: str = "active"
    # Slice NX.1 — Chalyb cross-system user id. The Chalyb platform's
    # Supabase auth.users.id for the human who owns this tenant. NULL for
    # CLI-created tenants (the historical case before the integration).
    # Used by:
    #   - balance.py / reporter.py:   bound into the outbound usage POST
    #   - service.py provisioning:    duplicate detection so POST /api/admin/tenants
    #                                 is idempotent on Chalyb retries
    # Without this field on the model + the SELECT below, the reporter
    # AttributeErrors on every LLM call (and the chip stays "— tokens").
    external_user_id: str | None = None
    # Slice NX.4 — cached Chalyb token balance. Updated by the outbound
    # usage reporter after each LLM call. Templates render these directly
    # in the nav chip without making any network call. NULL until the
    # tenant's first LLM call gets reported.
    cached_balance_remaining: int | None = None
    cached_balance_unlimited: int = 0   # 0/1 in SQLite; cast to bool when reading
    cached_balance_monthly_used: int | None = None
    cached_balance_at: str | None = None
    # Token T1 — outcome of the most recent outbound usage report, so the
    # chip / diag can show "balance sync failing" instead of a quietly
    # stale number. None until the first report attempt.
    last_usage_report_at: str | None = None
    last_usage_report_ok: int | None = None   # 1 ok / 0 fail / None never tried
    last_usage_report_error: str | None = None
    # DEPRECATED (dormant). upload-post integration (migration 022) —
    # retired in favor of Zernio. Column kept (sqlite column-drop is a
    # destructive table rebuild) but no longer written or read by app
    # code. Safe to drop in a future schema-rebuild migration.
    upload_post_profile_username: str | None = None
    # Zernio integration (migration 028/029). Each tenant maps to ONE
    # Zernio profile, CREATED via Zernio's POST /profiles (returns a
    # server `_id` like `prof_abc123`) from the publish dashboard.
    # Zernio scopes connected accounts + posts by `zernio_profile_id`.
    # `zernio_profile_name` is the operator-chosen display name (mig 029).
    # NULL = tenant has not created a profile yet.
    zernio_profile_id: str | None = None
    zernio_profile_name: str | None = None


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    email: str
    role: str = "owner"
    created_at: str


class ApiTokenRow(BaseModel):
    """Persisted shape — never carries the raw token, only the hash."""

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    hash: str
    scope: Literal["full", "read"] = "full"
    created_at: str
    last_used_at: str | None = None


# ---------- Personas + connected accounts ----------


class PersonaRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    name: str
    primary_language: str
    target_languages: list[str] = Field(default_factory=list)
    voice_prompt: str
    routing_tags: list[str] = Field(default_factory=list)
    created_at: str


class ConnectedAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    platform: str
    external_id: str
    display_name: str | None = None
    oauth_blob: dict[str, object] | None = None
    created_at: str
    # Phase 2: OAuth refresh + lifecycle tracking.
    refresh_token: str | None = None
    expires_at: str | None = None
    scopes: list[str] = Field(default_factory=list)
    status: str = "active"  # active | auth_failed | disabled
    # Migration 021 added native-OAuth columns to support an in-house
    # Connect flow that was later scrapped (switching to upload-post).
    # The columns are still on disk — leaving them as opaque optional
    # passthroughs so existing rows keep validating without forcing
    # a destructive down-migration on prod. Drop in a future cleanup
    # when nothing else reads them.
    access_token_encrypted: bytes | None = None
    refresh_token_encrypted: bytes | None = None
    token_type: str | None = None
    platform_user_id: str | None = None
    platform_username: str | None = None
    platform_avatar_url: str | None = None
    daily_publish_count: int = 0
    daily_publish_window_start: str | None = None


# ---------- Pipeline ----------


class StreamRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    vod_url: str
    platform: str
    title: str | None = None
    channel: str | None = None
    duration_s: float
    source_video_path: str
    source_audio_path: str
    status: str = "ingested"
    created_at: str
    # Phase L.1 — live RTMP ingest. is_live flips to True the moment
    # MediaMTX's runOnReady fires; live_started_at / live_ended_at
    # bracket the actual RTMP push window. status takes 'live' while
    # the push is active, 'live_ended' for the 24h retention window,
    # then 'ingested' once the operator runs the pipeline.
    is_live: bool = False
    live_started_at: str | None = None
    live_ended_at: str | None = None


class LiveStreamKeyRow(BaseModel):
    """Phase L.1 — per-tenant RTMP stream key. OBS uses key_value in
    its push URL: rtmp://live.chalybclip.chalyb.com/live/<key_value>.

    One active key per tenant at a time. Rotation creates a new row +
    sets revoked_at on the previous one. Never echo key_value to a
    response template that mixes admin + tenant scope — it's a secret."""

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    key_value: str
    created_at: str
    revoked_at: str | None = None
    last_used_at: str | None = None


class TranscriptRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_id: str
    tenant_id: str
    language: str
    duration_s: float
    model: str
    segments_json: str
    created_at: str


class CandidateRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    stream_id: str
    tenant_id: str
    ts: float
    score: float
    reason: str
    evidence: dict[str, object] = Field(default_factory=dict)
    created_at: str
    # Phase 2: vision-LLM rescore verdict (NULL until --vision-rescore runs).
    rescore_score: float | None = None
    rescore_reason: str | None = None
    rescore_model: str | None = None


class ClipRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    stream_id: str
    tenant_id: str
    candidate_id: str | None = None
    start_s: float
    end_s: float
    duration_s: float
    width: int
    height: int
    path: str
    smart_crop_box: dict[str, float] | None = None
    thumbnail_frame_path: str | None = None
    status: str = "cut"
    created_at: str
    # Per-clip overlay customization set in the clip editor screen.
    # Empty / None means "fall back to the brand-kit defaults end-to-end".
    # Schema is intentionally loose (dict[str, object]) — the renderer
    # picks the keys it knows; adding a new overlay knob doesn't need a
    # schema migration.
    overlay_config: dict[str, object] | None = None
    # Slice G.2 — cached publishability verdict. None = never scored.
    # `compute_publishability` runs in the editor + after save; the
    # cache lets the inbox + streams grid render a status chip without
    # recomputing per row.
    publishability_score: int | None = None
    publishability_status: str | None = None  # publish_ready | needs_edit | reject
    # Slice G.4b — when the operator auto-trims around integrity
    # issues, we shrink start_s/end_s/duration_s to the clean window
    # but stash the originals here so the "Revert trim" button can
    # restore them. NULL means the clip has never been trimmed.
    original_start_s: float | None = None
    original_end_s: float | None = None
    # Render Migration T1 — background-render state machine. Replaces
    # the legacy "render inline in the HTTP request" path that was
    # killing Railway requests on long clips. See
    # chalybclip/db/migrations/020_clip_render_state.sql for the column
    # rationale.
    render_state: str = "idle"        # idle | rendering | ready | failed
    render_progress_pct: int = 0      # 0-100, updated by capture_progress
    render_error: str | None = None   # short message on failure
    render_started_at: str | None = None  # ISO timestamp


class VariantRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    clip_id: str
    tenant_id: str
    persona_id: str
    language: str
    caption: str
    title_card_text: str = ""
    hashtags: list[str] = Field(default_factory=list)
    model: str | None = None
    created_at: str


class LLMCallRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    purpose: str
    provider: str
    model: str
    quality: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd_micros: int = 0
    status: str = "ok"
    error: str | None = None
    attempts: int = 1
    ts: str
    # Token T3 — the stream this cost belongs to (None for non-pipeline
    # calls). Stamped from the run's structlog stream_id contextvar.
    stream_id: str | None = None


class ProviderSpend(BaseModel):
    """Token T3 — one provider's roll-up within a stream's run."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    tokens: int = 0
    cost_usd_micros: int = 0
    calls: int = 0


class StreamSpend(BaseModel):
    """Token T3 — actual spend for one stream's run (vs the estimate).
    Sums llm_calls (Claude + transcription + any provider) tagged with
    this stream_id."""

    model_config = ConfigDict(extra="forbid")

    stream_id: str
    total_tokens: int = 0
    total_cost_usd_micros: int = 0
    total_calls: int = 0
    by_provider: list[ProviderSpend] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        """USD (not micros) for display."""
        return self.total_cost_usd_micros / 1_000_000.0


class PublishJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    clip_id: str
    variant_id: str
    account_id: str
    platform: str
    status: str = "pending"
    attempts: int = 0
    last_error: str | None = None
    scheduled_for: str | None = None
    external_id: str | None = None
    created_at: str
    # Phase 2: native-publisher metadata (TikTok/YT shareable URL, raw API response bits).
    external_url: str | None = None
    platform_metadata: dict[str, object] | None = None


class ZernioPublishRow(BaseModel):
    """One Zernio publish we fired (migration 030, status cols in 031).

    Local, tenant-scoped record of every POST /posts — Zernio's own
    GET /posts is company-key-wide, so per-tenant history MUST come
    from here. `post_id` is Zernio's post _id (the job handle the
    /job/{post_id} page polls). `status`/`platforms_json` are fed by
    the post.* webhooks (NULL until the first event lands)."""

    model_config = ConfigDict(extra="forbid")
    post_id: str
    tenant_id: str
    clip_id: str
    platforms: str  # csv of platform ids, e.g. "tiktok,instagram"
    content: str | None = None
    created_at: str
    status: str | None = None  # Zernio vocab: scheduled/published/failed/…
    # When the post is set to publish (mode=schedule). NULL for immediate
    # posts. Migration 048 — lets the Publicados list show the real publish
    # time instead of the batch creation time.
    scheduled_for: str | None = None
    platforms_json: str | None = None  # per-platform results, verbatim JSON
    updated_at: str | None = None
    options_json: str | None = None  # publish-extras snapshot (migration 033)


class HubPublishJobRow(BaseModel):
    """One internal-API publish job (migration 032).

    Created by /api/internal/v1/publish (ChalyOBS, Chalyb). The
    idempotency_key dedups caller retries per tenant; zernio_post_id
    links to the Zernio post once accepted, and the phase-2 webhook
    processor keeps status/platforms_json live from there."""

    model_config = ConfigDict(extra="forbid")
    job_id: str
    tenant_id: str
    idempotency_key: str | None = None
    source: str
    mode: str  # now | queue | schedule | draft
    targets: str  # csv of resolved platform ids
    video_url: str
    title: str | None = None
    caption: str | None = None
    scheduled_for: str | None = None
    zernio_post_id: str | None = None
    status: str = "pending"
    platforms_json: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str | None = None


class ZernioEventRow(BaseModel):
    """One inbound Zernio webhook event (migration 031).

    `event_id` is Zernio's stable payload.id — the at-least-once dedup
    key. `payload` is the raw JSON body verbatim (replay + later-phase
    backfills read it). `tenant_id` is NULL until (or unless) the
    background processor resolves the profileId/post id to a tenant."""

    model_config = ConfigDict(extra="forbid")
    event_id: str
    type: str
    payload: str
    profile_id: str | None = None
    tenant_id: str | None = None
    received_at: str
    processed: bool = False
    processed_at: str | None = None


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    type: str
    payload: dict[str, object] = Field(default_factory=dict)
    ts: str


class VisualSignalRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_id: str
    tenant_id: str
    ts_offset_s: float
    scene_cut: bool = False
    face_emotion: str | None = None
    motion_energy: float | None = None
    text_changed: bool = False


class WebhookSubscription(BaseModel):
    """Phase 2: outbound HMAC-signed event delivery to subscriber URLs."""

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    url: str
    types: list[str] = Field(default_factory=list)
    secret: str
    status: str = "active"  # active | disabled
    created_at: str
    last_dispatch_ts: str | None = None
    failure_count: int = 0


class WebhookSecretVersion(BaseModel):
    """Phase 3: a prior webhook secret kept alive for a rotation grace window.

    Subscribers may verify HMAC signatures against any unexpired secret in
    `webhook_secret_versions` for the subscription, in addition to the
    current secret on the `webhook_subscriptions` row.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    subscription_id: str
    tenant_id: str
    secret: str
    expires_at: str
    created_at: str


class PublishMetric(BaseModel):
    """Phase 3: one engagement-stats snapshot for a published clip.

    Each `(publish_job_id, fetched_at)` pair is one row; the dashboard's
    outcome card reads the latest row per job, and the calibration loop
    reads the time series. Some platforms won't expose every column - we
    leave the missing ones NULL rather than fabricating zeros.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    publish_job_id: str
    platform: str
    fetched_at: str
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    retention_pct: float | None = None  # 0.0-1.0
    ctr: float | None = None  # 0.0-1.0
    raw_metadata: dict[str, object] | None = None
    created_at: str


# ---------- Speakers (voice-markers spec slice B.2) ----------


class SpeakerRow(BaseModel):
    """Persistent voice identity for one tenant.

    Embedded vector is a 192-dim float list from pyannote/embedding,
    averaged across all VODs we've matched this identity in (weighted
    by total_speech_s). Stored as a JSON list in the embedding_json
    column for portability across SQLite / Postgres without a vector
    extension.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    display_name: str
    is_self: bool = False
    preferred_brand_kit_id: str | None = None
    embedding: list[float] | None = None
    embedding_dim: int | None = None
    total_speech_s: float = 0.0
    sample_audio_path: str | None = None
    created_at: str
    updated_at: str


class VodSpeakerRow(BaseModel):
    """Per-VOD resolution of one within-VOD speaker label to a persistent identity.

    `resolved_speaker_id=None` means the speaker is unknown — the user
    can label them via the dashboard's /speakers page (slice E).
    `confidence` is the cosine similarity to the matched speaker; below
    `DiarizationConfig.match_threshold` we leave resolved_speaker_id=None.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    stream_id: str
    tenant_id: str
    speaker_label: str
    resolved_speaker_id: str | None = None
    confidence: float | None = None
    total_speech_s: float = 0.0
    embedding: list[float] | None = None
    created_at: str


# ---------- Brand kits (voice-markers spec slice C.1) ----------


class CustomTriggerPhrases(BaseModel):
    """Per-kit additions to the tenant base trigger list.

    Spec section 9 hard rule: these are ADDITIVE (set() dedup at scan
    time), not overrides. Empty default = inherit tenant base unchanged.
    """

    model_config = ConfigDict(extra="forbid")
    forward: list[str] = Field(default_factory=list)
    retroactive: list[str] = Field(default_factory=list)


class BrandKitRow(BaseModel):
    """One visual identity (colors, fonts, logo, captions, auto-publish opt-in).

    Each tenant has zero or more kits; exactly one CAN be flagged
    `is_default=True` (enforced via a partial unique index in migration
    006). Kits are assigned per-speaker via `speakers.preferred_brand_kit_id`.

    Asset paths are storage-agnostic strings — local paths in dev,
    eventually S3/R2 keys in production via the Storage abstraction
    (see docs/production_deploy.md §3).
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    name: str
    is_default: bool = False

    # Colors
    primary_color: str
    accent_color: str
    text_color: str = "#FFFFFF"

    # Typography
    font_family: str = "Inter"
    font_weight: int = 800

    # Assets
    logo_url: str | None = None
    logo_dark_url: str | None = None
    watermark_url: str | None = None
    intro_sting_url: str | None = None
    outro_sting_url: str | None = None

    # Caption style — opaque JSON; the renderer reads keys it knows.
    caption_style: dict[str, object] | None = None

    # Layout
    default_layout: str = "pip"  # pip | split_stack | blurred_bg

    # Social handles
    handle_tiktok: str | None = None
    handle_youtube: str | None = None
    handle_instagram: str | None = None
    handle_kick: str | None = None

    # AI generation metadata
    ai_generated: bool = False
    ai_prompt: str | None = None
    ai_provider: str | None = None

    # Per-kit auto-publish (default OFF — review-first per spec §9)
    auto_publish_enabled: bool = False
    auto_publish_platforms: list[str] = Field(default_factory=list)
    auto_publish_delay_min: int = 60

    # Per-kit custom trigger phrases — additively merged with the
    # tenant base at scan time.
    custom_trigger_phrases: CustomTriggerPhrases = Field(
        default_factory=CustomTriggerPhrases
    )

    # Slice H.1 — user-level editor preferences. The clip editor's
    # right panel auto-saves to these so re-opening any clip prefills
    # the operator's last choices instead of resetting to "off" /
    # platform defaults / placeholder URL.
    default_platform: str | None = None
    banner_enabled_default: bool = False
    banner_show_context_default: bool = False
    banner_show_safezones_default: bool = False

    # Slice I.1 — Clip Style preset + Kick banner variant + top hook.
    # `clip_style` is the keystone editor abstraction (Repost Page Viral
    # / Clean Creator / Gaming Chaos / Documentary / Minimal Native).
    # Picking a clip style flips banner_variant + top_hook + caption
    # preset in one shot. None → falls back to "repost_page_viral".
    clip_style: str | None = None
    bottom_banner_style: str | None = None
    banner_live_badge_default: bool = False
    top_hook_enabled_default: bool = False
    top_hook_style_default: str | None = None

    # Slice K.5 — operator's default target platform. Picking one
    # (TikTok / Reels / Shorts / Kick / All) auto-configures every
    # downstream editor setting via PLATFORM_PRESETS.
    target_platform: str | None = None
    # Slice O.1 — pro-tier toggle for the bottom-right "chalybclip.com"
    # credit. Free tier ALWAYS gets the credit burned regardless of
    # this value; pro+ tiers respect it (default ON so newly-upgraded
    # tenants keep crediting until they explicitly opt out).
    show_chalybclip_credit: bool = True

    # Pro-tier toggle for the end-card "outro" splash appended to every
    # exported clip (the bundled chalybclip end card). Free tier ALWAYS
    # gets it regardless of this value; pro+ tiers respect it (default
    # ON, mirroring show_chalybclip_credit).
    show_chalybclip_outro: bool = True

    # Publishing safe trap (migration 042). When `safe_schedule_enabled`,
    # auto-publish schedules into the next compliant window and the drain
    # hard-gates blocked posts; otherwise the trap is advisory only.
    # `safety_policy` is a {platform: {field: value}} override map overlaid
    # on the built-in PLATFORM_DEFAULTS; None == pure defaults.
    safe_schedule_enabled: bool = False
    safety_policy: dict[str, object] | None = None
    content_timezone: str = "UTC"

    created_at: str
    updated_at: str


# ---------- Drive watches (voice-markers spec slice E.4) ----------


class DriveWatchRow(BaseModel):
    """One row in `drive_watches` — a folder ChalyClip polls for new VODs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tenant_id: str
    folder_id: str
    folder_name: str | None = None
    refresh_token: str
    access_token: str | None = None
    access_token_expires_at: str | None = None
    last_polled_at: str | None = None
    # Drive file IDs we've already ingested — the dedup key.
    seen_file_ids: list[str] = Field(default_factory=list)
    enabled: bool = True
    created_at: str
    updated_at: str


class ChannelWatchRow(BaseModel):
    """One row in `channel_watches` — a creator channel ChalyClip polls for
    new VODs (YouTube / Twitch / Kick). The auto-ingest counterpart to
    `DriveWatchRow`: instead of a Drive folder it watches a channel URL,
    and `seen_video_ids` is the per-video dedup key."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tenant_id: str
    platform: str  # youtube | twitch | kick
    channel_url: str
    channel_label: str | None = None
    persona_id: str
    language: str | None = None
    last_polled_at: str | None = None
    seen_video_ids: list[str] = Field(default_factory=list)
    # Per-video ingest failure counts: {video_id: attempts}. A video that
    # keeps failing (e.g. yt-dlp "This video is not available") is retried up
    # to a ceiling then parked into `seen_video_ids` so it stops failing every
    # poll. A successful ingest clears its entry.
    failed_video_attempts: dict[str, int] = Field(default_factory=dict)
    max_per_poll: int = 3
    enabled: bool = True
    # How many times a day to poll this channel. The loop only runs the
    # yt-dlp listing once ≥ 24h/polls_per_day has elapsed since the last
    # poll (1 = once a day, 4 = every 6h, 24 = hourly). Default 1.
    polls_per_day: int = 1
    created_at: str
    updated_at: str


class DriveExportSettingsRow(BaseModel):
    """One row in `drive_export_settings` — a tenant's clip → Drive
    EXPORT destination (task #31). The OUTPUT mirror of DriveWatchRow.

    `enabled` is the auto-save-on-render toggle. `is_connected`
    (refresh_token present) gates the per-clip manual export button
    independently of `enabled`."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    enabled: bool = False
    folder_id: str | None = None
    folder_name: str | None = None
    refresh_token: str | None = None
    access_token: str | None = None
    access_token_expires_at: str | None = None
    created_at: str
    updated_at: str

    @property
    def is_connected(self) -> bool:
        """True once the OAuth connect flow has stored a refresh token
        AND a destination folder is chosen — i.e. exports can actually
        run."""
        return bool(self.refresh_token and self.folder_id)

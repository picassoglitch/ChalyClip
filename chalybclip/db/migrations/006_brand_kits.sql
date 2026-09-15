-- Voice-markers spec slice C.1 — brand kits.
--
-- A brand kit is one streamer's visual identity: colors, fonts, logo,
-- caption style, social handles, and an optional set of custom trigger
-- phrases that extend the tenant-level base list at scan time. Kits are
-- per-tenant and can be assigned to specific speakers (so a multi-host
-- VOD renders each speaker's clips with their own kit).
--
-- Asset URLs are storage-agnostic: today they're local filesystem paths
-- under <output_dir>/brand_kits/<kit_id>/<asset>; in production they'll
-- be S3/R2 keys. The Storage abstraction (slice E / docs/production_deploy.md)
-- will swap the read path without touching this schema.
--
-- This migration also upgrades speakers.preferred_brand_kit_id from a
-- forward-ref TEXT column into a real FK, now that brand_kits exists.

CREATE TABLE brand_kits (
    id                        TEXT PRIMARY KEY,
    tenant_id                 TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name                      TEXT NOT NULL,
    is_default                INTEGER NOT NULL DEFAULT 0,             -- 0/1

    -- Colors (hex strings)
    primary_color             TEXT NOT NULL,
    accent_color              TEXT NOT NULL,
    text_color                TEXT NOT NULL DEFAULT '#FFFFFF',

    -- Typography
    font_family               TEXT NOT NULL DEFAULT 'Inter',
    font_weight               INTEGER NOT NULL DEFAULT 800,

    -- Asset paths/keys (local paths in dev, S3 keys in prod)
    logo_url                  TEXT,
    logo_dark_url             TEXT,
    watermark_url             TEXT,
    intro_sting_url           TEXT,
    outro_sting_url           TEXT,

    -- Caption style (JSON blob — see voice_markers_spec.md §3.4)
    caption_style_json        TEXT,

    -- Layout preferences
    default_layout            TEXT NOT NULL DEFAULT 'pip',           -- pip | split_stack | blurred_bg

    -- Social handles (rendered into watermark text)
    handle_tiktok             TEXT,
    handle_youtube            TEXT,
    handle_instagram          TEXT,
    handle_kick               TEXT,

    -- AI generation metadata (filled when assets come from slice D's logo gen)
    ai_generated              INTEGER NOT NULL DEFAULT 0,
    ai_prompt                 TEXT,
    ai_provider               TEXT,

    -- Per-kit auto-publish opt-in. Default OFF — review-first is the
    -- spec's locked decision (voice_markers_spec.md §9). Operator flips
    -- this per-kit to enable the auto-publish worker (slice E).
    auto_publish_enabled      INTEGER NOT NULL DEFAULT 0,
    auto_publish_platforms_json TEXT,                                  -- JSON list, e.g. ["tiktok","shorts"]
    auto_publish_delay_min    INTEGER NOT NULL DEFAULT 60,

    -- Custom trigger phrases — additively merged with tenant base at
    -- scan time. JSON: {"forward": [...], "retroactive": [...]}
    custom_trigger_phrases_json TEXT,

    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL
);
CREATE INDEX idx_brand_kits_tenant ON brand_kits (tenant_id);
-- Only ONE kit per tenant is allowed to be the default. Enforced at the
-- index level via a partial unique constraint (SQLite ≥ 3.8).
CREATE UNIQUE INDEX idx_brand_kits_one_default_per_tenant
    ON brand_kits (tenant_id)
    WHERE is_default = 1;

-- Upgrade speakers.preferred_brand_kit_id from a forward-ref TEXT into
-- a real FK. SQLite doesn't support ALTER TABLE ADD CONSTRAINT, so we
-- rebuild the table — same data, new FK. (preserves existing rows;
-- vod_speakers FK to speakers stays intact via ON DELETE CASCADE.)
CREATE TABLE speakers_new (
    id                        TEXT PRIMARY KEY,
    tenant_id                 TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    display_name              TEXT NOT NULL,
    is_self                   INTEGER NOT NULL DEFAULT 0,
    preferred_brand_kit_id    TEXT REFERENCES brand_kits(id) ON DELETE SET NULL,
    embedding_json            TEXT,
    embedding_dim             INTEGER,
    total_speech_s            REAL NOT NULL DEFAULT 0.0,
    sample_audio_path         TEXT,
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL
);
INSERT INTO speakers_new SELECT * FROM speakers;
DROP TABLE speakers;
ALTER TABLE speakers_new RENAME TO speakers;
CREATE INDEX idx_speakers_tenant ON speakers (tenant_id);

-- Phase 1 schema, lock-down version. Any change is a numbered migration.
--
-- Conventions:
--   * IDs are TEXT (ULID with kind prefix: ten_, usr_, tok_, str_, clp_, var_, ...)
--   * Timestamps are TEXT (ISO 8601, UTC) for portability + readability
--   * Booleans are INTEGER 0/1 (SQLite has no native bool)
--   * JSON columns end in _json and store stringified JSON
--   * Every domain table carries tenant_id with a (tenant_id, ...) composite index
--     so cross-tenant scans are never the cheap path
--
-- The `schema_version` table is created and bumped by the migration runner
-- (`chalybclip/db/migrations.py`) — not declared here.

-- ---------- Tenants + identity ----------

CREATE TABLE tenants (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE users (
    id         TEXT PRIMARY KEY,
    tenant_id  TEXT NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
    email      TEXT NOT NULL,
    role       TEXT NOT NULL DEFAULT 'owner',
    created_at TEXT NOT NULL,
    UNIQUE (tenant_id, email)
);
CREATE INDEX idx_users_tenant ON users (tenant_id);

CREATE TABLE api_tokens (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
    hash         TEXT NOT NULL UNIQUE,
    scope        TEXT NOT NULL DEFAULT 'full',
    created_at   TEXT NOT NULL,
    last_used_at TEXT
);
CREATE INDEX idx_api_tokens_tenant ON api_tokens (tenant_id);

-- ---------- Personas + connected accounts ----------

CREATE TABLE personas (
    id                    TEXT PRIMARY KEY,
    tenant_id             TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name                  TEXT NOT NULL,
    primary_language      TEXT NOT NULL,
    target_languages_json TEXT NOT NULL DEFAULT '[]',
    voice_prompt          TEXT NOT NULL,
    routing_tags_json     TEXT NOT NULL DEFAULT '[]',
    created_at            TEXT NOT NULL
);
CREATE INDEX idx_personas_tenant ON personas (tenant_id);

CREATE TABLE connected_accounts (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    platform        TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    display_name    TEXT,
    oauth_blob_json TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_connected_accounts_tenant ON connected_accounts (tenant_id);

-- ---------- Pipeline domain ----------

CREATE TABLE streams (
    id                 TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    vod_url            TEXT NOT NULL,
    platform           TEXT NOT NULL,
    title              TEXT,
    channel            TEXT,
    duration_s         REAL NOT NULL,
    source_video_path  TEXT NOT NULL,
    source_audio_path  TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'ingested',
    created_at         TEXT NOT NULL
);
CREATE INDEX idx_streams_tenant_created ON streams (tenant_id, created_at);

CREATE TABLE transcripts (
    stream_id     TEXT PRIMARY KEY REFERENCES streams(id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    language      TEXT NOT NULL,
    duration_s    REAL NOT NULL,
    model         TEXT NOT NULL,
    segments_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX idx_transcripts_tenant ON transcripts (tenant_id);

CREATE TABLE candidates (
    id            TEXT PRIMARY KEY,
    stream_id     TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    ts            REAL NOT NULL,
    score         REAL NOT NULL,
    reason        TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL
);
CREATE INDEX idx_candidates_tenant_stream ON candidates (tenant_id, stream_id, ts);

CREATE TABLE clips (
    id                   TEXT PRIMARY KEY,
    stream_id            TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    tenant_id            TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    candidate_id         TEXT REFERENCES candidates(id) ON DELETE SET NULL,
    start_s              REAL NOT NULL,
    end_s                REAL NOT NULL,
    duration_s           REAL NOT NULL,
    width                INTEGER NOT NULL,
    height               INTEGER NOT NULL,
    path                 TEXT NOT NULL,
    smart_crop_box_json  TEXT,
    thumbnail_frame_path TEXT,
    status               TEXT NOT NULL DEFAULT 'cut',
    created_at           TEXT NOT NULL
);
CREATE INDEX idx_clips_tenant_created ON clips (tenant_id, created_at);
CREATE INDEX idx_clips_stream ON clips (stream_id);

CREATE TABLE variants (
    id               TEXT PRIMARY KEY,
    clip_id          TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    tenant_id        TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    persona_id       TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    language         TEXT NOT NULL,
    caption          TEXT NOT NULL,
    title_card_text  TEXT NOT NULL DEFAULT '',
    hashtags_json    TEXT NOT NULL DEFAULT '[]',
    model            TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX idx_variants_tenant_created ON variants (tenant_id, created_at);
CREATE INDEX idx_variants_clip ON variants (clip_id);

-- LLM call audit log. Every router call writes one row regardless of outcome.
CREATE TABLE llm_calls (
    id               TEXT PRIMARY KEY,
    tenant_id        TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    purpose          TEXT NOT NULL,
    provider         TEXT NOT NULL,
    model            TEXT NOT NULL,
    quality          TEXT NOT NULL,
    input_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0,
    cost_usd_micros  INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL DEFAULT 'ok',
    error            TEXT,
    attempts         INTEGER NOT NULL DEFAULT 1,
    ts               TEXT NOT NULL
);
CREATE INDEX idx_llm_calls_tenant_ts ON llm_calls (tenant_id, ts);

CREATE TABLE publish_jobs (
    id             TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    clip_id        TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    variant_id     TEXT NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
    account_id     TEXT NOT NULL REFERENCES connected_accounts(id) ON DELETE RESTRICT,
    platform       TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    attempts       INTEGER NOT NULL DEFAULT 0,
    last_error     TEXT,
    scheduled_for  TEXT,
    external_id    TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_publish_jobs_tenant_status ON publish_jobs (tenant_id, status, scheduled_for);

CREATE TABLE events (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    ts           TEXT NOT NULL
);
CREATE INDEX idx_events_tenant_ts ON events (tenant_id, ts);
CREATE INDEX idx_events_tenant_type ON events (tenant_id, type, ts);

-- Vision pipeline writes here in Task 5; the table exists from day one.
CREATE TABLE visual_signals (
    stream_id     TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    ts_offset_s   REAL NOT NULL,
    scene_cut     INTEGER NOT NULL DEFAULT 0,
    face_emotion  TEXT,
    motion_energy REAL,
    text_changed  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (stream_id, ts_offset_s)
);
CREATE INDEX idx_visual_signals_tenant ON visual_signals (tenant_id, stream_id, ts_offset_s);

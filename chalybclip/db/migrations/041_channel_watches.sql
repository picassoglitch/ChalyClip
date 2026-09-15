-- Channel auto-ingest — watch a creator's channel for new VODs.
--
-- One row per (tenant, channel_url). The channel-poll loop lists the
-- channel's recent uploads via yt-dlp (flat, no download), and for each
-- video whose id is NOT in `seen_video_ids_json` it ingests the VOD and
-- kicks off the full pipeline (transcribe -> detect -> clip -> variants ->
-- auto-publish, the last gated by the safe trap).
--
-- Mirrors `drive_watches` (migration 008): `seen_video_ids_json` is the
-- exact dedup key, `last_polled_at` advances only on a clean pass, and
-- `enabled = 0` pauses a watch without losing its seen-set.
--
--   platform        — youtube | twitch | kick (yt-dlp handles all three)
--   channel_url      — the channel / @handle / videos URL to poll
--   persona_id       — required: the pipeline needs a persona for variants
--   max_per_poll     — cap the first poll so a big back-catalog doesn't
--                      flood the pipeline; later polls only see new uploads

CREATE TABLE channel_watches (
    id                  TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    platform            TEXT NOT NULL,
    channel_url         TEXT NOT NULL,
    channel_label       TEXT,
    persona_id          TEXT NOT NULL,
    language            TEXT,
    last_polled_at      TEXT,
    seen_video_ids_json TEXT NOT NULL DEFAULT '[]',
    max_per_poll        INTEGER NOT NULL DEFAULT 3,
    enabled             INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    UNIQUE (tenant_id, channel_url)
);

CREATE INDEX idx_channel_watches_tenant_enabled
    ON channel_watches (tenant_id, enabled);

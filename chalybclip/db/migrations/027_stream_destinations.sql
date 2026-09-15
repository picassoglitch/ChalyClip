-- Multistream M1 — per-tenant restream destinations.
--
-- The live-ingest relay (chalybclip-live / MediaMTX) fans one ingest out to
-- each of the streamer's platforms (Twitch / YouTube / Kick / custom RTMP)
-- with `ffmpeg -c copy`. This table holds where to push + the (encrypted)
-- stream key per destination.
--
-- `stream_key_enc` is Fernet-encrypted at rest (chalybclip.secrets). The key
-- is WRITE-ONLY in the UI — never returned to the browser, never logged
-- (CLAUDE rule #10). It's decrypted only server-side, only on the
-- internal-bearer relay-fetch path.
--
-- The full RTMP push target is `ingest_url + <decrypted key>`. For known
-- platforms `ingest_url` is templated (Twitch/YouTube); Kick/custom carry
-- the user-supplied URL (Kick is per-user IVS).

CREATE TABLE IF NOT EXISTS stream_destinations (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    platform        TEXT NOT NULL,          -- twitch | youtube | kick | custom
    label           TEXT,
    ingest_url      TEXT NOT NULL,          -- base RTMP url up to (not incl) the key
    stream_key_enc  TEXT NOT NULL,          -- Fernet-encrypted stream key
    enabled         INTEGER NOT NULL DEFAULT 1,
    last_status     TEXT,                   -- null | connecting | live | failed | stopped
    last_status_at  TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_stream_destinations_tenant
    ON stream_destinations (tenant_id);

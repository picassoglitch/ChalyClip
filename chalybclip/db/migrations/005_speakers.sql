-- Voice-markers spec slice B.2 — speaker identities + per-VOD resolution.
--
-- `speakers` holds the tenant's persistent voice identities. Each row is one
-- person (the tenant themselves, a co-host, a recurring guest, ...) with a
-- pyannote embedding vector that we cosine-similarity-match new diarizations
-- against. Display name + preferred brand kit live here too so the renderer
-- can pick the right caption style + handle per speaker.
--
-- `vod_speakers` is the per-VOD resolution table — one row per
-- (stream_id, SPEAKER_NN label) pair. `resolved_speaker_id` points at the
-- persistent `speakers` row when the embedding match cleared the threshold;
-- it stays NULL when the speaker is unknown / pending user labeling.
--
-- The brand_kits table arrives in slice C; for now `speakers.preferred_brand_kit_id`
-- is nullable text so the FK landing in 006 can reference it without
-- breaking forward-compatibility.

CREATE TABLE speakers (
    id                        TEXT PRIMARY KEY,
    tenant_id                 TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    display_name              TEXT NOT NULL,
    is_self                   INTEGER NOT NULL DEFAULT 0,         -- 0/1 — true for the tenant's own voice
    preferred_brand_kit_id    TEXT,                                -- FK added in slice C
    embedding_json            TEXT,                                -- JSON list[float], ~192 dims for pyannote-3.1
    embedding_dim             INTEGER,                             -- length of the embedding vector
    total_speech_s            REAL NOT NULL DEFAULT 0.0,           -- cumulative speech across all matched VODs
    sample_audio_path         TEXT,                                -- short WAV clip on disk for UI playback
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL
);
CREATE INDEX idx_speakers_tenant ON speakers (tenant_id);

-- One row per (stream, within-VOD speaker label) — populated after each
-- diarization run. `resolved_speaker_id` is NULL for unknown speakers
-- pending operator labeling; once labeled, the labeling step copies the
-- chosen speaker's id here and (if the new speaker is auto-created)
-- folds the embedding into speakers.embedding_json.
CREATE TABLE vod_speakers (
    id                    TEXT PRIMARY KEY,
    stream_id             TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    tenant_id             TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    speaker_label         TEXT NOT NULL,                  -- 'SPEAKER_00' etc. from pyannote
    resolved_speaker_id   TEXT REFERENCES speakers(id) ON DELETE SET NULL,
    confidence            REAL,                            -- cosine sim to the matched speaker, [0,1]
    total_speech_s        REAL NOT NULL DEFAULT 0.0,
    embedding_json        TEXT,                            -- this-VOD-only embedding; used for later merges
    created_at            TEXT NOT NULL,
    UNIQUE (stream_id, speaker_label)
);
CREATE INDEX idx_vod_speakers_stream ON vod_speakers (stream_id);
CREATE INDEX idx_vod_speakers_tenant ON vod_speakers (tenant_id);
CREATE INDEX idx_vod_speakers_speaker ON vod_speakers (tenant_id, resolved_speaker_id);

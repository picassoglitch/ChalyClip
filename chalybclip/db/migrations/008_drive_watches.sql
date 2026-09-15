-- Voice-markers spec slice E.4 — Google Drive folder watches.
--
-- One row per (tenant, folder). Used by `chalybclip drive poll` and the
-- in-process scheduler to find new VODs landing in a streamer's Drive
-- inbox folder and ingest them automatically.
--
-- Spec §1.2.a flow:
--   1. User connects Drive via OAuth (refresh_token persisted here)
--   2. User picks a folder; we store its Drive file_id + display name
--   3. Polling cron lists files modified since `last_polled_at` and
--      whose Drive id is NOT in `seen_file_ids_json` -> ingest each
--   4. After ingest enqueue, append the file id to seen_file_ids_json
--      and update last_polled_at + updated_at.
--
-- `enabled = 0` lets the operator pause a watch without deleting it,
-- which preserves seen_file_ids_json so re-enabling doesn't reingest.
--
-- The schema is intentionally storage-agnostic. The same table works
-- when we add Dropbox / S3 source buckets later — only the
-- DriveClient protocol implementation changes.

CREATE TABLE drive_watches (
    id                       TEXT PRIMARY KEY,
    tenant_id                TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    folder_id                TEXT NOT NULL,
    folder_name              TEXT,
    refresh_token            TEXT NOT NULL,
    access_token             TEXT,
    access_token_expires_at  TEXT,
    last_polled_at           TEXT,
    seen_file_ids_json       TEXT NOT NULL DEFAULT '[]',
    enabled                  INTEGER NOT NULL DEFAULT 1,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    UNIQUE (tenant_id, folder_id)
);

CREATE INDEX idx_drive_watches_tenant_enabled
    ON drive_watches (tenant_id, enabled);

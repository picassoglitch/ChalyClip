-- Task #31 — per-tenant clip → Drive EXPORT destination.
--
-- The mirror of drive_watches (008): that table is the INPUT side (watch
-- a folder, ingest VODs). This is the OUTPUT side — where a mid-tier
-- (pro / all_access) tenant's rendered clips get saved.
--
-- One row per tenant (tenant_id is the PK): a tenant has a single Drive
-- export destination. `enabled` is the auto-save-on-render toggle; the
-- per-clip manual "Export to Drive" button works whenever the tenant is
-- CONNECTED (refresh_token present) regardless of `enabled`.
--
-- refresh_token / access_token are populated by the OAuth connect flow
-- (deferred follow-up — needs the Google Cloud OAuth app). Until then a
-- row may exist with enabled=0 and NULL tokens (not yet connected).
--
-- Storage-agnostic like drive_watches: the same row works for Dropbox /
-- S3 destinations later — only the DriveUploadClient impl changes.

CREATE TABLE drive_export_settings (
    tenant_id                TEXT PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    -- Auto-save every rendered clip to Drive. Off by default — opt-in.
    enabled                  INTEGER NOT NULL DEFAULT 0,
    -- Destination folder the clips land in (Drive file_id + display name).
    folder_id                TEXT,
    folder_name              TEXT,
    -- Per-tenant OAuth, populated by the connect flow. NULL = not yet
    -- connected (the export service raises "not connected" until set).
    refresh_token            TEXT,
    access_token             TEXT,
    access_token_expires_at  TEXT,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL
);

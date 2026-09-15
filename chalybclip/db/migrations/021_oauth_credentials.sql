-- Wave 1 of the per-tenant OAuth connect flow (Pattern A).
--
-- The previous shape stored access tokens as plaintext JSON inside
-- `oauth_blob_json`. That worked for Phase 1's "paste a Buffer token"
-- flow but is unsafe for the native TikTok / IG / YT connect flows
-- where we'll be sitting on user-level refresh tokens long-term.
--
-- New columns:
--
--   access_token_encrypted   Fernet-encrypted access token. NULL for
--                            rows that pre-date this migration (read
--                            via the oauth_blob_json fallback until
--                            backfilled).
--
--   refresh_token_encrypted  Fernet-encrypted refresh token. NULL for
--                            providers that don't ship a classic
--                            refresh token (Meta: only long-lived
--                            access token, refreshed in place).
--
--   token_type               'bearer' (TikTok, Google) — classic
--                            access+refresh model.
--                            'long_lived' (Meta) — single long-lived
--                            token, refresh by re-exchanging the
--                            token itself before expires_at.
--                            Drives which refresh strategy the
--                            scheduler picks per row.
--
--   platform_user_id         TikTok open_id, IG-Business id, YouTube
--                            channel id. Auto-populated from the
--                            provider's user-info call at connect
--                            time. Different from `external_id`
--                            which historically was operator-typed.
--
--   platform_username        @handle as returned by the provider.
--                            Read-only on the dashboard; never typed
--                            by the operator.
--
--   platform_avatar_url      For the read-only post-connect display.
--
--   daily_publish_count      TikTok's 25 videos/day per account cap.
--   daily_publish_window_start ISO date the counter started in. Reset
--                            when the date rolls.
--
-- Indexes added: tenant_id + platform + platform_user_id so a re-
-- connect of the same TikTok account by the same tenant overwrites
-- the existing row instead of creating a duplicate.

ALTER TABLE connected_accounts ADD COLUMN access_token_encrypted   BLOB;
ALTER TABLE connected_accounts ADD COLUMN refresh_token_encrypted  BLOB;
ALTER TABLE connected_accounts ADD COLUMN token_type               TEXT;
ALTER TABLE connected_accounts ADD COLUMN platform_user_id         TEXT;
ALTER TABLE connected_accounts ADD COLUMN platform_username        TEXT;
ALTER TABLE connected_accounts ADD COLUMN platform_avatar_url      TEXT;
ALTER TABLE connected_accounts ADD COLUMN daily_publish_count      INTEGER NOT NULL DEFAULT 0;
ALTER TABLE connected_accounts ADD COLUMN daily_publish_window_start TEXT;

-- Re-connect dedup. NULL platform_user_id (Buffer / pre-migration
-- rows) doesn't participate; the index treats NULLs as distinct.
CREATE INDEX idx_connected_accounts_platform_user
    ON connected_accounts (tenant_id, platform, platform_user_id);

-- Community notifications: settings + notify ledger (Hub phase 11).
--
-- Discord/Telegram are the streamer's community channels (not clip
-- targets). When "Avisar a mi comunidad" is on, post.published fires a
-- rich embed / text announcement to the configured channel.

-- Per-tenant settings. discord/telegram account ids are the Zernio
-- social-account ids of the connected community channels.
CREATE TABLE IF NOT EXISTS zernio_community_settings (
    tenant_id           TEXT PRIMARY KEY,
    enabled             INTEGER NOT NULL DEFAULT 0,  -- "Avisar a mi comunidad"
    discord_account_id  TEXT,
    telegram_account_id TEXT,
    brand_name          TEXT,                        -- webhook display name
    brand_avatar_url    TEXT,
    weekly_digest       INTEGER NOT NULL DEFAULT 0,  -- default OFF
    updated_at          TEXT NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

-- Notify ledger: one row per ORIGINAL published post we announced.
-- Two jobs:
--   1. idempotency — webhooks are at-least-once, so a re-delivered
--      post.published must not re-announce (claim is once-only).
--   2. loop guard — the announcement is ITSELF a post to Discord/
--      Telegram; its post.published must not trigger another announce.
--      We record the notification post id and skip it.
CREATE TABLE IF NOT EXISTS zernio_community_notifications (
    source_post_id       TEXT PRIMARY KEY,   -- the clip post we announced
    tenant_id            TEXT NOT NULL,
    notification_post_id TEXT,               -- the Discord/Telegram post we created
    sent_at              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_community_notif_post
    ON zernio_community_notifications (notification_post_id);

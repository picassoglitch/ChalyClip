-- External (natively-authored) posts for the unified calendar (Hub
-- phase 8).
--
-- The post.external.* webhooks report posts a streamer authored
-- directly on a platform (not through the hub). Unlike hub publishes
-- and Zernio-scheduled posts — which we already track per-tenant in
-- zernio_publishes — the external payload carries the Zernio
-- social-account id but NO profileId, so we CAN'T resolve the tenant
-- at write time. We key these by (account_id, platform-native post_id)
-- and resolve to a tenant at READ time: the calendar route matches
-- account_id against the viewing tenant's own connected accounts. That
-- match IS the isolation boundary (a tenant only ever sees posts for
-- accounts they own on Zernio).
--
-- Idempotent: Zernio's first sync delivers every existing native post
-- as post.external.created, so created == updated == upsert. A
-- post.external.deleted flips status to 'deleted' (the calendar greys
-- it out / hides it) rather than dropping the row.

CREATE TABLE IF NOT EXISTS zernio_calendar (
    account_id    TEXT NOT NULL,    -- Zernio social account id (the tenant link)
    post_id       TEXT NOT NULL,    -- platform-native post id
    platform      TEXT,
    content       TEXT,
    url           TEXT,
    thumbnail_url TEXT,
    media_type    TEXT,
    published_at  TEXT,             -- ISO-8601; the calendar date
    status        TEXT NOT NULL DEFAULT 'active',  -- active | deleted
    deleted_at    TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (account_id, post_id)
);

CREATE INDEX IF NOT EXISTS idx_calendar_account_date
    ON zernio_calendar (account_id, published_at DESC);

-- Global, operator-editable key/value store (slice O.57).
--
-- The landing page's "what does an editor cost" comparison shows a
-- ChalybClip price. Hard-coding it in i18n.py meant every price tweak was
-- a code change + redeploy. This table lets the operator set it from the
-- admin UI (/dashboard/settings/site) and the public landing reads it at
-- render time.
--
-- Deliberately NOT tenant-scoped — these are platform-wide values, the
-- same exception TenantsRepo is for tenant rows. Keys are free-form
-- strings; values are stored as TEXT (the caller owns any parsing). The
-- table is generic so future site-wide knobs (announcement banner, demo
-- video URL, etc.) land here without another migration.
--
-- Current keys:
--   landing_price_es  -- price string shown on the ES landing (e.g. "US$ 29 / mes")
--   landing_price_en  -- price string shown on the EN landing (e.g. "$29 / month")
-- Either may be absent → the landing falls back to the i18n placeholder.

CREATE TABLE IF NOT EXISTS platform_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

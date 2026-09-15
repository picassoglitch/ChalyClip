-- Token T1 — make usage-report failures visible.
--
-- The outbound usage reporter (integrations/chalyb/reporter.py) is
-- fire-and-forget and swallows every error so a network hiccup never
-- breaks a successful LLM response. The downside: when reporting is
-- BROKEN (wrong CHALYB_ADMIN_TOKEN, wrong CHALYB_BASE_URL, Chalyb
-- rejecting the POST), Chalyb never deducts and the cached balance
-- silently goes stale — the chip shows the same number forever and
-- nobody can tell. Operators were flying blind.
--
-- These columns persist the OUTCOME of the most recent report attempt
-- so the balance chip + the /diag page can surface "balance sync is
-- failing" instead of a quietly-wrong number.
--   last_usage_report_at    — ISO timestamp of the last attempt
--   last_usage_report_ok    — 1 success / 0 failure
--   last_usage_report_error — short reason on failure (NULL on success)

ALTER TABLE tenants ADD COLUMN last_usage_report_at    TEXT;
ALTER TABLE tenants ADD COLUMN last_usage_report_ok    INTEGER;
ALTER TABLE tenants ADD COLUMN last_usage_report_error TEXT;

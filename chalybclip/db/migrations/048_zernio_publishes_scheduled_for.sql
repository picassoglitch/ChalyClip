-- 048_zernio_publishes_scheduled_for.sql
--
-- Record WHEN a scheduled post will publish, not just when it was created.
-- Before this, the Publicados list could only show created_at, so a whole
-- auto-program batch shared one timestamp (the moment of the click) — which
-- read as "many posts at the same time". NULL for immediate posts.
ALTER TABLE zernio_publishes ADD COLUMN scheduled_for TEXT;

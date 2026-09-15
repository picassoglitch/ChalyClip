-- Drop the now-dead `content_strategy` column from autopublish_settings.
--
-- It only ever drove the flat interval-drip cadence (unique = 30 min,
-- variations = 60 min). All publishing now schedules through the per-platform
-- rulebook (chalybclip.publish.pacing), which spaces posts by each platform's
-- min_gap + daily cap — the drip is gone, so the setting does nothing.
--
-- SQLite (>= 3.35) and Postgres both support ALTER TABLE ... DROP COLUMN; the
-- column has no index/constraint, so the drop is safe on both backends.
ALTER TABLE autopublish_settings DROP COLUMN content_strategy;

-- Auto-publish content strategy → drives the queue drip cadence.
--
-- When auto-publish schedules clips (post_mode='queue'), it now drips one
-- post at a fixed interval instead of best-time slots:
--   unique      -- each clip is a distinct video  → one post every 30 min
--   variations  -- same video reposted with edits → one post every 60 min,
--                  spacing spans platforms so the near-duplicates don't bunch
ALTER TABLE autopublish_settings
    ADD COLUMN content_strategy TEXT NOT NULL DEFAULT 'unique';

-- Per-channel poll cadence (resource-saving schedule).
--
-- Replaces the fixed every-15-min poll with a per-watch "N times a day"
-- schedule. The channel-poll loop still ticks often (cheap: a DB read +
-- a clock check), but the expensive yt-dlp listing for a watch only runs
-- when it's DUE — i.e. when at least 24h/polls_per_day has elapsed since
-- its last poll. polls_per_day=1 → once a day; 4 → every 6h; 24 → hourly.
-- Default 1 (max resource saving) for existing + new watches.

ALTER TABLE channel_watches ADD COLUMN polls_per_day INTEGER NOT NULL DEFAULT 1;

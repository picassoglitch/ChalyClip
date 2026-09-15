-- Per-video ingest failure counts so a permanently-unavailable video gets
-- retried a few times then PARKED, instead of failing on every poll forever.
--
-- Background: the channel poller intentionally leaves a failed video OUT of
-- seen_video_ids_json so a TRANSIENT error (YouTube bot-check, flaky network,
-- a not-yet-published premiere) retries on the next poll. But a genuinely
-- dead video (private / deleted / region-locked — yt-dlp "This video is not
-- available") then fails on EVERY poll indefinitely: every "scan now" reports
-- videos_ingested=0 / videos_failed=1 and looks broken.
--
-- This column records {video_id: attempt_count}. Once a video hits the retry
-- ceiling the poller adds it to seen_video_ids_json (parks it) and drops it
-- from here; a later successful ingest also clears its entry.
--
-- JSON object, default '{}'. Portable SQL (SQLite + Postgres).

ALTER TABLE channel_watches ADD COLUMN failed_video_attempts_json TEXT NOT NULL DEFAULT '{}';

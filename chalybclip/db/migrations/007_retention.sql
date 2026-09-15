-- Voice-markers spec slice E.1 — per-tenant retention windows.
--
-- Spec §9 locks the defaults: 30 days for raw VODs, 90 days for rendered
-- clips, 365 days for transcripts. All three are tenant-configurable so
-- compliance-sensitive tenants (legal, healthcare, etc.) can shorten;
-- archival tenants can extend.
--
-- The sweeper (chalybclip/retention/service.py) hard-deletes both DB rows
-- and on-disk artifacts past their windows. No soft-delete — once it's
-- gone, it's gone. Run it as a daily cron via `chalybclip retention sweep`.
--
-- NULL means "tenant inherits the system default". Stored as ints (days)
-- rather than seconds because the resolution we care about is daily and
-- the JSON-round-trip through the dashboard form is friendlier.

ALTER TABLE tenants ADD COLUMN retention_vod_days INTEGER;
ALTER TABLE tenants ADD COLUMN retention_clip_days INTEGER;
ALTER TABLE tenants ADD COLUMN retention_transcript_days INTEGER;

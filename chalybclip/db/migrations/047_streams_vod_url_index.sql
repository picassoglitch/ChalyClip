-- 047_streams_vod_url_index.sql
--
-- Channel auto-ingest dedup. The poll loop can re-detect a VOD it has
-- already ingested — e.g. its seen-set write was lost to a "database is
-- locked" during a load spike, or a prior pipeline step failed and the
-- poller retried the whole callback. Without a (tenant_id, vod_url)
-- lookup, ingest_vod mints a fresh ULID on every call and re-downloads
-- the entire VOD into a duplicate stream row. StreamsRepo.find_by_vod_url
-- uses this index to reuse the already-ingested stream's id instead, so a
-- re-ingest is a no-op (filesystem cache hit + INSERT OR IGNORE).
--
-- Deliberately NON-unique: production already carries duplicate rows from
-- the pre-fix behavior, so a UNIQUE constraint would fail to apply. The
-- repo resolves to the oldest match to converge on one canonical stream.
CREATE INDEX IF NOT EXISTS idx_streams_tenant_vod
    ON streams (tenant_id, vod_url);

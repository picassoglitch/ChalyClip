-- Clip render state — Render Migration T1.
--
-- The clip download path used to render the MP4 inline inside the HTTP
-- request handler (`await record_clip_to_mp4(...)` in dashboard.py).
-- For a 35s clip that's ~4.7 min of headless Chromium seek-and-shoot
-- screenshots, which on Railway exceeds the request timeout — the
-- operator sees a permanent "Preparing…" spinner with no error and no
-- file.
--
-- T1 moves render to a background task. The clip row gets four new
-- columns the UI's polling Download button reads:
--
--   render_state         — 'idle' (no render attempted yet, the default)
--                          'rendering' (background task in flight)
--                          'ready' (clip_render_<res>.mp4 exists on disk)
--                          'failed' (background task threw; render_error
--                                   has the message)
--   render_progress_pct  — 0-100, written by the recorder's
--                          capture_progress emitter
--   render_error         — short error message from the failure path
--                          (capped at ~300 chars so the column doesn't
--                          balloon on ffmpeg essays)
--   render_started_at    — ISO timestamp, used to detect stuck
--                          backgrounds (state="rendering" for >N min
--                          → mark failed via a sweeper or on next poll)
--
-- Idempotency: render_state="ready" + the cache file existing is the
-- terminal good state. State is reset to "idle" whenever the overlay
-- config changes (the existing _clip_overlay_save path already nukes
-- the cache file — we extend it to also reset the state columns).

ALTER TABLE clips ADD COLUMN render_state TEXT NOT NULL DEFAULT 'idle';
ALTER TABLE clips ADD COLUMN render_progress_pct INTEGER NOT NULL DEFAULT 0;
ALTER TABLE clips ADD COLUMN render_error TEXT;
ALTER TABLE clips ADD COLUMN render_started_at TEXT;

-- Index so the dashboard's "any renders in flight for this stream?"
-- query (used by the streams list to show a "renders pending" chip)
-- doesn't full-scan.
CREATE INDEX idx_clips_render_state
    ON clips (tenant_id, render_state)
    WHERE render_state IN ('rendering', 'failed');

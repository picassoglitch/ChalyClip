# Phase L — Live ingest (architectural spec)

**Status:** Scoping. No code yet. Lock the design here before opening
implementation slices.

**Decisions confirmed (from the scoping AskUserQuestion round):**

| Decision | Choice |
|---|---|
| Ingest path | RTMP push from OBS (operator-side) |
| Latency target | Soft-live, 2–5 minutes from moment → clip in inbox |
| VOD persistence | Save full recording for 24h, then auto-delete |
| Auto-publish timing | Fires WHILE the live is running (paid tiers); per-clip undo still applies |

## Why this is a real architecture change, not a feature flag

The current pipeline assumes one shape:

```
upload mp4 → Whisper (whole file) → detect (whole transcript) → cut → ...
```

Every step waits for the previous one to *finish*. That's fine for a
3-hour recorded VOD. For live it doesn't work — there's no "finish"
until the streamer hits End Stream. We need to process partial
transcripts on a rolling window, and a "candidate detected at t=842s"
can't be cut until the stream has gone past `t + post_roll_s` so the
tail of the clip exists on disk.

Three things that need to be new:

1. **Ingest layer** — accept RTMP and turn it into chunks that
   downstream code can consume.
2. **Pipeline runner mode** — a long-running process that ticks every
   N seconds, processes new chunks, emits clips as their tail
   boundaries land.
3. **Lifecycle state** — `stream.status` grows new values
   (`live` / `live_ended`) and the retention sweeper knows about
   live recordings.

Everything downstream of "clip cut" (overlays, captions, recorder,
publish, auto-publish, undo window) stays as-is. That's the
load-bearing simplification — it means Phase L only adds new code,
it doesn't refactor the editor / publish / billing surfaces.

## Architecture overview

```
                  ┌────────────────────────────────────────────┐
                  │   Streamer's OBS / Streamlabs              │
                  │   RTMP push: rtmp://live.chalybclip.chalyb.com/live│
                  │   /<stream_key>                             │
                  └────────────────────┬───────────────────────┘
                                       │ RTMP
                  ┌────────────────────▼───────────────────────┐
                  │   MediaMTX (Railway sidecar service)       │
                  │   - Auth via webhook → ChalyClip            │
                  │   - HLS muxer: 6s segments                 │
                  │   - MP4 recorder: appending file on disk   │
                  │   - Webhooks on publish-start / -end       │
                  └────────────────────┬───────────────────────┘
                                       │ HLS segments
                                       │ + MP4 mirror
                                       │ + webhook events
                  ┌────────────────────▼───────────────────────┐
                  │   ChalyClip live runner (FastAPI + worker)  │
                  │                                            │
                  │   on RTMP start:                           │
                  │     create stream row (is_live=True)       │
                  │     emit live.started event                │
                  │                                            │
                  │   every 30s while live:                    │
                  │     Modal Whisper on last 30s of segments  │
                  │     append to live_transcript              │
                  │     run detect on rolling 90s window       │
                  │     for each candidate whose tail is past: │
                  │       cut from the live MP4 mirror         │
                  │       run G.3 framing + cut overlays       │
                  │       create clip row + emit clip.cut      │
                  │       (existing variants stub fires)       │
                  │       (existing auto-publish dispatcher    │
                  │        sees the new clip + enqueues if     │
                  │        the brand kit has auto_publish=on)  │
                  │                                            │
                  │   on RTMP end:                             │
                  │     final transcribe + detect pass for     │
                  │     the trailing window                    │
                  │     stream.status → live_ended             │
                  │     emit live.ended event                  │
                  │     schedule retention sweep (24h)         │
                  │                                            │
                  │   retention sweep (24h after live_ended):  │
                  │     delete live MP4 mirror + HLS segments  │
                  │     keep clip rows + clip MP4s             │
                  │     stream.status → expired                │
                  └────────────────────────────────────────────┘
```

## Why MediaMTX (vs nginx-rtmp / SRS / homegrown)

MediaMTX is the right pick:

- Single Go binary, runs in Docker on Railway with ~50 MB RAM
- Built-in RTMP ingest + HLS muxer + MP4 recorder, no plugins
- Webhook hooks for publish-start / publish-end / auth — exactly the
  three events we need
- Active maintenance + stable API surface
- Apache 2.0 license

nginx-rtmp is older + needs nginx + rtmp-module-specific config DSL.
SRS works but its HTTP callback story is less clean than MediaMTX's.
Homegrown Python RTMP is months of work for no benefit.

## Data model changes

```sql
-- migration 019_live_ingest.sql

-- new fields on streams
ALTER TABLE streams ADD COLUMN is_live INTEGER NOT NULL DEFAULT 0;
ALTER TABLE streams ADD COLUMN live_started_at TEXT;
ALTER TABLE streams ADD COLUMN live_ended_at TEXT;

-- per-tenant RTMP key. one active key at a time; rotation invalidates
-- the previous one. operator can copy the URL+key from the dashboard.
CREATE TABLE IF NOT EXISTS live_stream_keys (
    id            TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    key_value     TEXT NOT NULL UNIQUE,
    created_at    TEXT NOT NULL,
    revoked_at    TEXT,
    last_used_at  TEXT,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);
CREATE INDEX idx_live_stream_keys_active
    ON live_stream_keys(tenant_id) WHERE revoked_at IS NULL;
```

New `stream.status` values:
- `live` — RTMP push currently active
- `live_ended` — recording finished, in 24h retention window
- `expired` — retention swept, recording gone, only clips remain

## On-disk layout

```
/data/live/
  <stream_id>/
    source.mp4              # MediaMTX appends this as the live runs
    segments/
      seg_NNNNNN.ts         # HLS segments (transient; reaped every hour)
    transcripts/
      live_transcript.json  # accumulating Whisper output, segments merged
```

After 24h: the entire `/data/live/<stream_id>/` directory is reaped.
Clip MP4s live in `/data/out/<stream_id>/clips/` (existing layout)
and are NOT affected by the live retention sweep.

## RTMP authentication

OBS hits `rtmp://live.chalybclip.chalyb.com/live/<stream_key>`. MediaMTX is
configured to POST to ChalyClip's auth webhook before accepting:

```
POST /api/internal/live/authorize
Authorization: Bearer <CHALYBCLIP_INTERNAL_SIGNING_SECRET>
{
  "stream_key": "slk_01XXXXX",
  "client_ip": "..."
}

200 OK  → MediaMTX accepts the push
401     → MediaMTX rejects
```

ChalyClip looks up the key in `live_stream_keys`, validates it's not
revoked, creates the `streams` row with `is_live=1`, returns the
`stream_id` MediaMTX should use for the recording path.

## Pipeline runner — the new piece

This is the only genuinely new long-running process. Lives in
`chalybclip/live/runner.py` (new module). Started as a background task
when a `live.started` event fires for a stream; stops when
`live.ended` fires.

Inputs per tick (every 30s while live):
- Read new HLS segments since last tick (file mtime > last_seen)
- Concatenate via ffmpeg to a temp WAV (Whisper input)
- Send to Modal Whisper (existing provider, no changes)
- Append result to `live_transcript.json`

Detection per tick:
- Re-run `detect_voice_triggers` against the last 90s of transcript
  (existing function, no changes — slice O.46 already made it
  tolerant of None brand kits)
- For each candidate, store with a `tail_ready_at` timestamp =
  `now() + post_roll_s`

Clip cut per tick:
- For each pending candidate whose `tail_ready_at < now()`:
  - The live MP4 mirror has the bytes we need
  - Run the existing `cut_clips` path with that file as the source
    (same _ffmpeg_fast_cut + _ffmpeg_reformat_9_16 from slice O.55)
  - Insert clip row, emit `clip.cut` event
  - The existing pipeline runs from here unchanged:
    - Variants stub (slice O.46)
    - Auto-publish dispatcher (slice G.5 — already runs every 60s,
      will pick up the new clip on its next tick)

On `live.ended`:
- Process the trailing 30s + post_roll_s of segments
- One final detect pass
- Cut any remaining clips
- Set `stream.status = "live_ended"`, `live_ended_at = now()`
- Emit `live.ended` event
- Schedule the retention sweep for `live_ended_at + 24h`

## Cost model (~per live hour)

| Item | Cost |
|---|---|
| MediaMTX hosting (Railway, 1 vCPU 1GB) | ~$0.05 |
| Modal Whisper (small, ~5–10s per minute of audio) | ~$1.50 |
| ffmpeg cut (Railway CPU, ~1s per clip, ~10 clips/hr typical) | ~$0.02 |
| Storage (1080p MP4 ~5 GB/hr, auto-deleted at 24h) | ~$0.10 |
| **Total per live hour** | **~$1.70** |

For the 10 active users currently on the platform doing 2-hour
streams a few times a week, that's ~$15-30/mo aggregate extra. chalyb
charges these to the operator's token balance via the existing usage
reporter — no new billing code.

## Open architectural decisions

These are flagged for explicit decision before the L.1 slice opens.
Each one only matters once — picking now avoids rework.

1. **MediaMTX as Railway sidecar or separate service?** Sidecar is
   easier (single deploy, shared volume). Separate is more
   resilient (MediaMTX crash doesn't restart the dashboard).
   **Recommendation**: separate service, both share the `/data`
   volume.

2. **Stream key format**: ULID (`slk_01XXX`) or JWT? ULID is
   simpler + matches the rest of the codebase. JWT would let us
   embed tenant_id without a DB lookup. **Recommendation**: ULID;
   the auth webhook lookup is one indexed query, not worth the JWT
   complexity.

3. **HLS segment duration**: 6s (smooth) vs 10s (cheaper Whisper
   calls). **Recommendation**: 6s — the tick cadence stays at 30s
   so we batch 5 segments per tick either way; smaller segments
   give the operator a sharper "the AI saw this 30s ago" signal.

4. **Whisper input format**: feed Modal raw HLS .ts segments
   directly (Modal's whisper container handles it) or pre-mux to
   WAV in ChalyClip? **Recommendation**: pre-mux to WAV. .ts is
   container-y; WAV makes Modal's job a pure decode, ~30% faster.

5. **Auto-publish kill-switch**: per-tenant "pause auto-publish
   while live" toggle for the cases where the streamer realizes
   mid-stream they don't want clips going out? **Recommendation**:
   yes — a single button on the live dashboard. Implement in the
   L.4 slice alongside the auto-publish wiring.

## Phased delivery plan

Each phase is a self-contained shipping slice. Operator gets
incremental value at each step; we can pause between phases without
leaving the codebase in a bad state.

### L.1 — MediaMTX + record-only (rough size: M)

Deliverables:
- MediaMTX deployed as a Railway service
- `live_stream_keys` table + migration 019
- `POST /api/internal/live/authorize` endpoint
- Webhook receivers for publish-start / publish-end → creates
  `streams` row with `is_live=1`, sets status
- Dashboard page: "Live" tab. Shows current stream key, RTMP URL,
  rotate button, live-stream list (in-progress + recent)
- After live ends, the recorded MP4 is treated like an uploaded
  VOD — operator manually clicks "Run pipeline" to process it

What works after L.1:
- Operator can stream from OBS, recording lands on disk
- Manual VOD-mode processing works on the recording
- No clips during the live, no auto-publish, no fancy stuff

### L.2 — Live transcription only (rough size: M)

Deliverables:
- `chalybclip/live/runner.py` with the tick loop
- Background task that starts on `live.started`, stops on `live.ended`
- Every 30s: read new segments, Modal Whisper, append transcript
- Dashboard live page surfaces the rolling transcript so the
  operator can see real-time captions of their own stream

What works after L.2:
- Operator sees the live transcript on the dashboard as it streams
- No clips yet — just the transcript surface
- Useful as a sanity check that the Whisper pipeline survives the
  live cadence before we wire it to clip generation

### L.3 — Live clipping (rough size: L)

Deliverables:
- Detection step in the live runner — every tick, detect against
  rolling 90s
- Pending-candidates queue with `tail_ready_at` timestamps
- Clip cut from the live MP4 mirror when tail is ready
- Existing variants + overlay + publish pipeline picks up the new
  clips without modification

What works after L.3:
- Clips appear in the inbox 2–5 minutes after the moment happens
- Operator approves them in the editor like any other clip
- Auto-publish is NOT yet enabled during live — manual approve only

### L.4 — Live auto-publish (rough size: S)

Deliverables:
- Wire G.5's `dispatch_for_clip` to fire on every live-cut clip
- Per-stream "auto-publish while live" toggle on the live dashboard
- Per-tenant kill-switch button: "pause auto-publish" (existing
  brand-kit setting can drive this)

What works after L.4:
- Approved clips from a live stream auto-publish per the existing
  brand-kit rules
- Operator can pause auto-publish mid-stream

### L.5 — Retention sweep (rough size: S)

Deliverables:
- Extend the existing retention sweeper (slice E.1) to handle
  `stream.status="live_ended"` after 24h
- Sweep `/data/live/<stream_id>/source.mp4` and segments dir
- Keep clip MP4s + clip rows
- `stream.status` → `expired`

What works after L.5:
- Live recordings auto-delete after 24h
- Disk usage stays bounded

## Out of scope for Phase L

Explicitly NOT covered:

- **Real-time (<1s) clipping** — that needs WebRTC, streaming
  Whisper, mid-segment transcript stitching. Order of magnitude
  more complex. Defer to a future Phase M.
- **Multi-camera live mixing** — operator brings a switched feed
  via OBS, we treat it as one source. If they want ChalyClip to do
  the switching, that's a different product.
- **Live captions burned into the live stream** — that's
  re-streaming territory + extra infra. Out of scope.
- **Live clips embedded in the live stream itself** ("replay during
  live") — fun but unrelated to the export flow Phase L addresses.

## Test plan (per phase)

Each slice ships with:

- Unit tests on the new pipe (auth webhook, runner tick, detection
  on partial transcript)
- An end-to-end happy-path test: simulated MediaMTX webhook events
  + a fixture MP4 fed as the "live recording" → expected clip
  rows + manifest
- The existing test suite must stay green — Phase L should not
  touch any code path that VOD-only tests exercise

## Local dev story

`docker compose up mediamtx` in the repo gives the operator a local
MediaMTX bound to `rtmp://localhost:1935/live`. OBS can push to it,
the local ChalyClip dev server receives the webhooks the same way
prod would. Reproducing a production bug locally is one command.

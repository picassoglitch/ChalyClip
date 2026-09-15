# ChalybClip — Voice-Marker Clip Extraction + Branding System

Spec for Claude Code. Phase 0 add-on to existing ChalybClip scaffold.

**Execution model:** POST-STREAM BATCH. Users upload finished VODs or
connect Google Drive / cloud storage for automated ingestion. ChalybClip
scans the audio for voice-trigger phrases and extracts clips around those
timestamps. Nothing runs during the stream itself.

---

## 1. Voice-marker clip extraction

### 1.1 Concept

The streamer says trigger phrases during their stream as natural verbal
bookmarks. After the stream, ChalybClip processes the VOD, transcribes the
audio, finds those phrases, and extracts clips around each timestamp.

Two phrases, two extraction behaviors:

| Phrase (ES) | EN equivalent | Extracts |
|---|---|---|
| `clipea esto` | "clip this" | Forward window: 30s starting AT the phrase timestamp |
| `clipeaste eso` | "you clipped that" | Retroactive window: 60s ending AT the phrase timestamp |

The retroactive case is the natural one in real streaming: the funny moment
ends, then the streamer says "clipeaste eso" to mark it. The forward case is
for things the streamer anticipates ("watch this — clipea esto").

Since this is post-stream processing, there is no buffer to maintain. The
full VOD is the buffer. We just seek into it.

### 1.2 Input sources

Three ways VODs reach ChalybClip:

| Source | Mechanism | Tenant action |
|---|---|---|
| Direct upload | Web UI drag-drop or chunked HTTP upload | Drop file on dashboard |
| Google Drive auto-watch | Drive API: watch folder, ingest new files | Connect Drive once, drop VODs in folder |
| Twitch/Kick VOD pull | Platform API → download VOD by ID | Connect account, click "import VOD" |

All three converge on the same ingestion endpoint that uploads to S3/R2 and
enqueues a `process_vod` job.

#### 1.2.a Google Drive watch (the automation path)

Setup once, then forget:

1. User connects Google Drive (OAuth)
2. User picks/creates a folder, e.g. "/ChalybClip Inbox"
3. User configures their streaming setup to save VODs there
   (OBS → Output → save to Drive-synced folder; or Streamlabs Cloud)
4. ChalybClip's Drive watcher polls folder every 60s (or uses Drive push
   notifications)
5. New file → download to S3/R2 → enqueue `process_vod`
6. User sees clips in their inbox the next morning

Use Drive push notifications (`changes.watch`) for tenants on paid tiers to
eliminate polling — webhooks fire within seconds.

#### 1.2.b Direct upload (the manual path)

For one-off VODs or users who don't want Drive integration. Standard chunked
upload with tus.io protocol or S3 multipart presigned URLs. UI shows upload
progress + thumbnail of first frame as the upload finishes.

#### 1.2.c Twitch/Kick VOD pull

Helix: `GET /videos?user_id=...` → VOD URL → download via streamlink → upload
to S3. For Twitch sub-only VODs, the tenant's OAuth token is used. For Kick,
scrape the VOD m3u8 (Kick has no official VOD API; verify current state
when implementing).

### 1.3 Processing pipeline

Single `process_vod` job runs end-to-end per VOD:

```
1. Fetch VOD from source storage → local /tmp
        ▼
2. Extract audio track (ffmpeg → 16kHz mono WAV)
        ▼
3. Speaker diarization (pyannote-audio 3.1)
   - Returns segments [start, end, speaker_label]
   - Labels stable WITHIN a VOD (SPEAKER_00, SPEAKER_01, …)
   - Per-speaker embeddings exported for cross-VOD identity resolution
        ▼
4. Speaker identity resolution
   - Match each speaker's embedding against tenant's known-speaker DB
   - Unknown speakers stored as new identities for user to label later
        ▼
5. Transcribe full audio with faster-whisper
   - Returns segments with [start, end, text, confidence]
   - 4hr stream: ~5min processing on RTX 4060 / L4 GPU
        ▼
6. Align diarization + transcription
   - For each whisper segment, assign speaker_id by max temporal overlap
        ▼
7. Scan segments for trigger phrases (per-speaker scoped)
   - Fuzzy match (Levenshtein, max_dist=2) against configured trigger list
   - Triggers attributed to the speaker who said them
   - Per-speaker cooldown
        ▼
8. For each detected trigger, compute clip window
   - forward:    [trigger_ts, trigger_ts + 30s]
   - retroactive: [trigger_ts - 60s, trigger_ts]
        ▼
9. Extract each clip with ffmpeg (stream copy, no re-encode)
        ▼
10. Enqueue clip_render jobs
    - Brand kit resolution: speaker.preferred_kit → tenant.default_kit
        ▼
11. Mark VOD as processed, notify user
```

Diarization runs before Whisper because pyannote-3.1 is faster (~30-60s on a
4hr VOD on GPU) and the speaker timing lets us later restrict transcription
to known-speech regions if we want to optimize further. For Phase 0, run
both independently and align after.

### 1.4 Whisper model choice

| Model | VRAM | 4hr VOD processing time | Quality |
|---|---|---|---|
| tiny | 1 GB | ~2 min | Too many false positives in ES |
| base | 1 GB | ~3 min | Acceptable |
| small | 2 GB | ~5 min | Recommended default |
| medium | 5 GB | ~12 min | Best for noisy streams |
| large-v3 | 10 GB | ~25 min | Overkill |

Use `faster-whisper` with CTranslate2 backend, `compute_type="float16"` on
GPU. The RTX 4060 (8GB) handles `small` and `medium` comfortably.

For language: `language="es"` for Aldo's tenant. Set `language=None` for
auto-detect on multi-tenant cloud. For Spanglish streams, `small` with
`language=None` handles code-switching reliably enough.

### 1.5 Job orchestration

```
Redis + RQ (existing ChalybClip queue)

Queues:
  - drive_ingest        (light)
  - vod_process         (GPU-bound, dedicated worker pool)
  - clip_render         (GPU + CPU, separate worker pool)
  - publish             (network-bound, can be many)

A 4hr VOD with 25 detected triggers produces:
  1 vod_process job + 25 clip_render jobs
```

VOD processing is the bottleneck — run a small pool of GPU workers (1-2 to
start, autoscale on queue depth).

### 1.6 Config (per tenant, durable in DB)

```json
{
  "voice_triggers": {
    "enabled": true,
    "language": "es",
    "whisper_model": "small",
    "forward_phrases": ["clipea esto", "clip this", "clipéalo"],
    "retroactive_phrases": ["clipeaste eso", "lo clipeaste", "ya lo clipeaste"],
    "forward_duration_s": 30,
    "retroactive_lookback_s": 60,
    "cooldown_s": 10,
    "min_confidence": 0.7,
    "fuzzy_max_distance": 2
  },
  "diarization": {
    "enabled": true,
    "model": "pyannote/speaker-diarization-3.1",
    "match_threshold": 0.75,
    "min_speech_for_id_s": 30
  },
  "ingestion": {
    "drive": {
      "enabled": true,
      "folder_id": "1AbCdEf...",
      "use_webhooks": false
    },
    "twitch_vod_auto": false,
    "kick_vod_auto": false
  },
  "publishing": {
    "default_mode": "review_first",
    "auto_publish_undo_window_min": 60
  },
  "retention": {
    "vod_days": 30,
    "clip_days": 90,
    "transcript_days": 365,
    "delete_originals_after_processing": false
  }
}
```

Custom trigger phrases live on the brand kit (`brand_kits.custom_trigger_phrases`),
NOT on the tenant. This lets a tenant have one kit with "monchi esto" and
another kit with "córtalo", picked per speaker. Resolution at scan time
merges tenant base list + kit additions; `set()` dedupes.

### 1.7 False-positive handling

- **Confidence threshold:** skip Whisper segments below 0.7 avg log-prob
- **Cooldown:** 10s between same-kind triggers per speaker
- **Fuzzy match cap:** Levenshtein distance ≤ 2
- **UI inbox:** every detected clip lands in the inbox with the detected
  phrase + 5s of surrounding transcript shown, so the user can verify
- **Auto-discard:** clips marked rejected feed a per-tenant blocklist

## 2. Other trigger types (post-stream variants)

Same idea, different signal extraction from the VOD:

| Source | What gets scanned | Example |
|---|---|---|
| Voice phrase | Whisper transcript | "clipea esto" |
| Audio peak | RMS loudness on audio track | Sudden scream/laugh (configurable dB threshold) |
| Chat overlay | Twitch/Kick chat log (downloaded with VOD) | N+ "LUL"/"KEKW" in 5s window |
| Visual | Frame-based ML on sampled frames | "Victory" screen, kill cam |
| Manual | User scrubs VOD and marks moments | Click + drag on timeline |

All produce `(timestamp, kind, metadata)` tuples that go into the same
clip-window extraction step.

For Phase 0 ship voice + manual. Chat-overlay and audio-peak in Phase 1.
Visual ML in Phase 2.

## 3. Branding system (UI + asset generation)

### 3.1 Brand Kit data model

```sql
CREATE TABLE brand_kits (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  name TEXT NOT NULL,              -- "AARA", "Aldo Villanueva", "Chalyb Academy"
  is_default BOOLEAN DEFAULT false,
  -- Colors
  primary_color TEXT NOT NULL,     -- "#FF3366"
  accent_color TEXT NOT NULL,      -- "#FFD700"
  text_color TEXT DEFAULT '#FFFFFF',
  -- Typography
  font_family TEXT DEFAULT 'Inter',
  font_weight INT DEFAULT 800,
  -- Assets (S3 keys)
  logo_url TEXT,                   -- transparent PNG, square
  logo_dark_url TEXT,              -- light version for dark gameplay
  watermark_url TEXT,              -- handle as image
  intro_sting_url TEXT,            -- 1-2s mp4 (optional)
  outro_sting_url TEXT,
  -- Caption style
  caption_style JSONB,             -- see 3.4
  -- Layout preferences
  default_layout TEXT,             -- 'split_stack' | 'pip' | 'blurred_bg'
  -- Social handles (rendered into watermark)
  handle_tiktok TEXT,
  handle_youtube TEXT,
  handle_instagram TEXT,
  handle_kick TEXT,
  -- AI generation metadata
  ai_generated BOOLEAN DEFAULT false,
  ai_prompt TEXT,
  ai_provider TEXT,
  -- Per-kit auto-publish opt-in (default OFF; review-first is the default UX)
  auto_publish_enabled BOOLEAN DEFAULT false,
  auto_publish_platforms TEXT[] DEFAULT '{}',   -- e.g. {'tiktok','shorts'}
  auto_publish_delay_min INT DEFAULT 60,        -- undo window before push
  -- Custom trigger phrases — per-kit overrides + additions
  custom_trigger_phrases JSONB DEFAULT '{"forward":[], "retroactive":[]}',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_brand_kits_tenant ON brand_kits(tenant_id);

-- Speakers: persistent identities across VODs for a tenant
CREATE TABLE speakers (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  display_name TEXT NOT NULL,                -- "Aldo", "Cano", "Guest A"
  is_self BOOLEAN DEFAULT false,             -- the tenant's own voice
  preferred_brand_kit_id UUID REFERENCES brand_kits(id),
  embedding BYTEA,                           -- pyannote speaker embedding, ~512 floats
  embedding_dim INT DEFAULT 512,
  sample_audio_url TEXT,                     -- short clip for user to verify identity
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_speakers_tenant ON speakers(tenant_id);

-- VOD-scoped speaker labels (resolved against persistent speakers table)
CREATE TABLE vod_speakers (
  id UUID PRIMARY KEY,
  vod_id UUID NOT NULL REFERENCES vods(id) ON DELETE CASCADE,
  speaker_label TEXT NOT NULL,               -- "SPEAKER_00" from pyannote
  resolved_speaker_id UUID REFERENCES speakers(id),  -- NULL if unknown
  confidence FLOAT,                          -- cosine sim to matched speaker
  total_speech_s FLOAT,                      -- how much they spoke in this VOD
  UNIQUE(vod_id, speaker_label)
);
CREATE INDEX idx_vod_speakers_vod ON vod_speakers(vod_id);
```

**Speaker matching:** when a new VOD finishes diarization, each `SPEAKER_NN`
embedding is compared (cosine similarity) against the tenant's `speakers`
table. Match if similarity > 0.75. Unmatched speakers create a new pending
row that the user can label in the UI (or merge with an existing one).

**Brand kit resolution at clip-render time:**

1. If `trigger.speaker.preferred_brand_kit_id` → use it
2. Else use `tenant.default_brand_kit`
3. Else use system default

### 3.2 Brand Kit UI page

Route: `/brand-kits/:id`

```
┌───────────────────────────────────────────────────────────┐
│  Brand Kit: AARA                          [Set as default]│
├───────────────────────────────────────────────────────────┤
│  ┌─ Identity ────────────────────┐  ┌─ Live Preview ────┐│
│  │ Name:    [AARA            ]   │  │                   ││
│  │ Primary: [#FF3366  ] [pick]   │  │   [facecam]       ││
│  │ Accent:  [#FFD700  ] [pick]   │  │                   ││
│  │ Font:    [Inter      ▾]       │  │   [gameplay]      ││
│  └───────────────────────────────┘  │   "EPIC PLAY"     ││
│                                     │   @aara_art       ││
│  ┌─ Logo ────────────────────────┐  │   [logo]          ││
│  │  [▢ drop logo here]           │  └───────────────────┘│
│  │  ── OR ──                     │                       │
│  │  [Generate with AI ▾]         │  ┌─ Layout ──────────┐│
│  │   Style: [Minimal mark    ▾]  │  │ ( ) Split stack   ││
│  │   [Generate variants]         │  │ (•) PiP           ││
│  └───────────────────────────────┘  │ ( ) Blurred bg    ││
│                                     └───────────────────┘│
│  ┌─ Captions ────────────────────┐                       │
│  │ Style: [Karaoke pop ▾]        │                       │
│  │ Size:  [72px ──●──]           │                       │
│  │ Stroke:[4px ─●────]           │                       │
│  └───────────────────────────────┘  [Save]  [Test render]│
└───────────────────────────────────────────────────────────┘
```

### 3.3 Asset generation (both upload AND AI)

Every asset field supports two input paths: upload or AI-generate. UI shows
both side-by-side, user picks per field.

#### 3.3.a Logo generation

Two-stage pipeline: Anthropic API call → returns SVG code → sanitize +
rasterize → store SVG + PNG (512, 1024). UI shows 3 variants per generation
(run the call 3 times with slight prompt variance: "minimal", "geometric",
"monogram"). User picks one.

Style presets to expose in UI:
- Minimal mark / monogram
- Geometric / abstract shape
- Streamer logo (badge-style)
- Glitch / cyber

#### 3.3.b Thumbnail generation

Frame extraction + text overlay rather than image gen — cheaper, more
on-brand, uses real footage:

1. Extract 3 candidate frames (peak motion / peak audio / midpoint)
2. Score and pick best (prefer faces)
3. Composite variants for 16:9 (1280×720), 9:16 (1080×1920), 1:1 (1080×1080)

For users who want AI-generated title text, call the LLM separately to
generate 3 viral hook variants from the clip transcript.

### 3.4 Caption style schema

```json
{
  "preset": "karaoke_pop",
  "font_family": "Inter",
  "font_weight": 900,
  "font_size_px": 72,
  "fill_color": "#FFFFFF",
  "stroke_color": "#000000",
  "stroke_width_px": 5,
  "highlight_color": "#FFD700",
  "highlight_mode": "active_word",
  "position": "upper_third",
  "max_words_per_line": 4,
  "animation": "pop_in",
  "shadow": {
    "enabled": true,
    "offset_y": 3,
    "blur": 6,
    "color": "rgba(0,0,0,0.6)"
  },
  "censor_swears": true
}
```

Caption presets to ship:
- `karaoke_pop` — word-by-word highlight, scale bounce (default)
- `typewriter` — char-by-char reveal
- `bold_block` — 2-3 word chunks, hard cut
- `subtle` — single line bottom, no animation (for serious content)

### 3.5 Watermark/handle rendering

Burned into every export:
- **Handle:** top-left, 24px from edges, 32px font, brand accent color
- **Logo:** bottom-right, 24px from edges, 5-8% of frame width (~80px on a
  1080-wide canvas)

Both pulled from the active Brand Kit at render time.

### 3.6 Intro/outro stings (optional)

If brand kit has `intro_sting_url`, prepend to clip (max 1.5s, auto-trimmed).
Same for outro. Stings are MP4s with alpha channel preferred (.mov ProRes
4444) but plain MP4 works.

## 4. UI routes

| Route | Purpose |
|---|---|
| `/inbox` | Upload VOD, view processing status, list captured clips |
| `/sources` | Connect Google Drive, Twitch, Kick; configure auto-ingest |
| `/sources/drive` | Pick Drive folder to watch, set polling/webhook mode |
| `/triggers` | List trigger phrases per kit, edit, add custom phrases |
| `/triggers/test` | Paste sample transcript, see which phrases would fire |
| `/brand-kits` | List of brand kits for tenant |
| `/brand-kits/new` | Create kit (upload OR AI generate per asset) |
| `/brand-kits/:id` | Edit kit + live preview |
| `/speakers` | List detected speakers across all VODs, label/merge, assign preferred brand kit |
| `/speakers/:id` | Speaker detail: sample audio playback, all clips by this speaker |
| `/clips` | All processed clips, filterable by VOD, trigger kind, status |
| `/clips/:id` | Single clip editor: brand kit, layout, caption preset, regenerate |
| `/vods/:id` | VOD detail: transcript with trigger markers highlighted, jump to clip |

### 4.1 Inbox UI behavior

After a VOD finishes processing, the inbox shows clips grouped by VOD with
the detected phrase + transcript context + speaker label. Bulk publish uses
default brand kit + default layout + auto-captions + AI-generated titles,
pushes to selected platforms.

### 4.2 VOD timeline view

For each VOD show the transcript inline with detected trigger phrases
highlighted. User can click any trigger to jump to that clip in the inbox.

## 5. Rendering pipeline (per clip)

```
1. Source clip (extracted MP4 from VOD)
        ▼
2. Trim & normalize to 1080×1920 canvas (9:16)
        ▼
3. Layout composer (split_stack / pip / blurred_bg)
        ▼
4. Caption pass (Whisper transcribe → styled ASS → burn-in)
   Note: we already have segment-level transcript from VOD processing,
         but re-transcribe at word level for caption sync
        ▼
5. Brand pass (handle top-left, logo bottom-right, color bar if template uses one)
        ▼
6. Intro/outro stings (concat if configured)
        ▼
7. Audio normalize (-14 LUFS, AAC 192kbps)
        ▼
8. Export (H.264, yuv420p, 8-12 Mbps, 1080×1920)
        ▼
9. Thumbnail generation (1280×720 + 1080×1920 + 1080×1080)
        ▼
10. LLM variant generation (3 title/hook variants per platform)
        ▼
11. Stage to publish (per-platform metadata)
```

Each step is a discrete worker job so failures retry idempotently.

## 6. Tech stack additions

| Component | Library |
|---|---|
| STT | `faster-whisper` (CTranslate2 backend) |
| Diarization | `pyannote.audio` 3.1 + HuggingFace token in `HF_TOKEN` env |
| Speaker embeddings | `pyannote/embedding` model, cosine sim via numpy |
| Drive integration | `google-api-python-client`, `google-auth` |
| VOD download | `streamlink` for Twitch/Kick |
| Video processing | `ffmpeg-python` wrapper |
| Caption rendering | `aeneas` for forced alignment, `ass` lib for subtitle gen |
| SVG sanitize | `bleach` + custom allowlist |
| SVG rasterize | `cairosvg` |
| Thumbnail compositing | `Pillow` |
| Frame analysis | `opencv-python` (face detection, scene change) |
| Job queue | Redis + RQ (consistent with ChalybClip v0.5) |
| Upload protocol | `tus.py` (server side) or S3 multipart presigned URLs |
| WebSocket UI events | FastAPI WebSocket + Angular RxJS |

## 7. Phase 0 deliverables (in order)

1. **VOD ingestion (direct upload)** — chunked upload endpoint + S3 storage
   + `vod_process` job stub. VOD rows include `retention_until` set from
   tenant config.
2. **Diarization worker** — pyannote-3.1, emits speaker turns + per-speaker
   embeddings, stored on `vod_speakers`.
3. **Whisper transcription worker** — `faster-whisper` on GPU, full VOD →
   segments JSON stored alongside VOD.
4. **Diarization-aware trigger scanner** — align Whisper + pyannote, fuzzy
   phrase match per speaker, per-speaker cooldown.
5. **Speaker identity resolution** — embedding match against `speakers`
   table, auto-create pending speakers for new voices.
6. **Clip extraction** — ffmpeg stream copy per window, store raw clips,
   enqueue `clip_render` with `speaker_id` annotation.
7. **Brand Kit CRUD** — DB migration (kits + speakers + vod_speakers) +
   Angular forms + asset upload to S3. Includes per-kit auto-publish opt-in
   + custom trigger phrases JSON.
8. **Render pipeline v1** — single layout (PiP), captions, watermark, no
   stings yet. Brand kit resolved from speaker → tenant default.
9. **Inbox UI** — list VODs + their captured clips, group by speaker,
   manual review, single-clip preview.
10. **Speakers admin UI** — `/speakers` page to label/merge unknown
    speakers, assign preferred brand kits.
11. **AI logo generation** — Anthropic API call, SVG sanitize, rasterize,
    store. 3 variants per generation.
12. **Thumbnail generator** — frame extract + composite, all 3 aspect
    ratios (16:9, 9:16, 1:1).
13. **Google Drive watcher** — OAuth flow, folder watch, auto-ingest.
    Polling for Phase 0; webhooks for Pro+.
14. **Retention sweeper** — daily cron deleting VODs / clips / transcripts
    past their retention windows. Hard-delete; no soft-delete.
15. **Auto-publish worker** — for kits with `auto_publish_enabled`, push
    clips after `auto_publish_delay_min` undo window expires.
16. **Multi-platform publish** — TikTok, YouTube Shorts, Instagram Reels.

Phase 0 ends at deliverable 16 — that's the MVP loop: drop VOD in Drive →
wake up to branded vertical clips → review or let auto-publish fire.

17. *Twitch/Kick VOD pull, stings, multi-layout, caption preset library —
    polish.*

## 8. Cost model considerations

VOD processing is GPU-bound and the dominant cost. Diarization (~30-60s) +
Whisper (~5min) per 4hr VOD:

| Tier | VOD hours/month | GPU cost (cloud) | Reasonable price |
|---|---|---|---|
| Free | 5 | ~$1.20 | $0 (gated by watermark + 720p, no diarization) |
| Starter | 30 | ~$7 | $19/mo |
| Pro | 100 | ~$22 | $49/mo |
| Studio | 300 | ~$60 | $119/mo |

Free tier disables diarization to save cost — single-kit branding only.
Diarization unlocks at Starter.

For Aldo's own tenant, run both Whisper and pyannote on Quantor (RTX 4060,
8GB) to bypass cloud GPU cost entirely. Memory budget: 2GB Whisper small +
2GB pyannote = 4GB, fits comfortably with headroom for the 4060.

## 9. Locked decisions

These are decided. Do not re-prompt:

- **Auto-publish vs review-first:** Review-first by default. Per-kit opt-in
  (`brand_kits.auto_publish_enabled`) flips a kit into auto-publish mode
  with a configurable undo window (default 60 min).
- **VOD retention:** 30-day default for VODs, 90-day for rendered clips,
  365-day for transcripts. All configurable via `tenant.retention.*`. Daily
  sweeper job hard-deletes past retention.
- **Trigger phrase customization:** Per-kit JSON arrays at
  `brand_kits.custom_trigger_phrases` (forward + retroactive). Additively
  merged with the tenant-level base list at scan time; `set()` dedupes.
- **Multi-streamer in one VOD:** Full pyannote-audio 3.1 diarization in
  Phase 0. Per-speaker triggers, per-speaker cooldowns, per-speaker
  preferred brand kits. Speaker identities persist across VODs via
  embedding match (threshold 0.75 cosine sim).

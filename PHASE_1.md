## Phase 1 — Multi-tenant core + local vision (3 weeks)

**Goal:** Tenancy and SQLite persistence become the load-bearing foundation. Once the DB shape and tenant enforcement are locked, we add the three new detectors (chat heat, audio energy, visual signals), the FastAPI + HTMX dashboard, and the Buffer publisher — all building against the same DB-backed repos.

**Sequencing principle:** the DB schema and tenancy contract are settled in Task 0 *before any detector, vision, or dashboard work starts*. Schema changes after Task 0 require a migration. Detectors, vision, and the dashboard get to write against a stable target instead of getting rewritten when the DB evolves.

**In scope:**
- Tenancy schema + enforcement (CLAUDE.md hard rule #1 becomes the actual contract).
- DB persistence (aiosqlite) is the source of truth. JSON manifests become read-models regenerated from the DB.
- Phase 0 services dual-write to the DB during a backward-compatible transition; the CLI keeps working unchanged.
- Event log table written from day one — every state transition emits.
- Three new detectors: chat heat, audio energy, visual signals.
- Local vision pipeline (PySceneDetect, MediaPipe, OpenCV) — heuristics only. No multimodal LLM scoring yet.
- Smart 9:16 crop via local face detection (no LLM).
- Auto-thumbnail picker via local heuristic.
- `LLMRouter.complete_multimodal()` capability + variant generator's multimodal opt-in.
- FastAPI REST API + HTMX server-rendered dashboard.
- Buffer publisher.

**Out of scope (Phase 2+):**
- Multimodal LLM scoring (the cheap-heuristic → premium-vision two-stage funnel).
- Vision-LLM-driven smart crop / thumbnail (Phase 1 ships face-detect-only versions).
- Native TikTok / YT Shorts publishers (Phase 2).
- MCP server (Phase 2).
- Webhook dispatch (Phase 2 — Phase 1 only writes the event log; subscribers are Phase 2).
- Auth UI / billing / Stripe (Phase 3+).
- Cloud migration (Aurora, ECS, SQS, S3 — Phase 3).

---

## Exit criterion

```bash
# Phase 1 schema + a tenant
$ chalybclip db init
$ chalybclip tenants add aldo "Aldo Villanueva"
$ chalybclip tokens issue --tenant aldo --scope full   # prints `tok_...`

# Phase 0 CLI keeps working, now writing through the DB:
$ chalybclip process <vod_url> --persona aldo_villanueva --tenant aldo
# → 4 detectors fire; visual signals contribute to candidates
# → smart crop centers on faces (was center-crop in Phase 0)
# → variants generated; multimodal opt-in available per persona
# → publish_jobs created for Aldo's connected Buffer accounts
# → events table records every state transition

# FastAPI + HTMX dashboard:
$ uvicorn chalybclip.api.app:app --port 8000
$ open http://localhost:8000/streams
# → list of streams, click in to see candidates with all four signal types,
#   approve / edit / publish clips

# Buffer publisher worker:
$ chalybclip publish --tenant aldo
# → drains pending publish_jobs, pushes to Buffer, marks them sent
```

The acceptance demo: Aldo runs his own VOD through the dashboard, clicks "approve all" on the clip review page, the variants land in his Buffer queue, and every transition shows up in `events`.

---

## Task list (in order)

> Tasks 0-2 are the foundation; nothing in Tasks 3+ ships until the schema and the tenancy contract are locked.

### 0. Tenancy + SQLite foundation (Days 1-3) — LOCK-DOWN TASK

This is the load-bearing task. After it merges, **the schema is frozen for Phase 1**: any change is a numbered migration. Detectors, vision, dashboard, and publisher all build against repos defined here.

- [ ] **Schema** — `chalybclip/db/schema.sql`, full Phase 1 schema in one shot (not added-to-incrementally):
  - `schema_version(version)` — single-row, the migration runner checks it.
  - `tenants(id PK, name, created_at)`
  - `users(id PK, tenant_id FK, email, role, created_at)` — Phase 1 has one user per tenant; row exists so Phase 3 auth has a target.
  - `api_tokens(id PK, tenant_id FK, hash, scope, created_at, last_used_at)` — `hash` is sha256 of `tok_<ulid>`, never the raw token.
  - `personas(id PK, tenant_id FK, name, primary_language, target_languages_json, voice_prompt, routing_tags_json, created_at)`
  - `connected_accounts(id PK, tenant_id FK, platform, external_id, display_name, oauth_blob_json, created_at)`
  - `streams(id PK, tenant_id FK, vod_url, platform, title, channel, duration_s, status, created_at)`
  - `transcripts(stream_id PK FK, tenant_id FK, language, model, segments_json)` — JSON column for now; Phase 3 normalizes if needed.
  - `candidates(id PK, stream_id FK, tenant_id FK, ts, score, reason, evidence_json, created_at)`
  - `clips(id PK, stream_id FK, tenant_id FK, candidate_id FK, start_s, end_s, duration_s, width, height, path, smart_crop_box_json, thumbnail_frame_path, status, created_at)`
  - `variants(id PK, clip_id FK, tenant_id FK, persona_id FK, language, caption, title_card_text, hashtags_json, model, created_at)`
  - `llm_calls(id PK, tenant_id FK, purpose, provider, model, quality, input_tokens, output_tokens, cost_usd_micros, status, error, attempts, ts)`
  - `publish_jobs(id PK, tenant_id FK, clip_id FK, variant_id FK, account_id FK, platform, status, attempts, last_error, scheduled_for, created_at)`
  - `events(id PK, tenant_id FK, type, payload_json, ts)`
  - `visual_signals(stream_id FK, tenant_id FK, ts_offset_s, scene_cut, face_emotion, motion_energy, text_changed, PRIMARY KEY (stream_id, ts_offset_s))` — Phase 1 lays the table; the vision pipeline (Task 5) populates it.
  - **Index discipline:** every domain table has a composite index `(tenant_id, <hot column>)`. Cross-tenant scans are impossible by construction.
- [ ] **Migration runner** — `chalybclip/db/migrations.py`:
  - Forward-only, hand-rolled. SQL files numbered `001_init.sql`, `002_*.sql`, etc.
  - Reads `schema_version`, applies anything newer in a transaction, bumps version.
  - No Alembic in Phase 1. Revisit if churn justifies it.
- [ ] **Connection pool** — `chalybclip/db/connection.py`:
  - aiosqlite pool with `WAL` + `foreign_keys = ON` + `synchronous = NORMAL`.
  - Lifespan-managed in FastAPI, fixture-managed in tests.
- [ ] **Tenancy context** — `chalybclip/tenancy/context.py`:
  - `current_tenant_id() -> str` (raises if unbound).
  - `bound_tenant(tenant_id)` async context manager that sets a `contextvars.ContextVar`.
  - `assert_tenant(expected: str)` for service-function preconditions.
- [ ] **Tenancy middleware** — `chalybclip/tenancy/middleware.py`:
  - FastAPI middleware: extract tenant from `Authorization: Bearer tok_...`, hash → look up `api_tokens.hash`, bind `current_tenant_id` for the request.
  - 401 on missing / unknown / expired token. 403 if a handler queries another tenant.
- [ ] **Repository layer** — `chalybclip/db/repos.py`:
  - One repo class per table. CRUD only — no business logic.
  - Every read/write asserts `tenant_id == current_tenant_id()` before issuing SQL. Compromised handler can't bypass it.
  - Async (aiosqlite). Methods return Pydantic models, not raw rows.
- [ ] **CLI hooks** — `chalybclip db init`, `chalybclip tenants add <id> <name>`, `chalybclip tokens issue --tenant <id> --scope <full|read>`.
- [ ] **Tests — the lock-down bar:**
  - Round-trip insert/select on every table.
  - Cross-tenant access attempts always raise (each repo has a "tenant B reading tenant A's row" test).
  - Foreign-key violations rejected (delete a referenced row → error, not silent corruption).
  - Migration runner: applies 001 cleanly on empty DB; idempotent on re-run; refuses to downgrade.
  - Tenancy contextvar: nested `bound_tenant` blocks restore on exit; `current_tenant_id()` outside any block raises.
  - 100+ fast unit tests; this is the point at which schema/tenancy mistakes become hard to make.

**Exit criterion for Task 0:** `pytest tests/db tests/tenancy` is fully green, the schema file is reviewed, and the schema-version table reads `1`. After this, no further DB-shape changes happen without a numbered migration file.

### 1. Persistence migration (Day 4)

Phase 0 services start dual-writing through the repos. JSON files keep being produced for backward-compatible CLI behavior; DB rows become canonical.

- [ ] Each Phase 0 service (`ingest_vod`, `transcribe`, `detect_voice_triggers`, `cut_clips`, `generate_variants`) writes to the DB *and* the existing JSON file in one transaction (DB first; JSON write is best-effort).
- [ ] Idempotency on insert conflicts: fall through to the existing-row read (no overwrite without `force`).
- [ ] Router moves `llm_calls.jsonl` write into a `llm_calls` row. The JSONL stays as a debug breadcrumb (writes after the DB row commits).
- [ ] `manifest.json` becomes a render of DB state, regenerated by `chalybclip render-manifest <stream_id>`. The orchestrator still writes it after a `process_vod` call so existing tests pass.
- [ ] Tests: re-running `process_vod` against the DB doesn't duplicate rows; aborting mid-pipeline and resuming still picks the cached DB state.

### 2. Event log baseline (Day 5)

Wired up before any new feature so every transition lands in `events` from day one.

- [ ] `chalybclip/events/log.py` — `emit(type, payload)` reads `current_tenant_id()` and writes to `events`.
- [ ] Hook into existing services: emit on `stream.created`, `stream.processed`, `clip.ready_for_review`, `clip.approved`, `clip.published`, `clip.failed`, `publish_job.failed`, `llm.fallback`, `llm.exhausted`.
- [ ] Tests: each transition writes the expected event row; cross-tenant emission is impossible.

### 3. Chat heat detector (Day 6)

- [ ] Extend `ingest` to pull chat replay where the platform supports it (Kick chat-replay JSON; Twitch `/v5/videos/<id>/comments`).
- [ ] `chalybclip/ingest/chat_replay.py` — fetch + normalize to `ChatMessage(ts, user, text)`.
- [ ] `chalybclip/detect/chat_heat.py` — rolling baseline_window_s msg/sec; spike when `current_rate > spike_ratio × baseline` AND `current_rate >= absolute_floor_msg_per_s`.
- [ ] Rename `detect_voice_triggers` → `detect_candidates`; voice + chat fuse into one Candidate stream.
- [ ] Tests: synthetic chat replay → expected spikes; chat-replay-not-available platforms degrade silently.

### 4. Audio energy detector (Day 7)

- [ ] **Dep already present** — `numpy` ships with the existing stack via `faster-whisper`. No `librosa`/`scipy` needed; `numpy.fft` plus a manual RMS window is enough.
- [ ] `chalybclip/detect/audio_energy.py` — windowed RMS over the existing 16 kHz mono WAV; spike detection vs `baseline_window_s` baseline; require `sustain_s` continuous over-threshold to fire (suppresses one-frame pops).
- [ ] Wire into `detect_candidates`.
- [ ] Tests: known-loud audio fixtures fire; quiet ones don't.

### 5. Local vision pipeline (Days 8-10)

- [ ] **New deps — confirm with Aldo:** `scenedetect`, `mediapipe`, `opencv-python`. CPU-only; MediaPipe optionally uses GPU.
- [ ] `chalybclip/vision/scene_detect.py` — PySceneDetect adapter; emit `SceneCut(ts)`.
- [ ] `chalybclip/vision/face_emotion.py` — MediaPipe Face Mesh sampled at 2 fps; label ∈ {neutral, smile, laugh, shock, anger, sad}.
- [ ] `chalybclip/vision/motion.py` — OpenCV frame-diff magnitude per second.
- [ ] `chalybclip/vision/frame_sampler.py` — `sample_frames(video, ts, n)`; saves to `<stream_dir>/frames/` and indexes paths via the DB.
- [ ] `chalybclip/vision/service.py::analyze_video(tenant_id, stream)` — runs all three local detectors, writes `visual_signals` rows, returns a `VisualSignalTrack`.
- [ ] CLI: `chalybclip analyze-video <stream_id>` (between `transcribe` and `detect` in the orchestrator).
- [ ] Tests: tiny test video with a known scene cut, smile frame, motion burst.

### 6. Visual signals → detector fusion (Day 11)

- [ ] `chalybclip/detect/visual_signals.py` — fuse scene_cut + face_emotion + motion_energy into Candidates.
- [ ] `detect_candidates` now merges voice + chat + audio + visual into one stream of Candidates with composite scores and per-signal sub-evidence.
- [ ] Tests: synthetic per-signal hits → merged candidates with correct composite scores.

### 7. Smart crop + auto-thumbnail (Day 12)

- [ ] `chalybclip/clip/smart_crop.py` — face-detect-driven 9:16 crop center; falls back to center-crop when no face. Reuses MediaPipe from §5.
- [ ] `chalybclip/clip/thumbnail.py` — pick best frame: highest face-emotion confidence × no-motion-blur heuristic.
- [ ] `cut_clips` consumes both; clip rows carry `smart_crop_box_json` + `thumbnail_frame_path`.
- [ ] Tests: a centered-face frame produces a crop centered on the face.

### 8. LLMRouter vision capability + variant multimodal opt-in (Day 13)

- [ ] `LLMRouter.complete_multimodal(tenant_id, purpose, system, user, images, schema, quality)` — same shape as `complete()` plus an `images` list (S3 URLs in Phase 3; local paths re-encoded to base64 in Phase 1).
- [ ] `AnthropicProvider.complete_multimodal()` — concrete impl using Claude vision message format.
- [ ] `chalybclip/llm/frame_cache.py` — local frame cache keyed by `(stream_id, ts)`; reused across multiple LLM calls on the same clip.
- [ ] `generate_variants` gets an opt-in flag `use_vision`; when true, samples 3 frames via `frame_sampler` and calls `complete_multimodal`.
- [ ] Tests: fake provider replays a multimodal response; cache hits skip re-encoding.

### 9. FastAPI REST API (Days 14-15)

- [ ] `chalybclip/api/app.py` — FastAPI app, lifespan-managed DB pool, tenancy middleware mounted globally.
- [ ] Routes (all tenant-scoped, all async):
  - `POST /streams` — kick off `process_vod` (background task).
  - `GET /streams` / `GET /streams/{id}`
  - `GET /streams/{id}/candidates`
  - `GET /streams/{id}/clips`
  - `GET /clips/{id}` / `PATCH /clips/{id}` (approve / reject / edit caption)
  - `POST /clips/{id}/publish` — enqueue publish_job(s) for the clip's connected accounts.
  - `GET /personas` / `POST /personas` / `PATCH /personas/{id}`
  - `GET /llm-calls` (cost dashboard)
- [ ] Tests: httpx AsyncClient, two tenants, full isolation verified.

### 10. HTMX dashboard (Days 16-17)

- [ ] `chalybclip/api/templates/` — Jinja2 + HTMX. Server-rendered, no React.
- [ ] Pages:
  - `/streams` — list + "process new VOD" form.
  - `/streams/{id}` — stream detail: candidates list (per-signal evidence), clips grid, run-summary card.
  - `/clips/{id}` — clip player, frame strip, variants picker (radio buttons), approve / edit / publish buttons.
  - `/settings/llm` — per-purpose model + quality selector.
  - `/settings/personas` — persona CRUD.
  - `/connected-accounts` — Buffer connection (manual API key entry in Phase 1; OAuth in Phase 3).
- [ ] Static assets minimal: HTMX from CDN, Pico.css for baseline styling.

### 11. Buffer publisher (Days 18-19)

- [ ] `chalybclip/publish/buffer.py` — thin Buffer API client (httpx).
- [ ] `chalybclip/publish/service.py::run_publish_jobs(tenant_id, *, max_jobs=50)` — pulls ready `publish_jobs`, posts to Buffer, marks sent / failed, retries with backoff.
- [ ] CLI: `chalybclip publish --tenant <id>` runs one drain pass.
- [ ] Background task in FastAPI lifespan kicks the same drain every 60s.
- [ ] Tests: respx mocks Buffer API; verify retry on 5xx, give-up after N attempts, successful posts mark the row.

### 12. Polish (Days 20-21)

- [ ] Docs update: README adds DB init + `uvicorn` quick-start.
- [ ] PHASE_2 backlog stub.
- [ ] Run on Aldo's actual stream; iterate persona prompts based on what comes back.
- [ ] PR opened.

---

## Day-by-day rough plan

| Day | Focus |
|---|---|
| 1-3 | **Task 0 — Tenancy + SQLite foundation (LOCK-DOWN)** |
| 4 | Persistence migration; Phase 0 services dual-write |
| 5 | Event log baseline wired in |
| 6 | Chat heat detector + chat replay ingest |
| 7 | Audio energy detector |
| 8-10 | Local vision pipeline (scene cuts, face/emotion, motion) |
| 11 | Visual signals fused into the detector stream |
| 12 | Smart crop + auto-thumbnail |
| 13 | LLMRouter vision + variant multimodal opt-in |
| 14-15 | FastAPI REST API |
| 16-17 | HTMX dashboard |
| 18-19 | Buffer publisher |
| 20-21 | Polish, demo, README, PR |

---

## New dependencies (confirm with Aldo before adding)

| Package | Purpose | Module | Phase 1 task |
|---|---|---|---|
| `fastapi` | REST API | `api/` | 9 |
| `uvicorn[standard]` | ASGI server | `api/` | 9 |
| `jinja2` | HTMX dashboard templates | `api/` | 10 |
| `python-multipart` | form posts (HTMX file uploads) | `api/` | 10 |
| `scenedetect` | PySceneDetect | `vision/scene_detect.py` | 5 |
| `mediapipe` | face mesh + emotion | `vision/face_emotion.py` | 5 |
| `opencv-python` | motion energy + frame sampling | `vision/motion.py`, `vision/frame_sampler.py` | 5 |

`aiosqlite`, `httpx`, and `numpy` (transitive via faster-whisper) are already in `pyproject.toml`. `scipy` and `librosa` are deliberately NOT added — `numpy.fft` is enough for Phase 1 audio energy.

---

## Notes for the implementer

- **Schema lock:** after Task 0 lands, no Phase 1 task changes the schema without a numbered migration file. If a detector or dashboard task needs a new column, that's a sub-PR before the feature work.
- **Don't break the Phase 0 CLI.** Every `chalybclip ingest|transcribe|detect|cut|variants|process` command keeps working; they just dual-write through the DB.
- **DB is the source of truth**; JSON files are read-models, regenerated on demand.
- **Tenancy isn't optional.** Service functions still take `tenant_id` as the first positional arg, but FastAPI handlers read it from the contextvars set by middleware. No `os.getenv("TENANT")`-style shortcuts. Repos enforce it independently — defense in depth.
- **Events from day 1.** Task 2 wires the event log before any feature work, so when detectors and the dashboard arrive they're already emitting transitions.
- **HTMX, not React.** Server-rendered + `hx-*` attributes. React energy goes into the Phase 3 marketing site.
- **Local vision = local CV libraries.** No LLM calls in §3-7. Multimodal scoring is a Phase 2 thing; Phase 1 only adds the *capability* (router + frame cache) so Phase 2 plugs in cleanly.
- **No new auth.** API tokens are static rows in `api_tokens` for Phase 1; Phase 3 introduces real auth + Cognito.
- **Free tier story stays viable.** Free-tier streamers get all four trigger types and face-detect smart crop; only multimodal scoring + vision-aware captions are Pro features (added Phase 2).

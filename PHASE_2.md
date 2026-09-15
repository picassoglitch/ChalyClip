## Phase 2 - quality first, then native reach (3 weeks)

**Goal:** Two unlocks in this order. (1) Clip quality goes from "loud + voice
trigger fired" to "the actual on-screen reaction" via a vision-LLM rescore on
top of the 4-signal heuristic. (2) Streamers stop renting Buffer - we publish
to TikTok and YouTube Shorts directly with proper OAuth refresh.

**Sequencing principle:** quality first, then reach. If clips still pick the
loudest moment instead of the best human reaction, multi-platform publishing
just distributes mediocre content faster. Schema migration 002 + a hard
budget governor land first as load-bearing infra; then the vision-LLM bet;
then publishers; then the operational layer (cost cards, webhooks).

**In scope:**
- Schema migration 002: webhook_subscriptions, OAuth refresh tokens on
  connected_accounts, platform external_ids on publish_jobs, candidate
  rescore columns, **per-tenant budget + quota columns** (Task 1 enforces).
- **Hard budget governor** (Task 1, lock-down): daily LLM USD ceiling per
  tenant, per-platform publish quotas, rescore concurrency cap, cooldown
  after repeated low-confidence rescore verdicts. Must land before any
  vision-LLM scope-up.
- `FrameStore` protocol around the in-memory frame cache (memory backend in
  Phase 2; S3 backend lands in Phase 3 without changing callers).
- Vision-LLM candidate rescoring: cheap-heuristic top-K -> premium vision
  rescore -> publish-ready ranking.
- Real face-emotion via vision LLM (drop the Phase 1 "neutral when face
  detected" placeholder).
- Vision-LLM smart crop + thumbnail upgrade.
- **Confidence-breakdown panel** on clip review (motion score, face
  presence, speaking intensity, reaction confidence, rescore delta) - the
  observability layer that lets us debug "why did the system pick THIS
  clip?" instead of guessing.
- TikTok Content Posting API publisher.
- YouTube Data API (Shorts) publisher.
- OAuth refresh + token rotation on connected_accounts.
- Dashboard cost-projection cards on /dashboard/llm-calls.
- Webhook dispatch (HMAC-signed, retry/give-up).

**Out of scope (Phase 3+):**
- Cloud migration (Aurora/Postgres, ECS, SQS-backed worker pool, S3 for
  clip + frame storage, Cognito auth, Stripe billing, marketing site).
- MCP server. *(Since shipped as Phase 3 #3 — lives in `chalybclip/mcp_server/`, run via `chalybclip mcp serve`.)*
- **SSE-based live pipeline progress.** Tempting but explodes scope on
  connection lifecycle, reconnect, event buffering, partial state sync,
  proxy quirks. The dashboard needs reliable scoring + publishing +
  economics first - "live orchestration theater" is the wrong investment
  at this phase.
- **OCR text-changed detector** + **scipy/librosa audio classifier.**
  Seductive but low ROI relative to the vision-LLM rescore. Webhooks +
  budget governance are higher-leverage at this phase. Revisit in Phase 3.
- Persona prompt iteration UI in-place editing.
- **Auto-publish from score thresholds.** The scoring system is still
  immature; auto-publishing bad clips at scale is the fastest brand-poison
  path. Phase 2 keeps humans in the approval loop without exception.

---

## Phase 2 hard rules (in addition to CLAUDE.md)

1. **Humans approve every publish.** Phase 2 does NOT introduce auto-publish
   from rescore-score thresholds. The progression that earns auto-publish
   is: rank -> human approve -> multi-publish -> collect engagement metrics
   -> learn what actually performs -> THEN consider thresholds. We're at
   step 1; auto-publish is years from "earned."
2. **Vision-LLM is opt-in, not default.** `--vision-rescore` flag, persona-
   level setting, dashboard toggle. Default off so free-tier streamers
   don't pay for premium tokens. (Phase 3 ties this to billing.)
3. **No LLM call leaves the router.** Same as Phase 1 - every multimodal
   call goes through `LLMRouter.complete_multimodal` so the budget governor
   and the cost log see it.
4. **Budget enforcement is process-wide.** A drained budget refuses the
   next call regardless of which tenant code path requests it. The
   governor is not advice; it's a hard ceiling that returns a typed
   `BudgetExceededError`.

---

## Exit criterion

```bash
# 1. Phase 1 still works:
chalybclip db init
chalybclip tenants add aldo "Aldo Villanueva"
chalybclip tokens issue --tenant aldo --scope full

# 2. Set the budget governor (Phase 2 makes this required for vision-rescore):
chalybclip tenants set-budget aldo --daily-usd 5.00 --rescore-cap 8

# 3. Run a VOD with vision-rescore enabled:
chalybclip process <vod_url> \
    --persona aldo_villanueva --tenant aldo \
    --vision-rescore
# -> top-K rescored by Claude vision; publish-ready clips ranked by both
#    audio/chat AND visual content (e.g. shock-face onset wins over a
#    one-frame loud spike).
# -> face_emotion populated with real labels (smile/laugh/shock/...).
# -> if the daily budget hits the ceiling mid-run, rescoring halts cleanly,
#    earlier clips keep their verdicts, log row says "budget exhausted".

# 4. Open the dashboard. Each clip page now shows:
#    motion score, face presence, speaking intensity, reaction confidence,
#    rescore delta -> heuristic-only score vs. vision-rescored score.

# 5. Connect TikTok + YT (OAuth) on the dashboard:
open http://localhost:8000/dashboard/connected-accounts
# -> "Connect TikTok" -> OAuth round-trip -> account row with refresh_token

# 6. Publish to TikTok directly (no Buffer):
chalybclip publish --tenant aldo
# -> tiktok publish_job: 200, external_id = TikTok video id, status=sent
# -> yt   publish_job: 200, external_id = YouTube video id, status=sent
# -> token expired mid-drain -> automatic OAuth refresh -> retry succeeds.

# 7. /dashboard/llm-calls shows projected EOM spend per purpose + model.

# 8. Webhook subscriber gets clip.published events:
curl -X POST http://localhost:8000/webhooks \
     -H "Authorization: Bearer tok_..." \
     -d '{"url": "https://my-handler.example/", "types": ["clip.published"]}'
# -> dispatcher posts HMAC-signed payloads; retries 5xx; gives up after N.
```

The acceptance demo: Aldo enables `--vision-rescore` (under his daily $
ceiling), the top clip on the dashboard is the actual moment he reacts on
camera (not the loudest two seconds), the confidence-breakdown panel shows
why it won, his TikTok + YT queues both fill from one approval click, and
his Discord bot subscribed to webhooks announces the new uploads.

---

## Task list (in order)

> Tasks 0-1 are the foundation. Tasks 2-6 are the quality bet. Task 7 is
> observability. Tasks 8-10 are native reach. Tasks 11-12 are the operational
> layer. Task 13 is polish.

### 0. Schema migration 002 (Day 1) - LOCK-DOWN TASK

After this merges, **schema is frozen at version 2 for Phase 2.** Any further
change is a numbered migration.

- [ ] `chalybclip/db/migrations/002_phase2.sql`:
  - `webhook_subscriptions(id, tenant_id, url, types_json, secret, status,
    created_at, last_dispatch_ts, failure_count)`
  - Extend `connected_accounts`: `refresh_token`, `expires_at`, `scopes_json`,
    `status` (`active` / `auth_failed`).
  - Extend `publish_jobs`: `external_url`, `platform_metadata_json`.
  - Extend `candidates`: `rescore_score`, `rescore_reason`, `rescore_model`
    (nullable; set by vision-rescore only when opted in).
  - Extend `tenants`: `daily_llm_budget_usd_micros` (NULL = unlimited),
    `daily_publish_limit` (NULL = unlimited), `rescore_concurrency_cap`
    (default 4).
  - Each new query target gets a composite `(tenant_id, ...)` index.
- [ ] Migration runner test: 001+002 cleanly, idempotent re-run.
- [ ] Tests: every new column round-trips; old code paths still work.

**Exit:** schema_version == 2 and `pytest tests/db` is green.

### 1. Hard budget governor (Day 2-3) - LOCK-DOWN TASK

Lock-down because every Phase 2 LLM/publish path adds load - without a
governor in place first, vision-rescore + multimodal smart crop will
silently burn money under usage spikes.

- [ ] `chalybclip/governance/budget.py`:
  - `BudgetGovernor(db, *, clock)` with `check_llm_spend(tenant_id, projected_usd_micros)`,
    `record_llm_spend(...)`, `check_publish_quota(tenant_id, platform)`.
  - Reads daily totals from `llm_calls` + `publish_jobs` filtered to today
    (UTC). Cheap because both tables are tenant-indexed.
  - Concurrency cap: in-process `asyncio.Semaphore` keyed by tenant_id,
    sized from `tenants.rescore_concurrency_cap`.
  - Cooldown: if last K rescore verdicts for this tenant returned scores
    below `low_confidence_threshold`, refuse new rescores for `cooldown_s`.
- [ ] `chalybclip/errors.py` adds `BudgetExceededError` + `QuotaExceededError`
      + `CooldownActiveError`.
- [ ] `LLMRouter.complete` / `complete_multimodal` consult the governor
      before issuing the call. Failure raises `BudgetExceededError`, which
      higher-up callers catch + emit `llm.budget_exhausted` event + halt
      cleanly (don't propagate to a 500).
- [ ] `chalybclip/publish/service.py`: `run_publish_jobs` checks the publish
      quota per (tenant, platform) before each enqueue dispatch.
- [ ] CLI: `chalybclip tenants set-budget <id> --daily-usd N --rescore-cap N
      --publish-limit N`.
- [ ] Tests:
  - llm_calls totals projecting tomorrow's first call across the ceiling
    -> `BudgetExceededError`.
  - Concurrency cap holds: 5 concurrent rescores against cap=2 -> 3 waiters.
  - Cooldown: 3 consecutive verdicts with score<0.3 -> next call refused
    inside the cooldown window.
  - Publish quota: 51st publish_job for `daily_publish_limit=50` -> refused.

### 2. FrameStore protocol (Day 4)

- [ ] `chalybclip/llm/frame_store.py`:
  - `FrameStore` protocol: `get`, `put`, `clear`.
  - `MemoryFrameStore` (existing FrameCache, renamed + adapted).
- [ ] `generate_variants(frame_store=...)` and the new vision-rescore service
      both accept the protocol.
- [ ] Tests: protocol conformance + drop-in for existing `FrameCache` tests.

### 3. Vision-LLM rescore wiring (Day 5-6)

- [ ] `chalybclip/detect/vision_rescore.py::rescore_candidates(...)`:
      samples 5 frames per top-K candidate, calls
      `router.complete_multimodal`, persists `rescore_score / rescore_reason
      / rescore_model` on the candidate row. Hits the `BudgetGovernor`
      first; honors the per-tenant concurrency semaphore.
- [ ] CLI flag: `chalybclip detect --vision-rescore`,
      `chalybclip process --vision-rescore`.
- [ ] Pipeline orchestrator opt-in: `process_vod(use_vision_rescore=False)`
      default; flag flips it.
- [ ] Tests: fake provider replays a rescored response; candidates are
      reordered by `rescore_score` when present, fall back to `score` else.
      Budget exhausted mid-run halts cleanly without losing the earlier
      verdicts.

### 4. Vision-LLM rescore prompt + schema (Day 7)

- [ ] `RescoreVerdict(score: float[0,1], face_emotion: FaceEmotion | None,
      reason: str)` + tight system prompt (~150 words, persona-agnostic
      scoring rubric).
- [ ] User prompt carries ts, candidate.reason, transcript snippet,
      chat snippet.
- [ ] The verdict's `face_emotion` propagates back into `visual_signals`
      for the matching second.
- [ ] Tests: shape + round-trip.

### 5. Vision-LLM smart crop + thumbnail upgrade (Day 8-9)

The Phase 1 columns (`smart_crop_box_json`, `thumbnail_frame_path`) stay -
only the picker logic changes.

- [ ] `chalybclip/clip/smart_crop_vision.py` + `thumbnail_vision.py`:
      vision-LLM pickers; fall back to Phase 1 face-detect heuristic on
      LLM error or budget refusal.
- [ ] CLI flag `--vision-crop`, persona-level opt-in.
- [ ] Tests: fake provider returns a crop box; cut_clips uses it; on LLM
      failure the heuristic still produces a clip.

### 6. Vision-driven face emotion (Day 10)

- [ ] `chalybclip/vision/face_emotion_vision.py`: replaces the Haar-Cascade-
      only path. The LLM gets one frame at the candidate ts, returns a
      `FaceEmotion` label.
- [ ] Caching: per-second emotion labels share the FrameStore so a 10-min
      VOD doesn't fire 600 separate calls.
- [ ] Tests: fake provider drives 5 frames; the visual_signals fan-in uses
      them; the strong-emotion edge in `detect_visual_candidates` actually
      fires.

### 7. Confidence-breakdown panel (Day 11)

The observability layer. Without this, "why did the system pick THIS clip?"
is unanswerable, and we'll waste cycles guessing instead of measuring.

- [ ] `/dashboard/clips/{id}` adds a "Why this clip?" panel:
  - **motion score** (from visual_signals.motion_energy averaged over the
    clip window)
  - **face presence** (frac of seconds with face detected)
  - **speaking intensity** (transcript word density during the clip)
  - **reaction confidence** (rescore_score if present, else None)
  - **rescore delta** (rescore_score - heuristic_score; positive = vision
    LLM thought this clip was *better* than the heuristic ranking)
- [ ] Persisted shape: a `clip_breakdowns` view joining clips +
      visual_signals + transcripts (read-only; no new table).
- [ ] Tests: rendered HTML carries the right numbers for a synthetic clip;
      panel degrades gracefully when no rescore is available.

### 8. TikTok Content Posting API publisher (Day 12-14)

- [ ] `chalybclip/publish/tiktok.py` - `TikTokClient` (httpx), implements
      `init_video_upload`, `upload_chunks`, `publish`. Same transient/fatal
      error split as `BufferClient`.
- [ ] `chalybclip/publish/service.py` dispatcher routes by
      `connected_accounts.platform`.
- [ ] OAuth round-trip on `/dashboard/connected-accounts`: "Connect TikTok"
      -> redirect -> callback writes refresh_token + expires_at.
- [ ] Publish quota check: `BudgetGovernor.check_publish_quota` runs
      before each TikTok call.
- [ ] Tests: respx mocks TikTok endpoints; happy path writes external_id
      + external_url; 401 with refresh available -> auto-refresh -> retry.

### 9. YouTube Data API (Shorts) publisher (Day 15-16)

- [ ] `chalybclip/publish/youtube.py` - YouTube Data API v3 with the
      resumable upload protocol.
- [ ] OAuth: same flow as TikTok behind a shared `OAuthFlow` helper.
- [ ] Tests: respx mocks Data API surface.

### 10. OAuth refresh + dispatcher integration (Day 17)

- [ ] `chalybclip/publish/oauth.py::refresh_if_expiring(account, *, db, http)`:
      called once per drain pass before posting; on 401 mid-call, force a
      refresh + one retry.
- [ ] Refresh failure flips `connected_accounts.status` to `auth_failed`,
      emits `connected_account.auth_failed` event.
- [ ] Tests: expired-on-the-clock refresh; 401 mid-call; refresh failure
      surfaces as `auth_failed`.

### 11. Dashboard cost-projection cards (Day 18)

- [ ] `/dashboard/llm-calls` roll-up cards: total this month, projected
      EOM, per-purpose breakdown, per-model breakdown, **budget headroom
      bar** (consumed / `daily_llm_budget_usd_micros`).
- [ ] HTMX `hx-trigger="every 60s"` keeps it live without any of the SSE
      complexity.
- [ ] Tests: rendered with seeded llm_calls + a tenant budget; the totals
      add up; budget bar shows the right percentage.

### 12. Webhook dispatch (Day 19-20)

- [ ] `chalybclip/webhooks/`:
  - `repos.py` CRUD on `webhook_subscriptions`.
  - `service.py::run_webhook_dispatch(tenant_id)` reads new event rows
    since `last_dispatch_ts`, filters by subscribed types, signs payloads
    with HMAC-SHA256(secret), posts to `url`.
  - Retry/give-up shape mirrors the publisher.
- [ ] FastAPI lifespan: another loop kicks `run_webhook_dispatch` every 30s.
- [ ] CLI: `chalybclip webhooks send --tenant <id>` for one-shot drains.
- [ ] REST: `POST /webhooks`, `GET /webhooks`, `DELETE /webhooks/{id}`.
- [ ] Dashboard: /dashboard/webhooks list + create form.
- [ ] Tests: respx mocks subscriber endpoints; HMAC signature in headers;
      5xx retried, fatal gives up after N.

### 13. Polish (Day 21)

- [ ] README: `--vision-rescore`, native publisher quick-start, webhook
      subscriber pattern, budget governor flags.
- [ ] PHASE_3.md backlog stub: cloud migration, MCP, SSE live progress,
      OCR, audio classifier, persona prompt iteration UI, **auto-publish
      from learned engagement metrics** (the earned-it endpoint).
- [ ] Run on a real Aldo VOD with rescore + native TikTok publish; iterate
      on prompts based on the verdicts.
- [ ] Performance check: a 1-hour VOD's vision-rescore finishes in <2 min
      with standard quality and <8 min with premium - and stays inside a
      $5 daily budget on a fresh tenant.

---

## Day-by-day rough plan

| Day | Focus |
|---|---|
| 1 | Task 0 - Schema migration 002 (LOCK-DOWN) |
| 2-3 | Task 1 - Hard budget governor (LOCK-DOWN) |
| 4 | Task 2 - FrameStore protocol |
| 5-6 | Task 3 - Vision-LLM rescore wiring |
| 7 | Task 4 - Rescore prompt + schema |
| 8-9 | Task 5 - Vision-LLM smart crop + thumbnail |
| 10 | Task 6 - Vision-driven face emotion |
| 11 | Task 7 - Confidence-breakdown panel |
| 12-14 | Task 8 - TikTok Content Posting API publisher |
| 15-16 | Task 9 - YouTube Shorts publisher |
| 17 | Task 10 - OAuth refresh + dispatcher integration |
| 18 | Task 11 - Dashboard cost-projection cards |
| 19-20 | Task 12 - Webhook dispatch (REST + worker + dashboard) |
| 21 | Task 13 - Polish, demo, README, PHASE_3 stub |

---

## New dependencies

**Zero expected.** Vision-LLM uses the existing Anthropic SDK; TikTok + YT
clients use existing httpx; HMAC uses stdlib `hmac`/`hashlib`; the budget
governor + FrameStore are pure Python. If a task discovers it genuinely
needs a new dep, that's a sub-PR before the feature work.

---

## Notes for the implementer

- **Schema lock at v2 after Task 0.** Any new column needs a 003+ migration.
- **Budget governor is wrapped around every external call.** LLM calls
  inside the router; publish calls inside the dispatcher. There's no
  "fast path" that bypasses it.
- **Vision-LLM is opt-in.** Default off everywhere - `--vision-rescore`
  flag, persona setting, dashboard toggle.
- **One publisher protocol.** Phase 1's `Publisher` shape (Buffer) stays;
  TikTok + YT are new implementations behind the same `service.py`
  dispatcher.
- **OAuth UX in dev** uses `http://localhost:8000/dashboard/oauth/callback`;
  Phase 3 (cloud) swaps in the real URL.
- **Webhook secret rotation** is Phase 3. Phase 2 stores one secret per
  subscription, generated at create time, returned once in the API
  response (mirrors api_tokens).
- **Tenant isolation everywhere**, same as Phase 1. Webhook subscriptions,
  OAuth tokens, rescore columns, budget rows - all tenant-scoped.
- **Humans approve every publish.** No automatic threshold-based
  publishing in Phase 2 under any circumstance. The earned-it path is in
  the PHASE_3 backlog stub.

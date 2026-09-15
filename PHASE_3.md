## Phase 3 - cloud + earned automation (backlog stub)

**Goal:** turn the local single-machine ChalyClip into a deployable
multi-tenant SaaS, then — *only after* engagement metrics show our scoring
predicts performance — earn the right to auto-publish high-confidence
clips at scale.

This is a backlog. The order isn't fixed; the user-facing direction is.

---

## Hard rules (carried over + extended from Phase 2)

1. **Auto-publish remains earned.** Phase 3 still does NOT enable
   threshold-based auto-publish until two things land:
     * (a) ingestion of engagement metrics (views, retention, CTR) per
       published clip, persisted alongside `publish_jobs`.
     * (b) a calibration pass showing rescore_score correlates with the
       engagement signal that matters - separately for each connected
       platform.
   Without (a) + (b), the scoring system is still guessing and any
   automatic distribution is a brand-poison risk.
2. **Brand safety guardrails before auto-publish.** A clip with NSFW /
   slur / out-of-context risk must never auto-publish. The vision-LLM
   already returns a `face_emotion`; we add a `brand_safety_verdict`
   that's a hard veto.
3. **Cloud migration is opt-in per environment.** Local SQLite stays the
   default for solo creators. The cloud path is a different deployment,
   not a forced upgrade.

---

## In scope

### Infrastructure & ops

- **Aurora / Postgres migration.** SQLite stays for local; production
  swaps to Postgres behind the same `Database` interface. The migration
  files are SQLite-flavored today; we add a parallel Postgres dialect
  generator and a one-shot `chalybclip db migrate-to-postgres` tool.
- **ECS / Fargate worker pool.** The Phase 2 publisher + webhook drains
  run as background tasks inside the FastAPI process; Phase 3 splits
  them into worker containers consuming SQS. Same `run_publish_jobs` /
  `run_webhook_dispatch` entry points.
- **S3-backed `FrameStore`.** The `MemoryFrameStore` becomes a fallback;
  the worker pool reads frames from S3 keyed by `(stream_id, ts)`.
- **Cognito-backed auth.** `api_tokens` keep working as a bypass for
  service-to-service; the dashboard's `/login` swaps in a Cognito
  hosted-UI redirect. Per-user (not just per-tenant) sessions.
- **Stripe billing.** `tenants.daily_llm_budget_usd_micros` becomes a
  per-plan default; the dashboard's "increase budget" flow lands as a
  Stripe checkout session.

### Distribution & integration

- **MCP server.** Expose the dashboard's read endpoints + a small set of
  state-transition tools as an MCP server so external agents (Claude
  Code, Cursor, etc.) can drive a tenant directly.
- **Webhook secret rotation.** Phase 2 stores one secret per
  subscription; Phase 3 lets the dashboard rotate without dropping
  in-flight deliveries. New secret coexists with old for a TTL window.
- **Native publishers: Instagram Reels.** Same `Publisher` protocol; the
  hard part is the Graph API approval cycle.

### Quality

- **Engagement metrics ingestion.** Per platform, pull views / retention
  / CTR for each `publish_jobs` row hourly. Store in a new
  `publish_metrics` table (one row per (job, fetch_ts)). The dashboard
  shows a per-clip outcome card.
- **Calibration loop.** A separate batch job correlates rescore_score
  with the engagement signal per platform; surfaces the calibration
  curve on the dashboard.
- **OCR text-change detector.** Deferred from Phase 2 because vision-LLM
  rescore + scene-cut already cover the high-leverage axes; revisit
  here once we want to spot title-card moments specifically.
- **scipy/librosa-driven audio classifier.** Spectral / pitch / formant
  features for laughter / shouting detection that the in-process RMS
  spike doesn't catch.
- **Persona prompt iteration UI.** Phase 2's `/dashboard/personas` is
  read + simple-edit; Phase 3 lets you A/B two persona prompts on the
  same clip set, measure variant CTR, and promote the winner.

### Earned automation (the destination, not the next step)

After (a) engagement ingestion + (b) calibration land, and only then:

- **Auto-publish from learned thresholds.** The dashboard's clip review
  page gains an "auto-publish high-confidence" toggle, gated by the
  brand-safety verdict. Defaults off forever; turning it on requires
  a per-tenant explicit opt-in *and* a minimum calibration confidence.
- **Per-platform auto-publish rules.** Different thresholds per
  TikTok / YT / Reels because each platform's engagement curve looks
  different.

### Scope-explosion magnets we're still NOT shipping

Per Phase 2's scope-adjusted spec, these stay out:

- **SSE-based live pipeline progress.** HTMX `hx-trigger="every 60s"` is
  good enough for the cards we have; SSE adds connection lifecycle +
  retry + reconnect + proxy-edge-cases + buffering for marginal UX
  upside. Revisit only if the worker fan-out demands it.
- **MCP-driven auto-approval.** Even when external agents can drive the
  tenant, approval stays human until calibration earns automation.

---

## Notes for the implementer

- **Migration v3 lock-down.** First Phase 3 task adds `publish_metrics`,
  `webhook_secret_versions`, and any auth columns we need. Schema gets
  frozen at v3 the same way v2 was.
- **Budget governor extends.** Phase 2's per-tenant daily LLM ceiling
  becomes per-plan; per-platform publish quotas grow into per-day +
  per-hour rate limits to honor TikTok's burst caps.
- **Confidence-breakdown panel** stays the observability hook. Engagement
  ingestion adds two columns to it: "actual views (24h)" and "predicted
  vs actual delta", so calibration is visible per clip.
- **Webhook subscribers don't change.** Phase 2's HMAC + type-filter shape
  is the contract; Phase 3 only adds secret-rotation under it.

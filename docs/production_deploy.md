# Production deploy — path from localhost to selling

This is the *path*, not the *recipe*. Each section describes what changes
between today's localhost-only dev setup and a paid-tier SaaS deployment,
plus which commits / modules already accommodate the change and which are
still local-only. Use this to evaluate any new code: does it land us closer
to or further from a deployable build?

The full step-by-step deploy guide ships at the end of Phase 0 (slice E).
This file is the contract every intermediate commit gets evaluated against.

## 1. Posture summary

| Area | Today (localhost dev) | Production target | Already ready? |
|---|---|---|---|
| HTTP server | uvicorn single-process, port 8000 | uvicorn behind nginx/Caddy + TLS | ✅ uvicorn is prod-grade; just front it |
| Database | SQLite file at `./chalybclip.db` | Postgres 15+ (managed: Neon/Supabase/RDS) | ⚠️ aiosqlite hardcoded; need driver abstraction |
| Storage (videos, clips, manifests) | `./out/` local dir | S3 / Cloudflare R2 | ❌ local-FS-only today; needs storage abstraction |
| Auth | Dashboard cookie + bearer tokens | Same + (eventually) email/password + OAuth | ✅ token model is already multi-tenant |
| LLM | Anthropic direct, cost-tracked | Same — Anthropic only | ✅ done |
| Whisper | faster-whisper, subprocess-isolated | Same on a GPU worker pool | ✅ subprocess isolation already in place |
| Diarization | pyannote-3.1, subprocess-isolated, skippable when absent | Same on a GPU worker pool | ✅ done (slice B) |
| Multi-tenancy | One tenant ("aldo") in single SQLite DB | N tenants in shared Postgres, row-level isolation | ✅ every query already filters on tenant_id |
| Job queue | FastAPI BackgroundTasks (in-process) | Redis + RQ (out-of-process workers) | ❌ in-process today; needs queue abstraction |
| Secrets | `.env` file | Environment vars from secret store (AWS SM / Doppler / 1Password) | ✅ all secrets already read from env |
| Billing | none | Stripe metered ($/clip processed or $/VOD hour) | ❌ not started |
| Observability | structlog → stdout | structlog → stdout (Loki / DataDog ingest) | ✅ structured logs already JSON-able |
| Backups | none | Postgres point-in-time + S3 versioning | n/a; falls out of choosing managed services |

## 2. Hard rules every commit must respect

Every PR / commit lands closer to or further from prod. These rules keep us
from accidentally adding tech debt that has to be unwound later:

1. **No hardcoded file paths.** Everything that touches disk reads
   `Settings.default_output_dir` (env: `CHALYBCLIP_DEFAULT_OUTPUT_DIR`) or a
   tenant-scoped subdir of it. The eventual S3 client will swap in here.

2. **No raw SQL in service code.** Service functions go through repos
   (`chalybclip.db.repos`). The repos hide aiosqlite vs asyncpg. When we swap
   drivers, only the repo layer changes.

3. **Secrets via `chalybclip.settings`.** Never `os.getenv` directly in
   business logic. The Settings class is the abstraction; in production the
   same class reads from environment vars set by the secret store.

4. **Background work goes through an injectable runner.** Today
   `app.state.pipeline_runner = default_pipeline_runner` runs in-process via
   `BackgroundTasks`. When we swap to RQ, only the runner changes. Service
   code stays identical.

5. **No `subprocess.run` to local binaries without explicit path resolution.**
   ffmpeg is located via `_ensure_ffmpeg_on_path()` in run.py for local dev;
   production containers bake ffmpeg into the image at a known path. Both
   paths converge on "ffmpeg is callable from `subprocess.Popen`."

6. **Tenant-isolation tests are mandatory for any new domain table.** If you
   add a table without a `test_*_isolated_per_tenant` test, the PR is
   incomplete.

7. **All long-running compute (Whisper, pyannote, ffmpeg cut, vision LLM) runs
   in a subprocess.** Hard crashes (CUDA OOM, segfault) must not be able to
   kill the dashboard process. Slice B will inherit this pattern from the
   existing Whisper worker.

8. **All publishing is per-tenant-credentialed and per-tenant-budgeted.**
   `BudgetGovernor.check_llm_spend` already gates LLM cost; the auto-publish
   worker (slice E) will gate publishing rate per tenant.

## 3. Storage abstraction plan (the biggest local-to-prod gap)

Right now, every artifact lives under `./out/<stream_id>/`. The dashboard
serves files directly with `FileResponse`. Production needs:

- VODs uploaded to S3/R2 via presigned multipart URLs (browser → S3
  directly, never through the dashboard process)
- Rendered clips uploaded to S3/R2 with public-read or signed download URLs
- The dashboard serves redirects to signed URLs, not file bytes

**Migration path** (not yet executed):

1. Introduce `chalybclip.storage.Storage` protocol: `put(key, data) → url`,
   `get(key) → bytes`, `presigned_upload(key) → url`.
2. Implement `LocalStorage` (today's behavior) and `S3Storage` (boto3).
3. `Settings.storage_backend` picks one. Default `local` in dev, `s3` in
   prod (env var `CHALYBCLIP_STORAGE=s3` + `CHALYBCLIP_S3_BUCKET=...`).
4. Every place that writes to `out/<stream_id>/` swaps to
   `storage.put(f"streams/{stream_id}/...")`.
5. Every place that reads switches from `FileResponse(Path)` to
   `RedirectResponse(storage.url(key))`.

This is a single coordinated commit when we tackle it (probably slice E or
just before public launch). Until then, keep file paths going through
`Settings` so the swap stays cheap.

## 4. Database abstraction plan

SQLite is sufficient through "small SaaS." Postgres becomes mandatory when:
- N tenants > ~100 with concurrent writes
- Aurora-style read replicas needed for analytics
- We want point-in-time-recovery beyond `cp chalybclip.db chalybclip.db.bak`

**Migration path:**

1. The repo layer already hides driver specifics. Each repo method uses
   parameter-substituted SQL (`?` placeholders for SQLite, `$1` for asyncpg
   — automatable with a small wrapper).
2. Add `chalybclip/db/postgres.py` mirroring the aiosqlite path but using
   asyncpg.
3. `Database(path_or_dsn)` decides driver based on scheme: `sqlite:///path`
   or `postgresql://user@host/db`.
4. Migrations: switch from raw SQL files to Alembic (Pydantic-style versioned
   migrations) when the schema-version count crosses ~10.

For the foreseeable future (Phase 0 + first paying tenants), SQLite + WAL is
plenty.

## 5. Worker pool plan

Today's `BackgroundTasks` runs pipelines inline in the FastAPI process.
Crash-isolated Whisper has already moved the heaviest step out. The full
worker pool plan:

| Job kind | Today | Production |
|---|---|---|
| `vod_process` (transcribe + detect + cut + variants) | FastAPI BackgroundTask | Redis + RQ, GPU worker pool (autoscale on queue depth) |
| `clip_render` (per-clip ffmpeg + thumbnail) | inline within vod_process | RQ CPU worker pool |
| `publish` | drain loop in dashboard process (P3) | RQ network worker pool |
| `metrics_ingest` | drain loop in dashboard | RQ scheduler |
| `webhook_dispatch` | drain loop in dashboard | RQ scheduler |

The dashboard process eventually does only request handling + scheduling.
All compute moves to dedicated worker containers.

## 6. Billing surface plan

Two metered axes:

- **VOD-hour** (storage + processing): $0.10–0.30/hr depending on tier
- **Clip-render** (output): $0.05 per rendered clip on free, included on
  paid tiers

Stripe metered billing with daily usage push from the dashboard. Per-tenant
`stripe_customer_id` on the `tenants` table; `tenants.daily_llm_budget_usd_micros`
already exists and would extend to include compute budgets.

Implementation lands in a separate phase (post-Phase-0). Not blocking the
voice-markers feature work.

## 7. Status checklist (kept current as we ship)

- [x] Anthropic-only LLM surface (commit `b3746f2`)
- [x] Whisper subprocess isolation (commit `3526332`)
- [x] Auto-free stale port on boot (commit `ef069ed`)
- [x] Data export tool for migrations / offboarding (commit `1d99971`)
- [x] Retroactive trigger window (commit `a60a7b6`)
- [x] Diarization subprocess worker, gracefully skippable (commit `a4b9c67`)
- [x] Speaker identity persistence + cosine-sim resolution (commit `4567927`)
- [x] Per-speaker trigger attribution + cooldown (commit `c8fc7fb`)
- [x] Brand kits schema + repo + per-speaker resolution (commit `2327d97`)
- [x] Per-kit custom_trigger_phrases wired into detect (commit `f8a68bf`)
- [x] Brand-kit CRUD dashboard pages (commit `54d0fd1`)
- [x] Brand-kit handle burned into clips (commit `a09a523`)
- [x] Caption preset library (commit `8651ed5`)
- [x] AI logo generation (commit `f50d66a`)
- [x] Brand-kit thumbnail compositing (commit `99cba4e`)
- [x] Retention sweeper (slice E.1)
- [x] Auto-publish dispatcher with undo window (slice E.2)
- [x] Inbox UI grouped by VOD/speaker (slice E.3)
- [x] Drive watcher polling architecture + FakeDriveClient (slice E.4)
- [ ] Real GoogleDriveClient + OAuth handshake routes (slice E.4 follow-up)
- [ ] S3/R2 storage abstraction (deferred — needed before public launch)
- [ ] Postgres driver (deferred — needed before scale)
- [ ] Redis + RQ worker pool (deferred — needed before scale)
- [ ] Stripe billing integration (deferred — needed before public launch)

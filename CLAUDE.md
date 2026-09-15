# ChalyClip — Claude Code Project Rules

## What this is

ChalyClip is a multi-tenant SaaS that turns a streamer's VOD into a multi-platform short-form clip pipeline. Voice cues + chat heat + audio peaks + visual signals detect clip-worthy moments; an LLM generates persona-flavored captions, hooks, and viral-moment selections (self-hosted open models by default — see rule 3); local Whisper handles transcription on the user's GPU.

**Read these first, in this order:**
1. `docs/chalybclip_spec.md` — full architectural spec (v0.5)
2. `PHASE_0.md` — concrete first-week tasks (the spike we're building)
3. This file — coding rules and conventions

The spec is the source of truth for *what* to build. PHASE_0.md is the source of truth for *what to build first*. This file is the source of truth for *how to write the code*.

---

## Hard rules (do not violate)

1. **Tenant isolation.** Every query against a domain table filters on `tenant_id` first. Every service function takes `tenant_id` as the first non-self argument. No exceptions. *(Phase 0 is single-tenant — but write the function signatures with `tenant_id` from day 1 even if it defaults to `"default"`.)*

2. **No business logic in route handlers.** FastAPI routes and MCP tool handlers both delegate to the same service functions in `chalybclip/<module>/service.py`. Routes are thin: parse inputs, call service, return response.

3. **All LLM calls go through `LLMRouter`.** Never call a vendor SDK or LLM HTTP endpoint directly outside `chalybclip/llm/`. The router handles cost tracking, retries, fallback, structured output validation. Bypassing it breaks billing and reliability. Default routing is the `openllm` / `openllm_vision` providers (any OpenAI-compatible self-hosted runtime — Ollama, vLLM, LM Studio — zero per-token cost, keyless). Anthropic remains a configured-but-unrouted provider; pointing a purpose back at it (or adding it as a fallback) is a deliberate config change in `config/llm.example.yaml`, never a code change. Do NOT route anything at a paid API by default.

4. **Idempotent pipeline steps.** Every step in the VOD pipeline must be safely re-runnable. If `transcribe` is run twice on the same stream, the second run is a no-op (or produces identical output). State is in DB + filesystem, not in memory.

5. **Structured output for LLM responses.** All LLM calls use Pydantic schemas via the router. No string parsing, no regex over LLM output, no "the LLM usually returns JSON-ish text and we hope for the best."

6. **Cost tracking is non-optional.** Every LLM call writes a row to `llm_calls` with `input_tokens`, `output_tokens`, `cost_usd_micros`, `tenant_id`, `purpose`. The router does this automatically — don't bypass it.

7. **IDs are ULIDs, not UUIDs.** Use `python-ulid`. ULIDs sort by creation time, which makes pagination and debugging much nicer. ID prefix per entity type: `str_` (stream), `clp_` (clip), `var_` (variant), `job_` (publish_job), `ten_` (tenant), `usr_` (user), `tok_` (api_token).

8. **Async by default.** All I/O is async. Use `asyncio`, `httpx.AsyncClient`, `aiosqlite` (Phase 0–2) / `asyncpg` or async SQLAlchemy (Phase 3+). The only sync code is CPU-bound work that genuinely doesn't await anything.

9. **Spanish-first content, but bilingual code.** Logs, comments, identifiers, error messages → English. Default content language for variant generation → Spanish (with EN as a configurable secondary). Persona voice prompts are written in the persona's target language.

10. **Never log secrets, OAuth tokens, full chat replays, or full VOD URLs.** Logs go to stdout/CloudWatch; treat them as semi-public. Use structured logging with fields, not f-string concatenation.

---

## Strong conventions

- **Pydantic v2 everywhere** for inputs, outputs, config, schemas. Single source of truth.
- **Typer or Click for CLI.** Every command supports `--json` for machine-readable output. Same JSON shape as the equivalent REST endpoint.
- **`pytest` + `pytest-asyncio`.** Tests live next to the module they test under `tests/<module>/`.
- **Dependency injection via constructor args**, not module-level singletons. Easier to test and swap.
- **Config via Pydantic Settings.** Environment variables override YAML; YAML overrides defaults. No `os.getenv` scattered through code.
- **Errors are typed.** Use custom exception classes (`ChalybClipError`, `LLMError`, `QuotaExceeded`, etc.). Don't catch `Exception` broadly except at process boundaries.
- **Format with `ruff` (which now does both linting and formatting).** Type-check with `mypy --strict` on `chalybclip/` (tests can be looser).

---

## Directory layout

```
chalybclip/
├── ingest/        # VOD download, chat replay, audio extraction, platform detection
├── transcribe/    # STT — AssemblyAI (default) + local faster-whisper provider
├── diarize/       # speaker diarization (pyannote subprocess, optional extra)
├── detect/        # voice/chat/audio/visual trigger fusion + vision-LLM rescore (scoring lives here — no separate score/ module)
├── clip/          # ffmpeg cut, reformat, captions, overlays, hybrid recorder
├── vision/        # local CV (PySceneDetect, OpenCV motion, face presence)
├── llm/           # LLMRouter + Anthropic provider client
├── variants/      # variant generator (uses llm.router)
├── branding/      # brand kits — colors, handles, thumbnail compositing
├── channels/      # channel auto-ingest — watch YT/Twitch/Kick for new VODs
├── drive/         # Google Drive folder watches (upload automation path)
├── jobs/          # pipeline job dispatch (DB-backed, restart-safe)
├── recovery/      # re-dispatch orphaned in-process ingest jobs after restart
├── retention/     # retention sweeper — deletes aged artifacts, bounds disk
├── publish/       # publish surfaces (Zernio-backed) + safe-trap scheduling
├── safety/        # anti-shadowban posting windows (spacing, caps, jitter)
├── metrics/       # per-platform engagement-metrics ingestion
├── governance/    # budget governor — LLM spend ceilings, caps, cooldowns
├── cost/          # LLM-spend projections for the dashboard cost cards
├── db/            # SQLite/Postgres persistence — migrations + repos
├── events/        # append-only event log
├── webhooks/      # HMAC-signed webhook dispatch to subscriber URLs
├── tenancy/       # tenant context + enforcement
├── api/           # FastAPI REST + HTMX dashboard + Pydantic schemas
├── mcp_server/    # MCP server — thin translation layer over the REST surface
└── integrations/  # external systems — chalyb (auth/tiers), zernio, storage, chalybobs
```

Still phase-pending (no directory yet): `auth/` (Phase 3+ — chalyb SSO covers it today), `billing/` (Phase 5), `workers/` (Phase 3+ GPU/CPU workers).

Each module has:
- `__init__.py` — re-exports the public surface
- `service.py` — business logic (the functions routes/MCP/CLI all call)
- `models.py` — Pydantic schemas (or SQLAlchemy if persistent)
- `<specific>.py` — concrete implementations

---

## Stack pin

- Python **3.11+** (match faster-whisper requirements)
- Open-model LLMs via any OpenAI-compatible runtime (Ollama/vLLM/LM Studio), spoken over `httpx` — no vendor SDK; `anthropic` SDK stays installed for the optional paid provider
- `faster-whisper` (CUDA build) for STT
- `yt-dlp` for VOD download
- `ffmpeg` (system binary) called via `subprocess.run` or `ffmpeg-python`
- `pydantic` v2
- `pydantic-settings` for config
- `typer` for CLI
- `httpx` for HTTP
- `aiosqlite` (Phase 0–2)
- `python-ulid`
- `pyyaml`
- `python-dotenv`

Dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`.

---

## Working with this codebase

When adding a new feature:
1. Read the relevant section of `docs/chalybclip_spec.md` first.
2. If the feature touches LLM calls, the path is always: schema in `chalybclip/llm/schemas.py` → method in `LLMRouter` → call from a service function. Not the other way around.
3. Write the service function with a typed signature before the route/CLI/MCP wrapper. The wrapper should be 5–10 lines.
4. Add a test that hits the service function with a fake LLM router (`tests/_fakes/fake_llm.py`).
5. Update `PHASE_X.md` if you've completed an exit criterion.

When debugging an LLM call:
- Check `llm_calls` table first — the call was logged whether it succeeded or failed.
- The `evidence` and `payload_json` columns have the input/output for replay.
- If the structured-output validation failed, the raw response is in the error log.

When unsure:
- Defer to the spec.
- Default to the simpler, more-tenant-isolated, more-idempotent option.
- Ask Aldo before adding a new top-level dependency.

---

## What this repo is NOT

- Not a video editor — clip windows are bounded edits, not full editing surface.
- Not a CRM — audience analytics stays per-platform.
- Not a stream recorder — we consume finalized VODs, never live audio/video.
- Not single-user — even Phase 0 writes its function signatures multi-tenant.

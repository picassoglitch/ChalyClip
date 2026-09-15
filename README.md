## ChalyClip

Multi-tenant SaaS that turns streamer VODs into multi-platform short-form clips.

**Status:** Phase 2 complete. The Phase 0 CLI (`chalybclip process …`) still works; on top of that the project now ships:

- Multi-tenant core: SQLite + tenancy contract, API tokens, dual-write through repos.
- 4-way detector fan-in (voice + chat heat + audio energy + visual signals).
- Local vision pipeline (PySceneDetect + OpenCV motion + Haar Cascade face presence).
- **Vision-LLM rescore** — promotes real on-screen reactions over loud-but-empty
  heuristic hits, gated by the budget governor. Opt-in via `--vision-rescore`.
- **Vision-LLM smart crop, thumbnail, and face emotion** with quiet fallback to
  the Phase 1 heuristic on any LLM error.
- **Hard budget governor** — daily LLM USD ceiling, per-platform publish quotas,
  rescore concurrency caps, low-confidence cooldown. Wraps every external-cost
  path; no fast-path bypass.
- **Confidence-breakdown panel** on the clip review page (motion / face presence /
  speaking intensity / reaction confidence / rescore delta) so "why did the
  system pick THIS clip?" stops being a guessing game.
- LLM router with vision capability (`complete_multimodal`) + `FrameStore`
  protocol around the in-memory frame cache (S3 backend pluggable in Phase 3).
- FastAPI REST API + HTMX dashboard with cookie-auth + cost projection cards.
- Multi-platform publishers behind one `Publisher` protocol: Buffer, **TikTok
  Content Posting API**, **YouTube Data API (Shorts)**, and **Instagram Reels
  (Graph API)**, with shared OAuth refresh + auth-failed lifecycle.
- **Webhook dispatch** — HMAC-signed delivery of event rows to subscriber URLs,
  type-filter wildcards, automatic disable after consecutive failures.

---

## Prerequisites

| Tool | Why | How (Windows) | How (macOS / Linux) |
|---|---|---|---|
| Python 3.11 | runtime (the optional `local-whisper` extra needs <= 3.11/3.12) | `uv python install 3.11` | `uv python install 3.11` |
| `uv` | Python + venv + dep manager | `pip install --user uv` | `pipx install uv` or `pip install --user uv` |
| ffmpeg | cuts + reformats clips | `winget install Gyan.FFmpeg` | `brew install ffmpeg` / `apt install ffmpeg` |
| AssemblyAI API key | transcription default (cloud STT, no GPU needed) | `setx CHALYBCLIP_ASSEMBLYAI_API_KEY ...` | `export CHALYBCLIP_ASSEMBLYAI_API_KEY=...` |
| NVIDIA GPU + driver | ONLY for the optional local-Whisper path (CPU works, just slow) | install the NVIDIA driver matching your GPU | same |
| Anthropic API key | variant generation via Claude | `setx ANTHROPIC_API_KEY ...` (new shell after) | `export ANTHROPIC_API_KEY=...` |

> Note: `uv` and `winget` install ffmpeg into a directory that's only added to PATH after a fresh shell. Open a new terminal before running `ffmpeg -version` to verify.

---

## Install

```bash
git clone <repo>
cd QuantorClipAI

# Create the venv with Python 3.11 specifically
uv venv --python 3.11 .venv

# Install runtime + dev deps (editable)
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
# (drop --python on macOS / Linux if your default python is the venv one)

# Optional extras — the default transcription provider is AssemblyAI (cloud),
# so neither is required for a normal install:
#   pip install 'chalybclip[local-whisper]'   # local faster-whisper STT (GPU/CPU),
#                                           # for CHALYBCLIP_TRANSCRIBE_PROVIDER=local
#   pip install 'chalybclip[diarize]'         # pyannote speaker diarization for the
#                                           # local path (pulls torch, ~2 GB; needs HF_TOKEN)

# Copy config templates
cp .env.example .env                                  # then add ANTHROPIC_API_KEY
cp config/chalybclip.example.yaml config/chalybclip.yaml
cp config/personas.example.yaml config/personas.yaml
cp config/llm.example.yaml      config/llm.yaml
```

### Verify the install

```bash
# Activate the venv first:
#   Windows:  .venv\Scripts\activate
#   *nix:     source .venv/bin/activate

ffmpeg -version                                        # binary on PATH?
chalybclip version                                       # entry-point installed?
pytest                                                  # smoke + unit tests

# Only if you installed the local-whisper extra:
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"   # 1+ means CUDA works
```

> The faster-whisper CUDA check uses **CTranslate2**, not PyTorch — `import torch` won't work because PyTorch isn't a dependency (unless you installed the `diarize` extra).

### First-run gotchas

- **Kick VODs need browser cookies.** Kick blocks anonymous `yt-dlp` with
  HTTP 403 / Cloudflare. Add this to your `.env`:
  ```
  CHALYBCLIP_COOKIES_FROM_BROWSER=chrome
  ```
  (or `edge` / `firefox` / `brave` / `chromium` — pick the browser
  you're already logged into Kick on). yt-dlp pulls cookies from that
  profile so the request authenticates. The browser must have visited
  Kick at least once. Twitch and YouTube generally work without this.
- **Use the venv's Python.** Activate `.venv` before running `chalybclip`
  or `python run.py`. If you installed the `local-whisper` extra, note
  that system Python 3.13/3.14 doesn't work with faster-whisper. The
  activated prompt should look like `(.venv) PS C:\...\QuantorClipAI>`.
- **Transcription is AssemblyAI by default.** Set
  `CHALYBCLIP_ASSEMBLYAI_API_KEY` in `.env` (cloud STT — word timestamps +
  diarization in one call, no GPU). To transcribe locally instead, run
  `pip install 'chalybclip[local-whisper]'` and set
  `CHALYBCLIP_TRANSCRIBE_PROVIDER=local`.
- **Local Whisper only: model downloads on first transcribe.** With the
  `local` provider, the first `chalybclip transcribe` (or first dashboard
  pipeline run) pulls the `medium` model (~770 MB). Don't kill the
  process while it's downloading.
- **Set the LLM budget governor before flipping vision-rescore on.**
  `chalybclip tenants set-budget aldo --daily-usd 5.00` keeps a runaway
  loop from burning premium tokens.

---

## Run

```bash
# Full pipeline against a real Kick / Twitch / YouTube VOD
chalybclip process "https://kick.com/<channel>/videos/<vod_id>" \
    --persona aldo_villanueva \
    --language es \
    --output-dir ./out
```

Per-step subcommands (handy when iterating on one stage):

```bash
chalybclip ingest      "<vod_url>" -o ./out
chalybclip transcribe  <stream_id> -o ./out
chalybclip detect      <stream_id> -o ./out
chalybclip cut         <stream_id> -o ./out
chalybclip variants    <clip_id>   --persona aldo_villanueva
```

Every command supports `--json` (machine-readable output) and `--force` (ignore the on-disk cache).

### Multi-tenant DB + dashboard (Phase 1+)

```bash
# 1. Initialize SQLite + apply migrations (current head: schema v2)
chalybclip db init

# 2. Create a tenant + issue an API token (the raw token is shown ONCE)
chalybclip tenants add aldo "Aldo Villanueva"
chalybclip tokens issue --tenant aldo --scope full
# → tok_01H...

# 3. Set the budget governor knobs before flipping vision-rescore on
chalybclip tenants set-budget aldo --daily-usd 5.00 --rescore-cap 8

# 4. Boot the API + HTMX dashboard
uvicorn 'chalybclip.api.app:create_app' --factory \
    --host 0.0.0.0 --port 8000

# 5. Open http://localhost:8000/ — dashboard access is via chalyb SSO
#    (GET /auth/sso?token=<jwt>); the in-app token login page was removed.
#    The tok_... API token is for Authorization: Bearer calls + MCP.

# 6. Drain pending publish_jobs manually (the API lifespan also kicks
#    a drain every 60s when started in production mode):
chalybclip publish --tenant aldo

# 7. Drain pending webhook subscriptions (every 30s in lifespan mode):
chalybclip webhooks send --tenant aldo
```

The CLI keeps writing JSON manifests to disk and dual-writes through the
DB when `CHALYBCLIP_DB_PATH` is set; the dashboard reads from the DB.

### Vision-LLM rescore (Phase 2 opt-in)

```bash
# Run the pipeline with vision-rescore on:
chalybclip process "<vod_url>" \
    --persona aldo_villanueva --tenant aldo \
    --vision-rescore
# → top-K candidates resampled by Claude vision, re-ranked by reaction
#   confidence; rescore_score / rescore_reason / rescore_model persisted.
# → if today's budget hits the ceiling, rescoring halts cleanly and
#   earlier verdicts stay; emits llm.budget_exhausted.
```

Subscribe a webhook to receive HMAC-signed event deliveries:

```bash
curl -X POST http://localhost:8000/webhooks \
     -H "Authorization: Bearer tok_..." \
     -d '{"url": "https://my-handler.example/", "types": ["clip.published", "clip.approved"]}'
# → response includes `secret` ONCE; subsequent reads omit it.
# → subscriber receives:
#     X-Chalybclip-Signature: hex(hmac_sha256(secret, body))
#     X-Chalybclip-Timestamp: 2026-...

# Rotate the secret without dropping in-flight deliveries (24h grace by default):
curl -X POST http://localhost:8000/webhooks/<sub_id>/rotate-secret \
     -H "Authorization: Bearer tok_..." \
     -d '{"grace_s": 86400}'
# → response includes the NEW secret once; the prior secret stays valid
#   until `prior_secret_expires_at`. List active prior secrets via
#   GET /webhooks/<sub_id>/secrets.
```

### MCP server (Phase 3 opt-in)

Run a local MCP server so external agents (Claude Code, Cursor, ...) can
drive the same tenant the dashboard does:

```bash
CHALYBCLIP_API_TOKEN=tok_... chalybclip mcp serve
# Or pass the token explicitly:
chalybclip mcp serve --token tok_...
```

The server is a thin translation layer over the REST surface — it adds no
new business logic. Tools include reads (`list_streams`, `get_stream`,
`list_candidates`, `list_clips`, `get_clip` with breakdown,
`list_personas`, `list_llm_calls`, `get_cost_projection`,
`get_calibration`) and state transitions (`update_clip_status`,
`publish_clip`). Tenant is resolved from the API token at boot; no
cross-tenant access. State-transition tools require `scope=full`.

See [docs/mcp_integration.md](docs/mcp_integration.md) for sample
config snippets to register the server with Claude Code, Cursor, and
Claude Desktop.

### Inspecting the publish queue

```bash
chalybclip queue list --tenant aldo
# →  Pending (3): per-platform rows, attempts, created_at
#    Recently sent (5): with external_id
#    Failed (1): with last_error
```

Read-only — no drain happens. The lifespan auto-drain runs every 60s
when the API server is up; this command is just for "did the worker
pick up the new job yet?" while iterating locally.

### Auto-drains (when the API is up)

`uvicorn 'chalybclip.api.app:create_app' --factory` boots with three
background loops by default:

| Loop | Cadence | What it does |
|---|---|---|
| publish | 60s | drains `publish_jobs` for every tenant via `run_publish_jobs` |
| webhook | 30s | delivers new events for active subscriptions via `run_webhook_dispatch` |
| metrics | 1h | pulls engagement stats per `sent` job via `run_metrics_ingest` |

Tests pass `enable_background_drains=False` so loops never spin during
test runs.

### Output layout (Phase 0)

```
out/<stream_id>/
├── manifest.json            # everything about this run, plus llm spend
├── stream.json              # ingest output
├── candidates.json          # detect output
├── clips_manifest.json      # cut output
├── llm_calls.jsonl          # one row per LLM call (tenant, purpose, tokens, cost)
├── source/
│   ├── video.mp4
│   ├── audio.wav            # 16 kHz mono PCM
│   └── transcript.json      # whisper output with word-level timestamps
└── clips/
    └── <clp_*>/
        ├── clip.mp4         # 9:16, libx264 + aac
        ├── metadata.json    # trigger, score, transcript snippet, window
        └── variants.json    # N persona-flavored captions
```

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | secrets + per-machine knobs (`ANTHROPIC_API_KEY`, `CHALYBCLIP_WHISPER_*`, `CHALYBCLIP_LOG_*`) |
| `config/chalybclip.yaml` | trigger phrases, fuzzy distance, clip windows, encoder settings |
| `config/personas.yaml` | persona voice prompts (the system prompt fragment fed to the LLM) |
| `config/llm.yaml` | provider/model selection per purpose, retry + pricing tables |

Env vars override YAML, YAML overrides defaults (see CLAUDE.md).

---

## Repo layout

```
.
├── CLAUDE.md                # coding rules + project context (read first)
├── PHASE_0.md               # the current phase's tasks
├── README.md                # this file
├── docs/
│   └── chalybclip_spec.md     # full architectural spec (v0.5)
├── config/                  # YAML templates → personal copies
├── chalybclip/                # source
│   ├── ingest/              # yt-dlp + ffmpeg audio extract
│   ├── transcribe/          # AssemblyAI (default) / faster-whisper (optional extra)
│   ├── detect/              # voice trigger fuzzy match
│   ├── clip/                # ffmpeg fast-cut + 9:16 reformat
│   ├── llm/                 # router + Anthropic provider + structured output
│   ├── variants/            # persona-flavored caption generation
│   ├── pipeline.py          # process_vod orchestrator + manifest
│   ├── settings.py          # pydantic-settings for env vars
│   ├── config.py            # YAML config models
│   ├── logging.py           # structlog setup
│   ├── errors.py            # typed exception hierarchy
│   ├── ids.py               # ULID minting (clp_, str_, var_, ...)
│   └── cli.py               # typer entry point
└── tests/
    └── test_end_to_end.py   # phase-0 exit-criterion smoke test
```

---

## Reading order for contributors / Claude Code

1. `CLAUDE.md` — coding rules + conventions
2. `docs/chalybclip_spec.md` — what we're building end-to-end
3. `PHASE_0.md` — what to build *right now*
4. The code

---

## Status (today)

Phase 0 + Phase 1 + Phase 2 complete. `pytest` runs 470+ tests in ~20s; `ruff` and `mypy --strict` are clean across 89+ source files. The pipeline runs against real VODs once you've set `ANTHROPIC_API_KEY` and `CHALYBCLIP_ASSEMBLYAI_API_KEY` (or installed the `local-whisper` extra for on-device STT).

Phase 3 backlog: cloud migration (Aurora/Postgres + ECS workers + S3 frame
storage + Cognito + Stripe billing), MCP server, SSE live progress, OCR text-
changed detector, scipy/librosa audio classifier, persona-prompt iteration UI
in-place editing, and — *only after* engagement metrics + brand safety guardrails
land — auto-publish from learned scores. See [PHASE_3.md](PHASE_3.md).

---

## License

Private. All rights reserved (for now).

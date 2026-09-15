# Modal Pipeline Worker — Deploy & Operations Runbook (Phase 2b)

Moves `process_vod` off the Railway web container onto Modal CPU workers.
Design + rationale: `docs/phase2_offload_design.md`. Dispatcher contract:
`chalybclip/jobs/modal.py`. Worker: `infra/modal_pipeline_app.py`.

## Prerequisites

- Phase 2a merged and deployed (clip artifacts persist to R2 — without it
  the worker's clips die with the container; the worker refuses to run).
- `CHALYBCLIP_OBJECT_STORAGE_*` vars set on Railway (done 2026-07-08).
- A Modal account + `pip install modal` + `modal token new` on the
  operator's machine.

## 1. Create the worker secret (once)

The worker is a full ChalybClip process minus the web server — it needs the
same env Railway has, minus web-only bits. Create ONE Modal secret named
`chalybclip-pipeline-env` containing:

| Variable | Value / notes |
| --- | --- |
| `MODAL_BEARER_TOKEN` | same string as Railway's `CHALYBCLIP_MODAL_TOKEN` |
| `DATABASE_URL` | Railway Postgres **public** URL (`DATABASE_PUBLIC_URL` from the Postgres service — the internal hostname doesn't resolve outside Railway). **Required**; the worker 500s without it. |
| `CHALYBCLIP_OBJECT_STORAGE_BUCKET` / `_ENDPOINT` / `_ACCESS_KEY_ID` / `_SECRET_ACCESS_KEY` / `_REGION` / `_PREFIX` / `_PUBLIC_BASE_URL` / `_PRESIGN_TTL_S` | copy from Railway. **Bucket required**; the worker 500s without it. |
| `ANTHROPIC_API_KEY` | copy from Railway (detect/viral + hooks + variants) |
| `CHALYBCLIP_TRANSCRIBE_PROVIDER` | copy from Railway (`modal` or `assemblyai`) |
| `CHALYBCLIP_MODAL_ENDPOINT_URL` / `CHALYBCLIP_MODAL_TOKEN` / `CHALYBCLIP_MODAL_MODEL` | copy from Railway — the pipeline worker calls the **whisper** app over HTTP like any other client |
| `CHALYBCLIP_TRANSCRIBE_AUDIO_VIA_OBJECT_STORAGE` | `1` — **worker-specific.** The WAV lives on the worker's ephemeral disk, so the provider uploads it to R2 and hands the whisper app a presigned URL instead of a `CHALYBCLIP_PUBLIC_URL`-signed one (which would 410). |
| `CHALYBCLIP_ASSEMBLYAI_API_KEY` | copy if provider=assemblyai |
| `CHALYBCLIP_COOKIES_TXT` (+ `CHALYBCLIP_COOKIES_TXT_B64=1` if base64) | copy from Railway — Modal egress IPs are datacenter IPs, YouTube bot-gates them the same way |
| `CHALYBCLIP_CHALYB_BASE_URL` / `CHALYBCLIP_CHALYB_ADMIN_TOKEN` | copy from Railway (usage reporting + post-run balance refresh) |
| `CHALYBCLIP_PUBLIC_URL` | copy from Railway (harmless; some URL-building paths read it) |
| `CHALYBCLIP_DEFAULT_OUTPUT_DIR` | optional; defaults to `/tmp/chalybclip-out` on the worker |

Do NOT set `CHALYBCLIP_JOB_DISPATCHER` in the secret — the worker calls the
runner directly and must never dispatch to itself.

CLI form (fill in values):

```bash
modal secret create chalybclip-pipeline-env \
  MODAL_BEARER_TOKEN=... DATABASE_URL=... \
  CHALYBCLIP_OBJECT_STORAGE_BUCKET=... CHALYBCLIP_OBJECT_STORAGE_ENDPOINT=... \
  CHALYBCLIP_OBJECT_STORAGE_ACCESS_KEY_ID=... CHALYBCLIP_OBJECT_STORAGE_SECRET_ACCESS_KEY=... \
  CHALYBCLIP_OBJECT_STORAGE_PUBLIC_BASE_URL=... \
  ANTHROPIC_API_KEY=... \
  CHALYBCLIP_TRANSCRIBE_PROVIDER=modal \
  CHALYBCLIP_MODAL_ENDPOINT_URL=... CHALYBCLIP_MODAL_TOKEN=... \
  CHALYBCLIP_TRANSCRIBE_AUDIO_VIA_OBJECT_STORAGE=1 \
  CHALYBCLIP_COOKIES_TXT="..." \
  CHALYBCLIP_CHALYB_BASE_URL=... CHALYBCLIP_CHALYB_ADMIN_TOKEN=... \
  CHALYBCLIP_PUBLIC_URL=https://chalybclip.chalyb.com
```

## 2. Deploy the worker

From the repo root (the image bundles the local `chalybclip/` + `config/`):

```bash
modal deploy infra/modal_pipeline_app.py
```

Copy the printed endpoint URL. Concurrency/cost ceiling (default 5
simultaneous runs, each 8 CPU): re-deploy with
`CHALYBCLIP_MODAL_PIPELINE_MAX_CONTAINERS=<n>` in the shell to change.

Re-deploy after any merge that touches pipeline code — the worker ships a
snapshot of the local source, it does not track Railway deploys.

## 3. Flip Railway

```bash
railway variables --set CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL=<printed URL> \
                  --set CHALYBCLIP_JOB_DISPATCHER=modal
```

(`CHALYBCLIP_MODAL_TOKEN` is already set from the whisper integration.)

## 4. Verify

1. Start a URL-VOD run from the dashboard. The progress card should tick
   through the six steps exactly as before (events come from the worker
   via shared Postgres).
2. `railway logs` should show `jobs.modal.dispatch` and NO
   `pipeline.step.*` work — the box only polls. CPU should stay flat.
3. `modal app logs chalybclip-pipeline` shows the actual run.
4. When it finishes: clips + thumbnails render in the dashboard (serving
   falls back to R2 keys `clips/<tenant>/<clip>/...` — the local files
   never existed on the web box).
5. Check `llm_calls` rows landed for the run (cost tracking unaffected).

## Rollback

Unset `CHALYBCLIP_JOB_DISPATCHER` (or set `in_process`) and redeploy —
dispatch reverts to in-process immediately. In-flight Modal runs finish on
their own and stay visible (their events keep flowing). The dispatcher
also falls back to in-process BY ITSELF at boot if the endpoint/token
vars are missing.

## Known behavior & follow-ups

- `upload://` and `live://` streams still run in-process on the web box
  (dispatcher routes them to the fallback — their source bytes only exist
  there). Follow-up: worker-pull of ingested sources.
- Operator-triggered editor renders / burn-in / downloads still run
  ffmpeg on the web box (user-triggered bursts; Phase 3 candidate).
- A re-dispatched (recovery) run on a fresh container redoes all steps —
  correct via DB idempotency, just not cached like a warm local re-run.
- Transient `work/<tenant>/<stream>/audio.wav` objects are deleted after
  each transcript; add an R2 lifecycle rule (expire `work/` after 7 days)
  as the backstop for crashed runs.
- The recovery sweeper re-dispatches through the configured dispatcher,
  so orphaned runs also go to Modal. Its 2h event-silence rule is
  unchanged; the dispatcher never marks a run failed on a mere poll
  timeout.

# Phase 2 — Storage/Compute Offload Design

**Problem.** The Railway web container serves FastAPI *and* runs the whole VOD
pipeline in-process. Measured 2026-07-08: ~9 sustained CPU cores during
`analyze_video` (OpenCV/PySceneDetect) plus ffmpeg cuts, with several runs in
flight — the dashboard lags whenever a pipeline is running. CPU capping was
explicitly rejected; the sanctioned fix is moving the compute off the box.

**Shape of the fix.** Two independent halves, shipped as three PR-sized
stages:

- **2a — R2 is the durable home of clip artifacts.** Pipeline outputs
  (clip MP4 + thumbnail) upload to the existing `ArtifactStore` (R2) as they
  are produced; the dashboard serves them from R2 whenever the local copy is
  gone. This makes the web box's disk a cache, not a source of truth — the
  precondition for running the pipeline on a machine whose disk evaporates
  when the run ends.
- **2b — the pipeline runs on Modal.** A `ModalJobDispatcher` (the stub in
  `chalybclip/jobs/modal.py` made real) POSTs the `PipelineKickoff` to a new
  Modal app (`infra/modal_pipeline_app.py`) that runs the *entire*
  `default_pipeline_runner` — download, analyze, transcribe, detect, cut,
  variants — against the same Railway Postgres and the same R2 bucket. The
  web box's remaining pipeline work is: enqueue + poll.

Everything reuses the pattern proven by `infra/modal_whisper_app.py`:
bearer-token `fastapi_endpoint`, Modal's 303→poll→200 protocol, secrets via
Modal Secret, config via `CHALYBCLIP_*` env vars.

---

## Stage A (PR 1) — Phase 2a: serve clip artifacts from R2

Today only the publish render (`clips/{tenant}/{clip}/clip_render_1080.mp4`)
reaches R2, and only at publish time. Everything the dashboard shows —
`clip.path` MP4, thumbnail JPEG — is local-only.

### Changes

1. **Key builders move to the storage package.**
   `artifact_key_for_clip()` currently lives in `api/routers/internal.py`;
   the pipeline can't import from `api`. New
   `chalybclip/integrations/storage/keys.py` with the full key family:
   `clip_media_key`, `clip_thumbnail_key`, `clip_render_key` — all under the
   existing `clips/{tenant_id}/{clip_id}/…` namespace. `internal.py`
   re-exports for backward compat.

2. **Offload after cut.** `offload_clip_artifacts(store, tenant_id, clips)`
   in `chalybclip/clip/offload.py`: for each clip, upload `clip.path` and
   `thumbnail_frame_path` to R2. Idempotent (`exists()` short-circuit),
   non-fatal (an R2 hiccup logs and continues — local serving still works),
   called from `pipeline.py` right after the cut step. No new pipeline step
   name, so the dashboard's six-step progress card is untouched.

3. **Serve-from-R2 fallback.** The two browser-facing endpoints
   (`/dashboard/clips/{id}/media` with `source=original`, and
   `/dashboard/clips/{id}/thumbnail`) keep serving the local file when it
   exists and otherwise 302 to `store.public_url(key)` /
   `store.presigned_url(key)` — same resolution logic as
   `resolve_publish_media_url`.

4. **Rehydration for byte-needing paths.** Renders, waveform computation,
   and downloads run ffmpeg on the web box against `clip.path`. New
   `ensure_local_clip(store, tenant_id, clip)` downloads the R2 copy back
   into place when the local file is missing; wired into
   `ensure_clip_rendered`, the download endpoint, and waveform.

5. **Retention deletes R2 too.** The clip-window sweep in
   `retention/service.py` deletes the R2 key family alongside the clip dir
   (reprocess already deletes the render key). R2 mirrors the clip row's
   lifecycle — no orphaned bucket objects.

Opt-in stays as-is: no `CHALYBCLIP_OBJECT_STORAGE_BUCKET` → behavior unchanged.

## Stage B (PR 2) — Phase 2b: ModalJobDispatcher

1. **Shared Modal HTTP client.** Extract the 303-poll protocol + error
   classification from `transcribe/providers/modal_whisper.py` into
   `chalybclip/integrations/modal_http.py`; the whisper provider and the new
   dispatcher both use it.

2. **Real `ModalJobDispatcher`** (`chalybclip/jobs/modal.py`):
   - Config: `CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL` (new) +
     `CHALYBCLIP_MODAL_TOKEN` (existing). Missing config → constructor raises →
     `create_app`'s existing defensive boot falls back to in-process.
   - `dispatch_pipeline`: same dedup as in-process (skip if
     `stream_id in active_stream_ids()`), `register(stream_id)`, spawn a
     poller task that POSTs `{auth_token, tenant_id, stream:
     stream.model_dump(), persona_id, language}` and follows the 303-poll
     protocol until terminal; `unregister` in `finally`.
   - Failure surfacing: a dispatch/HTTP failure emits the same top-level
     `pipeline.failed` event `default_pipeline_runner` would, so the
     dashboard explains instead of spinning.
   - **Hybrid fallback:** non-`http(s)` VOD URLs (`upload://`, `live://` —
     source bytes exist only on the web box) route to a wrapped
     `InProcessJobDispatcher`. URL VODs — the overwhelming majority and the
     entire measured CPU load — go to Modal.
   - No local semaphore for Modal dispatches: concurrency/cost is bounded on
     the Modal side (`max_containers`), not by queueing on the web box.

3. **`jobs.active` semantics preserved.** Registration still spans dispatch
   → completion, so the dashboard's in-flight detection and the disk
   reclaimers behave identically. If the web box restarts mid-run the Modal
   run keeps going and keeps writing step events to Postgres; the recovery
   sweeper's existing event-silence rules (2 h) apply unchanged, and a
   re-dispatch is safe because every step is idempotent (DB upserts; the
   worker's fresh disk just means a cold re-run).

4. **Worker-mode audio for Modal Whisper.** On the worker, the whisper
   provider can't hand Modal a Railway-signed audio URL (the WAV was never
   on Railway). New flag `CHALYBCLIP_TRANSCRIBE_AUDIO_VIA_OBJECT_STORAGE`
   (default off; on in the worker env): upload `source/audio.wav` to R2 and
   pass a presigned URL as `audio_url`. AssemblyAI needs nothing — it
   uploads bytes directly.

## Stage C (PR 3) — Phase 2b: the Modal pipeline app + rollout docs

1. **`infra/modal_pipeline_app.py`** — Modal app `chalybclip-pipeline`:
   - Image: `debian_slim(python 3.11)` + `apt ffmpeg` + deps from
     `pyproject.toml` + `add_local_python_source("chalybclip")`. CPU-only
     (transcription is remote; diarization stays disabled as in prod).
   - Function: `cpu=8`, `memory=16 GiB`, `timeout=8 h`,
     `max_containers` env-tunable (the real concurrency/cost ceiling),
     secret `chalybclip-pipeline-env`.
   - Endpoint: POST, verifies bearer, reconstructs the kickoff, and calls
     **the same `default_pipeline_runner`** used in-process — so events,
     `pipeline.failed` surfacing, source reclaim, base-fee charge, and
     balance refresh all behave identically. Output dir is the container's
     ephemeral disk; artifacts persist via Stage A's R2 offload; rows/events
     go straight to Railway Postgres (`DATABASE_URL` in the secret).
   - Brand kits are DB rows (colors/handles via drawtext; the Kick logo is
     rasterized from in-package SVG) — no web-box asset files needed.
2. **`docs/modal_pipeline_runbook.md`** — the secret's env checklist
   (DATABASE_PUBLIC_URL, Anthropic key, R2 vars, whisper endpoint vars,
   cookies for yt-dlp, `CHALYBCLIP_TRANSCRIBE_AUDIO_VIA_OBJECT_STORAGE=1`),
   deploy commands, Railway flip, verification steps, rollback.

### Rollout / rollback

1. Merge A → Railway deploy. R2 vars are already set in prod, so uploads +
   fallback serving activate immediately; behavior otherwise unchanged.
2. `modal deploy infra/modal_pipeline_app.py`, create the
   `chalybclip-pipeline-env` secret, set
   `CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL` on Railway.
3. Flip `CHALYBCLIP_JOB_DISPATCHER=modal`. Rollback at any point = unset it
   (in-flight Modal runs finish on their own; events keep flowing).

### Explicitly out of scope (follow-ups)

- **Editor/publish renders** (`ensure_clip_rendered`, burn-in, outro) still
  run ffmpeg on the web box — user-triggered bursts, small next to pipeline
  load. Offloading them is the natural Phase 3.
- **`upload://` and `live://` pipelines** stay in-process (their runners
  ingest local bytes before `process_vod`). Routing them to Modal needs a
  worker-pull of the ingested source (R2 or signed URL) — follow-up.
- **Modal-side step caching across retries** — a re-dispatched run redoes
  all steps on a fresh disk. Correct (idempotent), just not maximally cheap;
  persisting step JSONs to R2 could come later if retry cost ever matters.

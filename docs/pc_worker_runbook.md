# PC worker runbook — pipeline + LLM on operator hardware

Goal: zero paid APIs. The Quantor PC (RTX 4060) runs the full VOD pipeline
(ingest → local-GPU whisper → detect → cut → variants) and serves the
open LLM (Ollama); Railway keeps only the web app, Postgres and publish
scheduling. Modal, Anthropic and AssemblyAI are no longer in the hot path.

## How it fits together

```
Railway (web box)                          Quantor PC
─────────────────                          ──────────
ModalJobDispatcher ── POST kickoff ──────▶ chalybclip worker :8100
  (unchanged code;      via tunnel          ├─ default_pipeline_runner
   endpoint URL is                          │   ├─ yt-dlp ingest (home IP —
   the PC tunnel)                           │   │   no datacenter bot-gate)
                                            │   ├─ faster-whisper (CUDA)
dashboard "Generar 5",                      │   ├─ detect + hooks → Ollama
compose hooks ─────── /v1 proxy ─────────▶  │   └─ R2 upload + Postgres rows
  (OPENLLM_BASE_URL)   (bearer-authed)      └─ /v1/* → localhost:11434 (Ollama)
```

The worker speaks the exact HTTP contract of `infra/modal_pipeline_app.py`
(POST kickoff → 303 → poll → terminal JSON), so the web box's dispatcher
is reused as-is.

## PC side (one-time)

1. `scripts\make_worker_env.ps1` — writes `worker.env` (gitignored) from
   the Railway project's variables: public Postgres URL, R2 creds, the
   shared worker token (`CHALYBCLIP_MODAL_TOKEN`), Zernio key, signing
   secret. Plus PC-local settings (`CHALYBCLIP_TRANSCRIBE_PROVIDER=local`,
   `CHALYBCLIP_WHISPER_DEVICE=cuda`).
2. Ollama with `qwen2.5:7b-instruct-q4_K_M` pulled (config/llm.yaml points
   at that tag). For the vision purposes pull `qwen2.5vl:7b` too.
3. Tunnel (stable public URL for Railway → PC): Tailscale Funnel —
   `chalyb.com` DNS is on GoDaddy, so a named Cloudflare Tunnel isn't
   available without a DNS move.

   ```
   tailscale funnel --bg --https=443 http://127.0.0.1:8100
   ```

   One port covers both jobs AND the LLM: the worker proxies `/v1/*` to
   the local Ollama behind the same bearer token, so the GPU is never
   raw on the internet.

## Run it

```
powershell -ExecutionPolicy Bypass -File scripts\run_pc_worker.ps1
```

Health check: `GET /healthz` → `{"ok": true, "active_jobs": N}`.
`ok: false` means preflight would refuse runs (missing DATABASE_URL or
R2 bucket — regenerate worker.env).

To survive reboots, register it as a scheduled task at logon:

```
schtasks /Create /TN "ChalybClip PC Worker" /SC ONLOGON ^
  /TR "powershell -ExecutionPolicy Bypass -File C:\Users\picasso\Projects\QuantorClipAI\scripts\run_pc_worker.ps1"
```

## Railway side (the flip)

```
CHALYBCLIP_JOB_DISPATCHER=modal                     # the dispatcher is protocol-generic
CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL=https://<funnel-host>/
CHALYBCLIP_OPENLLM_BASE_URL=https://<funnel-host>/v1
OPENLLM_API_KEY=<value of CHALYBCLIP_MODAL_TOKEN>   # the /v1 proxy's bearer
```

`CHALYBCLIP_MODAL_TOKEN` already exists on Railway and is what the worker
validates. After a green week, the paid keys can be deleted:
`ANTHROPIC_API_KEY`, `CHALYBCLIP_ASSEMBLYAI_API_KEY`,
`CHALYBCLIP_MODAL_ENDPOINT_URL` (whisper) — see limitations first.

## Failure modes

- **PC off / tunnel down**: dispatch fails → `pipeline.failed` event on the
  stream (dashboard shows it); re-run once the worker is back. Railway-side
  hook generation falls back to deterministic titles (never blocks
  publishing). Scheduled Zernio posts are unaffected (Railway + R2 serve
  the media).
- **Run dies mid-flight**: the recovery sweeper's event-silence rules own
  it, same as Modal runs.
- **Duplicate dispatch after a lost response**: the worker dedupes per
  stream and redirects to the running job.

## Known limitations (follow-ups)

- `upload://` and `live://` sources still run in-process on Railway (their
  files exist only on the web box's disk) and use its transcribe provider —
  the last place a paid transcription can occur. Follow-up: worker-side
  whisper endpoint speaking the modal_whisper contract, or full source
  offload via R2.
- The worker's job ledger is in-memory: restarting the worker mid-run loses
  the poll (run keeps writing events; the sweeper reconciles).

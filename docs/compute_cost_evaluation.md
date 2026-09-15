# Compute after the GPU box — what Modal did, what replaced it, what it costs

Decision record, 2026-09-15. The self-hosted machine that ran the PC worker
(RTX 4060, Ollama, faster-whisper) is gone and the engines are being rebuilt
on GCP. This is the evaluation of what the "Modal" pieces actually do, whether
any of it is still needed, and the cheapest setup that keeps the product whole.

## 1. What Modal was, and what it is now

Modal appears in three places. Only one of them was ever the Modal cloud
service; the other two are a protocol name that outlived the vendor.

| Piece | What it does | Status |
| --- | --- | --- |
| `infra/modal_whisper_app.py` + `transcribe_provider="modal"` | GPU Whisper on Modal (slice O.44) | **Superseded.** The PC worker ran faster-whisper locally; on GCP transcription is a metered API (below). The Modal account is not needed. |
| `infra/modal_pipeline_app.py` | The whole VOD pipeline on Modal CPU containers (Phase 2b) | **Superseded** by `chalybclip worker`: the same kickoff → 303 → poll → terminal-JSON contract, first on the PC (`docs/pc_worker_runbook.md`), now on Cloud Run. |
| `CHALYBCLIP_JOB_DISPATCHER=modal` + `CHALYBCLIP_MODAL_*` env vars | The dispatcher that POSTs a run to *any* worker speaking that contract | **Still used, and vendor-neutral.** The hub's Terraform points `CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL` at the Cloud Run worker and `CHALYBCLIP_MODAL_TOKEN` is the shared bearer. The name is historical; nothing in it calls Modal. |

So: **Modal, the service, is not needed.** No account, no secrets, no deploys.
The two `infra/modal_*_app.py` files are kept only as reference for the
contract; they can be deleted once the Cloud Run worker has run a VOD end to
end.

What the dead box provided, and what has to be replaced, is not Modal but the
**GPU**: local Whisper, and Ollama serving `qwen2.5:7b` (text) and
`qwen2.5vl:7b` (vision) for every LLM purpose.

## 2. The workloads that need a home

| Workload | Where it ran | Needs GPU? | Per hour-long VOD |
| --- | --- | --- | --- |
| Pipeline: yt-dlp ingest, scene detection (OpenCV), ffmpeg cuts, variants, upload | PC worker | No, CPU-heavy (~9 cores measured on Railway) | ~20–30 min on 4 vCPU |
| Transcription with word timestamps | faster-whisper on the 4060 | Yes for speed; CPU works but slowly | 1 h of audio |
| Text LLM: `viral_detection`, `hook_generation`, `logo_generation`, `agent_decision` | Ollama qwen2.5:7b | Yes for a local model | ~15k input / 3k output tokens |
| Vision LLM: `vision_rescore`, `clip_scoring`, `variant_generation_visual`, smart-crop and thumbnail fallbacks | Ollama qwen2.5vl:7b | Yes for a local model | ~40–80 frames |

## 3. Options and cost

Prices are list prices at the time of writing; Cloud Run is us-central1,
CPU always allocated ($0.000018 per vCPU-second, $0.000002 per GiB-second).

### Pipeline CPU — Cloud Run worker (already in the hub's Terraform)

4 vCPU / 8 GiB for ~25 min ≈ **$0.13 per VOD-hour**, $0 when idle. Doubling
vCPUs halves the wall time for the same vCPU-seconds, so the size is about
turnaround, not cost. No alternative is cheaper without owning hardware.

### Transcription

| Option | Cost per audio hour | Notes |
| --- | --- | --- |
| **AssemblyAI** (`transcribe_provider=assemblyai`, implemented and tested, Task A1) | **~$0.17** ($0.15 + $0.02 speaker labels) | Minutes per hour of audio, diarization included, no GPU, no new code. |
| faster-whisper `small` int8 on the Cloud Run worker's CPU | ~$0.30–0.50 in vCPU-seconds | 1.5–2× realtime, so a 1 h VOD holds a worker slot for ~2 h; `medium` is slower still. Worse on both axes. |
| Groq `whisper-large-v3-turbo` | ~$0.04 | Cheapest, but the OpenAI-compatible transcribe provider is still a stub (`cloud_whisper.py`) and files over the size cap need chunking. Worth it past ~500 VOD-hours/month. |
| Cloud Run GPU (L4) | ~$0.84 per GPU-hour + 4 vCPU / 16 GiB | Only ~$0.05 per VOD in GPU time, but scale-to-zero needs a quota request and a 30–60 s model cold start on every wake; a min-instance is ~$600/month. |

**Pick: AssemblyAI.** Same money as CPU Whisper, ten times faster, and it is
the provider the Dockerfile already defaults to.

### Text LLM

| Option | Cost per VOD | Notes |
| --- | --- | --- |
| **Anthropic `claude-haiku-4-5`** (`anthropic` provider, already wired, cost-tracked) | **~$0.03** ($1 / $5 per Mtok) | Best structured-output reliability of the options; the router's pricing table meters it. Premium purposes on `claude-opus-5` ($5 / $25) are two calls per run at most. |
| Hosted open model through the `openllm` provider (Groq, OpenRouter) | ~$0.005 | Config-only switch (`CHALYBCLIP_OPENLLM_BASE_URL` + `OPENLLM_API_KEY`). Weaker JSON discipline; the router retries. |
| Ollama on a Cloud Run GPU | ~$0.40+ | Dozens of short calls per run keep the GPU instance alive for most of the pipeline; the worst option for sporadic traffic. |

**Pick: Anthropic Haiku 4.5 for `standard`, Opus 5 for `premium`.** The
difference to the cheapest hosted open model is two cents per VOD; the
difference in hook quality is the product.

### Vision LLM

Haiku 4.5 accepts images, so the same provider covers the multimodal
purposes: a 720p frame is ~1.2k tokens, so 60 frames per VOD ≈ **$0.07**. If
that ever matters, the vision purposes can be pointed at a hosted open VLM
via `openllm_vision`, or left to their deterministic fallbacks.

### The one cheaper path: your own GPU box again

The PC worker code is intact. A machine with a mid-range GPU, Ollama and a
Tailscale funnel brings API cost to $0 — and brings back the single point of
failure that took every engine down. Reasonable once volume justifies it;
not the default.

## 4. Totals

| Monthly VOD-hours | Worker CPU | AssemblyAI | LLM + vision | Total |
| --- | --- | --- | --- | --- |
| 0 | $0 | $0 | $0 | **$0** |
| 50 | ~$7 | ~$8.50 | ~$5 | **~$20** |
| 200 | ~$26 | ~$34 | ~$20 | **~$80** |

Against that, a GPU kept warm on Cloud Run is ~$600/month before it does any
work, and only starts to pay for itself somewhere past 500 VOD-hours/month.

## 5. What was changed to implement this

- `config/llm.example.yaml` (the router's default when `config/llm.yaml` is
  absent): every purpose routes to `anthropic`; `premium` is `claude-opus-5`
  with its pricing entry. The `openllm*` providers stay defined, unused.
- Hub Terraform (`picassoglitch/chalyb`, `infra/terraform`): the worker is
  4 vCPU / 8 GiB; `CHALYBCLIP_TRANSCRIBE_PROVIDER=assemblyai`; two new shared
  secrets, `assemblyai-api-key` and `anthropic-api-key`, injected as
  `CHALYBCLIP_ASSEMBLYAI_API_KEY` and `ANTHROPIC_API_KEY`; and an HMAC key on
  the engine's service account so `CHALYBCLIP_OBJECT_STORAGE_*` points boto3
  at the GCS media bucket through its S3 endpoint — without a bucket the
  worker refuses to run.
- Nothing Modal-specific was removed: the dispatcher and its env-var names
  are the worker protocol, and renaming them is churn with no saving.

## 6. Operator checklist

1. `printf '%s' "$KEY" | gcloud secrets versions add assemblyai-api-key --data-file=- --project=chalyb`
2. Same for `anthropic-api-key`.
3. Deploy (Cloud Build) so the revision picks up the new secret versions.
4. First run: watch the `usage_events` cost columns; the router and the
   AssemblyAI provider both meter into them.

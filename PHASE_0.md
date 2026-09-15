# Phase 0 — The Spike (1 week)

**COMPLETED** — kept for historical reference; see README for current status.

**Goal:** A single command — `chalybclip process <vod_url>` — that takes a Kick VOD URL and outputs a folder of vertical clips with cloud-LLM-generated Spanish captions, plus a JSON manifest.

**Out of scope for Phase 0** (these come in Phase 1+):
- Multi-tenancy enforcement (function signatures take `tenant_id` but it's hardcoded `"default"`)
- Database (filesystem + JSON manifest is the persistence)
- FastAPI dashboard
- MCP server
- Multiple trigger types (voice trigger only — chat heat, audio energy, visual signals come later)
- Multiple personas (one persona per run, passed as flag)
- Multiple platforms / publishers
- Auth, billing, quotas, webhooks

This is a spike. The point is to prove the spine works end-to-end on Aldo's actual streams, not to build the SaaS.

---

## Exit criterion

```bash
$ chalybclip process https://kick.com/aldovillanueva/videos/<vod_id> \
    --persona aldo_villanueva \
    --language es \
    --output-dir ./out \
    --json

# Produces:
./out/<stream_id>/
├── manifest.json              # Stream metadata, all clips, all variants
├── source/
│   ├── audio.wav              # 16kHz mono extract
│   └── transcript.json        # Whisper output with word timestamps
├── clips/
│   ├── clp_01H.../
│   │   ├── clip.mp4           # 9:16 vertical, captions burned
│   │   ├── metadata.json      # Trigger reason, score, transcript snippet
│   │   └── variants.json      # 5 LLM-generated caption variants
│   ├── clp_01J.../
│   └── ...
```

---

## Task list (in order)

### 0. Setup (Day 1, half day)
- [ ] `uv venv` and install deps from `pyproject.toml`
- [ ] Verify ffmpeg is in PATH: `ffmpeg -version`
- [ ] Verify CUDA is available for faster-whisper: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY`
- [ ] Copy `config/chalybclip.example.yaml` to `config/chalybclip.yaml`
- [ ] `pytest` runs (with no real tests yet — proves install works)

### 1. Ingest module (Day 1)
- [ ] `chalybclip/ingest/service.py::ingest_vod(tenant_id, vod_url, output_dir) -> Stream`
  - Use `yt-dlp` Python module (not subprocess — better error handling)
  - Detect Kick / Twitch / YouTube from URL pattern
  - Download VOD to `<output_dir>/<stream_id>/source/video.mp4`
  - Extract audio: `ffmpeg -i video.mp4 -ac 1 -ar 16000 audio.wav`
  - Return `Stream` Pydantic model: `{id, tenant_id, vod_url, source_video_path, source_audio_path, duration_s}`
- [ ] CLI: `chalybclip ingest <url>` calls service, prints JSON
- [ ] Smoke test: ingest a 10-min Kick VOD, verify outputs exist

### 2. Transcribe module (Day 2)
- [ ] `chalybclip/transcribe/service.py::transcribe(tenant_id, stream) -> Transcript`
  - Use `faster_whisper.WhisperModel("medium", device="cuda", compute_type="float16")`
  - Run on `stream.source_audio_path`
  - Return `Transcript` with word-level timestamps: `{stream_id, segments: [{ts, text, words: [{ts, text, prob}]}]}`
  - Save to `<stream_dir>/source/transcript.json`
- [ ] CLI: `chalybclip transcribe <stream_id>`
- [ ] Idempotency: if `transcript.json` exists, skip and return loaded version unless `--force`

### 3. Detect module — voice triggers only (Day 2)
- [ ] `chalybclip/detect/service.py::detect_voice_triggers(tenant_id, stream, transcript, config) -> list[Candidate]`
  - Phrase list from config: `["clipéalo", "clip this", "saca un clip", "guarda esto", "momento clip"]`
  - Fuzzy match (Levenshtein ≤ 2) over the transcript word sequence
  - For each match, emit `Candidate(timestamp, score, reason="voice", evidence={"phrase": ..., "transcript_snippet": ...})`
  - Score = `phrase_weight × confidence` (config-driven)
- [ ] Merge candidates within 30s into one (highest score wins, evidence union)
- [ ] CLI: `chalybclip detect <stream_id>` — prints candidates as JSON
- [ ] Phase 0 has only this one detector; chat/audio/visual come in Phase 1

### 4. Clip module (Day 3)
- [ ] `chalybclip/clip/service.py::cut_clips(tenant_id, stream, candidates, output_dir) -> list[Clip]`
  - For each candidate at time `t`:
    - Window: `[t - 30s, t + 15s]` (configurable per trigger type — defaults in config)
    - Cut: `ffmpeg -ss <start> -i <video> -t 45 -c copy <out>` (fast cut, may snap to keyframe — ok for spike)
    - Reformat 9:16: center-crop, optional face-detect (skip face-detect for spike — just center-crop)
    - Re-encode: `ffmpeg -i cut.mp4 -vf "crop=ih*9/16:ih,scale=1080:1920" -c:v libx264 -preset fast -c:a aac out.mp4`
  - Return `Clip` models, save `metadata.json` per clip
  - **Skip caption burning in Phase 0** — clips are clean, captions come back in Phase 1 once we have something to A/B against
- [ ] CLI: `chalybclip cut <stream_id>`

### 5. LLM router (Day 3–4)
- [ ] `chalybclip/llm/router.py::LLMRouter`
  - Phase 0 minimum: one method `complete(tenant_id, purpose, system, user, schema, quality="standard") -> T`
  - Loads provider config from `config/llm.yaml`
  - Reads API key from environment via `pydantic-settings`
  - Uses `anthropic` SDK; structured output via `tools=[{...}]` calling pattern OR via response prefill technique with JSON schema in system prompt
  - **Logs every call to `llm_calls.jsonl`** (Phase 0 doesn't have a DB yet — use a JSONL file in `<stream_dir>/llm_calls.jsonl`)
  - Handles 3 retries with exponential backoff
  - Provider fallback: stub for now, just Anthropic in Phase 0
- [ ] `chalybclip/llm/anthropic_provider.py` — concrete provider class
- [ ] `chalybclip/llm/schemas.py` — Pydantic models for structured outputs

### 6. Variant generator (Day 4)
- [ ] `chalybclip/variants/service.py::generate_variants(tenant_id, clip, persona, n=5) -> list[Variant]`
  - Loads persona config from `config/personas.yaml`
  - Builds prompt: persona voice prompt + transcript snippet + chat snippet (chat is empty for Phase 0) + clip metadata
  - Calls `LLMRouter.complete()` with `VariantBatch` schema (5 variants)
  - Returns variants, saves to `<clip_dir>/variants.json`
- [ ] Variant Pydantic model: `{id, language, caption, title_card_text, hashtags: list[str]}`
- [ ] CLI: `chalybclip variants <clip_id> --persona aldo_villanueva`

### 7. End-to-end orchestrator (Day 5)
- [ ] `chalybclip process <vod_url>` runs all 6 steps in sequence
- [ ] Each step is resumable: if `<stream_dir>/source/transcript.json` exists, skip transcribe; etc.
- [ ] Generate `manifest.json` at the stream root with full state
- [ ] `--json` flag prints the manifest to stdout when complete
- [ ] Smoke test on a real Aldo Villanueva Kick VOD: 1-hour stream → expect ~5–15 clips with captions

### 8. Polish (Day 5–7)
- [ ] Logging: structured logs via `structlog`, per-step timing
- [ ] Error handling: each step catches and re-raises typed errors with stream_id context
- [ ] One smoke test in `tests/test_end_to_end.py` using a tiny test VOD (5-min sample) — runs in CI
- [ ] README updated with actual install + run instructions based on what worked
- [ ] PR opened with all of the above

---

## Day-by-day rough plan

| Day | Focus |
|---|---|
| 1 | Setup + ingest module |
| 2 | Transcribe + voice trigger detection |
| 3 | Clip cutting + LLM router skeleton |
| 4 | Variant generator + first end-to-end run |
| 5 | Orchestrator + first real-VOD run |
| 6 | Polish, error handling, logging |
| 7 | Tests, README, demo to Aldo |

---

## Acceptance demo

You sit down with Aldo, run:

```bash
chalybclip process <his_latest_kick_vod_url> --persona aldo_villanueva --language es --output-dir ./demo
```

Expected:
- Stream takes ~ingest_time + ~12 min (Whisper) + ~clip_count × 5s (ffmpeg) + ~clip_count × 3s (LLM) for a 1hr VOD
- Output folder has 5–15 clips, each with 5 caption variants
- Captions actually sound like Aldo Villanueva (not generic)
- 9:16 framing is acceptable (will get smarter in Phase 2 with vision-guided crop)

If the captions are good and the clip windows feel right, Phase 0 ships and we move to Phase 1.

If captions are off, the issue is almost always the persona voice prompt — iterate on `config/personas.yaml` before changing code.

---

## Notes for the implementer

- **Don't optimize prematurely.** Phase 0 is a spike; readability beats performance.
- **Don't add a database.** Files + JSON manifests are deliberate. DB comes in Phase 1.
- **Don't add a web UI.** CLI only. Even the JSON output is mostly for piping into `jq` during dev.
- **Don't skip the structured output schemas.** The LLM router's value is partly in not parsing free-form text. Use Pydantic schemas from day 1.
- **Use `--force` flags for re-runs** — never delete the user's outputs without permission.
- **Hardcode `tenant_id="default"` in Phase 0,** but write the function signatures with the parameter so Phase 1 doesn't need to refactor every signature.

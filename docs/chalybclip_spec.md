# ChalyClip — System Specification (v0.5)

**Changes from v0.4:**
- Vision AI added as a fourth signal class alongside voice / chat / audio
- Two-stage scoring: cheap local vision heuristics for all clips, multimodal LLM scoring for top candidates (premium tier)
- Multimodal variant generator — LLM "sees" the clip when writing captions, hooks, and thumbnails
- Smart 9:16 cropping guided by vision rather than blind face-detect
- Auto-thumbnail picker
- Renamed §6 "LLM Provider Strategy" → "AI Provider Strategy" since it now includes vision models
- Vision adds modest cost; sized into the existing pricing tiers without restructuring them

---

## 1. Mission

Hosted SaaS that turns any streamer's VOD into a multi-platform clip pipeline.

**Updated principle (multimodal version):** Use the best signal for each job.
- **Whisper (local GPU)** — transcription
- **Heuristics (CPU)** — chat heat, audio energy, scene cuts, motion bursts, face/expression
- **Cloud text LLM (Anthropic Claude)** — variant captions, hooks, hashtags, viral-moment selection, agent decisions
- **Cloud multimodal LLM (Claude vision)** — clip-worthiness scoring, visual context for captions, smart cropping, thumbnail selection

Single-vendor by design (Anthropic only). The LLM router still supports an arbitrary fallback chain, so a future provider can be added later without code changes — just register a factory entry and reference it in the routing rules.

Multi-tenant SaaS from day 1. Public launch on free tier; paid tier designed and deferred. Aldo is user #1.

---

## 2. Block Diagram

```
[ User browser ] → [ Next.js web app ] → [ FastAPI API tier ]
                                               │
                                               ▼
                                          [ SQS queue ]
                                               │
                          ┌────────────────────┴────────────────────┐
                          ▼                                         ▼
                  [ GPU worker (Quantor) ]              [ CPU/API worker (cloud) ]
                  · Whisper transcription                · Trigger fusion
                  · PySceneDetect                        · Multimodal LLM scoring
                  · MediaPipe face/emotion               · Variant generation
                  · OpenCV motion                        · Smart crop / thumbnail
                                                         · Publish jobs
                          │                                         │
                          └────────────────────┬────────────────────┘
                                               ▼
                                  [ Aurora MySQL · S3 · Event log ]
                                               │
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
                  [ Dashboard ]         [ MCP Server ]        [ Webhooks ]
                                               │
                                               ▼
                                  [ Platform Publishers ]
```

Worker split is meaningful now: GPU worker on Quantor handles all the local-vision and STT work; cloud workers handle anything that's an API call (multimodal LLM, text LLM, publishing).

---

## 3. Single VOD Pipeline (now multimodal)

Seven steps. New step: **`analyze_video`**.

1. **`ingest`** — VOD + chat replay + extract audio.
2. **`transcribe`** — Whisper, word-level timestamps.
3. **`analyze_video`** *(NEW)* — local vision pass:
   - **PySceneDetect** — scene cuts (proxy for "something visually changed")
   - **MediaPipe** — face presence, expressions (laugh / shock / smile / neutral) per frame, sampled at 2 fps
   - **OpenCV motion energy** — frame-diff magnitude, identifies action bursts
   - **Optional OCR pass** (Tesseract or PaddleOCR) — detect on-screen text changes (donation alerts, browser content, code being typed)
   - Emits a per-second `visual_signal_track` with: `{scene_cut: bool, face_emotion: enum, motion_energy: float, text_changed: bool}`
4. **`detect`** — four detectors fuse into a single candidate stream:
   - Voice (transcript regex)
   - Chat heat (msg/sec baseline)
   - Audio energy (RMS spike)
   - **Visual** *(NEW)* — scene-cut + emotion-spike + motion-burst combined
5. **`score`** *(NEW)* — for top-N candidates (configurable, default top 20% by heuristic score), run multimodal LLM scoring (premium tier) — see §6.4.
6. **`cut`** — ffmpeg cuts; smart 9:16 crop guided by vision; captions burned; auto-thumbnail extracted.
7. **`generate_variants`** — variant generator now multimodal: LLM sees frames + transcript when writing hooks (premium tier) or text-only (standard tier).
8. **`notify`** — emit `clip.ready_for_review`.

All steps idempotent, resumable, individually addressable as MCP tools and CLI subcommands.

---

## 4. Personas (unchanged from v0.3)

User-defined personas with `voice_prompt`, target languages, routing tags, connected accounts. Vision adds value here too: art-content personas (like AARA) get *much* better captions when the LLM can see the work, not just read what was said about it.

---

## 5. Dashboard (small additions)

Same UX as v0.4 with these vision-related additions in `/review`:

- **Frame strip preview** — show the 5–10 frames the multimodal model used to score this clip, so you can see what it "saw"
- **Auto-thumbnail picker** — show the 3 best candidate thumbnails the vision model selected, click to choose
- **Smart-crop preview** — the 9:16 crop overlay shown over the original 16:9 frame, draggable to override
- **Visual evidence in trigger reason** — alongside chat snippets and transcript lines, show "scene cut + laugh detected at 14:32"

Same `/settings/llm` page — now also exposes "Use multimodal model for variants" toggle (Pro tier).

---

## 6. AI Provider Strategy (renamed from LLM Provider Strategy)

### 6.1 Multi-provider abstraction (extended)

Same `LLMRouter` from v0.4, now handles three modalities:

```python
# Text-only (variant generation, standard quality)
result = await router.complete(
    tenant_id, purpose="variant_generation",
    system=persona.voice_prompt, user=prompt,
    schema=VariantBatch, quality="standard",
)

# Multimodal (variant generation with visual context)
result = await router.complete_multimodal(
    tenant_id, purpose="variant_generation_visual",
    system=persona.voice_prompt, user=prompt,
    images=[frame_1, frame_2, frame_3],   # base64 frames or S3 URLs
    schema=VariantBatch, quality="premium",
)

# Multimodal scoring (clip-worthiness)
score = await router.score_clip(
    tenant_id, purpose="clip_scoring",
    images=sampled_frames, transcript_snippet=text,
    chat_context=chat_lines, schema=ClipScore,
)
```

### 6.2 Model strategy (extended)

| Quality | Use case | Default | Fallback |
|---|---|---|---|
| **Standard text** | Free + Pro variant generation | Claude Haiku-class | GPT-4o-mini-class |
| **Premium text** | Pro toggle + agent decisions | Claude Opus-class | GPT-4o-class |
| **Standard vision** *(NEW)* | Smart crop, thumbnail picking — runs on every clip in Pro tier | Claude Haiku-class (vision-capable) | GPT-4o-mini (vision) |
| **Premium vision** *(NEW)* | Multimodal scoring + multimodal variants — Pro premium toggle | Claude Opus-class (vision) | GPT-4o (vision) |

> *(OpenAI fallback removed in implementation — Anthropic is the only provider; see §"single-vendor" note below. The Fallback column is historical.)*

### 6.3 Two-stage scoring strategy (cost-controlled)

This is the key design pattern that makes vision affordable:

**Stage 1 — Heuristic scoring (free, runs on every candidate):**
- Voice trigger: transcript phrase match × confidence
- Chat heat: msg/sec ratio vs baseline
- Audio energy: RMS ratio vs baseline
- Visual heuristics (NEW): scene-cut weight + face-emotion weight + motion-burst weight
- Composite score = weighted sum

**Stage 2 — Multimodal LLM scoring (paid tier, runs on top-N candidates only):**
- Sample 3–5 frames from candidate window
- Send to vision LLM with: frames + transcript snippet + chat snippet + persona context
- LLM returns structured `ClipScore`:
  ```json
  {
    "clip_worthiness": 0.82,
    "reason": "Streamer reacts with shock to chat dropping a tweet showing the news",
    "best_thumbnail_frame_idx": 2,
    "suggested_hook": "Wait — they actually said that?",
    "suggested_crop": {"x": 0.3, "y": 0.1, "w": 0.4, "h": 1.0},
    "persona_fit": {"aldo_villanueva": 0.9, "chalyb_academy": 0.7, "aara": 0.1, "quantor": 0.2}
  }
  ```
- Final candidate score = `0.4 × heuristic + 0.6 × multimodal`

**Cost control:** only top-N (default top 20%) of heuristic candidates get stage-2. Free tier never gets stage 2. This bounds vision costs to a predictable fraction of total candidates.

### 6.4 Where vision shows up

| Feature | Vision used? | Tier |
|---|---|---|
| Trigger detection (visual signals) | Local CV (PySceneDetect, MediaPipe) | All tiers |
| Clip-worthiness scoring | Cloud multimodal LLM | Pro tier (premium toggle) |
| Smart 9:16 cropping | Local face-detect (free) → Cloud vision (Pro) | All tiers, better in Pro |
| Auto-thumbnail picker | Local heuristics (free) → Cloud vision (Pro) | All tiers, better in Pro |
| Variant captions with visual context | Cloud multimodal LLM | Pro tier (premium toggle) |
| First-frame title card selection | Cloud vision | Pro tier |

### 6.5 Cost rough order

Per stream, vision adds (premium tier):
- Stage-2 scoring: ~5 candidates × 3 frames × ~1.5K tokens/frame = ~22K vision tokens
- Multimodal variants: top 10 approved clips × 3 frames × 4 personas = ~180K vision tokens
- Estimated cost: well under $1/stream at current Haiku/4o-mini-class vision pricing, a few dollars/stream at premium model pricing

This fits the Pro tier ($19–29/mo) for normal usage. Heavy users brushing up against the limit get the same fair-use behavior as text tokens.

*(Verify pricing before launch — vision token costs shift more than text. Architectural point: two-stage scoring keeps total cost predictable regardless of provider rates.)*

### 6.6 Reliability + caching

- **Frame caching:** sampled frames stored in S3 keyed by `(stream_id, timestamp)`. Multiple LLM calls (scoring, captioning, thumbnail picking) on the same clip reuse the same frame URLs — sent as references rather than re-uploaded.
- **Same retry / fallback / circuit-breaker logic** as v0.4 text router.
- **Vision-specific:** if the multimodal call fails entirely, fall back to text-only generation on the transcript. Clip still ships, just slightly lower quality.

---

## 7. AI-Agent Layer (mostly unchanged)

Same MCP server, REST API, event log, webhooks. Two small additions:

- **`get_clip_frames(clip_id)`** — agent can fetch sampled frames if it wants to do its own multimodal reasoning
- **`rescore_clip(clip_id, model)`** — agent can request a re-score with a specific model (useful for the alter-ego to override low-confidence scores)

---

## 8. Schema (small additions)

Same as v0.4 plus:

```sql
CREATE TABLE visual_signals (
  stream_id TEXT NOT NULL REFERENCES streams(id),
  tenant_id TEXT NOT NULL REFERENCES tenants(id),
  ts_offset_s REAL NOT NULL,
  scene_cut BOOLEAN,
  face_emotion TEXT,             -- neutral | smile | laugh | shock | anger | sad
  motion_energy REAL,
  text_changed BOOLEAN,
  PRIMARY KEY (stream_id, ts_offset_s)
);

ALTER TABLE clips ADD COLUMN multimodal_score REAL;
ALTER TABLE clips ADD COLUMN multimodal_score_model TEXT;
ALTER TABLE clips ADD COLUMN multimodal_evidence TEXT;  -- JSON: best frames, suggested crop, etc.
ALTER TABLE clips ADD COLUMN thumbnail_frame_path TEXT;
ALTER TABLE clips ADD COLUMN smart_crop_box TEXT;       -- JSON: {x, y, w, h} normalized

-- frames cached for LLM reuse
CREATE TABLE clip_frames (
  clip_id TEXT NOT NULL REFERENCES clips(id),
  frame_idx INT NOT NULL,
  ts_offset_s REAL NOT NULL,
  s3_path TEXT NOT NULL,
  PRIMARY KEY (clip_id, frame_idx)
);
```

`llm_calls` table from v0.4 stays the same — vision calls log under `purpose=clip_scoring | variant_generation_visual` with their own token counts.

---

## 9. Repository Layout (small additions)

```
chalybclip/
├── ...
├── chalybclip/
│   ├── llm/                       # Now handles vision too
│   │   ├── router.py
│   │   ├── providers/
│   │   │   ├── anthropic.py       # supports text + vision (only vendor)
│   │   ├── frame_cache.py         # NEW
│   │   └── ...
│   ├── vision/                    # NEW
│   │   ├── scene_detect.py        # PySceneDetect wrapper
│   │   ├── face_emotion.py        # MediaPipe wrapper
│   │   ├── motion.py              # OpenCV motion energy
│   │   ├── ocr.py                 # Optional Tesseract/PaddleOCR
│   │   └── frame_sampler.py       # Smart frame sampling for LLM
│   ├── clip/
│   │   ├── engine.py
│   │   ├── smart_crop.py          # NEW: vision-guided 9:16
│   │   └── thumbnail.py           # NEW: auto-thumbnail picker
│   ├── detect/
│   │   ├── voice_triggers.py
│   │   ├── chat_heat.py
│   │   ├── audio_energy.py
│   │   ├── visual_signals.py      # NEW
│   │   └── scorer.py
│   ├── score/                     # NEW
│   │   └── multimodal_scorer.py
│   └── ...
```

---

## 10. Hosting & Deployment (essentially unchanged)

Same AWS layout. Vision additions don't change infrastructure shape — the local vision libraries (PySceneDetect, MediaPipe, OpenCV) run on Quantor alongside Whisper, the multimodal LLM is just additional API calls from the worker tier.

**Quantor GPU load with vision added:**
- Whisper medium ES on a 2hr VOD: ~12 min on RTX 4060
- MediaPipe face/emotion at 2 fps on 2hr: ~5 min (CPU + GPU mixed)
- PySceneDetect: ~2 min (CPU-bound)
- OpenCV motion: ~3 min (CPU-bound)

Total ~20 min per 2hr VOD on Quantor. Comfortably within overnight batch capacity for the free tier.

---

## 11. Pricing & Quotas (vision folded in)

| | Free | Pro | Scale |
|---|---|---|---|
| VODs | 1/week, max 2hr | Unlimited, max 8hr | Unlimited |
| Clips per VOD | 10 | Unlimited | Unlimited |
| Personas | 2 | Unlimited | Unlimited |
| Connected accounts | 4 | 20 | Unlimited |
| API/MCP | Read-only | Full read+write | Full + custom prompts |
| Auto-publish | ❌ | ✓ | ✓ |
| **Visual triggers (local CV)** | ✓ | ✓ | ✓ |
| **Multimodal scoring** | ❌ | ✓ (premium toggle) | ✓ |
| **Multimodal variants** | ❌ | ✓ (premium toggle) | ✓ |
| **Smart crop / auto-thumb** | Basic (face-detect only) | Vision-guided | Vision-guided |
| Monthly text LLM tokens | ~50K | ~2M | BYO keys |
| Monthly vision LLM tokens | 0 | ~500K | BYO keys |
| Suggested price | Free | $19–29/mo | $99+/mo |

Vision tokens are tracked separately from text tokens in `llm_calls` — costs are just enough different (vision input tokens are pricier per token but you use fewer of them) that separate budgets give cleaner control.

**The free tier story:** still ships great clips. Local vision (scene cuts, face/emotion, motion) catches the moments. It just doesn't get the LLM holistic scoring or vision-aware captions. That's a meaningful upgrade story for paid.

---

## 12. Phasing (vision folded into Phase 1 + 2)

**Phase 0 — Spike (1 week)**
- VOD download + Whisper + voice trigger + ffmpeg cut + cloud-LLM captions
- CLI only
- **Exit:** clips with cloud-LLM captions.

**Phase 1 — Multi-tenant core + local vision (3 weeks)**
- Tenancy schema
- All four trigger types (voice / chat / audio / **visual**)
- Local vision pipeline (PySceneDetect, MediaPipe, OpenCV)
- LLMRouter (text + vision-capable)
- Persona-aware variant generator (text-only by default; multimodal opt-in)
- FastAPI REST + HTMX dashboard
- Single publisher: Buffer
- **Exit:** Aldo's pipeline live with all four signal types feeding clip detection.

**Phase 2 — MCP + native publishers + multimodal scoring (3 weeks)**
- MCP server
- Native TikTok + YT Shorts adapters
- **Two-stage scoring (heuristic → multimodal LLM)**
- **Smart crop + auto-thumbnail with vision**
- Multi-persona / multi-account routing
- Analytics + leaderboards
- Webhooks
- **Exit:** premium clip quality features all working; alter-ego agent reviews clips with vision context.

**Phase 3 — Cloud migration (3–4 weeks)**
- SQLite → Aurora; FastAPI on ECS; S3, SQS, Cognito
- Quantor → SQS GPU worker (Whisper + local vision)
- Onboarding flow + marketing site
- Free tier quota enforcement (text + vision tokens)
- **Exit:** `chalybclip.chalyb.com` live, beta users.

**Phase 4 — Public launch.**
**Phase 5 — Paid tier (Stripe, Pro/Scale flags, BYO keys).**

---

## 13. Open Decisions (revised)

1. **Vision provider default** — Anthropic Claude only (Haiku for standard, Opus for premium). Earlier drafts kept an OpenAI fallback; removed for a single-vendor surface. The router still supports adding another provider later without code changes.
2. **OCR worth it?** OCR adds another local-CV dependency and slows the pipeline. Useful for catching donation alerts and on-screen text moments. My take: skip in Phase 1, add in Phase 2 if real streams show signal we're missing.
3. **Frame sampling rate** — 2 fps for emotion detection, 5 frames per candidate for LLM scoring. Sane defaults; tune from data.
4. **Free tier basic smart crop** — face-detect only, no LLM. Confirms that free tier still feels good even without vision LLM.
5. **Premium quality toggle granularity** — per-persona (carry from v0.4). Vision toggle separate from premium-text toggle? My take: combine them into one "Premium AI" switch per persona; complexity-vs-control tradeoff favors simplicity.
6. **Carry-overs:** product naming, AARA on TikTok, alter-ego scope, AWS region, watermark.

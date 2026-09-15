# Competitor feature parity — backlog (2026-05-13)

Reference: screenshots Aldo dropped of a Spanish-localized clipping tool
("Convertir video largo en cortos" flow). The competitor's pipeline UX
exposes nine features that ChalybClip either lacks today or has in a
weaker form. This doc triages each one to a future slice so nothing
gets lost while Phase 0 lands.

**This is a backlog, not a plan.** The numbered slices below ship after
slice D.4 (thumbnail compositing) closes. None of them block the spec
exit criteria for Phase 0.

---

## What the competitor shows

Each row maps a competitor feature → status in ChalybClip today → target
slice. Effort estimates: S = half-day, M = 1–2 days, L = 3+ days.

| # | Competitor feature | ChalybClip today | Target | Effort | Notes |
|---|--------------------|----------------|--------|--------|-------|
| 1 | Pre-trim range slider over VOD timeline before processing ("26 minutos seleccionados") | Whole VOD is always processed | F.1 | M | Massive cost saver — skips diarization + transcribe + detect for unselected ranges. Reuse `clip/breakdown.py` window math. |
| 2 | Visual caption preset cards (rendered preview, not text dropdown) | Dropdown with text labels (slice D.2) | F.2 | S | Render the four presets in `branding/captions.py` as inline SVG previews. Pure template change, no backend work. |
| 3 | Duration policy picker (Auto / <60s / 60-90s / 90-180s / custom) | Each candidate has its own implied window from triggers | F.3 | M | Maps to `clip/breakdown.py` `min_seconds` / `max_seconds`. Need a per-VOD override on top of brand-kit defaults. |
| 4 | Pre-flight time + cost estimate ("Tiempo necesario 26m", "60min" credit cost) | Cost log exists but isn't surfaced pre-flight | F.4 | S | Already have token estimates in `llm.router`; just need a `/dashboard/streams/estimate` endpoint that previews steps × current rates. |
| 5 | Upload progress bar with percentage ("Cargando tu video... 37.9%") | Form upload, no progress | F.5 | M | Use `tus.io` or streamed `multipart/form-data` with HTMX progress event. Backend already accepts uploads via `_stash_upload_to_tmp`. |
| 6 | Conversion progress with ETA ("Convirtiendo... 6%, quedan 9 minutos") | Step events visible on dashboard but no % or ETA | F.6 | M | Pipeline already writes `step.start` / `step.complete` events. Add a `progress` field to step events and a moving-average ETA estimator. |
| 7 | "You can close this page now" UX — truly background pipeline | Pipeline IS background (`background_tasks.add_task`) but the UI doesn't communicate it | F.7 | S | Template copy + a "you'll be notified" banner. Backend already supports it. |
| 8 | In-app task center / notifications panel | Dashboard has a per-stream progress card but no global notifications list | F.8 | M | New `notifications` table + a header dropdown listing recent step.complete / job.failed events for the tenant. |
| 9 | Email notification when clips finish | No outbound email | F.9 | M | SES / Postmark integration behind a `notifications.email` config block. Per-tenant opt-in. |
| 10 | 4-card skeleton grid during processing | Single pulsing progress card | F.10 | S | Template-only: render N empty cards (where N = expected clip count) and fade each one in as `clip.created` events arrive. |

---

## Slice F roadmap (after Phase 0 closes)

Group the table above into shippable slices:

### F.1 — Pre-trim range selector
- Frontend: HTML5 video element with two-handle range slider on the
  filmstrip thumbnails (`<canvas>`-rendered, sourced from
  `ffmpeg -vf fps=1/30` filmstrip).
- Backend: `streams.process()` accepts `start_s` / `end_s` kwargs;
  ingest still downloads the whole VOD but pipeline trims via
  `ffmpeg -ss -to` before audio extract.
- Persist on `streams.processing_range_s` (new column, nullable).

### F.2 — Visual caption preset cards
- Render each `CaptionStyle` as an inline SVG mock-up at template time.
- Replace the dropdown in `brand_kit_edit.html` with a card grid.

### F.3 — Duration policy picker
- New column `brand_kits.target_duration_policy` (enum:
  `auto` / `under_60` / `60_to_90` / `90_to_180` / `custom`).
- Per-stream override on the kickoff form.

### F.4 — Pre-flight estimate
- New endpoint `POST /dashboard/streams/estimate` takes
  `(duration_s, brand_kit_id)` returns
  `{minutes, llm_cost_usd_micros, gpu_minutes}`.
- Render on the kickoff form, recompute via HTMX as user adjusts trim.

### F.5–F.6 — Upload + conversion progress
- F.5 needs a tus.io endpoint (or streamed multipart). The dashboard
  uses HTMX SSE polling already; extend the existing progress card.
- F.6 needs a `progress_pct` field on the per-step event payload.
  Whisper already exposes per-segment progress; pipe it into events.

### F.7 — Background pipeline UX copy
- Pure template work. Add a "Puedes cerrar esta página" banner under
  the progress card, ES-first per CLAUDE.md.

### F.8 — In-app notifications
- New table `notifications` (`id`, `tenant_id`, `kind`, `payload_json`,
  `read_at`, `created_at`).
- Repo + dashboard header dropdown + dismiss endpoint.
- Pipeline emits a `notification` on every terminal step.

### F.9 — Email notifications
- Integration: Postmark (transactional + cheap) or SES (cheapest, but
  warm-up required). Default Postmark for dev velocity.
- Per-tenant opt-in via `tenants.notification_email`.

### F.10 — Skeleton grid
- Template change in `stream.html`: render N grey cards keyed by
  expected clip count (computed from voice-marker triggers + viral
  detector candidates). Each card fades in as its `clip.created`
  event arrives.

---

## What this doc explicitly does NOT promise

- **Order.** Phase 0 finishes first (slice D.4 + slice E). F-slices
  ship after.
- **All-or-nothing.** Each F-slice is independently valuable. F.2 (card
  grid) and F.10 (skeleton) are ~half-day wins; F.5/F.6 are the
  meaningful UX upgrade.
- **Feature parity for parity's sake.** ChalybClip's voice-marker triggers
  + per-speaker brand kits are *differentiators* the competitor doesn't
  have. Don't trade those for a prettier upload bar.

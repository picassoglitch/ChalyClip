# ChalybOBS → Hub Publish Contract

Standalone spec for the ChalybOBS repo. ChalybOBS Auto-Clip Mode publishes
finished clips through the Quantor Publish & Engagement Hub (inside the
ChalybClip backend). ChalybOBS never talks to Zernio directly — only this
internal API, with a service token.

## Auth

Every request: `Authorization: Bearer <service-token>`. The token is one
of the `name:token` pairs in the hub's `CHALYBCLIP_HUB_SERVICE_TOKENS`
(e.g. `chalybobs:tok_…`). Missing/unknown token → `401`. The API disabled
(no tokens configured on the hub) → `503`.

Base path: `https://<hub-host>/api/internal/v1`

---

## POST /publish — publish one clip

```jsonc
{
  "tenant_id": "ten_...",
  "clip": {
    "video_url": "https://...mp4",   // publicly fetchable; HEAD-validated
    "title": "Optional",
    "caption_default": "Caption sent to every target unless overridden",
    "duration_s": 45                  // informational
  },
  "targets": ["tiktok", "instagram", "youtube"],   // or "all_connected"
  "mode": "now",                       // now | queue | schedule | draft
  "scheduled_for": "2026-12-31T23:00:00Z",  // required for mode=schedule (unless use_best_time)
  "options": {
    "per_platform_captions": {
      "tiktok": "Caption solo TikTok",
      "youtube": { "title": "Título YT", "caption": "Caption YT" }
    },
    "first_comment": "Optional — only platforms that support it",
    "use_best_time": false             // mode=schedule: resolve from analytics
  },
  "source": "chalybobs",                 // chalybobs | chalybclip | chalyb
  "idempotency_key": "uuid"            // repeat key → returns the original job
}
```

**Response 200**
```jsonc
{
  "ok": true,
  "job_id": "hpj_...",
  "zernio_post_id": "post_...",
  "status": "publishing",              // draft|queued|scheduled|publishing
  "mode": "now",
  "scheduled_for": null,
  "platforms": [{ "platform": "tiktok", "status": "pending" }],
  "duplicate": false                   // true on an idempotency replay
}
```

**Structured errors** (never a raw Zernio 402): HTTP status + `error`
code + `message`:

| HTTP | `error` | Meaning |
|------|---------|---------|
| 402 | `plan_limit` | Zernio account/plan quota full. |
| 409 | `target_not_connected` | A requested platform isn't connected. |
| 409 | `no_zernio_profile` | Tenant has no Zernio profile yet. |
| 409 | `duplicate_content` | Zernio's 24h content-hash dedup. |
| 422 | `media_url_unreachable` / `media_url_not_video` / `media_url_invalid` | Bad `video_url`. |
| 404 | `unknown_tenant` | No such tenant. |
| 400 | `missing_scheduled_for` | `mode=schedule` without time or best-time. |

---

## GET /publish/{job_id}?tenant_id=ten_... — live status

Fed by Zernio webhooks (no polling Zernio). `status` advances
`publishing → published / failed / partial`; `platforms_json` carries
per-platform results once they land.

```jsonc
{ "ok": true, "job_id": "hpj_...", "status": "published",
  "targets": ["tiktok"], "zernio_post_id": "post_...",
  "platforms_json": "[{\"platform\":\"tiktok\",\"status\":\"published\"}]",
  "error": null, "created_at": "...", "updated_at": "..." }
```

---

## GET /accounts?tenant_id=ten_... — what can I publish to?

```jsonc
{ "ok": true, "connected": ["tiktok", "youtube"],
  "accounts": [{ "platform": "tiktok", "account_id": "acct_..." }] }
```

Call this before offering publish targets in ChalybOBS.

---

## POST /batch — a stream session's clips, anti-spam spread

```jsonc
{
  "tenant_id": "ten_...",
  "clips": [ { "video_url": "...", "title": "...", "caption_default": "..." } ],
  "targets": ["tiktok"],               // or "all_connected"
  "options": { "use_best_time": true },
  "source": "chalybobs",
  "idempotency_key": "uuid"            // per-clip key derives as "{key}:{index}"
}
```

The hub distributes the clips across days at no more than
`HUB_MAX_POSTS_PER_PLATFORM_PER_DAY` (default 4) per platform — never all
at once. **Response**: `{ ok, results:[{ok, job_id, scheduled_for, …}],
scheduled, failed }`. Per-clip failures don't abort the batch.

---

## Event relay (optional, push)

The hub can relay Zernio events to a ChalybOBS webhook (per-tenant
subscription) signed with `X-Chalybclip-Signature` (hex HMAC-SHA256 of the
body). Bodies are `{id, tenant_id, type:"zernio.post.published", payload:
{post_id, status, …}, ts}` — ids/status only, no user content.

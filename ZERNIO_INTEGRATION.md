# Zernio Integration — Quantor Publish & Engagement Hub

The single Zernio integration layer for the Quantor ecosystem. It lives
inside the ChalyClip backend and is consumed three ways:

- **ChalyClip** — the Publish Center dashboard (streamers, direct UI).
- **ChalyOBS** — the Android IRL app; its Auto-Clip Mode publishes via the
  internal API (`/api/internal/v1/*`) and never talks to Zernio directly.
- **Chalyb** — other ecosystem engines (content generators, ranking)
  use the same internal API + the analytics read.

Zernio is the upstream social API (`https://zernio.com/api/v1`, 14+
platforms, OAuth on their side). One company-wide API key; each ChalyClip
tenant is one Zernio `profileId`.

---

## Environment variables

All carry the `CHALYBCLIP_` prefix (Pydantic Settings).

| Var | Default | Purpose |
|-----|---------|---------|
| `CHALYBCLIP_ZERNIO_API_KEY` | — | Company-wide Zernio bearer key (`sk_...`). Required to publish. |
| `CHALYBCLIP_ZERNIO_BASE_URL` | `https://zernio.com/api/v1` | Override for staging. |
| `CHALYBCLIP_ZERNIO_WEBHOOK_SECRET` | — | HMAC secret to verify inbound webhooks. Unset → receiver 503s. |
| `CHALYBCLIP_PUBLIC_URL` | `http://localhost:8000` | Externally-reachable origin (connect redirect, signed clip URLs, webhook registration). |
| `CHALYBCLIP_HUB_SERVICE_TOKENS` | — | Comma-sep `name:token` pairs for the internal API. Unset → `/api/internal/v1/*` 503s. |
| `CHALYBCLIP_HUB_MAX_POSTS_PER_PLATFORM_PER_DAY` | `4` | Batch anti-spam cap. |
| `CHALYBCLIP_HUB_AUTO_RETRY_DELAY_S` | `600` | Delay before the one-shot auto-retry on a transient `post.failed` (`0`=inline, `<0`=off). |
| `CHALYBCLIP_HUB_MAX_BROADCASTS_PER_DAY` | `1` | Per-tenant daily broadcast cap (irreversible mass-DM guardrail). |
| `CHALYBCLIP_FEATURE_WHATSAPP` | `0` | WhatsApp seam (extra cost). Off → routes 404, UI hidden. |
| `CHALYBCLIP_FEATURE_ADS` | `0` | Ads seam (Meta spend). Off → routes 404, UI hidden. |

---

## Connect flow (white-label popup)

```
 Dashboard click
   │  window.open("", "zernio_connect", popup)         ← synchronous, beats popup blockers
   ▼
 POST /dashboard/publish/zernio/connect {platform}
   │  → ZernioClient.connect_url(platform, profile_id,
   │       redirect_url=PUBLIC_URL/.../connected?platform=X,
   │       headless=(platform needs a post-OAuth pick))
   ▼
 popup → Zernio hosted OAuth → redirect_url
   │
   ├─ standard: lands /connected?platform=X
   │     → postMessage({type:"zernio:connected", platform}) → window.close()
   │     → opener refreshes the account chips (no full reload)
   │
   └─ headless (Facebook): lands /connected?platform=facebook&tempToken=…
         → renders OUR page picker → POST /fb-page/select
         → postMessage + close
```

Discord/Telegram connect the same way (community channels — connectable,
never clip targets).

---

## Webhooks (the event backbone)

- **Inbound**: `POST /api/webhooks/zernio`. Verifies `X-Zernio-Signature`
  (lowercase hex HMAC-SHA256 of the raw body), dedups on `payload.id`
  (`zernio_events` PK), ACKs 2xx fast, processes in a background task.
- **Registration**: `chalybclip webhooks register-zernio` (idempotent
  create-or-update of the webhook at `{PUBLIC_URL}/api/webhooks/zernio`).
- **Events handled**: `post.scheduled/published/failed/partial/cancelled`
  + `post.platform.*` (publish-job status), `post.external.created/
  updated/deleted` (calendar), `account.connected/disconnected`,
  `comment.received` (inbox + contact seed), `message.received/sent` +
  `conversation.started` (DMs + contact seed),
  `account.ads.initial_sync_completed`, `whatsapp.number.*` (provisioning
  status). At-least-once delivery → every handler is idempotent.

### Outbound event fan-out
Each processed Zernio event is recorded as a `zernio.<type>` row in the
`events` table and relayed to the tenant's webhook subscriptions via the
existing HMAC-signed dispatcher (`X-Chalybclip-Signature`). So ChalyOBS /
Chalyb subscribers receive the relay. Relay bodies carry ids/status
only — never comment/message text.

---

## Internal Publish API (`/api/internal/v1/*`)

Auth: `Authorization: Bearer <service-token>` validated against
`CHALYBCLIP_HUB_SERVICE_TOKENS`. See `CHALYBOBS_PUBLISH_CONTRACT.md` for the
full request/response examples. Summary:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/internal/v1/publish` | Publish one clip (now/queue/schedule/draft), idempotent on `idempotency_key`. |
| GET | `/api/internal/v1/publish/{job_id}?tenant_id=` | Live job status (webhook-fed). |
| GET | `/api/internal/v1/accounts?tenant_id=` | Connected platforms. |
| POST | `/api/internal/v1/batch` | A session's clips, spread under the per-platform daily cap. |
| GET | `/api/internal/v1/analytics?tenant_id=&days=` | Performance read (live + snapshot fallback). |

Structured errors (never a raw Zernio 402): `plan_limit`,
`duplicate_content`, `target_not_connected`, `media_url_*`,
`no_zernio_profile`, `unknown_tenant`.

---

## Feature → endpoint map

| Feature | Dashboard route(s) | Zernio endpoint(s) |
|---------|--------------------|--------------------|
| Connect + headless FB | `/connect`, `/connected`, `/fb-pages`, `/fb-page/select` | `GET /connect/{platform}`, `GET/POST /connect/facebook/select-page` |
| Publish power-ups | `/post/{clip_id}`, `/bulk-post` | `POST /posts` (customContent, platformSpecificData, isDraft) |
| Drafts | `/draft/{id}/publish`, `/draft/{id}/delete` | `POST /posts`, `DELETE /posts/{id}` |
| Scheduling | `/schedule.json`, `/schedule/slots`, `/best-time.json`, `/schedule/cancel/{id}` | `GET/PUT/DELETE /queue/slots`, `GET /analytics/best-time`, `DELETE /posts/{id}` |
| Reliability | `/failed.json`, `/retry/{id}`, `/retry-all` | `GET /posts?status=failed`, `POST /posts/{id}/retry` |
| Analytics | `/rendimiento.json`, `/api/internal/v1/analytics` | `GET /analytics` |
| Calendar | `/calendar.json` | `post.external.*` webhooks |
| Inbox | `/inbox/comments*`, `/inbox/conversations*` | `GET/POST /inbox/comments/*`, `GET/POST/PUT /inbox/conversations/*` |
| Growth (Pro) | `/growth/*` | `/comment-automations`, `/contacts`, `/sequences`, `/broadcasts` |
| Community | `/community/settings*` | `POST /posts` (Discord embeds / Telegram) |
| Ads (flag) | `/ads/campaigns.json`, `/ads/boost` | `GET /ads/campaigns`, `POST /ads/boost` |
| WhatsApp (flag) | `/whatsapp/status.json` | `whatsapp.number.*` webhooks |

---

## Hardening

- **429 backoff**: `ZernioClient` retries 429s up to `max_retries` (3),
  honoring `Retry-After` when present, else exponential backoff
  (0.5·2ⁿ capped at 30s). Non-429 4xx/5xx are not retried.
- **No secrets in logs**: the client logs `method`/`path`/`status`/
  `attempt` only — never the api key or Bearer header. Treat logs as
  semi-public.
- **Tenant isolation**: webhook-fed stores (calendar, inbox, whatsapp)
  are keyed by Zernio `account_id` and resolved to a tenant at read time
  by matching against that tenant's connected accounts.

## CLI

| Command | When |
|---------|------|
| `chalybclip webhooks register-zernio` | Once / after PUBLIC_URL or secret change. |
| `chalybclip webhooks snapshot-analytics` | Daily (cron) — per-post metric snapshots. |
| `chalybclip webhooks community-digest` | Weekly (cron) — opt-in community digest. |

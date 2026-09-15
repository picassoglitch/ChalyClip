# ChalybClip ↔ Chalyb integration contract

This document is the SOURCE OF TRUTH for what ChalybClip must expose so Chalyb
(the operator dashboard at `chalyb.com`) can onboard and launch users into
ChalybClip seamlessly.

**Audience:** whoever is implementing the ChalybClip side. The Chalyb side is
already done and waiting — see `src/lib/engines/integrations/chalybclip.ts` in
the `chalyb` repo for the calling code.

---

## The model in 3 sentences

1. Chalyb is the **identity authority** — users authenticate there with
   Supabase Auth (Google OAuth + email/password).
2. When a Chalyb user activates ChalybClip, Chalyb calls ChalybClip's admin API
   to **provision a tenant** keyed by the Chalyb user id, and stores the
   returned `tenant_id` + `api_token` in its own database.
3. When the user clicks "Abrir ChalybClip" in Chalyb, Chalyb redirects to
   `https://chalybclip.chalyb.com/auth/sso?token=<signed>` — ChalybClip verifies the
   signature, creates its own session cookie, and renders its dashboard.

ChalybClip stays independently deployable. No shared database. No CORS gymnastics.
The shared secret (HMAC key) is the only coupling.

---

## Required env vars on the ChalybClip side

```bash
# Admin API authentication — Chalyb sends this as Bearer on /api/admin/* calls.
# Generate with: openssl rand -base64 48
CHALYB_ADMIN_TOKEN=...

# SSO token signing — must MATCH Chalyb's CHALYBCLIP_SSO_SECRET exactly.
# Same secret on both sides → ChalybClip can verify what Chalyb signed.
CHALYB_SSO_SECRET=...

# Where ChalybClip's user dashboard lives — used as the post-SSO redirect target.
CHALYBCLIP_PUBLIC_URL=https://chalybclip.chalyb.com
```

On the Chalyb side (`.env.local`) the same two secrets are:

```bash
CHALYBCLIP_ADMIN_TOKEN=<same as CHALYB_ADMIN_TOKEN above>
CHALYBCLIP_SSO_SECRET=<same as CHALYB_SSO_SECRET above>
```

---

## Endpoint 1: tenant provisioning

`POST {CHALYBCLIP_PUBLIC_URL}/api/admin/tenants`

Called by Chalyb when a user first activates ChalybClip (PRO live-bot
selection, ALL_ACCESS seed, admin grant, or paid MP upgrade).

### Request

```http
POST /api/admin/tenants HTTP/1.1
Authorization: Bearer <CHALYB_ADMIN_TOKEN>
Content-Type: application/json

{
  "external_user_id": "auth0|abc123def456",
  "email": "alexis.ralcantara@gmail.com",
  "display_name": "Alexis Ramírez"
}
```

- `external_user_id` — the Supabase Auth user id from Chalyb. ChalybClip
  should treat this as an opaque string and use it as the unique key for
  detecting existing tenants.
- `email` — for the tenant's primary user record. ChalybClip uses this for
  notifications, billing receipts, etc.
- `display_name` — friendly name. Optional fallback to email local-part if
  not provided.

### Response — 200 / 201 (first time)

```json
{
  "tenant_id": "ten_01HV3K9...",
  "api_token": "tok_01HV3K9..."
}
```

### Response — 409 (already exists)

```json
{
  "error": "duplicate",
  "tenant_id": "ten_01HV3K9...",
  "api_token": "tok_01HV3K9..."
}
```

**Chalyb treats 409 as success** — it stores the returned ids and moves on.
This makes admin re-grants and webhook retries safe.

### Response — 401 / 403

`Authorization` header missing or wrong. Body shape free-form. Chalyb logs
and stops retrying.

### Implementation notes for ChalybClip

- `tenant_id` should use the `ten_` ULID prefix per CLAUDE.md rule 7.
- `api_token` should use the `tok_` ULID prefix per the same rule. Store a
  hash, return the raw token only on creation — never return raw tokens
  again on subsequent reads.
- The Bearer admin token must be constant-time-compared, not `==`.
- This endpoint should be excluded from the regular tenant-scoping decorator
  (it creates the tenant rather than acting within one).

---

## Endpoint 2: SSO sign-in redirect

`GET {CHALYBCLIP_PUBLIC_URL}/auth/sso?token=<signed>`

Called when the user clicks "Abrir ChalybClip" in Chalyb. ChalybClip verifies
the signed token, creates its own session cookie, and redirects to the
dashboard.

### Token format

The token has shape `{base64url_payload}.{base64url_sig}` — same pattern as
a JWT but without the JWT header (we don't need algorithm agility).

**Payload (decoded):**

```json
{
  "user_id": "auth0|abc123def456",
  "email": "alexis.ralcantara@gmail.com",
  "tenant_id": "ten_01HV3K9...",
  "exp": 1700000000
}
```

- `user_id` — same Supabase Auth id used during provisioning. ChalybClip uses
  this to look up the tenant.
- `tenant_id` — the value returned by `/api/admin/tenants`. Chalyb sends
  it so ChalybClip doesn't need a second lookup before validating.
- `exp` — Unix seconds, **5 minutes from issue time**. ChalybClip MUST reject
  tokens past expiry.

**Signature:** HMAC-SHA256(`payload_base64url`, `CHALYB_SSO_SECRET`),
encoded as base64url.

### Verification algorithm (ChalybClip side)

```python
import base64, hmac, hashlib, json, time

def verify_sso_token(token: str, secret: str) -> dict:
    try:
        payload_b64, sig_b64 = token.split('.')
    except ValueError:
        raise ValueError('malformed token')

    expected_sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    ).decode().rstrip('=')

    # constant-time compare
    if not hmac.compare_digest(expected_sig, sig_b64.rstrip('=')):
        raise ValueError('bad signature')

    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + '=='))
    if payload['exp'] < time.time():
        raise ValueError('token expired')

    return payload  # { user_id, email, tenant_id, exp }
```

### Response

- **Valid token** → set ChalybClip session cookie scoped to the tenant from
  payload, redirect to `/dashboard`.
- **Invalid / expired** → render an error page or redirect to a login
  fallback. Don't expose validation details (don't say "expired" vs
  "bad sig" — generic "session inválida" is fine).

### Why HMAC and not JWT

Pragmatic: we control both ends, we don't need key rotation drama, and HMAC
is a single function call. JWT brings algorithm-agility + claims standards
that don't matter for this two-party trust relationship.

---

## Optional but recommended later

### Webhook back to Chalyb

When ChalybClip wants to tell Chalyb something (job completed, tenant ran
out of quota, payment failed inside ChalybClip), POST to:

`https://chalyb.com/api/engines/chalybclip/webhook`

Signed the same way (HMAC over the body, `X-Signature: sha256=<hex>` header).
**Not yet implemented on Chalyb** — flag if you need it and we'll build the
receiver.

### Tenant deactivation

`POST /api/admin/tenants/{tenant_id}/deactivate` (idempotent) — Chalyb calls
this when a subscriber downgrades or cancels. NOT YET CALLED from Chalyb's
side — we'd add it when there's a real need (right now downgrades just leave
the access record in place so re-upgrades are seamless).

---

## Local dev workflow

1. Pick a strong shared secret (`openssl rand -base64 48`), put the same
   value in both projects' env files as documented above.
2. Run ChalybClip locally on `http://localhost:8000` (FastAPI's default).
3. Override `external_url` and `admin_api_base` on the ChalybClip engines row
   in Supabase to point at localhost:
   ```sql
   update engines
     set external_url = 'http://localhost:8000',
         admin_api_base = 'http://localhost:8000/api/admin'
     where slug = 'chalybclip';
   ```
4. As a PRO user in Chalyb, click "Activar en vivo" on ChalybClip. Chalyb
   should POST `localhost:8000/api/admin/tenants` and store the returned
   ids. Check `engine_subscriptions` row in Supabase to verify.
5. Click "Abrir ChalybClip ↗" — should open a new tab to
   `localhost:8000/auth/sso?token=...` and land in ChalybClip's dashboard.

---

## Failure modes Chalyb handles already

| What happens | How Chalyb reacts |
|---|---|
| ChalybClip is down (network error) | Provisioning silently logs the error, subscription row is kept without external_user_id. Admin can retry. |
| `CHALYBCLIP_ADMIN_TOKEN` missing | Provisioning returns `not_configured`, no row written externally. |
| 409 duplicate response | Treated as success, returned ids are persisted. |
| 401/403 from admin endpoint | Logged loudly, no retry (will keep failing). |
| Launch click before provisioning ran | Toast: "No tienes acceso provisionado a este engine." |
| Launch click but no `external_url` | Toast: "Este engine aún no tiene URL externa. Estás en modo placeholder." |

---

## Checklist for "ChalybClip is onboarded"

- [ ] Both env vars set on ChalybClip side (`CHALYB_ADMIN_TOKEN`, `CHALYB_SSO_SECRET`)
- [ ] Same secrets set on Chalyb side (`CHALYBCLIP_ADMIN_TOKEN`, `CHALYBCLIP_SSO_SECRET`)
- [ ] `POST /api/admin/tenants` implemented per spec above
- [ ] `GET /auth/sso?token=...` implemented per spec above
- [ ] ChalybClip deployed to a public URL (Railway/Fly.io/Render — FastAPI + Python)
- [ ] `engines` row in Chalyb's Supabase has `external_url` and `admin_api_base` pointing at the deployed URL
- [ ] Smoke test: activate as PRO subscriber → `engine_subscriptions.external_user_id` gets populated → "Abrir ChalybClip" opens a working session

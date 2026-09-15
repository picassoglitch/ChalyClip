# Deploying MediaMTX for ChalybClip live ingest

> **Canonical deploy is now Path B — the `chalybclip-live` repo + S3-compatible
> object storage.** The MediaMTX service lives in its own repo
> (`picassoglitch/chalybclip-live`), records each stream, and uploads it to an
> object store; ChalybClip pulls it to auto-clip (Phase L.2). **Default store is
> Cloudflare R2** — $0 egress, so the once-per-stream download is free
> (~$0.005/stream vs ~$0.40 on metered-egress stores); cheapest for video and
> it isolates bulky recordings from the shared Supabase egress budget.
> Supabase/MinIO/S3 also work — it's all the same `CHALYBCLIP_LIVE_STORAGE_*`
> config, only the endpoint + keys change.
>
> The shared-`/data`-volume approach below is **Path A (legacy)** — kept for
> reference / single-box setups. It works, but doesn't scale (RTMP ingest and
> the clip workers are pinned to one volume). Prefer Path B.

---

## Path B — separate repo + object storage (recommended)

```
OBS ──rtmp──▶ chalybclip-live (MediaMTX svc) ──upload──▶ store/live/<stream_id>/
                     │ webhooks (authorize/started/ended)        ▲
                     ▼                                            │ pull
                  ChalybClip  ──auto-clip──────────────────────────┘
```

- **Service repo + full deploy steps:** `chalybclip-live/README.md`.
- **ChalybClip side:** set `CHALYBCLIP_LIVE_STORAGE_BUCKET`,
  `CHALYBCLIP_LIVE_STORAGE_ENDPOINT`, `CHALYBCLIP_LIVE_STORAGE_ACCESS_KEY_ID`,
  `CHALYBCLIP_LIVE_STORAGE_SECRET_ACCESS_KEY` (+ optional
  `CHALYBCLIP_LIVE_STORAGE_PREFIX`, `CHALYBCLIP_LIVE_STORAGE_REGION`) and
  `CHALYBCLIP_LIVE_RTMP_BASE_URL`. When the bucket is set, the live runner
  pulls recordings from the store instead of disk — no shared volume.

  **Cloudflare R2 (default):** create a bucket, create an R2 API token, and set
  `CHALYBCLIP_LIVE_STORAGE_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com`
  with `CHALYBCLIP_LIVE_STORAGE_REGION=auto`. $0 egress + a 10 GB / 1M-op free
  tier, so low volume is genuinely free and there's no monthly minimum.

  **Supabase Storage (alternative):** create a private bucket, grab S3 keys
  from *Project Settings → Storage*, set
  `CHALYBCLIP_LIVE_STORAGE_ENDPOINT=https://<ref>.supabase.co/storage/v1/s3`
  + `CHALYBCLIP_LIVE_STORAGE_REGION=<project region>`, and raise the bucket's
  file-size limit (default 50 MB). Note: ~$0.09/GB egress on every pull, and
  it draws from the project-wide egress budget.

The rest of this doc is **Path A (legacy, shared volume).**

---

# Path A (legacy) — shared `/data` volume

This is the operator-side deploy guide for the MediaMTX service that
sits in front of ChalybClip and accepts RTMP push from OBS.

**One-time setup**. Run through this once; MediaMTX then runs as a
separate Railway service indefinitely.

## Architecture refresher

```
OBS / Streamlabs
    │  rtmp://live.chalybclip.chalyb.com/live/<stream_key>
    ▼
┌────────────────────────────┐
│ MediaMTX (Railway service) │  config: infra/mediamtx.yml
│ - RTMP on :1935            │
│ - records to /data/live/   │
│ - calls ChalybClip webhooks  │
└──────────┬─────────────────┘
           │ webhooks (auth, started, ended)
           ▼
┌────────────────────────────┐
│ ChalybClip (existing service)│
│ - creates streams row      │
│ - existing VOD pipeline    │
└────────────────────────────┘
```

The two services share the same `/data` volume so MediaMTX's
recording lands where ChalybClip's existing pipeline expects to find
clip sources.

## Step 1 — Create the MediaMTX Railway service

In your Railway project (the one already running ChalybClip):

1. **+ New** → **Empty Service**
2. Settings → name it `mediamtx`
3. **Source** → switch to **Image** → use `bluenviron/mediamtx:latest`
4. **Networking** → enable Public Networking → expose port `1935`
   (TCP, not HTTP). The host gives you a TCP proxy host + port; CNAME
   `live.chalybclip.chalyb.com` to that host in DNS so the public
   endpoint is `live.chalybclip.chalyb.com:NNNN` — note the port.
5. **Volume** → attach the SAME volume that ChalybClip is mounted on
   (don't create a new one). Mount path `/data`. This is what makes
   the recording handoff work.
6. **Variables** → add the four below.

## Step 2 — Environment variables

These three URLs all point at your ChalybClip dashboard service
(same Railway project, different service):

```bash
CHALYBCLIP_AUTH_URL=https://chalybclip.chalyb.com/api/internal/live/authorize
CHALYBCLIP_STARTED_URL=https://chalybclip.chalyb.com/api/internal/live/started
CHALYBCLIP_ENDED_URL=https://chalybclip.chalyb.com/api/internal/live/ended
CHALYBCLIP_INTERNAL_SIGNING_SECRET=<SAME value already set on the ChalybClip service>
```

The signing secret MUST match what's set on the ChalybClip side — it's
the bearer MediaMTX passes back to ChalybClip on each webhook.

## Step 3 — Mount the config file

MediaMTX needs `infra/mediamtx.yml` from this repo. Two ways:

**Easy** — bake it into a small Dockerfile and point Railway at that
instead of the base MediaMTX image. Add a sibling `Dockerfile.mediamtx`:

```dockerfile
FROM bluenviron/mediamtx:latest
COPY infra/mediamtx.yml /mediamtx.yml
CMD ["/mediamtx", "/mediamtx.yml"]
```

In the MediaMTX service: Source → switch from Image to Dockerfile,
point at `Dockerfile.mediamtx`.

**Alternative** — use Railway's Config File feature to mount the YAML
at `/mediamtx.yml`. Slightly more setup; not recommended.

## Step 4 — Set the RTMP URL in ChalybClip

On the ChalybClip service (not MediaMTX), add the env var so the
dashboard knows what URL to show operators:

```bash
CHALYBCLIP_LIVE_RTMP_BASE_URL=rtmp://live.chalybclip.chalyb.com:NNNN/live
```

(Replace `NNNN` with the port Railway exposed in step 1.5.)

## Step 5 — Verify

1. Both services should show **Active** in Railway.
2. From ChalybClip's `/dashboard/live` page: the "Push URL for OBS"
   section should now be populated (not the "not configured" state).
3. Click "Generate stream key" — the key value should appear.
4. In OBS: Settings → Stream → Custom → paste the Server URL +
   Stream Key → OK → Start Streaming.
5. Within ~5 seconds the Live streams table on `/dashboard/live`
   should show a new row with status `live` and a pulsing dot.
6. Stop streaming in OBS — within ~5 seconds the row flips to
   `live_ended`.
7. Click "Open + run pipeline" on the row. The existing VOD pipeline
   runs against the recording.

## Troubleshooting

**OBS says "Failed to connect to server"**:
- Wrong port. Re-check Railway's exposed TCP port; Railway assigns a
  random external port that the operator's OBS URL must use.
- The TCP proxy URL is what Railway shows — copy that, not the HTTPS
  one.

**OBS connects but the publish is rejected**:
- Stream key is wrong / revoked. Re-generate from the live dashboard.
- `CHALYBCLIP_AUTH_URL` is unset or wrong on the MediaMTX service.
- `CHALYBCLIP_INTERNAL_SIGNING_SECRET` doesn't match across services.

**Streaming works, recording doesn't appear**:
- The volume isn't shared between services. Each Railway service must
  mount the same Volume; check the volume name in both services.

**Recording shows up but the pipeline fails to read it**:
- The recording path in MediaMTX has to match what the streams row
  carries as `source_video_path`. The L.1 webhook implementation
  derives this automatically; if you see a mismatch, check the
  webhook logs in Railway for the value MediaMTX sent.

## Cost expectation

MediaMTX itself is light: ~50 MB RAM, fractional CPU when no streams.
The bulk of the cost is bandwidth (recording 1080p at 4-8 Mbps adds
up over multi-hour streams). For 10 active users each doing 2-hour
streams a few times a week, expect ~$5-15/mo of Railway egress on
the MediaMTX service alone.

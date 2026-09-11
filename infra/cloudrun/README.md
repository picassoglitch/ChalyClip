# ChalybClip on Cloud Run

Deploy notes for the GCP migration. The infrastructure itself is Terraform in
the `nexo-ai` repo (`infra/terraform/`); this covers what is specific to
*this* application.

## Two services, one image

| Service | `NEXOCLIP_ROLE` | What it runs |
|---|---|---|
| `chalybclip` | `api` (default) | FastAPI dashboard + the hub contract (`POST /api/admin/tenants`, `GET /auth/sso`) |
| `chalybclip-worker` | `worker` | `nexoclip worker` — the pipeline |

`docker-entrypoint.sh` dispatches on the role, so both deploy from the same
image and there is one thing to build. Railway is unaffected: the role
defaults to `api` and runs exactly what it ran before.

The API dispatches pipeline runs to the worker over the kickoff/poll HTTP
contract that `ModalJobDispatcher` already speaks — point
`NEXOCLIP_MODAL_PIPELINE_ENDPOINT_URL` at the worker's URL and set
`NEXOCLIP_WORKER_TOKEN`. No new dispatcher is needed; Cloud Run is just
another host that answers that contract.

## Three things that will bite

**1. The worker needs CPU always allocated.** It answers the kickoff POST
immediately and does the work in an `asyncio` task. Cloud Run's default
withdraws CPU when a response is sent, which freezes the pipeline mid-job
with no error — the job simply never progresses. Deploy the worker with
`--no-cpu-throttling` (it is in `cloudbuild.yaml`). Even then, an instance
with no in-flight requests can be reclaimed; the API's polling is what keeps
it alive, so a job whose poller stops can still be lost. If that turns out to
happen in practice, the fix is `--min-instances=1` on the worker, at the cost
of an always-on instance.

**2. There is no persistent disk.** Cloud Run's filesystem is tmpfs — writes
to `/data` consume the instance's *memory* and vanish on reclaim. The
Dockerfile's `/data` defaults are Railway's, and must be overridden:

- `DATABASE_URL` → the Supabase Postgres DSN. Note it is read *without* the
  `NEXOCLIP_` prefix (`validation_alias="DATABASE_URL"` in `settings.py`).
  Leave it unset and you get a SQLite file that resets on every cold start.
- `NEXOCLIP_DEFAULT_OUTPUT_DIR` → a mounted GCS volume, or `/tmp` for scratch
  with artifacts uploaded to the bucket.

**3. Cold starts are slow, because the image is big.** ffmpeg, OpenCV and a
full Playwright Chromium. Startup CPU boost helps; if cold starts stay
painful, the Chromium layer is the biggest single win — it exists only for
pixel-exact caption rendering on `/clips/<id>/download`, so a variant image
without it would be much lighter for the API service.

## Env vars, and where each name comes from

Most settings take the `NEXOCLIP_` prefix. Three do not, because
`settings.py` gives them an explicit `validation_alias`:

| Env var | Notes |
|---|---|
| `DATABASE_URL` | No prefix. |
| `NEXO_AI_ADMIN_TOKEN` | No prefix. Must equal `CHALYBCLIP_ADMIN_TOKEN` in the hub's Vercel env. |
| `NEXO_AI_SSO_SECRET` | No prefix. Must equal `CHALYBCLIP_SSO_SECRET` in the hub's Vercel env, or every SSO launch fails signature verification. |

The docstring at the top of `nexoclip/api/routers/nexo_ai.py` calls this
`NEXOCLIP_NEXO_AI_SSO_SECRET`. That is stale — the alias in `settings.py` is
authoritative.

These two names still say `NEXO_AI` because this repo has not been through
the rebrand yet; only the hub has. Renaming the aliases to `CHALYB_*` is a
follow-up, and both sides must move together.

## Build

```sh
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_REGION=us-central1,_PROJECT=chalyb
```

The build timeout is 2400s and the machine is `E2_HIGHCPU_8` on purpose:
`playwright install --with-deps chromium` alone blows past Cloud Build's
10-minute default on the standard machine.

`cloudbuild.yaml` deploys with `--image` only. Everything else about the
services — service account, secrets, scaling, memory — is Terraform's, and
`gcloud run deploy` leaves unspecified fields alone, so the two do not fight
over the spec.

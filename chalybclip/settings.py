"""Process-wide settings driven by environment + `.env`.

Per CLAUDE.md: config flows through Pydantic Settings, not scattered
`os.getenv` calls. Phase 0 only needs the pieces that have an env var in
`.env.example`; YAML loading arrives when the detection phrase list does.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read once from environment / `.env`, then pass into service functions."""

    model_config = SettingsConfigDict(
        env_prefix="CHALYBCLIP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    default_tenant_id: str = "default"
    default_output_dir: str = "./out"
    db_path: str = "./chalybclip.db"

    # Postgres migration seam. When set, the app talks to Postgres via
    # asyncpg instead of the local SQLite file — `Database` dispatches on
    # the scheme. Railway injects a bare `DATABASE_URL` when a Postgres
    # service is attached, so we read that name directly (not the
    # CHALYBCLIP_ prefix). Unset → SQLite (current default).
    database_url: str | None = Field(
        default=None,
        validation_alias="DATABASE_URL",
    )

    whisper_device: str = "cuda"
    whisper_model: str = "medium"
    whisper_compute_type: str = "float16"

    # Slice F.8 — TranscribeProvider selection. "local" runs
    # faster-whisper on this host (current production behavior).
    # Other values map to cloud STT vendors (assemblyai / deepgram /
    # openai) when their providers ship in F.10+. Each cloud option
    # reads its own API key from the matching `<vendor>_api_key`
    # setting below.
    # Slice O.46 — variants generation is OFF by default. Operator
    # feedback: "they take a lot and don't do much". The variants step
    # used to fan out per-persona LLM caption drafts; with the editor's
    # overlay system + hook generator carrying most of the same value
    # without an LLM call per persona, the per-clip variant generation
    # was wasteful. When False, the pipeline still creates a single
    # stub VariantRow per clip so the publish flow keeps working
    # (publishers consume variant.caption / variant.hashtags).
    variants_enabled: bool = False

    # Auto-generate a viral hook (title line) per clip during the pipeline,
    # for clips at/above the tenant's auto-publish score threshold (so the
    # publish-worthy clips get a hook without a manual "Generar 5" click —
    # pre-filled in the editor, burned into the render, used by auto-publish).
    # One small LLM call per qualifying clip. Set False to disable.
    auto_hook_enabled: bool = True

    # Storage hygiene — delete the raw source VOD (downloaded video +
    # extracted audio) as soon as the pipeline finishes. The raw source is
    # by far the heaviest artifact on disk (~5-50 GB per 4hr stream) and is
    # only needed DURING processing: once clips are cut + the transcript is
    # persisted, it's dead weight. Deleting on completion is what keeps the
    # volume flat as the number of processed streams grows — without it, a
    # single active operator fills a 100+ GB volume in a month.
    #
    # Re-runs stay safe: transcribe + cut both serve their cached outputs
    # before touching the source, so a non-forced re-run of an already-
    # processed stream is a no-op even with the source gone. A `force=True`
    # re-run (or re-cut) needs the source and will re-download it.
    #
    # Set False to keep the source around (debugging, or to allow re-cuts
    # without re-download). The per-tenant retention window still applies as
    # a backstop for any source that escapes this path (e.g. a pipeline that
    # crashed before completion).
    delete_source_on_completion: bool = True

    # Free-disk preflight — refuse to START a VOD download when the output
    # volume has less than this many bytes free. A multi-hour VOD is 5-50 GB;
    # starting a download onto a near-full volume just produces a truncated
    # file that fails downstream and (until reclaimed) eats the last of the
    # disk. Failing fast with an actionable "disk full" message is strictly
    # better. Default 3 GiB — low enough to rarely false-block, high enough
    # that anything below it would fail mid-download anyway. Set 0 to disable.
    min_free_disk_bytes: int = Field(
        default=3 * 1024 * 1024 * 1024,
        validation_alias="CHALYBCLIP_MIN_FREE_DISK_BYTES",
    )

    # Disk-budget HIGH-WATER mark. The volume self-regulates against THIS,
    # not just the hard floor above: whenever free space drops below it, the
    # background watchdog (and the ingest preflight) evict the OLDEST raw
    # sources — heaviest, post-processing-disposable artifact — until free is
    # back above it. This is what bounds total disk regardless of how many
    # users/streams pile up; time-window retention only bounds AGE, not size.
    # Default 10 GiB headroom; must be >= min_free_disk_bytes. Set 0 to
    # disable the proactive watchdog (the hard floor + 24h sweep still apply).
    target_free_disk_bytes: int = Field(
        default=10 * 1024 * 1024 * 1024,
        validation_alias="CHALYBCLIP_TARGET_FREE_DISK_BYTES",
    )

    transcribe_provider: str = "local"
    assemblyai_api_key: str | None = None

    # AssemblyAI tuning.
    #
    # `assemblyai_language_mode` — what the AAI submit-time language
    # contract looks like per call. Three values:
    #
    #   "auto" (DEFAULT) — submit with `language_detection=true` +
    #     the code-switching prompt. AAI's Universal-3 Pro model
    #     handles ES/EN code-switching natively; the prompt
    #     instructs it to preserve each phrase in its original
    #     language instead of translating. This is the LATAM-creator
    #     default — every Mexican / Argentine / Spanish-speaking
    #     streamer who drops English filler ("clip that", "GG", etc)
    #     gets faithfully-transcribed captions in BOTH languages.
    #
    #   "es" | "en" — lock to a specific language. Submits
    #     `language_code=<code>` and drops language_detection + the
    #     code-switching prompt. Best accuracy for monolingual
    #     creators (most US English streamers, dedicated ES-only
    #     channels). Per-stream overrides through the existing
    #     `language` field on the pipeline call.
    #
    # `assemblyai_speech_models` — model ladder. The first model is
    # the quality target (`universal-3-pro` — best ES/EN accuracy +
    # native code-switching). The second is the 99-language fallback
    # (`universal-2`) for the rare case U3-Pro misses on an exotic
    # accent. AAI walks the ladder in order.
    #
    # `assemblyai_speaker_labels` — per-video diarization labels
    # (A / B / C). The pipeline reads these instead of running
    # pyannote separately (Task A2). Default True; flip False to
    # drop the ~$0.02/hr diarization upcharge if a particular run
    # doesn't need speaker separation.
    assemblyai_language_mode: str = "auto"
    assemblyai_speech_models: list[str] = Field(
        default_factory=lambda: ["universal-3-pro", "universal-2"]
    )
    assemblyai_speaker_labels: bool = True
    # Slice O.44 — Modal Whisper provider. Wired when
    # `transcribe_provider="modal"`. The endpoint is the URL Modal
    # exposes for the `transcribe` web function (printed by
    # `modal deploy infra/modal_whisper_app.py`). The token is a
    # shared bearer that the Modal app verifies — pick something
    # long and random.
    #   - modal_endpoint_url: e.g. https://username--chalybclip-whisper-transcribe.modal.run
    #   - modal_token: shared secret in the Authorization: Bearer header
    #   - modal_model: faster-whisper model size; "small" is the sweet
    #     spot on a T4 GPU (3-5 min per VOD-hour, accurate Spanish).
    #     "base" is faster but loses on accented speech; "medium" / "large-v3"
    #     are slower + costlier on Modal.
    modal_endpoint_url: str | None = None
    modal_token: str | None = None
    modal_model: str = "small"
    # Phase 2b — Modal PIPELINE worker (separate app from the whisper
    # one). The endpoint is the URL Modal prints for the `run_pipeline`
    # web function of `modal deploy infra/modal_pipeline_app.py`; the
    # bearer is the same `modal_token` shared secret. Set
    # `CHALYBCLIP_JOB_DISPATCHER=modal` to route http(s)-VOD pipeline runs
    # there. `modal_pipeline_timeout_s` bounds how long the dispatcher
    # polls one run before giving up on *tracking* it (the remote run
    # itself keeps going and keeps writing step events).
    modal_pipeline_endpoint_url: str | None = None
    modal_pipeline_timeout_s: float = 21600.0  # 6h — multi-hour VODs
    # Phase 2b — how the whisper provider hands Modal the audio. False
    # (default, Railway): HMAC-signed URL on our own public host. True
    # (the pipeline WORKER env): the WAV was extracted on the worker's
    # ephemeral disk and never existed on the web box, so upload it to
    # the object store and pass a presigned URL instead.
    transcribe_audio_via_object_storage: bool = False
    # Slice O.44 — HMAC secret for the internal signed-URL audio
    # fetch endpoint Modal pulls from. Different secret than the
    # SSO HMAC — this one is purely internal. Pick something random;
    # the audio URL only stays valid for 30 min so a leak is bounded.
    # Phase L.1 reuses this same secret for the MediaMTX webhook
    # bearer (internal-trust-only; not user-facing).
    internal_signing_secret: str | None = None

    # Phase L.1 — RTMP base URL the operator pastes into OBS. This is
    # the MediaMTX endpoint, NOT the ChalyClip API host. Example:
    # `rtmp://live.chalybclip.chalyb.com/live`
    # The dashboard appends the active stream key to this base when
    # displaying the OBS URL. Unset = the live dashboard renders a
    # "live ingest not configured" panel instead.
    live_rtmp_base_url: str | None = None

    # Phase L.2 — auto-clip after a live stream ends. When True (default),
    # the MediaMTX `live/ended` webhook kicks the full clip pipeline on the
    # recording automatically (transcribe → detect → cut → score), so a
    # streamer who pushed through our RTMP relay gets clips ready to publish
    # with zero dashboard interaction. Set False to keep live ingest as
    # recording-only (operator triggers "Run pipeline" manually).
    live_auto_clip_enabled: bool = Field(
        default=True,
        validation_alias="CHALYBCLIP_LIVE_AUTO_CLIP",
    )

    # Phase L.2 / Path B — object storage the live-ingest service (the
    # separate `chalybclip-live` MediaMTX deployment) uploads recordings to,
    # and that ChalyClip pulls them from. Vendor-neutral: ANY S3-compatible
    # store works via `endpoint` — Cloudflare R2 (recommended: $0 egress, so
    # the once-per-stream download is free; cheapest for video), Supabase
    # Storage, MinIO, Backblaze B2, AWS S3, … When the bucket is UNSET, live
    # ingest falls back to reading a shared `/data` volume (Path A).
    # Recordings land at `<prefix>/<stream_id>/<file>.mp4`.
    #
    # Cloudflare R2 example:
    #   CHALYBCLIP_LIVE_STORAGE_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    #   CHALYBCLIP_LIVE_STORAGE_REGION=auto
    #   (+ R2 API token access key id / secret)
    live_recording_storage_bucket: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_LIVE_STORAGE_BUCKET"
    )
    live_recording_storage_endpoint: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_LIVE_STORAGE_ENDPOINT"
    )
    live_recording_storage_access_key_id: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_LIVE_STORAGE_ACCESS_KEY_ID"
    )
    live_recording_storage_secret_access_key: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_LIVE_STORAGE_SECRET_ACCESS_KEY"
    )
    live_recording_storage_prefix: str = Field(
        default="live", validation_alias="CHALYBCLIP_LIVE_STORAGE_PREFIX"
    )
    live_recording_storage_region: str = Field(
        default="auto", validation_alias="CHALYBCLIP_LIVE_STORAGE_REGION"
    )

    # Phase 1 (object-storage offload) — the general ARTIFACT store. Renders
    # (and later sources/clips) are pushed here so the local Railway volume is
    # only bounded scratch, and the publisher fetches a presigned BUCKET url
    # (durable object, generous expiry, $0 R2 egress, zero load on our box)
    # instead of our time-boxed `/api/internal/clip` endpoint. Vendor-neutral
    # S3 (Cloudflare R2 recommended). When the bucket is UNSET, everything
    # falls back to local-disk serving (current behavior) — fully optional.
    #   CHALYBCLIP_OBJECT_STORAGE_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    #   CHALYBCLIP_OBJECT_STORAGE_REGION=auto  (+ R2 API access key id / secret)
    object_storage_bucket: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_OBJECT_STORAGE_BUCKET"
    )
    object_storage_endpoint: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_OBJECT_STORAGE_ENDPOINT"
    )
    object_storage_access_key_id: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_OBJECT_STORAGE_ACCESS_KEY_ID"
    )
    object_storage_secret_access_key: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_OBJECT_STORAGE_SECRET_ACCESS_KEY"
    )
    object_storage_prefix: str = Field(
        default="artifacts", validation_alias="CHALYBCLIP_OBJECT_STORAGE_PREFIX"
    )
    object_storage_region: str = Field(
        default="auto", validation_alias="CHALYBCLIP_OBJECT_STORAGE_REGION"
    )
    # Public base URL of the bucket (R2 public bucket r2.dev URL or a custom
    # domain bound to it). When set, the publisher gets a STABLE, non-expiring
    # object URL — the proper 403 fix, since presigned urls cap at 7 days
    # (SigV4) which is shorter than the vendor's retry tail. The clip is
    # public the moment it's published anyway, so the leak window is moot.
    # Unset → fall back to presigned (≤7d) urls.
    #   CHALYBCLIP_OBJECT_STORAGE_PUBLIC_BASE_URL=https://pub-<hash>.r2.dev
    object_storage_public_base_url: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_OBJECT_STORAGE_PUBLIC_BASE_URL"
    )
    # Presigned-URL lifetime handed to the publisher. Generous by default —
    # the object is durable, so this only bounds how long ONE url is valid;
    # the vendor's retry/reschedule tail can be days (see the 14d local-serve
    # floor this replaces).
    object_storage_presign_ttl_s: int = Field(
        default=14 * 24 * 3600,
        validation_alias="CHALYBCLIP_OBJECT_STORAGE_PRESIGN_TTL_S",
    )

    # Global cap on concurrently-running heavy pipelines (download + Whisper
    # + ffmpeg render). The in-process dispatcher gates every launch — API,
    # channel-poll, AND recovery-sweep — through one semaphore, so a redeploy
    # that re-dispatches a backlog (or a channel-poll burst) can't stampede
    # CPU/RAM/disk like the 11 vCPU spike we saw. Excess launches queue and
    # run as slots free. Default 2 — tune to the host's cores. Under Modal
    # (Phase 2) each job is isolated, so this becomes a dispatch-rate guard.
    max_concurrent_pipelines: int = Field(
        default=2, ge=1, validation_alias="CHALYBCLIP_MAX_CONCURRENT_PIPELINES"
    )

    # Slice F.8 — JobDispatcher selection. "in_process" runs pipeline
    # work via FastAPI BackgroundTasks on this host (current behavior).
    # "modal" hands the job off to a Modal app (planned in F.10+ once
    # the chalyb deployment lands).
    job_dispatcher: str = "in_process"
    # Hard cap on the transcribe step. faster-whisper occasionally hangs on
    # CUDA — silent stall, no exception, the asyncio.to_thread call never
    # returns. Without a timeout the whole pipeline waits forever and the
    # dashboard's progress card stays stuck on the pulsing dot.
    #
    # Slice O.12 — the OLD default of 1800 (30 min) was too tight for
    # multi-hour VODs. The pipeline now scales timeouts proportional to
    # stream duration via `whisper_timeout_multiplier`. The static
    # `whisper_timeout_s` value is the FLOOR (minimum) — actual timeout
    # = `max(whisper_timeout_s, duration_s * whisper_timeout_multiplier)`.
    #
    # 4× realtime is a generous cap on a healthy CUDA setup (whisper
    # typically runs 3-10× realtime; the buffer absorbs cold-start +
    # disk-flush + paging hiccups). On a slow CPU/GPU bump to 8×.
    whisper_timeout_s: float = 1800.0
    whisper_timeout_multiplier: float = 4.0

    # Slice O.12 — analyze_video (PySceneDetect + visual signals)
    # timeout multiplier. Actual cap = max(floor, duration_s * multiplier).
    # Slice O.27 — bumped 0.5× → 4× because the 0.5× value assumed GPU
    # but most prod hosts (Railway, Fly, Render) run on CPU where
    # PySceneDetect runs at ~1× realtime. A 87-second video hit the
    # 120-second floor with the old multiplier and skipped the step
    # entirely; 4× gives PySceneDetect 4 minutes on that video (plenty)
    # and proportionally more on longer ones.
    analyze_video_timeout_multiplier: float = 4.0

    # Admin tenants (CHALYBCLIP_ADMIN_TENANT_IDS) bypass the per-step
    # ceilings entirely. Useful for the operator dogfooding multi-hour
    # streams before paying users see them. Set to False to apply the
    # same caps to admins (safer for cost control once paying users land).
    admin_uncapped_pipeline: bool = True

    log_level: str = "INFO"
    log_format: str = "console"

    llm_default_provider: str = "anthropic"
    llm_default_quality: str = "standard"

    # yt-dlp browser-cookie pass-through. Set to "chrome" / "edge" / "firefox"
    # / "brave" / "chromium" to authenticate yt-dlp using cookies from a
    # logged-in browser session. Required for Kick (which 403s anonymous
    # scraping) and useful for age-gated YouTube. The browser must have
    # visited the platform at least once.
    cookies_from_browser: str | None = None

    # Alternative to cookies_from_browser: an absolute path to a Netscape-
    # format cookies.txt file (export with a "Get cookies.txt" browser
    # extension). yt-dlp reads the file directly so the browser can stay
    # open. When both this and cookies_from_browser are set, the file wins.
    cookies_file: str | None = None

    # Route ALL yt-dlp egress (VOD download + channel listing) and the Kick
    # API call through a proxy. A RESIDENTIAL/mobile proxy makes the request
    # look like a home connection instead of a datacenter — which is what
    # actually stops YouTube's "confirm you're not a bot" gate and Kick's
    # Cloudflare 403, WITHOUT cookies (which expire and get accounts banned).
    # This is the sustainable fix; cookies become an optional fallback for
    # login-walled content (e.g. X video). Format is a full proxy URL:
    #   http://user:pass@host:port  or  socks5://user:pass@host:port
    # Unset → direct connection (current behavior).
    ytdlp_proxy: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_YTDLP_PROXY"
    )

    # FREE, zero-infra YouTube bot-gate mitigation (no proxy, no cookies).
    # yt-dlp can dodge the "confirm you're not a bot" gate by extracting via a
    # different YouTube player client. Comma-separated, tried in order, e.g.
    # "tv,web_safari,mweb" (good gate-dodgers as of 2026; `web`/`android` now
    # demand a PO token). Maps to extractor_args youtube:player_client.
    # YouTube-SCOPED — never touches Twitch/Kick. Unset → yt-dlp's default.
    ytdlp_player_client: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_YTDLP_PLAYER_CLIENT"
    )

    # Base URL of a self-hosted bgutil PO-token provider (needs the yt-dlp
    # plugin `bgutil-ytdlp-pot-provider` installed + the provider running as a
    # small sidecar). Lets yt-dlp mint the Proof-of-Origin tokens YouTube
    # increasingly requires WITHOUT a browser or cookies. Maps to
    # extractor_args youtube:getpot_bgutil_baseurl. Unset → no PO provider.
    ytdlp_po_provider_url: str | None = Field(
        default=None, validation_alias="CHALYBCLIP_YTDLP_PO_PROVIDER_URL"
    )

    # Task 2b — VOD download speed.
    #
    # `ytdlp_concurrent_fragments` — yt-dlp's `concurrent_fragment_downloads`
    # option. Twitch / Kick / YouTube VODs are HLS / DASH segmented; the
    # native downloader fetches one segment at a time by default. Bumping
    # this to 16 typically gives 3-10× speedup on bandwidth-constrained
    # hosts (Railway egress, fly.io edge). Drop to 1 if the upstream CDN
    # rate-limits parallel fetches and you see 429s.
    ytdlp_concurrent_fragments: int = 16
    #
    # `ytdlp_use_aria2c` — opt into aria2c as the external downloader.
    # aria2c parallelises within a single segment via byte ranges; on
    # multi-GB sources it beats yt-dlp's native loop. Only takes effect
    # when aria2c is actually on PATH (we probe at request time and
    # silently fall back to the native downloader otherwise). False by
    # default because aria2c isn't installed on the Railway image yet.
    ytdlp_use_aria2c: bool = False
    #
    # `max_source_height` — cap the resolution yt-dlp asks for. Clips top
    # out at 1080×1920 for free/pro and 4K for all_access, so pulling
    # the platform's "best available" (often 4K+ on Kick / YouTube)
    # wastes bandwidth. Set to 1080 for Phase 0 defaults; tenants on the
    # all_access tier can bump this per-stream once the resolution-tier
    # plumbing surfaces it. 0 disables the cap (use platform's best).
    max_source_height: int = 1080

    # Chalyb integration — slice NX.1.
    # Both are symmetric secrets shared with Chalyb (which sees them as
    # CHALYBCLIP_ADMIN_TOKEN and CHALYBCLIP_SSO_SECRET on its side). When unset,
    # the new admin + SSO endpoints reject all traffic with 503 so we never
    # accept un-authenticated provisioning calls.
    #
    # We override the auto-prefix so the env vars are the simple
    # CHALYB_ADMIN_TOKEN / CHALYB_SSO_SECRET names that the spec doc
    # promises (instead of CHALYBCLIP_CHALYB_... which is the prefix the rest
    # of this Settings class uses).
    chalyb_admin_token: str | None = Field(
        default=None,
        validation_alias="CHALYB_ADMIN_TOKEN",
    )
    chalyb_sso_secret: str | None = Field(
        default=None,
        validation_alias="CHALYB_SSO_SECRET",
    )
    # Slice NX.3 — outbound URL to Chalyb. Used to push usage events back
    # to the platform after each LLM call. When unset, usage reporting is a
    # no-op (ChalyClip can run standalone without Chalyb knowing about it).
    chalyb_base_url: str | None = Field(
        default=None,
        validation_alias="CHALYB_BASE_URL",
    )
    # Token T3 — per-run BASE CHARGE in USD micros. The raw API cost of a
    # run (transcription + occasional Claude) can be near-zero, but each run
    # consumes the FULL stack we pay for: ChalyClip's Railway compute (ffmpeg
    # + render + orchestration), storage, bandwidth, AND a fair share of the
    # shared platform — Vercel (Chalyb), Supabase, Resend, the domain, etc.
    # This flat charge — reported as an `engine.base` usage event after every
    # successful run — recovers that fully-loaded overhead + margin so a
    # near-free-API run still draws down the quota.
    #
    # Fully-loaded model (estimates; refine with real bills):
    #   marginal (Railway compute+storage+bw)      ~$0.025/run
    #   shared platform allocation (Vercel+Supabase+
    #     Resend+domain, split across engines/runs) ~$0.020/run
    #   = fully-loaded cost ~$0.045 x ~1.3 margin   = ~$0.06/run
    # Default $0.06; at Chalyb's ~$4/1M-token rate that's ~15,000
    # token-equivalents/run. Set to 0 to disable. Drops toward $0.05 at
    # higher run volume (platform amortizes); raise pre-scale for cushion.
    pipeline_base_charge_usd_micros: int = Field(
        default=60_000,
        validation_alias="CHALYBCLIP_PIPELINE_BASE_CHARGE_USD_MICROS",
    )
    # Where ChalyClip is reachable from the public internet. Used as the
    # post-SSO landing origin if we ever need to build absolute URLs.
    # Defaults to localhost for dev. Same override pattern.
    public_url: str = Field(
        default="http://localhost:8000",
        validation_alias="CHALYBCLIP_PUBLIC_URL",
    )
    # Hard cap on a single stream upload's body size, in bytes. Default
    # 5 GiB — generous for a multi-hour VOD MP4. Streamed-chunk read
    # enforces this; anything over the cap raises 413 mid-upload so the
    # disk write stops immediately. Set lower on tight environments.
    max_upload_bytes: int = Field(
        default=5 * 1024 * 1024 * 1024,
        validation_alias="CHALYBCLIP_MAX_UPLOAD_BYTES",
    )

    # Slice O.9 — admin tenant allowlist. Comma-separated tenant IDs that
    # see operator-only nav items (LLM spend + LLM settings). All other
    # tenants get a creator-only nav. Empty / unset → no tenants are
    # admins (creator UX for everyone, including local dev). Set this to
    # your own tenant id to see the admin pages back.
    admin_tenant_ids: str = ""

    # Slice O.22 — chalyb is the gatekeeper now. ChalyClip no longer
    # serves a token-paste login form. Anyone hitting `/dashboard/login`
    # or any /dashboard/* page without a valid session cookie is bounced
    # to this URL. Production: https://chalyb.com/login. Set blank
    # to fall back to the legacy in-house login page (kept as
    # emergency-admin backstop; see dashboard.login_form).
    chalyb_login_url: str = Field(
        default="https://chalyb.com/login",
        validation_alias="CHALYB_LOGIN_URL",
    )

    # ------------------------------------------------------------------
    # Zernio integration — the multi-platform publish layer (replaces
    # upload-post.com). We hand Zernio a video URL + target accounts,
    # they do the actual OAuth + posting against TikTok / IG / YT / X /
    # LinkedIn / etc. (15 platforms). Multi-tenant via their `profileId`
    # model — one Zernio profileId per ChalyClip tenant.
    # ------------------------------------------------------------------
    #
    # zernio_api_key — single company-wide API key from Zernio (a
    # 67-char `sk_...` secret). Authenticates every call. Tenants never
    # see it. Header shape: `Authorization: Bearer <key>`.
    #
    # Env var: CHALYBCLIP_ZERNIO_API_KEY (per the class-level
    # env_prefix="CHALYBCLIP_"). Be explicit about this in the deploy
    # docs — the raw `ZERNIO_API_KEY` (what Zernio's own SDKs read by
    # default) is silently ignored.
    zernio_api_key: str | None = None

    # Base URL. Override only for tests / staging environments.
    # Env var: CHALYBCLIP_ZERNIO_BASE_URL.
    zernio_base_url: str = "https://zernio.com/api/v1"

    # Shared secret used to verify inbound Zernio webhook signatures.
    # Env var: CHALYBCLIP_ZERNIO_WEBHOOK_SECRET. When unset, the webhook
    # receiver returns 503 (refuses to trust unsigned-verifiable events).
    zernio_webhook_secret: str | None = None

    # ---------- Publish & Engagement Hub (internal service API) ----------

    # Service-to-service bearer tokens for /api/internal/v1/* — the
    # entry point ChalyOBS and Chalyb engines publish through. Format:
    # comma-separated `name:token` pairs ("chalybobs:tok_abc,chalyb:tok_x").
    # Env var: CHALYBCLIP_HUB_SERVICE_TOKENS. When unset, the internal
    # publish API returns 503 (no anonymous service access, ever).
    hub_service_tokens: str | None = None

    # Anti-spam policy for batch publishes: at most this many hub posts
    # per platform per tenant per UTC day; batch overflow rolls into the
    # next day's window. Env: CHALYBCLIP_HUB_MAX_POSTS_PER_PLATFORM_PER_DAY.
    hub_max_posts_per_platform_per_day: int = 4

    # On a post.failed with a transient error class, the hub fires ONE
    # automatic retry after this delay (seconds), then stops. 0 retries
    # immediately (inline); a NEGATIVE value disables auto-retry.
    # Env: CHALYBCLIP_HUB_AUTO_RETRY_DELAY_S.
    hub_auto_retry_delay_s: float = 600.0

    # Growth layer (phase 10) anti-spam: max broadcasts a tenant may
    # SEND per UTC day. A broadcast is a mass DM — a mistake is
    # irreversible spam — so the cap is deliberately low.
    # Env: CHALYBCLIP_HUB_MAX_BROADCASTS_PER_DAY.
    hub_max_broadcasts_per_day: int = 1

    # Phase-12 feature flags (default OFF). These carry extra cost /
    # complexity (Meta ad spend, WhatsApp per-number + per-message
    # fees), so the seams are built but the routes 404 and the UI is
    # hidden until a flag is turned on.
    # Env: CHALYBCLIP_FEATURE_WHATSAPP, CHALYBCLIP_FEATURE_ADS.
    feature_whatsapp: bool = False
    feature_ads: bool = False

    def db_target(self) -> str:
        """The connection target `Database(...)` should open. See
        `resolve_db_target` — this is the method form for direct callers."""
        return resolve_db_target(self)

    def hub_service_token_map(self) -> dict[str, str]:
        """Parse hub_service_tokens into {token: consumer_name}.

        Keyed by token (the lookup direction auth needs). Malformed
        pairs are skipped — fail-closed, never fail-open."""
        out: dict[str, str] = {}
        for pair in (self.hub_service_tokens or "").split(","):
            name, _, token = pair.strip().partition(":")
            if name and token:
                out[token.strip()] = name.strip()
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor; tests can call `get_settings.cache_clear()`."""
    return Settings()


def resolve_db_target(settings: object) -> str:
    """The connection target `Database(...)` should open: the Postgres DSN
    when `DATABASE_URL` is configured, else the local SQLite path.

    Single source of truth for the `database_url or db_path` choice. Reads
    attributes defensively (not via a method) so any settings-like object
    that only exposes `db_path` — e.g. lightweight test doubles — keeps
    working without needing to mirror the full Settings interface.
    """
    return getattr(settings, "database_url", None) or settings.db_path  # type: ignore[attr-defined, no-any-return]

"""ChalybClip CLI entry point.

Phase 0 commands (see PHASE_0.md):
    chalybclip ingest <vod_url>
    chalybclip transcribe <stream_id>
    chalybclip detect <stream_id>
    chalybclip cut <stream_id>
    chalybclip process <vod_url>          # orchestrates all of the above

Phase 1 admin commands (see PHASE_1.md):
    chalybclip db init
    chalybclip tenants add <id> "<name>"
    chalybclip tenants list
    chalybclip tokens issue --tenant <id> [--scope full|read]
    chalybclip tokens list   --tenant <id>
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from chalybclip.db import Database
    from chalybclip.db.models import ApiTokenRow, Tenant

app = typer.Typer(
    name="chalybclip",
    help="VOD-to-clips pipeline.",
    no_args_is_help=True,
)

db_app = typer.Typer(name="db", help="Database admin commands.", no_args_is_help=True)
tenants_app = typer.Typer(name="tenants", help="Tenant management.", no_args_is_help=True)
tokens_app = typer.Typer(name="tokens", help="API token management.", no_args_is_help=True)
webhooks_app = typer.Typer(
    name="webhooks", help="Webhook subscription dispatch.", no_args_is_help=True
)
mcp_app = typer.Typer(
    name="mcp", help="MCP server (stdio) for external agents.", no_args_is_help=True
)
retention_app = typer.Typer(
    name="retention",
    help="Per-tenant retention sweeper (voice-markers spec slice E.1).",
    no_args_is_help=True,
)
drive_app = typer.Typer(
    name="drive",
    help="Google Drive folder watches (voice-markers spec slice E.4).",
    no_args_is_help=True,
)
channel_app = typer.Typer(
    name="channel",
    help="Connected creator channels — auto-ingest new VODs (YouTube/Twitch/Kick).",
    no_args_is_help=True,
)
publish_app = typer.Typer(
    name="publish",
    help="Publishing ops — reprocess the scheduled queue, etc.",
    no_args_is_help=True,
)
app.add_typer(db_app)
app.add_typer(tenants_app)
app.add_typer(tokens_app)
app.add_typer(webhooks_app)
app.add_typer(mcp_app)
app.add_typer(retention_app)
app.add_typer(drive_app)
app.add_typer(channel_app)
app.add_typer(publish_app)


@mcp_app.command("serve")
def mcp_serve_cmd(
    token: str | None = typer.Option(
        None,
        "--token",
        help="Raw API token (tok_...). Falls back to CHALYBCLIP_API_TOKEN env var.",
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Boot the MCP server (stdio transport).

    The token resolves to a tenant id via api_tokens.lookup_by_hash; every
    tool call binds that tenant via bound_tenant(). No cross-tenant access;
    rejected tokens fail fast at boot.
    """
    from chalybclip.mcp_server import run_stdio_server
    from chalybclip.settings import get_settings, resolve_db_target

    # Same resolver as the web app + every other CLI command: prod runs on
    # Postgres via DATABASE_URL, and only an explicit --db-path should open
    # the SQLite fallback. (A DSN must stay a string — no Path() wrapping.)
    resolved_db: str | Path = db_path or resolve_db_target(get_settings())
    try:
        run_stdio_server(db_path=resolved_db, raw_token=token)
    except Exception as e:
        typer.echo(f"mcp server failed: {e}", err=True)
        raise typer.Exit(code=1) from e


@webhooks_app.command("send")
def webhooks_send_cmd(
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant id to drain."),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """One-shot drain pass over the tenant's active webhook subscriptions."""
    from chalybclip.db import apply_migrations
    from chalybclip.webhooks import run_webhook_dispatch

    async def _run() -> None:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            outcome = await run_webhook_dispatch(tenant_id, db)
            typer.echo(
                f"webhooks delivered={outcome.delivered} "
                f"failed={outcome.failed} disabled={outcome.disabled}"
            )
        finally:
            await db.close()

    asyncio.run(_run())


@webhooks_app.command("register-zernio")
def webhooks_register_zernio_cmd(
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Create-or-update the hub's webhook config on Zernio (idempotent).

    Points Zernio at {CHALYBCLIP_PUBLIC_URL}/api/webhooks/zernio with the
    hub's event list, keyed by CHALYBCLIP_ZERNIO_WEBHOOK_SECRET. Re-run
    after changing PUBLIC_URL or the secret — also re-activates a
    webhook Zernio auto-disabled after delivery failures.
    """
    import json

    from chalybclip.integrations.zernio import register_zernio_webhook
    from chalybclip.integrations.zernio.client import ZernioClient
    from chalybclip.settings import get_settings

    settings = get_settings()
    if not settings.zernio_api_key:
        typer.echo("error: CHALYBCLIP_ZERNIO_API_KEY is not configured", err=True)
        raise typer.Exit(code=1)
    secret = (settings.zernio_webhook_secret or "").strip()
    if not secret:
        typer.echo(
            "error: CHALYBCLIP_ZERNIO_WEBHOOK_SECRET is not configured "
            "(the receiver refuses unverifiable events)",
            err=True,
        )
        raise typer.Exit(code=1)
    public_url = (settings.public_url or "").rstrip("/")
    if not public_url or public_url.startswith(("http://localhost", "http://127.")):
        typer.echo(
            f"error: CHALYBCLIP_PUBLIC_URL ({public_url or 'unset'}) is not a "
            "publicly reachable origin — Zernio could not deliver to it",
            err=True,
        )
        raise typer.Exit(code=1)

    async def _run() -> None:
        client = ZernioClient(
            api_key=settings.zernio_api_key or "",
            base_url=settings.zernio_base_url,
        )
        result = await register_zernio_webhook(
            client,
            url=f"{public_url}/api/webhooks/zernio",
            secret=secret,
        )
        if json_out:
            typer.echo(json.dumps(result))
        else:
            typer.echo(
                f"webhook {result['action']}: id={result['webhook_id']} "
                f"url={result['url']} events={len(result['events'])}"
            )

    asyncio.run(_run())


@webhooks_app.command("snapshot-analytics")
def webhooks_snapshot_analytics_cmd(
    tenant_id: str = typer.Option(
        "", "--tenant", help="One tenant; omit for all bound tenants."
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Capture today's per-post metric snapshot (Hub phase 7).

    Run daily (cron). Idempotent per (post, UTC day) — a same-day
    re-run refreshes rows instead of duplicating. Persist-only: no
    scoring, no ML, just history for the clip-selection feedback loop.
    """
    from chalybclip.db import TenantsRepo, apply_migrations
    from chalybclip.integrations.zernio.client import ZernioClient
    from chalybclip.publish.analytics_service import snapshot_tenant
    from chalybclip.settings import get_settings

    settings = get_settings()
    if not settings.zernio_api_key:
        typer.echo("error: CHALYBCLIP_ZERNIO_API_KEY is not configured", err=True)
        raise typer.Exit(code=1)

    async def _run() -> None:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            client = ZernioClient(
                api_key=settings.zernio_api_key or "",
                base_url=settings.zernio_base_url,
            )
            tenants = (
                [tenant_id]
                if tenant_id
                else [t.id for t in await TenantsRepo(db).list_all()]
            )
            total = 0
            for tid in tenants:
                total += await snapshot_tenant(db, tid, client=client)
            typer.echo(f"snapshot tenants={len(tenants)} posts={total}")
        finally:
            await db.close()

    asyncio.run(_run())


@webhooks_app.command("community-digest")
def webhooks_community_digest_cmd(
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Send the weekly community digest to every tenant that opted in
    (Hub phase 11). Run weekly (cron). Default-off per tenant."""
    from chalybclip.db import ZernioCommunityRepo, apply_migrations
    from chalybclip.integrations.zernio.client import ZernioClient
    from chalybclip.publish.analytics_service import send_weekly_digest
    from chalybclip.settings import get_settings

    settings = get_settings()
    if not settings.zernio_api_key:
        typer.echo("error: CHALYBCLIP_ZERNIO_API_KEY is not configured", err=True)
        raise typer.Exit(code=1)

    async def _run() -> None:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            client = ZernioClient(
                api_key=settings.zernio_api_key or "",
                base_url=settings.zernio_base_url,
            )
            rows = await ZernioCommunityRepo(db).list_digest_tenants()
            sent = 0
            for row in rows:
                if await send_weekly_digest(db, row["tenant_id"], client=client):
                    sent += 1
            typer.echo(f"community-digest tenants={len(rows)} sent={sent}")
        finally:
            await db.close()

    asyncio.run(_run())


@publish_app.command("reprocess-queue")
def publish_reprocess_queue_cmd(
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant id to reprocess."),
    db_path: Path | None = typer.Option(None, "--db-path"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change; touch nothing."
    ),
    reschedule: bool = typer.Option(
        True, "--reschedule/--no-reschedule",
        help="After cleanup, re-render + re-schedule via the rulebook. "
        "Use --no-reschedule to do cleanup only, then click 'Auto-programar "
        "todo' in the dashboard (the proven web render path).",
    ),
) -> None:
    """Reprocess the scheduled queue through the new pipeline.

    Cancels the tenant's scheduled Zernio posts, resets each clip (approved +
    fresh overlay: hook + captions + source-credit marca), invalidates the
    cached render (local + object storage) so the re-render bakes the new
    overlays + the real outro, then (unless --no-reschedule) re-renders and
    re-schedules them under the per-platform rulebook.
    """
    import time

    from chalybclip.db import (
        AutoprogLocksRepo,
        AutopublishSettingsRepo,
        TenantsRepo,
        apply_migrations,
    )
    from chalybclip.integrations.zernio.client import ZernioClient
    from chalybclip.publish.reprocess import reprocess_scheduled_queue
    from chalybclip.settings import get_settings

    settings = get_settings()
    if not settings.zernio_api_key:
        typer.echo("error: CHALYBCLIP_ZERNIO_API_KEY is not configured", err=True)
        raise typer.Exit(code=1)

    async def _run() -> None:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            client = ZernioClient(
                api_key=settings.zernio_api_key or "",
                base_url=settings.zernio_base_url,
            )
            summary = await reprocess_scheduled_queue(
                db, tenant_id, client=client, settings=settings, dry_run=dry_run,
            )
            typer.echo(
                f"scheduled_found={summary.scheduled_found} "
                f"canceled={summary.canceled} cancel_failed={summary.cancel_failed} "
                f"clips_reset={summary.clips_reset} "
                f"skipped_missing_source={len(summary.skipped_missing_source)}"
            )
            if summary.skipped_missing_source:
                # Left untouched: their source file is gone, so a reset could
                # never re-render — the live scheduled post stays as-is.
                typer.echo(
                    "skipped (source missing): "
                    + ", ".join(summary.skipped_missing_source)
                )
            if dry_run:
                typer.echo(f"[dry-run] would reprocess {len(summary.clip_ids)} clip(s)")
                return
            if not reschedule:
                typer.echo(
                    "cleanup done — click 'Auto-programar todo' in the Publish "
                    "Center to re-render + re-schedule the reset clips."
                )
                return
            if not summary.clip_ids:
                typer.echo("nothing to re-schedule")
                return

            # Re-schedule through the rulebook — reuse the dashboard auto-program
            # worker (renders each clip + plans per-platform caps/gaps). Imported
            # lazily so the publish service stays free of the API layer.
            from chalybclip.api.routers.zernio import (
                _AUTOPROG,
                _account_map,
                _build_client,
                _connected_platforms,
                _run_growth_autoprog,
            )
            from chalybclip.tiers import zernio_account_limit

            tenant = await TenantsRepo(db).get(tenant_id)
            profile_id = tenant.zernio_profile_id if tenant else None
            base_url = (settings.public_url or "").rstrip("/")
            if not profile_id or not base_url:
                typer.echo(
                    "warn: no Zernio profile or public_url — clips were reset but "
                    "not re-scheduled. Click 'Auto-programar todo' in the dashboard.",
                    err=True,
                )
                return
            sched_client = _build_client()
            accounts = await sched_client.list_accounts(profile_id=profile_id)
            account_map = _account_map(accounts)
            connected = _connected_platforms(accounts)
            s = await AutopublishSettingsRepo(db).get(tenant_id) or {}
            want = [t for t in str(s.get("targets") or "").split(",") if t.strip()]
            targets = [t for t in want if t in connected] or sorted(connected)
            limit = zernio_account_limit(tenant.tier if tenant else "free")
            if limit is not None:
                targets = targets[:limit]

            lock_token = await AutoprogLocksRepo(db).acquire(tenant_id)
            if lock_token is None:
                typer.echo(
                    "warn: an auto-program run is already in progress; clips were "
                    "reset but not re-scheduled. Retry shortly.", err=True,
                )
                return
            _AUTOPROG[tenant_id] = {
                "state": "running", "total": len(summary.clip_ids),
                "done": 0, "scheduled": 0, "failed": 0, "results": [],
                "heartbeat": time.monotonic(),
            }
            await _run_growth_autoprog(
                tenant_id=tenant_id, clip_ids=summary.clip_ids, targets=targets,
                handle_suffix=str(s.get("tag_suffix") or ""), account_map=account_map,
                profile_id=profile_id,
                tenant_tier=(tenant.tier if tenant else None),
                base_url=base_url, session_cookie=None, db_target=db.target,
                lock_token=lock_token,
            )
            prog = _AUTOPROG.get(tenant_id, {})
            typer.echo(
                f"rescheduled={prog.get('scheduled', 0)} failed={prog.get('failed', 0)}"
            )
        finally:
            await db.close()

    asyncio.run(_run())


@app.callback()
def _root(
    log_level: str | None = typer.Option(
        None, "--log-level", help="DEBUG | INFO | WARNING | ERROR (overrides env)."
    ),
    log_format: str | None = typer.Option(
        None, "--log-format", help="`console` (default) or `json` for machine logs."
    ),
) -> None:
    """Configure logging once before any subcommand runs."""
    from chalybclip.logging import configure_logging
    from chalybclip.settings import get_settings

    settings = get_settings()
    configure_logging(
        level=log_level or settings.log_level,
        fmt=log_format or settings.log_format,
    )


@app.command()
def version() -> None:
    """Print the installed version."""
    from chalybclip import __version__

    typer.echo(__version__)


@app.command()
def worker(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8100, "--port", help="Bind port."),
) -> None:
    """Serve the PC worker — the full pipeline on this machine.

    Speaks the same kickoff/poll HTTP contract as the Modal worker, so the
    web box dispatches to it by pointing CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL
    at this app's (tunneled) URL. Requires CHALYBCLIP_WORKER_TOKEN (or
    CHALYBCLIP_MODAL_TOKEN), DATABASE_URL and CHALYBCLIP_OBJECT_STORAGE_BUCKET.
    """
    import uvicorn

    from chalybclip.workers import create_worker_app

    uvicorn.run(create_worker_app(), host=host, port=port, log_level="info")


@app.command()
def ingest(
    vod_url: str = typer.Argument(..., help="VOD URL (Kick / Twitch / YouTube)"),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory for stream artifacts."
    ),
    tenant_id: str = typer.Option(
        "default", "--tenant-id", help="Tenant owning the stream (Phase 0: hardcoded)."
    ),
    stream_id: str | None = typer.Option(
        None, "--stream-id", help="Resume an existing stream (skips ID generation)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-download and re-extract even if outputs exist."
    ),
    chat_replay: Path | None = typer.Option(
        None,
        "--chat-replay",
        help="Path to a JSONL of chat messages (Phase 1 doesn't fetch from platforms).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the Stream as JSON to stdout."),
) -> None:
    """Download a VOD and extract its audio. Idempotent."""
    from chalybclip.errors import IngestError
    from chalybclip.ingest import ingest_vod

    try:
        stream = asyncio.run(
            ingest_vod(
                tenant_id=tenant_id,
                vod_url=vod_url,
                output_dir=output_dir,
                stream_id=stream_id,
                force=force,
                chat_replay_source=chat_replay,
            )
        )
    except IngestError as e:
        typer.echo(f"ingest failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(stream.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {stream.id}")
    typer.echo(f"  platform: {stream.platform}")
    typer.echo(f"  title:    {stream.title or '(unknown)'}")
    typer.echo(f"  channel:  {stream.channel or '(unknown)'}")
    typer.echo(f"  duration: {stream.duration_s:.1f}s")
    typer.echo(f"  video:    {stream.source_video_path}")
    typer.echo(f"  audio:    {stream.source_audio_path}")


@app.command()
def transcribe(
    stream_id: str = typer.Argument(..., help="Stream ID produced by `chalybclip ingest`."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding `<stream_id>/`."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to CHALYBCLIP_DEFAULT_TENANT_ID."
    ),
    model_size: str | None = typer.Option(
        None, "--model", help="Whisper model size (overrides CHALYBCLIP_WHISPER_MODEL)."
    ),
    device: str | None = typer.Option(
        None, "--device", help="`cuda` or `cpu` (overrides CHALYBCLIP_WHISPER_DEVICE)."
    ),
    compute_type: str | None = typer.Option(
        None, "--compute-type", help="e.g. `float16`, `int8` (overrides env)."
    ),
    language: str = typer.Option("es", "--language", help="ISO 639-1 code, or `auto`."),
    force: bool = typer.Option(
        False, "--force", help="Re-transcribe even if `transcript.json` exists."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full Transcript as JSON."),
) -> None:
    """Run Whisper on an ingested stream and write `transcript.json`."""
    from chalybclip.errors import IngestError, TranscriptionError
    from chalybclip.ingest import load_stream
    from chalybclip.settings import get_settings
    from chalybclip.transcribe import transcribe as run_transcribe

    settings = get_settings()
    stream_dir = Path(output_dir).resolve() / stream_id

    try:
        stream = load_stream(stream_dir)
        transcript = asyncio.run(
            run_transcribe(
                tenant_id=tenant_id or settings.default_tenant_id,
                stream=stream,
                model_size=model_size or settings.whisper_model,
                device=device or settings.whisper_device,
                compute_type=compute_type or settings.whisper_compute_type,
                language=None if language == "auto" else language,
                force=force,
            )
        )
    except (IngestError, TranscriptionError) as e:
        typer.echo(f"transcribe failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(transcript.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {transcript.stream_id}")
    typer.echo(f"  model:    {transcript.model}")
    typer.echo(f"  language: {transcript.language}")
    typer.echo(f"  duration: {transcript.duration_s:.1f}s")
    typer.echo(f"  segments: {len(transcript.segments)}")
    word_count = sum(len(s.words) for s in transcript.segments)
    typer.echo(f"  words:    {word_count}")


@app.command(name="analyze-video")
def analyze_video_cmd(
    stream_id: str = typer.Argument(..., help="Stream ID produced by `chalybclip ingest`."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding `<stream_id>/`."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to CHALYBCLIP_DEFAULT_TENANT_ID."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run even when `visual_signals.json` exists."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the VisualSignalTrack as JSON."),
) -> None:
    """Run the local vision pipeline (scene cuts + motion + face/emotion)."""
    from chalybclip.errors import DetectionError, IngestError
    from chalybclip.ingest import load_stream
    from chalybclip.settings import get_settings
    from chalybclip.vision import analyze_video

    settings = get_settings()
    stream_dir = Path(output_dir).resolve() / stream_id
    effective_tenant = tenant_id or settings.default_tenant_id

    try:
        stream = load_stream(stream_dir)
        track = asyncio.run(
            analyze_video(
                tenant_id=effective_tenant,
                stream=stream,
                output_dir=Path(output_dir).resolve(),
                force=force,
            )
        )
    except (IngestError, DetectionError) as e:
        typer.echo(f"analyze-video failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(track.model_dump_json(indent=2))
        return

    cut_count = sum(1 for s in track.signals if s.scene_cut)
    smile_count = sum(1 for s in track.signals if s.face_emotion == "smile")
    motion_max = max(
        (s.motion_energy for s in track.signals if s.motion_energy is not None),
        default=0.0,
    )
    typer.echo(f"stream_id:  {track.stream_id}")
    typer.echo(f"  seconds:  {len(track.signals)}")
    typer.echo(f"  cuts:     {cut_count}")
    typer.echo(f"  smiles:   {smile_count}")
    typer.echo(f"  motion (max): {motion_max:.4f}")


@app.command()
def detect(
    stream_id: str = typer.Argument(..., help="Stream ID produced by `chalybclip ingest`."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding `<stream_id>/`."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to CHALYBCLIP_DEFAULT_TENANT_ID."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Path to chalybclip.yaml (defaults to config/chalybclip.yaml)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print candidates as JSON."),
) -> None:
    """Detect candidates (voice + chat + audio + visual) in an already-transcribed stream."""
    from chalybclip.config import load_config
    from chalybclip.detect import detect_candidates, save_candidates
    from chalybclip.detect.models import CandidateBatch
    from chalybclip.errors import DetectionError, IngestError, TranscriptionError
    from chalybclip.ingest import load_chat_replay, load_stream
    from chalybclip.settings import get_settings
    from chalybclip.transcribe import load_transcript
    from chalybclip.vision import load_visual_signals

    settings = get_settings()
    stream_dir = Path(output_dir).resolve() / stream_id
    effective_tenant = tenant_id or settings.default_tenant_id

    try:
        stream = load_stream(stream_dir)
        transcript = load_transcript(stream_dir)
        config = load_config(config_path)
        chat_replay = load_chat_replay(stream_dir, stream_id=stream.id, tenant_id=effective_tenant)
        visual_track = load_visual_signals(stream_dir)
        candidates = detect_candidates(
            tenant_id=effective_tenant,
            stream=stream,
            transcript=transcript,
            config=config.detection,
            chat_replay=chat_replay,
            visual_track=visual_track,
        )
    except (IngestError, TranscriptionError, DetectionError) as e:
        typer.echo(f"detect failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    batch = CandidateBatch(stream_id=stream.id, tenant_id=effective_tenant, candidates=candidates)
    save_candidates(stream_dir, batch)

    if json_output:
        typer.echo(batch.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {batch.stream_id}")
    typer.echo(f"  candidates: {len(candidates)}")
    for c in candidates:
        phrase = c.evidence.get("phrase", "?")
        snippet = c.evidence.get("transcript_snippet", "")
        typer.echo(
            f"  [{c.timestamp:>7.1f}s]  score={c.score:.3f}  phrase={phrase!r}  snippet={snippet!r}"
        )


@app.command()
def cut(
    stream_id: str = typer.Argument(..., help="Stream ID produced by `chalybclip ingest`."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding `<stream_id>/`."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to CHALYBCLIP_DEFAULT_TENANT_ID."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Path to chalybclip.yaml (defaults to config/chalybclip.yaml)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-cut every clip even if `clips_manifest.json` exists."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the manifest as JSON."),
) -> None:
    """Cut + 9:16 reformat one clip per detected candidate."""
    from chalybclip.clip import cut_clips
    from chalybclip.clip.models import ClipManifest
    from chalybclip.config import load_config
    from chalybclip.detect import load_candidates
    from chalybclip.errors import ClipError, DetectionError, IngestError
    from chalybclip.ingest import load_stream
    from chalybclip.settings import get_settings

    settings = get_settings()
    stream_dir = Path(output_dir).resolve() / stream_id
    effective_tenant = tenant_id or settings.default_tenant_id

    try:
        stream = load_stream(stream_dir)
        batch = load_candidates(stream_dir)
        config = load_config(config_path)
        clips = asyncio.run(
            cut_clips(
                tenant_id=effective_tenant,
                stream=stream,
                candidates=batch.candidates,
                output_dir=Path(output_dir).resolve(),
                config=config.clip,
                force=force,
            )
        )
    except (IngestError, DetectionError, ClipError) as e:
        typer.echo(f"cut failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    manifest = ClipManifest(stream_id=stream.id, tenant_id=effective_tenant, clips=clips)

    if json_output:
        typer.echo(manifest.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {manifest.stream_id}")
    typer.echo(f"  clips:    {len(clips)}")
    for c in clips:
        typer.echo(f"  [{c.start_s:>7.1f}s..{c.end_s:>7.1f}s]  {c.id}  -> {c.path}")


@app.command()
def process(
    vod_url: str = typer.Argument(..., help="VOD URL (Kick / Twitch / YouTube)"),
    persona: str = typer.Option(..., "--persona", help="Persona id from personas.yaml."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory for stream artifacts."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to CHALYBCLIP_DEFAULT_TENANT_ID."
    ),
    stream_id: str | None = typer.Option(
        None, "--stream-id", help="Resume an existing stream (skips ID generation)."
    ),
    language: str | None = typer.Option(
        None, "--language", help="ISO 639-1 code; defaults to persona's primary language."
    ),
    n: int = typer.Option(5, "--n", min=1, help="How many variants to generate per clip."),
    quality: str | None = typer.Option(
        None, "--quality", help="Override default quality (`standard` or `premium`)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run every step even when its output exists."
    ),
    no_db: bool = typer.Option(
        False, "--no-db", help="Skip dual-write to SQLite (filesystem only)."
    ),
    db_path: Path | None = typer.Option(
        None, "--db-path", help="Override CHALYBCLIP_DB_PATH for this run."
    ),
    chat_replay: Path | None = typer.Option(
        None,
        "--chat-replay",
        help="Path to a JSONL of chat messages, fed to the chat heat detector.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the full manifest as JSON when complete."
    ),
) -> None:
    """Run the full ingest -> transcribe -> detect -> cut -> variants pipeline."""
    from chalybclip.errors import (
        ClipError,
        DetectionError,
        IngestError,
        LLMError,
        TenancyError,
        TranscriptionError,
        VariantError,
    )
    from chalybclip.llm.config import Quality
    from chalybclip.pipeline import process_vod
    from chalybclip.settings import get_settings

    settings = get_settings()
    effective_tenant = tenant_id or settings.default_tenant_id
    quality_arg: Quality | None = None
    if quality is not None:
        if quality not in ("standard", "premium"):
            typer.echo(f"--quality must be 'standard' or 'premium', got {quality!r}", err=True)
            raise typer.Exit(code=2)
        quality_arg = quality  # type: ignore[assignment]

    effective_db_path = None if no_db else (str(db_path) if db_path else settings.db_path)

    try:
        manifest = asyncio.run(
            process_vod(
                tenant_id=effective_tenant,
                vod_url=vod_url,
                output_dir=Path(output_dir).resolve(),
                persona_id=persona,
                stream_id=stream_id,
                language=language,
                n_variants=n,
                quality=quality_arg,
                force=force,
                db_path=effective_db_path,
                chat_replay_source=chat_replay,
            )
        )
    except (
        IngestError,
        TranscriptionError,
        DetectionError,
        ClipError,
        VariantError,
        LLMError,
        TenancyError,
    ) as e:
        typer.echo(f"process failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(manifest.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {manifest.stream.id}")
    typer.echo(f"  persona:  {manifest.persona_id} ({manifest.persona_name})")
    typer.echo(f"  language: {manifest.language}")
    typer.echo(f"  duration: {manifest.stream.duration_s:.1f}s")
    typer.echo(f"  candidates: {len(manifest.candidates)}")
    typer.echo(f"  clips:    {len(manifest.clip_entries)}")
    typer.echo(
        f"  llm:      {manifest.llm_spend.total_calls} calls, "
        f"${manifest.llm_spend.total_cost_usd_micros / 1_000_000:.4f}"
    )
    for entry in manifest.clip_entries:
        c = entry.clip
        typer.echo(
            f"  - {c.id}  [{c.start_s:>7.1f}s..{c.end_s:>7.1f}s]  {len(entry.variants)} variants"
        )


# ---------------------------------------------------------------------------
# Phase 1 admin commands: db init, tenants add/list, tokens issue/list.
# ---------------------------------------------------------------------------


def _open_db(db_path: str | Path | None) -> Database:
    from chalybclip.db import Database
    from chalybclip.settings import get_settings, resolve_db_target

    if db_path:
        return Database(Path(db_path))
    # Prod runs on Postgres (DATABASE_URL); only `db_path` would open the
    # SQLite fallback. Use the same resolver the web app does so CLI commands
    # hit the SAME database the running service does — not an empty local file.
    return Database(resolve_db_target(get_settings()))


@db_app.command("init")
def db_init_cmd(
    db_path: Path | None = typer.Option(
        None, "--db-path", help="Override CHALYBCLIP_DB_PATH for this command."
    ),
) -> None:
    """Apply Phase 1 migrations against the configured SQLite file."""
    from chalybclip.db import apply_migrations

    async def _run() -> int:
        db = _open_db(db_path)
        try:
            return await apply_migrations(db)
        finally:
            await db.close()

    version = asyncio.run(_run())
    typer.echo(f"schema_version = {version}")


@tenants_app.command("add")
def tenants_add_cmd(
    tenant_id: str = typer.Argument(..., help="Stable tenant id, e.g. `aldo`."),
    name: str = typer.Argument(..., help="Display name."),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Create a new tenant row."""
    from chalybclip.db import TenantsRepo, apply_migrations

    async def _run() -> str:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            repo = TenantsRepo(db)
            t = await repo.create(tenant_id=tenant_id, name=name)
            return t.id
        finally:
            await db.close()

    created = asyncio.run(_run())
    typer.echo(f"created tenant: {created}")


@tenants_app.command("list")
def tenants_list_cmd(
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """List all tenants."""
    from chalybclip.db import TenantsRepo, apply_migrations

    async def _run() -> list[Tenant]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            return await TenantsRepo(db).list_all()
        finally:
            await db.close()

    tenants = asyncio.run(_run())
    if not tenants:
        typer.echo("(no tenants)")
        return
    for t in tenants:
        budget = (
            f"${t.daily_llm_budget_usd_micros / 1_000_000:.2f}"
            if t.daily_llm_budget_usd_micros is not None
            else "unlimited"
        )
        publish_cap = (
            str(t.daily_publish_limit) if t.daily_publish_limit is not None else "unlimited"
        )
        typer.echo(
            f"  {t.id}  {t.name}  ({t.created_at})  "
            f"budget={budget}  publish/d={publish_cap}  "
            f"rescore_cap={t.rescore_concurrency_cap}"
        )


@tenants_app.command("set-budget")
def tenants_set_budget_cmd(
    tenant_id: str = typer.Argument(..., help="Tenant id."),
    daily_usd: float | None = typer.Option(
        None,
        "--daily-usd",
        help="Daily LLM USD ceiling. Omit to leave unchanged; pass 0 = unlimited.",
    ),
    publish_limit: int | None = typer.Option(
        None, "--publish-limit", help="Daily publish_jobs ceiling. Pass 0 = unlimited."
    ),
    rescore_cap: int | None = typer.Option(
        None, "--rescore-cap", help="Max concurrent vision rescores per tenant.", min=1
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Tune the budget governor knobs for a tenant.

    Pass 0 (or any value <= 0) for `--daily-usd` / `--publish-limit` to
    flip the cap to NULL (= unlimited). Pass nothing to leave the column
    untouched.
    """
    from chalybclip.db import TenantsRepo, apply_migrations

    if daily_usd is None and publish_limit is None and rescore_cap is None:
        typer.echo("nothing to update; pass --daily-usd, --publish-limit, or --rescore-cap", err=True)
        raise typer.Exit(code=2)

    async def _run() -> Tenant:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            repo = TenantsRepo(db)
            if await repo.get(tenant_id) is None:
                typer.echo(f"unknown tenant: {tenant_id}", err=True)
                raise typer.Exit(code=1)
            kwargs: dict[str, int | None] = {}
            if daily_usd is not None:
                kwargs["daily_llm_budget_usd_micros"] = (
                    None if daily_usd <= 0 else round(daily_usd * 1_000_000)
                )
            if publish_limit is not None:
                kwargs["daily_publish_limit"] = (
                    None if publish_limit <= 0 else publish_limit
                )
            if rescore_cap is not None:
                kwargs["rescore_concurrency_cap"] = rescore_cap
            return await repo.set_budget(tenant_id, **kwargs)  # type: ignore[arg-type]
        finally:
            await db.close()

    updated = asyncio.run(_run())
    budget = (
        f"${updated.daily_llm_budget_usd_micros / 1_000_000:.2f}"
        if updated.daily_llm_budget_usd_micros is not None
        else "unlimited"
    )
    publish_cap = (
        str(updated.daily_publish_limit)
        if updated.daily_publish_limit is not None
        else "unlimited"
    )
    typer.echo(
        f"  budget={budget}  publish/d={publish_cap}  "
        f"rescore_cap={updated.rescore_concurrency_cap}"
    )


@tenants_app.command("set-retention")
def tenants_set_retention_cmd(
    tenant_id: str = typer.Argument(..., help="Tenant id."),
    vod_days: int | None = typer.Option(
        None,
        "--vod-days",
        help="Days to keep raw VODs. Pass 0 to clear back to the system default (7).",
        min=0,
    ),
    clip_days: int | None = typer.Option(
        None,
        "--clip-days",
        help="Days to keep rendered clips. Pass 0 to clear back to the default (90).",
        min=0,
    ),
    transcript_days: int | None = typer.Option(
        None,
        "--transcript-days",
        help="Days to keep Whisper transcripts. Pass 0 to clear back to the default (365).",
        min=0,
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Configure per-tenant retention windows (slice E.1)."""
    from chalybclip.db import TenantsRepo, apply_migrations

    if vod_days is None and clip_days is None and transcript_days is None:
        typer.echo(
            "nothing to update; pass --vod-days, --clip-days, or --transcript-days",
            err=True,
        )
        raise typer.Exit(code=2)

    async def _run() -> Tenant:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            repo = TenantsRepo(db)
            kw: dict[str, int | None] = {}
            # 0 means "clear to NULL" = inherit system default. Anything
            # else is the literal day count.
            if vod_days is not None:
                kw["retention_vod_days"] = None if vod_days == 0 else vod_days
            if clip_days is not None:
                kw["retention_clip_days"] = None if clip_days == 0 else clip_days
            if transcript_days is not None:
                kw["retention_transcript_days"] = (
                    None if transcript_days == 0 else transcript_days
                )
            return await repo.set_retention(tenant_id, **kw)
        finally:
            await db.close()

    updated = asyncio.run(_run())
    typer.echo(
        f"  vod_days={updated.retention_vod_days or 'default(7)'}  "
        f"clip_days={updated.retention_clip_days or 'default(90)'}  "
        f"transcript_days={updated.retention_transcript_days or 'default(365)'}"
    )


@retention_app.command("sweep")
def retention_sweep_cmd(
    tenant: str | None = typer.Option(
        None, "--tenant", help="Restrict the sweep to one tenant id. Default: all."
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Project output root. Defaults to CHALYBCLIP_DEFAULT_OUTPUT_DIR (./out).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be deleted without touching anything.",
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON object per report instead of text."
    ),
) -> None:
    """Sweep per-tenant retention windows. Hard-deletes past-cutoff artifacts.

    Wire as a daily cron in production (`docs/production_deploy.md` §6).
    """
    import json as _json

    from chalybclip.db import apply_migrations
    from chalybclip.retention import sweep_retention
    from chalybclip.settings import get_settings

    out_dir = output_dir or Path(get_settings().default_output_dir)

    async def _run() -> list[dict[str, object]]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            reports = await sweep_retention(
                db,
                output_dir=out_dir,
                tenant_id=tenant,
                dry_run=dry_run,
            )
            return [
                {
                    "tenant_id": r.tenant_id,
                    "vod_days": r.policy.vod_days,
                    "clip_days": r.policy.clip_days,
                    "transcript_days": r.policy.transcript_days,
                    "vods_deleted": r.vods_deleted,
                    "clips_deleted": r.clips_deleted,
                    "transcripts_deleted": r.transcripts_deleted,
                    "bytes_freed": r.bytes_freed,
                    "dry_run": r.dry_run,
                }
                for r in reports
            ]
        finally:
            await db.close()

    rows = asyncio.run(_run())
    if json_out:
        for row in rows:
            typer.echo(_json.dumps(row))
        return
    if not rows:
        typer.echo("(no tenants matched the sweep)")
        return
    label = "DRY RUN" if dry_run else "swept"
    for r in rows:
        mb_freed = r["bytes_freed"] / (1024 * 1024) if r["bytes_freed"] else 0  # type: ignore[operator]
        typer.echo(
            f"  [{label}] {r['tenant_id']}  "
            f"vods={r['vods_deleted']}  clips={r['clips_deleted']}  "
            f"transcripts={r['transcripts_deleted']}  freed={mb_freed:.1f} MB"
        )


@drive_app.command("add")
def drive_add_cmd(
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant the watch belongs to."),
    folder_id: str = typer.Argument(..., help="Drive folder id (or fake-client folder name)."),
    folder_name: str | None = typer.Option(
        None, "--folder-name", help="Optional human-readable label for the dashboard."
    ),
    refresh_token: str = typer.Option(
        ...,
        "--refresh-token",
        help=(
            "Drive OAuth refresh token. For dev with FakeDriveClient pass "
            "any non-empty placeholder."
        ),
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Register a new Drive folder to watch (slice E.4)."""
    from chalybclip.db import DriveWatchesRepo, TenantsRepo, apply_migrations
    from chalybclip.tenancy import bound_tenant

    async def _run() -> str:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            t = await TenantsRepo(db).get(tenant_id)
            if t is None:
                typer.echo(f"unknown tenant: {tenant_id}", err=True)
                raise typer.Exit(code=1)
            with bound_tenant(t.id):
                w = await DriveWatchesRepo(db).create(
                    folder_id=folder_id,
                    folder_name=folder_name,
                    refresh_token=refresh_token,
                )
            return w.id
        finally:
            await db.close()

    new = asyncio.run(_run())
    typer.echo(f"created drive watch: {new}")


@drive_app.command("list")
def drive_list_cmd(
    tenant_id: str = typer.Option(..., "--tenant"),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """List drive watches for one tenant."""
    from chalybclip.db import DriveWatchesRepo, TenantsRepo, apply_migrations
    from chalybclip.tenancy import bound_tenant

    async def _run() -> list[object]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            t = await TenantsRepo(db).get(tenant_id)
            if t is None:
                typer.echo(f"unknown tenant: {tenant_id}", err=True)
                raise typer.Exit(code=1)
            with bound_tenant(t.id):
                return list(await DriveWatchesRepo(db).list_for_tenant())
        finally:
            await db.close()

    rows = asyncio.run(_run())
    if not rows:
        typer.echo("(no watches)")
        return
    for w in rows:
        state = "enabled" if w.enabled else "paused"  # type: ignore[attr-defined]
        typer.echo(
            f"  {w.id}  folder={w.folder_id}  ({w.folder_name or '—'})  "  # type: ignore[attr-defined]
            f"{state}  seen={len(w.seen_file_ids)}  last_polled_at={w.last_polled_at or 'never'}"  # type: ignore[attr-defined]
        )


@drive_app.command("poll")
def drive_poll_cmd(
    tenant: str | None = typer.Option(
        None, "--tenant", help="Restrict polling to one tenant. Default: all."
    ),
    source_dir: Path | None = typer.Option(
        None,
        "--source-dir",
        help=(
            "DEV: poll a local directory via FakeDriveClient instead of Google. "
            "When omitted, the real GoogleDriveClient is used (deferred — currently "
            "unimplemented; poll exits 1 with a clear error)."
        ),
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Project output root. Defaults to CHALYBCLIP_DEFAULT_OUTPUT_DIR.",
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON object per report instead of text."
    ),
) -> None:
    """Poll every drive watch for new files and ingest them.

    For dev / tests, pass `--source-dir <path>` to use the on-disk
    FakeDriveClient. Drop video files into that directory; each new
    file gets ingested through the standard pipeline.
    """
    import json as _json

    from chalybclip.db import apply_migrations
    from chalybclip.drive import FakeDriveClient, poll_drive_watches
    from chalybclip.settings import get_settings

    out_dir = output_dir or Path(get_settings().default_output_dir)

    async def _run() -> list[dict[str, object]]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)

            # Real Google client integration is a follow-up — for now
            # CLI poll only works with --source-dir.
            if source_dir is None:
                typer.echo(
                    "drive poll: --source-dir is required (real Google client integration is "
                    "deferred to a follow-up commit)",
                    err=True,
                )
                raise typer.Exit(code=1)

            fake = FakeDriveClient(source_dir)

            async def _stub_ingest(tenant_id_: str, local_path: Path, name: str) -> None:
                # CLI mode just logs the file. Production wires this
                # to chalybclip.ingest.ingest_local_file via the in-process
                # scheduler — that lives in run.py's lifespan once the
                # Google client lands.
                typer.echo(f"  [{tenant_id_}] ingested: {name} -> {local_path}")

            reports = await poll_drive_watches(
                db,
                output_dir=out_dir,
                drive_client_factory=lambda _w: fake,
                ingest_callback=_stub_ingest,
                tenant_id=tenant,
            )
            return [
                {
                    "watch_id": r.watch_id,
                    "tenant_id": r.tenant_id,
                    "folder_id": r.folder_id,
                    "files_seen": r.files_seen,
                    "files_ingested": r.files_ingested,
                    "files_failed": r.files_failed,
                    "skipped_disabled": r.skipped_disabled,
                }
                for r in reports
            ]
        finally:
            await db.close()

    rows = asyncio.run(_run())
    if json_out:
        for row in rows:
            typer.echo(_json.dumps(row))
        return
    if not rows:
        typer.echo("(no watches matched)")
        return
    for r in rows:
        flag = " (paused)" if r["skipped_disabled"] else ""
        typer.echo(
            f"  {r['watch_id']}  tenant={r['tenant_id']}  folder={r['folder_id']}{flag}  "
            f"seen={r['files_seen']}  ingested={r['files_ingested']}  failed={r['files_failed']}"
        )


@channel_app.command("add")
def channel_add_cmd(
    channel_url: str = typer.Argument(..., help="Channel / @handle / videos URL to watch."),
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant the watch belongs to."),
    persona_id: str = typer.Option(
        ..., "--persona", help="Persona used for variant generation on ingested VODs."
    ),
    platform: str | None = typer.Option(
        None,
        "--platform",
        help="youtube | twitch | kick. Auto-detected from the URL when omitted.",
    ),
    language: str | None = typer.Option(
        None, "--language", help="Optional language hint for transcription."
    ),
    label: str | None = typer.Option(
        None, "--label", help="Optional human-readable label for the dashboard."
    ),
    max_per_poll: int = typer.Option(
        3, "--max-per-poll", help="Cap on VODs ingested per poll (and first-poll backfill)."
    ),
    polls_per_day: int = typer.Option(
        1, "--polls-per-day", min=1, max=96,
        help="How many times a day to poll (1=daily, 4=every 6h, 24=hourly). Saves resources.",
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Connect a creator channel so new VODs auto-ingest into ChalybClip."""
    from chalybclip.db import ChannelWatchesRepo, TenantsRepo, apply_migrations
    from chalybclip.ingest.service import detect_platform
    from chalybclip.tenancy import bound_tenant

    resolved_platform = platform or detect_platform(channel_url)

    async def _run() -> str:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            t = await TenantsRepo(db).get(tenant_id)
            if t is None:
                typer.echo(f"unknown tenant: {tenant_id}", err=True)
                raise typer.Exit(code=1)
            with bound_tenant(t.id):
                w = await ChannelWatchesRepo(db).create(
                    platform=resolved_platform,
                    channel_url=channel_url,
                    persona_id=persona_id,
                    channel_label=label,
                    language=language,
                    max_per_poll=max_per_poll,
                    polls_per_day=polls_per_day,
                )
            return w.id
        finally:
            await db.close()

    new = asyncio.run(_run())
    typer.echo(
        f"created channel watch: {new} (platform={resolved_platform}, "
        f"{polls_per_day}x/day)"
    )


@channel_app.command("schedule")
def channel_schedule_cmd(
    watch_id: str = typer.Argument(..., help="Channel watch id (chw_...)."),
    tenant_id: str = typer.Option(..., "--tenant"),
    per_day: int = typer.Option(
        ..., "--per-day", min=1, max=96,
        help="Times a day to poll (1=daily, 4=every 6h, 24=hourly).",
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Change how many times a day a channel is polled."""
    from chalybclip.db import ChannelWatchesRepo, TenantsRepo, apply_migrations
    from chalybclip.tenancy import bound_tenant

    async def _run() -> None:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            if await TenantsRepo(db).get(tenant_id) is None:
                typer.echo(f"unknown tenant: {tenant_id}", err=True)
                raise typer.Exit(code=1)
            with bound_tenant(tenant_id):
                w = await ChannelWatchesRepo(db).set_polls_per_day(watch_id, per_day)
            typer.echo(f"{w.id}: now {w.polls_per_day}x/day")
        finally:
            await db.close()

    asyncio.run(_run())


@channel_app.command("list")
def channel_list_cmd(
    tenant_id: str = typer.Option(..., "--tenant"),
    db_path: Path | None = typer.Option(None, "--db-path"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """List channel watches for one tenant."""
    import json as _json

    from chalybclip.db import ChannelWatchesRepo, TenantsRepo, apply_migrations
    from chalybclip.tenancy import bound_tenant

    async def _run() -> list[object]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            t = await TenantsRepo(db).get(tenant_id)
            if t is None:
                typer.echo(f"unknown tenant: {tenant_id}", err=True)
                raise typer.Exit(code=1)
            with bound_tenant(t.id):
                return list(await ChannelWatchesRepo(db).list_for_tenant())
        finally:
            await db.close()

    rows = asyncio.run(_run())
    if json_out:
        for w in rows:
            typer.echo(_json.dumps(w.model_dump()))  # type: ignore[attr-defined]
        return
    if not rows:
        typer.echo("(no channel watches)")
        return
    for w in rows:
        state = "enabled" if w.enabled else "paused"  # type: ignore[attr-defined]
        typer.echo(
            f"  {w.id}  {w.platform}  {w.channel_url}  ({w.channel_label or '—'})  "  # type: ignore[attr-defined]
            f"{state}  seen={len(w.seen_video_ids)}  "  # type: ignore[attr-defined]
            f"last_polled_at={w.last_polled_at or 'never'}"  # type: ignore[attr-defined]
        )


@channel_app.command("poll")
def channel_poll_cmd(
    tenant: str | None = typer.Option(
        None, "--tenant", help="Restrict polling to one tenant. Default: all."
    ),
    db_path: Path | None = typer.Option(None, "--db-path"),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON object per report instead of text."
    ),
) -> None:
    """Poll every channel watch for new VODs and report detections.

    CLI poll lists new uploads (via yt-dlp) and marks them detected, but
    does NOT run the heavy pipeline here — the running API server's
    channel-poll loop performs the real ingest + pipeline dispatch. Use
    this to verify wiring and see what would be picked up.
    """
    import json as _json

    from chalybclip.channels import poll_channel_watches
    from chalybclip.db import apply_migrations

    async def _run() -> list[dict[str, object]]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)

            async def _log_only(
                tenant_id_: str,
                vod_url: str,
                video_id: str,
                persona_id: str,
                language: str | None,
            ) -> None:
                typer.echo(f"  [{tenant_id_}] detected VOD {video_id}: {vod_url}")

            reports = await poll_channel_watches(
                db,
                ingest_callback=_log_only,
                tenant_id=tenant,
            )
            return [
                {
                    "watch_id": r.watch_id,
                    "tenant_id": r.tenant_id,
                    "channel_url": r.channel_url,
                    "videos_seen": r.videos_seen,
                    "videos_ingested": r.videos_ingested,
                    "videos_failed": r.videos_failed,
                    "skipped_disabled": r.skipped_disabled,
                }
                for r in reports
            ]
        finally:
            await db.close()

    rows = asyncio.run(_run())
    if json_out:
        for row in rows:
            typer.echo(_json.dumps(row))
        return
    if not rows:
        typer.echo("(no channel watches matched)")
        return
    for r in rows:
        flag = " (paused)" if r["skipped_disabled"] else ""
        typer.echo(
            f"  {r['watch_id']}  tenant={r['tenant_id']}  {r['channel_url']}{flag}  "
            f"seen={r['videos_seen']}  detected={r['videos_ingested']}  failed={r['videos_failed']}"
        )


@channel_app.command("enable")
def channel_enable_cmd(
    watch_id: str = typer.Argument(...),
    tenant_id: str = typer.Option(..., "--tenant"),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Resume a paused channel watch."""
    _channel_set_enabled(watch_id, tenant_id, db_path, enabled=True)


@channel_app.command("disable")
def channel_disable_cmd(
    watch_id: str = typer.Argument(...),
    tenant_id: str = typer.Option(..., "--tenant"),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Pause a channel watch without losing its seen-set."""
    _channel_set_enabled(watch_id, tenant_id, db_path, enabled=False)


@channel_app.command("remove")
def channel_remove_cmd(
    watch_id: str = typer.Argument(...),
    tenant_id: str = typer.Option(..., "--tenant"),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Delete a channel watch."""
    from chalybclip.db import ChannelWatchesRepo, TenantsRepo, apply_migrations
    from chalybclip.tenancy import bound_tenant

    async def _run() -> None:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            t = await TenantsRepo(db).get(tenant_id)
            if t is None:
                typer.echo(f"unknown tenant: {tenant_id}", err=True)
                raise typer.Exit(code=1)
            with bound_tenant(t.id):
                await ChannelWatchesRepo(db).delete(watch_id)
        finally:
            await db.close()

    asyncio.run(_run())
    typer.echo(f"removed channel watch: {watch_id}")


def _channel_set_enabled(
    watch_id: str, tenant_id: str, db_path: Path | None, *, enabled: bool
) -> None:
    from chalybclip.db import ChannelWatchesRepo, TenantsRepo, apply_migrations
    from chalybclip.tenancy import bound_tenant

    async def _run() -> None:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            t = await TenantsRepo(db).get(tenant_id)
            if t is None:
                typer.echo(f"unknown tenant: {tenant_id}", err=True)
                raise typer.Exit(code=1)
            with bound_tenant(t.id):
                await ChannelWatchesRepo(db).set_enabled(watch_id, enabled)
        finally:
            await db.close()

    asyncio.run(_run())
    typer.echo(f"{'enabled' if enabled else 'paused'} channel watch: {watch_id}")


@tokens_app.command("issue")
def tokens_issue_cmd(
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant the token authenticates."),
    scope: str = typer.Option("full", "--scope", help="`full` or `read`."),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Mint a new API token. Prints the raw token ONCE -- store it now."""
    from chalybclip.db import ApiTokensRepo, TenantsRepo, apply_migrations
    from chalybclip.tenancy import bound_tenant, mint_token

    if scope not in ("full", "read"):
        typer.echo(f"--scope must be 'full' or 'read', got {scope!r}", err=True)
        raise typer.Exit(code=2)

    async def _run() -> str:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            t = await TenantsRepo(db).get(tenant_id)
            if t is None:
                raise typer.Exit(code=1)
            raw, hashed = mint_token()
            with bound_tenant(tenant_id):
                await ApiTokensRepo(db).create(hash_=hashed, scope=scope)
            return raw
        finally:
            await db.close()

    raw = asyncio.run(_run())
    typer.echo(raw)
    typer.echo(
        "(store this token now -- it is not retrievable again; only its sha256 "
        "hash is persisted in the database)",
        err=True,
    )


@tokens_app.command("list")
def tokens_list_cmd(
    tenant_id: str = typer.Option(..., "--tenant"),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """List tokens for a tenant (hashes + scopes only -- no raw tokens)."""
    from chalybclip.db import ApiTokensRepo, apply_migrations
    from chalybclip.tenancy import bound_tenant

    async def _run() -> list[ApiTokenRow]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            with bound_tenant(tenant_id):
                return await ApiTokensRepo(db).list_for_tenant()
        finally:
            await db.close()

    rows = asyncio.run(_run())
    if not rows:
        typer.echo("(no tokens)")
        return
    for r in rows:
        last = r.last_used_at or "(never)"
        typer.echo(f"  {r.id}  scope={r.scope}  hash={r.hash[:16]}…  last_used={last}")


if __name__ == "__main__":
    app()

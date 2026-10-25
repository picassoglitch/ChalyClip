"""Production ingest callback for channel watches.

The drive watcher left its ingest callback stubbed; channel watches get
the real wiring: detect a VOD → `ingest_vod` (idempotent download + audio
extract) → persist the stream row → kick off the full pipeline via the
configured `JobDispatcher`. Downstream (clip → variants → auto-publish)
runs unchanged, with publishing gated by the safe trap.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path

import structlog

from chalybclip.db import Database, EventsRepo, StreamsRepo
from chalybclip.db.models import StreamRow
from chalybclip.ids import new_id
from chalybclip.jobs import JobDispatcher, PipelineKickoff
from chalybclip.tenancy import bound_tenant

from .service import ChannelIngestCallback

_log = structlog.get_logger(__name__)


def _pending_placeholder(
    *,
    stream_id: str,
    tenant_id: str,
    vod_url: str,
    platform: str,
    output_dir: Path,
) -> StreamRow:
    """A `status="pending"` StreamRow that makes a freshly-detected channel
    VOD show on the Streams list immediately (with "Downloading…"), the same
    way a manual URL job does — instead of staying invisible until the
    download + audio extract finish.

    The source paths match exactly where `ingest_vod(stream_id=...)` writes
    (`<output_dir>/<id>/source/{video.mp4,audio.wav}`), so the upsert with
    real metadata once the download lands is a no-op for those fields.
    """
    source_dir = output_dir / stream_id / "source"
    return StreamRow(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url=vod_url,
        platform=platform,
        title=None,
        channel=None,
        duration_s=0.0,
        source_video_path=str(source_dir / "video.mp4"),
        source_audio_path=str(source_dir / "audio.wav"),
        status="pending",
        created_at=datetime.now(UTC).isoformat(),
    )


def make_channel_ingest_callback(
    db: Database,
    dispatcher: JobDispatcher,
    *,
    output_dir: Path,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> ChannelIngestCallback:
    """Build the ingest callback the channel poller hands new VODs to.

    Each call ingests the VOD synchronously (fast — yt-dlp metadata + audio
    extract), persists the stream, emits `channel.vod_detected`, then
    dispatches the pipeline. `ingest_vod` is idempotent, so re-detecting an
    already-ingested VOD is a no-op download.
    """
    output_dir = Path(output_dir).resolve()

    async def _callback(
        tenant_id: str,
        vod_url: str,
        video_id: str,
        persona_id: str,
        language: str | None,
    ) -> None:
        from chalybclip.ingest import detect_platform, ingest_vod
        from chalybclip.integrations.chalyb.admission import AdmissionRefused
        from chalybclip.jobs.usage import admit_job, cancel_admission, readmit_job

        # Consumption contract: admit before downloading, re-admit with the real length
        # after. A refusal raises so the poller retries/parks the VOD instead of dropping it.
        try:
            admission = await admit_job(db, tenant_id=tenant_id, stream_id="")
        except AdmissionRefused as e:
            _log.info(
                "channel.vod_refused", tenant_id=tenant_id, video_id=video_id,
                reason=e.reason,
            )
            raise
        try:
            with bound_tenant(tenant_id):
                # Reuse the stream id of an already-ingested (tenant, vod_url) so ingest_vod cache-hits;
                # a fresh ULID per re-detection re-downloaded the VOD into a duplicate row.
                existing = await StreamsRepo(db).find_by_vod_url(vod_url)
                if existing is None:
                    # Write a 'pending' placeholder + detected event up front so the stream shows
                    # the moment it's pulled; reconcile_metadata below flips it to 'ingested'.
                    stream_id = new_id("str")
                    await StreamsRepo(db).upsert(
                        _pending_placeholder(
                            stream_id=stream_id,
                            tenant_id=tenant_id,
                            vod_url=vod_url,
                            platform=detect_platform(vod_url),
                            output_dir=output_dir,
                        )
                    )
                    await EventsRepo(db).emit(
                        type="channel.vod_detected",
                        payload={
                            "stream_id": stream_id,
                            "video_id": video_id,
                            "vod_url": vod_url,
                        },
                    )
                else:
                    stream_id = existing.id
                try:
                    stream = await ingest_vod(
                        tenant_id=tenant_id,
                        vod_url=vod_url,
                        output_dir=output_dir,
                        stream_id=stream_id,
                        cookies_from_browser=cookies_from_browser,
                        cookies_file=cookies_file,
                        db=db,
                    )
                except Exception:
                    # A failed download flips the placeholder to 'failed' instead of 'pending'
                    # forever; re-raise so the channel poller still applies its retry/park.
                    with contextlib.suppress(Exception):
                        await StreamsRepo(db).set_status(
                            stream_id, status="failed", only_if="pending"
                        )
                    raise
                # reconcile_metadata, not upsert: upsert is INSERT-OR-IGNORE and would not
                # update the pre-inserted placeholder (same path as the URL-job pipeline).
                await StreamsRepo(db).reconcile_metadata(
                    stream_id,
                    title=stream.title,
                    channel=stream.channel,
                    duration_s=stream.duration_s,
                )

            admission = await readmit_job(
                db, job_id=admission.job_id, stream_id=stream.id,
                source_minutes=float(stream.duration_s or 0) / 60.0,
            ) or admission
        except AdmissionRefused as e:
            _log.info(
                "channel.vod_refused", tenant_id=tenant_id, video_id=video_id,
                reason=e.reason,
            )
            raise
        except BaseException:
            await cancel_admission(db, admission)
            raise

        kickoff = PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
            language=language,
        ).with_admission(admission)
        # Decouple ingest from pipeline outcome. A successful download +
        # stream row IS the ingest as far as the poller's dedup is
        # concerned; the pipeline runs with its own retry/resume (step
        # events + dashboard re-run). If a pipeline step fails, it must NOT
        # propagate here — otherwise the poller counts the VOD as
        # un-ingested, never adds it to the seen-set, and re-runs the whole
        # thing next poll (the duplicate-ingest + re-download loop that also
        # amplifies host load). Download failures still raise from
        # ingest_vod above, so a genuinely failed ingest is still retried.
        try:
            await dispatcher.dispatch_pipeline(kickoff)
        except Exception as e:  # pipeline failure is not an ingest failure
            await cancel_admission(db, admission)
            _log.warning(
                "channel.pipeline_dispatch_failed",
                tenant_id=tenant_id,
                stream_id=stream.id,
                video_id=video_id,
                error=str(e),
            )
            return
        _log.info(
            "channel.vod_dispatched",
            tenant_id=tenant_id,
            stream_id=stream.id,
            video_id=video_id,
        )

    return _callback

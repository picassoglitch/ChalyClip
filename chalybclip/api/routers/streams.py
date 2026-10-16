"""Stream + candidate + clip-listing endpoints.

`POST /streams` is the kick-off point - it ingests the VOD synchronously
(needed to know stream metadata + on-disk paths), persists the row, then
schedules the rest of `process_vod` as a FastAPI BackgroundTask. The
ingest step itself is fast (yt-dlp metadata fetch); the heavy work is
Whisper + vision + variants which fire after the response returns.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from chalybclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    EventsRepo,
    StreamsRepo,
)
from chalybclip.db.usage_repos import UsageJobsRepo
from chalybclip.errors import ChalybClipError
from chalybclip.integrations.chalyb.admission import AdmissionRefused
from chalybclip.jobs.usage import admit_job, cancel_admission, readmit_job

from .._admission import (
    declared_body_bytes,
    effective_cap,
    parse_boost,
    probe_minutes,
    read_capped_form,
    refusal_http,
    too_large_http,
)
from .._pipeline import PipelineKickoff
from ..deps import get_db, require_full_scope, tenant_binder
from ..schemas import (
    CandidateResponse,
    ClipResponse,
    StreamCreateRequest,
    StreamResponse,
)

router = APIRouter(prefix="/streams", tags=["streams"])


@router.post(
    "",
    response_model=StreamResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_full_scope)],
)
async def create_stream(
    payload: StreamCreateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> StreamResponse:
    """Ingest a VOD and schedule the rest of the pipeline in the background.

    The ingest step (yt-dlp metadata + audio/video download) runs inline so
    the response can carry the resulting Stream id. Transcription, detection,
    cutting, and variant generation fire as a BackgroundTask after the
    response goes out.
    """
    # Imported here to avoid pulling yt-dlp into module-load time.
    from chalybclip.db.adapters import stream_to_row
    from chalybclip.ingest import ingest_vod
    from chalybclip.settings import get_settings

    output_dir = Path(get_settings().default_output_dir)
    # Consumption contract: admit before the (inline) download, re-admit
    # with the real duration once ingest knows it.
    try:
        admission = await admit_job(
            db, tenant_id=tenant_id, stream_id="", boost=payload.boost
        )
    except AdmissionRefused as e:
        raise refusal_http(e) from e
    try:
        try:
            stream = await ingest_vod(
                vod_url=payload.vod_url,
                tenant_id=tenant_id,
                output_dir=output_dir,
            )
        except ChalybClipError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e
        admission = await readmit_job(
            db, job_id=admission.job_id, stream_id=stream.id,
            source_minutes=float(stream.duration_s or 0) / 60.0,
        ) or admission
    except AdmissionRefused as e:
        # readmit_job already cancelled the reservation.
        raise refusal_http(e) from e
    except BaseException:
        await cancel_admission(db, admission)
        raise

    row = await StreamsRepo(db).upsert(stream_to_row(stream))
    await EventsRepo(db).emit(type="stream.created", payload={"stream_id": row.id})

    dispatcher = request.app.state.job_dispatcher
    kickoff = PipelineKickoff(
        tenant_id=tenant_id,
        stream=stream,
        persona_id=payload.persona_id,
        output_dir=output_dir,
        language=payload.language,
    ).with_admission(admission)
    await dispatcher.dispatch_pipeline(kickoff, background_tasks=background_tasks)
    return StreamResponse.model_validate(row.model_dump())


@router.post(
    "/upload",
    response_model=StreamResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_full_scope)],
)
async def upload_stream(
    background_tasks: BackgroundTasks,
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> StreamResponse:
    """Ingest an uploaded video file (no yt-dlp). Multipart fields: `file`,
    `persona_id`, optional `language`, optional `boost`.

    The body is read by hand (see `api/_admission.py`) so the hub is asked
    BEFORE any bytes are accepted, and the transfer is cut off once it
    passes the tier's `max_upload_mb`. After the file lands it's probed and
    re-admitted with the real duration before ingest runs.

    Same downstream contract as `POST /streams`: the response carries the
    new Stream id and the rest of the pipeline (transcribe, detect, cut,
    variants) runs as a BackgroundTask.
    """
    from chalybclip.db.adapters import stream_to_row
    from chalybclip.ingest import ingest_uploaded, is_ffmpeg_available
    from chalybclip.settings import get_settings

    if not is_ffmpeg_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ffmpeg is not installed on the server. Install it first: "
                "Windows -> 'winget install --id=Gyan.FFmpeg -e' (then reopen "
                "your shell so PATH refreshes); macOS -> 'brew install ffmpeg'; "
                "Debian/Ubuntu -> 'sudo apt install ffmpeg'."
            ),
        )

    output_dir = Path(get_settings().default_output_dir)
    declared = declared_body_bytes(request)
    query_boost = parse_boost(request.query_params.get("boost"))
    try:
        admission = await admit_job(
            db, tenant_id=tenant_id, stream_id="", upload_bytes=declared,
            boost=query_boost,
        )
    except AdmissionRefused as e:
        raise refusal_http(e) from e

    tmp_path: Path | None = None
    try:
        cap = effective_cap(admission.max_upload_bytes, _global_upload_cap())
        if cap is not None and declared > cap + 64 * 1024:
            raise too_large_http()
        file, fields = await read_capped_form(request, max_bytes=cap)
        persona_id = (fields.get("persona_id") or "").strip()
        if not persona_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="persona_id is required",
            )
        language = fields.get("language") or None
        boost = parse_boost(fields.get("boost"))
        if boost is None:
            boost = query_boost
        tmp_path = await _stash_upload_to_tmp(file, output_dir, max_bytes=cap)
        # Check twice: the real size + duration, same job id.
        admission = await readmit_job(
            db, job_id=admission.job_id,
            upload_bytes=tmp_path.stat().st_size,
            source_minutes=await probe_minutes(tmp_path),
            boost=boost,
        ) or admission
        try:
            stream = await ingest_uploaded(
                tenant_id=tenant_id,
                source_path=tmp_path,
                output_dir=output_dir,
                title=file.filename,
            )
        except ChalybClipError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            ) from e
        await UsageJobsRepo(db).set_stream(admission.job_id, stream.id)
    except AdmissionRefused as e:
        raise refusal_http(e) from e
    except BaseException:
        await cancel_admission(db, admission)
        raise
    finally:
        # ingest_uploaded moves the file out; if it didn't (cache hit, error),
        # don't leave a multi-hundred-MB orphan in the temp dir.
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    row = await StreamsRepo(db).upsert(stream_to_row(stream))
    await EventsRepo(db).emit(type="stream.created", payload={"stream_id": row.id})

    dispatcher = request.app.state.job_dispatcher
    kickoff = PipelineKickoff(
        tenant_id=tenant_id,
        stream=stream,
        persona_id=persona_id,
        output_dir=output_dir,
        language=language,
    ).with_admission(admission)
    await dispatcher.dispatch_pipeline(kickoff, background_tasks=background_tasks)
    return StreamResponse.model_validate(row.model_dump())


def _global_upload_cap() -> int:
    """CHALYBCLIP_MAX_UPLOAD_BYTES. `getattr` so test stubs that monkeypatch
    get_settings with a minimal object still work."""
    from chalybclip.settings import get_settings as _get_settings

    return int(getattr(_get_settings(), "max_upload_bytes", 5 * 1024 * 1024 * 1024))


async def _stash_upload_to_tmp(
    file: UploadFile, output_dir: Path, *, max_bytes: int | None = None
) -> Path:
    """Stream `file` to a tempfile under `output_dir/.uploads/`. Returns the path.

    `max_bytes` (the tier's cap from admission) tightens the global
    CHALYBCLIP_MAX_UPLOAD_BYTES; the write stops as soon as either is passed.

    We write into the same root as the eventual stream dir (not /tmp) to avoid
    cross-device moves on Windows. Chunk size is 1 MB — plenty fast on local
    SSD, won't pin a 500 MB upload in RAM.
    """
    import uuid

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upload missing filename",
        )

    uploads_dir = output_dir / ".uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_path = uploads_dir / f"upl_{uuid.uuid4().hex}{suffix}"

    # Hard cap to protect disk against runaway uploads (DoS / accident).
    # Read from settings so a tight environment can lower it. Default
    # 5 GiB — enough for a multi-hour VOD; everything bigger should be
    # registered as a vod_url and pulled via yt-dlp anyway. `getattr`
    # so test stubs that monkeypatch get_settings with a minimal object
    # still work without needing this field declared.
    global_cap = _global_upload_cap()
    tier_capped = max_bytes is not None and max_bytes < global_cap
    max_bytes = max_bytes if tier_capped and max_bytes is not None else global_cap

    chunk_size = 1024 * 1024
    bytes_written = 0
    with tmp_path.open("wb") as out:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                # Stop the write + delete the partial file so we don't
                # leak disk on the rejection.
                out.close()
                tmp_path.unlink(missing_ok=True)
                if tier_capped:
                    raise too_large_http()
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"upload exceeds maximum size of {max_bytes} bytes "
                        f"(CHALYBCLIP_MAX_UPLOAD_BYTES)"
                    ),
                )

    if bytes_written == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty upload",
        )
    return tmp_path


@router.get("", response_model=list[StreamResponse])
async def list_streams(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[StreamResponse]:
    rows = await StreamsRepo(db).list_for_tenant()
    return [StreamResponse.model_validate(r.model_dump()) for r in rows]


@router.get("/{stream_id}", response_model=StreamResponse)
async def get_stream(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> StreamResponse:
    row = await StreamsRepo(db).get(stream_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    return StreamResponse.model_validate(row.model_dump())


@router.get("/{stream_id}/candidates", response_model=list[CandidateResponse])
async def list_candidates(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[CandidateResponse]:
    # Stream existence check ensures we 404 instead of returning [] for
    # other-tenant ids - tighter contract for the dashboard.
    if await StreamsRepo(db).get(stream_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    rows = await CandidatesRepo(db).list_for_stream(stream_id)
    return [CandidateResponse.model_validate(r.model_dump()) for r in rows]


@router.get("/{stream_id}/clips", response_model=list[ClipResponse])
async def list_clips(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[ClipResponse]:
    if await StreamsRepo(db).get(stream_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    rows = await ClipsRepo(db).list_for_stream(stream_id)
    return [ClipResponse.model_validate(r.model_dump()) for r in rows]

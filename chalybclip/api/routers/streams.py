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
    File,
    Form,
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
from chalybclip.errors import ChalybClipError

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
    try:
        stream = await ingest_vod(
            vod_url=payload.vod_url,
            tenant_id=tenant_id,
            output_dir=output_dir,
        )
    except ChalybClipError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    row = await StreamsRepo(db).upsert(stream_to_row(stream))
    await EventsRepo(db).emit(type="stream.created", payload={"stream_id": row.id})

    dispatcher = request.app.state.job_dispatcher
    kickoff = PipelineKickoff(
        tenant_id=tenant_id,
        stream=stream,
        persona_id=payload.persona_id,
        output_dir=output_dir,
        language=payload.language,
    )
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
    file: UploadFile = File(...),
    persona_id: str = Form(...),
    language: str | None = Form(None),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> StreamResponse:
    """Ingest an uploaded video file (no yt-dlp). Streams the upload to a
    tempfile chunk-by-chunk so a 1 GB VOD doesn't get pinned in RAM.

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
    tmp_path = await _stash_upload_to_tmp(file, output_dir)
    try:
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
    finally:
        # ingest_uploaded moves the file out; if it didn't (cache hit, error),
        # don't leave a multi-hundred-MB orphan in the temp dir.
        if tmp_path.exists():
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
    )
    await dispatcher.dispatch_pipeline(kickoff, background_tasks=background_tasks)
    return StreamResponse.model_validate(row.model_dump())


async def _stash_upload_to_tmp(file: UploadFile, output_dir: Path) -> Path:
    """Stream `file` to a tempfile under `output_dir/.uploads/`. Returns the path.

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
    from chalybclip.settings import get_settings as _get_settings
    max_bytes = getattr(_get_settings(), "max_upload_bytes", 5 * 1024 * 1024 * 1024)

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

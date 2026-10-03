"""Route-side helpers for the consumption contract (admission + uploads).

The upload routes take the raw `Request` instead of `File(...)`/`Form(...)`
parameters on purpose: FastAPI parses (and spools to disk) the whole
multipart body BEFORE a handler with File params runs, which would accept
every byte before we could ask the hub. Here the order is:

  1. admit with the declared size (Content-Length) — refuse before reading;
  2. stream the body through a byte counter capped at the tier's
     `max_upload_mb` (from the admit response) and the global
     CHALYBCLIP_MAX_UPLOAD_BYTES, aborting with 413 the moment it's over;
  3. the caller probes the file and re-admits with the real numbers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import HTTPException, Request, status
from starlette.datastructures import UploadFile

from chalybclip.integrations.chalyb.admission import AdmissionRefused

_TOO_LARGE_MESSAGE = (
    "El archivo supera el tamaño máximo permitido para tu plan. Sube un "
    "video más ligero o mejora tu plan."
)


class UploadTooLarge(Exception):
    """Raised mid-stream once the received body passes the cap."""


def refusal_http(e: AdmissionRefused) -> HTTPException:
    """AdmissionRefused → the HTTP error a route returns (413 size, 402
    tokens, 429 caps/concurrency, 503 hub down) with the Spanish message."""
    return HTTPException(status_code=e.status_code, detail=e.user_message)


def declared_body_bytes(request: Request) -> int:
    """The request's Content-Length (0 when absent / chunked). Multipart
    framing adds a few hundred bytes on top of the file — negligible
    against MB-sized caps."""
    try:
        return max(0, int(request.headers.get("content-length") or 0))
    except ValueError:
        return 0


def parse_boost(value: object) -> bool | None:
    """Form/query boost toggle → True (asked), False (never), None (tier
    default). Unchecked checkboxes aren't sent, so absent → None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("1", "true", "on", "yes", "si", "sí"):
        return True
    if text in ("0", "false", "off", "no"):
        return False
    return None


def effective_cap(*caps: int | None) -> int | None:
    vals = [int(c) for c in caps if c is not None and int(c) > 0]
    return min(vals) if vals else None


async def _capped_stream(
    request: Request, max_bytes: int | None
) -> AsyncGenerator[bytes, None]:
    seen = 0
    async for chunk in request.stream():
        seen += len(chunk)
        if max_bytes is not None and seen > max_bytes:
            raise UploadTooLarge(seen)
        yield chunk


async def read_capped_form(
    request: Request, *, max_bytes: int | None
) -> tuple[UploadFile, dict[str, str]]:
    """Parse a multipart upload with ONE file part, stopping the transfer
    as soon as the body passes `max_bytes`. Returns (file, text fields)."""
    from starlette.formparsers import MultiPartParser

    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="expected a multipart/form-data upload",
        )
    # Leave room for the multipart framing around the file itself.
    stream_cap = None if max_bytes is None else max_bytes + 64 * 1024
    parser = MultiPartParser(
        request.headers, _capped_stream(request, stream_cap), max_files=1, max_fields=32
    )
    try:
        form = await parser.parse()
    except UploadTooLarge as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=_TOO_LARGE_MESSAGE
        ) from e
    file: UploadFile | None = None
    fields: dict[str, str] = {}
    for key, value in form.multi_items():
        if isinstance(value, str):
            fields[key] = value
        elif key == "file" or file is None:
            file = value
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="file is required"
        )
    return file, fields


def too_large_http() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=_TOO_LARGE_MESSAGE
    )


async def probe_minutes(path: Path) -> float:
    """ffprobe the stashed upload → minutes (0.0 when unreadable). Looked
    up through the module so tests can stub the shell-out."""
    import asyncio

    from chalybclip.ingest import service as ingest_service

    seconds = await asyncio.to_thread(ingest_service._ffprobe_duration, path)
    return max(0.0, float(seconds or 0.0)) / 60.0

"""Serve large MP4s without tripping Cloud Run's response-size cap.

Cloud Run rejects any HTTP/1 response with a Content-Length over 32 MiB
("Response size was too large") and the client sees a 500. A 50 s clip at
1080x1920 / CRF 19 is already ~35 MB, so a plain FileResponse breaks:

  * <video> asks for `Range: bytes=0-` (the whole file) → we cap every
    range at `MAX_RANGE_BYTES`; a short 206 is legal and browsers just
    request the next range.
  * Downloads / server-to-server fetches send no Range → stream the file
    chunked (no Content-Length), which Cloud Run doesn't size-limit.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import anyio
from fastapi import Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.responses import Response

# Well under Cloud Run's 32 MiB cap, big enough that playback needs few
# round-trips.
MAX_RANGE_BYTES = 8 * 1024 * 1024

_STREAM_CHUNK_BYTES = 1024 * 1024

_SINGLE_RANGE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", re.IGNORECASE)


def _capped_range(header: str, size: int) -> str | None:
    """Rewrite a single byte range so it spans at most MAX_RANGE_BYTES.
    Returns None when the header should be left alone (multi-range,
    suffix range, malformed, or already small enough)."""
    m = _SINGLE_RANGE.match(header)
    if m is None or not m.group(1):
        return None
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else size - 1
    if start >= size or end - start + 1 <= MAX_RANGE_BYTES:
        return None
    return f"bytes={start}-{start + MAX_RANGE_BYTES - 1}"


async def _iter_file(path: Path) -> AsyncIterator[bytes]:
    async with await anyio.open_file(path, "rb") as f:
        while chunk := await f.read(_STREAM_CHUNK_BYTES):
            yield chunk


def video_file_response(
    request: Request,
    path: Path,
    *,
    media_type: str = "video/mp4",
    filename: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> Response:
    """Drop-in for FileResponse on video endpoints (see module docstring)."""
    range_header = request.headers.get("range")
    if request.method == "HEAD":
        # Headers only (incl. the real Content-Length) — no body, no cap.
        return FileResponse(
            path=path, media_type=media_type, filename=filename, headers=headers
        )
    if range_header is not None:
        capped = _capped_range(range_header, path.stat().st_size)
        if capped is not None:
            # FileResponse reads Range from the ASGI scope when it runs.
            request.scope["headers"] = [
                (k, capped.encode("latin-1") if k == b"range" else v)
                for k, v in request.scope["headers"]
            ]
        return FileResponse(
            path=path, media_type=media_type, filename=filename, headers=headers
        )

    stream_headers = {"Accept-Ranges": "bytes", **(headers or {})}
    if filename and not any(k.lower() == "content-disposition" for k in stream_headers):
        # Match FileResponse's default for a named file.
        stream_headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return StreamingResponse(
        _iter_file(path), media_type=media_type, headers=stream_headers
    )

"""Recording object-storage handoff for live ingest (Phase L.2, Path B).

The live-ingest service (the separate `chalybclip-live` MediaMTX deployment)
uploads each finished recording to an S3-compatible bucket under
`<prefix>/<stream_id>/<file>.mp4`. ChalyClip pulls it from there to run the
clip pipeline — so the two services share object storage instead of a
Railway volume, and scale independently.

Vendor-neutral: anything S3-compatible works via the `endpoint` setting —
Supabase Storage (reuse the Supabase you already run), Cloudflare R2,
MinIO, Backblaze B2, AWS S3, …

`build_recording_store(settings)` returns a store only when a bucket is
configured; otherwise None, and live ingest falls back to reading a shared
`/data` volume (Path A).
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from chalybclip.settings import Settings

# Skip a fetched object smaller than this (bytes) — a 0-byte placeholder or
# a multipart upload that hasn't completed. R2 only lists an object once its
# upload finishes, but this guards against truncated/odd states too.
_MIN_RECORDING_BYTES = 1024


@runtime_checkable
class RecordingStore(Protocol):
    """Fetches a live recording for a stream onto local disk."""

    async def fetch_latest(self, *, stream_id: str, dest_dir: Path) -> Path | None:
        """Download the newest recording object for `stream_id` into
        `dest_dir` and return its local path, or None if none exists yet."""
        ...


class S3RecordingStore:
    """Pulls live recordings from any S3-compatible bucket (Supabase
    Storage, Cloudflare R2, MinIO, S3, …).

    Lists `<prefix>/<stream_id>/` and downloads the newest non-trivial
    object. `client` is an S3 client (boto3 or a stub in tests); all calls
    run in a worker thread since boto3 is synchronous.
    """

    def __init__(self, *, client: Any, bucket: str, prefix: str = "live") -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    async def fetch_latest(self, *, stream_id: str, dest_dir: Path) -> Path | None:
        return await asyncio.to_thread(self._fetch_sync, stream_id, Path(dest_dir))

    def _fetch_sync(self, stream_id: str, dest_dir: Path) -> Path | None:
        key_prefix = f"{self._prefix}/{stream_id}/"
        resp: Any = self._client.list_objects_v2(
            Bucket=self._bucket, Prefix=key_prefix
        )
        contents: list[Any] = list(resp.get("Contents") or [])
        candidates = [c for c in contents if int(c.get("Size") or 0) > _MIN_RECORDING_BYTES]
        if not candidates:
            return None
        # Newest by upload time. If a stream produced multiple segments, the
        # last one wins; single-segment streams (the norm) have exactly one.
        newest = max(candidates, key=lambda c: c["LastModified"])
        key = str(newest["Key"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(key).name
        self._client.download_file(self._bucket, key, str(dest))
        return dest


@runtime_checkable
class ArtifactStore(Protocol):
    """Durable off-box store for pipeline artifacts (renders, sources, clips).

    The local Railway volume is only bounded scratch; the authoritative copy
    of a servable artifact lives here, addressed by an opaque `key`. The
    publisher fetches a presigned GET url for the object instead of pulling
    from our own time-boxed endpoint.
    """

    async def upload(
        self, *, local_path: Path, key: str, content_type: str | None = None
    ) -> None: ...

    async def download(self, *, key: str, dest: Path) -> Path | None: ...

    async def presigned_url(self, *, key: str, ttl_seconds: int) -> str: ...

    def public_url(self, key: str) -> str | None: ...

    async def exists(self, *, key: str) -> bool: ...

    async def delete(self, *, key: str) -> None: ...


class S3ArtifactStore:
    """`ArtifactStore` over any S3-compatible bucket (Cloudflare R2, Supabase
    Storage, MinIO, S3, …). All boto3 calls are sync, so they run in a worker
    thread. `prefix` namespaces every key (e.g. `artifacts/<...>`)."""

    # S3 SigV4 caps presigned-URL validity at 7 days. Past that the URL is
    # rejected, so we clamp — and prefer a stable public URL when available.
    _PRESIGN_MAX_TTL_S = 7 * 24 * 3600

    def __init__(
        self,
        *,
        client: Any,
        bucket: str,
        prefix: str = "artifacts",
        public_base_url: str | None = None,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._public_base = (public_base_url or "").rstrip("/") or None

    def _full(self, key: str) -> str:
        k = key.strip("/")
        return f"{self._prefix}/{k}" if self._prefix else k

    def public_url(self, key: str) -> str | None:
        """Stable, non-expiring URL for the object — only when a public base
        (R2 public bucket / custom domain) is configured. None otherwise, so
        the caller falls back to a presigned (≤7d) url."""
        if not self._public_base:
            return None
        return f"{self._public_base}/{self._full(key)}"

    async def upload(
        self, *, local_path: Path, key: str, content_type: str | None = None
    ) -> None:
        extra = {"ContentType": content_type} if content_type else None
        await asyncio.to_thread(
            self._client.upload_file,
            str(local_path), self._bucket, self._full(key),
            extra,  # ExtraArgs (positional in boto3.upload_file)
        )

    async def download(self, *, key: str, dest: Path) -> Path | None:
        def _dl() -> Path | None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._client.download_file(self._bucket, self._full(key), str(dest))
            except Exception:
                return None
            return dest

        return await asyncio.to_thread(_dl)

    async def presigned_url(self, *, key: str, ttl_seconds: int) -> str:
        ttl = max(1, min(int(ttl_seconds), self._PRESIGN_MAX_TTL_S))
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            {"Bucket": self._bucket, "Key": self._full(key)},
            ttl,
        )

    async def exists(self, *, key: str) -> bool:
        def _head() -> bool:
            try:
                self._client.head_object(Bucket=self._bucket, Key=self._full(key))
            except Exception:
                return False
            return True

        return await asyncio.to_thread(_head)

    async def delete(self, *, key: str) -> None:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                self._client.delete_object,
                Bucket=self._bucket, Key=self._full(key),
            )


def build_artifact_store(settings: Settings) -> ArtifactStore | None:
    """Construct the artifact store from settings, or None when no bucket is
    configured (→ the pipeline serves artifacts from the local volume, the
    current behavior). boto3 is imported lazily."""
    bucket = (getattr(settings, "object_storage_bucket", None) or "").strip()
    if not bucket:
        return None

    import boto3  # lazy: only a hard dep when object storage is configured

    endpoint = (getattr(settings, "object_storage_endpoint", None) or "").strip()
    client = boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=getattr(settings, "object_storage_access_key_id", None),
        aws_secret_access_key=getattr(
            settings, "object_storage_secret_access_key", None
        ),
        region_name=(getattr(settings, "object_storage_region", None) or "auto"),
    )
    return S3ArtifactStore(
        client=client,
        bucket=bucket,
        prefix=(getattr(settings, "object_storage_prefix", None) or "artifacts"),
        public_base_url=getattr(settings, "object_storage_public_base_url", None),
    )


def build_recording_store(settings: Settings) -> RecordingStore | None:
    """Construct the object store from settings, or None when not configured.

    None means "no object storage" → live ingest reads the shared volume
    (Path A). boto3 is imported lazily so it's only a hard dependency when
    object storage is actually wired up.
    """
    bucket = (getattr(settings, "live_recording_storage_bucket", None) or "").strip()
    if not bucket:
        return None

    import boto3  # lazy: only needed when object storage is configured

    endpoint = (
        getattr(settings, "live_recording_storage_endpoint", None) or ""
    ).strip()
    client = boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=getattr(
            settings, "live_recording_storage_access_key_id", None
        ),
        aws_secret_access_key=getattr(
            settings, "live_recording_storage_secret_access_key", None
        ),
        region_name=(
            getattr(settings, "live_recording_storage_region", None) or "auto"
        ),
    )
    return S3RecordingStore(
        client=client,
        bucket=bucket,
        prefix=(getattr(settings, "live_recording_storage_prefix", None) or "live"),
    )


from .keys import (  # noqa: E402 — re-export the key builders
    clip_key_family,
    clip_media_key,
    clip_render_key,
    clip_thumbnail_key,
)

__all__ = [
    "ArtifactStore",
    "RecordingStore",
    "S3ArtifactStore",
    "S3RecordingStore",
    "build_artifact_store",
    "build_recording_store",
    "clip_key_family",
    "clip_media_key",
    "clip_render_key",
    "clip_thumbnail_key",
]

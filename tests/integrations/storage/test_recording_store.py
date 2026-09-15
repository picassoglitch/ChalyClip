"""Phase L.2 / Path B — S3-compatible recording store.

`S3RecordingStore` lists `<prefix>/<stream_id>/` in an S3-compatible
bucket (Supabase Storage, R2, MinIO, …) and downloads the newest
non-trivial object. `build_recording_store` returns None unless a bucket
is configured (so live ingest falls back to the shared-disk path). Both
are exercised here against a fake S3 client — no network, no boto3.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chalybclip.integrations.storage import (
    S3RecordingStore,
    build_recording_store,
)


class _FakeS3:
    """Minimal stand-in for the boto3 s3 client surface we use."""

    def __init__(self, objects: list[dict[str, Any]]) -> None:
        self._objects = objects
        self.downloaded: list[tuple[str, str]] = []

    def list_objects_v2(self, *, Bucket: str, Prefix: str) -> dict[str, Any]:  # noqa: N803
        return {
            "Contents": [o for o in self._objects if str(o["Key"]).startswith(Prefix)]
        }

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:  # noqa: N803
        Path(Filename).write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 4096)
        self.downloaded.append((Key, Filename))


async def test_fetch_latest_downloads_newest(tmp_path: Path) -> None:
    """Picks the newest object under live/<stream_id>/ and downloads it."""
    s3 = _FakeS3([
        {"Key": "live/str_1/2026-01-01_00-00-00.mp4", "Size": 5_000_000, "LastModified": 100},
        {"Key": "live/str_1/2026-01-01_01-00-00.mp4", "Size": 9_000_000, "LastModified": 200},
        {"Key": "live/str_OTHER/x.mp4", "Size": 9_000_000, "LastModified": 999},
    ])
    store = S3RecordingStore(client=s3, bucket="rec", prefix="live")

    got = await store.fetch_latest(stream_id="str_1", dest_dir=tmp_path / "in")

    assert got is not None
    assert got.name == "2026-01-01_01-00-00.mp4"   # newest by LastModified
    assert got.exists()
    # Only the matching stream's object was downloaded.
    assert len(s3.downloaded) == 1
    assert s3.downloaded[0][0] == "live/str_1/2026-01-01_01-00-00.mp4"


async def test_fetch_latest_none_when_empty(tmp_path: Path) -> None:
    """No objects yet (upload still in flight) → None, so the runner keeps
    polling."""
    s3 = _FakeS3([])
    store = S3RecordingStore(client=s3, bucket="rec", prefix="live")
    assert await store.fetch_latest(stream_id="str_1", dest_dir=tmp_path) is None


async def test_fetch_latest_skips_tiny_objects(tmp_path: Path) -> None:
    """A 0/near-0-byte placeholder is ignored (treated as not-yet-uploaded)."""
    s3 = _FakeS3([
        {"Key": "live/str_1/partial.mp4", "Size": 12, "LastModified": 100},
    ])
    store = S3RecordingStore(client=s3, bucket="rec", prefix="live")
    assert await store.fetch_latest(stream_id="str_1", dest_dir=tmp_path) is None
    assert s3.downloaded == []


def test_build_store_none_without_bucket() -> None:
    """No bucket configured → None (live ingest uses the shared-disk path)."""
    settings = SimpleNamespace(live_recording_storage_bucket=None)
    assert build_recording_store(settings) is None  # type: ignore[arg-type]

    settings = SimpleNamespace(live_recording_storage_bucket="   ")  # blank/whitespace
    assert build_recording_store(settings) is None  # type: ignore[arg-type]


def test_build_store_requires_boto3_only_when_configured() -> None:
    """With a bucket set, build_recording_store tries to import boto3. If
    boto3 isn't installed it raises ImportError — but only then, never on
    the unconfigured path."""
    settings = SimpleNamespace(
        live_recording_storage_bucket="rec",
        live_recording_storage_endpoint="https://acct.r2.cloudflarestorage.com",
        live_recording_storage_access_key_id="k",
        live_recording_storage_secret_access_key="s",
        live_recording_storage_region="auto",
        live_recording_storage_prefix="live",
    )
    try:
        store = build_recording_store(settings)  # type: ignore[arg-type]
    except ImportError:
        pytest.skip("boto3 not installed in this env")
    else:
        assert store is not None
        assert isinstance(store, S3RecordingStore)

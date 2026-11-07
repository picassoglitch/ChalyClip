"""Phase 1 — general artifact store (durable off-box pipeline artifacts).

`S3ArtifactStore` uploads/serves/deletes artifacts in an S3-compatible
bucket and mints presigned GET urls for the publisher. `build_artifact_store`
returns None unless a bucket is configured (→ local-disk serving, current
behavior). Exercised against a fake S3 client — no network, no boto3.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from chalybclip.integrations.storage import (
    ArtifactStore,
    S3ArtifactStore,
    build_artifact_store,
)


class _FakeS3:
    """Minimal stand-in for the boto3 s3 surface S3ArtifactStore uses."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}  # full key -> bytes
        self.presigned: list[tuple[str, str, int]] = []

    def upload_file(
        self, Filename: str, Bucket: str, Key: str, ExtraArgs: Any = None  # noqa: N803
    ) -> None:
        self.store[Key] = Path(Filename).read_bytes()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:  # noqa: N803
        if Key not in self.store:
            raise FileNotFoundError(Key)
        Path(Filename).write_bytes(self.store[Key])

    def generate_presigned_url(
        self, op: str, params: dict[str, str], ttl: int
    ) -> str:
        self.presigned.append((op, params["Key"], ttl))
        return f"https://bucket.example/{params['Key']}?exp={ttl}"

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self.store:
            raise FileNotFoundError(Key)
        return {"ContentLength": len(self.store[Key])}

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        self.store.pop(Key, None)


def _store(s3: _FakeS3, prefix: str = "artifacts") -> S3ArtifactStore:
    return S3ArtifactStore(client=s3, bucket="b", prefix=prefix)


async def test_upload_then_download_roundtrip(tmp_path: Path) -> None:
    s3 = _FakeS3()
    store = _store(s3)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video-bytes")

    await store.upload(local_path=src, key="str_1/clip.mp4", content_type="video/mp4")
    # Key is prefixed.
    assert "artifacts/str_1/clip.mp4" in s3.store

    out = await store.download(key="str_1/clip.mp4", dest=tmp_path / "got.mp4")
    assert out is not None
    assert out.read_bytes() == b"video-bytes"


async def test_download_missing_returns_none(tmp_path: Path) -> None:
    store = _store(_FakeS3())
    assert await store.download(key="nope.mp4", dest=tmp_path / "x.mp4") is None


async def test_presigned_url_is_prefixed_and_ttl_passed() -> None:
    s3 = _FakeS3()
    url = await _store(s3).presigned_url(key="str_1/clip.mp4", ttl_seconds=3600)
    assert "artifacts/str_1/clip.mp4" in url
    assert s3.presigned == [("get_object", "artifacts/str_1/clip.mp4", 3600)]


async def test_exists_true_false() -> None:
    s3 = _FakeS3()
    s3.store["artifacts/a.mp4"] = b"x"
    store = _store(s3)
    assert await store.exists(key="a.mp4") is True
    assert await store.exists(key="missing.mp4") is False


async def test_delete_removes_and_is_idempotent() -> None:
    s3 = _FakeS3()
    s3.store["artifacts/a.mp4"] = b"x"
    store = _store(s3)
    await store.delete(key="a.mp4")
    assert "artifacts/a.mp4" not in s3.store
    await store.delete(key="a.mp4")  # no raise on missing


async def test_no_prefix_keys_are_bare() -> None:
    s3 = _FakeS3()
    store = _store(s3, prefix="")
    src_present = "k.mp4"
    await store.presigned_url(key=src_present, ttl_seconds=1)
    assert s3.presigned[-1][1] == "k.mp4"


def test_build_artifact_store_none_without_bucket() -> None:
    s = SimpleNamespace(object_storage_bucket=None)
    assert build_artifact_store(s) is None  # type: ignore[arg-type]


def test_s3_artifact_store_satisfies_protocol() -> None:
    assert isinstance(_store(_FakeS3()), ArtifactStore)


async def test_upload_sends_disposition_and_metadata_and_reads_them_back(tmp_path: Path) -> None:
    class _MetaS3(_FakeS3):
        def __init__(self) -> None:
            super().__init__()
            self.extra: dict[str, Any] = {}

        def upload_file(self, Filename: str, Bucket: str, Key: str, ExtraArgs: Any = None) -> None:  # noqa: N803
            super().upload_file(Filename, Bucket, Key, ExtraArgs)
            self.extra[Key] = ExtraArgs

        def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            if Key not in self.store:
                from botocore.exceptions import ClientError

                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            out = super().head_object(Bucket=Bucket, Key=Key)
            # boto3 hands user metadata back without the x-amz-meta- prefix.
            out["Metadata"] = dict((self.extra.get(Key) or {}).get("Metadata") or {})
            return out

    s3 = _MetaS3()
    store = _store(s3)
    f = tmp_path / "e.mp4"
    f.write_bytes(b"x")
    await store.upload(
        local_path=f, key="clips/t/c/export_4k.mp4", content_type="video/mp4",
        content_disposition='attachment; filename="c.mp4"', metadata={"fp": "abc"},
    )
    assert s3.extra["artifacts/clips/t/c/export_4k.mp4"] == {
        "ContentType": "video/mp4",
        "ContentDisposition": 'attachment; filename="c.mp4"',
        "Metadata": {"fp": "abc"},
    }
    assert await store.metadata(key="clips/t/c/export_4k.mp4") == {"fp": "abc"}
    assert await store.metadata(key="clips/t/c/missing.mp4") is None


async def test_plain_upload_still_sends_no_extra_args(tmp_path: Path) -> None:
    calls: list[Any] = []

    class _S3(_FakeS3):
        def upload_file(self, Filename: str, Bucket: str, Key: str, ExtraArgs: Any = None) -> None:  # noqa: N803
            calls.append(ExtraArgs)
            super().upload_file(Filename, Bucket, Key, ExtraArgs)

    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    await _store(_S3()).upload(local_path=f, key="k")
    assert calls == [None]


async def test_metadata_raises_on_denied_but_not_on_missing() -> None:
    import pytest
    from botocore.exceptions import ClientError

    class _S3(_FakeS3):
        def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            code = "404" if Key.endswith("missing") else "403"
            raise ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": int(code)}}, "HeadObject")

    store = _store(_S3())
    assert await store.metadata(key="k/missing") is None
    with pytest.raises(ClientError):
        await store.metadata(key="k/denied")

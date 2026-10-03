"""Dashboard uploads run on the worker: the web box parks the raw file in
the bucket and dispatches; the worker downloads it and runs the normal
upload pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chalybclip.api import _pipeline
from chalybclip.jobs import ModalJobDispatcher, PipelineKickoff


class _FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_upload = False

    async def upload(self, *, local_path: Path, key: str, content_type: str | None = None) -> None:
        if self.fail_upload:
            raise RuntimeError("bucket down")
        self.objects[key] = local_path.read_bytes()

    async def download(self, *, key: str, dest: Path) -> Path | None:
        if key not in self.objects:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[key])
        return dest

    async def delete(self, *, key: str) -> None:
        self.objects.pop(key, None)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _FakeStore:
    s = _FakeStore()
    monkeypatch.setattr(
        "chalybclip.integrations.storage.build_artifact_store", lambda _settings: s
    )
    return s


def _modal_dispatcher(sent: list[PipelineKickoff]) -> ModalJobDispatcher:
    d = ModalJobDispatcher(endpoint_url="https://worker.test", bearer_token="tok")

    async def _dispatch(kickoff: PipelineKickoff, **_: Any) -> None:
        sent.append(kickoff)

    d.dispatch_pipeline = _dispatch  # type: ignore[method-assign]
    return d


async def test_web_box_parks_upload_and_dispatches(store: _FakeStore, tmp_path: Path) -> None:
    upload = tmp_path / "up.mov"
    upload.write_bytes(b"video-bytes")
    sent: list[PipelineKickoff] = []

    handed = await _pipeline._hand_upload_to_worker(
        dispatcher=_modal_dispatcher(sent),
        tenant_id="ten_A",
        stream_id="str_A",
        persona_id="per_A",
        tmp_path=upload,
        output_dir=tmp_path / "out",
        title="clip.mov",
        language="auto",
    )

    assert handed is True
    assert store.objects == {"uploads/ten_A/str_A/source.mov": b"video-bytes"}
    assert not upload.exists()  # local copy dropped
    (k,) = sent
    assert k.source_object_key == "uploads/ten_A/str_A/source.mov"
    assert k.stream.vod_url == "upload://clip.mov"
    assert k.title == "clip.mov"


async def test_in_process_dispatcher_keeps_local_path(store: _FakeStore, tmp_path: Path) -> None:
    upload = tmp_path / "up.mp4"
    upload.write_bytes(b"x")
    handed = await _pipeline._hand_upload_to_worker(
        dispatcher=object(),
        tenant_id="ten_A", stream_id="str_A", persona_id="per_A",
        tmp_path=upload, output_dir=tmp_path, title=None, language=None,
    )
    assert handed is False
    assert upload.exists() and store.objects == {}


async def test_bucket_failure_falls_back_to_in_process(store: _FakeStore, tmp_path: Path) -> None:
    store.fail_upload = True
    upload = tmp_path / "up.mp4"
    upload.write_bytes(b"x")
    sent: list[PipelineKickoff] = []
    handed = await _pipeline._hand_upload_to_worker(
        dispatcher=_modal_dispatcher(sent),
        tenant_id="ten_A", stream_id="str_A", persona_id="per_A",
        tmp_path=upload, output_dir=tmp_path, title=None, language=None,
    )
    assert handed is False and sent == [] and upload.exists()


async def test_worker_downloads_deletes_and_runs_upload_pipeline(
    store: _FakeStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.objects["uploads/ten_A/str_A/source.mov"] = b"video-bytes"
    ran: list[dict[str, Any]] = []

    async def fake_upload_runner(**kw: Any) -> None:
        ran.append({**kw, "bytes": Path(kw["tmp_path"]).read_bytes()})

    monkeypatch.setattr(_pipeline, "upload_pipeline_runner", fake_upload_runner)
    from chalybclip.ingest import Stream

    kickoff = PipelineKickoff(
        tenant_id="ten_A",
        stream=Stream(
            id="str_A", tenant_id="ten_A", vod_url="upload://clip.mov",
            platform="upload", duration_s=0.0,
            source_video_path=tmp_path / "v.mp4", source_audio_path=tmp_path / "a.wav",
        ),
        persona_id="per_A",
        output_dir=tmp_path / "out",
        language="es",
        source_object_key="uploads/ten_A/str_A/source.mov",
        title="clip.mov",
    )
    await _pipeline.remote_upload_runner(kickoff)

    assert store.objects == {}  # bucket copy removed
    (kw,) = ran
    assert kw["bytes"] == b"video-bytes"
    assert kw["stream_id"] == "str_A" and kw["title"] == "clip.mov"
    assert "dispatcher" not in kw  # runs locally on the worker


async def test_worker_missing_object_surfaces_failure(
    store: _FakeStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emitted: list[str] = []

    async def fake_emit(**kw: Any) -> None:
        emitted.append(str(kw["error"]))

    monkeypatch.setattr(_pipeline, "_emit_top_level_failure", fake_emit)
    from chalybclip.errors import IngestError
    from chalybclip.ingest import Stream

    kickoff = PipelineKickoff(
        tenant_id="ten_A",
        stream=Stream(
            id="str_A", tenant_id="ten_A", vod_url="upload://x.mp4",
            platform="upload", duration_s=0.0,
            source_video_path=tmp_path / "v.mp4", source_audio_path=tmp_path / "a.wav",
        ),
        persona_id="per_A",
        output_dir=tmp_path,
        source_object_key="uploads/ten_A/str_A/source.mp4",
    )
    with pytest.raises(IngestError):
        await _pipeline.remote_upload_runner(kickoff)
    assert emitted and "missing from object storage" in emitted[0]

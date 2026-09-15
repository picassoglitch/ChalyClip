"""LocalWhisperProvider CPU fallback — a CUDA-side native crash retries
once on cpu/int8 instead of failing the run with edit-your-.env advice."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chalybclip.errors import TranscriptionError
from chalybclip.transcribe.models import Transcript
from chalybclip.transcribe.providers.local_whisper import LocalWhisperProvider


def _transcript_json(stream_id: str = "str_x") -> str:
    return Transcript(
        stream_id=stream_id, tenant_id="t", language="en",
        duration_s=1.0, model="small", segments=[],
    ).model_dump_json()


class _Result:
    def __init__(self, returncode: int, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr


def _patch_run(monkeypatch: pytest.MonkeyPatch, behavior) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: Any) -> _Result:
        calls.append(cmd)
        return behavior(cmd)

    monkeypatch.setattr(
        "chalybclip.transcribe.providers.local_whisper.subprocess.run", fake_run
    )
    return calls


def _device_of(cmd: list[str]) -> str:
    return cmd[cmd.index("--device") + 1]


def _out_of(cmd: list[str]) -> Path:
    return Path(cmd[cmd.index("--out") + 1])


@pytest.mark.asyncio
async def test_cuda_crash_falls_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def behavior(cmd: list[str]) -> _Result:
        if _device_of(cmd) == "cuda":
            return _Result(3221226505, "faulthandler dump ...")  # 0xC0000409
        _out_of(cmd).write_text(_transcript_json(), encoding="utf-8")
        return _Result(0)

    calls = _patch_run(monkeypatch, behavior)
    provider = LocalWhisperProvider(
        model_size="small", device="cuda", compute_type="int8_float16"
    )
    result = await provider.transcribe(_req(tmp_path))

    assert result.language == "en"
    assert [_device_of(c) for c in calls] == ["cuda", "cpu"]
    # The fallback runs int8 on cpu regardless of the configured compute type.
    assert calls[1][calls[1].index("--compute") + 1] == "int8"


@pytest.mark.asyncio
async def test_cpu_crash_does_not_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _patch_run(monkeypatch, lambda cmd: _Result(1, "boom"))
    provider = LocalWhisperProvider(
        model_size="small", device="cpu", compute_type="int8"
    )
    with pytest.raises(TranscriptionError, match="exited with code 1"):
        await provider.transcribe(_req(tmp_path))
    assert len(calls) == 1  # already on cpu: no retry ladder


@pytest.mark.asyncio
async def test_double_crash_surfaces_second_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _patch_run(monkeypatch, lambda cmd: _Result(3221226505, "dead gpu"))
    provider = LocalWhisperProvider(
        model_size="small", device="cuda", compute_type="float16"
    )
    with pytest.raises(TranscriptionError, match="cpu/int8"):
        await provider.transcribe(_req(tmp_path))
    assert [_device_of(c) for c in calls] == ["cuda", "cpu"]


def _req(tmp_path: Path):
    from chalybclip.transcribe.providers.base import TranscribeRequest

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00")
    return TranscribeRequest(
        audio_path=audio, stream_id="str_x", tenant_id="t", language=None,
    )

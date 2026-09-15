"""Subprocess worker for Whisper transcription.

The dashboard process spawns this as a child so a hard Whisper crash
(CUDA OOM SIGABRT, driver SIGSEGV, etc.) only kills the worker — the
parent dashboard survives and emits a clean `pipeline.step.failed`
event with the worker's exit code + tail of stderr.

Contract:
    Args:
        --audio    path to source WAV
        --model    Whisper model size
        --device   cuda | cpu
        --compute  float16 / int8 / int8_float16 / ...
        --language ISO 639-1 code or empty string for auto-detect
        --out      path to write Transcript JSON
    Exit codes:
        0  — success; transcript written to --out
        != 0  — failure; stderr has the error
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path


def _main() -> int:
    # Hard-crash diagnostics — Whisper on CUDA on Windows can SIGABRT
    # without raising a Python exception. faulthandler at least dumps
    # a stack to stderr so the parent has something to report.
    faulthandler.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--compute", required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    from chalybclip.transcribe.models import Segment, Transcript, Word

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"audio file missing: {audio_path}", file=sys.stderr)
        return 2

    language = args.language or None

    try:
        model = WhisperModel(
            args.model, device=args.device, compute_type=args.compute
        )
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            beam_size=1,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"Whisper failed to start "
            f"({args.model}/{args.device}/{args.compute}): {e}",
            file=sys.stderr,
        )
        return 3

    segments: list[Segment] = []
    try:
        for seg in segments_iter:
            words = [
                Word(
                    ts=float(w.start),
                    end_ts=float(w.end),
                    text=w.word,
                    prob=float(w.probability),
                )
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(
                    ts=float(seg.start),
                    end_ts=float(seg.end),
                    text=seg.text,
                    words=words,
                )
            )
    except Exception as e:  # noqa: BLE001
        print(f"Whisper segment iteration failed: {e}", file=sys.stderr)
        return 4

    transcript = Transcript(
        stream_id=args.stream_id,
        tenant_id=args.tenant_id,
        language=info.language,
        duration_s=float(info.duration),
        model=args.model,
        segments=segments,
    )
    Path(args.out).write_text(
        transcript.model_dump_json(indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

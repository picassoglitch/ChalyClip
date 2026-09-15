"""Subprocess worker for pyannote-3.1 speaker diarization.

The dashboard process spawns this as a child so a hard pyannote crash
(CUDA OOM SIGABRT, torch / driver SIGSEGV) only kills the worker — the
parent dashboard survives and emits a clean `pipeline.step.failed`
event with the worker's exit code + stderr tail.

Exit codes:
    0  — success; Diarization JSON written to --out
    2  — audio file missing
    3  — pyannote.audio not importable (skippable case, parent should warn)
    4  — pyannote pipeline failed to load (e.g. HF_TOKEN missing / unauthorized)
    5  — diarization runtime error
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path


def _main() -> int:
    # Same diagnostic safety net we ship with Whisper — pyannote can also
    # SIGABRT on CUDA OOM. faulthandler at least dumps a partial stack.
    faulthandler.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="HuggingFace model id for the diarization pipeline.",
    )
    parser.add_argument(
        "--embedding-model",
        default="pyannote/embedding",
        help="HuggingFace model id for per-speaker voice embeddings.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device — cuda | cpu. Falls back to cpu if cuda fails.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"audio file missing: {audio_path}", file=sys.stderr)
        return 2

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print(
            "HF_TOKEN not set. pyannote-3.1 is a gated HuggingFace model — "
            "accept the license at https://hf.co/pyannote/speaker-diarization-3.1 "
            "and https://hf.co/pyannote/segmentation-3.0, create a read token "
            "at https://hf.co/settings/tokens, and add HF_TOKEN=hf_... to .env.",
            file=sys.stderr,
        )
        return 4

    try:
        import torch  # noqa: F401
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook  # noqa: F401
    except ImportError as e:
        print(
            f"pyannote.audio not installed: {e}. "
            f"Run: pip install 'pyannote.audio>=3.1,<4'",
            file=sys.stderr,
        )
        return 3

    try:
        pipeline = Pipeline.from_pretrained(args.model, use_auth_token=hf_token)
        if pipeline is None:
            print(
                f"Failed to load {args.model}. Most likely the HF_TOKEN "
                f"doesn't have access — accept the license at "
                f"https://hf.co/{args.model}",
                file=sys.stderr,
            )
            return 4
    except Exception as e:  # noqa: BLE001
        print(f"Failed to construct pyannote pipeline: {e}", file=sys.stderr)
        return 4

    # Move to GPU if requested + available.
    try:
        import torch as _torch

        if args.device == "cuda" and _torch.cuda.is_available():
            pipeline.to(_torch.device("cuda"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not move pipeline to {args.device}: {e}", file=sys.stderr)

    # ---- run diarization ----
    try:
        diarization = pipeline(str(audio_path))
    except Exception as e:  # noqa: BLE001
        print(f"diarization runtime error: {e}", file=sys.stderr)
        return 5

    # Collect turns.
    segments: list[dict[str, object]] = []
    speaker_totals: dict[str, float] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "ts": float(turn.start),
                "end_ts": float(turn.end),
                "speaker_label": str(speaker),
            }
        )
        speaker_totals[str(speaker)] = (
            speaker_totals.get(str(speaker), 0.0) + (turn.end - turn.start)
        )

    # ---- per-speaker embeddings ----
    # The pyannote embedding model returns a vector per audio segment.
    # We aggregate per speaker_label using a duration-weighted mean.
    embeddings_payload: list[dict[str, object]] = []
    try:
        from pyannote.audio import Inference, Model

        emb_model = Model.from_pretrained(
            args.embedding_model, use_auth_token=hf_token
        )
        inference = Inference(emb_model, window="whole")
        try:
            import torch as _torch

            if args.device == "cuda" and _torch.cuda.is_available():
                inference.to(_torch.device("cuda"))
        except Exception:
            pass

        # Compute one embedding per turn, then weighted-average per speaker.
        from pyannote.core import Segment as _PSeg

        sums: dict[str, list[float]] = {}
        weights: dict[str, float] = {}
        for seg in segments:
            duration = float(seg["end_ts"]) - float(seg["ts"])
            if duration < 0.5:
                continue  # skip too-short turns; embedding is noise
            crop = _PSeg(float(seg["ts"]), float(seg["end_ts"]))
            try:
                vec = inference.crop(str(audio_path), crop)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[warn] embedding crop failed for {seg['speaker_label']} "
                    f"@ {seg['ts']:.1f}-{seg['end_ts']:.1f}: {e}",
                    file=sys.stderr,
                )
                continue
            label = str(seg["speaker_label"])
            vec_list = [float(x) for x in vec.flatten().tolist()]
            cur = sums.get(label)
            if cur is None:
                sums[label] = [v * duration for v in vec_list]
                weights[label] = duration
            else:
                for i, v in enumerate(vec_list):
                    cur[i] += v * duration
                weights[label] += duration
        for label, summed in sums.items():
            w = weights[label]
            if w <= 0.0:
                continue
            avg = [v / w for v in summed]
            embeddings_payload.append(
                {
                    "speaker_label": label,
                    "embedding": avg,
                    "total_speech_s": speaker_totals.get(label, 0.0),
                }
            )
    except Exception as e:  # noqa: BLE001
        # Embeddings are nice-to-have. Segments alone are still useful.
        print(
            f"[warn] embedding extraction failed: {e}. Segments will be "
            f"emitted without speaker fingerprints.",
            file=sys.stderr,
        )

    payload = {
        "stream_id": args.stream_id,
        "tenant_id": args.tenant_id,
        "segments": segments,
        "embeddings": embeddings_payload,
        "skipped": False,
        "skip_reason": None,
    }
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

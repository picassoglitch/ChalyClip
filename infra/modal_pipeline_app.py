"""Modal app: the full ChalyClip VOD pipeline on a CPU worker (Phase 2b).

Deploy from the repo root (the image bundles the local `chalybclip/` source
plus `config/`, so deploy re-runs ship code changes):

    pip install modal
    modal token new                              # one-time Modal auth
    modal deploy infra/modal_pipeline_app.py

Modal prints the public endpoint URL — copy it into Railway as
CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL and set CHALYBCLIP_JOB_DISPATCHER=modal.
Full env checklist + rollout/rollback: docs/modal_pipeline_runbook.md.

The worker runs the SAME `default_pipeline_runner` the web box runs
in-process — against the shared Railway Postgres (step events, rows, LLM
cost tracking, failure surfacing, post-run balance refresh) and the shared
R2 bucket (Phase 2a uploads clip.mp4 + thumbnails there as part of the
run). The container's disk is scratch: the downloaded source evaporates
with the container, which is exactly the point — nothing heavy ever
touches the web box's CPU or volume.

Request JSON (built by chalybclip/jobs/modal.py::ModalJobDispatcher):
    {
        "auth_token": "<bearer, same as CHALYBCLIP_MODAL_TOKEN>",
        "tenant_id": "ten_...",
        "persona_id": "per_...",
        "language": "es" | null,
        "stream": { ...Stream.model_dump(mode="json")... },
    }

Response JSON (HTTP 200 either way — a "failed" body means the runner
already emitted its own pipeline.failed event, so the dispatcher only
logs it):
    {"stream_id": "...", "status": "done"|"failed",
     "error": "...", "n_clips": 3, "elapsed_s": 812.4}
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

# Cost ceiling: how many pipeline containers may run at once. Each is one
# full VOD run (8 CPUs for its duration). Read at DEPLOY time from the
# operator's shell — re-deploy to change it.
_MAX_CONTAINERS = int(os.environ.get("CHALYBCLIP_MODAL_PIPELINE_MAX_CONTAINERS", "5"))

# One full pipeline run: download + analyze + transcribe(remote) + detect
# + cut + variants. Multi-hour VODs on CPU need hours; 8h is the hard kill.
_TIMEOUT_S = 8 * 3600

_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    # ffmpeg: cut/reformat/audio-extract. libgl1 + libglib2.0-0: OpenCV's
    # runtime .so deps (opencv-python, not -headless, per the repo pin).
    # Fonts: the cut path's drawtext branding resolves
    # /usr/share/fonts/truetype/{dejavu,liberation}/... (clip/service.py,
    # thumbnail_brand.py) — without these the handle/watermark overlays
    # silently skip on the worker.
    .apt_install(
        "ffmpeg", "libgl1", "libglib2.0-0",
        "fonts-dejavu-core", "fonts-liberation",
    )
    # The package's own dependency list — single source of truth, so a
    # requirements bump on the web box reaches the worker on redeploy.
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("fastapi[standard]>=0.115")  # Modal 1.x web endpoint dep
    # Local files LAST (Modal 1.x constraint): the chalybclip package and
    # the config/ dir (pipeline reads config/chalybclip[.example].yaml
    # relative to CWD, which is /root in Modal containers).
    #
    # The gitignored operator-local config/chalybclip.yaml is EXCLUDED on
    # purpose: Railway prod deliberately runs on chalybclip.example.yaml
    # (the real one never ships), and the worker must use the same
    # detection/LLM defaults as the web box — not the deploy machine's
    # local experiments.
    .add_local_dir(
        "config",
        remote_path="/root/config",
        ignore=["chalybclip.yaml"],
    )
    .add_local_python_source("chalybclip")
    # add_local_python_source ships ONLY *.py files. The pipeline also
    # needs the package's data files at runtime, overlaid onto the same
    # /root/chalybclip tree:
    #   - db/migrations/*.sql + db/pg_baseline.sql — apply_migrations
    #     reads them at every db_session open (even against an
    #     already-migrated Postgres, it must read the files to discover
    #     versions); without them EVERY remote run dies at startup with
    #     MigrationError.
    #   - clip/assets/outro.mp4 — the outro end-card (append_outro only
    #     degrades to a warning, but worker clips would silently differ
    #     from in-process ones).
    .add_local_dir(
        "chalybclip/db/migrations", remote_path="/root/chalybclip/db/migrations"
    )
    .add_local_file(
        "chalybclip/db/pg_baseline.sql",
        remote_path="/root/chalybclip/db/pg_baseline.sql",
    )
    .add_local_dir(
        "chalybclip/clip/assets", remote_path="/root/chalybclip/clip/assets"
    )
)

app = modal.App("chalybclip-pipeline")


def _materialize_cookies_file() -> None:
    """Write inline yt-dlp cookies (CHALYBCLIP_COOKIES_TXT) to a real file
    and point CHALYBCLIP_COOKIES_FILE at it — the worker-side twin of
    run.py's boot step, needed because YouTube bot-gates datacenter IPs
    (Modal's included). Same base64 escape hatch."""
    raw = (os.environ.get("CHALYBCLIP_COOKIES_TXT") or "").strip()
    if not raw or os.environ.get("CHALYBCLIP_COOKIES_FILE"):
        return
    if os.environ.get("CHALYBCLIP_COOKIES_TXT_B64", "").strip() == "1":
        import base64

        raw = base64.b64decode(raw).decode("utf-8", errors="replace")
    dest = Path("/root/cookies.txt")
    dest.write_text(raw, encoding="utf-8")
    os.environ["CHALYBCLIP_COOKIES_FILE"] = str(dest)


def _worker_preflight() -> str | None:
    """Refuse to run when the results would evaporate with the container.

    Returns an error string (→ HTTP 500 at the endpoint) or None when the
    env is sane. Two hard requirements:
      * DATABASE_URL — without shared Postgres, rows/events land in a
        container-local SQLite file and the dashboard never sees the run;
      * CHALYBCLIP_OBJECT_STORAGE_BUCKET — without R2, the cut clips die
        with the container's disk (Phase 2a is what makes them durable).
    """
    if not (os.environ.get("DATABASE_URL") or "").strip():
        return (
            "worker misconfigured: DATABASE_URL is not set in the "
            "chalybclip-pipeline-env Modal secret — the run's rows/events "
            "would be written to a throwaway container-local SQLite file"
        )
    if not (os.environ.get("CHALYBCLIP_OBJECT_STORAGE_BUCKET") or "").strip():
        return (
            "worker misconfigured: CHALYBCLIP_OBJECT_STORAGE_BUCKET is not "
            "set in the chalybclip-pipeline-env Modal secret — cut clips "
            "would evaporate with the container's disk"
        )
    return None


@app.function(
    image=_IMAGE,
    cpu=8.0,  # analyze_video's 3 CV passes + up to 3 concurrent ffmpeg cuts
    memory=16384,
    timeout=_TIMEOUT_S,
    scaledown_window=300,  # warm 5 min — channel-poll bursts reuse the box
    max_containers=_MAX_CONTAINERS,
    secrets=[modal.Secret.from_name("chalybclip-pipeline-env")],
)
@modal.fastapi_endpoint(method="POST")
async def run_pipeline(payload: dict) -> dict:
    """Run the full pipeline for one serialized PipelineKickoff."""
    from fastapi import HTTPException

    expected_token = (
        os.environ.get("MODAL_BEARER_TOKEN")
        or os.environ.get("CHALYBCLIP_MODAL_TOKEN")
        or ""
    )
    request_token = str((payload or {}).get("auth_token", ""))
    # Constant-time compare (bytes — str compare_digest rejects non-ASCII):
    # this public URL's only guard is the token, and `!=` leaks a timing
    # oracle byte by byte.
    import hmac as _hmac

    if not expected_token or not _hmac.compare_digest(
        request_token.encode("utf-8"), expected_token.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid bearer token")

    preflight_error = _worker_preflight()
    if preflight_error is not None:
        raise HTTPException(status_code=500, detail=preflight_error)

    tenant_id = str((payload or {}).get("tenant_id") or "").strip()
    persona_id = str((payload or {}).get("persona_id") or "").strip()
    language = (payload or {}).get("language") or None
    stream_raw = (payload or {}).get("stream") or {}
    if not tenant_id or not persona_id or not stream_raw:
        raise HTTPException(
            status_code=400,
            detail="tenant_id, persona_id and stream are required",
        )

    # Config + cookies live under /root (see _IMAGE); make relative reads
    # (config/chalybclip.yaml) resolve regardless of Modal's default CWD.
    os.chdir("/root")
    _materialize_cookies_file()

    from chalybclip.api._pipeline import default_pipeline_runner
    from chalybclip.ingest import Stream
    from chalybclip.jobs import PipelineKickoff
    from chalybclip.settings import get_settings

    try:
        stream = Stream.model_validate(stream_raw)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"stream payload invalid: {e}"
        ) from e

    # The worker's own scratch root — NEVER the web box's path. The
    # Phase 2a offload inside the run persists what matters to R2.
    output_dir = Path(
        os.environ.get("CHALYBCLIP_DEFAULT_OUTPUT_DIR") or "/tmp/chalybclip-out"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    get_settings.cache_clear()  # pick up the env this container booted with

    kickoff = PipelineKickoff(
        tenant_id=tenant_id,
        stream=stream,
        persona_id=persona_id,
        output_dir=output_dir,
        language=language,
    )

    started_at = time.time()
    print(
        f"chalybclip-pipeline: run start stream={stream.id} "
        f"tenant={tenant_id} persona={persona_id}"
    )
    try:
        # The same runner the web box uses in-process: step events,
        # pipeline.failed surfacing, source reclaim, base-fee charge and
        # balance refresh all behave identically.
        await default_pipeline_runner(kickoff)
    except Exception as e:
        # The runner already emitted pipeline.failed to the shared DB —
        # return a terminal body (HTTP 200) so the dispatcher logs it
        # without double-emitting.
        print(
            f"chalybclip-pipeline: run FAILED stream={stream.id}: "
            f"{type(e).__name__}: {e}"
        )
        return {
            "stream_id": stream.id,
            "status": "failed",
            "error": f"{type(e).__name__}: {str(e)[:500]}",
            "elapsed_s": time.time() - started_at,
        }

    n_clips = _count_clips_from_manifest(output_dir / stream.id)
    print(
        f"chalybclip-pipeline: run done stream={stream.id} "
        f"n_clips={n_clips} elapsed_s={time.time() - started_at:.0f}"
    )
    return {
        "stream_id": stream.id,
        "status": "done",
        "error": None,
        "n_clips": n_clips,
        "elapsed_s": time.time() - started_at,
    }


def _count_clips_from_manifest(stream_dir: Path) -> int | None:
    """Best-effort clip count for the dispatcher's terminal log line."""
    try:
        manifest = json.loads(
            (stream_dir / "manifest.json").read_text(encoding="utf-8")
        )
        clips = manifest.get("clip_entries")
        return len(clips) if isinstance(clips, list) else None
    except Exception:
        return None

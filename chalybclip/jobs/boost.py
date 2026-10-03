"""Boost lane — run one admitted job on a dedicated Cloud Run Job execution.

When /usage/admit answers `lane: "boost"`, the run doesn't go to the shared
worker: the kickoff is persisted on its `usage_jobs` row and the engine's
`<engine>-boost` Cloud Run Job (8 vCPU / 32 GiB, max_retries 0) is started
with one per-execution env override, CHALYBCLIP_BOOST_JOB_ID. That
container (`python -m chalybclip.workers.boost_job`) loads the kickoff,
runs exactly that job through the same runner the worker uses, and exits.

    POST https://run.googleapis.com/v2/projects/{project}/locations/{region}
         /jobs/{CHALYBCLIP_BOOST_JOB_NAME}:run
    { "overrides": { "containerOverrides": [
        { "env": [ { "name": "CHALYBCLIP_BOOST_JOB_ID", "value": <id> } ] } ] } }

Auth is the metadata-server access token of the service's own SA (it holds
roles/run.jobsExecutorWithOverrides on the job) — no google-auth dependency.
Project/region come from GOOGLE_CLOUD_PROJECT / CLOUD_RUN_REGION, falling
back to the metadata server.

Fallback: boost not configured (CHALYBCLIP_BOOST_JOB_NAME unset), a source
the job can't reach (a file on this box's disk), or the run API failing →
the job is re-admitted with `boost: false` (so no boost fee is charged for
a run that didn't get the boost machine) and goes to the standard lane.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from chalybclip.settings import get_settings

from .base import JobDispatcher, PipelineKickoff

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

_log = logging.getLogger("chalybclip.jobs.boost")

_METADATA = "http://metadata.google.internal/computeMetadata/v1"
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}
_RUN_API = "https://run.googleapis.com/v2"
_HTTP_TIMEOUT = httpx.Timeout(20.0, connect=5.0)

BOOST_JOB_ID_ENV = "CHALYBCLIP_BOOST_JOB_ID"


class BoostDispatchError(Exception):
    """The Cloud Run Jobs run call could not be made or was rejected."""


def boost_configured() -> bool:
    return bool((get_settings().boost_job_name or "").strip())


def kickoff_to_payload(kickoff: PipelineKickoff) -> dict[str, Any]:
    """Serialize a kickoff for the boost job (output_dir is NOT carried:
    the job uses its own CHALYBCLIP_DEFAULT_OUTPUT_DIR)."""
    return {
        "tenant_id": kickoff.tenant_id,
        "persona_id": kickoff.persona_id,
        "language": kickoff.language,
        "source_object_key": kickoff.source_object_key,
        "title": kickoff.title,
        "stream": kickoff.stream.model_dump(mode="json"),
        "usage_job_id": kickoff.usage_job_id,
        "reservation_id": kickoff.reservation_id,
        "lane": kickoff.lane,
    }


def kickoff_from_payload(payload: dict[str, Any], *, output_dir: Any) -> PipelineKickoff:
    from pathlib import Path

    from chalybclip.ingest import Stream

    return PipelineKickoff(
        tenant_id=str(payload["tenant_id"]),
        stream=Stream.model_validate(payload["stream"]),
        persona_id=str(payload["persona_id"]),
        output_dir=Path(output_dir),
        language=payload.get("language"),
        source_object_key=payload.get("source_object_key"),
        title=payload.get("title"),
        usage_job_id=payload.get("usage_job_id"),
        reservation_id=payload.get("reservation_id"),
        lane=str(payload.get("lane") or "boost"),
    )


async def _metadata(client: httpx.AsyncClient, path: str) -> str:
    resp = await client.get(f"{_METADATA}/{path}", headers=_METADATA_HEADERS)
    resp.raise_for_status()
    return resp.text.strip()


async def _access_token(client: httpx.AsyncClient) -> str:
    resp = await client.get(
        f"{_METADATA}/instance/service-accounts/default/token", headers=_METADATA_HEADERS
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise BoostDispatchError("metadata server returned no access_token")
    return str(token)


def boost_run_url(*, project: str, region: str, job_name: str) -> str:
    return f"{_RUN_API}/projects/{project}/locations/{region}/jobs/{job_name}:run"


def boost_run_body(job_id: str) -> dict[str, Any]:
    return {
        "overrides": {
            "containerOverrides": [
                {"env": [{"name": BOOST_JOB_ID_ENV, "value": job_id}]}
            ]
        }
    }


async def start_boost_execution(
    job_id: str, *, client: httpx.AsyncClient | None = None
) -> str:
    """Start one execution of the boost Cloud Run Job for `job_id`. Returns
    the long-running operation name. Raises BoostDispatchError."""
    settings = get_settings()
    job_name = (settings.boost_job_name or "").strip()
    if not job_name:
        raise BoostDispatchError("CHALYBCLIP_BOOST_JOB_NAME is not set")
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
    try:
        project = (settings.boost_gcp_project or "").strip() or await _metadata(
            http, "project/project-id"
        )
        region = (settings.boost_gcp_region or "").strip()
        if not region:
            # "projects/<number>/regions/<region>"
            region = (await _metadata(http, "instance/region")).rsplit("/", 1)[-1]
        token = await _access_token(http)
        url = boost_run_url(project=project, region=region, job_name=job_name)
        resp = await http.post(
            url,
            json=boost_run_body(job_id),
            headers={"Authorization": f"Bearer {token}"},
        )
    except BoostDispatchError:
        raise
    except Exception as e:  # noqa: BLE001 — metadata/network failures
        raise BoostDispatchError(f"{type(e).__name__}: {e}") from e
    finally:
        if own_client:
            await http.aclose()
    if resp.status_code >= 300:
        raise BoostDispatchError(
            f"run API HTTP {resp.status_code}: {(resp.text or '')[:300]}"
        )
    try:
        op_name = str(resp.json().get("name") or "")
    except Exception:  # noqa: BLE001
        op_name = ""
    _log.info("boost execution started · job=%s operation=%s", job_id, op_name)
    return op_name


def _boost_capable(kickoff: PipelineKickoff) -> bool:
    """The boost container shares nothing with this box but the DB and the
    bucket: it can fetch an http(s) VOD or a parked upload, nothing else."""
    vod_url = str(getattr(kickoff.stream, "vod_url", "") or "")
    return vod_url.startswith(("http://", "https://")) or bool(kickoff.source_object_key)


class BoostLaneDispatcher(JobDispatcher):
    """Wraps the configured dispatcher: boost-lane kickoffs go to a Cloud
    Run Job execution, everything else (and every fallback) to `inner`."""

    def __init__(self, inner: JobDispatcher, *, client: httpx.AsyncClient | None = None):
        self.inner = inner
        self._client = client

    @property
    def name(self) -> str:
        return f"{self.inner.name}+boost"

    def __getattr__(self, item: str) -> Any:
        # drain(), _fallback, … — anything callers reach for on the inner.
        return getattr(self.inner, item)

    async def dispatch_pipeline(
        self,
        kickoff: PipelineKickoff,
        *,
        background_tasks: BackgroundTasks | None = None,
    ) -> None:
        if kickoff.lane == "boost":
            if await self._try_boost(kickoff):
                return
            kickoff = await downgrade_to_standard(kickoff)
        await self.inner.dispatch_pipeline(kickoff, background_tasks=background_tasks)

    async def _try_boost(self, kickoff: PipelineKickoff) -> bool:
        stream_id = kickoff.stream.id
        if not boost_configured():
            _log.warning(
                "boost lane requested but CHALYBCLIP_BOOST_JOB_NAME is unset — "
                "running on the standard worker · stream=%s", stream_id,
            )
            return False
        if not kickoff.usage_job_id:
            _log.warning("boost lane without a usage job id — standard · stream=%s", stream_id)
            return False
        if not _boost_capable(kickoff):
            _log.warning(
                "boost lane can't reach this source (local file) — standard · stream=%s",
                stream_id,
            )
            return False
        from chalybclip.db import Database
        from chalybclip.db.usage_repos import UsageJobsRepo
        from chalybclip.settings import resolve_db_target

        try:
            db = Database(resolve_db_target(get_settings()))
            await db.connect()
            try:
                await UsageJobsRepo(db).set_kickoff(
                    kickoff.usage_job_id, kickoff_to_payload(kickoff)
                )
            finally:
                await db.close()
            await start_boost_execution(kickoff.usage_job_id, client=self._client)
        except Exception as e:  # noqa: BLE001 — degrade to the standard lane
            _log.error(
                "boost dispatch failed (%s) — running on the standard worker · stream=%s",
                e, stream_id,
            )
            return False
        return True


async def downgrade_to_standard(kickoff: PipelineKickoff) -> PipelineKickoff:
    """Re-admit with boost=false (drops the boost fee) and return a
    standard-lane kickoff. A refusal on the re-admit propagates."""
    from dataclasses import replace

    from chalybclip.db import Database
    from chalybclip.settings import resolve_db_target

    from .usage import readmit_job

    if kickoff.usage_job_id:
        db = Database(resolve_db_target(get_settings()))
        await db.connect()
        try:
            await readmit_job(db, job_id=kickoff.usage_job_id, boost=False)
        finally:
            await db.close()
    return replace(kickoff, lane="standard")

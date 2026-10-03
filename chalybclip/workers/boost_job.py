"""Boost-lane entrypoint — run exactly ONE admitted job, then exit.

    python -m chalybclip.workers.boost_job

This is the container command of the `chalybclip-boost` Cloud Run Job
(8 vCPU / 32 GiB, max_retries 0). Each execution is started by the web box
(`jobs.boost.start_boost_execution`) with a per-execution env override:

    CHALYBCLIP_BOOST_JOB_ID = <usage_jobs.id>   (the hub's external_job_id)

The job loads that row from the shared Postgres, rebuilds the kickoff the
web box persisted, and runs it through the SAME runner the shared worker
uses (`remote_upload_runner` for a parked upload, `default_pipeline_runner`
otherwise). Those runners meter compute.seconds at the boost rate, settle
the reservation and drain the usage outbox before returning, so when this
process exits nothing is left running or unreported.

Exit codes: 0 success · 1 the run failed · 2 bad invocation (no id, row
missing, no kickoff) — the reservation is settled `failed` in that case
too so it doesn't hold the user's tokens until its TTL.

Needs the worker env: DATABASE_URL, CHALYB_BASE_URL + CHALYB_ADMIN_TOKEN
(metering + settle), the object-storage vars, provider keys, and
CHALYBCLIP_DEFAULT_OUTPUT_DIR (scratch disk).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from chalybclip.jobs.boost import BOOST_JOB_ID_ENV

_log = logging.getLogger("chalybclip.workers.boost_job")

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_BAD_INVOCATION = 2


def _normalize_db_env() -> None:
    """Same as the PC worker: Settings binds database_url to the un-prefixed
    DATABASE_URL alias, so accept CHALYBCLIP_DATABASE_URL too."""
    if not (os.environ.get("DATABASE_URL") or "").strip():
        prefixed = (os.environ.get("CHALYBCLIP_DATABASE_URL") or "").strip()
        if prefixed:
            os.environ["DATABASE_URL"] = prefixed


async def run_boost_job(job_id: str, *, runner: object | None = None) -> int:
    """Load + run one job. `runner` is injectable for tests."""
    from pathlib import Path

    from chalybclip.db import Database
    from chalybclip.db.usage_repos import UsageJobsRepo
    from chalybclip.jobs.boost import kickoff_from_payload
    from chalybclip.jobs.usage import settle_job
    from chalybclip.settings import get_settings, resolve_db_target

    settings = get_settings()
    db = Database(resolve_db_target(settings))
    await db.connect()
    try:
        row = await UsageJobsRepo(db).get(job_id)
        if row is None or not row.kickoff:
            _log.error("boost job %s: no usage_jobs row / kickoff — nothing to run", job_id)
            if row is not None:
                await settle_job(
                    db, tenant_id=row.tenant_id, job_id=row.id,
                    reservation_id=row.reservation_id, outcome="failed",
                )
            return EXIT_BAD_INVOCATION
        await UsageJobsRepo(db).set_status(job_id, "running")
    finally:
        await db.close()

    output_dir = Path(settings.default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kickoff = kickoff_from_payload(row.kickoff, output_dir=output_dir)

    if runner is None:
        from chalybclip.api._pipeline import default_pipeline_runner, remote_upload_runner

        run = remote_upload_runner if kickoff.source_object_key else default_pipeline_runner
    else:
        run = runner  # type: ignore[assignment]
    _log.info(
        "boost job start · job=%s stream=%s tenant=%s",
        job_id, kickoff.stream.id, kickoff.tenant_id,
    )
    try:
        await run(kickoff)
    except Exception:
        # The runner already emitted pipeline.failed and settled `failed`.
        _log.exception("boost job failed · job=%s stream=%s", job_id, kickoff.stream.id)
        return EXIT_RUN_FAILED
    _log.info("boost job done · job=%s stream=%s", job_id, kickoff.stream.id)
    return EXIT_OK


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    job_id = (os.environ.get(BOOST_JOB_ID_ENV) or "").strip()
    if not job_id:
        _log.error("%s is not set — nothing to run", BOOST_JOB_ID_ENV)
        return EXIT_BAD_INVOCATION
    _normalize_db_env()
    return asyncio.run(run_boost_job(job_id))


if __name__ == "__main__":
    sys.exit(main())

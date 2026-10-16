"""Boost lane — the Cloud Run Jobs run request (URL, overrides body,
metadata-server token), the dispatcher's routing/fallback, and the
one-shot entrypoint running exactly the persisted job."""

from __future__ import annotations

import json as _json
from pathlib import Path

import httpx
import pytest
import respx

from chalybclip.db import Database, TenantsRepo, apply_migrations
from chalybclip.db.usage_repos import UsageJobsRepo
from chalybclip.ingest import Stream
from chalybclip.jobs import PipelineKickoff
from chalybclip.jobs import boost as boost_mod
from chalybclip.jobs.boost import (
    BoostLaneDispatcher,
    kickoff_from_payload,
    kickoff_to_payload,
    start_boost_execution,
)
from chalybclip.settings import get_settings

_META = "http://metadata.google.internal/computeMetadata/v1"
_TOKEN_URL = f"{_META}/instance/service-accounts/default/token"
_RUN_URL = (
    "https://run.googleapis.com/v2/projects/chalyb-prod/locations/us-central1/"
    "jobs/chalybclip-boost:run"
)


@pytest.fixture
def boost_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHALYBCLIP_BOOST_JOB_NAME", "chalybclip-boost")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "chalyb-prod")
    monkeypatch.setenv("CLOUD_RUN_REGION", "us-central1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _kickoff(tmp_path: Path, **kw: object) -> PipelineKickoff:
    stream = Stream(
        id="str_1", tenant_id="ten_a", vod_url="https://youtu.be/x", platform="youtube",
        duration_s=0.0, source_video_path=tmp_path / "v.mp4",
        source_audio_path=tmp_path / "a.wav",
    )
    base: dict[str, object] = dict(
        tenant_id="ten_a", stream=stream, persona_id="p1", output_dir=tmp_path,
        usage_job_id="ujob_1", reservation_id="res-1", lane="boost",
    )
    base.update(kw)
    return PipelineKickoff(**base)  # type: ignore[arg-type]


@respx.mock
async def test_run_request_shape(boost_env) -> None:
    token = respx.get(_TOKEN_URL).mock(return_value=httpx.Response(
        200, json={"access_token": "ya29.tok", "expires_in": 3599}
    ))
    run = respx.post(_RUN_URL).mock(return_value=httpx.Response(
        200, json={"name": "projects/chalyb-prod/locations/us-central1/operations/op1"}
    ))
    op = await start_boost_execution("ujob_42")
    assert op.endswith("/operations/op1")
    assert token.calls.last.request.headers["metadata-flavor"] == "Google"
    req = run.calls.last.request
    assert req.headers["authorization"] == "Bearer ya29.tok"
    assert _json.loads(req.content) == {
        "overrides": {
            "containerOverrides": [
                {"env": [{"name": "CHALYBCLIP_BOOST_JOB_ID", "value": "ujob_42"}]}
            ]
        }
    }


@respx.mock
async def test_project_and_region_fall_back_to_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHALYBCLIP_BOOST_JOB_NAME", "chalybclip-boost")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    monkeypatch.setenv("CLOUD_RUN_REGION", "")
    get_settings.cache_clear()
    try:
        respx.get(f"{_META}/project/project-id").mock(
            return_value=httpx.Response(200, text="chalyb-prod")
        )
        respx.get(f"{_META}/instance/region").mock(
            return_value=httpx.Response(200, text="projects/123/regions/us-central1")
        )
        respx.get(_TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": "t"}))
        run = respx.post(_RUN_URL).mock(return_value=httpx.Response(200, json={"name": "op"}))
        await start_boost_execution("ujob_1")
        assert run.called
    finally:
        get_settings.cache_clear()


@respx.mock
async def test_run_api_rejection_raises(boost_env) -> None:
    respx.get(_TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": "t"}))
    respx.post(_RUN_URL).mock(return_value=httpx.Response(403, text="denied"))
    with pytest.raises(boost_mod.BoostDispatchError, match="403"):
        await start_boost_execution("ujob_1")


class _Inner:
    name = "fake"

    def __init__(self) -> None:
        self.kickoffs: list[PipelineKickoff] = []

    async def dispatch_pipeline(self, kickoff, *, background_tasks=None) -> None:
        self.kickoffs.append(kickoff)


async def test_boost_kickoff_goes_to_cloud_run_and_persists_kickoff(
    tmp_path: Path, boost_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "t.db"
    db = Database(db_file)
    await apply_migrations(db)
    await TenantsRepo(db).create(tenant_id="ten_a", name="A")
    await UsageJobsRepo(db).upsert(
        job_id="ujob_1", tenant_id="ten_a", stream_id="str_1", operation="clips.pipeline",
        job_class="job", reservation_id="res-1", lane="boost", boost=True,
        upload_mb=0, source_minutes=0,
    )
    await db.close()
    monkeypatch.setattr("chalybclip.settings.resolve_db_target", lambda _s: str(db_file))
    started: list[str] = []

    async def fake_start(job_id: str, *, client=None) -> str:
        started.append(job_id)
        return "op"

    monkeypatch.setattr(boost_mod, "start_boost_execution", fake_start)
    inner = _Inner()
    await BoostLaneDispatcher(inner).dispatch_pipeline(_kickoff(tmp_path))
    assert started == ["ujob_1"]
    assert inner.kickoffs == []
    db = Database(db_file)
    row = await UsageJobsRepo(db).get("ujob_1")
    await db.close()
    assert row is not None and row.kickoff is not None
    rebuilt = kickoff_from_payload(row.kickoff, output_dir=tmp_path / "job")
    assert rebuilt.stream.id == "str_1" and rebuilt.reservation_id == "res-1"
    assert rebuilt.lane == "boost"


async def test_boost_unconfigured_falls_back_to_standard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHALYBCLIP_BOOST_JOB_NAME", "")
    get_settings.cache_clear()
    downgraded: list[str] = []

    async def fake_downgrade(kickoff: PipelineKickoff) -> PipelineKickoff:
        from dataclasses import replace

        downgraded.append(kickoff.usage_job_id or "")
        return replace(kickoff, lane="standard")

    monkeypatch.setattr(boost_mod, "downgrade_to_standard", fake_downgrade)
    try:
        inner = _Inner()
        await BoostLaneDispatcher(inner).dispatch_pipeline(_kickoff(tmp_path))
    finally:
        get_settings.cache_clear()
    # Re-admitted with boost=false (no fee) and run on the shared worker.
    assert downgraded == ["ujob_1"]
    assert [k.lane for k in inner.kickoffs] == ["standard"]


async def test_standard_kickoff_passes_straight_through(tmp_path: Path) -> None:
    inner = _Inner()
    k = _kickoff(tmp_path, lane="standard")
    await BoostLaneDispatcher(inner).dispatch_pipeline(k)
    assert inner.kickoffs == [k]


def test_local_upload_source_is_not_boost_capable(tmp_path: Path) -> None:
    stream = Stream(
        id="str_2", tenant_id="ten_a", vod_url="upload://clip.mp4", platform="upload",
        duration_s=0.0, source_video_path=tmp_path / "v.mp4",
        source_audio_path=tmp_path / "a.wav",
    )
    assert not boost_mod._boost_capable(_kickoff(tmp_path, stream=stream))
    assert boost_mod._boost_capable(
        _kickoff(tmp_path, stream=stream, source_object_key="uploads/x.mp4")
    )


async def test_boost_entrypoint_runs_exactly_the_persisted_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chalybclip.workers import boost_job

    db_file = tmp_path / "t.db"
    db = Database(db_file)
    await apply_migrations(db)
    await TenantsRepo(db).create(tenant_id="ten_a", name="A")
    await UsageJobsRepo(db).upsert(
        job_id="ujob_9", tenant_id="ten_a", stream_id="str_1", operation="clips.pipeline",
        job_class="job", reservation_id="res-9", lane="boost", boost=True,
        upload_mb=0, source_minutes=0,
    )
    await UsageJobsRepo(db).set_kickoff("ujob_9", kickoff_to_payload(_kickoff(tmp_path)))
    await db.close()
    monkeypatch.setattr("chalybclip.settings.resolve_db_target", lambda _s: str(db_file))
    monkeypatch.setenv("CHALYBCLIP_DEFAULT_OUTPUT_DIR", str(tmp_path / "out"))
    get_settings.cache_clear()
    ran: list[PipelineKickoff] = []

    async def runner(kickoff: PipelineKickoff) -> None:
        ran.append(kickoff)

    async def failing(kickoff: PipelineKickoff) -> None:
        raise RuntimeError("boom")

    try:
        assert await boost_job.run_boost_job("ujob_9", runner=runner) == boost_job.EXIT_OK
        assert [k.stream.id for k in ran] == ["str_1"]
        assert ran[0].output_dir == tmp_path / "out"
        assert await boost_job.run_boost_job("ujob_9", runner=failing) == boost_job.EXIT_RUN_FAILED
        assert await boost_job.run_boost_job("missing", runner=runner) == (
            boost_job.EXIT_BAD_INVOCATION
        )
    finally:
        get_settings.cache_clear()


def test_entrypoint_without_job_id_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    from chalybclip.workers import boost_job

    monkeypatch.delenv("CHALYBCLIP_BOOST_JOB_ID", raising=False)
    assert boost_job.main() == boost_job.EXIT_BAD_INVOCATION

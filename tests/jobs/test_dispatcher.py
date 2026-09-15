"""Unit tests for the F.8 JobDispatcher abstraction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from chalybclip.errors import ChalybClipError
from chalybclip.jobs import (
    InProcessJobDispatcher,
    ModalJobDispatcher,
    PipelineKickoff,
    get_dispatcher,
)
from chalybclip.settings import get_settings


@dataclass
class _FakeStream:
    """Duck-typed stand-in for chalybclip.ingest.Stream that PipelineKickoff
    only needs as an opaque attribute — the dispatcher never inspects it."""

    id: str = "str_TEST"
    vod_url: str = "upload://test.mp4"


@dataclass
class _RunnerProbe:
    """Captures everything the dispatcher passes to the runner."""

    calls: list[PipelineKickoff] = field(default_factory=list)

    async def __call__(self, kickoff: PipelineKickoff) -> None:
        self.calls.append(kickoff)


class _FakeBackgroundTasks:
    """Mimics FastAPI.BackgroundTasks.add_task — captures the coro
    instead of scheduling it so the test can inspect deferral."""

    def __init__(self) -> None:
        self.deferred: list[tuple[object, tuple[object, ...]]] = []

    def add_task(self, func: object, *args: object) -> None:
        self.deferred.append((func, args))


def _make_kickoff(stream_id: str = "str_TEST") -> PipelineKickoff:
    return PipelineKickoff(
        tenant_id="ten_TEST",
        stream=cast("object", _FakeStream(id=stream_id)),  # type: ignore[arg-type]
        persona_id="per_TEST",
        output_dir=Path("./out"),
        language="es",
    )


@pytest.fixture(autouse=True)
def _clean_active_registry() -> None:
    """The jobs.active registry is module-global; a test that dispatches
    without draining its deferred task would leak a registration into the
    next test and trip the in-flight dedupe."""
    from chalybclip.jobs import active

    active._active.clear()


# ---- InProcessJobDispatcher ----


def test_in_process_with_background_tasks_defers_call() -> None:
    """API path: when BackgroundTasks is provided, the dispatcher must
    NOT run the runner inline — it defers via add_task so the HTTP
    response goes out first."""
    probe = _RunnerProbe()
    bt = _FakeBackgroundTasks()
    dispatcher = InProcessJobDispatcher(probe)
    kickoff = _make_kickoff()

    asyncio.run(dispatcher.dispatch_pipeline(kickoff, background_tasks=bt))  # type: ignore[arg-type]

    assert len(probe.calls) == 0, "runner ran inline; should have been deferred"
    assert len(bt.deferred) == 1
    func, args = bt.deferred[0]
    # Deferred through the registry-releasing concurrency-guarded wrapper
    # (not the raw runner), but the kickoff is threaded through and running
    # it invokes the runner.
    assert func == dispatcher._run_registered
    assert args == (kickoff,)
    asyncio.run(func(*args))
    assert probe.calls == [kickoff]


def test_in_process_without_background_tasks_runs_inline() -> None:
    """CLI / test path: without BackgroundTasks the dispatcher awaits
    the runner directly so callers can synchronously verify the run."""
    probe = _RunnerProbe()
    dispatcher = InProcessJobDispatcher(probe)
    kickoff = _make_kickoff()

    asyncio.run(dispatcher.dispatch_pipeline(kickoff, background_tasks=None))

    assert probe.calls == [kickoff]


def test_in_process_name_is_stable() -> None:
    """Logs + telemetry depend on a stable name."""
    assert InProcessJobDispatcher(lambda _: asyncio.sleep(0)).name == "in-process"


# ---- ModalJobDispatcher (construction contract; behavior in
# ---- test_modal_dispatcher.py) ----


def test_modal_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing without endpoint/token must fail loudly with an
    actionable message — `create_app`'s defensive boot catches this and
    falls back to in-process, so a half-configured deploy still serves."""
    monkeypatch.delenv("CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("CHALYBCLIP_MODAL_TOKEN", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ChalybClipError, match="misconfigured"):
        ModalJobDispatcher()


def test_modal_name_is_stable() -> None:
    d = ModalJobDispatcher(
        endpoint_url="https://modal.test", bearer_token="bear"
    )
    assert d.name == "modal"


# ---- factory.get_dispatcher ----


def test_factory_defaults_to_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env override → in-process dispatcher."""
    monkeypatch.delenv("CHALYBCLIP_JOB_DISPATCHER", raising=False)
    get_settings.cache_clear()
    probe = _RunnerProbe()
    d = get_dispatcher(runner=probe)
    assert isinstance(d, InProcessJobDispatcher)


def test_factory_picks_modal_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHALYBCLIP_JOB_DISPATCHER=modal (+ endpoint/token) → ModalJobDispatcher,
    with the in-process fallback wired when a runner is available."""
    monkeypatch.setenv("CHALYBCLIP_JOB_DISPATCHER", "modal")
    monkeypatch.setenv(
        "CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL", "https://modal.test/pipeline"
    )
    monkeypatch.setenv("CHALYBCLIP_MODAL_TOKEN", "bear")
    get_settings.cache_clear()
    try:
        d = get_dispatcher(runner=_RunnerProbe())
        assert isinstance(d, ModalJobDispatcher)
        assert isinstance(d._fallback, InProcessJobDispatcher)
        # No runner → no fallback, but the dispatcher still constructs.
        d2 = get_dispatcher(runner=None)
        assert isinstance(d2, ModalJobDispatcher)
        assert d2._fallback is None
    finally:
        get_settings.cache_clear()


def test_factory_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHALYBCLIP_JOB_DISPATCHER", "kubernetes")
    get_settings.cache_clear()
    with pytest.raises(ChalybClipError, match="unknown job_dispatcher"):
        get_dispatcher(runner=None)


def test_factory_requires_runner_for_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process needs an actual runner to wrap; calling without one
    is a programmer error and should error early, not at dispatch time."""
    monkeypatch.delenv("CHALYBCLIP_JOB_DISPATCHER", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ChalybClipError, match="runner"):
        get_dispatcher(runner=None)


# ---- Global concurrency cap ----


@pytest.mark.asyncio
async def test_dispatch_bounds_concurrency_to_cap() -> None:
    """At most `max_concurrency` runners execute heavy work at once; the
    rest queue on the semaphore and run as slots free."""
    live = 0
    peak = 0

    async def _slow_runner(_kickoff: PipelineKickoff) -> None:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            await asyncio.sleep(0.05)
        finally:
            live -= 1

    disp = InProcessJobDispatcher(_slow_runner, max_concurrency=2)
    # Fire 8 launches concurrently (recovery/CLI path → inline await, wrapped
    # in tasks just like the recovery sweep does). Distinct stream ids —
    # same-stream launches are deduped, which is its own test below.
    await asyncio.gather(
        *(
            asyncio.create_task(disp.dispatch_pipeline(_make_kickoff(f"str_{i}")))
            for i in range(8)
        )
    )
    assert peak <= 2, f"peak concurrency {peak} exceeded cap of 2"
    assert peak >= 1


@pytest.mark.asyncio
async def test_dispatch_runs_all_launches_despite_cap() -> None:
    """The cap throttles, it never drops work — all 8 still run."""
    ran = 0

    async def _runner(_kickoff: PipelineKickoff) -> None:
        nonlocal ran
        ran += 1
        await asyncio.sleep(0)

    disp = InProcessJobDispatcher(_runner, max_concurrency=1)
    await asyncio.gather(
        *(
            asyncio.create_task(disp.dispatch_pipeline(_make_kickoff(f"str_{i}")))
            for i in range(8)
        )
    )
    assert ran == 8


# ---- Same-stream in-flight dedupe ----


@pytest.mark.asyncio
async def test_dispatch_dedupes_same_stream_while_in_flight() -> None:
    """A second launch for a stream that is queued or running must be
    skipped — two concurrent runs of the same stream write over each
    other's source/*.part files (the prod fragment-corruption incident)."""
    started = asyncio.Event()
    release = asyncio.Event()
    ran = 0

    async def _runner(_kickoff: PipelineKickoff) -> None:
        nonlocal ran
        ran += 1
        started.set()
        await release.wait()

    disp = InProcessJobDispatcher(_runner, max_concurrency=2)
    first = asyncio.create_task(disp.dispatch_pipeline(_make_kickoff("str_dup")))
    await started.wait()

    # Duplicate launches while the first is mid-run: all skipped, even
    # though a semaphore slot is free.
    await disp.dispatch_pipeline(_make_kickoff("str_dup"))
    await disp.dispatch_pipeline(_make_kickoff("str_dup"))
    release.set()
    await first
    assert ran == 1

    # Registration released after the run — a genuine re-run goes through.
    release.set()
    await disp.dispatch_pipeline(_make_kickoff("str_dup"))
    assert ran == 2


@pytest.mark.asyncio
async def test_dispatch_dedupes_stream_queued_on_semaphore() -> None:
    """The dedupe must also cover a launch still WAITING for a slot — a
    queued run is in flight, not lost (the recovery sweeper was stacking
    duplicates of exactly these)."""
    release = asyncio.Event()
    ran: list[str] = []

    async def _runner(kickoff: PipelineKickoff) -> None:
        ran.append(kickoff.stream.id)
        await release.wait()

    disp = InProcessJobDispatcher(_runner, max_concurrency=1)
    hog = asyncio.create_task(disp.dispatch_pipeline(_make_kickoff("str_hog")))
    await asyncio.sleep(0)  # hog takes the only slot
    queued = asyncio.create_task(disp.dispatch_pipeline(_make_kickoff("str_q")))
    await asyncio.sleep(0)  # str_q now parked on the semaphore

    await disp.dispatch_pipeline(_make_kickoff("str_q"))  # duplicate: skipped
    release.set()
    await asyncio.gather(hog, queued)
    assert ran == ["str_hog", "str_q"]

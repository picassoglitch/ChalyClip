"""Token T2 — after a successful pipeline run, the balance is pulled
live so the chip reflects the run's spend without a manual click.

Pins: the refresh fires on success, is SKIPPED on failure (the run
re-raises before it), and the helper actually calls fetch_balance_now.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chalybclip.api import _pipeline
from chalybclip.db import Database, apply_migrations


def _kickoff(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id="ten_1",
        output_dir=tmp_path,
        persona_id="p1",
        language=None,
        stream=SimpleNamespace(id="str_1", vod_url="https://x/v"),
    )


def _stub_settings(tmp_path: Path) -> None:
    return SimpleNamespace(db_path=str(tmp_path / "x.db"))


async def test_default_runner_refreshes_balance_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_process_vod(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "chalybclip.pipeline.process_vod", fake_process_vod, raising=False,
    )
    monkeypatch.setattr(
        "chalybclip.settings.get_settings", lambda: _stub_settings(tmp_path),
    )
    called: list[dict] = []

    async def fake_refresh(**kwargs: object) -> None:
        called.append(dict(kwargs))

    monkeypatch.setattr(_pipeline, "_refresh_balance_after_run", fake_refresh)

    await _pipeline.default_pipeline_runner(_kickoff(tmp_path))

    assert len(called) == 1
    assert called[0]["tenant_id"] == "ten_1"


async def test_default_runner_skips_refresh_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def boom(**kwargs: object) -> None:
        raise RuntimeError("pipeline blew up")

    monkeypatch.setattr(
        "chalybclip.pipeline.process_vod", boom, raising=False,
    )
    monkeypatch.setattr(
        "chalybclip.settings.get_settings", lambda: _stub_settings(tmp_path),
    )

    # Don't hit a real DB on the failure-event path.
    async def noop_failure(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(_pipeline, "_emit_top_level_failure", noop_failure)

    called: list[dict] = []

    async def fake_refresh(**kwargs: object) -> None:
        called.append(dict(kwargs))

    monkeypatch.setattr(_pipeline, "_refresh_balance_after_run", fake_refresh)

    with pytest.raises(RuntimeError, match="blew up"):
        await _pipeline.default_pipeline_runner(_kickoff(tmp_path))

    # Failure re-raises before the refresh — chip isn't refreshed on a
    # run that didn't complete.
    assert called == []


async def test_refresh_helper_calls_fetch_balance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The helper pulls the live balance (one fetch) after the grace
    delay, swallowing errors."""
    # No 4s wait in tests.
    monkeypatch.setattr(_pipeline, "_BALANCE_REFRESH_GRACE_S", 0.0)

    db = Database(tmp_path / "real.db")
    await apply_migrations(db)
    await db.close()
    monkeypatch.setattr(
        "chalybclip.settings.get_settings",
        lambda: SimpleNamespace(db_path=str(tmp_path / "real.db")),
    )

    fetched: list[str] = []

    async def fake_fetch(db_arg: object, *, tenant_id: str) -> bool:
        fetched.append(tenant_id)
        return True

    monkeypatch.setattr(
        "chalybclip.integrations.chalyb.balance.fetch_balance_now", fake_fetch,
    )

    await _pipeline._refresh_balance_after_run(
        db_path=str(tmp_path / "real.db"), tenant_id="ten_z",
    )
    assert fetched == ["ten_z"]


async def test_base_fee_reported_with_configured_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Token T3 — a completed run reports a per-run base charge
    (engine.base) at the configured cost, idempotent per stream."""
    from chalybclip.settings import get_settings

    monkeypatch.setenv("CHALYBCLIP_PIPELINE_BASE_CHARGE_USD_MICROS", "50000")
    get_settings.cache_clear()

    db = Database(tmp_path / "b.db")
    await apply_migrations(db)

    calls: list[dict] = []

    async def fake_report_usage(db_arg: object, **kwargs: object) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(
        "chalybclip.integrations.chalyb.reporter.report_usage",
        fake_report_usage,
    )
    try:
        await _pipeline._charge_run_base_fee(db, "ten_1", "str_xyz")
    finally:
        await db.close()
        get_settings.cache_clear()

    assert len(calls) == 1
    c = calls[0]
    assert c["kind"] == "engine.base"
    assert c["cost_usd_micros"] == 50_000
    assert c["amount"] == 1
    assert c["provider"] == "chalybclip"
    assert c["source_id"] == "base_str_xyz"   # idempotent per stream
    assert c["operation"] == "pipeline_run"


async def test_base_fee_skipped_when_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Setting the base charge to 0 disables it — no event reported."""
    from chalybclip.settings import get_settings

    monkeypatch.setenv("CHALYBCLIP_PIPELINE_BASE_CHARGE_USD_MICROS", "0")
    get_settings.cache_clear()

    db = Database(tmp_path / "b.db")
    await apply_migrations(db)

    calls: list[dict] = []

    async def fake_report_usage(db_arg: object, **kwargs: object) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(
        "chalybclip.integrations.chalyb.reporter.report_usage",
        fake_report_usage,
    )
    try:
        await _pipeline._charge_run_base_fee(db, "ten_1", "str_xyz")
    finally:
        await db.close()
        get_settings.cache_clear()

    assert calls == []


async def test_refresh_helper_swallows_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fetch error must never propagate — the run already succeeded."""
    monkeypatch.setattr(_pipeline, "_BALANCE_REFRESH_GRACE_S", 0.0)

    db = Database(tmp_path / "real.db")
    await apply_migrations(db)
    await db.close()

    async def boom_fetch(db_arg: object, *, tenant_id: str) -> bool:
        raise RuntimeError("chalyb down")

    monkeypatch.setattr(
        "chalybclip.integrations.chalyb.balance.fetch_balance_now", boom_fetch,
    )

    # Must NOT raise.
    await _pipeline._refresh_balance_after_run(
        db_path=str(tmp_path / "real.db"), tenant_id="ten_z",
    )

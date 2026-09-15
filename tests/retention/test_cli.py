"""CLI smoke tests for `chalybclip retention sweep` + `tenants set-retention`."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from typer.testing import CliRunner

from chalybclip.cli import app
from chalybclip.db import (
    Database,
    StreamsRepo,
    TenantsRepo,
    apply_migrations,
)
from chalybclip.db.models import StreamRow
from chalybclip.tenancy import bound_tenant


def _seed_aged_stream(db_path: Path, *, tenant_id: str, days_old: int) -> Path:
    """Sync helper that creates a tenant + an aged stream + an on-disk
    source video. Returns the per-stream output dir."""
    import asyncio

    out_dir = db_path.parent / "out"
    stream_dir = out_dir / "str_seed"
    stream_dir.mkdir(parents=True, exist_ok=True)
    video = stream_dir / "source" / "video.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"\x00" * 256)
    audio = stream_dir / "source" / "audio.wav"
    audio.write_bytes(b"\x00" * 128)

    async def _seed() -> None:
        db = Database(db_path)
        try:
            await apply_migrations(db)
            await TenantsRepo(db).create(tenant_id=tenant_id, name="seed")
            with bound_tenant(tenant_id):
                created = (
                    _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=days_old)
                ).isoformat()
                await StreamsRepo(db).upsert(
                    StreamRow(
                        id="str_seed",
                        tenant_id=tenant_id,
                        vod_url="x",
                        platform="kick",
                        title=None,
                        channel=None,
                        duration_s=60.0,
                        source_video_path=str(video),
                        source_audio_path=str(audio),
                        status="ingested",
                        created_at=created,
                    )
                )
        finally:
            await db.close()

    asyncio.run(_seed())
    return out_dir


def test_retention_sweep_help(tmp_path: Path) -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["retention", "sweep", "--help"])
    assert r.exit_code == 0
    assert "sweep" in r.output.lower()


def test_retention_sweep_dry_run_reports_vod(tmp_path: Path) -> None:
    """`--dry-run` reports the VOD that would be deleted but doesn't act."""
    db_path = tmp_path / "x.db"
    out_dir = _seed_aged_stream(db_path, tenant_id="aldo", days_old=45)
    stream_dir = out_dir / "str_seed"
    assert stream_dir.exists()

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "retention",
            "sweep",
            "--db-path",
            str(db_path),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "DRY RUN" in r.output
    assert "vods=1" in r.output
    # Files still on disk.
    assert stream_dir.exists()


def test_retention_sweep_real_run_deletes(tmp_path: Path) -> None:
    db_path = tmp_path / "x.db"
    out_dir = _seed_aged_stream(db_path, tenant_id="aldo", days_old=45)
    stream_dir = out_dir / "str_seed"

    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "retention",
            "sweep",
            "--db-path",
            str(db_path),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "swept" in r.output
    assert not stream_dir.exists()


def test_retention_sweep_json_output(tmp_path: Path) -> None:
    """`--json` emits one JSON object per tenant."""
    import json

    db_path = tmp_path / "x.db"
    out_dir = _seed_aged_stream(db_path, tenant_id="aldo", days_old=45)
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "retention",
            "sweep",
            "--db-path",
            str(db_path),
            "--output-dir",
            str(out_dir),
            "--dry-run",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    line = r.output.strip().splitlines()[0]
    parsed = json.loads(line)
    assert parsed["tenant_id"] == "aldo"
    assert parsed["dry_run"] is True
    assert parsed["vods_deleted"] == 1


def test_tenants_set_retention_persists_overrides(tmp_path: Path) -> None:
    db_path = tmp_path / "x.db"
    runner = CliRunner()
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r = runner.invoke(
        app,
        [
            "tenants",
            "set-retention",
            "aldo",
            "--vod-days",
            "7",
            "--clip-days",
            "30",
            "--db-path",
            str(db_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "vod_days=7" in r.output
    assert "clip_days=30" in r.output
    # transcript_days untouched -> still NULL -> rendered as default(365).
    assert "transcript_days=default(365)" in r.output


def test_tenants_set_retention_zero_clears_to_default(tmp_path: Path) -> None:
    """vod_days=0 means 'go back to the system default', NOT '0 days'."""
    db_path = tmp_path / "x.db"
    runner = CliRunner()
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    runner.invoke(
        app,
        [
            "tenants",
            "set-retention",
            "aldo",
            "--vod-days",
            "5",
            "--db-path",
            str(db_path),
        ],
    )
    r = runner.invoke(
        app,
        [
            "tenants",
            "set-retention",
            "aldo",
            "--vod-days",
            "0",
            "--db-path",
            str(db_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "vod_days=default(7)" in r.output


def test_tenants_set_retention_no_args_exits_2(tmp_path: Path) -> None:
    db_path = tmp_path / "x.db"
    runner = CliRunner()
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r = runner.invoke(
        app, ["tenants", "set-retention", "aldo", "--db-path", str(db_path)]
    )
    assert r.exit_code == 2
    assert "nothing to update" in (r.stderr or r.output)

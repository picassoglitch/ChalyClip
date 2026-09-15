"""CLI smoke tests for `chalybclip drive {add,list,poll}` (slice E.4)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from chalybclip.cli import app


def _put_video(folder: Path, name: str, size: int = 64) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"\x00" * size)


def test_drive_add_then_list(tmp_path: Path) -> None:
    db_path = tmp_path / "x.db"
    runner = CliRunner()
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r1 = runner.invoke(
        app,
        [
            "drive",
            "add",
            "--tenant",
            "aldo",
            "folder_xyz",
            "--folder-name",
            "ChalyClip Inbox",
            "--refresh-token",
            "rt",
            "--db-path",
            str(db_path),
        ],
    )
    assert r1.exit_code == 0, r1.output
    assert "drv_" in r1.output

    r2 = runner.invoke(
        app, ["drive", "list", "--tenant", "aldo", "--db-path", str(db_path)]
    )
    assert r2.exit_code == 0, r2.output
    assert "folder_xyz" in r2.output
    assert "ChalyClip Inbox" in r2.output
    assert "enabled" in r2.output


def test_drive_list_unknown_tenant_exits_1(tmp_path: Path) -> None:
    db_path = tmp_path / "x.db"
    runner = CliRunner()
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    r = runner.invoke(
        app, ["drive", "list", "--tenant", "ten_nope", "--db-path", str(db_path)]
    )
    assert r.exit_code == 1
    assert "unknown tenant" in (r.stderr or r.output)


def test_drive_poll_with_source_dir_ingests(tmp_path: Path) -> None:
    """End-to-end: CLI polls a fake folder, fires the stub ingest
    callback for each new file, and the report row reflects what landed."""
    db_path = tmp_path / "x.db"
    out_dir = tmp_path / "out"
    src = tmp_path / "drop"
    _put_video(src, "stream_a.mp4")

    runner = CliRunner()
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    runner.invoke(
        app,
        [
            "drive",
            "add",
            "--tenant",
            "aldo",
            "folder_x",
            "--refresh-token",
            "dev",
            "--db-path",
            str(db_path),
        ],
    )
    r = runner.invoke(
        app,
        [
            "drive",
            "poll",
            "--tenant",
            "aldo",
            "--source-dir",
            str(src),
            "--output-dir",
            str(out_dir),
            "--db-path",
            str(db_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "ingested=1" in r.output
    # Stub callback echoed the per-file ingest line.
    assert "stream_a.mp4" in r.output


def test_drive_poll_without_source_dir_exits_1(tmp_path: Path) -> None:
    """Real GoogleDriveClient is deferred — poll without --source-dir
    fails loudly rather than silently doing nothing."""
    db_path = tmp_path / "x.db"
    runner = CliRunner()
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    r = runner.invoke(app, ["drive", "poll", "--db-path", str(db_path)])
    assert r.exit_code == 1
    assert "--source-dir is required" in (r.stderr or r.output)


def test_drive_poll_json_output(tmp_path: Path) -> None:
    import json

    db_path = tmp_path / "x.db"
    out_dir = tmp_path / "out"
    src = tmp_path / "drop"
    _put_video(src, "x.mp4")

    runner = CliRunner()
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    runner.invoke(
        app,
        [
            "drive",
            "add",
            "--tenant",
            "aldo",
            "folder_x",
            "--refresh-token",
            "dev",
            "--db-path",
            str(db_path),
        ],
    )
    r = runner.invoke(
        app,
        [
            "drive",
            "poll",
            "--source-dir",
            str(src),
            "--output-dir",
            str(out_dir),
            "--db-path",
            str(db_path),
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    # JSON-mode output: one object per line. The stub-ingest echo also
    # writes to stdout, so we filter.
    json_lines = [
        line for line in r.output.splitlines() if line.startswith("{")
    ]
    assert len(json_lines) == 1
    parsed = json.loads(json_lines[0])
    assert parsed["tenant_id"] == "aldo"
    assert parsed["files_ingested"] == 1

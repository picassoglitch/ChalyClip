"""CLI smoke tests for the Phase 1 admin commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from chalybclip.cli import app


def test_db_init_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["db", "init", "--help"])
    assert result.exit_code == 0


def test_db_init_brings_schema_to_current_version(tmp_path: Path) -> None:
    """`db init` runs every migration; current head is version 58
    (058_rebrand_brand_kit_columns)."""
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    result = runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "schema_version = 58" in result.output


def test_tenants_add_and_list(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    r1 = runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo Villanueva", "--db-path", str(db_path)]
    )
    assert r1.exit_code == 0, r1.output
    assert "created tenant: aldo" in r1.output

    r2 = runner.invoke(app, ["tenants", "list", "--db-path", str(db_path)])
    assert r2.exit_code == 0
    assert "aldo" in r2.output


def test_tokens_issue_then_list(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r1 = runner.invoke(
        app, ["tokens", "issue", "--tenant", "aldo", "--db-path", str(db_path)]
    )
    assert r1.exit_code == 0, r1.output
    raw_token = r1.output.strip().splitlines()[0]
    assert raw_token.startswith("tok_")

    r2 = runner.invoke(
        app, ["tokens", "list", "--tenant", "aldo", "--db-path", str(db_path)]
    )
    assert r2.exit_code == 0
    assert "scope=full" in r2.output
    # Raw token must NOT appear in the list output.
    assert raw_token not in r2.output


def test_tenants_set_budget_sets_dollars_micros(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r = runner.invoke(
        app,
        [
            "tenants",
            "set-budget",
            "aldo",
            "--daily-usd",
            "5.00",
            "--publish-limit",
            "50",
            "--rescore-cap",
            "8",
            "--db-path",
            str(db_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "$5.00" in r.output
    assert "publish/d=50" in r.output
    assert "rescore_cap=8" in r.output

    # `tenants list` reflects the new values.
    r2 = runner.invoke(app, ["tenants", "list", "--db-path", str(db_path)])
    assert "$5.00" in r2.output


def test_tenants_set_budget_zero_clears_to_unlimited(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    runner.invoke(
        app,
        ["tenants", "set-budget", "aldo", "--daily-usd", "5.00", "--db-path", str(db_path)],
    )
    # Now clear it.
    r = runner.invoke(
        app,
        ["tenants", "set-budget", "aldo", "--daily-usd", "0", "--db-path", str(db_path)],
    )
    assert r.exit_code == 0, r.output
    assert "budget=unlimited" in r.output


def test_tenants_set_budget_no_args_exits_2(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r = runner.invoke(
        app, ["tenants", "set-budget", "aldo", "--db-path", str(db_path)]
    )
    assert r.exit_code == 2
    assert "nothing to update" in (r.stderr or r.output)


def test_mcp_serve_help_is_documented(tmp_path: Path) -> None:
    """`chalybclip mcp serve --help` shows usage so agents can discover the command."""
    runner = CliRunner()
    result = runner.invoke(app, ["mcp", "serve", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.output or "stdio" in result.output


def test_mcp_serve_without_token_exits_1(tmp_path: Path) -> None:
    """No `--token` and no env var -> the server fails fast at boot."""
    import os

    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    # Make sure the env var is empty for this invocation.
    env = {k: v for k, v in os.environ.items() if k != "CHALYBCLIP_API_TOKEN"}
    env["CHALYBCLIP_API_TOKEN"] = ""
    result = runner.invoke(
        app, ["mcp", "serve", "--db-path", str(db_path)], env=env
    )
    assert result.exit_code == 1
    assert "no API token" in (result.stderr or result.output) or "CHALYBCLIP_API_TOKEN" in (
        result.stderr or result.output
    )


def test_tokens_issue_invalid_scope_exits_2(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r = runner.invoke(
        app,
        [
            "tokens",
            "issue",
            "--tenant",
            "aldo",
            "--scope",
            "admin",
            "--db-path",
            str(db_path),
        ],
    )
    assert r.exit_code == 2
    assert "must be 'full' or 'read'" in (r.stderr or r.output)

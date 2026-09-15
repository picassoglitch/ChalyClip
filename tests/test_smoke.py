"""Smoke test — proves the package installs and imports cleanly."""

def test_import() -> None:
    import chalybclip

    assert chalybclip.__version__


def test_cli_help_does_not_crash() -> None:
    from typer.testing import CliRunner

    from chalybclip.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

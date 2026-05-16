from __future__ import annotations

from typer.testing import CliRunner

from mystery import __version__
from mystery.cli import app

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("new", "play", "eval", "version"):
        assert cmd in result.stdout

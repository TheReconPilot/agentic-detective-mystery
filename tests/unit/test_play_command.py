from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from typer.testing import CliRunner

from mystery import cli as cli_module
from mystery.cli import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from mystery.models import CaseBible

runner = CliRunner()


def _setup(
    bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    chat_responses: list[str],
) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "42.json").write_text(bible.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("MYSTERY_CASES_DIR", str(cases_dir))
    monkeypatch.setattr(
        cli_module,
        "_default_embeddings_factory",
        lambda _settings: DeterministicFakeEmbedding(size=16),
    )
    monkeypatch.setattr(
        cli_module,
        "_default_chat_model_factory",
        lambda _settings: FakeListChatModel(responses=chat_responses),
    )


def test_play_winning_path_through_repl(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(valid_bible, tmp_path, monkeypatch, chat_responses=["I was nowhere near."])
    script = (
        "\n".join(
            [
                "examine",
                "move hallway",
                "examine",
                "notes",
                "ask butler alibi",
                "accuse butler",
            ],
        )
        + "\n"
    )

    result = runner.invoke(app, ["play", "--seed", "42"], input=script)

    assert result.exit_code == 0, result.output
    assert "case is solved" in result.output
    assert "Turns used:" in result.output


def test_play_help_command_lists_actions(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(valid_bible, tmp_path, monkeypatch, chat_responses=["unused"])
    result = runner.invoke(app, ["play", "--seed", "42"], input="help\naccuse butler\n")
    assert result.exit_code == 0, result.output
    assert "move <location>" in result.output
    assert "ask <suspect>" in result.output


def test_play_handles_bad_input_then_continues(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(valid_bible, tmp_path, monkeypatch, chat_responses=["unused"])
    result = runner.invoke(app, ["play", "--seed", "42"], input="teleport library\naccuse butler\n")
    assert result.exit_code == 0, result.output
    assert "Unknown command" in result.output
    assert "case is solved" in result.output


def test_play_eof_abandons_case(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup(valid_bible, tmp_path, monkeypatch, chat_responses=["unused"])
    result = runner.invoke(app, ["play", "--seed", "42"], input="")
    assert result.exit_code == 0
    assert "abandon" in result.output


def test_play_errors_on_missing_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYSTERY_CASES_DIR", str(tmp_path / "cases"))
    result = runner.invoke(app, ["play", "--seed", "99"])
    assert result.exit_code != 0
    assert "No case found" in result.output

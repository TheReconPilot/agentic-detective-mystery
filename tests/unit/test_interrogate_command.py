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


def _write_case(bible: CaseBible, cases_dir: Path, seed: int) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / f"{seed}.json").write_text(bible.model_dump_json(), encoding="utf-8")


def _stub_factories(monkeypatch: pytest.MonkeyPatch, *, response: str) -> None:
    monkeypatch.setattr(
        cli_module,
        "_default_embeddings_factory",
        lambda _settings: DeterministicFakeEmbedding(size=16),
    )
    monkeypatch.setattr(
        cli_module,
        "_default_chat_model_factory",
        lambda _settings: FakeListChatModel(responses=[response]),
    )


def test_interrogate_prints_suspect_response(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_dir = tmp_path / "cases"
    monkeypatch.setenv("MYSTERY_CASES_DIR", str(cases_dir))
    _write_case(valid_bible, cases_dir, seed=42)
    _stub_factories(monkeypatch, response="I was in the pantry, sir.")

    result = runner.invoke(
        app,
        ["interrogate", "--seed", "42", "--suspect", "butler", "where were you?"],
    )

    assert result.exit_code == 0, result.output
    assert "I was in the pantry, sir." in result.output
    assert "Hodges" in result.output  # the butler's display name


def test_interrogate_errors_on_missing_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYSTERY_CASES_DIR", str(tmp_path / "cases"))
    result = runner.invoke(
        app,
        ["interrogate", "--seed", "99", "--suspect", "butler", "anything"],
    )
    assert result.exit_code != 0
    assert "No case found" in result.output


def test_interrogate_errors_on_unknown_suspect(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_dir = tmp_path / "cases"
    monkeypatch.setenv("MYSTERY_CASES_DIR", str(cases_dir))
    _write_case(valid_bible, cases_dir, seed=42)
    _stub_factories(monkeypatch, response="unused")

    result = runner.invoke(
        app,
        ["interrogate", "--seed", "42", "--suspect", "ghost", "anything"],
    )
    assert result.exit_code != 0
    assert "Unknown suspect" in result.output

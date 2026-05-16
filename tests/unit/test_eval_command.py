"""CLI tests for `mystery eval`: solvability and optional consistency reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from typer.testing import CliRunner

from mystery import cli as cli_module
from mystery.cli import app
from mystery.evals.consistency import JudgeRuling

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from mystery.models import CaseBible

runner = CliRunner()


def _write_cases(bibles: list[CaseBible], cases_dir: Path) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    for bible in bibles:
        (cases_dir / f"{bible.seed}.json").write_text(
            bible.model_dump_json(),
            encoding="utf-8",
        )


class _AlwaysConsistentJudge:
    def judge(
        self,
        bible: CaseBible,
        suspect_id: str,
        question: str,
        response: str,
    ) -> JudgeRuling:
        del bible, suspect_id, question, response
        return JudgeRuling(verdict="consistent", reasoning="(test stub)")


def _stub_eval_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "_default_embeddings_factory",
        lambda _settings: DeterministicFakeEmbedding(size=16),
    )
    monkeypatch.setattr(
        cli_module,
        "_default_chat_model_factory",
        lambda _settings: FakeListChatModel(responses=[f"reply-{i}" for i in range(200)]),
    )
    monkeypatch.setattr(
        cli_module,
        "_default_judge_factory",
        lambda _settings: _AlwaysConsistentJudge(),
    )


def test_eval_reports_solvability_on_bundled_cases(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_dir = tmp_path / "evals" / "cases"
    bibles = [
        valid_bible.model_copy(update={"seed": 1}),
        valid_bible.model_copy(update={"seed": 2}),
    ]
    _write_cases(bibles, cases_dir)
    _stub_eval_factories(monkeypatch)

    result = runner.invoke(app, ["eval", "--cases-dir", str(cases_dir)])

    assert result.exit_code == 0, result.output
    assert "2/2" in result.output  # 2 successes
    assert "100%" in result.output


def test_eval_errors_when_cases_dir_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_eval_factories(monkeypatch)
    result = runner.invoke(app, ["eval", "--cases-dir", str(tmp_path / "missing")])

    assert result.exit_code != 0
    assert "No case JSON files found" in result.output


def test_eval_consistency_flag_runs_the_judge(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_dir = tmp_path / "evals" / "cases"
    _write_cases([valid_bible], cases_dir)
    _stub_eval_factories(monkeypatch)

    result = runner.invoke(
        app,
        ["eval", "--cases-dir", str(cases_dir), "--consistency"],
    )

    assert result.exit_code == 0, result.output
    assert "consistency eval" in result.output
    assert "overall:" in result.output

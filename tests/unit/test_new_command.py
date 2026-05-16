from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from mystery import cli as cli_module
from mystery.cli import app
from mystery.models import CaseBible

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def test_new_writes_bible_to_cases_dir(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_dir = tmp_path / "cases"
    monkeypatch.setenv("MYSTERY_CASES_DIR", str(cases_dir))

    class _StubLLM:
        def generate_bible(self, system: str, user: str) -> CaseBible:
            del system, user
            return valid_bible

    monkeypatch.setattr(cli_module, "_default_llm_factory", lambda _seed, _settings: _StubLLM())

    result = runner.invoke(app, ["new", "--seed", "42"])
    assert result.exit_code == 0, result.output

    out_path = cases_dir / "42.json"
    assert out_path.exists()

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert CaseBible.model_validate(written) == valid_bible


def test_new_surfaces_generation_failure(
    valid_bible: CaseBible,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYSTERY_CASES_DIR", str(tmp_path / "cases"))
    monkeypatch.setenv("MYSTERY_MAX_GEN_ATTEMPTS", "1")

    broken = valid_bible.model_copy(update={"killer_id": "ghost"})

    class _BadLLM:
        def generate_bible(self, system: str, user: str) -> CaseBible:
            del system, user
            return broken

    monkeypatch.setattr(cli_module, "_default_llm_factory", lambda _seed, _settings: _BadLLM())

    result = runner.invoke(app, ["new", "--seed", "1"])
    assert result.exit_code != 0

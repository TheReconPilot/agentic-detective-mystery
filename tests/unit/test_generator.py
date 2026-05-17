from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mystery.case_gen.generator import GenerationFailed, generate_bible
from mystery.case_gen.validate import BibleInvariantError

if TYPE_CHECKING:
    from mystery.models import CaseBible


class _ScriptedLLM:
    """Returns each scripted item in turn; raises Exceptions, yields CaseBibles."""

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.calls = 0
        self.user_prompts: list[str] = []

    def generate_bible(self, system: str, user: str) -> CaseBible:
        del system
        self.calls += 1
        self.user_prompts.append(user)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]


def test_returns_bible_on_first_success(valid_bible: CaseBible) -> None:
    llm = _ScriptedLLM([valid_bible])
    result = generate_bible(seed=1, llm=llm)
    assert result == valid_bible
    assert llm.calls == 1


def test_retries_on_invariant_violation(valid_bible: CaseBible) -> None:
    broken = valid_bible.model_copy(update={"killer_id": "ghost"})
    llm = _ScriptedLLM([broken, valid_bible])
    result = generate_bible(seed=1, llm=llm, max_attempts=3)
    assert result == valid_bible
    assert llm.calls == 2


def test_gives_up_after_max_attempts(valid_bible: CaseBible) -> None:
    broken = valid_bible.model_copy(update={"killer_id": "ghost"})
    llm = _ScriptedLLM([broken, broken, broken])
    with pytest.raises(GenerationFailed) as exc_info:
        generate_bible(seed=1, llm=llm, max_attempts=3)
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_error, BibleInvariantError)
    assert llm.calls == 3


def test_max_attempts_must_be_positive(valid_bible: CaseBible) -> None:
    llm = _ScriptedLLM([valid_bible])
    with pytest.raises(ValueError, match="max_attempts"):
        generate_bible(seed=1, llm=llm, max_attempts=0)


def test_retry_feeds_validation_error_back_into_prompt(valid_bible: CaseBible) -> None:
    """The retry prompt must surface the prior error so the LLM can self-correct."""
    broken = valid_bible.model_copy(update={"killer_id": "ghost"})
    llm = _ScriptedLLM([broken, valid_bible])
    generate_bible(seed=42, llm=llm, max_attempts=3)
    assert llm.calls == 2
    # First call: plain prompt, no error context.
    assert "previous attempt" not in llm.user_prompts[0].lower()
    # Second call: includes the BibleInvariantError text from the first failure.
    assert "ghost" in llm.user_prompts[1]
    assert "previous attempt" in llm.user_prompts[1].lower()

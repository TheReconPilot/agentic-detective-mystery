from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mystery.case_gen.generator import GenerationFailed, generate_bible
from mystery.case_gen.validate import BibleInvariantError

if TYPE_CHECKING:
    from mystery.models import CaseBible


class _ScriptedLLM:
    """Returns each scripted bible in turn; raises Exceptions, yields CaseBibles.

    Records BOTH the stage-1 (premise text) and stage-2 (structured bible)
    prompts so tests can assert on either. The stage-1 call is invoked exactly
    once per ``generate_bible``; only the stage-2 call retries.
    """

    def __init__(self, script: list[object], premise_text: str = "PREMISE TEXT") -> None:
        self._script = list(script)
        self.calls = 0
        self.user_prompts: list[str] = []
        self.premise_text = premise_text
        self.premise_calls = 0
        self.premise_user_prompts: list[str] = []

    def generate_premise_text(self, system: str, user: str) -> str:
        del system
        self.premise_calls += 1
        self.premise_user_prompts.append(user)
        return self.premise_text

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
    # Stage-1 fires exactly once per generation regardless of stage-2 retries.
    assert llm.premise_calls == 1


def test_retries_on_invariant_violation(valid_bible: CaseBible) -> None:
    broken = valid_bible.model_copy(update={"killer_id": "ghost"})
    llm = _ScriptedLLM([broken, valid_bible])
    result = generate_bible(seed=1, llm=llm, max_attempts=3)
    assert result == valid_bible
    assert llm.calls == 2
    # The premise is rolled once and reused across retries — not re-expanded.
    assert llm.premise_calls == 1


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


def test_stage1_premise_is_passed_to_stage2(valid_bible: CaseBible) -> None:
    """Stage-2 prompts must quote the stage-1 premise text as creative anchor."""
    llm = _ScriptedLLM([valid_bible], premise_text="A POIROT-FREE LOCKED-ROOM SETUP")
    generate_bible(seed=1, llm=llm)
    assert llm.premise_calls == 1
    assert "POIROT-FREE LOCKED-ROOM SETUP" in llm.user_prompts[0]


def test_stage2_prompt_contains_rolled_premise_constraints(valid_bible: CaseBible) -> None:
    """Stage-2 must include the rolled setting/era/cast as hard constraints.

    Without this, the LLM falls back to its prior and the diversity gain
    from rolling the premise in Python is lost.
    """
    llm = _ScriptedLLM([valid_bible])
    generate_bible(seed=1, llm=llm)
    from mystery.case_gen.premise import roll_premise

    premise = roll_premise(1)
    assert premise.setting in llm.user_prompts[0]
    assert premise.era in llm.user_prompts[0]
    # At least one rolled role should appear in the prompt.
    assert any(role in llm.user_prompts[0] for role in premise.cast_roles)


def test_retry_prompt_preserves_premise(valid_bible: CaseBible) -> None:
    """Retries must keep the same setting — we don't want to ricochet on schema fails."""
    broken = valid_bible.model_copy(update={"killer_id": "ghost"})
    llm = _ScriptedLLM([broken, valid_bible])
    generate_bible(seed=7, llm=llm, max_attempts=3)
    from mystery.case_gen.premise import roll_premise

    premise = roll_premise(7)
    assert premise.setting in llm.user_prompts[0]
    assert premise.setting in llm.user_prompts[1]


def test_asymmetric_edges_are_repaired_not_retried(valid_bible: CaseBible) -> None:
    """A one-way edge is mechanical — fix it, don't burn a retry on it.

    Small instruct models routinely list A->B and forget B->A. The repair
    pass adds the back-edge before validation so the bible passes on the
    first LLM attempt, instead of consuming all retries.
    """
    # Strip one back-edge from the valid bible to simulate the typical LLM mistake.
    from mystery.models import Location

    broken_locations = [
        Location(
            id=loc.id,
            name=loc.name,
            description=loc.description,
            connected_location_ids=(
                [n for n in loc.connected_location_ids if n != "library"]
                if loc.id == "hallway"
                else list(loc.connected_location_ids)
            ),
        )
        for loc in valid_bible.locations
    ]
    asymmetric = valid_bible.model_copy(update={"locations": broken_locations})

    llm = _ScriptedLLM([asymmetric])
    result = generate_bible(seed=1, llm=llm, max_attempts=3)

    # One LLM call only — the asymmetry was repaired, not retried.
    assert llm.calls == 1
    # Back-edge is present in the result.
    hallway = next(loc for loc in result.locations if loc.id == "hallway")
    assert "library" in hallway.connected_location_ids

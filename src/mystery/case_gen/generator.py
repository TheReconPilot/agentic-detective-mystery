"""Bible generation orchestration.

The generator is decoupled from any specific LLM via the ``BibleLLM`` protocol.
Tests inject a stub; production wires in the Ollama implementation from
``mystery.case_gen.llm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from mystery.case_gen.prompts import SYSTEM_PROMPT, retry_user_prompt, user_prompt
from mystery.case_gen.validate import validate_bible

if TYPE_CHECKING:
    from mystery.models import CaseBible


class BibleLLM(Protocol):
    """Anything that turns a (system, user) prompt pair into a parsed CaseBible."""

    def generate_bible(self, system: str, user: str) -> CaseBible: ...


class GenerationFailed(RuntimeError):
    """All retry attempts produced an invalid bible."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        super().__init__(
            f"bible generation failed after {attempts} attempts; last error: {last_error}",
        )
        self.attempts = attempts
        self.last_error = last_error


def generate_bible(
    seed: int,
    llm: BibleLLM,
    *,
    max_attempts: int = 3,
) -> CaseBible:
    """Generate a CaseBible, retrying on shape or invariant violations.

    The LLM is expected to honor ``seed`` so the same (seed, model) pair tends
    to produce the same case. The retry loop catches both Pydantic shape errors
    (raised by the LLM wrapper during structured-output parsing) and our own
    semantic invariants.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        prompt = (
            user_prompt(seed)
            if last_error is None
            else retry_user_prompt(seed, last_error, attempt)
        )
        try:
            bible = llm.generate_bible(SYSTEM_PROMPT, prompt)
            validate_bible(bible)
        except ValueError as e:
            last_error = e
            continue
        else:
            return bible

    assert last_error is not None  # the loop ran at least once
    raise GenerationFailed(max_attempts, last_error)

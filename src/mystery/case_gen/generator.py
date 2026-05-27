"""Bible generation orchestration.

The generator is decoupled from any specific LLM via the ``BibleLLM`` protocol.
Tests inject a stub; production wires in the Ollama implementation from
``mystery.case_gen.llm``.

Generation is two stages (see ``prompts.py`` for the why):

  1. ``roll_premise(seed)`` deterministically picks setting/era/cast/death from
     the curated lists in ``premise.py`` — pure Python, no LLM.
  2. ``llm.generate_premise_text(...)`` expands the rolled premise into a short
     atmospheric paragraph (no JSON schema attached, so a small model can
     commit fully to the setting).
  3. ``llm.generate_bible(...)`` does the constrained structured-output call,
     re-using the rolled premise as hard constraints. Retries on schema or
     invariant failures feed the error back in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from mystery.case_gen.premise import roll_premise
from mystery.case_gen.prompts import (
    PREMISE_EXPANSION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    premise_expansion_user_prompt,
    retry_user_prompt,
    user_prompt,
)
from mystery.case_gen.validate import validate_bible
from mystery.models import Alibi, CaseBible, Location, Suspect

if TYPE_CHECKING:
    from mystery.case_gen.premise import Premise


class BibleLLM(Protocol):
    """The two LLM calls the generator needs.

    ``generate_premise_text`` is unconstrained free-form chat; ``generate_bible``
    is structured-output bound to the CaseBible schema. Splitting them lets the
    same wrapper bind/unbind the schema between calls — and lets tests stub
    each independently.
    """

    def generate_premise_text(self, system: str, user: str) -> str: ...

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
    """Generate a CaseBible: roll premise → expand prose → structured bible.

    The premise is rolled (and the prose expanded) ONCE up front, then reused
    across retries. We don't want a retry to land in a different setting — the
    user's seed is meant to be stable, and stage 1 is the slow part.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    premise = roll_premise(seed)
    premise_text = llm.generate_premise_text(
        PREMISE_EXPANSION_SYSTEM_PROMPT,
        premise_expansion_user_prompt(premise),
    )

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        prompt = _build_stage2_prompt(seed, premise, premise_text, last_error, attempt)
        try:
            bible = llm.generate_bible(SYSTEM_PROMPT, prompt)
            bible = _repair_mechanical(bible)
            validate_bible(bible)
        except ValueError as e:
            last_error = e
            continue
        else:
            return bible

    assert last_error is not None  # the loop ran at least once
    raise GenerationFailed(max_attempts, last_error)


def _repair_mechanical(bible: CaseBible) -> CaseBible:
    """Auto-fix purely-mechanical violations the LLM keeps re-introducing.

    Bookkeeping mistakes — forgotten back-edges, witness IDs that don't match
    any suspect — burn whole retries on what amount to one-line fixes. We
    repair them in Python before validation. Creative invariants (killer
    alibi is a lie, killer is incriminated, etc.) are NOT auto-fixed — those
    still trigger a retry because they require the model to rework the story.
    """
    bible = _symmetrize_location_edges(bible)
    bible = _drop_unknown_alibi_witnesses(bible)
    return bible


def _drop_unknown_alibi_witnesses(bible: CaseBible) -> CaseBible:
    """Null out ``corroborating_witness_id``s that don't match any real suspect.

    Small models, especially on unfamiliar settings, occasionally cite a
    witness role rather than a suspect id ("waiter", "guard") — characters
    they never added to the suspects list. The schema allows ``None``, and
    a witness-less alibi is fine; better than the LLM ricocheting through
    five retries trying to invent a "waiter" suspect.
    """
    suspect_ids = {s.id for s in bible.suspects}
    changed = False
    new_suspects: list[Suspect] = []
    for s in bible.suspects:
        new_alibis: list[Alibi] = []
        for a in s.alibis:
            if (
                a.corroborating_witness_id is not None
                and a.corroborating_witness_id not in suspect_ids
            ):
                new_alibis.append(a.model_copy(update={"corroborating_witness_id": None}))
                changed = True
            else:
                new_alibis.append(a)
        new_suspects.append(s.model_copy(update={"alibis": new_alibis}))
    if not changed:
        return bible
    return bible.model_copy(update={"suspects": new_suspects})


def _symmetrize_location_edges(bible: CaseBible) -> CaseBible:
    adj: dict[str, list[str]] = {
        loc.id: list(loc.connected_location_ids) for loc in bible.locations
    }
    changed = False
    for src, neighbours in list(adj.items()):
        for dst in neighbours:
            if dst in adj and src not in adj[dst]:
                adj[dst].append(src)
                changed = True
    if not changed:
        return bible
    new_locations = [
        Location(
            id=loc.id,
            name=loc.name,
            description=loc.description,
            connected_location_ids=adj[loc.id],
        )
        for loc in bible.locations
    ]
    return bible.model_copy(update={"locations": new_locations})


def _build_stage2_prompt(
    seed: int,
    premise: Premise,
    premise_text: str,
    last_error: Exception | None,
    attempt: int,
) -> str:
    """Select first-attempt vs retry prompt; keep the premise anchored either way."""
    if last_error is None:
        return user_prompt(seed, premise, premise_text)
    return retry_user_prompt(seed, premise, last_error, attempt)

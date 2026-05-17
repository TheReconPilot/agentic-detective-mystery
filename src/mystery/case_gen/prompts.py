"""Prompts for case-bible generation.

Kept in plain Python strings (rather than LangChain PromptTemplate objects)
because the generator passes them through a Protocol that knows nothing about
LangChain. The Ollama wrapper applies them as system/user messages.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a mystery-game case designer. Your job is to invent a single self-contained
murder mystery and return it as a structured JSON document conforming to the
CaseBible schema.

Hard requirements:
- Exactly one killer, recorded in `killer_id`.
- 3 to 5 suspects. Give each a distinct archetype, motive (or None for the obvious
  innocent), a deception_policy in plain English, and a `voice` of 2-3 sentences
  describing how they talk — speech rhythm, a verbal tic or favourite phrase, and
  one topic they steer away from. Make the voices recognisably different from
  each other; flat NPC-speak ruins the game.
- 4 to 6 locations forming a small connected graph (use `connected_location_ids`).
- 4 to 8 physical clues scattered across locations. At least one clue MUST
  incriminate the killer; some clues may point to innocents as red herrings.
- Each suspect needs at least one alibi whose `time_window` covers `victim.time_of_death`.
- The killer's alibi covering the time of death MUST have `is_true: false`.
- All ids are lowercase snake_case slugs. Ids are unique within their type.
- Cross-references must resolve: every `location_id`, `corroborating_witness_id`,
  and `incriminates_suspect_ids` entry must refer to something that exists in the bible.

Style:
- Atmospheric but terse. Think Agatha Christie, not pulp.
- Time is measured in integer minutes from case t0 (the start of the evening).
"""

USER_PROMPT_TEMPLATE = """\
Generate a complete CaseBible with seed={seed}.

Use the seed as a creative anchor: different seeds should yield meaningfully
different settings, casts, and motives. Set the `seed` field of the bible to {seed}.
"""


def user_prompt(seed: int) -> str:
    return USER_PROMPT_TEMPLATE.format(seed=seed)


_RETRY_PROMPT_TEMPLATE = """\
Your previous attempt #{attempt} at seed={seed} failed validation:

    {error}

Generate a NEW complete CaseBible for seed={seed} that fixes this problem.
Be especially careful that every id you reference (locations in alibis and
clues, suspect ids in witnesses and clue.incriminates_suspect_ids, the
killer_id) refers to something you actually defined elsewhere in the bible.
Set the `seed` field of the bible to {seed}.
"""


def retry_user_prompt(seed: int, error: Exception, attempt: int) -> str:
    """Prompt for a retry attempt, surfacing the prior validation error.

    The bare retry loop produced ~50% failure rates on real LLMs because each
    attempt was blind to the previous mistake. Feeding the error back lets the
    model cross-check its ids on the next try.
    """
    return _RETRY_PROMPT_TEMPLATE.format(seed=seed, error=error, attempt=attempt)

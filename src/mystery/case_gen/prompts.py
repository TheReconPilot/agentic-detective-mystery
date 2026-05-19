"""Prompts for case-bible generation.

Kept in plain Python strings (rather than LangChain PromptTemplate objects)
because the generator passes them through a Protocol that knows nothing about
LangChain. The Ollama wrapper applies them as system/user messages.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a mystery-game case designer. Your job is to invent a single self-contained
murder mystery and return it as a structured JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXACT FIELD NAMES — copy these exactly, no variations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Top-level: seed, victim, suspects, locations, clues, killer_id, canonical_timeline
  (Do NOT add: case_title, case_summary, witnesses, title, or any other top-level key)

victim:
  name, role, location_of_death_id, time_of_death
  (NOT: victim_id, location_of_death, death_location, death_time)

suspects[] — each has:
  id, name, archetype, motive, alibis, knowledge, deception_policy, voice
  id is a slug like "butler" or "mrs_harlow"; name is the full human name like "James Weston"
  Both id AND name are required — do not omit name.
  (NOT: suspect_id, relation, description, alibi — the field is "id" and "alibis")

suspects[].alibis[] — each has:
  location_id, time_window, is_true, corroborating_witness_id
  time_window is a JSON array of two integers: [start_minutes, end_minutes]
  (NOT: witness_id, time, time_range — time_window MUST be [int, int])

locations[] — each has:
  id, name, description, connected_location_ids
  (NOT: location_id, location_name — the field is "id")

clues[] — each has:
  id, location_id, description, incriminates_suspect_ids
  (NOT: clue_id — the field is "id")

canonical_timeline[] — each has:
  time, actor_id, location_id, description
  time is an integer (minutes from t0), description is a string sentence
  (NOT: event, timestamp, events — field is "description"; time is int NOT string)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hard requirements:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Exactly one killer, recorded in `killer_id`.
- 3 to 5 suspects. Give each a distinct archetype, motive (or null for the obvious
  innocent), a deception_policy in plain English, a `knowledge` list of discrete
  facts this character truthfully knows (one sentence each), and a `voice` of 2-3
  sentences describing how they talk — speech rhythm, a verbal tic or favourite
  phrase, and one topic they steer away from.
- 4 to 6 locations forming a small connected graph (use `connected_location_ids`).
  Edges must be bidirectional: if A lists B in connected_location_ids then B must
  list A too.
- 4 to 8 physical clues scattered across locations. At least one clue MUST
  incriminate the killer; some clues may point to innocents as red herrings.
- Each suspect needs at least one alibi whose `time_window` covers
  `victim.time_of_death`. time_window is [start, end] as integers.
- The killer's alibi covering the time of death MUST have `is_true: false`.
- All ids are lowercase snake_case slugs (e.g. "butler", "library"). Unique per type.
- Cross-references must resolve: every `location_id`, `corroborating_witness_id`,
  and `incriminates_suspect_ids` entry must match an id that exists in the bible.

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

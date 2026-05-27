"""Prompts for case-bible generation.

Kept in plain Python strings (rather than LangChain PromptTemplate objects)
because the generator passes them through a Protocol that knows nothing about
LangChain. The Ollama wrapper applies them as system/user messages.

Generation is two-stage:

  1. A short *premise expansion* prompt asks the LLM, with no JSON schema
     attached, to flesh out a Python-rolled (setting, era, cast, death) into
     a four-to-six sentence atmospheric paragraph plus a victim sketch. This
     stage exists to free the model from its "Edwardian manor" prior; with no
     schema competing for attention, the small instruct models commit to the
     rolled setting properly.

  2. The structured-output prompt (`SYSTEM_PROMPT` + `user_prompt`) reuses
     the rolled premise as hard constraints AND quotes the expanded paragraph
     as creative anchor, then asks for the full CaseBible. Because the cast
     roles are dictated up-front, the model can't fall back on butler/maid/
     cook regardless of how heavily its prior pulls that direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mystery.case_gen.premise import Premise


SYSTEM_PROMPT = """\
You are a mystery-game case designer. Your job is to invent a single self-contained
murder mystery and return it as a structured JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXACT FIELD NAMES — copy these exactly, no variations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Top-level: seed, victim, suspects, locations, clues, killer_id, canonical_timeline
  (Do NOT add: case_title, case_summary, witnesses, title, or any other top-level key)

victim:
  name, role, location_of_death_id, time_of_death, forensic_details
  forensic_details is one sentence describing what a careful look at the body
  reveals: cause of death, signs of struggle, distinctive marks. Phrase it as
  *properties* of the method (e.g. "puncture mark on the neck consistent with
  injection") — do NOT name the killer.
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
  id, location_id, description, incriminates_suspect_ids, forensic_details
  forensic_details is one sentence about what a closer analysis reveals —
  material origin, manufacturing marks, chemical signature, technique used.
  Phrase as PROPERTIES that narrow the suspect set (e.g. "a small medicinal
  vial of the type kept by household staff for pest control") — do NOT name
  a suspect outright. The surface `description` should stay generic;
  forensic_details is what the player earns by spending a turn on `analyze`.
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
  list A too. Location names MUST fit the setting given in the user prompt — do
  not invent libraries, studies, drawing rooms, or hallways unless the setting
  is literally a manor house.
- 4 to 8 physical clues scattered across locations. At least one clue MUST
  incriminate the killer; some clues may point to innocents as red herrings.
- Each suspect needs at least one alibi whose `time_window` covers
  `victim.time_of_death`. time_window is [start, end] as integers.
- The killer's alibi covering the time of death MUST have `is_true: false`.
- All ids are lowercase snake_case slugs (e.g. "butler", "library"). Unique per type.
- Cross-references must resolve: every `location_id`, `corroborating_witness_id`,
  and `incriminates_suspect_ids` entry must match an id that exists in the bible.
- The suspects you create are the ONLY characters in the case. Do not name a
  witness or incriminated party that is not one of the suspects in this bible.
  If an alibi has no corroborator among the suspects, use `null`, not a made-up id.
- The VICTIM is a separate character from every suspect. The victim has their
  own name and role; no suspect may share the victim's name. If the rolled
  cast contains a role that would obviously be killed in this setting (e.g.
  "the head chef" in a restaurant), keep that role as a SUSPECT and invent
  a different victim (a visiting critic, a senior partner, a relative — pick
  whichever fits).

Style:
- Atmospheric but terse. Stay in the setting given by the user — every name,
  location, motive, and clue should feel native to it. Period and idiom should
  match the era.
- Time is measured in integer minutes from case t0 (the start of the case).
"""


PREMISE_EXPANSION_SYSTEM_PROMPT = """\
You are a mystery-story brainstormer. The user will hand you a rolled premise:
a setting, an era, a list of suspect roles, and how the victim died. Your job
is to flesh it out as PLAIN PROSE — no JSON, no lists, no headers.

Write four to six sentences that:
  - name the victim (a name that fits the era and setting),
  - give the victim a one-phrase role in this world — a role DIFFERENT from
    every suspect role; the victim is a separate character, never one of the
    listed suspects,
  - place the death in a specific room/area within the setting,
  - hint at one tension between two of the listed suspect roles,
  - establish the time-of-day and mood.

Do not invent suspect roles beyond those given. Do not name suspects yet —
keep them as their roles. Stay terse and atmospheric. No meta-commentary.
"""


def premise_expansion_user_prompt(premise: Premise) -> str:
    """Render a rolled premise as the stage-1 free-text expansion prompt."""
    roles_lines = "\n".join(f"  - {r}" for r in premise.cast_roles)
    return (
        f"Setting: {premise.setting}\n"
        f"Era: {premise.era}\n"
        f"Suspect roles:\n{roles_lines}\n"
        f"How the victim died: {premise.death_scenario}\n\n"
        "Write the four-to-six-sentence prose expansion now."
    )


def user_prompt(seed: int, premise: Premise, premise_text: str) -> str:
    """Stage-2 prompt: ask for the full structured CaseBible.

    The rolled premise goes in as bullet-pointed hard constraints; the
    stage-1 prose paragraph follows as creative anchor. Both are needed —
    the constraints stop the model from drifting to butler/maid/cook, and
    the prose gives it a concrete victim, mood, and tension to build on.
    """
    return (
        f"Generate a complete CaseBible with seed={seed}.\n\n"
        f"HARD CONSTRAINTS (do not deviate):\n"
        f"{premise.to_constraint_text()}\n\n"
        f"Use this brainstormed prose as creative anchor — borrow its victim "
        f"name, room of death, and tensions; expand into the full bible:\n\n"
        f"  {premise_text.strip()}\n\n"
        f"Set the `seed` field of the bible to {seed}."
    )


_RETRY_PROMPT_TEMPLATE = """\
Your previous attempt #{attempt} at seed={seed} failed validation:

    {error}

Generate a NEW complete CaseBible for seed={seed} that fixes this problem.
The HARD CONSTRAINTS below still apply unchanged — do not change the setting,
era, cast roles, or death scenario:

{constraints}

Be especially careful that every id you reference (locations in alibis and
clues, suspect ids in witnesses and clue.incriminates_suspect_ids, the
killer_id) refers to something you actually defined elsewhere in the bible.
Set the `seed` field of the bible to {seed}.
"""


def retry_user_prompt(seed: int, premise: Premise, error: Exception, attempt: int) -> str:
    """Prompt for a retry attempt, surfacing the prior validation error.

    The bare retry loop produced ~50% failure rates on real LLMs because each
    attempt was blind to the previous mistake. Feeding the error back lets the
    model cross-check its ids on the next try. The rolled premise is repeated
    because retries otherwise lose the anchor and drift back to defaults.
    """
    return _RETRY_PROMPT_TEMPLATE.format(
        seed=seed,
        error=error,
        attempt=attempt,
        constraints=premise.to_constraint_text(),
    )

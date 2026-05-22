"""Examine tool: surface all clues at the current location.

``examine victim`` is a separate branch that only fires when the player is in
the death room; it reveals the victim's ``forensic_details`` (cause of death,
puncture marks, etc.) and is the cheapest way to nail down the *method* before
chasing alibis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


_VICTIM_TARGETS = frozenset({"victim", "body", "corpse"})


def _apply_examine_victim(state: GameState, bible: CaseBible) -> dict[str, Any]:
    """Forensic look at the body — only meaningful in the death room."""
    death_room = bible.victim.location_of_death_id
    if state["current_location_id"] != death_room:
        return {
            "last_output": (
                f"The body isn't here. {bible.victim.name} was found in {death_room!r}."
            ),
        }
    forensic = bible.victim.forensic_details.strip()
    if not forensic:
        text = (
            f"You examine the body of {bible.victim.name}. Without proper "
            "forensic equipment the cause of death is not obvious."
        )
        note = f"VICTIM EXAM: {bible.victim.name} — no obvious cause of death visible."
    else:
        text = f"You examine the body of {bible.victim.name} closely.\n  - {forensic}"
        note = f"VICTIM EXAM: {forensic}"
    new_notebook = state["notebook"] + [note]
    return {
        "notebook": new_notebook,
        "turn_count": state["turn_count"] + 1,
        "last_output": text,
    }


def apply_examine(
    state: GameState,
    bible: CaseBible,
    target: str | None = None,
) -> dict[str, Any]:
    """Reveal all clues at the current location, append fresh ones to the notebook.

    Re-examining a room is allowed (it just costs a turn and re-prints) — the
    notebook never duplicates clues already seen.

    If ``target`` is one of the victim aliases, dispatch to the body-exam
    branch instead of the room sweep.
    """
    if target is not None and target.lower() in _VICTIM_TARGETS:
        return _apply_examine_victim(state, bible)
    if target is not None:
        return {
            "last_output": (
                f"You can examine the room (no argument) or the victim. "
                f"There is nothing called {target!r} to examine here."
            ),
        }
    here = state["current_location_id"]
    clues_here = [c for c in bible.clues if c.location_id == here]
    already = set(state["revealed_clue_ids"])

    fresh = [c for c in clues_here if c.id not in already]
    new_revealed = sorted(already | {c.id for c in clues_here})
    new_notebook = state["notebook"] + [f"[{c.id}] {c.description}" for c in fresh]

    if not clues_here:
        text = "You search the room thoroughly. Nothing of interest catches your eye."
    elif not fresh:
        # Re-printing descriptions (not just ids) on re-examine, so a player
        # who has forgotten the slug can still see "the muddy boots" and use
        # that phrase with `show`.
        text = "You see only the clues you have already catalogued:\n" + "\n".join(
            f"  - {c.description} [{c.id}]" for c in clues_here
        )
    else:
        text = "You find:\n" + "\n".join(f"  - {c.description} [{c.id}]" for c in fresh)

    examined = sorted(set(state["examined_location_ids"]) | {here})
    return {
        "revealed_clue_ids": new_revealed,
        "examined_location_ids": examined,
        "notebook": new_notebook,
        "turn_count": state["turn_count"] + 1,
        "last_output": text,
    }

"""Examine tool: surface all clues at the current location."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_examine(state: GameState, bible: CaseBible) -> dict[str, Any]:
    """Reveal all clues at the current location, append fresh ones to the notebook.

    Re-examining a room is allowed (it just costs a turn and re-prints) — the
    notebook never duplicates clues already seen.
    """
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

"""Topics tool: surface things the player can ``ask <suspect> about <X>``.

A scaffold for the hardest part of detective fiction — knowing *what* to ask.
The list is derived from current state: every other suspect, every revealed
clue, every visited location. Nothing here gives the case away; it just shows
the menu the LLM is already willing to talk about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_topics(state: GameState, bible: CaseBible) -> dict[str, Any]:
    """Render the askable-topics menu — free action, no turn cost."""
    suspect_lines = [f"  [{s.id}] {s.name} — {s.archetype}" for s in bible.suspects]

    revealed_clues = [c for c in bible.clues if c.id in state["revealed_clue_ids"]]
    if revealed_clues:
        clue_lines = [f"  [{c.id}] {c.description}" for c in revealed_clues]
    else:
        clue_lines = ["  (none yet — try 'examine' in a room)"]

    visited = state["visited_location_ids"]
    by_id = {loc.id: loc for loc in bible.locations}
    location_lines = [f"  [{lid}] {by_id[lid].name}" for lid in sorted(visited) if lid in by_id]

    text = (
        "You can use 'ask <suspect> about <topic>' as a shortcut. "
        "Topics resolve to clues, locations, or other people.\n\n"
        f"People:\n{chr(10).join(suspect_lines)}\n\n"
        f"Clues you've found:\n{chr(10).join(clue_lines)}\n\n"
        f"Places you've been:\n{chr(10).join(location_lines)}"
    )
    return {"last_output": text}

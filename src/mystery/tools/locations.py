"""Locations tool: show current room + adjacent rooms. Free action (no turn cost).

Without this the player can only learn the map by attempting bad moves and
reading the error from :func:`mystery.tools.move.apply_move`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_locations(state: GameState, bible: CaseBible) -> dict[str, Any]:
    by_id = {loc.id: loc for loc in bible.locations}
    here = by_id[state["current_location_id"]]
    if here.connected_location_ids:
        exits = "\n".join(
            f"  - [{lid}] {by_id[lid].name}" for lid in here.connected_location_ids if lid in by_id
        )
        exits_block = f"From here you can go to:\n{exits}"
    else:
        exits_block = "There are no exits from this room."
    text = f"You are in {here.name} [{here.id}]. {here.description}\n{exits_block}"
    return {"last_output": text}

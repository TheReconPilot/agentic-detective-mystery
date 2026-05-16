"""Move tool: change current_location_id along the location graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_move(state: GameState, bible: CaseBible, location_id: str) -> dict[str, Any]:
    """Return the partial state update for a move attempt.

    Refuses to move to unknown locations or to non-adjacent ones, with a
    message instead of an exception — bad moves are gameplay, not errors.
    """
    all_locations = {loc.id: loc for loc in bible.locations}
    if location_id not in all_locations:
        return {"last_output": f"There is no place called {location_id!r}."}

    current = all_locations[state["current_location_id"]]
    if location_id == current.id:
        return {"last_output": f"You are already in {current.name}."}

    if location_id not in current.connected_location_ids:
        connected = ", ".join(current.connected_location_ids) or "(nowhere)"
        return {
            "last_output": (
                f"You cannot reach {location_id!r} directly from {current.id}. "
                f"From here you can go to: {connected}."
            ),
        }

    new_loc = all_locations[location_id]
    visited = sorted(set(state["visited_location_ids"]) | {location_id})
    return {
        "current_location_id": location_id,
        "visited_location_ids": visited,
        "turn_count": state["turn_count"] + 1,
        "last_output": f"You enter {new_loc.name}. {new_loc.description}",
    }

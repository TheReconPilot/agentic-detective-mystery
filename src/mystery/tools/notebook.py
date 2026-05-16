"""Notebook tool: display the player's accumulated notes. Free action (no turn cost)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mystery.graph.state import GameState


def apply_notebook(state: GameState) -> dict[str, Any]:
    if not state["notebook"]:
        return {"last_output": "Your notebook is empty."}
    text = "Your notebook:\n" + "\n".join(f"  {line}" for line in state["notebook"])
    return {"last_output": text}

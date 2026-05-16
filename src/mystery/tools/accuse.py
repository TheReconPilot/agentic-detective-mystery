"""Accuse tool: terminal action. Resolves against bible.killer_id and ends the game."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mystery.graph.state import AccusationResult

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_accuse(state: GameState, bible: CaseBible, suspect_id: str) -> dict[str, Any]:
    suspect_ids = {s.id for s in bible.suspects}
    if suspect_id not in suspect_ids:
        # Don't end the game on a typo — give the player back their turn.
        return {"last_output": f"There is no suspect {suspect_id!r}. The accusation is withdrawn."}

    correct = suspect_id == bible.killer_id
    accusation = AccusationResult(
        accused_id=suspect_id,
        correct=correct,
        actual_killer_id=bible.killer_id,
    )

    if correct:
        accused = next(s for s in bible.suspects if s.id == suspect_id)
        text = (
            f"You accuse {accused.name}. After a long pause, they confess. "
            f"The case is solved in {state['turn_count'] + 1} turns."
        )
    else:
        actual = next(s for s in bible.suspects if s.id == bible.killer_id)
        text = (
            f"You accuse {suspect_id}. They are innocent. "
            f"The real killer was {actual.name} — and they have already fled."
        )

    return {
        "accusation": accusation,
        "done": True,
        "turn_count": state["turn_count"] + 1,
        "last_output": text,
    }

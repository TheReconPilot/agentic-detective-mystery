"""Accuse tool: terminal action. Resolves against bible.killer_id and ends the game."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mystery.graph.state import AccusationResult
from mystery.tools._resolve import format_suspect_roster, resolve_suspect

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_accuse(state: GameState, bible: CaseBible, suspect_id: str) -> dict[str, Any]:
    suspect = resolve_suspect(bible, suspect_id)
    if suspect is None:
        # Don't end the game on a typo — give the player back their turn.
        roster = format_suspect_roster(bible)
        return {
            "last_output": (
                f"I'm not sure who you mean by {suspect_id!r}. The accusation is withdrawn.\n"
                f"You can accuse any of these (by id, name, or archetype):\n{roster}"
            )
        }

    correct = suspect.id == bible.killer_id
    accusation = AccusationResult(
        accused_id=suspect.id,
        correct=correct,
        actual_killer_id=bible.killer_id,
    )

    if correct:
        text = (
            f"You accuse {suspect.name}. After a long pause, they confess. "
            f"The case is solved in {state['turn_count'] + 1} turns."
        )
    else:
        actual = next(s for s in bible.suspects if s.id == bible.killer_id)
        text = (
            f"You accuse {suspect.name}. They are innocent. "
            f"The real killer was {actual.name} — and they have already fled."
        )

    return {
        "accusation": accusation,
        "done": True,
        "turn_count": state["turn_count"] + 1,
        "last_output": text,
    }

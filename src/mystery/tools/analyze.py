"""Analyze tool: forensic drill-down on a single revealed clue.

Where ``examine`` is a wide sweep that surfaces what's *in* a room, ``analyze``
is a narrow drill into one already-revealed clue's ``forensic_details``. The
case generator's brief is to fill those details with *properties* of the
killer (chemical signature, manufacturing origin, technique) rather than a
direct attribution, so the player still has to triangulate — but at least the
trail isn't gated behind ambiguous one-line descriptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mystery.tools._resolve import format_revealed_clues, resolve_clue

if TYPE_CHECKING:
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_analyze(state: GameState, bible: CaseBible, clue_id: str) -> dict[str, Any]:
    """Drill into a clue the player has already revealed.

    Refuses unrevealed clues with the same "go examine first" message as
    ``show`` — the player must have *seen* a clue to study it.
    """
    revealed = state["revealed_clue_ids"]
    clue = resolve_clue(bible, clue_id, revealed)
    if clue is None:
        inventory = format_revealed_clues(bible, revealed)
        exists = any(c.id == clue_id for c in bible.clues)
        if exists and clue_id not in revealed:
            return {
                "last_output": (
                    f"You haven't found {clue_id!r} yet. Examine the room "
                    "where it's hiding before you can analyze it.\n"
                    f"Clues you have found so far:\n{inventory}"
                ),
            }
        return {
            "last_output": (
                f"I'm not sure which clue you mean by {clue_id!r}.\n"
                f"Clues you have found so far:\n{inventory}"
            ),
        }

    forensic = clue.forensic_details.strip()
    if not forensic:
        text = (
            f"You study {clue.description.rstrip('.')} carefully. Without a "
            "proper lab there's nothing more to learn from it."
        )
        return {
            "turn_count": state["turn_count"] + 1,
            "last_output": text,
        }

    text = f"You study {clue.description.rstrip('.')} closely.\n  - {forensic}"
    note = f"FORENSICS [{clue.id}]: {forensic}"
    # Don't duplicate the forensic note on repeat analyze of the same clue.
    new_notebook = state["notebook"]
    if note not in new_notebook:
        new_notebook = [*new_notebook, note]
    return {
        "notebook": new_notebook,
        "turn_count": state["turn_count"] + 1,
        "last_output": text,
    }

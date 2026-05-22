"""Sidecar save file for an in-progress game.

The REPL writes the player-facing slice of ``GameState`` to
``cases/{seed}.save.json`` after every turn that produced a state update, and
restores it the next time ``play`` is invoked. ``pending_action`` and
``last_output`` are deliberately excluded — both are turn-scoped scratch state
the REPL re-sets per dispatch.

A finished game (``done=True``) is not saved; the save file is removed instead,
so the next ``play`` of the same seed starts cleanly.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime in function bodies
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

# Pydantic v2 needs Commitment available at runtime to validate the typed field
# below — a TYPE_CHECKING-only import would NameError when the model is built.
from mystery.models import Commitment  # noqa: TC001

if TYPE_CHECKING:
    from mystery.graph.state import GameState


class SaveState(BaseModel):
    """Serialisable snapshot of an in-progress game."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    seed: int
    current_location_id: str
    revealed_clue_ids: list[str]
    visited_location_ids: list[str]
    examined_location_ids: list[str]
    notebook: list[str]
    turn_count: int
    suspect_commitments: dict[str, list[Commitment]]


def save_path_for(cases_dir: Path, seed: int) -> Path:
    return cases_dir / f"{seed}.save.json"


def write_save(state: GameState, seed: int, path: Path) -> None:
    """Write the player-facing slice of ``state`` to ``path`` atomically.

    Atomic via write-then-rename so a crash mid-write cannot leave a partial
    file that would refuse to deserialise on the next launch.
    """
    snapshot = SaveState(
        seed=seed,
        current_location_id=state["current_location_id"],
        revealed_clue_ids=list(state["revealed_clue_ids"]),
        visited_location_ids=list(state["visited_location_ids"]),
        examined_location_ids=list(state["examined_location_ids"]),
        notebook=list(state["notebook"]),
        turn_count=state["turn_count"],
        suspect_commitments={k: list(v) for k, v in state["suspect_commitments"].items()},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def read_save(path: Path) -> SaveState | None:
    """Read a sidecar save file. Returns None if it doesn't exist."""
    if not path.exists():
        return None
    return SaveState.model_validate_json(path.read_text(encoding="utf-8"))


def apply_save_to_state(state: GameState, snapshot: SaveState) -> GameState:
    """Overlay a save snapshot onto a freshly-initialised state in-place."""
    state["current_location_id"] = snapshot.current_location_id
    state["revealed_clue_ids"] = list(snapshot.revealed_clue_ids)
    state["visited_location_ids"] = list(snapshot.visited_location_ids)
    state["examined_location_ids"] = list(snapshot.examined_location_ids)
    state["notebook"] = list(snapshot.notebook)
    state["turn_count"] = snapshot.turn_count
    state["suspect_commitments"] = {k: list(v) for k, v in snapshot.suspect_commitments.items()}
    return state


def remove_save(path: Path) -> None:
    """Remove a save file if it exists. Safe to call when no save is present."""
    path.unlink(missing_ok=True)

"""Unit tests for the four game-state tools (move/examine/notebook/accuse)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from mystery.graph.state import GameState, initial_state
from mystery.tools.accuse import apply_accuse
from mystery.tools.examine import apply_examine
from mystery.tools.move import apply_move
from mystery.tools.notebook import apply_notebook

if TYPE_CHECKING:
    from mystery.models import CaseBible


def _merge(state: GameState, update: dict[str, Any]) -> GameState:
    """Test helper: simulate LangGraph's state-merge step."""
    return cast("GameState", {**state, **update})


# ---------- move ----------


def test_move_to_connected_location_succeeds(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)  # starts in library
    update = apply_move(state, valid_bible, "hallway")
    assert update["current_location_id"] == "hallway"
    assert "hallway" in update["visited_location_ids"]
    assert update["turn_count"] == state["turn_count"] + 1


def test_move_to_unknown_location_rejected(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_move(state, valid_bible, "atlantis")
    assert "no place called" in update["last_output"]
    assert "current_location_id" not in update


def test_move_to_non_adjacent_location_rejected(valid_bible: CaseBible) -> None:
    """Library and garden are not directly connected — only via hallway."""
    state = initial_state(valid_bible)
    update = apply_move(state, valid_bible, "garden")
    assert "cannot reach" in update["last_output"]
    assert "current_location_id" not in update


def test_move_to_current_location_rejected(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_move(state, valid_bible, state["current_location_id"])
    assert "already in" in update["last_output"]


def test_visited_locations_deduplicated(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_move(state, valid_bible, "hallway")
    state = _merge(state, update)
    update = apply_move(state, valid_bible, "library")  # backtrack
    state = _merge(state, update)
    update = apply_move(state, valid_bible, "hallway")
    state = _merge(state, update)
    assert sorted(state["visited_location_ids"]) == ["hallway", "library"]


# ---------- examine ----------


def test_examine_in_room_with_clues_reveals_them(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)  # library has torn_letter
    update = apply_examine(state, valid_bible)
    assert "torn_letter" in update["revealed_clue_ids"]
    assert any("torn letter" in line.lower() for line in update["notebook"])
    assert update["turn_count"] == 1


def test_examine_re_examining_does_not_duplicate_notebook_entries(
    valid_bible: CaseBible,
) -> None:
    state = initial_state(valid_bible)
    first = apply_examine(state, valid_bible)
    state = _merge(state, first)
    second = apply_examine(state, valid_bible)
    assert second["notebook"] == state["notebook"]  # no new lines
    assert "already catalogued" in second["last_output"]


def test_examine_in_empty_room_is_a_no_op_except_turn_cost(
    valid_bible: CaseBible,
) -> None:
    """Move to the garden — no clues there — and examine."""
    state = initial_state(valid_bible)
    state = _merge(state, apply_move(state, valid_bible, "hallway"))
    state = _merge(state, apply_move(state, valid_bible, "garden"))
    starting_turn = state["turn_count"]
    update = apply_examine(state, valid_bible)
    assert update["revealed_clue_ids"] == []
    assert update["notebook"] == state["notebook"]
    assert update["turn_count"] == starting_turn + 1
    assert "Nothing of interest" in update["last_output"]


# ---------- notebook ----------


def test_notebook_shows_initial_victim_note(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_notebook(state)
    assert "VICTIM" in update["last_output"]
    assert "turn_count" not in update  # free action


def test_notebook_when_empty(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state = _merge(state, {"notebook": []})
    update = apply_notebook(state)
    assert update["last_output"] == "Your notebook is empty."


# ---------- accuse ----------


def test_accuse_correct_killer_ends_game_with_win(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_accuse(state, valid_bible, suspect_id="butler")
    assert update["done"] is True
    assert update["accusation"].correct is True
    assert update["accusation"].accused_id == "butler"
    assert update["accusation"].actual_killer_id == "butler"
    assert "case is solved" in update["last_output"]


def test_accuse_wrong_suspect_ends_game_with_loss(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_accuse(state, valid_bible, suspect_id="niece")
    assert update["done"] is True
    assert update["accusation"].correct is False
    assert update["accusation"].actual_killer_id == "butler"
    assert "innocent" in update["last_output"]


def test_accuse_unknown_suspect_is_withdrawn(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_accuse(state, valid_bible, suspect_id="ghost")
    assert "done" not in update
    assert "accusation" not in update
    assert "withdrawn" in update["last_output"]

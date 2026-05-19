"""Tests for the free-action discovery tools (suspects, locations)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mystery.graph.state import initial_state
from mystery.tools.locations import apply_locations
from mystery.tools.suspects import apply_suspects

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_apply_suspects_lists_every_suspect_with_id_and_archetype(
    valid_bible: CaseBible,
) -> None:
    update = apply_suspects(valid_bible)
    text = update["last_output"]
    for s in valid_bible.suspects:
        assert s.id in text
        assert s.name in text
        assert s.archetype in text
    # Free action: no turn_count bump.
    assert "turn_count" not in update


def test_apply_locations_shows_current_room_and_exits(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)  # library
    update = apply_locations(state, valid_bible)
    text = update["last_output"]
    assert "Library" in text
    # library only connects to hallway in the fixture bible.
    assert "hallway" in text
    assert "turn_count" not in update


def test_apply_locations_handles_dead_end_room(valid_bible: CaseBible) -> None:
    """A room with no connections still renders sensibly (no crash, no comma soup)."""
    bible = valid_bible.model_copy(deep=True)
    bible.locations[0].connected_location_ids = []
    state = initial_state(bible)
    update = apply_locations(state, bible)
    assert "no exits" in update["last_output"].lower()

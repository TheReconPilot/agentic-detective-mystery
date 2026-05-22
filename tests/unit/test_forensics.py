"""Tests for `examine victim` and `analyze <clue>`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mystery.graph.state import initial_state
from mystery.tools.analyze import apply_analyze
from mystery.tools.examine import apply_examine

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_examine_victim_in_death_room_reveals_forensics(valid_bible: CaseBible) -> None:
    valid_bible.victim.forensic_details = "Puncture mark on the neck consistent with injection."
    state = initial_state(valid_bible)  # starts in library = death room

    update = apply_examine(state, valid_bible, target="victim")
    assert "Puncture mark" in update["last_output"]
    assert any("VICTIM EXAM" in line and "Puncture mark" in line for line in update["notebook"])
    assert update["turn_count"] == state["turn_count"] + 1


def test_examine_victim_outside_death_room_refuses(valid_bible: CaseBible) -> None:
    valid_bible.victim.forensic_details = "Puncture mark on the neck."
    state = initial_state(valid_bible)
    state["current_location_id"] = "hallway"

    update = apply_examine(state, valid_bible, target="victim")
    assert "body isn't here" in update["last_output"]
    assert "turn_count" not in update  # no turn spent on a refused action


def test_examine_victim_with_no_forensic_details_says_so(valid_bible: CaseBible) -> None:
    # The fixture leaves forensic_details empty; behaviour should degrade gracefully.
    state = initial_state(valid_bible)
    update = apply_examine(state, valid_bible, target="victim")
    assert "not obvious" in update["last_output"]
    assert update["turn_count"] == state["turn_count"] + 1


def test_examine_unknown_target_refuses(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_examine(state, valid_bible, target="chandelier")
    assert "nothing called" in update["last_output"]
    assert "turn_count" not in update


def test_examine_no_target_still_sweeps_room(valid_bible: CaseBible) -> None:
    """The legacy no-arg path must remain a room sweep, not a no-op."""
    state = initial_state(valid_bible)
    update = apply_examine(state, valid_bible)
    assert "torn_letter" in update["revealed_clue_ids"]


def test_analyze_revealed_clue_returns_forensic_details(valid_bible: CaseBible) -> None:
    # Mutate the fixture: tag the muddy_boots clue with forensic detail.
    for clue in valid_bible.clues:
        if clue.id == "muddy_boots":
            clue.forensic_details = "Mud matches the loam from the garden's eastern bed."
    state = initial_state(valid_bible)
    state["revealed_clue_ids"] = ["muddy_boots"]

    update = apply_analyze(state, valid_bible, "muddy_boots")
    assert "garden's eastern bed" in update["last_output"]
    assert any("FORENSICS [muddy_boots]" in line for line in update["notebook"])
    assert update["turn_count"] == state["turn_count"] + 1


def test_analyze_unrevealed_clue_refuses(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    # muddy_boots exists in the bible but the player hasn't examined the hallway yet.
    update = apply_analyze(state, valid_bible, "muddy_boots")
    assert "haven't found" in update["last_output"]
    assert "turn_count" not in update


def test_analyze_unknown_clue_refuses(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_analyze(state, valid_bible, "ghost_clue")
    assert "not sure which clue" in update["last_output"]


def test_analyze_with_no_forensic_details_still_costs_a_turn(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["revealed_clue_ids"] = ["muddy_boots"]
    update = apply_analyze(state, valid_bible, "muddy_boots")
    assert "nothing more to learn" in update["last_output"]
    assert update["turn_count"] == state["turn_count"] + 1


def test_analyze_does_not_duplicate_notebook_entry(valid_bible: CaseBible) -> None:
    for clue in valid_bible.clues:
        if clue.id == "muddy_boots":
            clue.forensic_details = "Mud from the eastern bed."
    state = initial_state(valid_bible)
    state["revealed_clue_ids"] = ["muddy_boots"]

    first = apply_analyze(state, valid_bible, "muddy_boots")
    state2 = {**state, **first}
    second = apply_analyze(state2, valid_bible, "muddy_boots")  # type: ignore[arg-type]
    forensic_lines = [line for line in second["notebook"] if "FORENSICS [muddy_boots]" in line]
    assert len(forensic_lines) == 1

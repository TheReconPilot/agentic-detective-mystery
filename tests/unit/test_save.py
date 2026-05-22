"""Save/load roundtrip and atomicity tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mystery.graph.state import initial_state
from mystery.models import Commitment
from mystery.save import (
    apply_save_to_state,
    read_save,
    remove_save,
    save_path_for,
    write_save,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mystery.models import CaseBible


def test_read_save_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_save(tmp_path / "missing.json") is None


def test_save_load_roundtrip(tmp_path: Path, valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["current_location_id"] = "hallway"
    state["revealed_clue_ids"] = ["muddy_boots"]
    state["visited_location_ids"] = ["library", "hallway"]
    state["examined_location_ids"] = ["library"]
    state["notebook"] = ["VICTIM: Lord Ashworth.", "[muddy_boots] muddy boots"]
    state["turn_count"] = 5
    state["suspect_commitments"] = {
        "butler": [
            Commitment(
                claimed_location_id="garden",
                claimed_time_window=(45, 75),
                named_witness_ids=[],
                denied_facts=[],
                summary="They claimed to be in the garden between 45 and 75.",
            )
        ]
    }

    path = save_path_for(tmp_path, valid_bible.seed)
    write_save(state, valid_bible.seed, path)

    snapshot = read_save(path)
    assert snapshot is not None
    assert snapshot.seed == valid_bible.seed
    assert snapshot.current_location_id == "hallway"
    assert snapshot.revealed_clue_ids == ["muddy_boots"]
    assert snapshot.turn_count == 5
    assert snapshot.suspect_commitments["butler"][0].claimed_location_id == "garden"

    fresh = initial_state(valid_bible)
    apply_save_to_state(fresh, snapshot)
    assert fresh["current_location_id"] == "hallway"
    assert fresh["revealed_clue_ids"] == ["muddy_boots"]
    assert fresh["turn_count"] == 5
    assert fresh["suspect_commitments"]["butler"][0].summary.startswith("They claimed")


def test_remove_save_is_idempotent(tmp_path: Path, valid_bible: CaseBible) -> None:
    path = save_path_for(tmp_path, valid_bible.seed)
    remove_save(path)  # no-op on missing path
    state = initial_state(valid_bible)
    write_save(state, valid_bible.seed, path)
    assert path.exists()
    remove_save(path)
    assert not path.exists()


def test_write_save_is_atomic_via_rename(tmp_path: Path, valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    path = save_path_for(tmp_path, valid_bible.seed)
    write_save(state, valid_bible.seed, path)
    # The tmp sibling used during the rename dance should not be left behind.
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []

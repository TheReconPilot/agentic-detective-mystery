"""Tests for the notebook's derived suspect-summary section."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mystery.graph.state import initial_state
from mystery.models import Commitment
from mystery.tools.notebook import apply_notebook

if TYPE_CHECKING:
    from mystery.models import CaseBible


def _commitment(
    location: str | None = None,
    window: tuple[int, int] | None = None,
    witnesses: list[str] | None = None,
    summary: str = "(claim)",
) -> Commitment:
    return Commitment(
        claimed_location_id=location,
        claimed_time_window=window,
        named_witness_ids=witnesses or [],
        denied_facts=[],
        summary=summary,
    )


def test_summary_marks_uncorroborated_claim_with_warning(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["suspect_commitments"] = {
        "butler": [
            _commitment(location="garden", window=(45, 75), summary="They claimed the garden."),
        ]
    }
    update = apply_notebook(state, valid_bible)
    out = update["last_output"]
    assert "Suspect summary:" in out
    assert "butler (Mr. Hodges)" in out
    assert "claimed: garden [45-75]" in out
    assert "⚠ no corroborator named" in out


def test_summary_marks_mutual_corroboration_with_check(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["suspect_commitments"] = {
        "niece": [_commitment(location="hallway", window=(50, 70), witnesses=["cook"])],
        "cook": [_commitment(location="hallway", window=(50, 70), witnesses=["niece"])],
    }
    update = apply_notebook(state, valid_bible)
    out = update["last_output"]
    assert "✓ corroborated by Mrs. Pell" in out  # niece is corroborated by cook
    assert "✓ corroborated by Eleanor Ashworth" in out  # cook is corroborated by niece
    assert "⚠ no corroborator" not in out


def test_summary_flags_revealed_clue_pointing_at_suspect(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["revealed_clue_ids"] = ["muddy_boots"]
    state["suspect_commitments"] = {
        "butler": [_commitment(location="garden", summary="They claimed the garden.")],
    }
    update = apply_notebook(state, valid_bible)
    out = update["last_output"]
    # muddy_boots incriminates butler in the fixture
    assert "⚠ clue [muddy_boots] points here" in out


def test_summary_omitted_when_no_commitments(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_notebook(state, valid_bible)
    assert "Suspect summary:" not in update["last_output"]


def test_summary_skips_suspects_with_no_commitments(valid_bible: CaseBible) -> None:
    """Only suspects we've actually interrogated should appear in the rollup."""
    state = initial_state(valid_bible)
    state["suspect_commitments"] = {
        "butler": [_commitment(location="garden", summary="They claimed the garden.")],
    }
    update = apply_notebook(state, valid_bible)
    out = update["last_output"]
    assert "butler" in out
    # niece and cook have no commitments and should not appear.
    assert "Eleanor Ashworth" not in out
    assert "Mrs. Pell" not in out


def test_summary_uses_only_revealed_clues_for_flags(valid_bible: CaseBible) -> None:
    """A clue in the bible but not yet found by the player must not be flagged."""
    state = initial_state(valid_bible)
    # muddy_boots NOT in revealed_clue_ids
    state["suspect_commitments"] = {
        "butler": [_commitment(location="garden", summary="They claimed the garden.")],
    }
    update = apply_notebook(state, valid_bible)
    assert "clue [muddy_boots]" not in update["last_output"]

"""Shared test fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from mystery.models import (
    Alibi,
    CaseBible,
    Clue,
    Location,
    Suspect,
    TimelineEvent,
    Victim,
)


def _make_valid_bible() -> CaseBible:
    """Return a hand-rolled valid bible used as a baseline for invariant tests."""
    return CaseBible(
        seed=42,
        victim=Victim(
            name="Lord Ashworth",
            role="host of the dinner party",
            location_of_death_id="library",
            time_of_death=60,
        ),
        locations=[
            Location(
                id="library",
                name="Library",
                description="A dim room lined with leather-bound books.",
                connected_location_ids=["hallway"],
            ),
            Location(
                id="hallway",
                name="Hallway",
                description="A long marble corridor.",
                connected_location_ids=["library", "garden"],
            ),
            Location(
                id="garden",
                name="Garden",
                description="A moonlit hedge maze.",
                connected_location_ids=["hallway"],
            ),
        ],
        suspects=[
            Suspect(
                id="butler",
                name="Mr. Hodges",
                archetype="butler",
                motive="dismissed without pension that morning",
                alibis=[
                    Alibi(
                        location_id="garden",
                        time_window=(45, 75),
                        is_true=False,
                        corroborating_witness_id=None,
                    )
                ],
                knowledge=["The library door sticks when the air is humid."],
                deception_policy="Lies about being in the garden; truthful about the household.",
                voice=(
                    "Clipped, formal, lapses into 'sir' every other sentence. "
                    "Goes very quiet whenever the garden is mentioned."
                ),
            ),
            Suspect(
                id="niece",
                name="Eleanor Ashworth",
                archetype="estranged-niece",
                motive="cut out of the will last week",
                alibis=[
                    Alibi(
                        location_id="hallway",
                        time_window=(50, 70),
                        is_true=True,
                        corroborating_witness_id="cook",
                    )
                ],
                knowledge=["Lord Ashworth changed his will on Tuesday."],
                deception_policy="Truthful; nervous and over-explains.",
                voice=(
                    "Trails off, restarts mid-sentence, fills silences with "
                    "'I mean—'. Brittle, on the edge of tears."
                ),
            ),
            Suspect(
                id="cook",
                name="Mrs. Pell",
                archetype="cook",
                motive=None,
                alibis=[
                    Alibi(
                        location_id="hallway",
                        time_window=(50, 70),
                        is_true=True,
                        corroborating_witness_id="niece",
                    )
                ],
                knowledge=["Dinner was served at half past eight."],
                deception_policy="Truthful.",
                voice=(
                    "Warm, motherly, uses cooking metaphors for everything. "
                    "Calls the detective 'love'."
                ),
            ),
        ],
        clues=[
            Clue(
                id="muddy_boots",
                location_id="hallway",
                description="A pair of muddy boots tucked behind the umbrella stand.",
                incriminates_suspect_ids=["butler"],
            ),
            Clue(
                id="torn_letter",
                location_id="library",
                description="A torn letter mentioning a disinheritance.",
                incriminates_suspect_ids=["butler", "niece"],
            ),
        ],
        killer_id="butler",
        canonical_timeline=[
            TimelineEvent(
                time=45,
                actor_id="butler",
                location_id="library",
                description="The butler enters the library through the side door.",
            ),
            TimelineEvent(
                time=60,
                actor_id="butler",
                location_id="library",
                description="The butler poisons Lord Ashworth's brandy.",
            ),
        ],
    )


@pytest.fixture
def valid_bible() -> CaseBible:
    return _make_valid_bible()


@pytest.fixture
def valid_bible_dict() -> dict[str, Any]:
    return _make_valid_bible().model_dump()

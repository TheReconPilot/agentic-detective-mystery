"""Tests for the topics command and `ask <suspect> about <topic>` expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mystery.graph.router import parse_action
from mystery.graph.state import TopicsAction, initial_state
from mystery.tools.interrogate import _expand_topic_question
from mystery.tools.topics import apply_topics

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_topics_verb_parses_to_topics_action() -> None:
    assert parse_action("topics") == TopicsAction()


def test_apply_topics_lists_people_clues_locations(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["revealed_clue_ids"] = ["muddy_boots"]
    state["visited_location_ids"] = ["library", "hallway"]
    out = apply_topics(state, valid_bible)["last_output"]
    assert "People:" in out
    assert "Clues you've found:" in out
    assert "Places you've been:" in out
    assert "[muddy_boots]" in out
    assert "[hallway]" in out


def test_apply_topics_with_no_clues_shows_hint(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    out = apply_topics(state, valid_bible)["last_output"]
    assert "none yet" in out


def test_expand_about_clue_uses_clue_description(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["revealed_clue_ids"] = ["muddy_boots"]
    expanded = _expand_topic_question(state, valid_bible, "about muddy_boots")
    assert "muddy boots" in expanded.lower()
    assert "What can you tell me about it?" in expanded


def test_expand_about_clue_by_description_word(valid_bible: CaseBible) -> None:
    """Token-matching against the clue description should also resolve."""
    state = initial_state(valid_bible)
    state["revealed_clue_ids"] = ["muddy_boots"]
    expanded = _expand_topic_question(state, valid_bible, "about the boots")
    assert "muddy boots" in expanded.lower()


def test_expand_about_suspect(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    expanded = _expand_topic_question(state, valid_bible, "about niece")
    assert "Eleanor Ashworth" in expanded


def test_expand_about_visited_location(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["visited_location_ids"] = ["library", "hallway"]
    expanded = _expand_topic_question(state, valid_bible, "about hallway")
    assert "Hallway" in expanded
    assert "Were you there" in expanded


def test_expand_passes_through_unresolved_topic(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    original = "about the weather"
    assert _expand_topic_question(state, valid_bible, original) == original


def test_expand_passes_through_non_about_question(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    original = "Where were you at the time of the murder?"
    assert _expand_topic_question(state, valid_bible, original) == original


def test_expand_handles_article_prefixes(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    state["revealed_clue_ids"] = ["muddy_boots"]
    expanded = _expand_topic_question(state, valid_bible, "about the boots")
    assert "muddy boots" in expanded.lower()


def test_expand_does_not_resolve_unrevealed_clue(valid_bible: CaseBible) -> None:
    """A clue the player hasn't found shouldn't expand — it would leak knowledge."""
    state = initial_state(valid_bible)  # nothing revealed
    original = "about muddy_boots"
    assert _expand_topic_question(state, valid_bible, original) == original

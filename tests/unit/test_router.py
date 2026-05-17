from __future__ import annotations

import pytest

from mystery.graph.router import ParseError, parse_action
from mystery.graph.state import (
    AccuseAction,
    ExamineAction,
    HelpAction,
    InterrogateAction,
    MoveAction,
    NotebookAction,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("move library", MoveAction(location_id="library")),
        ("go library", MoveAction(location_id="library")),
        ("examine", ExamineAction()),
        ("look", ExamineAction()),
        ("notes", NotebookAction()),
        ("n", NotebookAction()),
        ("accuse butler", AccuseAction(suspect_id="butler")),
        ("help", HelpAction()),
        ("?", HelpAction()),
    ],
)
def test_parses_simple_commands(text: str, expected: object) -> None:
    assert parse_action(text) == expected


def test_parses_interrogate_with_multiword_question() -> None:
    result = parse_action("ask butler where were you last night")
    assert isinstance(result, InterrogateAction)
    assert result.suspect_id == "butler"
    assert result.question == "where were you last night"


def test_verbs_are_case_insensitive() -> None:
    assert parse_action("MOVE library") == MoveAction(location_id="library")


def test_location_id_keeps_case() -> None:
    """Bibles use lowercase ids, but the parser must not pre-normalize — the
    tool decides whether the id resolves."""
    result = parse_action("move Library")
    assert isinstance(result, MoveAction)
    assert result.location_id == "Library"


@pytest.mark.parametrize(
    ("text", "expected_substring"),
    [
        ("", "Empty"),
        ("   ", "Empty"),
        ("teleport library", "Unknown command"),
        ("move", "Usage: move"),
        ("ask butler", "Usage: ask"),
        ("accuse", "Usage: accuse"),
    ],
)
def test_returns_parse_error_for_bad_input(text: str, expected_substring: str) -> None:
    result = parse_action(text)
    assert isinstance(result, ParseError)
    assert expected_substring in result.message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Verb synonyms an LLM is likely to use.
        ("investigate", ExamineAction()),
        ("inspect", ExamineAction()),
        (
            "question butler about it",
            InterrogateAction(suspect_id="butler", question="about it"),
        ),
        (
            "interview butler what time",
            InterrogateAction(suspect_id="butler", question="what time"),
        ),
        ("walk library", MoveAction(location_id="library")),
        # Filler words after the verb.
        ("move to library", MoveAction(location_id="library")),
        ("go to the library", MoveAction(location_id="library")),
        ("walk into the garden", MoveAction(location_id="garden")),
        ("accuse the butler", AccuseAction(suspect_id="butler")),
        # Markdown wrappers and leading list markers.
        ("**examine**", ExamineAction()),
        ("- examine", ExamineAction()),
        ("1. examine", ExamineAction()),
        ("`examine`", ExamineAction()),
        # Quoted/punctuated arguments.
        ("examine.", ExamineAction()),
        ("examine\n", ExamineAction()),
        ("accuse butler.", AccuseAction(suspect_id="butler")),
    ],
)
def test_tolerates_llm_formatting_noise(text: str, expected: object) -> None:
    """The detective LLM emits markdown, fillers, and punctuation. Accept them."""
    assert parse_action(text) == expected


def test_interrogate_strips_quotes_from_suspect_and_question() -> None:
    """LLMs love to wrap things in quotes — strip them so the suspect id resolves."""
    result = parse_action('ask "butler" "where were you"')
    assert isinstance(result, InterrogateAction)
    assert result.suspect_id == "butler"
    assert result.question == "where were you"


def test_interrogate_strips_colon_after_suspect_id() -> None:
    """'ask butler: where were you' is a natural LLM format. Don't break on it."""
    result = parse_action("ask butler: where were you")
    assert isinstance(result, InterrogateAction)
    assert result.suspect_id == "butler"
    assert result.question == "where were you"

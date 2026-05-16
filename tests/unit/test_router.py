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

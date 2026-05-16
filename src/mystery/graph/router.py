"""Deterministic parser from raw player input to typed Actions.

A deliberate non-LLM choice: a regex/split parser is faster, free, and the
command surface is small enough that natural-language understanding would
buy nothing here. The only LLM call per turn is inside the suspect agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from mystery.graph.state import (
    AccuseAction,
    Action,
    ExamineAction,
    HelpAction,
    InterrogateAction,
    MoveAction,
    NotebookAction,
)


@dataclass(frozen=True)
class ParseError:
    """Soft error: bad input. The REPL prints ``message`` and re-prompts."""

    message: str


_MOVE_VERBS = {"move", "go", "goto"}
_INTERROGATE_VERBS = {"ask", "interrogate", "talk"}
_EXAMINE_VERBS = {"examine", "look", "search"}
_NOTEBOOK_VERBS = {"notebook", "notes", "n"}
_ACCUSE_VERBS = {"accuse"}
_HELP_VERBS = {"help", "?", "h"}

HELP_TEXT = (
    "Commands:\n"
    "  move <location>           move to a connected location\n"
    "  ask <suspect> <question>  interrogate a suspect\n"
    "  examine                   look around the current location for clues\n"
    "  notes                     show your notebook\n"
    "  accuse <suspect>          end the game by naming the killer\n"
    "  help                      show this message"
)


def parse_action(user_input: str) -> Action | ParseError:
    """Parse one line of player input. Verbs are case-insensitive; args are not."""
    raw = user_input.strip()
    if not raw:
        return ParseError("Empty input. Type 'help' for commands.")

    parts = raw.split(maxsplit=2)
    verb = parts[0].lower()
    rest = parts[1:]

    if verb in _MOVE_VERBS:
        if len(rest) < 1:
            return ParseError("Usage: move <location>")
        return MoveAction(location_id=rest[0])

    if verb in _INTERROGATE_VERBS:
        if len(rest) < 2:
            return ParseError("Usage: ask <suspect> <question>")
        return InterrogateAction(suspect_id=rest[0], question=rest[1])

    if verb in _EXAMINE_VERBS:
        return ExamineAction()

    if verb in _NOTEBOOK_VERBS:
        return NotebookAction()

    if verb in _ACCUSE_VERBS:
        if len(rest) < 1:
            return ParseError("Usage: accuse <suspect>")
        return AccuseAction(suspect_id=rest[0])

    if verb in _HELP_VERBS:
        return HelpAction()

    return ParseError(f"Unknown command {verb!r}. Type 'help' for commands.")

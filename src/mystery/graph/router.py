"""Deterministic parser from raw player input to typed Actions.

A deliberate non-LLM choice: a regex/split parser is faster, free, and the
command surface is small enough that natural-language understanding would
buy nothing here. The only LLM call per turn is inside the suspect agent.

Tolerance vs. strictness: the parser is strict about *verbs* (you must use
one of the listed words) but lenient about *formatting* — it strips markdown
wrappers, surrounding quotes/punctuation, and filler words like "to"/"the"
that LLM-produced commands routinely include.
"""

from __future__ import annotations

from dataclasses import dataclass

from mystery.graph.state import (
    AccuseAction,
    Action,
    AnalyzeAction,
    ExamineAction,
    HelpAction,
    InterrogateAction,
    LocationsAction,
    MoveAction,
    NotebookAction,
    ShowAction,
    SuspectsAction,
)


@dataclass(frozen=True)
class ParseError:
    """Soft error: bad input. The REPL prints ``message`` and re-prompts."""

    message: str


_MOVE_VERBS = {"move", "go", "goto", "walk"}
_INTERROGATE_VERBS = {"ask", "interrogate", "talk", "question", "interview"}
_EXAMINE_VERBS = {"examine", "look", "search", "investigate", "inspect"}
_ANALYZE_VERBS = {"analyze", "analyse", "forensic", "forensics"}
_NOTEBOOK_VERBS = {"notebook", "notes", "n"}
_ACCUSE_VERBS = {"accuse"}
_SHOW_VERBS = {"show", "present", "confront"}
_HELP_VERBS = {"help", "?", "h"}
_SUSPECTS_VERBS = {"suspects", "who", "people"}
_LOCATIONS_VERBS = {"locations", "map", "where", "exits"}

# Filler words after verb that should be dropped, e.g. "move to library".
_FILLER_AFTER_VERB = {"to", "the", "into", "toward", "towards", "at", "with"}

# Characters stripped from individual argument tokens (quotes, punctuation,
# trailing colon/comma the LLM tends to attach).
_ARG_STRIP = "*`'\"“”‘’.,:;!?()[]{}<>"  # noqa: RUF001 — intentional unicode quotes

# Wrappers stripped from the entire input before tokenisation (markdown bold,
# leading bullet/numbering, trailing newlines).
_INPUT_STRIP = "*_`#> \t\n\r"

HELP_TEXT = (
    "Commands (synonyms shown in parens):\n"
    "  move <location>           (go/goto/walk) move to a connected location\n"
    "  ask <suspect> <question>  (question/interview/talk) interrogate a suspect\n"
    "  examine                   (look/search/investigate/inspect) look for clues\n"
    "  examine victim            in the death room, perform a forensic look at the body\n"
    "  analyze <clue>            (forensics) drill into a revealed clue\n"
    "  show <suspect> <clue>     (present/confront) confront a suspect with a clue;\n"
    "                            clue can be its id or any word from its description\n"
    "  notes                     show your notebook\n"
    "  suspects                  (who) list everyone you can interrogate\n"
    "  locations                 (map/where) show this room and where you can go\n"
    "  accuse <suspect>          end the game by naming the killer\n"
    "  help                      show this message"
)


def _clean_arg(token: str) -> str:
    """Strip surrounding punctuation/quotes/markdown from one argument token."""
    return token.strip(_ARG_STRIP)


def _strip_leading_filler(tokens: list[str]) -> list[str]:
    """Drop leading filler words like 'to' / 'the' that come after a verb."""
    i = 0
    while i < len(tokens) and tokens[i].lower() in _FILLER_AFTER_VERB:
        i += 1
    return tokens[i:]


def _split_verb(raw: str) -> tuple[str, list[str]] | None:
    """Extract a verb token from the (already-pre-cleaned) input."""
    raw = raw.strip(_INPUT_STRIP)
    if not raw:
        return None
    # Strip a leading list marker like "1." or "-" that some LLMs emit.
    head, _, rest = raw.partition(" ")
    if head.rstrip(".)").isdigit() or head in {"-", "*", "•"}:
        raw = rest.lstrip()
    parts = raw.split(maxsplit=2)
    if not parts:
        return None
    # Strip markdown/quotes from the verb but preserve "?" since it IS a verb.
    stripped = parts[0].lower().strip("*`'\"“”‘’.,:;!()[]{}<>")  # noqa: RUF001
    verb = stripped or parts[0].lower()
    return verb, parts[1:]


def parse_action(user_input: str) -> Action | ParseError:
    """Parse one line of player input. Verbs are case-insensitive; args are not.

    Tolerates LLM-style noise (markdown wrappers, surrounding quotes, filler
    words after verbs, trailing punctuation) without devolving into NLU.
    """
    if not user_input.strip():
        return ParseError("Empty input. Type 'help' for commands.")

    head = _split_verb(user_input)
    if head is None:
        return ParseError("Empty input. Type 'help' for commands.")
    verb, rest = head
    rest = _strip_leading_filler(rest)

    if verb in _MOVE_VERBS:
        # Re-split fully — "to the library" must collapse to "library", not
        # leave a multi-word token from the maxsplit=2 of _split_verb.
        all_args = _strip_leading_filler(" ".join(rest).split())
        if not all_args:
            return ParseError("Usage: move <location>")
        location_id = _clean_arg(all_args[0])
        if not location_id:
            return ParseError("Usage: move <location>")
        return MoveAction(location_id=location_id)

    if verb in _INTERROGATE_VERBS:
        if len(rest) < 2:
            return ParseError("Usage: ask <suspect> <question>")
        suspect_id = _clean_arg(rest[0])
        question = rest[1].strip().strip(_ARG_STRIP)
        if not suspect_id or not question:
            return ParseError("Usage: ask <suspect> <question>")
        return InterrogateAction(suspect_id=suspect_id, question=question)

    if verb in _EXAMINE_VERBS:
        # Optional target — "examine victim" goes to the forensics branch in
        # apply_examine. No target → legacy room sweep.
        all_args = _strip_leading_filler(" ".join(rest).split())
        if not all_args:
            return ExamineAction()
        target = _clean_arg(all_args[0]).lower()
        return ExamineAction(target=target or None)

    if verb in _ANALYZE_VERBS:
        all_args = _strip_leading_filler(" ".join(rest).split())
        if not all_args:
            return ParseError("Usage: analyze <clue>")
        clue_id = " ".join(_clean_arg(t) for t in all_args if _clean_arg(t)).strip()
        if not clue_id:
            return ParseError("Usage: analyze <clue>")
        return AnalyzeAction(clue_id=clue_id)

    if verb in _NOTEBOOK_VERBS:
        return NotebookAction()

    if verb in _ACCUSE_VERBS:
        all_args = _strip_leading_filler(" ".join(rest).split())
        if not all_args:
            return ParseError("Usage: accuse <suspect>")
        suspect_id = _clean_arg(all_args[0])
        if not suspect_id:
            return ParseError("Usage: accuse <suspect>")
        return AccuseAction(suspect_id=suspect_id)

    if verb in _SHOW_VERBS:
        all_args = _strip_leading_filler(" ".join(rest).split())
        if not all_args:
            return ParseError("Usage: show <suspect> <clue>")
        suspect_id = _clean_arg(all_args[0])
        # Strip filler again between the two args so "show butler the boots"
        # collapses to (butler, boots) — LLMs often slot articles in there.
        remaining = _strip_leading_filler(all_args[1:])
        if not remaining:
            return ParseError("Usage: show <suspect> <clue>")
        # Preserve the rest of the line as a multi-word clue reference so
        # players can type "show butler torn letter" or "show butler the
        # muddy boots" — resolve_clue handles the token matching.
        clue_id = " ".join(_clean_arg(t) for t in remaining if _clean_arg(t)).strip()
        if not suspect_id or not clue_id:
            return ParseError("Usage: show <suspect> <clue>")
        return ShowAction(suspect_id=suspect_id, clue_id=clue_id)

    if verb in _HELP_VERBS:
        return HelpAction()

    if verb in _SUSPECTS_VERBS:
        return SuspectsAction()

    if verb in _LOCATIONS_VERBS:
        return LocationsAction()

    return ParseError(f"Unknown command {verb!r}. Type 'help' for commands.")

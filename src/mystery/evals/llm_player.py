"""LLM-driven detective playtester.

Unlike :mod:`mystery.evals.optimal_player`, which DFSes the bible to win by
construction, this player only sees what the game would show a human: the
current room, the visible exits, the cast of suspects, the notebook, and the
last tool output. It then has to *decide* what to do next via an LLM call.

That makes it the only test surface that exercises the full chain:

* the router's tolerance to natural-language commands the LLM emits,
* the observation we render each turn (does it contain enough to play?),
* the suspect agent's grounding under non-canned questions,
* whether a generated case is solvable without privileged knowledge.

Bugs that show up here cannot be caught by canned ``FakeListChatModel`` tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from langchain_core.messages import HumanMessage, SystemMessage

from mystery.graph.game import build_game_graph
from mystery.graph.router import ParseError, parse_action
from mystery.graph.state import GameState, initial_state

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.graph.state import Action
    from mystery.models import CaseBible


_DETECTIVE_SYSTEM = """\
You are the player in a text-based detective game. Your goal is to identify
the killer and accuse them. You must output exactly ONE command per turn,
no commentary, no markdown, no quotes — just the raw command line.

Available commands (you MUST use these exact verbs):
  move <location_id>            move to a connected location
  ask <suspect_id> <question>   interrogate a suspect (question can be many words)
  examine                       look around the current location for clues
  notes                         show your notebook (free action)
  accuse <suspect_id>           end the game by naming the killer

Strategy:
- Examine every reachable location to discover clues.
- Interrogate every suspect about their whereabouts, the victim, each other.
- When you are confident, accuse exactly one suspect. The game then ends.
- Use ONLY the ids shown in the observation (snake_case). Do not invent ids.

Output format: one command line. Nothing else.
"""


@dataclass
class PlaytestStep:
    turn: int
    observation: str
    raw_command: str
    parsed_kind: str  # action kind, or "parse_error"
    output: str


@dataclass
class PlaytestReport:
    seed: int
    success: bool
    turns_used: int
    accused: str
    actual_killer: str
    parse_errors: int
    repeated_actions: int
    steps: list[PlaytestStep] = field(default_factory=list)

    def as_text(self) -> str:
        head = (
            f"seed={self.seed} success={self.success} turns={self.turns_used} "
            f"accused={self.accused or '(none)'} actual={self.actual_killer} "
            f"parse_errors={self.parse_errors} repeats={self.repeated_actions}"
        )
        body = "\n".join(
            f"  t{s.turn} [{s.parsed_kind}] {s.raw_command!r}\n"
            f"      -> {s.output.splitlines()[0][:120] if s.output else ''}"
            for s in self.steps
        )
        return f"{head}\n{body}"


def render_observation(state: GameState, bible: CaseBible) -> str:
    """Render the textual state the LLM detective sees each turn.

    Deliberately bible-redacted: it shows what a player at this state would
    legitimately know (current room, exits, suspect roster, revealed clues,
    last tool output). It does NOT leak the killer, deception policies,
    motives, alibi truth values, or the canonical timeline.
    """
    here = next(loc for loc in bible.locations if loc.id == state["current_location_id"])
    exits = ", ".join(here.connected_location_ids) or "(none)"
    suspect_ids = ", ".join(s.id for s in bible.suspects)
    revealed = ", ".join(state["revealed_clue_ids"]) or "(none yet)"
    notes_tail = state["notebook"][-6:] if state["notebook"] else []
    notes_block = "\n".join(f"  - {line}" for line in notes_tail) or "  (empty)"
    last = state["last_output"] or "(none)"

    return (
        f"VICTIM: {bible.victim.name} ({bible.victim.role}), "
        f"found in {bible.victim.location_of_death_id} at t={bible.victim.time_of_death}.\n"
        f"YOU ARE IN: {here.id} — {here.name}. {here.description}\n"
        f"EXITS: {exits}\n"
        f"SUSPECTS (use these ids): {suspect_ids}\n"
        f"CLUES FOUND: {revealed}\n"
        f"NOTEBOOK (last entries):\n{notes_block}\n"
        f"LAST RESULT: {last}\n"
        f"TURN: {state['turn_count']}\n"
        f"\nWhat is your next command?"
    )


def _ask_detective(chat_model: BaseChatModel, observation: str) -> str:
    messages = [
        SystemMessage(content=_DETECTIVE_SYSTEM),
        HumanMessage(content=observation),
    ]
    response = chat_model.invoke(messages)
    raw = str(response.content).strip()
    # Take only the first non-empty line, in case the model can't help itself.
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "//", "<")):
            # Strip surrounding quotes / backticks if any.
            return line.strip("`'\" ")
    return raw


def play_with_llm(
    bible: CaseBible,
    vectorstore: Chroma,
    suspect_chat_model: BaseChatModel,
    detective_chat_model: BaseChatModel,
    *,
    max_turns: int = 60,
) -> PlaytestReport:
    """Run a full LLM-vs-LLM playtest. Returns a structured report."""
    graph = build_game_graph(bible, vectorstore, suspect_chat_model)
    state = initial_state(bible)
    steps: list[PlaytestStep] = []
    parse_errors = 0
    repeated_actions = 0
    last_raw: str | None = None
    consecutive_parse_errors = 0

    while not state["done"] and state["turn_count"] < max_turns:
        observation = render_observation(state, bible)
        raw = _ask_detective(detective_chat_model, observation)

        if raw == last_raw:
            repeated_actions += 1
        last_raw = raw

        parsed: Action | ParseError = parse_action(raw)
        if isinstance(parsed, ParseError):
            parse_errors += 1
            consecutive_parse_errors += 1
            steps.append(
                PlaytestStep(
                    turn=state["turn_count"],
                    observation=observation,
                    raw_command=raw,
                    parsed_kind="parse_error",
                    output=parsed.message,
                ),
            )
            state = cast(
                "GameState",
                {**state, "last_output": f"PARSE ERROR: {parsed.message}"},
            )
            if consecutive_parse_errors >= 5:
                # Detective is hopelessly confused; abort to keep eval bounded.
                break
            continue

        consecutive_parse_errors = 0
        state["pending_action"] = parsed
        state = cast("GameState", graph.invoke(state))
        steps.append(
            PlaytestStep(
                turn=state["turn_count"],
                observation=observation,
                raw_command=raw,
                parsed_kind=parsed.kind,
                output=state["last_output"],
            ),
        )

    accusation = state["accusation"]
    return PlaytestReport(
        seed=bible.seed,
        success=bool(accusation and accusation.correct),
        turns_used=state["turn_count"],
        accused=accusation.accused_id if accusation else "",
        actual_killer=bible.killer_id,
        parse_errors=parse_errors,
        repeated_actions=repeated_actions,
        steps=steps,
    )

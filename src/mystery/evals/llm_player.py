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

Strategy (follow in order):
1. If the current room is NOT YET examined, your next command is `examine`.
2. If there are still rooms in ROOMS NOT YET SEARCHED, your next command is
   `move <one of them>` (must be adjacent — use EXITS).
3. Once every room is searched, interrogate each suspect about the victim,
   their whereabouts, and the specific clues you have collected.
4. When the evidence points clearly to one suspect, `accuse <suspect_id>`.
   The game ends instantly on accusation, right or wrong, so don't guess —
   but DO accuse before your turns run out; an indecisive detective loses.

Hard rules:
- NEVER repeat the same command twice in a row. If LAST RESULT says you have
  already catalogued every clue here, you must MOVE somewhere else or `ask`
  a suspect — do not `examine` again.
- Use ONLY the ids shown in the observation (snake_case). Do not invent ids.
- Write questions as natural English sentences, not snake_case_strings.

Output format: ONE command line. Nothing else.
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


def render_observation(state: GameState, bible: CaseBible, *, max_turns: int | None = None) -> str:
    """Render the textual state the LLM detective sees each turn.

    Deliberately bible-redacted: it shows what a player at this state would
    legitimately know (current room, exits, suspect roster, revealed clues,
    last tool output). It does NOT leak the killer, deception policies,
    motives, alibi truth values, or the canonical timeline.

    Surfaces *which* rooms have not yet been visited so a confused detective
    is nudged to explore — early playtests showed the LLM happily looping on
    `examine` in a fully-searched room without trying other locations.
    """
    here = next(loc for loc in bible.locations if loc.id == state["current_location_id"])
    exits = ", ".join(here.connected_location_ids) or "(none)"
    suspect_ids = ", ".join(s.id for s in bible.suspects)
    revealed = ", ".join(state["revealed_clue_ids"]) or "(none yet)"
    notes_block = (
        "\n".join(f"  - {line}" for line in state["notebook"]) if state["notebook"] else "  (empty)"
    )
    last = state["last_output"] or "(none)"
    all_locs = {loc.id for loc in bible.locations}
    unexamined = sorted(all_locs - set(state["examined_location_ids"]))
    unexamined_str = ", ".join(unexamined) or "(none — you have searched everywhere)"
    here_examined = "yes" if here.id in state["examined_location_ids"] else "NO, not yet"

    return (
        f"VICTIM: {bible.victim.name} ({bible.victim.role}), "
        f"found in {bible.victim.location_of_death_id} at t={bible.victim.time_of_death}.\n"
        f"YOU ARE IN: {here.id} — {here.name}. {here.description}\n"
        f"  examined this room? {here_examined}\n"
        f"EXITS: {exits}\n"
        f"SUSPECTS (use these ids): {suspect_ids}\n"
        f"CLUES FOUND: {revealed}\n"
        f"ROOMS NOT YET SEARCHED (need `examine` after moving): {unexamined_str}\n"
        f"NOTEBOOK:\n{notes_block}\n"
        f"LAST RESULT: {last}\n"
        f"TURN: {state['turn_count']}"
        f"{f' of {max_turns} (commit soon!)' if max_turns else ''}\n"
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


def _progress_score(state: GameState) -> int:
    """A monotonic 'detective is making progress' counter.

    Increases when the player examines a new room or reveals a new clue.
    Used as the no-progress abort signal: a small LLM detective will happily
    shuttle between two rooms without ever examining, and consecutive-repeat
    detection misses alternating cycles.
    """
    return len(state["examined_location_ids"]) + len(state["revealed_clue_ids"])


def play_with_llm(
    bible: CaseBible,
    vectorstore: Chroma,
    suspect_chat_model: BaseChatModel,
    detective_chat_model: BaseChatModel,
    *,
    max_turns: int = 60,
    max_consecutive_repeats: int = 4,
    no_progress_window: int = 12,
) -> PlaytestReport:
    """Run a full LLM-vs-LLM playtest. Returns a structured report.

    ``max_consecutive_repeats`` is a guardrail: the LLM detective will
    occasionally lock into a loop (e.g. re-examining a searched room) and
    consume every remaining turn. Once that many *consecutive identical*
    commands have been issued, abort and mark the case unsolved — that's a
    real failure to report, not an excuse to keep spending tokens.

    ``no_progress_window`` is a second guardrail for alternating-cycle loops
    (e.g. move A → move B → move A → …) that consecutive-repeat detection
    misses. If the progress score does not increase for that many turns,
    abort. Set high enough that legitimate "all rooms searched, now I'm
    interrogating" phases can run.
    """
    graph = build_game_graph(bible, vectorstore, suspect_chat_model)
    state = initial_state(bible)
    steps: list[PlaytestStep] = []
    parse_errors = 0
    repeated_actions = 0
    last_raw: str | None = None
    consecutive_repeats = 0
    consecutive_parse_errors = 0
    best_progress = _progress_score(state)
    turns_since_progress = 0

    while not state["done"] and state["turn_count"] < max_turns:
        observation = render_observation(state, bible, max_turns=max_turns)
        raw = _ask_detective(detective_chat_model, observation)

        if raw == last_raw:
            repeated_actions += 1
            consecutive_repeats += 1
        else:
            consecutive_repeats = 0
        last_raw = raw
        if consecutive_repeats >= max_consecutive_repeats:
            # Aborted loop: record it so the report is honest about why we stopped.
            steps.append(
                PlaytestStep(
                    turn=state["turn_count"],
                    observation=observation,
                    raw_command=raw,
                    parsed_kind="aborted_loop",
                    output=f"detective issued {raw!r} {consecutive_repeats + 1}x — abort",
                ),
            )
            break

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

        progress = _progress_score(state)
        if progress > best_progress:
            best_progress = progress
            turns_since_progress = 0
        else:
            turns_since_progress += 1
        if turns_since_progress >= no_progress_window and not state["done"]:
            steps.append(
                PlaytestStep(
                    turn=state["turn_count"],
                    observation="",
                    raw_command="",
                    parsed_kind="aborted_no_progress",
                    output=(
                        f"no new clues or examined rooms for {turns_since_progress} turns "
                        f"— aborting"
                    ),
                ),
            )
            break

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

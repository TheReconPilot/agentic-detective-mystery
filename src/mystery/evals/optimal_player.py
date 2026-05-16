"""A heuristic player that visits every location and accuses the killer.

This is *not* a fair player: it sees the bible and walks an optimal path. Its
purpose is to verify that generated cases are mechanically solvable — every
clue location reachable, the killer accusable, the game terminating. If
``play_to_solve`` ever fails, the generator produced a broken case.

It's also the spine of the difficulty eval: mean turn count over many cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from mystery.graph.game import build_game_graph
from mystery.graph.state import (
    AccuseAction,
    ExamineAction,
    GameState,
    MoveAction,
    initial_state,
)

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.graph.state import Action
    from mystery.models import CaseBible


@dataclass
class SolvabilityReport:
    """Outcome of a single optimal-player run."""

    seed: int
    success: bool
    turns: int
    accused: str
    actual_killer: str
    locations_visited: int
    clues_revealed: int
    transcript: list[str] = field(default_factory=list)


def plan_examine_all(bible: CaseBible) -> list[Action]:
    """Build an action plan that visits every location reachable from the start.

    DFS through the location graph: examine, descend, backtrack. The final
    action is a placeholder — the caller appends the accusation.
    """
    adj = {loc.id: list(loc.connected_location_ids) for loc in bible.locations}
    start = bible.victim.location_of_death_id

    plan: list[Action] = [ExamineAction()]
    visited: set[str] = {start}

    def dfs(current: str) -> None:
        for next_loc in adj.get(current, []):
            if next_loc in visited:
                continue
            visited.add(next_loc)
            plan.append(MoveAction(location_id=next_loc))
            plan.append(ExamineAction())
            dfs(next_loc)
            plan.append(MoveAction(location_id=current))

    dfs(start)
    return plan


def play_to_solve(
    bible: CaseBible,
    vectorstore: Chroma,
    chat_model: BaseChatModel,
    *,
    max_turns: int = 200,
) -> SolvabilityReport:
    """Drive the compiled game graph through an optimal sequence and accuse the killer."""
    graph = build_game_graph(bible, vectorstore, chat_model)
    state = initial_state(bible)
    transcript: list[str] = []

    plan = plan_examine_all(bible)
    plan.append(AccuseAction(suspect_id=bible.killer_id))

    for action in plan:
        if state["turn_count"] >= max_turns:
            break
        state["pending_action"] = action
        state = cast("GameState", graph.invoke(state))
        snippet = state["last_output"].splitlines()[0][:80] if state["last_output"] else ""
        transcript.append(f"{action.kind}: {snippet}")
        if state["done"]:
            break

    accusation = state["accusation"]
    return SolvabilityReport(
        seed=bible.seed,
        success=bool(accusation and accusation.correct),
        turns=state["turn_count"],
        accused=accusation.accused_id if accusation else "",
        actual_killer=bible.killer_id,
        locations_visited=len(state["visited_location_ids"]),
        clues_revealed=len(state["revealed_clue_ids"]),
        transcript=transcript,
    )

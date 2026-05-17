"""Compile the per-turn LangGraph.

Topology: one entry point, six leaf nodes (one per action kind), all leading
to END. The graph processes exactly one player input per invocation; the REPL
loop lives in :mod:`mystery.cli`.

This is a deliberately thin graph. Each node delegates to a pure
``apply_*`` function from :mod:`mystery.tools`, so the game logic is
testable without any LangGraph dependency. LangGraph's contribution here is
the explicit dispatch topology and the state-merge protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langgraph.graph import END, StateGraph

from mystery.graph.router import HELP_TEXT
from mystery.graph.state import (
    AccuseAction,
    GameState,
    InterrogateAction,
    MoveAction,
    ShowAction,
)
from mystery.tools.accuse import apply_accuse
from mystery.tools.examine import apply_examine
from mystery.tools.interrogate import apply_interrogate
from mystery.tools.move import apply_move
from mystery.tools.notebook import apply_notebook
from mystery.tools.show import apply_show

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from mystery.agents.commitments import CommitmentExtractor
    from mystery.models import CaseBible

_ACTION_NODES = ("move", "examine", "notebook", "accuse", "interrogate", "show", "help")


def _route(state: GameState) -> str:
    """Conditional entry point: dispatch on the action's discriminator field."""
    action = state["pending_action"]
    if action is None:
        msg = "graph invoked with no pending_action; the REPL must set one"
        raise RuntimeError(msg)
    return action.kind


def _make_move_node(bible: CaseBible) -> Callable[[GameState], dict[str, Any]]:
    def node(state: GameState) -> dict[str, Any]:
        action = state["pending_action"]
        assert isinstance(action, MoveAction)
        return apply_move(state, bible, action.location_id)

    return node


def _make_examine_node(bible: CaseBible) -> Callable[[GameState], dict[str, Any]]:
    def node(state: GameState) -> dict[str, Any]:
        return apply_examine(state, bible)

    return node


def _notebook_node(state: GameState) -> dict[str, Any]:
    return apply_notebook(state)


def _help_node(_state: GameState) -> dict[str, Any]:
    return {"last_output": HELP_TEXT}


def _make_accuse_node(bible: CaseBible) -> Callable[[GameState], dict[str, Any]]:
    def node(state: GameState) -> dict[str, Any]:
        action = state["pending_action"]
        assert isinstance(action, AccuseAction)
        return apply_accuse(state, bible, action.suspect_id)

    return node


def _make_interrogate_node(
    bible: CaseBible,
    vectorstore: Chroma,
    chat_model: BaseChatModel,
    commitment_extractor: CommitmentExtractor | None,
) -> Callable[[GameState], dict[str, Any]]:
    def node(state: GameState) -> dict[str, Any]:
        action = state["pending_action"]
        assert isinstance(action, InterrogateAction)
        return apply_interrogate(
            state,
            bible,
            vectorstore,
            chat_model,
            action.suspect_id,
            action.question,
            commitment_extractor=commitment_extractor,
        )

    return node


def _make_show_node(
    bible: CaseBible,
    vectorstore: Chroma,
    chat_model: BaseChatModel,
    commitment_extractor: CommitmentExtractor | None,
) -> Callable[[GameState], dict[str, Any]]:
    def node(state: GameState) -> dict[str, Any]:
        action = state["pending_action"]
        assert isinstance(action, ShowAction)
        return apply_show(
            state,
            bible,
            vectorstore,
            chat_model,
            action.suspect_id,
            action.clue_id,
            commitment_extractor=commitment_extractor,
        )

    return node


def build_game_graph(
    bible: CaseBible,
    vectorstore: Chroma,
    chat_model: BaseChatModel,
    commitment_extractor: CommitmentExtractor | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the per-turn dispatcher graph.

    ``StateGraph`` is cast to ``Any`` so the heavily-overloaded ``add_node``
    signature stops fighting our plain ``Callable[[GameState], dict[str, Any]]``
    node functions. The dispatch shape is exercised by the integration tests
    in ``tests/integration/test_game_loop.py``.
    """
    builder = cast("Any", StateGraph(GameState))

    builder.add_node("move", _make_move_node(bible))
    builder.add_node("examine", _make_examine_node(bible))
    builder.add_node("notebook", _notebook_node)
    builder.add_node("accuse", _make_accuse_node(bible))
    builder.add_node(
        "interrogate",
        _make_interrogate_node(bible, vectorstore, chat_model, commitment_extractor),
    )
    builder.add_node(
        "show",
        _make_show_node(bible, vectorstore, chat_model, commitment_extractor),
    )
    builder.add_node("help", _help_node)

    builder.set_conditional_entry_point(_route, {n: n for n in _ACTION_NODES})

    for node_name in _ACTION_NODES:
        builder.add_edge(node_name, END)

    compiled: CompiledStateGraph[Any, Any, Any, Any] = builder.compile()
    return compiled

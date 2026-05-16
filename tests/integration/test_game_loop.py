"""End-to-end: a scripted player solves the bundled case via the compiled graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.graph.game import build_game_graph
from mystery.graph.state import (
    AccuseAction,
    ExamineAction,
    GameState,
    InterrogateAction,
    MoveAction,
    NotebookAction,
    initial_state,
)
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index

if TYPE_CHECKING:
    from mystery.graph.state import Action
    from mystery.models import CaseBible


def _step(graph: Any, state: GameState, action: Action) -> GameState:
    state = cast("GameState", {**state, "pending_action": action})
    return cast("GameState", graph.invoke(state))


def test_scripted_player_solves_the_case(valid_bible: CaseBible) -> None:
    """Walk a player through examining, interrogating, and accusing — and win.

    This is the spine of what M6's optimal-player eval will automate: visit
    every location, examine, talk to everyone, accuse the killer. Here we
    use a canned chat response — the eval will use a real LLM.
    """
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(
        responses=["I was polishing silver in the pantry, sir. I saw nothing."],
    )
    graph = build_game_graph(valid_bible, vectorstore, chat)

    state = initial_state(valid_bible)

    # 1. Examine the library (where the body is) — finds the torn letter.
    state = _step(graph, state, ExamineAction())
    assert "torn_letter" in state["revealed_clue_ids"]

    # 2. Move to the hallway.
    state = _step(graph, state, MoveAction(location_id="hallway"))
    assert state["current_location_id"] == "hallway"

    # 3. Examine the hallway — finds the muddy boots.
    state = _step(graph, state, ExamineAction())
    assert "muddy_boots" in state["revealed_clue_ids"]

    # 4. Notebook is a free action: no turn cost, all clues catalogued.
    pre_turn = state["turn_count"]
    state = _step(graph, state, NotebookAction())
    assert state["turn_count"] == pre_turn
    assert "muddy_boots" in state["last_output"]
    assert "torn_letter" in state["last_output"]

    # 5. Interrogate the butler.
    state = _step(graph, state, InterrogateAction(suspect_id="butler", question="alibi?"))
    assert "Hodges" in state["last_output"]
    assert "polishing silver" in state["last_output"]

    # 6. Accuse the butler. Win.
    state = _step(graph, state, AccuseAction(suspect_id="butler"))
    assert state["done"] is True
    assert state["accusation"] is not None
    assert state["accusation"].correct is True


def test_wrong_accusation_ends_game_with_loss(valid_bible: CaseBible) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["unused"])
    graph = build_game_graph(valid_bible, vectorstore, chat)

    state = initial_state(valid_bible)
    state = _step(graph, state, AccuseAction(suspect_id="niece"))

    assert state["done"] is True
    assert state["accusation"] is not None
    assert state["accusation"].correct is False
    assert state["accusation"].actual_killer_id == "butler"


def test_unknown_suspect_accusation_does_not_end_game(valid_bible: CaseBible) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["unused"])
    graph = build_game_graph(valid_bible, vectorstore, chat)

    state = initial_state(valid_bible)
    state = _step(graph, state, AccuseAction(suspect_id="phantom"))

    assert state["done"] is False
    assert state["accusation"] is None
    assert "withdrawn" in state["last_output"]


def test_full_thorough_investigation(valid_bible: CaseBible) -> None:
    """A long, methodical game: visit every location, interrogate every suspect,
    examine everywhere, then accuse. Mirrors the optimal-player eval (M6)."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(
        responses=[f"answer-{i}" for i in range(20)],  # plenty of canned replies
    )
    graph = build_game_graph(valid_bible, vectorstore, chat)
    state = initial_state(valid_bible)

    # Examine the starting location.
    state = _step(graph, state, ExamineAction())
    # Walk the location graph: library → hallway → garden → hallway → library.
    state = _step(graph, state, MoveAction(location_id="hallway"))
    state = _step(graph, state, ExamineAction())
    state = _step(graph, state, MoveAction(location_id="garden"))
    state = _step(graph, state, ExamineAction())  # empty room
    state = _step(graph, state, MoveAction(location_id="hallway"))
    state = _step(graph, state, MoveAction(location_id="library"))

    # Interrogate every suspect.
    for suspect in valid_bible.suspects:
        state = _step(
            graph,
            state,
            InterrogateAction(suspect_id=suspect.id, question="alibi?"),
        )

    # All revealed clues should be in the notebook.
    for clue in valid_bible.clues:
        assert clue.id in state["revealed_clue_ids"]
    # Every location visited.
    assert sorted(state["visited_location_ids"]) == sorted(loc.id for loc in valid_bible.locations)

    # Finally accuse — and win.
    state = _step(graph, state, AccuseAction(suspect_id=valid_bible.killer_id))
    assert state["done"] is True
    assert state["accusation"] is not None
    assert state["accusation"].correct is True

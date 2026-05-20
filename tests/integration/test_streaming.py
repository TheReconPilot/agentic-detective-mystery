"""Streaming path: when a callback is set, chunks land in the callback and
``last_output`` is suppressed so the REPL doesn't double-print."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.graph.game import build_game_graph
from mystery.graph.state import GameState, InterrogateAction, initial_state
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index
from mystery.tools.interrogate import apply_interrogate

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_apply_interrogate_streams_chunks_to_callback(valid_bible: CaseBible) -> None:
    """When a stream_callback is set the LLM reply is forwarded to it and
    ``last_output`` is emptied so the CLI's after-graph print is a no-op."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    # FakeListChatModel.stream() emits the response one character at a time —
    # that's enough to verify chunks are being forwarded incrementally.
    chat = FakeListChatModel(responses=["I was nowhere near."])

    chunks: list[str] = []
    state = initial_state(valid_bible)
    update = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        chat,
        suspect_id="butler",
        question="alibi?",
        stream_callback=chunks.append,
    )

    assert update["last_output"] == ""
    joined = "".join(chunks)
    # The wrapper writes a "thinking…" indicator first, then erases it and
    # paints the speaker prefix once the first chunk arrives. We assert the
    # *visible* result rather than the raw chunk sequence so the test isn't
    # over-coupled to ANSI escape choices.
    assert "Mr. Hodges is thinking" in joined
    assert "Mr. Hodges: " in joined
    assert "I was nowhere near." in joined
    assert joined.endswith("\n")
    assert update["turn_count"] == state["turn_count"] + 1


def test_graph_routes_stream_callback_through_interrogate(
    valid_bible: CaseBible,
) -> None:
    """build_game_graph captures the callback and the interrogate node uses it."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["streamed reply"])
    chunks: list[str] = []
    graph = build_game_graph(
        valid_bible,
        vectorstore,
        chat,
        stream_callback=chunks.append,
    )

    state = initial_state(valid_bible)
    state = cast(
        "GameState",
        {
            **state,
            "pending_action": InterrogateAction(suspect_id="butler", question="alibi?"),
        },
    )
    state = cast("GameState", graph.invoke(cast("dict[str, Any]", state)))

    assert state["last_output"] == ""
    assert "streamed reply" in "".join(chunks)


def test_apply_interrogate_without_callback_preserves_last_output(
    valid_bible: CaseBible,
) -> None:
    """Regression: existing tests that depend on ``last_output`` must keep working."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["I plead the fifth."])

    state = initial_state(valid_bible)
    update = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        chat,
        suspect_id="butler",
        question="alibi?",
    )
    assert "plead the fifth" in update["last_output"]
    assert update["last_output"].startswith("Mr. Hodges: ")

"""Integration: `show` dispatches through the graph and carries commitments forward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.graph.game import build_game_graph
from mystery.graph.state import (
    ExamineAction,
    GameState,
    InterrogateAction,
    ShowAction,
    initial_state,
)
from mystery.models import Commitment
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index

if TYPE_CHECKING:
    from mystery.models import CaseBible, Suspect


class _RecordingChat:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.system_prompts: list[str] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.system_prompts.append(str(messages[0].content))

        class _Resp:
            content: str

        r = _Resp()
        r.content = self._replies.pop(0)
        return r


class _StaticExtractor:
    def __init__(self, commitments: list[Commitment]) -> None:
        self._commitments = list(commitments)

    def extract(self, suspect: Suspect, question: str, answer: str) -> Commitment | None:
        del suspect, question, answer
        if not self._commitments:
            return None
        return self._commitments.pop(0)


def test_show_through_graph_renders_clue_and_extracts_commitment(
    valid_bible: CaseBible,
) -> None:
    """Examine → ask → show; the show prompt must contain both the prior commitment
    AND the confronting clue description, and a new commitment is appended."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = _RecordingChat(
        replies=[
            "I was nowhere near the library, sir.",  # ask
            "T-the letter is not mine — I demand you put it down!",  # show
        ],
    )
    extractor = _StaticExtractor(
        [
            Commitment(summary="They claimed never to have been in the library."),
            Commitment(summary="When shown the letter, they denied ownership in panic."),
        ],
    )

    graph = build_game_graph(valid_bible, vectorstore, cast("Any", chat), extractor)
    state = initial_state(valid_bible)

    # Step 1: examine the library so torn_letter is revealed.
    state["pending_action"] = ExamineAction()
    state = cast("GameState", graph.invoke(state))
    assert "torn_letter" in state["revealed_clue_ids"]

    # Step 2: ask the butler something — establishes a commitment.
    state["pending_action"] = InterrogateAction(
        suspect_id="butler",
        question="Where were you?",
    )
    state = cast("GameState", graph.invoke(state))

    # Step 3: show the butler the torn letter.
    state["pending_action"] = ShowAction(suspect_id="butler", clue_id="torn_letter")
    state = cast("GameState", graph.invoke(state))

    # The show-turn system prompt must include both blocks.
    show_prompt = chat.system_prompts[-1]
    assert "holding up a piece of evidence" in show_prompt
    assert "torn letter" in show_prompt.lower()
    assert "previously told this detective" in show_prompt
    assert "never to have been in the library" in show_prompt

    # Two commitments now in state: one from `ask`, one from `show`.
    butler_commitments = state["suspect_commitments"]["butler"]
    assert len(butler_commitments) == 2
    assert butler_commitments[-1].summary.startswith("When shown the letter")


def test_show_unrevealed_clue_via_graph_is_a_soft_refusal(valid_bible: CaseBible) -> None:
    """The graph path returns a soft error message, not an exception."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["never called"])

    graph = build_game_graph(valid_bible, vectorstore, chat)
    state = initial_state(valid_bible)
    # No examine — torn_letter is not in revealed_clue_ids.
    state["pending_action"] = ShowAction(suspect_id="butler", clue_id="torn_letter")
    state = cast("GameState", graph.invoke(state))

    assert "haven't found 'torn_letter'" in state["last_output"]
    # Turn was NOT consumed.
    assert state["turn_count"] == 0

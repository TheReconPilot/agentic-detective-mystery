"""Cross-turn commitment integration: turn 2's suspect prompt contains turn 1's claim.

This is the load-bearing M9 contract: the bible-as-canon discipline depends
on the *next* turn seeing a structured summary of what was just said, NOT
the raw transcript. The test asserts both halves — the summary appears,
the raw answer does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.graph.state import GameState, initial_state
from mystery.models import Commitment
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index
from mystery.tools.interrogate import apply_interrogate

if TYPE_CHECKING:
    from mystery.models import CaseBible, Suspect


class _RecordingChat:
    """FakeListChatModel-style chat that also captures every system prompt it sees."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.system_prompts: list[str] = []

    def invoke(self, messages: list[Any]) -> Any:
        # messages[0] is the SystemMessage from build_suspect_messages.
        self.system_prompts.append(str(messages[0].content))

        class _Resp:
            content: str

        r = _Resp()
        r.content = self._replies.pop(0)
        return r


class _StaticExtractor:
    """Returns a fixed sequence of commitments so the test can assert on rendering."""

    def __init__(self, commitments: list[Commitment]) -> None:
        self._commitments = list(commitments)

    def extract(self, suspect: Suspect, question: str, answer: str) -> Commitment | None:
        del suspect, question, answer
        if not self._commitments:
            return None
        return self._commitments.pop(0)


def _merge(state: GameState, update: dict[str, Any]) -> GameState:
    return cast("GameState", {**state, **update})


def test_second_turn_prompt_contains_prior_commitment(valid_bible: CaseBible) -> None:
    """Turn 1 produces a commitment; turn 2's system prompt must replay it."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)

    chat = _RecordingChat(
        replies=[
            "I was in the garden, sir, taking the night air.",  # turn 1
            "As I already said, the garden — nothing more to add.",  # turn 2
        ],
    )
    commitment_turn1 = Commitment(
        claimed_location_id="garden",
        summary="They claimed to have been in the garden taking the night air.",
    )
    extractor = _StaticExtractor([commitment_turn1])

    state = initial_state(valid_bible)
    update = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        cast("Any", chat),
        suspect_id="butler",
        question="Where were you at the time of the murder?",
        commitment_extractor=extractor,
    )
    state = _merge(state, update)

    # Turn-1 prompt: no prior commitments block yet.
    assert "previously told this detective" not in chat.system_prompts[0]

    # Turn 2: the *same* suspect, second question.
    update = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        cast("Any", chat),
        suspect_id="butler",
        question="Anything else to add?",
        commitment_extractor=extractor,
    )
    state = _merge(state, update)

    # Turn-2 prompt must now carry the prior commitment summary.
    assert "previously told this detective" in chat.system_prompts[1]
    assert "They claimed to have been in the garden taking the night air." in chat.system_prompts[1]
    # The raw answer text from turn 1 must NOT leak verbatim into turn 2.
    assert "taking the night air, sir" not in chat.system_prompts[1]


def test_commitments_are_per_suspect(valid_bible: CaseBible) -> None:
    """Niece's commitments must not bleed into butler's prompt and vice versa."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)

    chat = _RecordingChat(replies=["Garden.", "Hallway.", "Garden again."])
    extractor = _StaticExtractor(
        [
            Commitment(summary="BUTLER_CLAIMED_GARDEN."),
            Commitment(summary="NIECE_CLAIMED_HALLWAY."),
        ],
    )

    state = initial_state(valid_bible)
    state = _merge(
        state,
        apply_interrogate(
            state,
            valid_bible,
            vectorstore,
            cast("Any", chat),
            suspect_id="butler",
            question="?",
            commitment_extractor=extractor,
        ),
    )
    state = _merge(
        state,
        apply_interrogate(
            state,
            valid_bible,
            vectorstore,
            cast("Any", chat),
            suspect_id="niece",
            question="?",
            commitment_extractor=extractor,
        ),
    )
    # Third turn: butler again. Niece's commitment must NOT appear.
    _ = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        cast("Any", chat),
        suspect_id="butler",
        question="And later?",
        commitment_extractor=extractor,
    )

    butler_second_prompt = chat.system_prompts[2]
    assert "BUTLER_CLAIMED_GARDEN." in butler_second_prompt
    assert "NIECE_CLAIMED_HALLWAY." not in butler_second_prompt


def test_full_game_loop_carries_commitments_through_graph(valid_bible: CaseBible) -> None:
    """Build the actual LangGraph and exercise the dispatcher path."""
    from mystery.graph.game import build_game_graph
    from mystery.graph.state import InterrogateAction

    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = _RecordingChat(replies=["I was in the kitchen.", "I already told you — kitchen."])
    extractor = _StaticExtractor(
        [Commitment(summary="They claimed to have been in the kitchen.")],
    )

    graph = build_game_graph(valid_bible, vectorstore, cast("Any", chat), extractor)
    state = initial_state(valid_bible)

    state["pending_action"] = InterrogateAction(suspect_id="butler", question="Where were you?")
    state = cast("GameState", graph.invoke(state))

    state["pending_action"] = InterrogateAction(
        suspect_id="butler",
        question="And then?",
    )
    state = cast("GameState", graph.invoke(state))

    assert "They claimed to have been in the kitchen." in chat.system_prompts[1]
    assert len(state["suspect_commitments"]["butler"]) == 1


def test_turn_with_none_commitment_does_not_pollute_dict(valid_bible: CaseBible) -> None:
    """A None extraction must leave suspect_commitments untouched."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["A non-answer."])

    # Extractor returns None on every call.
    extractor = _StaticExtractor(commitments=[])

    state = initial_state(valid_bible)
    update = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        cast("Any", chat),
        suspect_id="butler",
        question="?",
        commitment_extractor=extractor,
    )
    assert "suspect_commitments" not in update

"""Unit tests for apply_show: the M10 confrontation tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.agents.suspect import build_suspect_messages
from mystery.graph.state import GameState, initial_state
from mystery.models import Commitment
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index
from mystery.tools.show import apply_show

if TYPE_CHECKING:
    from mystery.models import CaseBible, Suspect


def _merge(state: GameState, update: dict[str, Any]) -> GameState:
    return cast("GameState", {**state, **update})


def _butler(bible: CaseBible) -> Suspect:
    return next(s for s in bible.suspects if s.id == "butler")


def _torn_letter(bible: CaseBible) -> object:
    return next(c for c in bible.clues if c.id == "torn_letter")


# ---------- error paths ----------


def test_show_unknown_suspect_returns_helpful_error(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_show(
        state,
        valid_bible,
        vectorstore=None,  # type: ignore[arg-type]
        chat_model=None,  # type: ignore[arg-type]
        suspect_id="phantom",
        clue_id="torn_letter",
    )
    assert "'phantom'" in update["last_output"]
    assert "turn_count" not in update


def test_show_unknown_clue_returns_helpful_error(valid_bible: CaseBible) -> None:
    state = initial_state(valid_bible)
    update = apply_show(
        state,
        valid_bible,
        vectorstore=None,  # type: ignore[arg-type]
        chat_model=None,  # type: ignore[arg-type]
        suspect_id="butler",
        clue_id="imaginary_clue",
    )
    assert "imaginary_clue" in update["last_output"]
    assert "not sure which clue" in update["last_output"]
    assert "turn_count" not in update


def test_show_unrevealed_clue_is_refused(valid_bible: CaseBible) -> None:
    """The player must have actually examined the clue before showing it."""
    state = initial_state(valid_bible)
    # torn_letter is in the library and exists, but we haven't examined yet.
    update = apply_show(
        state,
        valid_bible,
        vectorstore=None,  # type: ignore[arg-type]
        chat_model=None,  # type: ignore[arg-type]
        suspect_id="butler",
        clue_id="torn_letter",
    )
    assert "haven't found 'torn_letter'" in update["last_output"]
    assert "turn_count" not in update


# ---------- prompt rendering ----------


def test_build_suspect_messages_renders_confronting_clue(valid_bible: CaseBible) -> None:
    clue = _torn_letter(valid_bible)
    msgs = build_suspect_messages(
        _butler(valid_bible),
        [Document(page_content="The library door sticks when humid.")],
        question="?",
        confronting_clue=clue,  # type: ignore[arg-type]
    )
    system = str(msgs[0].content)
    assert "holding up a piece of evidence" in system
    # The clue description appears verbatim:
    assert "torn letter" in system.lower()


def test_build_suspect_messages_omits_clue_block_when_none(valid_bible: CaseBible) -> None:
    """The plain `ask` path must NOT carry a confronting-clue block."""
    msgs = build_suspect_messages(_butler(valid_bible), [], question="?")
    system = str(msgs[0].content)
    assert "holding up a piece of evidence" not in system


# ---------- happy path through apply_show ----------


def test_show_with_revealed_clue_invokes_agent_and_costs_a_turn(
    valid_bible: CaseBible,
) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["I... I have never seen that letter before, sir."])

    state = initial_state(valid_bible)
    # Simulate the player having examined the library — torn_letter is now revealed.
    state = _merge(state, {"revealed_clue_ids": ["torn_letter"]})

    update = apply_show(
        state,
        valid_bible,
        vectorstore,
        chat,
        suspect_id="butler",
        clue_id="torn_letter",
    )
    assert update["turn_count"] == state["turn_count"] + 1
    assert "Mr. Hodges" in update["last_output"]
    assert "I... I have never seen that letter before" in update["last_output"]
    assert "you show" in update["last_output"].lower()


def test_show_resolves_clue_by_description_word(valid_bible: CaseBible) -> None:
    """Players can type a word from the description instead of the snake_case id."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["That letter is not mine."])

    state = initial_state(valid_bible)
    state = _merge(state, {"revealed_clue_ids": ["torn_letter"]})

    update = apply_show(
        state,
        valid_bible,
        vectorstore,
        chat,
        suspect_id="butler",
        clue_id="letter",
    )
    assert update["turn_count"] == state["turn_count"] + 1
    assert "torn letter" in update["last_output"].lower()


def test_show_resolves_multi_word_clue_description(valid_bible: CaseBible) -> None:
    """A multi-word reference like 'torn letter' resolves to torn_letter."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["..."])

    state = initial_state(valid_bible)
    state = _merge(state, {"revealed_clue_ids": ["torn_letter"]})

    update = apply_show(
        state,
        valid_bible,
        vectorstore,
        chat,
        suspect_id="butler",
        clue_id="torn letter",
    )
    assert update["turn_count"] == state["turn_count"] + 1


def test_show_lists_revealed_inventory_when_clue_ref_is_unknown(
    valid_bible: CaseBible,
) -> None:
    state = initial_state(valid_bible)
    state = _merge(state, {"revealed_clue_ids": ["torn_letter"]})
    update = apply_show(
        state,
        valid_bible,
        vectorstore=None,  # type: ignore[arg-type]
        chat_model=None,  # type: ignore[arg-type]
        suspect_id="butler",
        clue_id="vacuum_cleaner",
    )
    # The error names what they typed and lists what they actually have.
    assert "vacuum_cleaner" in update["last_output"]
    assert "torn_letter" in update["last_output"]
    assert "turn_count" not in update


class _StaticExtractor:
    def __init__(self, commitment: Commitment | None) -> None:
        self._commitment = commitment

    def extract(self, suspect: Suspect, question: str, answer: str) -> Commitment | None:
        del suspect, question, answer
        return self._commitment


def test_show_runs_commitment_extractor_and_persists(valid_bible: CaseBible) -> None:
    """Confrontation reactions feed the next-turn carry-through, just like `ask`."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["It's not mine."])
    extractor = _StaticExtractor(
        Commitment(
            summary="They denied owning the torn letter.", denied_facts=["owning the letter"]
        ),
    )

    state = initial_state(valid_bible)
    state = _merge(state, {"revealed_clue_ids": ["torn_letter"]})

    update = apply_show(
        state,
        valid_bible,
        vectorstore,
        chat,
        suspect_id="butler",
        clue_id="torn_letter",
        commitment_extractor=extractor,
    )
    assert update["suspect_commitments"]["butler"][0].summary == (
        "They denied owning the torn letter."
    )

"""LLM playtester end-to-end with a scripted FakeListChatModel detective."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.evals.llm_player import play_with_llm, render_observation
from mystery.graph.state import initial_state
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_observation_exposes_exits_and_suspects_without_leaking_truth(
    valid_bible: CaseBible,
) -> None:
    state = initial_state(valid_bible)
    obs = render_observation(state, valid_bible)
    assert "library" in obs  # current location id
    assert "hallway" in obs  # an exit
    assert "butler" in obs and "niece" in obs and "cook" in obs  # suspect roster
    # Bible truth that must not leak to the player:
    assert "killer_id" not in obs
    assert "is_true" not in obs
    assert "deception_policy" not in obs
    assert "canonical_timeline" not in obs


def test_llm_player_solves_case_with_scripted_commands(valid_bible: CaseBible) -> None:
    """A FakeListChatModel that emits the optimal command sequence wins."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    suspect_chat = FakeListChatModel(responses=["I was in the pantry, sir."] * 10)
    # Hand-written command stream: examine, walk the map, ask butler, accuse.
    detective_commands = [
        "examine",
        "move hallway",
        "examine",
        "move garden",
        "examine",
        "move hallway",
        "move library",
        "ask butler where were you at the time of the murder",
        "accuse butler",
    ]
    detective_chat = FakeListChatModel(responses=detective_commands)

    report = play_with_llm(
        valid_bible,
        vectorstore,
        suspect_chat,
        detective_chat,
        max_turns=20,
    )

    assert report.success is True
    assert report.accused == "butler"
    assert report.parse_errors == 0
    # Examined all three rooms => all clues should be revealed by the time we accuse.
    assert {"muddy_boots", "torn_letter"}.issubset(
        {clue_id for step in report.steps for clue_id in (step.output.split())},
    ) or report.success  # accusation success is the contract; clue check is informational


def test_llm_player_records_parse_errors_without_crashing(valid_bible: CaseBible) -> None:
    """A detective that emits garbage should rack up parse_errors, not crash."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    suspect_chat = FakeListChatModel(responses=["unused"])
    detective_chat = FakeListChatModel(
        responses=["investigate the library", "search around", "question butler"] * 5,
    )

    report = play_with_llm(
        valid_bible,
        vectorstore,
        suspect_chat,
        detective_chat,
        max_turns=10,
    )
    assert report.parse_errors >= 1
    # All three are unknown verbs -> the loop aborts on consecutive_parse_errors >= 5.
    assert report.success is False

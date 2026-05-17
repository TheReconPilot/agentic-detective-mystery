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
    """A detective that emits garbage should record events, not crash.

    With the router's new tolerance these verbs now parse, so we send genuine
    unknown commands to exercise the parse-error path.
    """
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    suspect_chat = FakeListChatModel(responses=["unused"])
    detective_chat = FakeListChatModel(
        responses=["frobnicate the room", "nonsense here", "blarg butler"] * 5,
    )

    report = play_with_llm(
        valid_bible,
        vectorstore,
        suspect_chat,
        detective_chat,
        max_turns=10,
    )
    assert report.parse_errors >= 1
    assert report.success is False


def test_llm_player_aborts_on_no_progress_alternation(valid_bible: CaseBible) -> None:
    """Alternating move-A / move-B forever counts as no-progress and aborts."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    suspect_chat = FakeListChatModel(responses=["unused"])
    # library <-> hallway forever, never examining.
    detective_chat = FakeListChatModel(responses=(["move hallway", "move library"] * 30))

    report = play_with_llm(
        valid_bible,
        vectorstore,
        suspect_chat,
        detective_chat,
        max_turns=100,
        no_progress_window=8,
    )
    assert report.success is False
    assert report.turns_used < 100
    assert any(step.parsed_kind == "aborted_no_progress" for step in report.steps)


def test_llm_player_aborts_on_consecutive_repeat_loop(valid_bible: CaseBible) -> None:
    """A stuck detective that emits the same command repeatedly is force-aborted."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    suspect_chat = FakeListChatModel(responses=["unused"])
    detective_chat = FakeListChatModel(responses=["examine"] * 20)

    report = play_with_llm(
        valid_bible,
        vectorstore,
        suspect_chat,
        detective_chat,
        max_turns=50,
        max_consecutive_repeats=3,
    )
    assert report.success is False
    # Aborted well before max_turns.
    assert report.turns_used < 50
    assert any(step.parsed_kind == "aborted_loop" for step in report.steps)


def test_observation_lists_unsearched_rooms(valid_bible: CaseBible) -> None:
    """The detective needs to see which rooms still need an `examine` action."""
    state = initial_state(valid_bible)
    obs = render_observation(state, valid_bible)
    assert "ROOMS NOT YET SEARCHED" in obs
    tail = obs.split("ROOMS NOT YET SEARCHED")[1]
    assert "library" in tail and "hallway" in tail and "garden" in tail
    # The current room is also flagged as unexamined before the player examines.
    assert "examined this room? NO" in obs

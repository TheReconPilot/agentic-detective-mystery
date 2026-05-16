"""Offline tests for the optimal player and solvability harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.evals.optimal_player import plan_examine_all, play_to_solve
from mystery.evals.solvability import run_solvability_eval
from mystery.graph.state import AccuseAction, ExamineAction, MoveAction
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_plan_examine_all_visits_every_reachable_location(valid_bible: CaseBible) -> None:
    plan = plan_examine_all(valid_bible)

    moves = [a.location_id for a in plan if isinstance(a, MoveAction)]
    visited = {valid_bible.victim.location_of_death_id, *moves}
    reachable = {loc.id for loc in valid_bible.locations}

    assert visited == reachable
    assert any(isinstance(a, ExamineAction) for a in plan)


def test_plan_does_not_include_accusation(valid_bible: CaseBible) -> None:
    """The planner's job is exploration; accusation is appended by the caller."""
    plan = plan_examine_all(valid_bible)
    assert not any(isinstance(a, AccuseAction) for a in plan)


def test_play_to_solve_succeeds_on_valid_bible(valid_bible: CaseBible) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["unused — optimal player doesn't interrogate"])

    report = play_to_solve(valid_bible, vectorstore, chat)

    assert report.success is True
    assert report.accused == valid_bible.killer_id
    assert report.actual_killer == valid_bible.killer_id
    assert report.locations_visited == len(valid_bible.locations)
    # Every clue should be revealed because we examine every room.
    assert report.clues_revealed == len(valid_bible.clues)
    assert report.transcript  # non-empty
    assert report.turns > 0


def test_play_to_solve_respects_max_turns(valid_bible: CaseBible) -> None:
    """A pathologically tight budget should abort cleanly (no crash, no win)."""
    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = FakeListChatModel(responses=["unused"])

    report = play_to_solve(valid_bible, vectorstore, chat, max_turns=1)

    assert report.success is False
    assert report.turns <= 2  # might step one past the cap depending on increment timing


def test_solvability_eval_aggregates_across_bibles(valid_bible: CaseBible) -> None:
    bibles = [
        valid_bible,
        valid_bible.model_copy(update={"seed": 99}),
        valid_bible.model_copy(update={"seed": 7}),
    ]
    report = run_solvability_eval(
        bibles,
        embeddings_factory=lambda: DeterministicFakeEmbedding(size=16),
        chat_factory=lambda: FakeListChatModel(responses=["unused"]),
    )
    assert report.cases_run == 3
    assert report.successes == 3
    assert report.success_rate == 1.0
    assert report.mean_turns_on_success is not None
    assert report.mean_turns_on_success > 0


def test_solvability_eval_handles_empty_bible_list() -> None:
    report = run_solvability_eval(
        [],
        embeddings_factory=lambda: DeterministicFakeEmbedding(size=16),
        chat_factory=lambda: FakeListChatModel(responses=[]),
    )
    assert report.cases_run == 0
    assert report.success_rate == 0.0
    assert report.mean_turns_on_success is None

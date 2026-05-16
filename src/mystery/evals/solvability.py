"""Aggregate ``play_to_solve`` over many bibles into a solvability report."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING

from mystery.evals.optimal_player import play_to_solve
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

    from mystery.evals.optimal_player import SolvabilityReport
    from mystery.models import CaseBible


@dataclass
class SolvabilityEvalReport:
    cases_run: int
    successes: int
    success_rate: float
    mean_turns_on_success: float | None
    per_case: list[SolvabilityReport] = field(default_factory=list)


def run_solvability_eval(
    bibles: list[CaseBible],
    embeddings_factory: Callable[[], Embeddings],
    chat_factory: Callable[[], BaseChatModel],
    *,
    max_turns: int = 200,
) -> SolvabilityEvalReport:
    """Run ``play_to_solve`` on each bible and aggregate.

    Factories are called once per bible so concrete clients (ChatOllama,
    OllamaEmbeddings) get fresh connections per case — important when state
    bleed-over would be confusing.
    """
    if not bibles:
        return SolvabilityEvalReport(
            cases_run=0,
            successes=0,
            success_rate=0.0,
            mean_turns_on_success=None,
            per_case=[],
        )

    per_case: list[SolvabilityReport] = []
    for bible in bibles:
        embeddings = embeddings_factory()
        chat = chat_factory()
        vectorstore = build_index(build_chunks(bible), embeddings)
        per_case.append(play_to_solve(bible, vectorstore, chat, max_turns=max_turns))

    successes = sum(1 for r in per_case if r.success)
    winning_turns = [r.turns for r in per_case if r.success]

    return SolvabilityEvalReport(
        cases_run=len(per_case),
        successes=successes,
        success_rate=successes / len(per_case),
        mean_turns_on_success=mean(winning_turns) if winning_turns else None,
        per_case=per_case,
    )

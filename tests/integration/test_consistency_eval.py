"""Offline tests for the consistency-eval harness using a scripted judge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.evals.consistency import (
    DEFAULT_QUESTIONS,
    JudgeRuling,
    run_consistency_eval,
)
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index

if TYPE_CHECKING:
    from mystery.models import CaseBible


@dataclass
class _ScriptedJudge:
    """Returns a queued ruling per call, or a default if the queue is empty."""

    rulings: list[JudgeRuling] = field(default_factory=list)
    default: JudgeRuling = field(
        default_factory=lambda: JudgeRuling(verdict="consistent", reasoning="(default)"),
    )
    calls: int = 0

    def judge(
        self,
        bible: CaseBible,
        suspect_id: str,
        question: str,
        response: str,
    ) -> JudgeRuling:
        del bible, suspect_id, question, response
        self.calls += 1
        if self.rulings:
            return self.rulings.pop(0)
        return self.default


def test_consistency_eval_records_one_per_suspect_per_question(valid_bible: CaseBible) -> None:
    vectorstore = build_index(build_chunks(valid_bible), DeterministicFakeEmbedding(size=16))
    chat = FakeListChatModel(responses=[f"answer-{i}" for i in range(50)])
    judge = _ScriptedJudge()

    report = run_consistency_eval(valid_bible, vectorstore, chat, judge)

    expected_total = len(valid_bible.suspects) * len(DEFAULT_QUESTIONS)
    assert report.total == expected_total
    assert judge.calls == expected_total
    # All records are unique (suspect, question) pairs.
    pairs = {(r.suspect_id, r.question) for r in report.records}
    assert len(pairs) == expected_total


def test_consistency_rate_computes_correctly(valid_bible: CaseBible) -> None:
    vectorstore = build_index(build_chunks(valid_bible), DeterministicFakeEmbedding(size=16))
    chat = FakeListChatModel(responses=[f"answer-{i}" for i in range(50)])

    # Make every other ruling "contradicts".
    n = len(valid_bible.suspects) * len(DEFAULT_QUESTIONS)
    rulings = [
        JudgeRuling(verdict="contradicts" if i % 2 == 0 else "consistent", reasoning="r")
        for i in range(n)
    ]
    judge = _ScriptedJudge(rulings=rulings)

    report = run_consistency_eval(valid_bible, vectorstore, chat, judge)

    assert report.consistent + report.contradicts == report.total
    assert report.contradicts == n - n // 2  # ceil(n/2) for the even-indexed contradictions
    assert 0.0 <= report.consistency_rate <= 1.0


def test_consistency_eval_handles_refused_verdict(valid_bible: CaseBible) -> None:
    vectorstore = build_index(build_chunks(valid_bible), DeterministicFakeEmbedding(size=16))
    chat = FakeListChatModel(responses=[f"answer-{i}" for i in range(50)])
    judge = _ScriptedJudge(
        default=JudgeRuling(verdict="refused", reasoning="suspect declined to answer"),
    )

    report = run_consistency_eval(valid_bible, vectorstore, chat, judge)

    assert report.refused == report.total
    assert report.contradicts == 0
    assert report.consistency_rate == 0.0


def test_custom_questions_override_default(valid_bible: CaseBible) -> None:
    vectorstore = build_index(build_chunks(valid_bible), DeterministicFakeEmbedding(size=16))
    chat = FakeListChatModel(responses=[f"answer-{i}" for i in range(20)])
    judge = _ScriptedJudge()

    report = run_consistency_eval(
        valid_bible,
        vectorstore,
        chat,
        judge,
        questions=("Only this one question.",),
    )

    assert report.total == len(valid_bible.suspects)
    assert all(r.question == "Only this one question." for r in report.records)

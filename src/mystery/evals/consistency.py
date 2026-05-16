"""Consistency eval: do suspect responses contradict the bible?

Workflow per case:

1. For each suspect, ask a small set of standardised questions about their
   movements and the victim — this is the same data path the player would
   use, including the character-scoped retriever.
2. Hand each (bible, suspect, question, response) to a ``ConsistencyJudge``.
3. Aggregate verdicts into a report. "Contradicts" is the failure case the
   whole project exists to prevent.

The judge is a Protocol so tests can stub it; the production implementation
is ``LLMConsistencyJudge``, which uses an LLM that sees the *full bible* as
context — the omniscient judge the suspect agent is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from mystery.agents.suspect import respond_as_suspect
from mystery.rag.retriever import suspect_retriever

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.models import CaseBible

Verdict = Literal["consistent", "contradicts", "refused"]


class JudgeRuling(BaseModel):
    """Structured output the LLM judge returns."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    reasoning: str = Field(description="One short sentence justifying the verdict.")


@dataclass
class InterrogationRecord:
    suspect_id: str
    question: str
    response: str
    ruling: JudgeRuling


@dataclass
class ConsistencyReport:
    seed: int
    records: list[InterrogationRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def consistent(self) -> int:
        return sum(1 for r in self.records if r.ruling.verdict == "consistent")

    @property
    def contradicts(self) -> int:
        return sum(1 for r in self.records if r.ruling.verdict == "contradicts")

    @property
    def refused(self) -> int:
        return sum(1 for r in self.records if r.ruling.verdict == "refused")

    @property
    def consistency_rate(self) -> float:
        return self.consistent / self.total if self.total else 0.0


class ConsistencyJudge(Protocol):
    def judge(
        self,
        bible: CaseBible,
        suspect_id: str,
        question: str,
        response: str,
    ) -> JudgeRuling: ...


_JUDGE_SYSTEM = """\
You are an impartial judge in a murder-mystery game. You will see the full
case bible (the absolute truth) and a transcript of one detective question
and one suspect response. Your task is to classify the suspect's response as:

- "consistent" — anything the suspect said that *could* be true is permitted
  by the bible, *or* the suspect lied in a way their deception_policy allows.
  Lies-by-policy count as consistent, because the suspect agent is supposed
  to lie within that policy.
- "contradicts" — the suspect asserted something the bible directly
  contradicts AND their deception_policy does not license the lie.
- "refused" — the suspect declined to answer, deflected entirely, or said
  they don't know.

Return a structured ruling: the verdict plus one short sentence of reasoning.
"""

_JUDGE_USER_TEMPLATE = """\
=== BIBLE (ground truth) ===
{bible_json}

=== INTERROGATION ===
Suspect: {suspect_id}
Question: {question}
Response: {response}

Classify the response.
"""


class LLMConsistencyJudge:
    """Production judge: uses a chat model with the full bible as context."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._structured = chat_model.with_structured_output(JudgeRuling)

    def judge(
        self,
        bible: CaseBible,
        suspect_id: str,
        question: str,
        response: str,
    ) -> JudgeRuling:
        user = _JUDGE_USER_TEMPLATE.format(
            bible_json=bible.model_dump_json(indent=2),
            suspect_id=suspect_id,
            question=question,
            response=response,
        )
        messages = [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=user)]
        result = self._structured.invoke(messages)
        return cast("JudgeRuling", result)


DEFAULT_QUESTIONS = (
    "Where were you at the time of the murder?",
    "Did you see anyone else nearby that evening?",
    "What was your relationship to the victim?",
    "Is there anything you would like the detective to know?",
)


def run_consistency_eval(
    bible: CaseBible,
    vectorstore: Chroma,
    suspect_chat_model: BaseChatModel,
    judge: ConsistencyJudge,
    *,
    questions: tuple[str, ...] = DEFAULT_QUESTIONS,
) -> ConsistencyReport:
    """For each suspect and each question, collect the response and a verdict."""
    report = ConsistencyReport(seed=bible.seed)
    for suspect in bible.suspects:
        retriever = suspect_retriever(vectorstore, suspect_id=suspect.id)
        for question in questions:
            response = respond_as_suspect(
                suspect,
                retriever,
                suspect_chat_model,
                question=question,
            )
            ruling = judge.judge(bible, suspect.id, question, response)
            report.records.append(
                InterrogationRecord(
                    suspect_id=suspect.id,
                    question=question,
                    response=response,
                    ruling=ruling,
                ),
            )
    return report

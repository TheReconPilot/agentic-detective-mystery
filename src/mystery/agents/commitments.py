"""Extract structured commitments from a suspect's free-text answer.

A Commitment is the slice of an interrogation answer we feed back into the
next turn's suspect prompt. The full transcript is NOT persisted: see the
class docstring on :class:`mystery.models.Commitment` for the rationale.

This module defines the extractor abstraction (``CommitmentExtractor``) so
the production path can use a structured-output LLM call while tests inject
a deterministic stub. The null extractor exists for offline tests and for
disabling the feature entirely without branching the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage

from mystery.models import Commitment

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from mystery.models import Suspect


_EXTRACTOR_SYSTEM = """\
You distil one short COMMITMENT record from a suspect's answer in a murder
mystery game. A commitment captures what the suspect just *committed to*:
where they claim they were, when, who they say can vouch for them, and any
fact they explicitly denied. You are NOT judging truthfulness — the suspect
may be lying — you are just recording what they said.

Rules:
- claimed_location_id: a snake_case room id if the suspect named one, else null.
- claimed_time_window: (start, end) minutes if a time was given, else null.
- named_witness_ids: other suspect ids they named as witnesses (lowercase).
- denied_facts: short sentences for things the suspect explicitly denied.
- summary: ONE short sentence paraphrasing the claim, written in third person
  ("They claimed …"). This is what we replay verbatim next turn.

If the answer contains no factual claim (deflection, refusal, generic
banter), return an empty commitment with summary describing the deflection.
"""


def _user_prompt(suspect: Suspect, question: str, answer: str) -> str:
    return (
        f"Suspect: {suspect.name} (id={suspect.id}, archetype={suspect.archetype}).\n"
        f"Detective asked: {question}\n"
        f"Suspect answered: {answer}\n\n"
        "Return one COMMITMENT record summarising what the suspect just committed to."
    )


class CommitmentExtractor(Protocol):
    """Extract a Commitment from a (suspect, question, answer) triple.

    Returning ``None`` is an explicit signal that nothing worth carrying
    forward was said — distinct from an empty Commitment, which still gets
    rendered ("They deflected the question").
    """

    def extract(self, suspect: Suspect, question: str, answer: str) -> Commitment | None: ...


class NullCommitmentExtractor:
    """No-op extractor used by tests and the offline default.

    Skips the structured-output round trip entirely. Production wires the
    LLM-backed extractor in :func:`mystery.cli._default_commitment_extractor_factory`.
    """

    def extract(self, suspect: Suspect, question: str, answer: str) -> Commitment | None:
        del suspect, question, answer
        return None


class LLMCommitmentExtractor:
    """Structured-output extractor backed by a langchain chat model.

    Uses ``with_structured_output(Commitment)`` so the model is forced to
    emit a schema-valid record. Failures surface as ``None`` rather than
    crashing the interrogation turn — a missed extraction degrades the
    feature, but a thrown exception would break the game loop.
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._structured = chat_model.with_structured_output(Commitment)

    def extract(self, suspect: Suspect, question: str, answer: str) -> Commitment | None:
        messages = [
            SystemMessage(content=_EXTRACTOR_SYSTEM),
            HumanMessage(content=_user_prompt(suspect, question, answer)),
        ]
        try:
            result = self._structured.invoke(messages)
        except Exception:
            return None
        if result is None:
            return None
        return cast("Commitment", result)

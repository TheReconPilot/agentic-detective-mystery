"""Unit tests for the Commitment model, extractor protocol, and prompt rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from mystery.agents.commitments import (
    LLMCommitmentExtractor,
    NullCommitmentExtractor,
)
from mystery.agents.suspect import build_suspect_messages
from mystery.models import Commitment, Suspect

if TYPE_CHECKING:
    from mystery.models import CaseBible


def _butler(bible: CaseBible) -> Suspect:
    return next(s for s in bible.suspects if s.id == "butler")


# ---------- Commitment model ----------


def test_commitment_accepts_full_record() -> None:
    c = Commitment(
        claimed_location_id="garden",
        claimed_time_window=(45, 75),
        named_witness_ids=["cook"],
        denied_facts=["I was never in the library."],
        summary="They claimed to have been in the garden from 45 to 75 minutes.",
    )
    assert c.claimed_location_id == "garden"
    assert c.claimed_time_window == (45, 75)
    assert c.named_witness_ids == ["cook"]
    assert c.denied_facts == ["I was never in the library."]


def test_commitment_summary_is_required() -> None:
    """The summary is what we replay verbatim next turn — it must exist."""
    with pytest.raises(ValidationError):
        Commitment.model_validate({})  # missing required summary


def test_commitment_forbids_extras() -> None:
    with pytest.raises(ValidationError):
        Commitment.model_validate({"summary": "ok", "spurious": "no"})


def test_commitment_defaults_for_partial_record() -> None:
    """Most extracted commitments will only have a summary."""
    c = Commitment(summary="They refused to answer.")
    assert c.claimed_location_id is None
    assert c.claimed_time_window is None
    assert c.named_witness_ids == []
    assert c.denied_facts == []


# ---------- NullCommitmentExtractor ----------


def test_null_extractor_returns_none(valid_bible: CaseBible) -> None:
    extractor = NullCommitmentExtractor()
    result = extractor.extract(_butler(valid_bible), "where were you?", "I was in the pantry.")
    assert result is None


# ---------- LLMCommitmentExtractor ----------


class _FakeStructured:
    """Stand-in for chat.with_structured_output(Commitment).invoke()."""

    def __init__(self, result: Commitment | None | Exception) -> None:
        self._result = result

    def invoke(self, _messages: object) -> Commitment | None:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeChat:
    def __init__(self, structured: _FakeStructured) -> None:
        self._structured = structured

    def with_structured_output(self, _schema: object) -> _FakeStructured:
        return self._structured


def test_llm_extractor_returns_commitment_on_success(valid_bible: CaseBible) -> None:
    expected = Commitment(summary="They claimed to be in the garden.")
    chat = _FakeChat(_FakeStructured(expected))
    extractor = LLMCommitmentExtractor(chat)  # type: ignore[arg-type]
    result = extractor.extract(_butler(valid_bible), "where?", "the garden")
    assert result == expected


def test_llm_extractor_returns_none_when_structured_output_raises(
    valid_bible: CaseBible,
) -> None:
    """A failed extraction must not crash the interrogation turn."""
    chat = _FakeChat(_FakeStructured(RuntimeError("model blew up")))
    extractor = LLMCommitmentExtractor(chat)  # type: ignore[arg-type]
    result = extractor.extract(_butler(valid_bible), "where?", "irrelevant")
    assert result is None


def test_llm_extractor_returns_none_when_structured_output_returns_none(
    valid_bible: CaseBible,
) -> None:
    chat = _FakeChat(_FakeStructured(None))
    extractor = LLMCommitmentExtractor(chat)  # type: ignore[arg-type]
    result = extractor.extract(_butler(valid_bible), "where?", "irrelevant")
    assert result is None


# ---------- Prompt rendering with prior commitments ----------


def test_system_message_omits_commitments_block_when_none(valid_bible: CaseBible) -> None:
    msgs = build_suspect_messages(_butler(valid_bible), [], question="?")
    system = str(msgs[0].content)
    assert "previously told this detective" not in system


def test_system_message_renders_commitments_when_provided(valid_bible: CaseBible) -> None:
    prior = [
        Commitment(summary="They claimed to have been in the garden at 45-75."),
        Commitment(summary="They denied hearing any argument in the library."),
    ]
    msgs = build_suspect_messages(
        _butler(valid_bible),
        [],
        question="And before that?",
        prior_commitments=prior,
    )
    system = str(msgs[0].content)
    assert "previously told this detective" in system
    assert "They claimed to have been in the garden at 45-75." in system
    assert "They denied hearing any argument in the library." in system
    # The reinforcement line that ties commitments to the deception policy:
    assert "Stay consistent with those prior statements" in system


def test_commitments_block_does_not_leak_raw_transcript(valid_bible: CaseBible) -> None:
    """We render the structured summary, not the raw answer the suspect gave.

    If the raw answer leaked into the prompt, the LLM would treat its own
    lies as ground truth and the bible-as-canon discipline would erode.
    """
    prior = [Commitment(summary="They claimed to be in the garden.")]
    msgs = build_suspect_messages(
        _butler(valid_bible),
        [Document(page_content="The library door sticks when humid.")],
        question="next?",
        prior_commitments=prior,
    )
    system = str(msgs[0].content)
    # The structured summary appears, but no transcript-style "Q: ... A: ..."
    assert "They claimed to be in the garden." in system
    assert "Q:" not in system
    assert "A:" not in system


# ---------- apply_interrogate wiring ----------


class _FakeRetrieverlessChat:
    """Just returns a fixed reply; doesn't depend on retriever output."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last_messages: list[object] | None = None

    def invoke(self, messages: list[object]) -> object:
        self.last_messages = messages

        class _Resp:
            content: str

        r = _Resp()
        r.content = self._reply
        return r


class _StaticExtractor:
    """Deterministic extractor that emits a fixed commitment for assertions."""

    def __init__(self, commitment: Commitment | None) -> None:
        self._commitment = commitment

    def extract(self, suspect: Suspect, question: str, answer: str) -> Commitment | None:
        del suspect, question, answer
        return self._commitment


def test_apply_interrogate_appends_commitment_to_state(valid_bible: CaseBible) -> None:
    """A successful interrogation persists the extracted commitment under suspect_id."""
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    from mystery.graph.state import initial_state
    from mystery.rag.chunks import build_chunks
    from mystery.rag.indexer import build_index
    from mystery.tools.interrogate import apply_interrogate

    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = _FakeRetrieverlessChat("I was polishing silver, sir.")
    extractor = _StaticExtractor(Commitment(summary="They said they were polishing silver."))

    state = initial_state(valid_bible)
    update = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        chat,  # type: ignore[arg-type]
        suspect_id="butler",
        question="Where were you?",
        commitment_extractor=extractor,
    )
    assert "suspect_commitments" in update
    assert update["suspect_commitments"]["butler"][0].summary == (
        "They said they were polishing silver."
    )


def test_apply_interrogate_skips_state_update_when_extractor_returns_none(
    valid_bible: CaseBible,
) -> None:
    """If the extractor declined, we should not write an empty/partial dict."""
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    from mystery.graph.state import initial_state
    from mystery.rag.chunks import build_chunks
    from mystery.rag.indexer import build_index
    from mystery.tools.interrogate import apply_interrogate

    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = _FakeRetrieverlessChat("...")
    extractor = _StaticExtractor(None)

    state = initial_state(valid_bible)
    update = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        chat,  # type: ignore[arg-type]
        suspect_id="butler",
        question="?",
        commitment_extractor=extractor,
    )
    assert "suspect_commitments" not in update


def test_apply_interrogate_with_no_extractor_dep_does_not_crash(
    valid_bible: CaseBible,
) -> None:
    """Optional dep: callers may omit the extractor entirely (offline tests)."""
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    from mystery.graph.state import initial_state
    from mystery.rag.chunks import build_chunks
    from mystery.rag.indexer import build_index
    from mystery.tools.interrogate import apply_interrogate

    embeddings = DeterministicFakeEmbedding(size=16)
    vectorstore = build_index(build_chunks(valid_bible), embeddings)
    chat = _FakeRetrieverlessChat("Nothing relevant.")

    state = initial_state(valid_bible)
    update = apply_interrogate(
        state,
        valid_bible,
        vectorstore,
        chat,  # type: ignore[arg-type]
        suspect_id="butler",
        question="?",
    )
    assert "suspect_commitments" not in update
    assert update["turn_count"] == 1

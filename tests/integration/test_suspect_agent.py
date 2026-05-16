"""End-to-end: retriever + chat model produce a suspect's response."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mystery.agents.suspect import respond_as_suspect
from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index
from mystery.rag.retriever import suspect_retriever

if TYPE_CHECKING:
    from mystery.models import CaseBible


def test_respond_as_suspect_returns_chat_model_output(valid_bible: CaseBible) -> None:
    index = build_index(build_chunks(valid_bible), DeterministicFakeEmbedding(size=16))
    retriever = suspect_retriever(index, suspect_id="butler", k=4)
    butler = next(s for s in valid_bible.suspects if s.id == "butler")
    chat = FakeListChatModel(responses=["I was in the pantry polishing silverware."])

    reply = respond_as_suspect(butler, retriever, chat, question="Where were you at nine o'clock?")

    assert reply == "I was in the pantry polishing silverware."

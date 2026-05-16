"""Integration test: end-to-end RAG pipeline upholds character-scope isolation.

Uses ``DeterministicFakeEmbedding`` so the test is offline and reproducible.
We don't care about ranking quality — the assertion is purely about *which*
documents the metadata filter permits to surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from mystery.rag.chunks import build_chunks
from mystery.rag.indexer import build_index
from mystery.rag.retriever import suspect_retriever

if TYPE_CHECKING:
    from langchain_chroma import Chroma

    from mystery.models import CaseBible


@pytest.fixture
def index(valid_bible: CaseBible) -> Chroma:
    chunks = build_chunks(valid_bible)
    embeddings = DeterministicFakeEmbedding(size=16)
    return build_index(chunks, embeddings)


@pytest.mark.parametrize(
    "querying_suspect",
    ["butler", "niece", "cook"],
)
def test_suspect_never_sees_another_suspects_private_chunks(
    valid_bible: CaseBible,
    index: Chroma,
    querying_suspect: str,
) -> None:
    retriever = suspect_retriever(index, suspect_id=querying_suspect, k=100)

    # Use each other suspect's private knowledge as the query — the most
    # adversarial possible probe — and verify none of their chunks leak.
    other_suspects = [s for s in valid_bible.suspects if s.id != querying_suspect]
    for other in other_suspects:
        for fact in other.knowledge:
            results = retriever.invoke(fact)
            leaked = [doc for doc in results if doc.metadata.get("character_id") == other.id]
            assert not leaked, (
                f"suspect={querying_suspect!r} leaked {other.id!r} chunks "
                f"when probing with that suspect's own knowledge: {leaked}"
            )


def test_suspect_can_retrieve_their_own_knowledge(
    valid_bible: CaseBible,
    index: Chroma,
) -> None:
    retriever = suspect_retriever(index, suspect_id="butler", k=100)
    butler = next(s for s in valid_bible.suspects if s.id == "butler")

    results = retriever.invoke(butler.knowledge[0])
    own = [doc for doc in results if doc.metadata.get("character_id") == "butler"]
    assert own, "butler's retriever returned none of butler's own chunks"


def test_world_chunks_are_visible_to_every_suspect(
    valid_bible: CaseBible,
    index: Chroma,
) -> None:
    for suspect in valid_bible.suspects:
        retriever = suspect_retriever(index, suspect_id=suspect.id, k=100)
        results = retriever.invoke("the library")
        world = [doc for doc in results if doc.metadata.get("scope") == "world"]
        assert world, f"suspect {suspect.id} cannot see any world chunks"

"""Persisted Chroma indexes round-trip and don't re-embed on second load."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from mystery.rag.indexer import get_or_build_index, load_index
from mystery.rag.retriever import suspect_retriever

if TYPE_CHECKING:
    from pathlib import Path

    from mystery.models import CaseBible


def test_get_or_build_embeds_then_loads(valid_bible: CaseBible, tmp_path: Path) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    index_dir = tmp_path / "chroma"

    # First call: directory doesn't exist, embeds and writes.
    first = get_or_build_index(valid_bible, embeddings, index_dir)
    assert index_dir.exists() and any(index_dir.iterdir())

    # Second call: directory exists; should not raise and should return a usable store.
    second = get_or_build_index(valid_bible, embeddings, index_dir)
    retriever = suspect_retriever(second, suspect_id="butler", k=10)
    docs = retriever.invoke("garden")
    assert docs, "loaded index returned no documents"
    del first


def test_load_index_returns_same_documents_as_build(
    valid_bible: CaseBible,
    tmp_path: Path,
) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    index_dir = tmp_path / "chroma"

    built = get_or_build_index(valid_bible, embeddings, index_dir)
    built_count = built._collection.count()

    loaded = load_index(embeddings, str(index_dir))
    assert loaded._collection.count() == built_count

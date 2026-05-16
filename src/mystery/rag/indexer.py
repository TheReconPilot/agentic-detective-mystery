"""Build a Chroma index from a list of Chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from mystery.rag.chunks import Chunk

DEFAULT_COLLECTION = "case_bible"


def to_documents(chunks: list[Chunk]) -> list[Document]:
    """Project Chunks into langchain Documents, carrying scoping metadata."""
    return [Document(page_content=c.text, metadata=c.metadata(), id=c.id) for c in chunks]


def build_index(
    chunks: list[Chunk],
    embeddings: Embeddings,
    *,
    persist_directory: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """Create a Chroma store over the chunks.

    ``persist_directory=None`` yields an ephemeral in-memory store — used in tests.
    """
    return Chroma.from_documents(
        documents=to_documents(chunks),
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

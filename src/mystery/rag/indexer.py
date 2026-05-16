"""Build, load, and persist a Chroma index for a case bible."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.documents import Document

from mystery.rag.chunks import build_chunks

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.embeddings import Embeddings

    from mystery.models import CaseBible
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


def load_index(
    embeddings: Embeddings,
    persist_directory: str,
    *,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """Open an existing Chroma store at ``persist_directory`` without re-embedding."""
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def get_or_build_index(
    bible: CaseBible,
    embeddings: Embeddings,
    persist_directory: Path,
    *,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """Load the index at ``persist_directory`` if it exists; otherwise embed and persist.

    The first call for a (bible, embed-model) pair pays the embedding cost; every
    subsequent call is a directory read. Caller is responsible for invalidating
    the directory if the bible changes — typically by deleting it.
    """
    if persist_directory.exists() and any(persist_directory.iterdir()):
        return load_index(embeddings, str(persist_directory), collection_name=collection_name)

    persist_directory.mkdir(parents=True, exist_ok=True)
    return build_index(
        build_chunks(bible),
        embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )

"""Character-scoped retrievers — the keystone of the anti-drift design.

A suspect's retriever sees their own private chunks plus world chunks; it can
never reach another suspect's private knowledge. The integration test
``tests/integration/test_rag_scope_isolation.py`` enforces this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.retrievers import BaseRetriever


def _suspect_filter(suspect_id: str) -> dict[str, Any]:
    return {
        "$or": [
            {"character_id": {"$eq": suspect_id}},
            {"scope": {"$eq": "world"}},
        ],
    }


def suspect_retriever(
    vectorstore: Chroma,
    suspect_id: str,
    *,
    k: int = 4,
) -> BaseRetriever:
    """Build a retriever scoped to one suspect plus world-shared chunks."""
    return vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": _suspect_filter(suspect_id)},
    )

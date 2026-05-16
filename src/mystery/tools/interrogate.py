"""Interrogate tool: wraps the suspect agent into a state-update.

Distinct from the suspect agent itself: this layer adds bookkeeping (turn
count, validation of the suspect id), produces the rendered output line,
and is the only place the agent is invoked during a game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mystery.agents.suspect import respond_as_suspect
from mystery.rag.retriever import suspect_retriever

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_interrogate(
    state: GameState,
    bible: CaseBible,
    vectorstore: Chroma,
    chat_model: BaseChatModel,
    suspect_id: str,
    question: str,
) -> dict[str, Any]:
    suspect = next((s for s in bible.suspects if s.id == suspect_id), None)
    if suspect is None:
        ids = ", ".join(s.id for s in bible.suspects)
        return {"last_output": f"There is no suspect {suspect_id!r}. Known: {ids}."}

    retriever = suspect_retriever(vectorstore, suspect_id=suspect_id)
    reply = respond_as_suspect(suspect, retriever, chat_model, question=question)

    return {
        "turn_count": state["turn_count"] + 1,
        "last_output": f"{suspect.name}: {reply}",
    }

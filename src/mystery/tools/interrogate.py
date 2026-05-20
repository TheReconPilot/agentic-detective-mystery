"""Interrogate tool: wraps the suspect agent into a state-update.

Distinct from the suspect agent itself: this layer adds bookkeeping (turn
count, validation of the suspect id), produces the rendered output line,
and is the only place the agent is invoked during a game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mystery.agents.commitments import NullCommitmentExtractor
from mystery.agents.suspect import respond_as_suspect
from mystery.rag.retriever import suspect_retriever
from mystery.tools._resolve import format_suspect_roster, resolve_suspect
from mystery.tools._streaming import StreamingPrefix

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.agents.commitments import CommitmentExtractor
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_interrogate(
    state: GameState,
    bible: CaseBible,
    vectorstore: Chroma,
    chat_model: BaseChatModel,
    suspect_id: str,
    question: str,
    commitment_extractor: CommitmentExtractor | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    suspect = resolve_suspect(bible, suspect_id)
    if suspect is None:
        roster = format_suspect_roster(bible)
        return {
            "last_output": (
                f"I'm not sure who you mean by {suspect_id!r}. "
                f"You can interrogate any of these (by id, name, or archetype):\n{roster}"
            )
        }

    retriever = suspect_retriever(vectorstore, suspect_id=suspect.id)
    prior = state["suspect_commitments"].get(suspect.id, [])

    wrapped = StreamingPrefix(stream_callback, suspect.name) if stream_callback else None
    reply = respond_as_suspect(
        suspect,
        retriever,
        chat_model,
        question=question,
        prior_commitments=prior,
        stream_callback=wrapped,
    )
    if wrapped is not None:
        wrapped.finalize()

    extractor: CommitmentExtractor = commitment_extractor or NullCommitmentExtractor()
    new_commitment = extractor.extract(suspect, question, reply)
    # Streaming path has already painted the reply to the terminal; suppress
    # the duplicate display by emptying last_output. Non-streaming path
    # behaves exactly as before.
    rendered = "" if stream_callback is not None else f"{suspect.name}: {reply}"
    update: dict[str, Any] = {
        "turn_count": state["turn_count"] + 1,
        "last_output": rendered,
    }
    if new_commitment is not None:
        # Copy the suspect-keyed dict so LangGraph sees an actual change.
        updated_commitments = {k: list(v) for k, v in state["suspect_commitments"].items()}
        updated_commitments.setdefault(suspect.id, []).append(new_commitment)
        update["suspect_commitments"] = updated_commitments
    return update

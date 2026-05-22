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
from mystery.tools._resolve import format_suspect_roster, resolve_clue, resolve_suspect
from mystery.tools._streaming import StreamingPrefix

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.agents.commitments import CommitmentExtractor
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def _expand_topic_question(state: GameState, bible: CaseBible, question: str) -> str:
    """Expand ``about <topic>`` into a full natural-language question.

    Topics resolve in this order: revealed clue → other suspect → visited
    location. Unresolved or non-topic questions pass through unchanged so
    the existing free-form path is unaffected.
    """
    stripped = question.strip()
    lower = stripped.lower()
    prefix = None
    for cand in ("about the ", "about a ", "about an ", "about "):
        if lower.startswith(cand):
            prefix = cand
            break
    if prefix is None:
        return question
    topic = stripped[len(prefix) :].strip(" ?.,!:;")
    if not topic:
        return question

    clue = resolve_clue(bible, topic, state["revealed_clue_ids"])
    if clue is not None:
        return (
            f"I want to ask you about something specific: {clue.description} "
            "What can you tell me about it?"
        )

    suspect = resolve_suspect(bible, topic)
    if suspect is not None:
        return (
            f"What can you tell me about {suspect.name}? Where they were tonight, "
            "what you know of them — anything."
        )

    by_id = {loc.id: loc for loc in bible.locations}
    for loc in bible.locations:
        if loc.id in state["visited_location_ids"] and (
            loc.id.lower() == topic.lower() or loc.name.lower() == topic.lower()
        ):
            return (
                f"What can you tell me about {by_id[loc.id].name}? Were you there "
                "tonight? Did you see anyone?"
            )

    return question


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

    expanded_question = _expand_topic_question(state, bible, question)
    retriever = suspect_retriever(vectorstore, suspect_id=suspect.id)
    prior = state["suspect_commitments"].get(suspect.id, [])

    wrapped = StreamingPrefix(stream_callback, suspect.name) if stream_callback else None
    reply = respond_as_suspect(
        suspect,
        retriever,
        chat_model,
        question=expanded_question,
        prior_commitments=prior,
        stream_callback=wrapped,
    )
    if wrapped is not None:
        wrapped.finalize()

    extractor: CommitmentExtractor = commitment_extractor or NullCommitmentExtractor()
    new_commitment = extractor.extract(suspect, expanded_question, reply)
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

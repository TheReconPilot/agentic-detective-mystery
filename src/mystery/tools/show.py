"""Show tool: confront a suspect with a specific revealed clue.

The M10 payoff for M9: when a clue contradicts a prior commitment, this is
where the deception policy is supposed to crack. The suspect agent receives
the clue's description in the system prompt on top of the persona / voice /
commitments scaffolding, then the same commitment extractor runs on the
reaction so the confrontation feeds the next-turn carry-through.

A clue must already be in ``state["revealed_clue_ids"]`` to be shown — the
player can't bluff their way through by naming clues they haven't earned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mystery.agents.commitments import NullCommitmentExtractor
from mystery.agents.suspect import respond_as_suspect
from mystery.rag.retriever import suspect_retriever
from mystery.tools._resolve import (
    format_revealed_clues,
    format_suspect_roster,
    resolve_clue,
    resolve_suspect,
)
from mystery.tools._streaming import StreamingPrefix

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from mystery.agents.commitments import CommitmentExtractor
    from mystery.graph.state import GameState
    from mystery.models import CaseBible


def apply_show(
    state: GameState,
    bible: CaseBible,
    vectorstore: Chroma,
    chat_model: BaseChatModel,
    suspect_id: str,
    clue_id: str,
    commitment_extractor: CommitmentExtractor | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    suspect = resolve_suspect(bible, suspect_id)
    if suspect is None:
        roster = format_suspect_roster(bible)
        return {
            "last_output": (
                f"I'm not sure who you mean by {suspect_id!r}. "
                f"You can confront any of these (by id, name, or archetype):\n{roster}"
            )
        }

    clue = resolve_clue(bible, clue_id, state["revealed_clue_ids"])
    if clue is None:
        inventory = format_revealed_clues(bible, state["revealed_clue_ids"])
        # Distinguish "no such clue at all" from "exists but not yet found",
        # since the player's next move is different (rethink vs. go examine).
        exists = any(c.id == clue_id for c in bible.clues)
        if exists and clue_id not in state["revealed_clue_ids"]:
            return {
                "last_output": (
                    f"You haven't found {clue_id!r} yet. "
                    f"You can only confront a suspect with clues you have actually examined.\n"
                    f"Clues you have found so far:\n{inventory}"
                ),
            }
        return {
            "last_output": (
                f"I'm not sure which clue you mean by {clue_id!r}. "
                f"You can confront with any clue you have found (by id or by description):\n"
                f"{inventory}"
            ),
        }

    retriever = suspect_retriever(vectorstore, suspect_id=suspect.id)
    prior = state["suspect_commitments"].get(suspect.id, [])
    # The "question" we pass is the framing the suspect sees: the system
    # prompt's clue block already handles the confrontation, so the user
    # message just narrates the moment in the detective's voice.
    framing_question = "I'm showing you this. What do you have to say about it?"

    if stream_callback is not None:
        stream_callback(f"You show {suspect.name} the evidence: {clue.description}\n")
    wrapped = StreamingPrefix(stream_callback, suspect.name) if stream_callback else None
    reply = respond_as_suspect(
        suspect,
        retriever,
        chat_model,
        question=framing_question,
        prior_commitments=prior,
        confronting_clue=clue,
        stream_callback=wrapped,
    )
    if wrapped is not None:
        wrapped.finalize()

    extractor: CommitmentExtractor = commitment_extractor or NullCommitmentExtractor()
    new_commitment = extractor.extract(suspect, framing_question, reply)
    rendered = (
        ""
        if stream_callback is not None
        else f"You show {suspect.name} the evidence: {clue.description}\n{suspect.name}: {reply}"
    )
    update: dict[str, Any] = {
        "turn_count": state["turn_count"] + 1,
        "last_output": rendered,
    }
    if new_commitment is not None:
        updated_commitments = {k: list(v) for k, v in state["suspect_commitments"].items()}
        updated_commitments.setdefault(suspect.id, []).append(new_commitment)
        update["suspect_commitments"] = updated_commitments
    return update
